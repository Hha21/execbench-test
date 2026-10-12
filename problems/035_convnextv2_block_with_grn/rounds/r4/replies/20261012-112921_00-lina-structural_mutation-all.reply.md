### Rationale
I ported lineage D's bulk-plane conv into A's K1 for the W=7, 14 and 449×7 rows. The shipped build passes 16/16 in run_tests. K1 got 8–19 µs faster on the five rows I probed. Four steps are not done: the 10-seed-round fp64 precision log, Step 0, Step 5, and the nwmin sweep. The run_tests drift is large, so treat the projected score with care.

```json solution-spec
{"name": "r4-a-e1-bulkplane-wprep", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "description": "Lineage A E1: bulk-plane depthwise conv (per-lane 1-D cp.async.bulk of channel row ranges into bank-skewed shared slots, warp per output row segment, halving-shuffle LN partials) plus a coalesced 16-byte weight conversion spread over all K1 CTAs for W=7/14/449x7 rows; cuBLASLt FP16 pipeline unchanged.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda", "-lcublasLt"]}}}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

extern "C" size_t convblk_ws_bytes(int B, int C, int H, int W);
extern "C" int convblk_run(const float* x, const float* dww, const float* dwb, const float* lnw, const float* lnb,
                           const float* w1, const float* b1, const float* gw, const float* gb, const float* w2,
                           const float* b2, float* out, void* ws, int B, int C, int H, int W, float grn_eps,
                           float ln_eps, cudaStream_t st);

void run(torch::Tensor x, torch::Tensor dww, torch::Tensor dwb, torch::Tensor lnw, torch::Tensor lnb,
         torch::Tensor w1, torch::Tensor b1, torch::Tensor gw, torch::Tensor gb, torch::Tensor w2, torch::Tensor b2,
         double eps, double ln_eps, torch::Tensor out) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  const int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  size_t wb = convblk_ws_bytes(B, C, H, W);
  TORCH_CHECK(wb > 0, "unsupported shape or cuBLASLt heuristic failure");
  auto ws = torch::empty({(int64_t)wb}, x.options().dtype(torch::kUInt8));
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  int rc = convblk_run(x.data_ptr<float>(), dww.data_ptr<float>(), dwb.data_ptr<float>(), lnw.data_ptr<float>(),
                       lnb.data_ptr<float>(), w1.data_ptr<float>(), b1.data_ptr<float>(), gw.data_ptr<float>(),
                       gb.data_ptr<float>(), w2.data_ptr<float>(), b2.data_ptr<float>(), out.data_ptr<float>(),
                       ws.data_ptr(), B, C, H, W, (float)eps, (float)ln_eps, st);
  TORCH_CHECK(rc == 0, "convblk_run failed with code ", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "convnextv2 block"); }
```

```cuda file=kernel.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) return 100 + (int)e_; } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) return 200 + (int)s_; } while (0)

static constexpr int NC = 4;  // copies of the GRN sum-of-squares accumulator (cuts same-address atomic contention)

static __device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
static __device__ __forceinline__ float4 ld4(const float* p) { return __ldg(reinterpret_cast<const float4*>(p)); }

struct K1P {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gb, *w2, *b2;
  __half *dbuf, *w1h, *w2h;
  float *sumsq, *uvec, *cvec;
  float2* stats;
  int B, C, H, W, M, RT, nTW, nTH, nGrp, nch, nrole, LD, doW2, PS;
  int CG, CS, STG, Wbox;  // TMA path only
  int SL, units, mask;    // bulk-plane path only
};

// Extra CTAs of K1 (TMA path and the old W=16 fallback): zero the GRN workspace; W1' = W1 * lnw -> fp16 with
// u[k] = sum_c W1'[k][c] and cst[k] = sum_c lnb[c] W1[k][c] + b1[k]; for the scale-h branch also W2 -> fp16 with
// b2' (hi/lo columns).
static __device__ void k1_role(const K1P& p, int rb, int lane, int warp, int nw) {
  const int gw = rb * nw + warp;
  const int gtid = gw * 32 + lane, gn = p.nrole * nw * 32;
  const int C = p.C, K4 = 4 * p.C;
  for (int i = gtid; i < p.B * NC * K4; i += gn) p.sumsq[i] = 0.f;
  for (int rr = 0; rr < 2; ++rr) {
    const int k = gw * 2 + rr;
    if (k < K4) {
      const float* wr = p.w1 + (size_t)k * C;
      __half* o = p.w1h + (size_t)k * C;
      float ua = 0.f, ca = 0.f;
      for (int c4 = lane * 4; c4 < C; c4 += 128) {
        float4 w = __ldg(reinterpret_cast<const float4*>(wr + c4));
        float4 l = __ldg(reinterpret_cast<const float4*>(p.lnw + c4));
        float4 bb = __ldg(reinterpret_cast<const float4*>(p.lnb + c4));
        ua += w.x * l.x + w.y * l.y + w.z * l.z + w.w * l.w;
        ca += w.x * bb.x + w.y * bb.y + w.z * bb.z + w.w * bb.w;
        *reinterpret_cast<uint2*>(o + c4) = make_uint2(pack_h2(w.x * l.x, w.y * l.y), pack_h2(w.z * l.z, w.w * l.w));
      }
      ua = warp_sum(ua);
      ca = warp_sum(ca);
      if (lane == 0) { p.uvec[k] = ua; p.cvec[k] = ca + p.b1[k]; }
    }
  }
  if (p.doW2) {
    for (int rr = 0; rr < 2; ++rr) {
      const int c = gw * 2 + rr;
      if (c < C) {
        const float* wr = p.w2 + (size_t)c * K4;
        __half* o = p.w2h + (size_t)c * p.LD;
        float dot = 0.f;
        for (int k = lane * 4; k < K4; k += 128) {
          float4 w = __ldg(reinterpret_cast<const float4*>(wr + k));
          float4 g = __ldg(reinterpret_cast<const float4*>(p.gb + k));
          dot += w.x * g.x + w.y * g.y + w.z * g.z + w.w * g.w;
          *reinterpret_cast<uint2*>(o + k) = make_uint2(pack_h2(w.x, w.y), pack_h2(w.z, w.w));
        }
        dot = warp_sum(dot);
        const float bb = p.b2[c] + dot;
        if (lane < 8) {
          __half hi = __float2half_rn(bb);
          __half v = lane == 0 ? hi : (lane == 1 ? __float2half_rn(bb - __half2float(hi)) : __float2half_rn(0.f));
          o[K4 + lane] = v;
        }
      }
    }
  }
}

// Old depthwise conv for the shapes the bulk-plane kernel does not take (W = 16 fallback only).
template <int WT>
__global__ void __launch_bounds__(192) k1_dw(const K1P p) {
  extern __shared__ float sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  if ((int)blockIdx.x < p.nrole) { k1_role(p, blockIdx.x, lane, warp, nw); return; }
  int t = blockIdx.x - p.nrole;
  const int cg = t % p.nGrp; t /= p.nGrp;
  const int tw = t % p.nTW; t /= p.nTW;
  const int th = t % p.nTH;
  const int b = t / p.nTH;
  const int chunk = cg * nw + warp;
  if (chunk >= p.nch) return;
  const int C = p.C, H = p.H, W = p.W, RT = p.RT, PS = p.PS;
  const int c0 = chunk * 32;
  const int h0 = th * RT, w0 = tw * WT;
  const int rows = min(RT, H - h0);
  constexpr int CW = WT + 6;
  const int RG = (RT + 6) * CW;
  float* pl = sm + warp * (32 * PS + 32 * 49);
  float* wsm = pl + 32 * PS;

  const float* xb = p.x + (size_t)(b * C + c0) * H * W;
  for (int cl = 0; cl < 32; ++cl) {
    const float* xp = xb + (size_t)cl * H * W;
    float* dp = pl + cl * PS;
    for (int rem = lane; rem < RG; rem += 32) {
      const int rr = rem / CW, cc = rem - rr * CW;
      const int gh = h0 - 3 + rr, gw = w0 - 3 + cc;
      const bool ok = gh >= 0 && gh < H && gw >= 0 && gw < W;
      const float* src = ok ? xp + gh * W + gw : p.x;
      uint32_t d = (uint32_t)__cvta_generic_to_shared(dp + rem);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(ok ? 4 : 0));
    }
  }
  for (int e = lane; e < 32 * 49; e += 32) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(wsm + e);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(p.dww + (size_t)c0 * 49 + e));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();

  float wr[49];
  const float* wp = wsm + lane * 49;
#pragma unroll
  for (int k = 0; k < 49; ++k) wr[k] = wp[k];
  const float bias = __ldg(p.dwb + c0 + lane);
  const float* pt = pl + lane * PS;
  for (int r = 0; r < rows; ++r) {
    float acc[WT];
#pragma unroll
    for (int w = 0; w < WT; ++w) acc[w] = bias;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
      float xr[CW];
#pragma unroll
      for (int q = 0; q < CW; ++q) xr[q] = pt[(r + i) * CW + q];
#pragma unroll
      for (int j = 0; j < 7; ++j)
#pragma unroll
        for (int w = 0; w < WT; ++w) acc[w] = fmaf(xr[w + j], wr[i * 7 + j], acc[w]);
    }
    const size_t mrow = ((size_t)b * H + h0 + r) * W + w0;
#pragma unroll
    for (int w = 0; w < WT; ++w) {
      const float dv = acc[w];
      p.dbuf[(mrow + w) * C + c0 + lane] = __float2half_rn(dv);
      const float s = warp_sum(dv), q2 = warp_sum(dv * dv);
      if (lane == 0) p.stats[(size_t)chunk * p.M + mrow + w] = make_float2(s, q2);
    }
  }
}

// ---- TMA-staged depthwise conv for W % 4 == 0 -------------------------------------------------------------
// CTA = (image b, RT output rows at full width, CG channels). x arrives through a 3-D tensor map over
// [B*C][H][W] in boxes {W+8, RT+6, CS} that start at (w=-4, h=h0-3): the TMA unit zero-fills the halo (the inner
// start coordinate must be a multiple of 16 B, hence -4, not -3). A STG-deep mbarrier ring walks the CG channels,
// CS per stage. Thread = (channel slot, row strip of R rows, 4 output columns). Writes d channel-major (fp16
// d[c][m], the B operand of GEMM1 transposed) and per-(channel group, token) partial (sum d, sum d^2).
struct __align__(64) TMap { unsigned long long v[16]; };

static __device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
static __device__ __forceinline__ void tma3(const TMap* tm, float* dst, unsigned long long* bar, int bytes, int x, int y,
                                            int z) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(bar)), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :: "r"(su32(dst)), "l"(tm), "r"(x), "r"(y), "r"(z), "r"(su32(bar)) : "memory");
}
static __device__ __forceinline__ void mbar_wait(unsigned long long* bar, uint32_t par) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }"
               :: "r"(su32(bar)), "r"(par) : "memory");
}

template <int R>
__global__ void __launch_bounds__(256) k1_tma(const __grid_constant__ TMap tmap, const K1P p) {
  extern __shared__ unsigned char smraw[];
  __shared__ __align__(8) unsigned long long bar[4];
  const int tid = threadIdx.x;
  if ((int)blockIdx.x < p.nrole) {
    const int nw = blockDim.x >> 5;
    if ((tid >> 5) < nw) k1_role(p, blockIdx.x, tid & 31, tid >> 5, nw);
    return;
  }
  int t = blockIdx.x - p.nrole;
  const int g = t % p.nGrp; t /= p.nGrp;
  const int th = t % p.nTH;
  const int b = t / p.nTH;
  const int H = p.H, W = p.W, RT = p.RT, CS = p.CS, CG = p.CG, Wbox = p.Wbox, STG = p.STG;
  const int c0 = g * CG, h0 = th * RT, nst = CG / CS;
  const int CHF = Wbox * (RT + 6), stf = CS * CHF;
  float* ring = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(smraw) + 127) & ~(uintptr_t)127);
  const int ringf = max(STG * stf, CS * RT * W * 2);
  float* wsm = ring + ringf;
  float* bsm = wsm + CG * 52;
  const TMap* tm = &tmap;
  const int zb = b * p.C + c0;
  if (tid == 0) {
    for (int s = 0; s < STG; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&bar[s])));
    asm volatile("fence.mbarrier_init.release.cluster;");
    for (int k = 0; k < min(STG, nst); ++k) tma3(tm, ring + k * stf, &bar[k], stf * 4, -4, h0 - 3, zb + k * CS);
  }
  for (int e = tid; e < CG * 49; e += blockDim.x) {
    const int c = e / 49;
    wsm[c * 52 + e - c * 49] = __ldg(p.dww + (size_t)c0 * 49 + e);
  }
  for (int e = tid; e < CG; e += blockDim.x) bsm[e] = __ldg(p.dwb + c0 + e);
  __syncthreads();

  const int NQ = W >> 2, NS = RT / R;
  const int q = tid % NQ, t2 = tid / NQ, strip = t2 % NS, slot = t2 / NS;
  const int r0 = strip * R;
  const size_t HW = (size_t)H * W;
  float sa[R][4], qa[R][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int u = 0; u < 4; ++u) { sa[r][u] = 0.f; qa[r][u] = 0.f; }

  for (int k = 0; k < nst; ++k) {
    const int s = k % STG;
    mbar_wait(&bar[s], (uint32_t)((k / STG) & 1));
    const int cl = k * CS + slot;
    const float* pl = ring + s * stf + slot * CHF + r0 * Wbox + 4 * q;
    float wv[52];
#pragma unroll
    for (int i = 0; i < 13; ++i) {
      const float4 v = *reinterpret_cast<const float4*>(wsm + cl * 52 + 4 * i);
      wv[4 * i] = v.x; wv[4 * i + 1] = v.y; wv[4 * i + 2] = v.z; wv[4 * i + 3] = v.w;
    }
    const float bias = bsm[cl];
    float acc[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < 4; ++u) acc[r][u] = bias;
#pragma unroll
    for (int ii = 0; ii < R + 6; ++ii) {
      float xv[12];
#pragma unroll
      for (int v4 = 0; v4 < 3; ++v4) {
        const float4 v = *reinterpret_cast<const float4*>(pl + ii * Wbox + 4 * v4);
        xv[4 * v4] = v.x; xv[4 * v4 + 1] = v.y; xv[4 * v4 + 2] = v.z; xv[4 * v4 + 3] = v.w;
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int i = ii - r;
        if (i >= 0 && i < 7) {
#pragma unroll
          for (int j = 0; j < 7; ++j)
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[r][u] = fmaf(xv[u + 1 + j], wv[i * 7 + j], acc[r][u]);
        }
      }
    }
    __half* dcol = p.dbuf + (size_t)(c0 + cl) * p.M + (size_t)b * HW + 4 * q;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int h = h0 + r0 + r;
      if (h < H) {
        *reinterpret_cast<uint2*>(dcol + (size_t)h * W) =
            make_uint2(pack_h2(acc[r][0], acc[r][1]), pack_h2(acc[r][2], acc[r][3]));
#pragma unroll
        for (int u = 0; u < 4; ++u) { sa[r][u] += acc[r][u]; qa[r][u] = fmaf(acc[r][u], acc[r][u], qa[r][u]); }
      }
    }
    __syncthreads();
    if (tid == 0 && k + STG < nst) tma3(tm, ring + s * stf, &bar[s], stf * 4, -4, h0 - 3, zb + (k + STG) * CS);
  }
  // partial LayerNorm statistics: reduce the CS channel slots through shared memory (ring is idle now)
  float2* red = reinterpret_cast<float2*>(ring);
  const int ntok = RT * W;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int u = 0; u < 4; ++u) red[slot * ntok + (r0 + r) * W + 4 * q + u] = make_float2(sa[r][u], qa[r][u]);
  __syncthreads();
  float2* st = p.stats + (size_t)g * p.M + (size_t)b * HW + (size_t)h0 * W;
  for (int tk = tid; tk < ntok; tk += blockDim.x) {
    if (h0 + tk / W < H) {
      float2 a = red[tk];
      for (int sl = 1; sl < CS; ++sl) { const float2 v = red[sl * ntok + tk]; a.x += v.x; a.y += v.y; }
      st[tk] = a;
    }
  }
}

// ---- bulk-plane depthwise conv + coalesced weight pass for W = 7 / 14 (7x7, 14x14, 449x7) --------------------
static __device__ __forceinline__ void expect_tx(unsigned long long* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(su32(bar)), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)) : "memory");
}

// W1 row k -> fp16 W1' = W1 * lnw (8 columns per lane, 16-byte loads/stores) with u[k], cst[k].
static __device__ void w1_row(const K1P& p, int k, int lane) {
  const int C = p.C;
  const float* wr = p.w1 + (size_t)k * C;
  __half* o = p.w1h + (size_t)k * C;
  float4 wa[3], wb[3];
#pragma unroll
  for (int it = 0; it < 3; ++it) {
    const int c8 = lane * 8 + it * 256;
    wa[it] = make_float4(0.f, 0.f, 0.f, 0.f);
    wb[it] = wa[it];
    if (c8 < C) { wa[it] = ld4(wr + c8); wb[it] = ld4(wr + c8 + 4); }
  }
  float ua = 0.f, ca = 0.f;
#pragma unroll
  for (int it = 0; it < 3; ++it) {
    const int c8 = lane * 8 + it * 256;
    if (c8 < C) {
      const float4 l0 = ld4(p.lnw + c8), l1 = ld4(p.lnw + c8 + 4), q0 = ld4(p.lnb + c8), q1 = ld4(p.lnb + c8 + 4);
      const float w[8] = {wa[it].x, wa[it].y, wa[it].z, wa[it].w, wb[it].x, wb[it].y, wb[it].z, wb[it].w};
      const float l[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
      const float q[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      uint32_t pk[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ua += w[2 * i] * l[2 * i] + w[2 * i + 1] * l[2 * i + 1];
        ca += w[2 * i] * q[2 * i] + w[2 * i + 1] * q[2 * i + 1];
        pk[i] = pack_h2(w[2 * i] * l[2 * i], w[2 * i + 1] * l[2 * i + 1]);
      }
      *reinterpret_cast<uint4*>(o + c8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }
  ua = warp_sum(ua);
  ca = warp_sum(ca);
  if (lane == 0) { p.uvec[k] = ua; p.cvec[k] = ca + p.b1[k]; }
}

// W2 row c -> fp16 (LD-strided rows) with b2' = b2 + W2 gb as hi/lo columns 4C, 4C+1 (only when the hidden is big).
static __device__ void w2_row(const K1P& p, int c, int lane) {
  const int K4 = 4 * p.C;
  const float* wr = p.w2 + (size_t)c * K4;
  __half* o = p.w2h + (size_t)c * p.LD;
  float dot = 0.f;
  for (int base = 0; base < K4; base += 768) {
    float4 wa[3], wb[3], ga[3], gc[3];
#pragma unroll
    for (int it = 0; it < 3; ++it) {
      const int k8 = base + lane * 8 + it * 256;
      wa[it] = make_float4(0.f, 0.f, 0.f, 0.f);
      wb[it] = wa[it]; ga[it] = wa[it]; gc[it] = wa[it];
      if (k8 < K4) { wa[it] = ld4(wr + k8); wb[it] = ld4(wr + k8 + 4); ga[it] = ld4(p.gb + k8); gc[it] = ld4(p.gb + k8 + 4); }
    }
#pragma unroll
    for (int it = 0; it < 3; ++it) {
      const int k8 = base + lane * 8 + it * 256;
      if (k8 < K4) {
        dot += wa[it].x * ga[it].x + wa[it].y * ga[it].y + wa[it].z * ga[it].z + wa[it].w * ga[it].w +
               wb[it].x * gc[it].x + wb[it].y * gc[it].y + wb[it].z * gc[it].z + wb[it].w * gc[it].w;
        *reinterpret_cast<uint4*>(o + k8) =
            make_uint4(pack_h2(wa[it].x, wa[it].y), pack_h2(wa[it].z, wa[it].w), pack_h2(wb[it].x, wb[it].y),
                       pack_h2(wb[it].z, wb[it].w));
      }
    }
  }
  dot = warp_sum(dot);
  const float bb = p.b2[c] + dot;
  if (lane < 8) {
    __half hi = __float2half_rn(bb);
    __half v = lane == 0 ? hi : (lane == 1 ? __float2half_rn(bb - __half2float(hi)) : __float2half_rn(0.f));
    o[K4 + lane] = v;
  }
}

// Halving reduction across the warp: v[0..2H) -> v[0..H), each lane keeping half of the values.
template <int H>
static __device__ __forceinline__ void halve(float* v, bool hi, int xm) {
#pragma unroll
  for (int i = 0; i < H; ++i) {
    const float send = hi ? v[i] : v[i + H];
    const float keep = hi ? v[i + H] : v[i];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, xm);
  }
}

// CTA = (image b, 32-channel chunk, tile of RT output rows). Lane = channel; each lane issues one 1-D bulk copy of
// its channel's rows [hlo, hhi) (rounded out to 16 B) into a padded shared slot, so a warp reads 32 different
// channels without bank conflicts when HW is odd. Warp = (output row, 7-column segment); the 49 taps live in
// registers. LayerNorm partials (sum, sum of squares over the 32 channels) are reduced with a halving shuffle
// tree. While the copies are in flight every warp converts its share of the weights (W1', W2, GRN workspace).
template <int W, int SEGS>
__global__ void __launch_bounds__(448) k1_bulk(const K1P p) {
  extern __shared__ __align__(128) float smb[];
  __shared__ __align__(8) unsigned long long bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  const int C = p.C, H = p.H, SL = p.SL;
  const bool doconv = (p.mask & 1) != 0;
  int t = blockIdx.x;
  const int th = t % p.nTH; t /= p.nTH;
  const int chunk = t % p.nch;
  const int b = t / p.nch;
  const int c0 = chunk * 32;
  const int h0 = th * p.RT;
  const int rows = min(p.RT, H - h0);
  const int hlo = max(h0 - 3, 0), hhi = min(H, h0 + rows + 3);
  const int HW = H * W;
  float* wsm = smb + 32 * SL;
  const size_t g0 = ((size_t)(b * C + c0 + lane) * H + hlo) * W;
  const int shift = (int)(g0 & 3);
  const int slot = (HW & 1) ? ((lane >> 2) + 8 * shift) : lane;

  if (doconv) {
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(su32(&bar)));
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    if (warp == 0) {
      const size_t a = g0 & ~(size_t)3;
      const size_t gend = ((size_t)(b * C + c0 + lane) * H + hhi) * W;
      const size_t aend = (gend + 3) & ~(size_t)3;
      const uint32_t bytes = (uint32_t)((aend - a) * 4);
      expect_tx(&bar, bytes + (lane == 0 ? 32 * 49 * 4 : 0));
      bulk_g2s(smb + slot * SL, p.x + a, bytes, &bar);
      if (lane == 0) bulk_g2s(wsm, p.dww + (size_t)c0 * 49, 32 * 49 * 4, &bar);
    }
  }

  if (p.mask & 2) {
    const int gwid = blockIdx.x * nw + warp, nwt = gridDim.x * nw;
    const int nW2 = p.doW2 ? C : 0;
    for (int u = gwid; u < p.units; u += nwt) {
      if (u < nW2) w2_row(p, u, lane);
      else w1_row(p, u - nW2, lane);
    }
    const int n4 = p.B * NC * 4 * C / 4;
    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x)
      reinterpret_cast<float4*>(p.sumsq)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  if (!doconv || warp >= SEGS * rows) return;
  const int r = warp / SEGS;
  const int seg = (SEGS == 1) ? 0 : (warp % SEGS);
  const int hout = h0 + r;
  const float bias = __ldg(p.dwb + c0 + lane);
  mbar_wait(&bar, 0);
  float wr[49];
#pragma unroll
  for (int k = 0; k < 49; ++k) wr[k] = wsm[lane * 49 + k];
  const int off0 = slot * SL + shift - hlo * W;
  float acc[7];
#pragma unroll
  for (int w = 0; w < 7; ++w) acc[w] = bias;
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    const int h = hout - 3 + i;
    if ((unsigned)h < (unsigned)H) {
      const int rb = off0 + h * W;
      float xr[13];
#pragma unroll
      for (int q = 0; q < 13; ++q) {
        const int cc = seg * 7 - 3 + q;
        if (SEGS == 1 && (q < 3 || q > 9)) xr[q] = 0.f;
        else xr[q] = ((unsigned)cc < (unsigned)W) ? smb[rb + cc] : 0.f;
      }
#pragma unroll
      for (int j = 0; j < 7; ++j)
#pragma unroll
        for (int w = 0; w < 7; ++w) {
          if (SEGS == 1 && (w + j < 3 || w + j > 9)) continue;
          acc[w] = fmaf(xr[w + j], wr[i * 7 + j], acc[w]);
        }
    }
  }
  const size_t m0 = (size_t)b * HW + (size_t)hout * W + seg * 7;
  float v[16];
#pragma unroll
  for (int w = 0; w < 7; ++w) {
    p.dbuf[(m0 + w) * C + c0 + lane] = __float2half_rn(acc[w]);
    v[2 * w] = acc[w];
    v[2 * w + 1] = acc[w] * acc[w];
  }
  v[14] = 0.f;
  v[15] = 0.f;
  halve<8>(v, (lane & 16) != 0, 16);
  halve<4>(v, (lane & 8) != 0, 8);
  halve<2>(v, (lane & 4) != 0, 4);
  halve<1>(v, (lane & 2) != 0, 2);
  const float tot = v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
  const int idx = (((lane >> 4) & 1) << 3) | (((lane >> 3) & 1) << 2) | (((lane >> 2) & 1) << 1) | ((lane >> 1) & 1);
  if (idx < 14 && !(lane & 1))
    reinterpret_cast<float*>(p.stats)[((size_t)chunk * p.M + m0 + (idx >> 1)) * 2 + (idx & 1)] = tot;
}

static __device__ __forceinline__ float gelu_erf(float z) {
  return 0.5f * z * (1.f + erff(z * 0.70710678118654752f));
}

// In-place LN-correction + bias + exact GELU on the fp16 GEMM1 output:
//   z = rstd_t * (Y[t][k] - mean_t * u[k]) + cst[k];  h = gelu(z)
// per-(image, k) sum of squares into sumsq; writes the two ones-columns (4C, 4C+1) that carry b2' through GEMM2.
__global__ void k2_gelu(__half* Y, const float* uvec, const float* cvec, const float2* stats, float* sumsq, int HW, int C,
                        int LD, int tokb, int TS, int nch, int M, float ln_eps) {
  __shared__ float red[5120];
  __shared__ float s_mean[256], s_rstd[256];
  const int K4 = 4 * C, V = C / 2, Vt = V + 1;
  const int kc = threadIdx.x % Vt, ts = threadIdx.x / Vt;
  const int b = blockIdx.y, t0 = blockIdx.x * tokb, tend = min(t0 + tokb, HW);
  for (int i = threadIdx.x; i < tend - t0; i += blockDim.x) {
    const int m = b * HW + t0 + i;
    float s = 0.f, q = 0.f;
    for (int ch = 0; ch < nch; ++ch) { float2 v = stats[(size_t)ch * M + m]; s += v.x; q += v.y; }
    const float mean = s / C;
    const float var = fmaxf(q / C - mean * mean, 0.f);
    s_mean[i] = mean;
    s_rstd[i] = rsqrtf(var + ln_eps);
  }
  __syncthreads();
  __half* base = Y + (size_t)b * HW * LD;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  if (kc == V) {
    for (int t = t0 + ts; t < tend; t += TS)
      *reinterpret_cast<uint4*>(base + (size_t)t * LD + K4) = make_uint4(0x3C003C00u, 0u, 0u, 0u);
  } else {
    float uu[8], cc[8];
    {
      float4 a = __ldg(reinterpret_cast<const float4*>(uvec + kc * 8));
      float4 c = __ldg(reinterpret_cast<const float4*>(uvec + kc * 8 + 4));
      uu[0] = a.x; uu[1] = a.y; uu[2] = a.z; uu[3] = a.w; uu[4] = c.x; uu[5] = c.y; uu[6] = c.z; uu[7] = c.w;
      a = __ldg(reinterpret_cast<const float4*>(cvec + kc * 8));
      c = __ldg(reinterpret_cast<const float4*>(cvec + kc * 8 + 4));
      cc[0] = a.x; cc[1] = a.y; cc[2] = a.z; cc[3] = a.w; cc[4] = c.x; cc[5] = c.y; cc[6] = c.z; cc[7] = c.w;
    }
    for (int t = t0 + ts; t < tend; t += 4 * TS) {
      uint4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int tt = t + u * TS;
        if (tt < tend) v[u] = *reinterpret_cast<const uint4*>(base + (size_t)tt * LD + kc * 8);
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int tt = t + u * TS;
        if (tt < tend) {
          const float mu = s_mean[tt - t0], rs = s_rstd[tt - t0];
          __half2* hp = reinterpret_cast<__half2*>(&v[u]);
          uint32_t o[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float2 f = __half22float2(hp[q]);
            float g0 = gelu_erf(rs * (f.x - mu * uu[2 * q]) + cc[2 * q]);
            float g1 = gelu_erf(rs * (f.y - mu * uu[2 * q + 1]) + cc[2 * q + 1]);
            acc[2 * q] += g0 * g0;
            acc[2 * q + 1] += g1 * g1;
            o[q] = pack_h2(g0, g1);
          }
          *reinterpret_cast<uint4*>(base + (size_t)tt * LD + kc * 8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
    }
  }
  if (TS > 1 && kc < V && ts > 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) red[(ts - 1) * K4 + kc * 8 + i] = acc[i];
  }
  __syncthreads();
  if (kc < V && ts == 0) {
    for (int q = 1; q < TS; ++q) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += red[(q - 1) * K4 + kc * 8 + i];
    }
    float* dst = sumsq + ((size_t)b * NC + (blockIdx.x % NC)) * K4 + kc * 8;
    atomicAdd(reinterpret_cast<float4*>(dst), make_float4(acc[0], acc[1], acc[2], acc[3]));
    atomicAdd(reinterpret_cast<float4*>(dst + 4), make_float4(acc[4], acc[5], acc[6], acc[7]));
  }
}

// s[k] = 1 + gw[k] * gf[k] / (mean_k gf + eps), gf = sqrt(sum of the NC sumsq copies). 256 threads.
static __device__ void compute_s(float* s, float* red8, const float* ss, const float* gw, int K4, float eps) {
  const int tid = threadIdx.x;
  float loc = 0.f;
  for (int k = tid; k < K4; k += 256) {
    float q = ss[k];
#pragma unroll
    for (int c = 1; c < NC; ++c) q += ss[c * K4 + k];
    float g = sqrtf(q);
    s[k] = g;
    loc += g;
  }
  loc = warp_sum(loc);
  if ((tid & 31) == 0) red8[tid >> 5] = loc;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red8[i];
  const float inv = 1.f / (tot / K4 + eps);
  for (int k = tid; k < K4; k += 256) s[k] = 1.f + __ldg(gw + k) * s[k] * inv;
  __syncthreads();
}

// Branch A: per-image fp16 W2_b = W2 * diag(s_b), with b2' = b2 + W2 gb as hi/lo in columns 4C, 4C+1.
// The first four float4 of each W2 row are loaded before the (dependent) statistics reduction.
__global__ void __launch_bounds__(256) k3a_w2(const float* w2, const float* gb, const float* b2, const float* gw,
                                              const float* ss, __half* w2s, int C, int LD, float eps) {
  extern __shared__ float sm[];
  const int K4 = 4 * C, b = blockIdx.y;
  const int lane = threadIdx.x & 31, c = blockIdx.x * 8 + (threadIdx.x >> 5);
  const float* wr = w2 + (size_t)min(c, C - 1) * K4;
  float4 wpf[4], gpf[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = lane * 4 + 128 * i;
    wpf[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    gpf[i] = wpf[i];
    if (k < K4) {
      wpf[i] = __ldg(reinterpret_cast<const float4*>(wr + k));
      gpf[i] = __ldg(reinterpret_cast<const float4*>(gb + k));
    }
  }
  compute_s(sm, sm + K4, ss + (size_t)b * NC * K4, gw, K4, eps);
  if (c >= C) return;
  __half* o = w2s + ((size_t)b * C + c) * LD;
  float dot = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = lane * 4 + 128 * i;
    if (k < K4) {
      float4 w = wpf[i], g = gpf[i];
      float4 sv = *reinterpret_cast<const float4*>(sm + k);
      dot += w.x * g.x + w.y * g.y + w.z * g.z + w.w * g.w;
      *reinterpret_cast<uint2*>(o + k) = make_uint2(pack_h2(w.x * sv.x, w.y * sv.y), pack_h2(w.z * sv.z, w.w * sv.w));
    }
  }
  for (int k = lane * 4 + 512; k < K4; k += 128) {
    float4 w = __ldg(reinterpret_cast<const float4*>(wr + k));
    float4 g = __ldg(reinterpret_cast<const float4*>(gb + k));
    float4 sv = *reinterpret_cast<const float4*>(sm + k);
    dot += w.x * g.x + w.y * g.y + w.z * g.z + w.w * g.w;
    *reinterpret_cast<uint2*>(o + k) = make_uint2(pack_h2(w.x * sv.x, w.y * sv.y), pack_h2(w.z * sv.z, w.w * sv.w));
  }
  dot = warp_sum(dot);
  const float bb = b2[c] + dot;
  if (lane < 8) {
    __half hi = __float2half_rn(bb);
    __half v = lane == 0 ? hi : (lane == 1 ? __float2half_rn(bb - __half2float(hi)) : __float2half_rn(0.f));
    o[K4 + lane] = v;
  }
}

// Branch B: scale the hidden rows in place (columns < 4C), fp16.
__global__ void __launch_bounds__(256) k3b_scale(__half* Y, const float* gw, const float* ss, int HW, int C, int LD,
                                                 int tokb, float eps) {
  extern __shared__ float sm[];
  const int K4 = 4 * C, V = C / 2, b = blockIdx.y;
  compute_s(sm, sm + K4, ss + (size_t)b * NC * K4, gw, K4, eps);
  const int t0 = blockIdx.x * tokb, ntok = min(tokb, HW - t0);
  __half* base = Y + ((size_t)b * HW + t0) * LD;
  const int total = ntok * V;
  for (int i0 = threadIdx.x; i0 < total; i0 += 1024) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = i0 + u * 256;
      if (i < total) {
        const int t = i / V, kc = i - t * V;
        v[u] = *reinterpret_cast<const uint4*>(base + (size_t)t * LD + kc * 8);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = i0 + u * 256;
      if (i < total) {
        const int t = i / V, kc = i - t * V;
        __half2* hp = reinterpret_cast<__half2*>(&v[u]);
        uint32_t o[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float2 f = __half22float2(hp[q]);
          o[q] = pack_h2(f.x * sm[kc * 8 + 2 * q], f.y * sm[kc * 8 + 2 * q + 1]);
        }
        *reinterpret_cast<uint4*>(base + (size_t)t * LD + kc * 8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
  }
}

// Channels-last GEMM2 result d2[m][c] -> NCHW out[b][c][hw] = x + d2 (32x32 smem transpose).
__global__ void k4_transpose_add(const float* x, const float* d2, float* out, int HW, int C) {
  __shared__ float tile[32][33];
  const int b = blockIdx.z, hw0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
  const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int hw = hw0 + ty + 8 * j, c = c0 + tx;
    tile[ty + 8 * j][tx] = (hw < HW && c < C) ? d2[((size_t)b * HW + hw) * C + c] : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = c0 + ty + 8 * j, hw = hw0 + tx;
    if (hw < HW && c < C) {
      const size_t i = ((size_t)b * C + c) * HW + hw;
      out[i] = x[i] + tile[tx][ty + 8 * j];
    }
  }
}

// ------------------------------------------------------------------ host side
static const size_t kCublasWs = 32u << 20;
static constexpr int kR = 2;  // output rows per thread in k1_tma

struct Plan {
  int B, C, H, W, M, HW, K4, LD;
  bool bB, useCL, tma, bulk;
  int WT, NWc, nch, nGrp, RT, nTW, nTH, ntiles, nrole, PS, thr1;
  int CG, CS, STG, Wbox, SL, nconv;
  size_t smem1;
  int tokb, TS2, thr2;
  size_t off_d, off_y, off_w1h, off_w2h, off_ss, off_uv, off_st, off_d2, off_cw, total;
  cublasLtMatmulDesc_t md1, md2;
  cublasLtMatrixLayout_t la1, lb1, ld1, la2, lb2, lc2;
  cublasLtMatmulHeuristicResult_t h1[8], h2[8];
  int n1, n2;
};

static cublasLtHandle_t g_lt = nullptr;
static std::map<uint64_t, Plan*> g_plans;
// g_mask / g_k1mask: stop after a launch / ablate K1 halves; g_forceold: old K1 for W=7/14; g_nwmin: min warps per K1 CTA
// (all are diagnostics for the bench; the defaults are what ships)
static int g_idx1 = 0, g_idx2 = 0, g_mask = 99, g_k1mask = 3, g_forceold = 0, g_nwmin = 8;

extern "C" void convblk_set_idx(int a, int b) { g_idx1 = a; g_idx2 = b; }
extern "C" void convblk_set_mask(int m) { g_mask = m; }
extern "C" void convblk_set_k1mask(int m) { g_k1mask = m; }
extern "C" void convblk_set_forceold(int m) { g_forceold = m; }
extern "C" void convblk_set_nwmin(int m) { g_nwmin = std::max(1, std::min(m, 14)); }

static size_t al256(size_t x) { return (x + 255) & ~(size_t)255; }

// A function's max dynamic shared memory is process-wide state, and the portal sets up every shape before it times
// any: only ever raise it, so a plan made later for a smaller shape cannot shrink the limit an earlier plan relies on.
static cudaError_t raise_smem(const void* f, int bytes) {
  static std::map<const void*, int> cur;
  int& c = cur[f];
  if (bytes <= c) return cudaSuccess;
  c = bytes;
  return cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

static cublasStatus_t mk_layout(cublasLtMatrixLayout_t* l, cudaDataType t, uint64_t rows, uint64_t cols, int64_t ld,
                                int batch, int64_t stride) {
  cublasStatus_t s = cublasLtMatrixLayoutCreate(l, t, rows, cols, ld);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  if (batch > 1) {
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
  }
  return s;
}

static int heur(cublasLtMatmulDesc_t md, cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c,
                uint32_t alA, uint32_t alB, uint32_t alC, cublasLtMatmulHeuristicResult_t* out, int* n) {
  cublasLtMatmulPreference_t pref;
  CB(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsb = kCublasWs;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alA, sizeof(alA));
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alB, sizeof(alB));
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alC, sizeof(alC));
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alC, sizeof(alC));
  int cnt = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_lt, md, a, b, c, c, pref, 8, out, &cnt);
  cublasLtMatmulPreferenceDestroy(pref);
  CB(s);
  if (cnt == 0) return 300;
  *n = cnt;
  return 0;
}

// TMA conv tiling: tallest tile (RT in 8/4/2 rows) and widest channel group CG (a divisor of C, multiple of 4)
// that still give >= 296 conv CTAs (2 per SM); CS channels per stage so the CTA has <= 256 threads and a stage is
// <= 18 KB.
static bool plan_tma(Plan* P) {
  const int B = P->B, C = P->C, H = P->H, W = P->W;
  if (W % 4 || W + 8 > 256 || C % 4) return false;
  int bestRT = 0, bestCG = 0, bestCtas = -1;
  for (int rt : {8, 4, 2}) {
    if (rt > H) continue;
    const int tiles = B * ((H + rt - 1) / rt);
    for (int cg = C; cg >= 4; --cg) {
      if (C % cg || cg % 4) continue;
      const int ctas = tiles * (C / cg);
      if (ctas >= 296) { bestRT = rt; bestCG = cg; bestCtas = ctas; break; }
      if (ctas > bestCtas && bestCtas < 296) { bestCtas = ctas; bestRT = rt; bestCG = cg; }
    }
    if (bestCtas >= 296) break;
  }
  if (!bestRT) return false;
  const int Wbox = W + 8, tpc = (W / 4) * (bestRT / kR);
  int cs = 0;
  for (int s = bestCG; s >= 1; --s)
    if (bestCG % s == 0 && s * tpc <= 256 && (size_t)s * Wbox * (bestRT + 6) * 4 <= 18432) { cs = s; break; }
  if (!cs || cs * tpc < 32) return false;
  P->RT = bestRT; P->CG = bestCG; P->CS = cs; P->Wbox = Wbox;
  P->nTH = (H + bestRT - 1) / bestRT;
  P->nGrp = C / bestCG;
  P->nch = P->nGrp;
  P->STG = std::min(4, bestCG / cs);
  P->thr1 = cs * tpc;
  P->ntiles = B * P->nTH;
  const size_t stf = (size_t)cs * Wbox * (bestRT + 6);
  const size_t ringf = std::max((size_t)P->STG * stf, (size_t)cs * bestRT * W * 2);
  P->smem1 = 128 + 4 * (ringf + (size_t)bestCG * 53);
  const int nw = P->thr1 / 32;
  P->nrole = (P->K4 + 2 * nw - 1) / (2 * nw);
  return true;
}

// Bulk-plane conv tiling (W = 7 or 14, C a multiple of 32 up to 768): one CTA per (image, 32-channel chunk, row tile),
// the fewest row tiles that give >= 296 CTAs; one warp per (output row, 7-column segment), at least g_nwmin warps so
// the weight pass has hands.
static bool plan_bulk(Plan* P) {
  const int B = P->B, C = P->C, H = P->H, W = P->W;
  if ((W != 7 && W != 14) || (C % 32) || C > 768) return false;
  const int nch = C / 32, BC = B * nch, segs = W / 7;
  int nTH = std::min(H, (296 + BC - 1) / BC);
  const int RT = (H + nTH - 1) / nTH;
  nTH = (H + RT - 1) / RT;
  const int items = RT * segs;
  if (items > 14) return false;
  const int rowsMax = std::min(H, RT + 6);
  int SL = (rowsMax * W + 8 + 3) & ~3;
  if (((SL / 4) & 1) == 0) SL += 4;  // SL/4 odd: slots land on distinct banks
  P->RT = RT; P->nTH = nTH; P->nch = nch; P->SL = SL; P->nrole = 0;
  P->thr1 = 32 * std::max(g_nwmin, items);
  P->smem1 = (size_t)4 * (32 * SL + 32 * 49);
  P->nconv = BC * nTH;
  return true;
}

static int make_plan(Plan* P, int B, int C, int H, int W) {
  P->B = B; P->C = C; P->H = H; P->W = W;
  P->HW = H * W; P->M = B * P->HW; P->K4 = 4 * C; P->LD = P->K4 + 8;
  const int M = P->M, HW = P->HW, K4 = P->K4, LD = P->LD;
  P->bB = (size_t)B * C * LD * 2 > ((size_t)16 << 20);
  P->useCL = !(HW % 8 == 0 && HW >= 256);
  P->tma = (M % 8 == 0) && plan_tma(P);
  if (P->tma) {
    CK(raise_smem((const void*)k1_tma<kR>, (int)P->smem1));
  } else if (!g_forceold && plan_bulk(P)) {
    P->bulk = true;
  } else {
    P->WT = (W % 16 == 0) ? 16 : (W % 14 == 0) ? 14 : (W % 7 == 0) ? 7 : 0;
    if (!P->WT || (C % 32)) return 1;
    P->nch = C / 32;
    P->nTW = W / P->WT;
    auto smem_of = [&](int rt, int nwc) {
      int ps = ((rt + 6) * (P->WT + 6)) | 1;
      return (size_t)4 * nwc * ((size_t)32 * ps + 32 * 49);
    };
    int nwc = 0, rt = 1;
    if ((int64_t)M * C >= ((int64_t)8 << 20)) {
      for (int c : {6, 5, 4, 3, 2, 1})
        if (P->nch % c == 0 && smem_of(1, c) <= 120000) { nwc = c; break; }
      if (!nwc) return 2;
      for (int r = 1; r <= std::min(H, 8); ++r) {
        int ctas = (P->nch / nwc) * B * ((H + r - 1) / r) * P->nTW;
        if (smem_of(r, nwc) <= 120000 && ctas >= 300) rt = r;
      }
    } else {
      nwc = 1;
      for (int r = 1; r <= std::min(H, 8); ++r) {
        int ctas = P->nch * B * ((H + r - 1) / r) * P->nTW;
        if (ctas >= 150) rt = r;
      }
    }
    P->NWc = nwc;
    P->nGrp = P->nch / nwc;
    P->RT = rt;
    P->PS = ((rt + 6) * (P->WT + 6)) | 1;
    P->smem1 = smem_of(rt, nwc);
    P->nTH = (H + rt - 1) / rt;
    P->ntiles = B * P->nTH * P->nTW;
    P->nrole = (K4 + 2 * nwc - 1) / (2 * nwc);
    P->thr1 = 32 * nwc;
    if (P->WT == 7) CK(raise_smem((const void*)k1_dw<7>, (int)P->smem1));
    if (P->WT == 14) CK(raise_smem((const void*)k1_dw<14>, (int)P->smem1));
    if (P->WT == 16) CK(raise_smem((const void*)k1_dw<16>, (int)P->smem1));
  }
  int tokb = 4;
  while (tokb < 256 && (HW + tokb - 1) / tokb > 48 * NC) tokb *= 2;
  P->tokb = tokb;
  const int Vt = C / 2 + 1;
  P->TS2 = std::min(4, 1024 / Vt);
  P->thr2 = Vt * P->TS2;
  CK(raise_smem((const void*)k3a_w2, 4 * K4 + 64));
  CK(raise_smem((const void*)k3b_scale, 4 * K4 + 64));

  P->off_d = 0;
  P->off_y = al256((size_t)M * C * 2);
  P->off_w1h = P->off_y + al256((size_t)M * LD * 2);
  P->off_w2h = P->off_w1h + al256((size_t)K4 * C * 2);
  P->off_ss = P->off_w2h + al256((size_t)(P->bB ? 1 : B) * C * LD * 2);
  P->off_uv = P->off_ss + al256((size_t)B * NC * K4 * 4);
  P->off_st = P->off_uv + al256((size_t)2 * K4 * 4);
  P->off_d2 = P->off_st + al256((size_t)P->nch * M * 8);
  P->off_cw = P->off_d2 + (P->useCL ? al256((size_t)M * C * 4) : 0);
  P->total = P->off_cw + kCublasWs;

  if (!g_lt) CB(cublasLtCreate(&g_lt));
  cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  CB(cublasLtMatmulDescCreate(&P->md1, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CB(cublasLtMatmulDescSetAttribute(P->md1, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CB(mk_layout(&P->la1, CUDA_R_16F, C, K4, C, 1, 0));
  if (P->tma) {
    // d is channel-major d[c][m]: B = (d stored M x C, ld M)^T
    CB(cublasLtMatmulDescSetAttribute(P->md1, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
    CB(mk_layout(&P->lb1, CUDA_R_16F, M, C, M, 1, 0));
  } else {
    CB(cublasLtMatmulDescSetAttribute(P->md1, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CB(mk_layout(&P->lb1, CUDA_R_16F, C, M, C, 1, 0));
  }
  CB(mk_layout(&P->ld1, CUDA_R_16F, K4, M, LD, 1, 0));
  int rc = heur(P->md1, P->la1, P->lb1, P->ld1, 16, 16, 16, P->h1, &P->n1);
  if (rc) return rc;

  CB(cublasLtMatmulDescCreate(&P->md2, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CB(cublasLtMatmulDescSetAttribute(P->md2, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  CB(cublasLtMatmulDescSetAttribute(P->md2, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  if (!P->useCL) {
    // NCHW batched: D_b^T [C][HW] = W2_b [C][LD] x h_b^T, beta=1 with C = x.
    CB(mk_layout(&P->la2, CUDA_R_16F, LD, HW, LD, B, (int64_t)HW * LD));
    CB(mk_layout(&P->lb2, CUDA_R_16F, LD, C, LD, B, P->bB ? 0 : (int64_t)C * LD));
    CB(mk_layout(&P->lc2, CUDA_R_32F, HW, C, HW, B, (int64_t)C * HW));
  } else if (!P->bB) {
    // channels-last, per image: D_b [C][HW] (ld C) = W2_b-view x h_b-view, then K4 transposes and adds x.
    CB(mk_layout(&P->la2, CUDA_R_16F, LD, C, LD, B, (int64_t)C * LD));
    CB(mk_layout(&P->lb2, CUDA_R_16F, LD, HW, LD, B, (int64_t)HW * LD));
    CB(mk_layout(&P->lc2, CUDA_R_32F, C, HW, C, B, (int64_t)C * HW));
  } else {
    CB(mk_layout(&P->la2, CUDA_R_16F, LD, C, LD, 1, 0));
    CB(mk_layout(&P->lb2, CUDA_R_16F, LD, M, LD, 1, 0));
    CB(mk_layout(&P->lc2, CUDA_R_32F, C, M, C, 1, 0));
  }
  rc = heur(P->md2, P->la2, P->lb2, P->lc2, 16, 16, 16, P->h2, &P->n2);
  return rc;
}

static Plan* get_plan(int B, int C, int H, int W, int* err) {
  uint64_t key = ((uint64_t)B << 48) ^ ((uint64_t)C << 32) ^ ((uint64_t)H << 16) ^ (uint64_t)W ^
                 ((uint64_t)g_forceold << 62) ^ ((uint64_t)g_nwmin << 56);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan* P = new Plan();
  int rc = make_plan(P, B, C, H, W);
  if (rc) { *err = rc; delete P; return nullptr; }
  g_plans[key] = P;
  return P;
}

extern "C" void convblk_counts(int B, int C, int H, int W, int* o) {
  int err = 0;
  Plan* P = get_plan(B, C, H, W, &err);
  if (!P) { o[0] = -err; return; }
  o[0] = P->n1; o[1] = P->n2; o[2] = P->RT; o[3] = P->tma ? P->CS : P->NWc; o[4] = P->bulk ? P->nconv : P->ntiles;
  o[5] = P->nrole; o[6] = P->bB; o[7] = P->useCL;
}

extern "C" size_t convblk_ws_bytes(int B, int C, int H, int W) {
  int err = 0;
  Plan* P = get_plan(B, C, H, W, &err);
  return P ? P->total : 0;
}

extern "C" int convblk_run(const float* x, const float* dww, const float* dwb, const float* lnw, const float* lnb,
                           const float* w1, const float* b1, const float* gw, const float* gb, const float* w2,
                           const float* b2, float* out, void* wsp, int B, int C, int H, int W, float grn_eps,
                           float ln_eps, cudaStream_t st) {
  int err = 0;
  Plan* P = get_plan(B, C, H, W, &err);
  if (!P) return err;
  char* ws = static_cast<char*>(wsp);
  __half* dbuf = reinterpret_cast<__half*>(ws + P->off_d);
  __half* y = reinterpret_cast<__half*>(ws + P->off_y);
  __half* w1h = reinterpret_cast<__half*>(ws + P->off_w1h);
  __half* w2h = reinterpret_cast<__half*>(ws + P->off_w2h);
  float* ss = reinterpret_cast<float*>(ws + P->off_ss);
  float* uvec = reinterpret_cast<float*>(ws + P->off_uv);
  float* cvec = uvec + P->K4;
  float2* stats = reinterpret_cast<float2*>(ws + P->off_st);
  float* d2 = reinterpret_cast<float*>(ws + P->off_d2);
  void* cw = ws + P->off_cw;

  K1P k;
  memset(&k, 0, sizeof(k));
  k.x = x; k.dww = dww; k.dwb = dwb; k.lnw = lnw; k.lnb = lnb; k.w1 = w1; k.b1 = b1; k.gb = gb; k.w2 = w2; k.b2 = b2;
  k.dbuf = dbuf; k.w1h = w1h; k.w2h = w2h; k.sumsq = ss; k.uvec = uvec; k.cvec = cvec; k.stats = stats;
  k.B = B; k.C = C; k.H = H; k.W = W; k.M = P->M; k.RT = P->RT; k.nTW = P->nTW; k.nTH = P->nTH; k.nGrp = P->nGrp;
  k.nch = P->nch; k.nrole = P->nrole; k.LD = P->LD; k.doW2 = P->bB ? 1 : 0; k.PS = P->PS;
  k.CG = P->CG; k.CS = P->CS; k.STG = P->STG; k.Wbox = P->Wbox;
  k.SL = P->SL; k.units = (P->bB ? C : 0) + P->K4; k.mask = g_k1mask;
  if (P->tma) {
    // the tensor map holds x's address, which moves every call: encode it on the host each call
    CUtensorMap tm;
    cuuint64_t gdim[3] = {(cuuint64_t)W, (cuuint64_t)H, (cuuint64_t)B * C};
    cuuint64_t gstr[2] = {(cuuint64_t)W * 4, (cuuint64_t)H * W * 4};
    cuuint32_t box[3] = {(cuuint32_t)P->Wbox, (cuuint32_t)(P->RT + 6), (cuuint32_t)P->CS};
    cuuint32_t es[3] = {1, 1, 1};
    CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, (void*)x, gdim, gstr, box, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return 400 + (int)r;
    TMap tmv;
    memcpy(&tmv, &tm, sizeof(tmv));
    const int g1 = P->nrole + P->nGrp * P->ntiles;
    k1_tma<kR><<<g1, P->thr1, P->smem1, st>>>(tmv, k);
  } else if (P->bulk) {
    if (W == 7) k1_bulk<7, 1><<<P->nconv, P->thr1, P->smem1, st>>>(k);
    else k1_bulk<14, 2><<<P->nconv, P->thr1, P->smem1, st>>>(k);
  } else {
    const int g1 = P->nrole + P->nGrp * P->ntiles;
    if (P->WT == 7) k1_dw<7><<<g1, P->thr1, P->smem1, st>>>(k);
    else if (P->WT == 14) k1_dw<14><<<g1, P->thr1, P->smem1, st>>>(k);
    else k1_dw<16><<<g1, P->thr1, P->smem1, st>>>(k);
  }
  CK(cudaGetLastError());
  if (g_mask < 2) return 0;

  const float alpha = 1.f, beta0 = 0.f, beta1 = 1.f;
  const int i1 = std::min(g_idx1, P->n1 - 1), i2 = std::min(g_idx2, P->n2 - 1);
  CB(cublasLtMatmul(g_lt, P->md1, &alpha, w1h, P->la1, dbuf, P->lb1, &beta0, y, P->ld1, y, P->ld1, &P->h1[i1].algo, cw,
                    kCublasWs, st));
  if (g_mask < 3) return 0;

  const int nb = (P->HW + P->tokb - 1) / P->tokb;
  k2_gelu<<<dim3(nb, B), P->thr2, 0, st>>>(y, uvec, cvec, stats, ss, P->HW, C, P->LD, P->tokb, P->TS2, P->nch, P->M, ln_eps);
  CK(cudaGetLastError());
  if (g_mask < 4) return 0;
  if (P->bB) {
    k3b_scale<<<dim3(nb, B), 256, 4 * P->K4 + 64, st>>>(y, gw, ss, P->HW, C, P->LD, P->tokb, grn_eps);
  } else {
    k3a_w2<<<dim3((C + 7) / 8, B), 256, 4 * P->K4 + 64, st>>>(w2, gb, b2, gw, ss, w2h, C, P->LD, grn_eps);
  }
  CK(cudaGetLastError());
  if (g_mask < 5) return 0;

  if (!P->useCL) {
    CB(cublasLtMatmul(g_lt, P->md2, &alpha, y, P->la2, w2h, P->lb2, &beta1, x, P->lc2, out, P->lc2, &P->h2[i2].algo,
                      cw, kCublasWs, st));
  } else {
    CB(cublasLtMatmul(g_lt, P->md2, &alpha, w2h, P->la2, y, P->lb2, &beta0, d2, P->lc2, d2, P->lc2, &P->h2[i2].algo,
                      cw, kCublasWs, st));
    if (g_mask < 6) return 0;
    k4_transpose_add<<<dim3((P->HW + 31) / 32, (C + 31) / 32, B), dim3(32, 8), 0, st>>>(x, d2, out, P->HW, C);
    CK(cudaGetLastError());
  }
  return 0;
}
```

```yaml design-card
id: r4-a-e1-bulkplane-wprep
parents: [r2-lib-cublaslt-fp16-tmaconv]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: oneshot
  launch: split
  tile: rows8
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  Lineage A / E1 (H2, H18). For W=7, 14 and 449x7 rows K1 stages x with one 1-D cp.async.bulk per lane (channel)
  of that channel's row range, rounded out to 16 B, into bank-skewed padded shared slots (lane/4 + 8*(addr&3) slot
  order, SL/4 odd, so odd-HW rows read conflict-free; 14x14 keeps a 4-way conflict). One warp per (output row,
  7-column segment) with the 49 taps in registers writes channels-last fp16 d and reduces the LN partials over the
  32 channels with a 16-shuffle halving tree. Every K1 CTA also converts a share of W1' = W1*lnw (and W2 plus b2' hi/lo
  when the hidden is large) with 16-byte loads and stores while the plane copies are in flight, and zeroes the GRN
  workspace. The old role CTAs and the 8 GB/s/SM LDGSTS staging are gone on these rows. The rest of the pipeline
  (cuBLASLt FP16 GEMMs, K2, K3, K4) is unchanged.
expected_effect:
  S: {pct: -10, confidence: medium}
  M: {pct: -15, confidence: medium}
  L: {pct: -20, confidence: medium}
resources_sm100a:
  regs_per_thread: 72
  smem_per_cta_bytes: 14000
  threads_per_cta: 256
  ctas_per_sm: 3
  bytes_in_flight_per_sm: 40000
  launches_per_call: 5
knobs: {RT: "fewest row tiles giving >=296 CTAs", nwmin: 8, SL: "rows*W+8 rounded to 4, SL/4 odd", nwmin_probe_value: 4}
dispatch:
  - {max_tokens: null, kernel: "k1_bulk<7,1> (W=7) | k1_bulk<14,2> (W=14) | k1_tma (W%4==0) + cublasLt + k2_gelu + k3a|k3b + cublasLt (+k4)", meta: {per-shape plan keyed on (B,C,H,W)}}
tests: [H2, H18]
findings:
  - "K1-only, harness_time on the rented B200, old K1 vs new K1 with a 4-warp minimum, interleaved x3 (spread <=0.3 us): 2,768 18.1 -> 9.7 us; 8,768 23.2 -> 12.2; 32,768 33.4 -> 14.2; 16,384 28.9 -> 16.7; 1,160,449,7 17.9 -> 5.2. Gains are 8.4 / 11.0 / 19.2 / 12.2 / 12.7 us, so the success lines (>=8 at 8,768, >=12 at 32,768) are met [probe_b200]"
  - "ablations of the 4-warp-minimum build, K1 only (conv-only / weights-only / both): 2,768 3.9 / 8.9 / 9.7; 8,768 4.3 / 11.0 / 12.2; 32,768 8.7 / 9.4 / 14.1; 16,384 13.2 / 5.9 / 16.7; 1,160 5.0 / 3.1 / 5.2. The bulk-plane conv is 4-9 us at C=768 (D reported 1.5-10). The weight pass alone is 8.9-11 us at C=768 (above the 8 us line) but overlaps the conv almost fully at 2,768 and 8,768, so the weight conversion, not the conv staging, now bounds K1 on the C=768 rows with B<=8 [probe_b200]"
  - "run_tests 16/16 PASS, rented, 4-warp-minimum build, vs the merge's A-path rows in the same container (parent-equivalent): 8,768 50.0 vs 62.7; 1,160 42.1 vs 54.3; 16,768 61.1 vs 77.6; 16,384 76.5 vs 89.1; 32,768 75.6 vs 95.2. Rows owned by D/B paths in the merge: 2,768 56.2 vs 34.2 (D), 8,384 57.3 vs 53.6 (D). Projected portal 0.6432 [run_tests]"
  - "run_tests 16/16 PASS, rented, shipped build (8-warp minimum), vs the same merge reference: 8,768 44.4 vs 62.7 (-29%); 1,160 34.9 vs 54.3 (-36%); 16,768 53.2 vs 77.6 (-31%); 16,384 71.5 vs 89.1 (-20%); 32,768 70.1 vs 95.2 (-26%); 2,768 48.5 vs 34.2 (D wins); 8,384 50.4 vs 53.6; TMA rows 1,192 27.4, 1,96 28.7, 2,96 33.6, 4,192 36.7, 4,96 44.0, 8,192 51.8, 8,320 89.4, 32,128 291.2, 64,96 1144.1. Projected portal 0.6703 +-0.008 [run_tests]"
  - "CAUTION on the 0.6703: the two run_tests runs in this session differ by 5-12 us on rows whose code did not change (TMA rows: 1,192 34.6 -> 27.4, 4,192 42.3 -> 36.7, 8,192 58.9 -> 51.8). So the nwmin 4 -> 8 gain (about -5 us on the C=768 rows) is NOT separated from cross-run drift, and the 0.6703 includes that drift. Use the 4-warp comparison and the probe K1 deltas as the real gain; rank 8 vs 4 warps with an interleaved same-container test [run_tests]"
  - "per-row gain vs the merge's A-path rows (same container): 5 rows by 12.6-24.5 us (8,768, 16,768, 32,768, 16,384, 1,160); the card success line (8-20 us on >=5 of 7 rows) is met on 5 of 7. 2,768 and 8,384 stay with D's paths in the merge. Suggested merge: this kernel for 8,768, 16,768, 32,768, 16,384, 1,160 and, if the interleaved comparison confirms it, for the TMA rows; D at 2,768 and 8,384; B on the small rows where it won [INFERRED]"
  - "bank-skewed slots: plane copies must start 16-B aligned in shared memory, so channel slots can only sit at multiples of 4 floats; with identical shifts that is a 4-way conflict minimum, which the 14x14 rows keep. For odd HW the shift (global float index & 3) equals channel mod 4 (up to a constant), so slot = lane/4 + 8*shift with SL/4 odd places the 32 lanes on 32 distinct banks [INFERRED, SASS clean: 0 spills, 70-72 regs]"
  - "per-lane 1-D bulk copies (32 small copies per CTA, 200-400 B each, outward-rounded to 16 B) were fast enough: bulk copy issue rate did not show up as a bound at any row [probe_b200]"
  - "NOT DONE: Step 0 (parent and r0 anchor interleaved x3, and the r3-a-e1 custom path vs the library path at 8,768; the r3-a-e1 source was not loaded). Step 4's 10-seeded-round fp64 precision log: not run. Numerics are unchanged from the parent: d is one fp32 conv rounded once to fp16, the same W1' rounding, and K2/K3/GEMMs untouched. run_tests passes 16/16 x 10 rounds, but the owed max |err| and beyond-atol counts are still missing for these 7 rows. Step 5 (K2 vs plain copy, H29): not run. The nwmin 4/8/12 interleaved sweep: not run [session cut by a usage limit]"
  - "dead end avoided: more warps per CTA cost registers, and at 70-72 regs a 10-14 warp CTA would fall to 1-2 CTAs/SM and a second wave at 2,768 (336 CTAs); 8 warps keeps 3 CTAs/SM = 444 slots >= 384 CTAs at the C=768 rows [INFERRED]"
  - "W2 rows (hidden >16 MB: B>=8 at C=768, B=16 at C=384) are still one warp per row with 4 sequential 768-column rounds; it is the longest chain in the weight pass at 8,768 and 16,768. Splitting a W2 row over 4 warps with a CTA-level dot reduction is the next lever here (worth about 3-5 us at those rows) [INFERRED]"
paths:
  - {max_tokens: 3200, lang: cuda, width: 128, threads: 256, rows: 7, grid: oneshot, mem: tma, launches: 5, cluster: 1}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 8, grid: oneshot, mem: tma, launches: 5, cluster: 1}
runs_on:
  H200: {runs: true, representative: false, note: "bulk copies and TMA maps work on sm_90; cuBLASLt kernels differ"}
  A100: {runs: false, representative: false}
  L40S: {runs: false, representative: false}
risks:
  - "precision unchanged from the parent but the 10-round per-row beyond-atol log is still owed; parent max |err| was 2.37-2.39e-3 against atol 2.4e-3 on three rows"
  - "the 4 -> 8 warp-minimum gain and the 0.6703 projection are mixed with 5-12 us of cross-run drift"
  - "x base pointers are assumed 16-B aligned (the harness keeps 256-B alignment) for the 1-D bulk copies; the outward rounding stays inside the tensor because B*C*H*W is a multiple of 4"
  - "g_forceold / g_nwmin / g_k1mask are bench-only setters with defaults 0 / 8 / 3; run() never calls them"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```