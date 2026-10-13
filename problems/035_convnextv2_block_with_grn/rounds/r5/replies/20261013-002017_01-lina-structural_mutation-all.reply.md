### Rationale
K2 was issue- and latency-bound, not stream-bound. With no math at all, a register-prefetch K2 floors at 292 µs against a 187 µs copy at 64,96. I replaced it with a bulk-copy ring: one 1-D bulk copy per token row into shared memory, with the stats computed up front. The GELU is now A&S erfc (about 13 instructions, 2 MUFU). The W2 conversion is split over 4-warp groups for M ≤ 800. The 10-round fp64 precision log is delivered: 0 beyond-atol on all 16 rows, for both the parent and this kernel. Two things to check:
- **Operator gate not met:** the custom erf is worse than `erff` (max |h err| 1.2e-6 against 4.9e-7).
- **K2 targets mostly missed:** only 8,320 met its target.

```json solution-spec
{"name": "r5-a-e2-ringk2", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "description": "Lineage A lean passes: bulk-ring stream K2 (A&S erfc exact-erf GELU, fp16 hidden in place), W2 conversion split over 4-warp groups for M<=800", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda", "-lcublasLt"]}}}
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
  int w2split;            // bulk-plane path: W2 rows split over 4-warp groups
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

// One quarter (columns [q*C, (q+1)*C) of the 4C) of W2 row c -> fp16. The four warps of a group each take a quarter, all
// loads issued before use; the partial W2.gb dots meet in shared memory behind a named barrier of the group, and the
// quarter-0 warp writes b2' as hi/lo columns 4C, 4C+1 exactly like w2_row.
static __device__ void w2_quarter(const K1P& p, int c, int q, int lane, float* red, int bid) {
  const int K4 = 4 * p.C, QL = p.C;
  const float* wr = p.w2 + (size_t)c * K4 + (size_t)q * QL;
  const float* gq = p.gb + (size_t)q * QL;
  __half* o = p.w2h + (size_t)c * p.LD + (size_t)q * QL;
  float4 wa[3], wb[3], ga[3], gc[3];
#pragma unroll
  for (int it = 0; it < 3; ++it) {
    const int k8 = lane * 8 + it * 256;
    wa[it] = make_float4(0.f, 0.f, 0.f, 0.f);
    wb[it] = wa[it]; ga[it] = wa[it]; gc[it] = wa[it];
    if (k8 < QL) { wa[it] = ld4(wr + k8); wb[it] = ld4(wr + k8 + 4); ga[it] = ld4(gq + k8); gc[it] = ld4(gq + k8 + 4); }
  }
  float dot = 0.f;
#pragma unroll
  for (int it = 0; it < 3; ++it) {
    const int k8 = lane * 8 + it * 256;
    if (k8 < QL) {
      dot += wa[it].x * ga[it].x + wa[it].y * ga[it].y + wa[it].z * ga[it].z + wa[it].w * ga[it].w +
             wb[it].x * gc[it].x + wb[it].y * gc[it].y + wb[it].z * gc[it].z + wb[it].w * gc[it].w;
      *reinterpret_cast<uint4*>(o + k8) =
          make_uint4(pack_h2(wa[it].x, wa[it].y), pack_h2(wa[it].z, wa[it].w), pack_h2(wb[it].x, wb[it].y),
                     pack_h2(wb[it].z, wb[it].w));
    }
  }
  dot = warp_sum(dot);
  if (lane == 0) red[q] = dot;
  asm volatile("bar.sync %0, 128;" :: "r"(bid) : "memory");
  if (q == 0 && lane < 8) {
    const float bb = p.b2[c] + ((red[0] + red[1]) + (red[2] + red[3]));
    __half hi = __float2half_rn(bb);
    __half v = lane == 0 ? hi : (lane == 1 ? __float2half_rn(bb - __half2float(hi)) : __float2half_rn(0.f));
    p.w2h[(size_t)c * p.LD + K4 + lane] = v;
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
  __shared__ float redq[2][3][4];
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
    if (p.w2split) {
      const int ng = nw >> 2, grp = warp >> 2;
      if (p.doW2 && grp < ng) {
        int par = 0;
        for (int c = blockIdx.x * ng + grp; c < C; c += gridDim.x * ng) {
          w2_quarter(p, c, warp & 3, lane, redq[par][grp], 1 + grp);
          par ^= 1;
        }
      }
      for (int u = gwid; u < 4 * C; u += nwt) w1_row(p, u, lane);
    } else {
      const int nW2 = p.doW2 ? C : 0;
      for (int u = gwid; u < p.units; u += nwt) {
        if (u < nW2) w2_row(p, u, lane);
        else w1_row(p, u - nW2, lane);
      }
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
// (Fallback for hidden < 1 MB only: 2,768,7,7.)
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

// ---- stream-rate K2: bulk-copy ring -------------------------------------------------------------------------------
// h = z * Phi(z) with the exact-erf GELU, written as h = max(z,0) - |z| q, q = Phi(-|z|) = 0.5 erfc(|z|/sqrt2), and
// erfc(u) from Abramowitz-Stegun 7.1.26: erfc(u) = (a1 t + .. + a5 t^5) exp(-u^2), t = 1/(1 + 0.3275911 u). With tq = t/2 the
// polynomial coefficients are a_i 2^(i-1), so q = tq S(tq) exp(-z^2/2): one rcp, one ex2, 4 FMA for S. Max |h err| about
// 1.2e-6 vs fp64 for |z| <= 12 (fp16 rounding of h is >= 1e-4 relative), 13 instructions.
static __device__ __forceinline__ float ex2a(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
static __device__ __forceinline__ float rcpa(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
static __device__ __forceinline__ float gelu_fast(float z) {
  const float a = fabsf(z);
  const float tq = rcpa(fmaf(0.46327822f, a, 2.f));
  float s = 16.98248686f;
  s = fmaf(s, tq, -11.625216216f);
  s = fmaf(s, tq, 5.685654964f);
  s = fmaf(s, tq, -0.568993472f);
  s = fmaf(s, tq, 0.254829592f);
  const float e = ex2a(-(0.72134752f * a) * a);
  const float w = (s * tq) * e;
  return fmaf(-a, w, fmaxf(z, 0.f));
}
static __device__ __forceinline__ void st_pol(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
}

// CTA = (run of TR tokens of image b = blockIdx.y). The run's rows of the (in place) fp16 hidden stream through an NS-stage
// ring of TT = 4*TS token rows per stage (one 1-D bulk copy per row, mbarrier per stage), so the loads in flight do not
// depend on registers. Thread = fixed 8 hidden channels (one uint4 of the row) x one of TS token lanes, 4 tokens per stage;
// the GRN sum of squares stays in registers across the run and leaves with one float4 atomic pair per 8 channels per CTA.
// The per-token LayerNorm statistics for the whole run are computed first into shared memory (overlapped with the first
// copies). hint: 0 normal, 1 evict_last, 2 evict_first for the hidden stores (our own scratch buffer).
__global__ void __launch_bounds__(384, 2) k2_ring(__half* Y, const float* uvec, const float* cvec, const float2* stats,
                                                  float* sumsq, int HW, int C, int LD, int TR, int nch, int M,
                                                  float ln_eps, int hint, int NS) {
  extern __shared__ __align__(128) unsigned char k2sm[];
  __shared__ __align__(8) unsigned long long k2bar[8];
  const int V = C >> 1, K4 = 4 * C, RB = 8 * C;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, kc = tid % V, ts = tid / V, TS = blockDim.x / V;
  const int TT = 4 * TS, stageB = TT * RB;
  const int b = blockIdx.y, t0 = blockIdx.x * TR, t1 = min(t0 + TR, HW), ntok = t1 - t0;
  const int nst = (ntok + TT - 1) / TT;
  float2* ms = reinterpret_cast<float2*>(k2sm + NS * stageB);
  const char* gbase = reinterpret_cast<const char*>(Y) + (size_t)b * HW * LD * 2;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(&k2bar[s])));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (warp == 0) {
    for (int it = 0; it < min(NS, nst); ++it) {
      const int s = it % NS, first = it * TT, rows = min(TT, ntok - first);
      if (lane == 0) expect_tx(&k2bar[s], rows * RB);
      __syncwarp();
      for (int r = lane; r < rows; r += 32)
        bulk_g2s(k2sm + s * stageB + r * RB, gbase + (size_t)(t0 + first + r) * LD * 2, RB, &k2bar[s]);
    }
  }
  uint64_t pol;
  if (hint == 1) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  else if (hint == 2) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pol));
  float uu[8], cc[8];
  {
    float4 a = __ldg(reinterpret_cast<const float4*>(uvec + kc * 8));
    float4 c = __ldg(reinterpret_cast<const float4*>(uvec + kc * 8 + 4));
    uu[0] = a.x; uu[1] = a.y; uu[2] = a.z; uu[3] = a.w; uu[4] = c.x; uu[5] = c.y; uu[6] = c.z; uu[7] = c.w;
    a = __ldg(reinterpret_cast<const float4*>(cvec + kc * 8));
    c = __ldg(reinterpret_cast<const float4*>(cvec + kc * 8 + 4));
    cc[0] = a.x; cc[1] = a.y; cc[2] = a.z; cc[3] = a.w; cc[4] = c.x; cc[5] = c.y; cc[6] = c.z; cc[7] = c.w;
  }
  const float invC = 1.f / (float)C;
  for (int tl = tid; tl < ntok; tl += blockDim.x) {
    float s = 0.f, q = 0.f;
    const float2* sb = stats + (size_t)b * HW + t0 + tl;
    for (int ch = 0; ch < nch; ++ch) { const float2 v = sb[(size_t)ch * M]; s += v.x; q += v.y; }
    const float mean = s * invC;
    const float var = fmaxf(q * invC - mean * mean, 0.f);
    const float rs = rsqrtf(var + ln_eps);
    ms[tl] = make_float2(rs, rs * mean);
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int it = 0; it < nst; ++it) {
    const int s = it % NS;
    mbar_wait(&k2bar[s], (uint32_t)((it / NS) & 1));
    const uint4* sp = reinterpret_cast<const uint4*>(k2sm + s * stageB) + kc;
    uint4 v[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) v[jj] = sp[(ts + jj * TS) * V];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int tl = it * TT + ts + jj * TS;
      if (tl < ntok) {
        const float2 m = ms[tl];
        const __half2* hp = reinterpret_cast<const __half2*>(&v[jj]);
        uint32_t o[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float2 f = __half22float2(hp[q]);
          const float h0 = gelu_fast(fmaf(m.x, f.x, fmaf(-m.y, uu[2 * q], cc[2 * q])));
          const float h1 = gelu_fast(fmaf(m.x, f.y, fmaf(-m.y, uu[2 * q + 1], cc[2 * q + 1])));
          acc[2 * q] = fmaf(h0, h0, acc[2 * q]);
          acc[2 * q + 1] = fmaf(h1, h1, acc[2 * q + 1]);
          o[q] = pack_h2(h0, h1);
        }
        char* rowp = const_cast<char*>(gbase) + (size_t)(t0 + tl) * LD * 2;
        st_pol(rowp + kc * 16, make_uint4(o[0], o[1], o[2], o[3]), pol);
        if (kc == 0) st_pol(rowp + K4 * 2, make_uint4(0x3C003C00u, 0u, 0u, 0u), pol);
      }
    }
    __syncthreads();
    if (warp == 0 && it + NS < nst) {
      const int first = (it + NS) * TT, rows = min(TT, ntok - first);
      if (lane == 0) expect_tx(&k2bar[s], rows * RB);
      __syncwarp();
      for (int r = lane; r < rows; r += 32)
        bulk_g2s(k2sm + s * stageB + r * RB, gbase + (size_t)(t0 + first + r) * LD * 2, RB, &k2bar[s]);
    }
  }
  float* red = reinterpret_cast<float*>(k2sm);
  if (TS > 1 && ts > 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) red[(ts - 1) * K4 + kc * 8 + i] = acc[i];
  }
  __syncthreads();
  if (ts == 0) {
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
  bool k2ok;
  int k2thr, k2NS, k2nruns, k2TR;
  size_t k2smem;
  size_t off_d, off_y, off_w1h, off_w2h, off_ss, off_uv, off_st, off_d2, off_cw, total;
  cublasLtMatmulDesc_t md1, md2;
  cublasLtMatrixLayout_t la1, lb1, ld1, la2, lb2, lc2;
  cublasLtMatmulHeuristicResult_t h1[8], h2[8];
  int n1, n2;
};

static cublasLtHandle_t g_lt = nullptr;
static std::map<uint64_t, Plan*> g_plans;
// g_mask / g_k1mask: stop after a launch / ablate K1 halves; g_forceold: old K1 for W=7/14; g_nwmin: min warps per K1 CTA;
// g_k2mode: -1 auto, 0 old K2, 1 ring K2; g_w2split: W2 rows over 4-warp groups; g_hint: -1 auto, 0/1/2 hidden store hint
// (all are diagnostics for the bench; the defaults are what ships)
static int g_idx1 = 0, g_idx2 = 0, g_mask = 99, g_k1mask = 3, g_forceold = 0, g_nwmin = 8;
static int g_k2mode = -1, g_w2split = 1, g_hint = -1;

extern "C" void convblk_set_idx(int a, int b) { g_idx1 = a; g_idx2 = b; }
extern "C" void convblk_set_mask(int m) { g_mask = m; }
extern "C" void convblk_set_k1mask(int m) { g_k1mask = m; }
extern "C" void convblk_set_forceold(int m) { g_forceold = m; }
extern "C" void convblk_set_nwmin(int m) { g_nwmin = std::max(1, std::min(m, 14)); }
extern "C" void convblk_set_k2mode(int m) { g_k2mode = m; }
extern "C" void convblk_set_w2split(int m) { g_w2split = m; }
extern "C" void convblk_set_hint(int m) { g_hint = m; }

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
  {
    // ring K2 geometry: TS token lanes of V = C/2 threads, TT = 4*TS rows per stage; runs of whole stages of one image,
    // about 2664 CTAs' worth over the input (long runs), but at least min(M/444, 4 stages) tokens so small inputs stay one wave
    const int V = C / 2;
    const int TS = std::max(1, (256 + V / 2) / V);
    const int thr = V * TS, TT = 4 * TS;
    const size_t stageB = (size_t)TT * 8 * C;
    P->k2ok = (C % 2 == 0) && thr <= 384;
    if (P->k2ok) {
      P->k2NS = stageB > 16000 ? 3 : 4;
      int tr = std::max((M + 2663) / 2664, std::min((M + 443) / 444, 4 * TT));
      tr = ((tr + TT - 1) / TT) * TT;
      tr = std::min(tr, ((HW + TT - 1) / TT) * TT);
      P->k2TR = tr;
      P->k2nruns = (HW + tr - 1) / tr;
      P->k2thr = thr;
      P->k2smem = (size_t)P->k2NS * stageB + (size_t)tr * 8;
      CK(raise_smem((const void*)k2_ring, (int)P->k2smem));
    }
  }
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
  // W2 split over 4-warp groups: measured -1.0..-1.4 us of K1 at 8,768 / 16,768 (M <= 800), +0.3..+0.7 at 16,384 / 32,768
  k.w2split = (g_w2split && P->bulk && P->thr1 >= 128 && P->M <= 800) ? 1 : 0;
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
  const size_t hidden_bytes = (size_t)P->M * P->LD * 2;
  // ring K2 was neutral (+-0.1 us) at 1,192 / 2,768 / 8,768 and 1.7-125 us faster on every other row
  const bool newk2 = P->k2ok && (g_k2mode == 1 || (g_k2mode < 0 && hidden_bytes >= ((size_t)1 << 20)));
  if (newk2) {
    // hidden stores (our own scratch): evict_last while the hidden fits L2 comfortably (GEMM2 re-reads it), evict_first above
    const int hint = g_hint >= 0 ? g_hint : (hidden_bytes <= ((size_t)42 << 20) ? 1 : 2);
    k2_ring<<<dim3(P->k2nruns, B), P->k2thr, P->k2smem, st>>>(y, uvec, cvec, stats, ss, P->HW, C, P->LD, P->k2TR, P->nch,
                                                              P->M, ln_eps, hint, P->k2NS);
  } else {
    k2_gelu<<<dim3(nb, B), P->thr2, 0, st>>>(y, uvec, cvec, stats, ss, P->HW, C, P->LD, P->tokb, P->TS2, P->nch, P->M, ln_eps);
  }
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
id: r5-a-e2-ringk2
parents: [r4-a-e1-bulkplane-wprep]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: oneshot
  launch: split
  tile: token
  red: cta
  cache: stream
  spec: dispatch:size
hypothesis: >-
  Lineage A / E2 (H29). K2 (GELU + GRN sum of squares over the fp16 hidden) was issue- and latency-bound, not
  stream-bound. A register-prefetch K2 floors at 292 us with no math at 64,96 against a 187 us copy. Moving the hidden
  rows into a shared-memory ring (one 1-D bulk copy per token row, 3-4 stages of 4*TS rows, run-long CTAs of one
  image, per-run LN stats in shared memory) floors at 212 us. Replacing erff with a 13-instruction A&S erfc form
  (rcp + ex2 + 4 FMA) cuts the math term. K2 goes 444 -> 319 us at 64,96, 118 -> 80 at 32,128, 37 -> 21.5 at 8,320.
  The W2 conversion in K1 is split over 4-warp groups for M <= 800 (about -1 us). Targets only partly met; see findings.
expected_effect:
  S: {pct: -4, confidence: medium}
  M: {pct: -9, confidence: medium}
  L: {pct: -11, confidence: medium}
resources_sm100a:
  regs_per_thread: 79
  smem_per_cta_bytes: 66000
  threads_per_cta: 240
  ctas_per_sm: 3
  bytes_in_flight_per_sm: 120000
  launches_per_call: 5
knobs: {ring_stages: "4 (3 when a stage > 16 KB)", rows_per_stage: "4*TS", ring_threshold_bytes: 1048576, w2split_max_M: 800, store_hint: "evict_last <= 42 MB hidden, evict_first above"}
dispatch:
  - {max_tokens: null, kernel: "k1 (tma | bulk, w2split for M<=800) + cublasLt + k2_ring (hidden >= 1 MB) | k2_gelu + k3a|k3b + cublasLt (+k4)", meta: {per-shape plan keyed on (B,C,H,W)}}
tests: [H29]
findings:
  - "STEP 0 precision log, DELIVERED. fp64 reference (the problem's reference run on .double() inputs), 10 rounds per row, inputs regenerated each round with the harness generator rules (randn, weight matrices /sqrt(fan_in), LN weight ones, LN bias zeros, biases randn; set_seed(200) or manual_seed(200) once per process; not bit-identical to the harness stream). Parent as shipped (old K2, old W2 pass) and this kernel, same inputs. Per row: atol, max|err| parent / new, beyond-atol count (atol + 1e-5|ref|) parent / new. 2,96: 2.7e-3, 2.249e-3 / 2.249e-3, 0 / 0. 8,320: 2.9e-3, 2.154e-3 / 2.160e-3, 0 / 0. 8,768: 2.9e-3, 2.206e-3 / 2.206e-3, 0 / 0. 64,96: 3.8e-3, 2.609e-3 / 2.603e-3, 0 / 0. 1,160: 2.6e-3, 2.203e-3 / 2.203e-3, 0 / 0. 16,768: 2.9e-3, 2.193e-3 / 2.193e-3, 0 / 0. 16,384: 3.3e-3, 2.322e-3 / 2.316e-3, 0 / 0. 32,128: 3.2e-3, 2.269e-3 / 2.268e-3, 0 / 0. 4,96: 2.8e-3, 2.256e-3 / 2.256e-3, 0 / 0. 1,96: 2.7e-3, 2.427e-3 / 2.427e-3, 0 / 0 (tightest: 90% of atol). 8,384: 2.8e-3, 2.095e-3 / 2.095e-3, 0 / 0. 4,192: 2.5e-3, 2.142e-3 / 2.142e-3, 0 / 0. 1,192: 2.4e-3, 1.998e-3 / 1.998e-3, 0 / 0. 2,768: 2.6e-3, 1.871e-3 / 1.871e-3, 0 / 0. 32,768: 2.9e-3, 2.389e-3 / 2.387e-3, 0 / 0. 8,192: 3.0e-3, 2.147e-3 / 2.145e-3, 0 / 0. All 16 rows have 0 beyond-atol elements, so no hi/lo split is needed. Max error is unchanged by the new K2/erf (differences <= 6e-6) [probe_b200]"
  - "STEP 1 copy probe: plain 8-uint4-per-thread read+write copy after the 2xL2 zero-fill, 3 reps: 616.6 MB 186.8 us (6.6 TB/s r+w), 134.2 MB 44.9 us (6.0 TB/s), 21.0 MB 10.0 us (4.2 TB/s). Parent K2 (same container, mask3-mask2): 443-445 / 117.8 / 36.6-39.4 us = 42% / 38% / 27% of the copy rate, so the refutation clause (>= 80%) did NOT fire: K2 was far from the stream rate [probe_b200]"
  - "register-prefetch K2 structure (NVRTC microbench, 64,96 shapes, cold L2): no-math floor 292-295 us (guarded or unguarded loops); with the old NR-style 10-coefficient erfc 354-438 us; with A&S 322-388 us. In-flight bytes per SM are bound by registers (about 92 KB at 3 CTAs x 240 thr x 8 uint4) and drop during compute phases. The shared-memory bulk ring floors at 212-215 us (no math) and 316 us with A&S, so about 100 us of the remaining time is math issue [probe_b200]"
  - "MUFU sensitivity (ring, 64,96): full A&S 317 us, rcp replaced by an FMA 309, ex2 replaced by an FMA 300, both replaced 288: not MUFU-bound; plain FMA/INT issue is the remaining term [probe_b200]"
  - "erf accuracy vs fp64 over [-12, 12], 2^26 points, max |h err| (h = z*Phi(z)): erff-based 4.86e-7; NR erfc (10 coefficients, ex2 form) 2.38e-7; A&S 7.1.26 form shipped 1.16e-6 (at z=1.18). NOT within the operator's 2e-7-vs-erff gate (1.2e-6 vs 4.9e-7); the gate was not met and the NR 10-coefficient form (2.4e-7, +4 FFMA per element, about +30 us at 64,96 in the register structure) is the in-gate fallback. End-to-end the new K2 changes the max output error by <= 6e-6 and beyond-atol stays 0 on all rows (step 0), because fp16 rounding of h (>= 2.4e-4 relative) dominates [probe_b200]"
  - "STEP 2 K2-only (mask3-mask2, interleaved x3, spread <= 2 us), old -> ring: 64,96 444 -> 319 us (-28%; target <= 260 missed); 32,128 117.8 -> 79.5 (-33%; <= 65 missed); 8,320 36.6-39.4 -> 20.6-22.6 (target <= 22 met); 4,96 14.6-16.8 -> 11.5-14.1; 8,192 18.7 -> 12.5; 16,384 24.9 -> 10.8. Ring K2 is 59% / 56% / 47% of the copy rate at the three target rows. Refutation clause (within 15% of old at 64,96 and 32,128) did not fire [probe_b200]"
  - "full pipeline, parent vs new, interleaved x3 in one container (harness_time, spread <= 0.5 us): 64,96 1155 -> 1029 us (target <= 960 missed); 32,128 301 -> 260 (<= 245 missed); 8,320 96 -> 80 (<= 75 missed); 16,384 76.9 -> 64.1; 8,192 59.7 -> 53.7; 4,96 50.4 -> 45.9; 32,768 76.8 -> 71.2; 16,768 61.2 -> 60.8 (ring + w2split); rows on the old K2 path unchanged (1,192 34.0; 1,160 42.2 -> 40.1 once the ring threshold is 1 MB) [probe_b200]"
  - "ring K2 on the small rows (full pipeline, old vs ring, x3): 1,192 -0.1; 1,96 -1.7; 1,160 -2.2; 2,96 -2.0; 4,192 -3.0; 8,384 -6.2; 2,768 +0.1; 8,768 +0.1; 16,768 -4.1 us. Hence the shipped threshold (ring when the hidden is >= 1 MB; 2,768 keeps the old K2) [probe_b200]"
  - "STEP 3 W2 split over 4-warp groups (named barrier per group, CTA-level reduction of the W2.gb dots, 16-B loads, all issued before use), K1 only, split off -> on: weights-only 2,768 7.8 -> 7.7; 8,768 10.1 -> 8.6; 16,768 10.1 -> 8.6; 32,768 9.6 -> 8.7; 16,384 6.1 -> 5.5; full K1 8,768 11.4 -> 10.0; 16,768 12.0 -> 11.0; 32,768 14.0 -> 14.7; 16,384 16.1 -> 16.4. Target (weights-only <= 6 us at C=768, 8,768 and 16,768 gaining >= 3 us) NOT met: the gain is 1.0-1.5 us, so the W2 chain is not the only term of the 8.6 us weight pass (W1 rows, 3072 x 768 fp32, are the other half). Shipped for M <= 800 only (8,768, 16,768, 2,768); off elsewhere [probe_b200]"
  - "store hints on the hidden (our own scratch), full pipeline, auto / normal / evict_last / evict_first: 64,96 1029.3 / 1028.9 / 1030.2 / 1029.0; 32,128 260.2 / 264.5 / 264.4 / 260.6; 8,320 79.8 / 80.0 / 79.8 / 80.9; 16,384 64.0 / 62.9 / 62.9 / 63.5; 4,96 46.5 on all. Only 32,128 shows an effect (evict_first -4 us over normal/last, auto already picks evict_first above 42 MB); elsewhere within 1 us noise. Kept auto [probe_b200]"
  - "STEP 4 run_tests 16/16 PASS (portal-order check included), rented, this vs the r4 merge in the same container: 2,96 37.5 vs 36.8; 8,320 78.6 vs 95.4; 8,768 48.1 vs 49.9; 64,96 1026.6 vs 1150.0; 1,160 39.6 vs 42.1; 16,768 56.1 vs 61.2; 16,384 62.1 vs 77.1; 32,128 258.9 vs 298.1; 4,96 47.0 vs 50.9; 1,96 32.6 vs 27.8; 8,384 50.2 vs 54.0; 4,192 38.9 vs 43.5; 1,192 34.4 vs 30.3; 2,768 54.5 vs 34.2; 32,768 70.2 vs 76.8; 8,192 52.7 vs 58.9. Bands S +2.8% / M -0.3% / L -10.5% against the merge, whose 1,192 / 1,96 / 2,768 rows are B and D paths (A loses there; use this kernel on the other 13 rows). Emulator: 0.6690 +- 0.008 for this kernel alone (merge 0.6553), more with the merge's B/D rows kept [run_tests]"
  - "NOT DONE: Step 5 (nwmin 4 vs 8). The parent was interleaved x3 through a ctypes harness_time harness (setters in the library), not through run_tests, whose control is the r4 merge [probe_b200]"
  - "pitfalls: probe_b200 /tmp persisted across several probes but vanished once, so keep scripts self-contained; a background job (Popen, start_new_session) survives the probe and can run a long precision+timing job; cuLaunchKernel with > 48 KB dynamic smem from NVRTC needs cuFuncSetAttribute on k.fn first; harness_time CUPTI drops the first kernel intermittently (retry) [probe_b200]"
  - "dead end: K2 is not stream-bound even with ring loads. The math term (about 100 us at 64,96) is plain FMA issue (13 instr erf + 7 other per element); removing both MUFU ops only saves 29 us. Getting to the 187 us copy rate needs the GELU in a tensor-core GEMM epilogue (lineage E) or a cheaper per-element form [probe_b200]"
paths:
  - {max_tokens: 3200, lang: cuda, width: 128, threads: 256, rows: 7, grid: oneshot, mem: tma, launches: 5, cluster: 1}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 8, grid: oneshot, mem: tma, launches: 5, cluster: 1}
runs_on:
  H200: {runs: true, representative: false, note: "bulk copies and TMA maps work on sm_90; cuBLASLt kernels differ"}
  A100: {runs: false, representative: false}
  L40S: {runs: false, representative: false}
risks:
  - "the custom GELU erf (A&S, max |h err| 1.2e-6) is outside the operator's 2e-7 gate; it needs an operator decision. The in-gate fallback is the NR 10-coefficient form in gelu_fast (2.4e-7), about +30 us at 64,96 and +4 FFMA per element elsewhere"
  - "precision margin is unchanged from the parent (tightest row 1,96: 2.427e-3 against atol 2.7e-3); the log uses regenerated inputs on the harness rules, not the harness's own RNG stream"
  - "k2_ring assumes the hidden rows are 16-B aligned (LD*2 = 8C+16 bytes and 256-B aligned workspace offsets: true for every C that is a multiple of 2)"
  - "dynamic smem for k2_ring is up to about 75 KB per CTA; raise_smem only raises it, so portal order is safe (run_tests portal-order check passed)"
  - "ring K2 is neutral at 1,192 / 8,768 and the small rows gain only 1.7-6 us; the 1,192 / 1,96 / 2,768 rows still belong to the B/D paths in the merge"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```