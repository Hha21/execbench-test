### Rationale
Lineage B, E2 (tests H12). The legacy `mma.sync` FP16 path runs at 2048 FLOP/clk/SM, which is 0.455 PF/s at the 1.5 GHz portal lock and a quarter of the tcgen05 rate. That is enough to move the bottleneck off the GEMM maths everywhere except the largest size.

The kernel is still one cooperative launch with two grid syncs. Phase 1 converts the weights to fp16 in their native [N][K] layout, which is already what `mma.sync` wants for B, so no transpose is needed. In the same pass it computes row sums, folding LayerNorm into the GEMM1 epilogue (`y = r·(acc − μ·s1) + c1`) and `gb` into `c2`. The depthwise conv writes raw fp16 `d`, using smem planes for 7×7/14×14 and 8-channel tiles where 32-channel tiles would leave SMs idle.

Both GEMMs run 4-stage cp.async rings with ldmatrix fragments. GEMM2 applies the GRN scale in an smem pass over each A stage.

Measured: the stubbed floor F is 4.6 µs. All 16 workloads pass. Times are 28–55 µs on the S rows, 48 µs at 2,768,7,7, 180 µs at 32,768,7,7 and 1787 µs at the largest. Projected score 0.561 (anchor 0.304).

```json solution-spec
{"name": "r1-convnext-coop-hmma-ring", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "description": "One cooperative launch: dwconv (+LN partials) -> FP16 mma.sync GEMM1 with LN folded into the epilogue, GELU, GRN sums -> FP16 mma.sync GEMM2 with the GRN scale applied in smem, residual.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, FP16 mma.sync GEMMs (fp32 accumulate).
//  P1: weights -> fp16 (native [N][K] = mma col-major B; lnw folded into W1) with row sums s1, c1, c2;
//      depthwise 7x7 conv -> xh fp16 [C][Mp] (raw d), LN partial sums per (token, 8*CPW-channel chunk); zero GRN sums
//  P2: hidden fp16 [4C][Mp] = gelu(rstd*(xh @ W1'^T - mean*s1) + c1)  (LayerNorm folded into the epilogue);
//      sumsq[b][k] += hidden^2
//  P3: out = x + (hidden * s_b) @ W2^T + c2,  s_b[k] = 1 + gw[k]*nf[b][k], c2 = b2 + W2 gb  (split-K atomics at small M)
//  GEMM main loops: S-stage cp.async rings for A and B, ldmatrix fragments, mma.sync.m16n8k16.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define NT 256
#define MAXIMG 5
#define RING_BYTES 81920
#define TABK 1024
#define AUX_FLOATS (MAXIMG * TABK)
#define MISC_FLOATS (MAXIMG * 128 + 8 + 64 + 4 * 128)
#define SMEM_BYTES (RING_BYTES + AUX_FLOATS * 4 + MISC_FLOATS * 4)
#define PLANE_FLOATS 676
#define P1_SLOTS 512

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* xh; float* part; void* hidden; float* sumsq; void* w1h; void* w2h; float* s1; float* c1; float* c2;
  int B, C, H, W, HW, M, Mp, C4, nparts;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb, p1_cpw;
  int cfg2, cfg3, split3;
  int mask, force_split;
};

__device__ __forceinline__ float gelu_f(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ unsigned smem_u32(const void* ptr) { return (unsigned)__cvta_generic_to_shared(ptr); }
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  unsigned r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float2 unpack2(unsigned u) {
  __half2 h = *reinterpret_cast<__half2*>(&u);
  return __half22float2(h);
}
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }
__device__ __forceinline__ void ldsm_x4(unsigned a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(unsigned a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(unsigned a, unsigned& r0, unsigned& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------- phase 1
// one warp per weight row: W1' = fp16(lnw * W1), s1 = sum_k W1'[n][k], c1 = b1 + W1 lnb; W2h = fp16(W2), c2 = b2 + W2 gb
__device__ void weights_pass(const Params& p, int wid, int wstride) {
  const int lane = threadIdx.x & 31;
  const int C = p.C, C4 = p.C4;
  for (int row = wid; row < C4 + C; row += wstride) {
    const bool first = row < C4;
    const int L = first ? C : C4;
    const float* src = first ? p.w1 + (size_t)row * C : p.w2 + (size_t)(row - C4) * C4;
    __half* dst = first ? reinterpret_cast<__half*>(p.w1h) + (size_t)row * C
                        : reinterpret_cast<__half*>(p.w2h) + (size_t)(row - C4) * C4;
    const float* dv = first ? p.lnb : p.gb;
    float ssum = 0.f, dsum = 0.f;
#pragma unroll 4
    for (int i = lane * 8; i < L; i += 256) {
      float4 a = __ldg(reinterpret_cast<const float4*>(src + i)), b = __ldg(reinterpret_cast<const float4*>(src + i + 4));
      const float4 da = __ldg(reinterpret_cast<const float4*>(dv + i)), db = __ldg(reinterpret_cast<const float4*>(dv + i + 4));
      dsum += a.x * da.x + a.y * da.y + a.z * da.z + a.w * da.w + b.x * db.x + b.y * db.y + b.z * db.z + b.w * db.w;
      if (first) {
        const float4 sa = __ldg(reinterpret_cast<const float4*>(p.lnw + i)), sb = __ldg(reinterpret_cast<const float4*>(p.lnw + i + 4));
        a.x *= sa.x; a.y *= sa.y; a.z *= sa.z; a.w *= sa.w; b.x *= sb.x; b.y *= sb.y; b.z *= sb.z; b.w *= sb.w;
      }
      uint4 o; o.x = pack2(a.x, a.y); o.y = pack2(a.z, a.w); o.z = pack2(b.x, b.y); o.w = pack2(b.z, b.w);
      *reinterpret_cast<uint4*>(dst + i) = o;
      if (first) {
        const float2 f0 = unpack2(o.x), f1 = unpack2(o.y), f2 = unpack2(o.z), f3 = unpack2(o.w);
        ssum += f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
      }
    }
    ssum = warp_sum(ssum); dsum = warp_sum(dsum);
    if (lane == 0) {
      if (first) { p.s1[row] = ssum; p.c1[row] = __ldg(p.b1 + row) + dsum; }
      else p.c2[row - C4] = __ldg(p.b2 + row - C4) + dsum;
    }
  }
}

// 4 rows x TW cols output patch of one channel; wt = 49 weights (registers)
template <int TW>
__device__ __forceinline__ void conv_patch(const float* __restrict__ xc, const float* wt, int H, int W, int h0, int w0,
                                           float (&acc)[4][TW]) {
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int cc = 0; cc < TW; cc++) acc[r][cc] = 0.f;
#pragma unroll
  for (int i = 0; i < 10; i++) {
    const int hh = h0 - 3 + i;
    const bool rowok = (hh >= 0) && (hh < H);
    float v[TW + 6];
    if (TW == 4) {
      const float* rp = xc + hh * W;
      float4 L = make_float4(0.f, 0.f, 0.f, 0.f), Cq = L, R = L;
      if (rowok) {
        if (w0 >= 4) L = __ldg(reinterpret_cast<const float4*>(rp + w0 - 4));
        Cq = __ldg(reinterpret_cast<const float4*>(rp + w0));
        if (w0 + 4 < W) R = __ldg(reinterpret_cast<const float4*>(rp + w0 + 4));
      }
      v[0] = L.y; v[1] = L.z; v[2] = L.w; v[3] = Cq.x; v[4] = Cq.y; v[5] = Cq.z; v[6] = Cq.w;
      v[7] = R.x; v[8] = R.y; v[9] = R.z;
    } else {
#pragma unroll
      for (int jj = 0; jj < TW + 6; jj++) {
        const int ww = w0 - 3 + jj;
        v[jj] = (rowok && ww >= 0 && ww < W) ? __ldg(xc + hh * W + ww) : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int ki = i - r;
      if (ki >= 0 && ki < 7) {
#pragma unroll
        for (int cc = 0; cc < TW; cc++)
#pragma unroll
          for (int j = 0; j < 7; j++) acc[r][cc] = fmaf(v[cc + j], wt[ki * 7 + j], acc[r][cc]);
      }
    }
  }
}

// patch tiles: tile = (image, 8*CPW-channel chunk, 4*rg rows, lpr*TW cols); warp = CPW channels; lane = 4 x TW patch
template <int TW>
__device__ void phase1_patch(const Params& p, float* smem, float* aux) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, Mp = p.Mp, nparts = p.nparts, CPW = p.p1_cpw, TC = 8 * CPW;
  float* st_s = smem;   // [8][P1_SLOTS][2]
  float* wsm = aux;     // [TC][49]
  __half* xh = reinterpret_cast<__half*>(p.xh);
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg, rpt = lpr * TW;
  const int rgi = lane / lpr, cgi = lane - rgi * lpr;
  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    const int per = nparts * hb * wb;
    const int b = tile / per; int rem = tile - b * per;
    const int cch = rem / (hb * wb); rem -= cch * (hb * wb);
    const int hbi = rem / wb, wbi = rem - hbi * wb;
    for (int i = tid; i < 8 * rows_per_tile * rpt * 2; i += NT) st_s[i] = 0.f;
    for (int i = tid; i < TC * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * TC * 49 + i);
    __syncthreads();
    const int h0 = hbi * rows_per_tile + rgi * 4, w0 = (wbi * lpr + cgi) * TW;
    const bool lane_ok = (rgi < rg) && (w0 < W);
    float* sts = st_s + warp * rows_per_tile * rpt * 2;
#pragma unroll 1
    for (int cl = 0; cl < CPW; cl++) {
      const int clc = warp * CPW + cl, c = cch * TC + clc;
      if (lane_ok) {
        float wt[49];
#pragma unroll
        for (int i = 0; i < 49; i++) wt[i] = wsm[clc * 49 + i];
        const float bias = __ldg(p.dwb + c);
        float acc[4][TW];
        conv_patch<TW>(p.x + ((size_t)b * C + c) * HW, wt, H, W, h0, w0, acc);
        __half* xrow = xh + (size_t)c * Mp;
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int h = h0 + r;
          if (h < H) {
            const int t = b * HW + h * W + w0;
            const int slot = (rgi * 4 + r) * rpt + cgi * TW;
            float o[TW];
#pragma unroll
            for (int cc = 0; cc < TW; cc++) {
              o[cc] = acc[r][cc] + bias;
              sts[(slot + cc) * 2] += o[cc]; sts[(slot + cc) * 2 + 1] += o[cc] * o[cc];
            }
            if (TW == 4) *reinterpret_cast<uint2*>(xrow + t) = make_uint2(pack2(o[0], o[1]), pack2(o[2], o[3]));
            else xrow[t] = __float2half_rn(o[0]);
          }
        }
      }
    }
    __syncthreads();
    const int nslots = rows_per_tile * rpt;
    for (int slot = tid; slot < nslots; slot += NT) {
      const int row = slot / rpt, col = slot - row * rpt;
      const int h = hbi * rows_per_tile + row, w = wbi * rpt + col;
      if (h < H && w < W) {
        float S1 = 0.f, S2 = 0.f;
#pragma unroll
        for (int ws = 0; ws < 8; ws++) { S1 += st_s[(ws * nslots + slot) * 2]; S2 += st_s[(ws * nslots + slot) * 2 + 1]; }
        *reinterpret_cast<float2*>(p.part + ((size_t)cch * Mp + (size_t)b * HW + h * W + w) * 2) = make_float2(S1, S2);
      }
    }
    __syncthreads();
  }
}

// small images (H, W <= 14, W % 7 == 0): zero-padded smem planes per warp; a lane computes 7 outputs of a row.
__device__ void phase1_planes(const Params& p, float* smem, float* aux) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, Mp = p.Mp, nparts = p.nparts, CPW = p.p1_cpw, TC = 8 * CPW;
  float* wsm = aux;
  float* st = aux + 32 * 49;              // [HW][2]
  float* pln = smem;                      // [8][PLANE_FLOATS]
  const int PW = W + 6, PS = (H + 6) * PW;
  const int SEG = W / 7, LPP = H * SEG;
  int PPW = (4 * LPP <= 32) ? 4 : ((2 * LPP <= 32) ? 2 : 1);
  if (PPW > CPW) PPW = CPW;
  float* mypl = pln + warp * PLANE_FLOATS;
  __half* xh = reinterpret_cast<__half*>(p.xh);
  for (int i = tid; i < 8 * PLANE_FLOATS; i += NT) pln[i] = 0.f;  // borders stay zero
  const int pl = lane / LPP, r = lane - pl * LPP;
  const bool act = pl < PPW;
  const int h = act ? r / SEG : 0, w0 = act ? (r - h * SEG) * 7 : 0;

  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    const int b = tile / nparts, cch = tile - b * nparts;
    for (int i = tid; i < 2 * HW; i += NT) st[i] = 0.f;
    for (int i = tid; i < TC * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * TC * 49 + i);
    __syncthreads();
#pragma unroll 1
    for (int pass = 0; pass < CPW / PPW; pass++) {
      const int cl0 = warp * CPW + pass * PPW;
      const float* src = p.x + ((size_t)b * C + cch * TC + cl0) * HW;
      for (int i = lane; i < PPW * HW; i += 32) {
        const int q = i / HW, e = i - q * HW, hh = e / W, ww = e - hh * W;
        mypl[q * PS + (hh + 3) * PW + ww + 3] = __ldg(src + i);
      }
      __syncwarp();
      if (act) {
        const int clc = cl0 + pl, c = cch * TC + clc;
        const float* wrow = wsm + clc * 49;
        float acc[7];
#pragma unroll
        for (int cc = 0; cc < 7; cc++) acc[cc] = 0.f;
        const float* base = mypl + pl * PS + h * PW + w0;
#pragma unroll 1
        for (int i = 0; i < 7; i++) {
          float v[13], wr[7];
#pragma unroll
          for (int jj = 0; jj < 13; jj++) v[jj] = base[i * PW + jj];
#pragma unroll
          for (int j = 0; j < 7; j++) wr[j] = wrow[i * 7 + j];
#pragma unroll
          for (int cc = 0; cc < 7; cc++)
#pragma unroll
            for (int j = 0; j < 7; j++) acc[cc] = fmaf(v[cc + j], wr[j], acc[cc]);
        }
        const float bias = __ldg(p.dwb + c);
        __half* xrow = xh + (size_t)c * Mp + (size_t)b * HW + h * W + w0;
        const int slot0 = h * W + w0;
#pragma unroll
        for (int cc = 0; cc < 7; cc++) {
          const float o = acc[cc] + bias;
          xrow[cc] = __float2half_rn(o);
          atomicAdd(&st[(slot0 + cc) * 2], o);
          atomicAdd(&st[(slot0 + cc) * 2 + 1], o * o);
        }
      }
      __syncwarp();
    }
    __syncthreads();
    for (int slot = tid; slot < HW; slot += NT)
      *reinterpret_cast<float2*>(p.part + ((size_t)cch * Mp + (size_t)b * HW + slot) * 2) = make_float2(st[2 * slot], st[2 * slot + 1]);
    __syncthreads();
  }
}

// generic fallback (32-channel chunks): one token per lane, direct global loads
__device__ void phase1_tokens(const Params& p, float* smem, float* aux) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nparts = p.nparts;
  float* st_s = smem;
  float* wsm = aux;
  __half* xh = reinterpret_cast<__half*>(p.xh);
  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    const int cch = tile % nparts, tok0 = (tile / nparts) * 32;
    for (int i = tid; i < 8 * 32 * 2; i += NT) st_s[i] = 0.f;
    for (int i = tid; i < 32 * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * 32 * 49 + i);
    __syncthreads();
    const int t = tok0 + lane;
    const bool tv = t < M;
    const int tt = tv ? t : 0;
    const int bb = tt / HW, rem = tt - bb * HW, th = rem / W, tw = rem - th * W;
#pragma unroll 1
    for (int cl = 0; cl < 4; cl++) {
      const int clc = warp * 4 + cl, c = cch * 32 + clc;
      float acc = 0.f;
      for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) {
          const int hh = th - 3 + i, ww = tw - 3 + j;
          if (tv && hh >= 0 && hh < H && ww >= 0 && ww < W)
            acc = fmaf(__ldg(p.x + ((size_t)bb * C + c) * HW + hh * W + ww), wsm[clc * 49 + i * 7 + j], acc);
        }
      if (tv) {
        const float o = acc + __ldg(p.dwb + c);
        xh[(size_t)c * Mp + t] = __float2half_rn(o);
        st_s[(warp * 32 + lane) * 2] += o; st_s[(warp * 32 + lane) * 2 + 1] += o * o;
      }
    }
    __syncthreads();
    if (tid < 32 && tok0 + tid < M) {
      float S1 = 0.f, S2 = 0.f;
      for (int ws = 0; ws < 8; ws++) { S1 += st_s[(ws * 32 + tid) * 2]; S2 += st_s[(ws * 32 + tid) * 2 + 1]; }
      *reinterpret_cast<float2*>(p.part + ((size_t)cch * Mp + tok0 + tid) * 2) = make_float2(S1, S2);
    }
    __syncthreads();
  }
}

__device__ void phase1(const Params& p, float* smem, float* aux) {
  const int G = gridDim.x, warp = threadIdx.x >> 5;
  for (int i = blockIdx.x * NT + threadIdx.x; i < p.B * p.C4; i += G * NT) p.sumsq[i] = 0.f;
  // few conv tiles: the other CTAs convert the weights meanwhile
  const bool split_roles = 2 * p.p1_tiles <= G;
  if (split_roles && blockIdx.x >= p.p1_tiles) { weights_pass(p, (blockIdx.x - p.p1_tiles) * 8 + warp, (G - p.p1_tiles) * 8); return; }
  if (!split_roles) weights_pass(p, blockIdx.x * 8 + warp, G * 8);
  if (p.p1_mode == 2) phase1_planes(p, smem, aux);
  else if (p.p1_mode == 1) { if (p.p1_twr == 4) phase1_patch<4>(p, smem, aux); else phase1_patch<1>(p, smem, aux); }
  else phase1_tokens(p, smem, aux);
}

// ---------------------------------------------------------------- GEMM phases (FP16 mma.sync, cp.async rings)
// A = act fp16 [K][Mp] -> smem [k][m] (ldmatrix.trans); B = W fp16 [N][K] -> smem [n][k] (ldmatrix).
// 8 warps = WM x WN spatial warps x KS K-groups (each K-group runs its own S-stage ring).
template <int BM, int BN, int WM, int WN, int KS, int BKc, int S, int PHASE>
__device__ void gemm_phase(const Params& p, float* smem, float* aux, float* misc, int split) {
  constexpr int WS = WM * WN, GT = WS * 32;
  constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
  constexpr int AS_LD = BM + 8, BS_LD = BKc + 8;
  constexpr int A_STAGE = BKc * AS_LD, B_STAGE = BN * BS_LD, STAGE_H = A_STAGE + B_STAGE;  // halves
  constexpr int CS_LD = BM + 4;
  constexpr int AM8 = BM / 8, ACH = BKc * AM8, AQ = ACH / GT, KSTEPA = GT / AM8;
  constexpr int BK8 = BKc / 8, BCH = BN * BK8, BQ = (BCH + GT - 1) / GT;
  constexpr int NSPLIT = NT / BM;
  static_assert(WS * KS == 8, "8 warps");
  static_assert(S >= 3, "ring depth");
  static_assert(KS * S * STAGE_H * 2 <= RING_BYTES, "ring smem");
  static_assert(KS * BN * CS_LD * 4 <= RING_BYTES, "epilogue smem");
  static_assert(GT % AM8 == 0 && ACH % GT == 0, "A mapping");
  static_assert(WTM % 16 == 0 && WTN % 8 == 0 && BKc % 16 == 0 && BM <= 128 && BN <= 128 && NT % BM == 0, "tile");
  float* sq_s = misc; float* inv_s = misc + MAXIMG * 128; float* red_s = inv_s + 8;
  float* mu_s = red_s + 64; float* rs_s = mu_s + 128; float* cs1_s = rs_s + 128; float* cc_s = cs1_s + 128;
  float* tab_s = aux;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kg = warp / WS, ws = warp - kg * WS, wm = ws / WN, wn = ws - wm * WN;
  const int gt = ws * 32 + lane;
  const int M = p.M, Mp = p.Mp, HW = p.HW;
  const int N = (PHASE == 2) ? p.C4 : p.C;
  const int K = (PHASE == 2) ? p.C : p.C4;
  const __half* __restrict__ Wh = reinterpret_cast<const __half*>(PHASE == 2 ? p.w1h : p.w2h);
  const __half* __restrict__ Ag = reinterpret_cast<const __half*>(PHASE == 2 ? p.xh : p.hidden);
  const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
  const int ntiles = mt * nt * split, nk = K / BKc;
  __half* gsm = reinterpret_cast<__half*>(smem) + kg * S * STAGE_H;
  const unsigned g_s = smem_u32(gsm);
  const int lj = lane >> 3, lr = lane & 7;
  const int a_lo = ((lj >> 1) * 8 + lr) * AS_LD + (lj & 1) * 8 + wm * WTM;
  const int b_lo = (wn * WTN + (lj >> 1) * 8 + lr) * BS_LD + (lj & 1) * 8;
  const int b2_lo = (wn * WTN + lr) * BS_LD + (lj & 1) * 8;
  const int a_m8 = (gt % AM8) * 8, a_kq = gt / AM8;

  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    int rem = tile;
    const int nti = rem % nt; rem /= nt;
    const int mti = rem % mt; const int s = rem / mt;
    const int m0 = mti * BM, n0 = nti * BN;
    const int c0 = (int)(((long)nk * s) / split), c1 = (int)(((long)nk * (s + 1)) / split);
    const int b_first = m0 / HW;
    const int nimg = min(MAXIMG, (min(m0 + BM, M) - 1) / HW - b_first + 1);
    const int kbase = c0 * BKc, krange = (c1 - c0) * BKc;
    const bool use_tab = krange <= TABK;

    // per-column epilogue constants: s1 / c1 (phase 2), c2 (phase 3)
    for (int i = tid; i < BN; i += NT) {
      const int col = n0 + i;
      if (PHASE == 2) { cs1_s[i] = col < N ? __ldcg(p.s1 + col) : 0.f; cc_s[i] = col < N ? __ldcg(p.c1 + col) : 0.f; }
      else cc_s[i] = col < N ? __ldcg(p.c2 + col) : 0.f;
    }
    int ia_l = 0, ib_l = 0, bmask = 0; float inva = 0.f, invb = 0.f;
    const float* sqa = p.sumsq; const float* sqb = p.sumsq;
    if constexpr (PHASE == 2) {
      // LayerNorm statistics: NSPLIT threads per token sum the per-chunk partials, reduced through smem
      float* red2 = aux;
      {
        const int tok = tid % BM, js = tid / BM;
        const int t = min(m0 + tok, M - 1);
        float S1 = 0.f, S2 = 0.f;
#pragma unroll 4
        for (int ch = js; ch < p.nparts; ch += NSPLIT) {
          const float2 pp = __ldcg(reinterpret_cast<const float2*>(p.part + ((size_t)ch * Mp + t) * 2));
          S1 += pp.x; S2 += pp.y;
        }
        red2[(js * BM + tok) * 2] = S1; red2[(js * BM + tok) * 2 + 1] = S2;
      }
      __syncthreads();
      if (tid < BM) {
        float S1 = 0.f, S2 = 0.f;
#pragma unroll
        for (int js = 0; js < NSPLIT; js++) { S1 += red2[(js * BM + tid) * 2]; S2 += red2[(js * BM + tid) * 2 + 1]; }
        const float mean = S1 / (float)p.C;
        const float var = fmaxf(S2 / (float)p.C - mean * mean, 0.f);
        mu_s[tid] = mean; rs_s[tid] = rsqrtf(var + p.ln_eps);
      }
    } else {
      float sacc[MAXIMG];
#pragma unroll
      for (int im = 0; im < MAXIMG; im++) sacc[im] = 0.f;
      for (int k4 = tid; k4 < K / 4; k4 += NT) {
#pragma unroll
        for (int im = 0; im < MAXIMG; im++) {
          if (im < nimg) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(p.sumsq + (size_t)(b_first + im) * K) + k4);
            sacc[im] += sqrtf(v.x) + sqrtf(v.y) + sqrtf(v.z) + sqrtf(v.w);
          }
        }
      }
#pragma unroll
      for (int im = 0; im < MAXIMG; im++) {
        const float v = warp_sum(sacc[im]);
        if (lane == 0) red_s[warp * MAXIMG + im] = v;
      }
      __syncthreads();
      if (tid < nimg) {
        float tot = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < 8; w2++) tot += red_s[w2 * MAXIMG + tid];
        inv_s[tid] = 1.f / (tot / (float)K + p.eps);
      }
      __syncthreads();
      if (use_tab) {
        for (int idx = tid; idx < nimg * krange; idx += NT) {
          const int im = idx / krange, kk = idx - im * krange, k = kbase + kk;
          tab_s[im * TABK + kk] = fmaf(__ldg(p.gw + k) * inv_s[im], sqrtf(__ldcg(p.sumsq + (size_t)(b_first + im) * K + k)), 1.f);
        }
      }
      const int t0 = m0 + a_m8;
      const int tf = min(t0, M - 1), tl = min(t0 + 7, M - 1);
      const int ia = tf / HW, ib = tl / HW, bnd = (ia + 1) * HW;
#pragma unroll
      for (int e = 0; e < 8; e++) bmask |= (t0 + e >= bnd) ? (1 << e) : 0;
      ia_l = min(ia - b_first, MAXIMG - 1); ib_l = min(ib - b_first, MAXIMG - 1);
      inva = inv_s[ia_l]; invb = inv_s[ib_l];
      sqa = p.sumsq + (size_t)ia * K; sqb = p.sumsq + (size_t)ib * K;
      __syncthreads();
    }

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
      for (int j = 0; j < NI; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

    auto issue = [&](int jl, int st) {
      const int k0 = (c0 + kg + jl * KS) * BKc;
      const unsigned sa_ = g_s + st * STAGE_H * 2, sb_ = sa_ + A_STAGE * 2;
#pragma unroll
      for (int q = 0; q < AQ; q++) {
        const int id = q * GT + gt, kr = id / AM8, m8 = (id - kr * AM8) * 8;
        cp_async16(sa_ + (kr * AS_LD + m8) * 2, Ag + (size_t)(k0 + kr) * Mp + m0 + m8, 16);
      }
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt;
        if (BQ * GT == BCH || id < BCH) {
          const int n = id / BK8, k8 = (id - n * BK8) * 8;
          const bool ok = (n0 + n) < N;
          const __half* src = ok ? (Wh + (size_t)(n0 + n) * K + k0 + k8) : Wh;
          cp_async16(sb_ + (n * BS_LD + k8) * 2, src, ok ? 16 : 0);
        }
      }
    };
    auto scale = [&](int jl, int st) {
      const int k0 = (c0 + kg + jl * KS) * BKc;
      __half* As = gsm + st * STAGE_H;
#pragma unroll
      for (int q = 0; q < AQ; q++) {
        const int kr = q * KSTEPA + a_kq, k = k0 + kr;
        float sa, sb;
        if (use_tab) { sa = tab_s[ia_l * TABK + k - kbase]; sb = tab_s[ib_l * TABK + k - kbase]; }
        else {
          const float g = __ldg(p.gw + k);
          sa = fmaf(g * inva, sqrtf(__ldcg(sqa + k)), 1.f); sb = fmaf(g * invb, sqrtf(__ldcg(sqb + k)), 1.f);
        }
        uint4* ptr = reinterpret_cast<uint4*>(As + kr * AS_LD + a_m8);
        const uint4 v = *ptr;
        const unsigned w[4] = {v.x, v.y, v.z, v.w};
        unsigned o[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float2 f = unpack2(w[j]);
          const float s0 = ((bmask >> (2 * j)) & 1) ? sb : sa, s1v = ((bmask >> (2 * j + 1)) & 1) ? sb : sa;
          o[j] = pack2(f.x * s0, f.y * s1v);
        }
        *ptr = make_uint4(o[0], o[1], o[2], o[3]);
      }
    };
    auto compute = [&](int st) {
      const unsigned a_s = g_s + st * STAGE_H * 2, b_s = a_s + A_STAGE * 2;
#pragma unroll
      for (int ks = 0; ks < BKc / 16; ks++) {
        unsigned bf[NI][2];
#pragma unroll
        for (int ni = 0; ni + 1 < NI; ni += 2)
          ldsm_x4(b_s + ((ni * 8) * BS_LD + ks * 16 + b_lo) * 2, bf[ni][0], bf[ni][1], bf[ni + 1][0], bf[ni + 1][1]);
        if constexpr ((NI & 1) != 0)
          ldsm_x2(b_s + (((NI - 1) * 8) * BS_LD + ks * 16 + b2_lo) * 2, bf[NI - 1][0], bf[NI - 1][1]);
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
          unsigned af[4];
          ldsm_x4_t(a_s + ((ks * 16) * AS_LD + mi * 16 + a_lo) * 2, af[0], af[1], af[2], af[3]);
#pragma unroll
          for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af, bf[ni][0], bf[ni][1]);
        }
      }
    };
    auto gsync = [&]() {
      if (KS == 1) __syncthreads();
      else if (GT == 32) __syncwarp();
      else asm volatile("bar.sync %0, %1;" ::"r"(kg + 1), "r"(GT) : "memory");
    };

    const int kfirst = c0 + kg;
    const int nloc = (kfirst < c1) ? (c1 - kfirst + KS - 1) / KS : 0;
#pragma unroll
    for (int j = 0; j < S - 1; j++) { if (j < nloc) issue(j, j); cp_async_commit(); }
    if constexpr (PHASE == 3) {
      cp_async_wait<S - 2>();
      gsync();
      if (nloc > 0) scale(0, 0);
    }
    for (int j = 0; j < nloc; j++) {
      if constexpr (PHASE == 3) cp_async_wait<S - 3>(); else cp_async_wait<S - 2>();
      gsync();
      { const int jn = j + S - 1; if (jn < nloc) issue(jn, jn % S); cp_async_commit(); }
      if constexpr (PHASE == 3) { if (j + 1 < nloc) scale(j + 1, (j + 1) % S); }
      compute(j % S);
    }
    cp_async_wait<0>();

    // ---- epilogue: partial tiles -> smem Cs [KS][BN][CS_LD] (fp32)
    __syncthreads();
    float* Cs = smem;
    {
      const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
      for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++) {
          const int m = wm * WTM + mi * 16 + gid, n = wn * WTN + ni * 8 + 2 * tig;
          float* cp = Cs + (kg * BN + n) * CS_LD + m;
          cp[0] = acc[mi][ni][0]; cp[CS_LD] = acc[mi][ni][1];
          cp[8] = acc[mi][ni][2]; cp[CS_LD + 8] = acc[mi][ni][3];
        }
    }
    __syncthreads();
    if constexpr (PHASE == 2) {
      constexpr int G8 = BM / 8;
      __half* hid = reinterpret_cast<__half*>(p.hidden);
      for (int idx = tid; idx < BN * G8; idx += NT) {
        const int m8 = idx % G8, n_l = idx / G8, col = n0 + n_l;
        const int t0 = m0 + m8 * 8;
        float v[8];
        {
          const float* src = Cs + n_l * CS_LD + m8 * 8;
          float4 x0 = *reinterpret_cast<const float4*>(src), x1 = *reinterpret_cast<const float4*>(src + 4);
#pragma unroll
          for (int g = 1; g < KS; g++) {
            const float4 y0 = *reinterpret_cast<const float4*>(src + g * BN * CS_LD);
            const float4 y1 = *reinterpret_cast<const float4*>(src + g * BN * CS_LD + 4);
            x0.x += y0.x; x0.y += y0.y; x0.z += y0.z; x0.w += y0.w; x1.x += y1.x; x1.y += y1.y; x1.z += y1.z; x1.w += y1.w;
          }
          v[0] = x0.x; v[1] = x0.y; v[2] = x0.z; v[3] = x0.w; v[4] = x1.x; v[5] = x1.y; v[6] = x1.z; v[7] = x1.w;
        }
        const float s1n = cs1_s[n_l], c1n = cc_s[n_l];
        const float4 mu0 = *reinterpret_cast<const float4*>(mu_s + m8 * 8), mu1 = *reinterpret_cast<const float4*>(mu_s + m8 * 8 + 4);
        const float4 rs0 = *reinterpret_cast<const float4*>(rs_s + m8 * 8), rs1 = *reinterpret_cast<const float4*>(rs_s + m8 * 8 + 4);
        const float mu[8] = {mu0.x, mu0.y, mu0.z, mu0.w, mu1.x, mu1.y, mu1.z, mu1.w};
        const float rs[8] = {rs0.x, rs0.y, rs0.z, rs0.w, rs1.x, rs1.y, rs1.z, rs1.w};
#pragma unroll
        for (int e = 0; e < 8; e++) v[e] = gelu_f(fmaf(rs[e], v[e] - mu[e] * s1n, c1n));
        *reinterpret_cast<uint4*>(hid + (size_t)col * Mp + t0) =
            make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
        const int tf = min(t0, M - 1);
        const int ia = tf / HW, bnd = (ia + 1) * HW;
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int t = t0 + e;
          const float g2 = v[e] * v[e];
          if (t < M) { if (t < bnd) sa += g2; else sb += g2; }
        }
        const bool uni = (t0 + 7 < M) && (t0 + 7 < bnd);
        const int ia0 = __shfl_sync(0xffffffffu, ia, 0);
        if (__all_sync(0xffffffffu, uni && ia == ia0)) {
#pragma unroll
          for (int o = 1; o < G8 && o < 32; o <<= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
          if ((lane % G8) == 0) atomicAdd(&sq_s[(ia - b_first) * BN + n_l], sa);
        } else if (t0 < M) {
          atomicAdd(&sq_s[(ia - b_first) * BN + n_l], sa);
          if (sb != 0.f) atomicAdd(&sq_s[min(ia + 1 - b_first, MAXIMG - 1) * BN + n_l], sb);
        }
      }
      __syncthreads();
      for (int idx = tid; idx < nimg * BN; idx += NT) {
        const int im = idx / BN, n = idx - im * BN;
        const float v = sq_s[idx]; sq_s[idx] = 0.f;
        if (n0 + n < N) atomicAdd(p.sumsq + (size_t)(b_first + im) * N + n0 + n, v);
      }
      __syncthreads();
    } else {
      constexpr int NSTEP = NT / BM, NPT = BN / NSTEP;
      const int m_l = tid % BM, n_l0 = tid / BM;
      const int t = m0 + m_l;
      const bool tv = t < M;
      const int img = tv ? t / HW : b_first;
      const int hw = t - img * HW;
      const size_t obase = (size_t)img * N * HW + hw;
      if (split == 1) {
        float xv[NPT];
#pragma unroll
        for (int i = 0; i < NPT; i++) {  // issue every residual load of the tile before the first use
          const int col = n0 + n_l0 + NSTEP * i;
          xv[i] = (col < N && tv) ? __ldg(p.x + obase + (size_t)col * HW) : 0.f;
        }
#pragma unroll
        for (int i = 0; i < NPT; i++) {
          const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
          if (col < N && tv) {
            float v = 0.f;
#pragma unroll
            for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
            p.out[obase + (size_t)col * HW] = v + cc_s[n_l] + xv[i];
          }
        }
      } else {
#pragma unroll 4
        for (int i = 0; i < NPT; i++) {
          const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
          if (col < N && tv) {
            float v = 0.f;
#pragma unroll
            for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
            atomicAdd(p.out + obase + (size_t)col * HW, v);
          }
        }
      }
      __syncthreads();
    }
  }
}

template <int PHASE>
__device__ __forceinline__ void run_gemm(const Params& p, int cfg, int split, float* smem, float* aux, float* misc) {
  switch (cfg) {
    case 0: gemm_phase<128, 128, 2, 4, 1, 32, 4, PHASE>(p, smem, aux, misc, split); break;
    case 1: gemm_phase<128, 96, 2, 4, 1, 32, 4, PHASE>(p, smem, aux, misc, split); break;
    case 2: gemm_phase<64, 64, 2, 2, 2, 32, 4, PHASE>(p, smem, aux, misc, split); break;
    default: gemm_phase<32, 32, 1, 1, 8, 16, 3, PHASE>(p, smem, aux, misc, split); break;
  }
}

__device__ __forceinline__ void stub(const Params& p) { if (threadIdx.x == 0) p.part[blockIdx.x] = 0.f; }

// PH = 0: all phases in one cooperative launch with grid syncs; PH = 1..3: one phase per plain launch (probes).
// mask bits 1/2/4 select which phases do real work (the others write one word): probe-only, run() passes 7.
template <int PH>
__global__ void __launch_bounds__(NT, 2) convnext_k(Params p) {
  extern __shared__ float4 smem4[];
  float* smem = reinterpret_cast<float*>(smem4);
  float* aux = smem + RING_BYTES / 4;
  float* misc = aux + AUX_FLOATS;
  for (int i = threadIdx.x; i < MAXIMG * 128; i += NT) misc[i] = 0.f;
  __syncthreads();
  if (PH == 0 || PH == 1) { if (p.mask & 1) phase1(p, smem, aux); else stub(p); }
  if constexpr (PH == 0) cg::this_grid().sync();
  if (PH == 0 || PH == 2) {
    if ((p.mask & 4) && p.split3 > 1) {  // residual + c2 init for the split-K atomics of phase 3
      const long total = (long)p.M * p.C;
      for (long i = (long)blockIdx.x * NT + threadIdx.x; i < total; i += (long)gridDim.x * NT) {
        const int c = (int)((i / p.HW) % p.C);
        p.out[i] = __ldg(p.x + i) + __ldcg(p.c2 + c);
      }
    }
    if (p.mask & 2) run_gemm<2>(p, p.cfg2, 1, smem, aux, misc); else stub(p);
  }
  if constexpr (PH == 0) cg::this_grid().sync();
  if (PH == 0 || PH == 3) { if (p.mask & 4) run_gemm<3>(p, p.cfg3, p.split3, smem, aux, misc); else stub(p); }
}

// ---------------------------------------------------------------- host
struct Pick { int cfg; int split; };
static Pick pick_cfg(long M, long N, long K, int grid, int min_split, int max_split) {
  struct CfgD { int id, BM, BN, BK; double eff; };
  const CfgD cfgs[4] = {{0, 128, 128, 32, 0.80}, {1, 128, 96, 32, 0.78}, {2, 64, 64, 32, 0.60}, {3, 32, 32, 16, 0.40}};
  double best = 1e300; Pick r{0, min_split};
  for (int ci = 0; ci < 4; ci++) {
    const CfgD& c = cfgs[ci];
    if (c.id == 1 && (N % 96) != 0) continue;
    const long tiles = ((M + c.BM - 1) / c.BM) * ((N + c.BN - 1) / c.BN);
    const long nk = K / c.BK;
    for (int S = min_split; S <= max_split; S *= 2) {
      if (S > 1 && nk / S < 2) break;
      const long total = tiles * S;
      const long waves = (total + grid - 1) / grid;
      double cost = (double)c.BM * c.BN * ((double)K / S) / c.eff * (double)waves;
      if (S > 1) cost *= 1.08 + 0.03 * S;
      if (cost < best) { best = cost; r.cfg = c.id; r.split = S; }
    }
  }
  return r;
}

// fills the phase-1 geometry for the given channels-per-warp (tile = 8*cpw channels); returns the tile count
static int p1_geometry(Params& p, int cpw) {
  p.p1_cpw = cpw; p.nparts = p.C / (8 * cpw);
  p.p1_lpr = 0; p.p1_rg = 0; p.p1_wb = 0; p.p1_hb = 0;
  const int SEG = p.W / 7, LPP = p.H * SEG;
  const int PPW = (4 * LPP <= 32) ? 4 : ((2 * LPP <= 32) ? 2 : 1);
  const bool planes = (p.W % 7 == 0) && p.H <= 14 && p.W <= 14 && LPP <= 32 && LPP > 0 && 2 * p.HW <= 32 * 49 &&
                      PPW * (p.H + 6) * (p.W + 6) <= PLANE_FLOATS;
  if (planes) { p.p1_mode = 2; p.p1_twr = 7; p.p1_tiles = p.B * p.nparts; return p.p1_tiles; }
  const int tw = (p.W % 4 == 0) ? 4 : 1;
  const int cgt = p.W / tw;
  int lpr, rg, wb;
  if (cgt >= 32) { lpr = 32; rg = 1; wb = (cgt + 31) / 32; }
  else { lpr = cgt; rg = 32 / cgt; wb = 1; }
  if (rg > 8) rg = 8;
  const int rows = 4 * rg;
  if (rows <= p.H) {
    p.p1_mode = 1; p.p1_twr = tw; p.p1_lpr = lpr; p.p1_rg = rg; p.p1_wb = wb;
    p.p1_hb = (p.H + rows - 1) / rows;
    p.p1_tiles = p.B * p.nparts * p.p1_hb * p.p1_wb;
    return p.p1_tiles;
  }
  p.p1_cpw = 4; p.nparts = p.C / 32;
  p.p1_mode = 0; p.p1_twr = 1; p.p1_tiles = ((p.M + 31) / 32) * p.nparts;
  return p.p1_tiles;
}

extern "C" const char* convnext_launch(Params p, void* stream_v, int mode) {
  static int grid_size = 0;
  cudaError_t e;
  cudaStream_t stream = (cudaStream_t)stream_v;
  if (grid_size == 0) {
    e = cudaFuncSetAttribute(convnext_k<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    cudaFuncSetAttribute(convnext_k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(convnext_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(convnext_k<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int bps = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, convnext_k<0>, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    if (bps < 1) bps = 1;
    if (bps > 2) bps = 2;
    grid_size = sms * bps;
  }
  int smin = 1;
  if (p.force_split != 1) while ((long)p.C4 / smin > TABK) smin *= 2;
  const Pick k2 = pick_cfg(p.M, p.C4, p.C, grid_size, 1, 1);
  const Pick k3 = pick_cfg(p.M, p.C, p.C4, grid_size, smin, p.force_split == 1 ? 1 : 16);
  p.cfg2 = k2.cfg; p.cfg3 = k3.cfg; p.split3 = k3.split;
  // phase 1: 32-channel tiles (4 channels per warp); patch tiles drop to 8 channels when that leaves SMs idle
  if (p1_geometry(p, 4) < grid_size && p.p1_mode == 1) p1_geometry(p, 1);
  if (mode == 0) {
    void* args[] = {&p};
    e = cudaLaunchCooperativeKernel((const void*)convnext_k<0>, dim3(grid_size), dim3(NT), args, (size_t)SMEM_BYTES, stream);
  } else {
    convnext_k<1><<<grid_size, NT, SMEM_BYTES, stream>>>(p);
    convnext_k<2><<<grid_size, NT, SMEM_BYTES, stream>>>(p);
    convnext_k<3><<<grid_size, NT, SMEM_BYTES, stream>>>(p);
    e = cudaGetLastError();
  }
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <string>

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* xh; float* part; void* hidden; float* sumsq; void* w1h; void* w2h; float* s1; float* c1; float* c2;
  int B, C, H, W, HW, M, Mp, C4, nparts;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb, p1_cpw;
  int cfg2, cfg3, split3;
  int mask, force_split;
};
extern "C" const char* convnext_launch(Params p, void* stream, int mode);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  TORCH_CHECK(C % 32 == 0, "dim must be a multiple of 32");
  TORCH_CHECK(H * W >= 32, "H*W must be >= 32");
  const int HW = H * W, M = B * HW, C4 = 4 * C;
  const int Mp = ((M + 127) / 128) * 128;
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");

  // workspace (no state kept across calls)
  const size_t off_xh = 0;
  const size_t off_part = align256(off_xh + (size_t)C * Mp * 2);
  const size_t off_hidden = align256(off_part + (size_t)(C / 8) * Mp * 2 * 4);
  const size_t off_sumsq = align256(off_hidden + (size_t)C4 * Mp * 2);
  const size_t off_w1h = align256(off_sumsq + (size_t)B * C4 * 4);
  const size_t off_w2h = align256(off_w1h + (size_t)C * C4 * 2);
  const size_t off_s1 = align256(off_w2h + (size_t)C4 * C * 2);
  const size_t off_c1 = align256(off_s1 + (size_t)C4 * 4);
  const size_t off_c2 = align256(off_c1 + (size_t)C4 * 4);
  const size_t total = align256(off_c2 + (size_t)C * 4);
  torch::Tensor ws = torch::empty({(long)total}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();

  Params p;
  p.x = x.data_ptr<float>(); p.dww = dwconv_weight.data_ptr<float>(); p.dwb = dwconv_bias.data_ptr<float>();
  p.lnw = layernorm_weight.data_ptr<float>(); p.lnb = layernorm_bias.data_ptr<float>();
  p.w1 = pwconv1_weight.data_ptr<float>(); p.b1 = pwconv1_bias.data_ptr<float>();
  p.gw = grn_weight.data_ptr<float>(); p.gb = grn_bias.data_ptr<float>();
  p.w2 = pwconv2_weight.data_ptr<float>(); p.b2 = pwconv2_bias.data_ptr<float>();
  p.out = output.data_ptr<float>();
  p.xh = (void*)(wp + off_xh); p.part = (float*)(wp + off_part); p.hidden = (void*)(wp + off_hidden);
  p.sumsq = (float*)(wp + off_sumsq); p.w1h = (void*)(wp + off_w1h); p.w2h = (void*)(wp + off_w2h);
  p.s1 = (float*)(wp + off_s1); p.c1 = (float*)(wp + off_c1); p.c2 = (float*)(wp + off_c2);
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = HW; p.M = M; p.Mp = Mp; p.C4 = C4; p.nparts = C / 32;
  p.eps = (float)eps; p.ln_eps = (float)layer_norm_eps;
  p.p1_mode = 0; p.p1_twr = 1; p.p1_tiles = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_hb = 0; p.p1_wb = 0; p.p1_cpw = 4;
  p.cfg2 = 0; p.cfg3 = 0; p.split3 = 1;
  p.mask = 7; p.force_split = 0;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = convnext_launch(p, (void*)stream, 0);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (fused, FP16 mma.sync GEMMs)"); }
```

```yaml design-card
id: r1-convnext-coop-hmma-ring
parents: [r0-convnext-coop-fp32-anchor]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: cpasync
  st: direct
  grid: persistent
  launch: fused
  tile: rows128
  red: cta
  cache: default
  spec: all
hypothesis: >-
  Lineage B, E2 (H12). Legacy mma.sync.m16n8k16 FP16 with fp32 accumulate runs at 2048 FLOP/clk/SM, which is
  0.455 PF/s at 1.5 GHz, so the GEMMs stop being the bottleneck except at the top two sizes. The kernel stays one
  cooperative launch: phase 1 does the weight pass (fp16, native [N][K] layout; LN and gb folded into row sums) and
  the dwconv, writing raw fp16 d and LN partials; GEMM1 uses a 4-stage cp.async ring with the LN algebra in the
  epilogue, then GELU and fp16 hidden plus GRN sums; GEMM2 uses a 4-stage ring with the GRN scale as an smem pass,
  then the residual. Rented B200: 16/16 pass, -59/-65/-74 % vs the anchor in S/M/L; projected portal score 0.561.
expected_effect:
  S: {pct: -59, confidence: medium}
  M: {pct: -65, confidence: medium}
  L: {pct: -74, confidence: medium}
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 106272
  threads_per_cta: 256
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 151552
  launches_per_call: 1
knobs:
  NT: 256
  GRID: 296
  GEMM_CFGS: "128x128/128x96/64x64(KS2)/32x32(KS8,BK16)"
  STAGES: 4
  BK: 32
  TABK: 1024
  P1_CPW: "4, or 1 for patch tiles when 32-ch tiles < grid"
tests: [H12, H11, H4, H5, H17]
findings:
  - "H12 supported: mma.sync m16n8k16 f16->f32 = 2048 FLOP/clk/SM with 1 CTA/SM (4-16 warps, 8 or 16 independent chains) = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16; wall 520-546 TF/s at boost. Per-CTA clock64 with 2 CTAs/SM gives bogus 2700-4090 (CTAs not co-resident) [probe_b200]"
  - "H11 fixed floor F: this kernel with all three phases stubbed (296 CTAs, 104 KB smem, 2 grid.sync) = 4.2-4.7 us on every shape; three plain launches stubbed = 9.7-10.1 us [probe_b200 harness_time]"
  - "H4: full kernel, coop vs three plain launches, 3 interleaved repeats: 1,192,28,28 29.7 vs 36.3 us; 2,768,7,7 46.7 vs 50.2; 64,96,112,112 1793 vs 1749 (coop 2.5% slower at the largest). Regs: 128 for all four kernel variants (PH0 has 64 B spill) [probe_b200]"
  - "H5: split-K auto vs forced 1: 1,192,28,28 29.9 vs 30.6 us (no effect); 2,768,7,7 46.5 vs 64.6; 8,768,7,7 86.1 vs 120.6 (split worth 28-29% at C=768) [probe_b200]"
  - "run_tests (rented B200, us): 1,192 30.9; 1,96 28.3; 1,160,449,7 55.2; 2,96 42.2; 4,192 47.2; 8,384 85.8; 4,96 62.5; 8,192 66.1; 16,384 113.0; 2,768 47.9; 8,768 87.1; 16,768 107.9; 8,320 175.3; 32,768 179.5; 32,128 587.2; 64,96 1786.6; projected portal 0.561 +- 0.008 [run_tests]"
  - "per-phase (cumulative masks, includes F ~4.6): 1,192,28,28 P1 13.2 / +P2 20.1 / all 29.7; 1,160,449,7 17.1/30.6/56.4; 2,768,7,7 19.7/33.2/47.1; 32,768,7,7 48.2/106.0/180.6; 32,128,64,64 107/309/586; 64,96,112,112 446/1290/1793 [probe_b200]"
  - "precision: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <=1e-7 fraction beyond atol 2.4e-3 with randomized lnw/lnb (stricter than harness); raw fp16 d with LN folded into the GEMM1 epilogue (r*(acc - mu*s1) + c1) is as accurate as an fp16 LN'd operand [probe_b200]"
  - "phase-1 parallelism: 8-channel tiles (1 channel per warp) when 32-channel tiles < grid cut P1 at 1,192,28,28 from ~25 us to ~9 us; the 4x4 register-patch conv is latency-bound (10 serial row loads per channel), so tile count matters more than FMA rate at S [probe_b200]"
  - "dead end: applying 8-ch tiles to the 7x7 smem-plane path made the GEMM1 prologue sum 96 partials per token serially (P1+P2 at 8,768,7,7 61 -> 93 us); planes keep 32-ch tiles and the prologue now splits partials over NT/BM threads per token [probe_b200]"
  - "dead end: cp.async-staged smem bands for the dwconv (per-warp double buffer, 16B interior + 4B halo cp.async, float4 LDS) were slower than direct __ldg 4x4 patches: P1 41 vs 29 us at 1,192,28,28 and 515 vs 385 us at 64,96,112,112 (cause not isolated) [probe_b200]"
  - "GEMM main loop: register-prefetched 1-deep A -> 4-stage pure cp.async rings (LN algebra in epilogue, GRN as in-smem scale pass) cut P2 at 8,384,14,14 from ~31 to ~21 us but barely moved the largest (950 -> 888 us), where each tile is a serial prologue/3-chunk/GELU-epilogue chain [probe_b200]"
  - "GEMM2 epilogue: issuing all residual x loads of a tile before use (NPT registers) cut P3 at 64,96,112,112 from 896 to 504 us [probe_b200]"
  - "weight pass at C=768 (fp16 conversion + row sums, 19 MB) costs ~10 us of P1 (P1 with only the weight pass = 14.8-15.5 us vs F 4.6) [probe_b200]"
  - "success criteria: S rows <= 45 us missed at 1,160,449,7 (55) and 4,192,28,28 (47); 2,768,7,7 47.9 <= 60 met; 32,768,7,7 179.5 > 150 missed; largest 1787 <= 2500 met; F 4.6 <= 15 and HMMA 0.455 >= 0.15 PF/s, so neither refutation clause fires [run_tests, probe_b200]"
  - "remaining gaps: largest P2 ~840 us vs ~150 us roofline (short-K per-tile chain + erf GELU epilogue), P1 ~440 us vs ~105 us FMA bound; 1,160,449,7 P3 ~26 us (N=160 padded to 2x128 tiles); C=768 rows P3 60-75 us [probe_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: cpasync, wstat: false, launches: 1, prefetch: false, x: none, w: none, st: none, cluster: 1}
runs_on:
  H200: {runs: true, representative: false, note: "mma.sync/cp.async/ldmatrix/cooperative launch all exist on sm_90; 104 KB smem x2 fits; HMMA rate and clocks differ"}
  A100: {runs: true, representative: false, note: "sm_80 has mma.sync m16n8k16 and cp.async; 2 x 105 KB smem exceeds 164 KB so 1 CTA/SM (grid 108); correctness only"}
  L40S: {runs: true, representative: false, note: "100 KB smem per SM < 104 KB dynamic smem: launch would fail; not runnable as is"}
risks:
  - "cooperative launch needs 2 CTAs/SM at 104 KB dynamic smem and 128 regs; the grid comes from the occupancy API at first call"
  - "split-K phase 3 uses fp32 atomics into an x+c2-initialised output (order-nondeterministic, within tolerance)"
  - "hidden fp16 is rounded twice (GELU output, then after the GRN scale): err rms 4e-4 vs atol 2.4e-3, about 6 sigma"
  - "workspace up to ~0.9 GB at the largest size (xh fp16, LN partials, hidden fp16) via torch.empty each call"
  - "portal/rented ratio unknown for the mixed HMMA/SIMT/DRAM profile (anchor was 1.30-1.32 clock-bound)"
measure_first: [H200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```