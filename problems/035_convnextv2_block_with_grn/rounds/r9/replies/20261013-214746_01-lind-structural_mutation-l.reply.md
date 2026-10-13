### Rationale
E2 tested H33: GEMM2 run as cta_group::2 pairs, with g produced in shared memory by producer warps that run ahead of the MMA. ptxas will not mix cta_group::1 and ::2 in one kernel, so the C=768 instantiation became all-pair: TMEM alloc, GEMM1 and GEMM2 all use cta_group::2. All 16 rows pass on every build. The first version was slower because producers paid an L2 round per stage. Staging gw·gf in shared memory fixed part of that, and stamps then showed the producers' fence plus cluster-arrive chain gating the MMA. Giving each producer warp a whole stage fixed that and gave 32,768 −3.0 / −2.0 µs (−4.6 % / −3.1 %, two runs). At 2/8,768 the pair skeleton loses 1.4–2.1 µs, so pairs are dispatched only for C=768 with B ≥ 16, and every other row runs r8's exact code. The success bars are missed and H33 is mostly refuted as stated; the merge should take 32,768.

```json solution-spec
{"name": "r9-d-e2-pair-smemg", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN.
// Wide-C small-image rows (2/8/16/32,768,7,7 and 8/16,384,14,14): cooperative kernel tc_kernel with a tcgen05
//   kind::f16 swap-AB engine (A = fp16 weight tile, 128 rows, K-major; B = fp16 activations [channel][token],
//   MN-major; TMA 128B-swizzled ring; whole-warp TMA producer and MMA warp; double-buffered TMEM accumulators).
//   r8: 512 threads; the GEMM1 epilogue runs on all 16 warps (3-4 per TMEM quadrant) with the A&S erfc GELU.
//   r9 (PAIR instantiation, 16/32,768): launched as 2-CTA clusters and every tcgen05 op is cta_group::2 (ptxas
//   forbids mixing): GEMM1 pairs two 128-row hidden tiles over the same tokens (each CTA lands half of the d tile);
//   GEMM2 pairs M = 256 output channels (each CTA lands its own 128 W2 rows and half of the token tile: 32 KB stages,
//   6-deep ring); g = gw*gf*h is no longer stored by GEMM1 but produced in smem from the landed h half by 4 producer
//   warps, one warp per stage. 2/8,768 and 8/16,384 keep the r8 1SM path (fewer stages per CTA: the pair's fixed
//   costs do not pay there).
// Small-C rows (1/4/8,192,28,28; 1/2/4,96,56,56; 1,160,449,7): cooperative kernel sc_kernel, same engine with
//   N-chunked token tiles of one image, red.add GRN sums, and the GRN scale applied to the GEMM2 B tile in smem.
// Other rows: the r0 fp32 SIMT cooperative anchor (convnext_fused).
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <string.h>
namespace cg = cooperative_groups;

// =====================================================================================================
// r0 anchor path (unchanged)
// =====================================================================================================
#define NT 256
#define MAXIMG 5
#define GEMM_SMEM_FLOATS (128 * 132)
#define SMEM_BYTES (GEMM_SMEM_FLOATS * 4 + MAXIMG * 128 * 4 + 64)
#define P1_SLOTS 512

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};

__device__ __forceinline__ float gelu_f(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ void transpose_mat(const float* __restrict__ src, int R, int Cc, float* __restrict__ dst, float* tile) {
  const int rt = (R + 31) / 32, ct = (Cc + 31) / 32;
  const int tid = threadIdx.x, j = tid & 31, i0 = tid >> 5;
  for (int t = blockIdx.x; t < rt * ct; t += gridDim.x) {
    const int tr = t / ct, tc = t % ct;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, r = tr * 32 + i, c = tc * 32 + j;
      tile[i * 33 + j] = (r < R && c < Cc) ? src[(size_t)r * Cc + c] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, c = tc * 32 + i, r = tr * 32 + j;
      if (r < R && c < Cc) dst[(size_t)c * R + r] = tile[j * 33 + i];
    }
    __syncthreads();
  }
}

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

template <int MODE, int TWR>
__device__ void phase1_conv(const Params& p, float* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nch = p.nch;
  float* st_s = smem;
  float* wsm = smem + 8 * P1_SLOTS * 2;
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg;
  const int rpt = lpr * TWR;

  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    int b = 0, cch = 0, hbi = 0, wbi = 0, tok0 = 0;
    if (MODE == 1) {
      const int per = nch * hb * wb;
      b = tile / per; int rem = tile - b * per;
      cch = rem / (hb * wb); rem -= cch * (hb * wb);
      hbi = rem / wb; wbi = rem - hbi * wb;
    } else {
      cch = tile % nch; tok0 = (tile / nch) * (32 * TWR);
    }
    for (int i = tid; i < 8 * P1_SLOTS * 2; i += NT) st_s[i] = 0.f;
    for (int i = tid; i < 32 * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * 32 * 49 + i);
    __syncthreads();

    int rgi = 0, cgi = 0, h0 = 0, w0 = 0; bool lane_ok = true;
    if (MODE == 1) {
      rgi = lane / lpr; cgi = lane - rgi * lpr;
      lane_ok = rgi < rg;
      h0 = hbi * rows_per_tile + rgi * 4;
      w0 = (wbi * 32 + cgi) * TWR;
      lane_ok = lane_ok && (w0 < W);
    }
    int tk[MODE == 0 ? TWR : 1], tb[MODE == 0 ? TWR : 1], th[MODE == 0 ? TWR : 1], tw_[MODE == 0 ? TWR : 1];
    bool tv[MODE == 0 ? TWR : 1];
    if (MODE == 0) {
#pragma unroll
      for (int r = 0; r < TWR; r++) {
        const int t = tok0 + r * 32 + lane;
        tv[r] = t < M;
        const int tt = tv[r] ? t : 0;
        const int bb = tt / HW; const int rem = tt - bb * HW; const int hh = rem / W;
        tk[r] = t; tb[r] = bb; th[r] = hh; tw_[r] = rem - hh * W;
      }
    }

#pragma unroll 1
    for (int cl = 0; cl < 4; cl++) {
      const int clc = warp * 4 + cl;
      const int c = cch * 32 + clc;
      float wt[49];
#pragma unroll
      for (int i = 0; i < 49; i++) wt[i] = wsm[clc * 49 + i];
      const float bias = __ldg(p.dwb + c);
      float* xrow = p.xn + (size_t)c * Mp;
      float* sts = st_s + warp * P1_SLOTS * 2;
      if (MODE == 1) {
        if (lane_ok) {
          float acc[4][TWR];
          conv_patch<TWR>(p.x + ((size_t)b * C + c) * HW, wt, H, W, h0, w0, acc);
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int h = h0 + r;
            if (h < H) {
              const int t = b * HW + h * W + w0;
              const int slot = (rgi * 4 + r) * rpt + cgi * TWR;
              if (TWR == 4) {
                float4 o = make_float4(acc[r][0] + bias, acc[r][1] + bias, acc[r][2] + bias, acc[r][3] + bias);
                *reinterpret_cast<float4*>(xrow + t) = o;
                sts[(slot + 0) * 2] += o.x; sts[(slot + 0) * 2 + 1] += o.x * o.x;
                sts[(slot + 1) * 2] += o.y; sts[(slot + 1) * 2 + 1] += o.y * o.y;
                sts[(slot + 2) * 2] += o.z; sts[(slot + 2) * 2 + 1] += o.z * o.z;
                sts[(slot + 3) * 2] += o.w; sts[(slot + 3) * 2 + 1] += o.w * o.w;
              } else {
#pragma unroll
                for (int cc = 0; cc < TWR; cc++) {
                  const float o = acc[r][cc] + bias;
                  xrow[t + cc] = o;
                  sts[(slot + cc) * 2] += o; sts[(slot + cc) * 2 + 1] += o * o;
                }
              }
            }
          }
        }
      } else {
        float acc[TWR];
#pragma unroll
        for (int r = 0; r < TWR; r++) acc[r] = 0.f;
#pragma unroll
        for (int i = 0; i < 7; i++)
#pragma unroll
          for (int j = 0; j < 7; j++) {
            const float wv = wt[i * 7 + j];
#pragma unroll
            for (int r = 0; r < TWR; r++) {
              const int hh = th[r] - 3 + i, ww = tw_[r] - 3 + j;
              if (tv[r] && hh >= 0 && hh < H && ww >= 0 && ww < W)
                acc[r] = fmaf(__ldg(p.x + ((size_t)tb[r] * C + c) * HW + hh * W + ww), wv, acc[r]);
            }
          }
#pragma unroll
        for (int r = 0; r < TWR; r++) {
          if (tv[r]) {
            const float o = acc[r] + bias;
            xrow[tk[r]] = o;
            const int slot = r * 32 + lane;
            sts[slot * 2] += o; sts[slot * 2 + 1] += o * o;
          }
        }
      }
    }
    __syncthreads();
    const int nslots = (MODE == 1) ? rows_per_tile * rpt : 32 * TWR;
    for (int slot = tid; slot < nslots; slot += NT) {
      int t; bool valid;
      if (MODE == 1) {
        const int row = slot / rpt, col = slot - row * rpt;
        const int h = hbi * rows_per_tile + row, w = wbi * 32 * TWR + col;
        valid = (h < H) && (w < W);
        t = b * HW + h * W + w;
      } else {
        t = tok0 + slot; valid = t < M;
      }
      if (valid) {
        float S1 = 0.f, S2 = 0.f;
#pragma unroll
        for (int ws = 0; ws < 8; ws++) { S1 += st_s[(ws * P1_SLOTS + slot) * 2]; S2 += st_s[(ws * P1_SLOTS + slot) * 2 + 1]; }
        float* pp = p.part + ((size_t)cch * Mp + t) * 2;
        pp[0] = S1; pp[1] = S2;
      }
    }
    __syncthreads();
  }
}

__device__ void phase1(const Params& p, float* smem) {
  const int tid = threadIdx.x;
  for (int i = blockIdx.x * NT + tid; i < p.B * p.C4; i += gridDim.x * NT) p.sumsq[i] = 0.f;
  transpose_mat(p.w1, p.C4, p.C, p.w1t, smem);
  transpose_mat(p.w2, p.C, p.C4, p.w2t, smem);
  if (p.p1_mode == 1) {
    if (p.p1_twr == 4) phase1_conv<1, 4>(p, smem); else phase1_conv<1, 1>(p, smem);
  } else {
    if (p.p1_twr == 4) phase1_conv<0, 4>(p, smem);
    else if (p.p1_twr == 2) phase1_conv<0, 2>(p, smem);
    else phase1_conv<0, 1>(p, smem);
  }
}

template <int BM, int BN, int WM, int WN, int KS, int BKc, int PHASE>
__device__ void gemm_phase(const Params& p, float* smem, float* sq_s, float* inv_s, float* red_s, int split) {
  constexpr int WS = WM * WN;
  constexpr int GT = WS * 32;
  constexpr int TM = BM / (WM * 8), TN = BN / (WN * 4);
  constexpr int AM4 = BM / 4, BN4 = BN / 4;
  constexpr int AQ = (BKc * AM4) / GT, BQ = (BKc * BN4) / GT;
  constexpr int STAGE = BKc * (BM + BN);
  constexpr int CS_LD = BM + 4;
  constexpr int NSTEP = NT / BM, NPT = BN / NSTEP;
  static_assert(WS * KS == 8, "8 warps");
  static_assert((BKc * AM4) % GT == 0 && (BKc * BN4) % GT == 0, "quad mapping");
  static_assert(GT % AM4 == 0, "token quad constant per thread");
  static_assert(KS * 2 * STAGE <= GEMM_SMEM_FLOATS && KS * BN * CS_LD <= GEMM_SMEM_FLOATS, "smem");
  static_assert(TM == 8 || TM == 4, "TM");
  static_assert(TN == 8 || TN == 6, "TN");

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kg = warp / WS, ws = warp - kg * WS, wm = ws / WN, wn = ws - wm * WN;
  const int lm = lane >> 2, ln = lane & 3;
  const int gt = ws * 32 + lane;
  const int M = p.M, Mp = p.Mp, HW = p.HW;
  const int N = (PHASE == 2) ? p.C4 : p.C;
  const int K = (PHASE == 2) ? p.C : p.C4;
  const float* __restrict__ Ag = (PHASE == 2) ? p.xn : p.hidden;
  const float* __restrict__ Bg = (PHASE == 2) ? p.w1t : p.w2t;
  const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
  const int ntiles = mt * nt * split, nk = K / BKc;
  float* As = smem + kg * 2 * STAGE;
  float* Bs = As + 2 * BKc * BM;
  const int row_m = wm * (BM / WM) + lm * TM;
  const int col_n = wn * (BN / WN) + ln * TN;
  const int a_m4 = (gt % AM4) * 4;
  const int a_kq = gt / AM4;
  constexpr int KSTEP_A = GT / AM4;

  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    int rem = tile;
    const int ni = rem % nt; rem /= nt;
    const int mi = rem % mt; const int s = rem / mt;
    const int m0 = mi * BM, n0 = ni * BN;
    const int c0 = (int)(((long)nk * s) / split), c1 = (int)(((long)nk * (s + 1)) / split);
    const int b_first = m0 / HW;
    const int nimg = min(MAXIMG, (min(m0 + BM, M) - 1) / HW - b_first + 1);

    float tc0[4], tc1[4]; int timg[4]; bool tval[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int t = m0 + a_m4 + e;
      tval[e] = t < M;
      const int tt = tval[e] ? t : 0;
      if (PHASE == 2) {
        float S1 = 0.f, S2 = 0.f;
        for (int ch = 0; ch < p.nch; ch++) {
          const float2 pp = __ldg(reinterpret_cast<const float2*>(p.part + ((size_t)ch * Mp + tt) * 2));
          S1 += pp.x; S2 += pp.y;
        }
        const float mean = S1 / (float)p.C;
        float var = S2 / (float)p.C - mean * mean; var = fmaxf(var, 0.f);
        tc0[e] = mean; tc1[e] = rsqrtf(var + p.ln_eps); timg[e] = 0;
      } else {
        timg[e] = tt / HW - b_first; tc0[e] = 0.f; tc1[e] = 0.f;
      }
    }
    if (PHASE == 3) {
      for (int img = 0; img < nimg; img++) {
        const float* sq = p.sumsq + (size_t)(b_first + img) * K;
        float sacc = 0.f;
        for (int k = tid; k < K; k += NT) sacc += sqrtf(sq[k]);
        sacc = warp_sum(sacc);
        if (lane == 0) red_s[warp] = sacc;
        __syncthreads();
        if (tid == 0) {
          float tot = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < 8; w2++) tot += red_s[w2];
          inv_s[img] = 1.f / (tot / (float)K + p.eps);
        }
        __syncthreads();
      }
#pragma unroll
      for (int e = 0; e < 4; e++) tc1[e] = inv_s[timg[e] < MAXIMG ? timg[e] : 0];
    }

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    float4 ra[AQ], rb[BQ];
    auto load_chunk = [&](int kc) {
      const int k0 = kc * BKc;
#pragma unroll
      for (int q = 0; q < AQ; q++) {
        const int ka = k0 + q * KSTEP_A + a_kq;
        float4 v = __ldg(reinterpret_cast<const float4*>(Ag + (size_t)ka * Mp + m0 + a_m4));
        float vv[4] = {v.x, v.y, v.z, v.w};
        if (PHASE == 2) {
          const float lw = __ldg(p.lnw + ka), lb = __ldg(p.lnb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) vv[e] = tval[e] ? fmaf((vv[e] - tc0[e]) * tc1[e], lw, lb) : 0.f;
        } else {
          const float gwk = __ldg(p.gw + ka), gbk = __ldg(p.gb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float sq = __ldg(p.sumsq + (size_t)(b_first + timg[e]) * K + ka);
            vv[e] = tval[e] ? fmaf(vv[e], fmaf(gwk * tc1[e], sqrtf(sq), 1.f), gbk) : 0.f;
          }
        }
        ra[q] = make_float4(vv[0], vv[1], vv[2], vv[3]);
      }
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        const int col = n0 + n4;
        rb[q] = (col < N) ? __ldg(reinterpret_cast<const float4*>(Bg + (size_t)(k0 + kb) * N + col))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    };
    auto store_regs = [&](int buf) {
#pragma unroll
      for (int q = 0; q < AQ; q++)
        *reinterpret_cast<float4*>(As + (buf * BKc + q * KSTEP_A + a_kq) * BM + a_m4) = ra[q];
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        *reinterpret_cast<float4*>(Bs + (buf * BKc + kb) * BN + n4) = rb[q];
      }
    };
    auto gsync = [&]() {
      if (KS == 1) __syncthreads();
      else if (GT == 32) __syncwarp();
      else asm volatile("bar.sync %0, %1;" ::"r"(kg + 1), "r"(GT));
    };

    int kc = c0 + kg;
    if (kc < c1) { load_chunk(kc); store_regs(0); }
    gsync();
    for (int it = 0; kc < c1; it++, kc += KS) {
      const int buf = it & 1;
      const bool more = (kc + KS) < c1;
      if (more) load_chunk(kc + KS);
      const float* Ab = As + buf * BKc * BM + row_m;
      const float* Bb = Bs + buf * BKc * BN + col_n;
#pragma unroll 4
      for (int k = 0; k < BKc; k++) {
        float a[TM], b[TN];
        {
          const float4 a0 = *reinterpret_cast<const float4*>(Ab + k * BM);
          a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
          if (TM == 8) {
            const float4 a1 = *reinterpret_cast<const float4*>(Ab + k * BM + 4);
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
          }
        }
        if (TN == 8) {
          const float4 b0 = *reinterpret_cast<const float4*>(Bb + k * BN);
          const float4 b1 = *reinterpret_cast<const float4*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
        } else {
          const float2 b0 = *reinterpret_cast<const float2*>(Bb + k * BN);
          const float2 b1 = *reinterpret_cast<const float2*>(Bb + k * BN + 2);
          const float2 b2 = *reinterpret_cast<const float2*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b1.x; b[3] = b1.y; b[4] = b2.x; b[5] = b2.y;
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
      if (more) store_regs(buf ^ 1);
      gsync();
    }

    __syncthreads();
    float* Cs = smem;
#pragma unroll
    for (int j = 0; j < TN; j++) {
      float* dst = Cs + (kg * BN + col_n + j) * CS_LD + row_m;
      *reinterpret_cast<float4*>(dst) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
      if (TM == 8) *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
    }
    __syncthreads();
    const int m_l = tid % BM, n_l0 = tid / BM;
    const int t = m0 + m_l;
    const bool tv = t < M;
    const int img = tv ? t / HW : b_first;
    const int hw = t - img * HW;
    if (PHASE == 2) {
      const int img0 = __shfl_sync(0xffffffffu, img, 0);
      const bool uniform = __all_sync(0xffffffffu, (img == img0) && tv);
      const int il0 = img0 - b_first;
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        float v = 0.f;
#pragma unroll
        for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
        const float gl = gelu_f(v + __ldg(p.b1 + col));
        if (tv) p.hidden[(size_t)col * Mp + t] = gl;
        float g2 = tv ? gl * gl : 0.f;
        if (uniform) {
          g2 = warp_sum(g2);
          if (lane == 0) {
            if (il0 < MAXIMG) atomicAdd(&sq_s[il0 * BN + n_l], g2);
            else atomicAdd(p.sumsq + (size_t)img0 * N + col, g2);
          }
        } else if (tv) {
          const int il = img - b_first;
          if (il < MAXIMG) atomicAdd(&sq_s[il * BN + n_l], g2);
          else atomicAdd(p.sumsq + (size_t)img * N + col, g2);
        }
      }
      __syncthreads();
      for (int idx = tid; idx < nimg * BN; idx += NT) {
        const int im = idx / BN, n = idx - im * BN;
        const float v = sq_s[idx]; sq_s[idx] = 0.f;
        atomicAdd(p.sumsq + (size_t)(b_first + im) * N + n0 + n, v);
      }
      __syncthreads();
    } else {
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        if (col < N && tv) {
          float v = 0.f;
#pragma unroll
          for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
          const size_t idx = ((size_t)img * N + col) * HW + hw;
          if (split == 1) p.out[idx] = v + __ldg(p.b2 + col) + __ldg(p.x + idx);
          else atomicAdd(p.out + idx, v);
        }
      }
      __syncthreads();
    }
  }
}

template <int PHASE>
__device__ __forceinline__ void run_gemm(const Params& p, int cfg, int split, float* smem, float* sq_s, float* inv_s, float* red_s) {
  switch (cfg) {
    case 0: gemm_phase<128, 128, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 1: gemm_phase<128, 96, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 2: gemm_phase<64, 64, 2, 2, 2, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    default: gemm_phase<32, 32, 1, 1, 8, 16, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
  }
}

__global__ void __launch_bounds__(NT, 2) convnext_fused(Params p) {
  extern __shared__ float4 smem4[];
  float* smem = reinterpret_cast<float*>(smem4);
  float* sq_s = smem + GEMM_SMEM_FLOATS;
  float* inv_s = sq_s + MAXIMG * 128;
  float* red_s = inv_s + 8;
  for (int i = threadIdx.x; i < MAXIMG * 128; i += NT) sq_s[i] = 0.f;
  __syncthreads();
  cg::grid_group grid = cg::this_grid();

  phase1(p, smem);
  grid.sync();
  if (p.split3 > 1) {
    const long total = (long)p.M * p.C;
    for (long i = (long)blockIdx.x * NT + threadIdx.x; i < total; i += (long)gridDim.x * NT) {
      const int c = (int)((i / p.HW) % p.C);
      p.out[i] = p.x[i] + p.b2[c];
    }
  }
  run_gemm<2>(p, p.cfg2, 1, smem, sq_s, inv_s, red_s);
  grid.sync();
  run_gemm<3>(p, p.cfg3, p.split3, smem, sq_s, inv_s, red_s);
}

struct Pick { int cfg; int split; };
static Pick pick_cfg(long M, long N, long K, int grid, bool allow_split) {
  struct CfgD { int id, BM, BN, BK; double eff; };
  const CfgD cfgs[4] = {{0, 128, 128, 32, 0.80}, {1, 128, 96, 32, 0.78}, {2, 64, 64, 32, 0.60}, {3, 32, 32, 16, 0.40}};
  double best = 1e300; Pick r{0, 1};
  for (int ci = 0; ci < 4; ci++) {
    const CfgD& c = cfgs[ci];
    if (c.id == 1 && (N % 96) != 0) continue;
    const long tiles = ((M + c.BM - 1) / c.BM) * ((N + c.BN - 1) / c.BN);
    const long nk = K / c.BK;
    for (int S = 1; S <= (allow_split ? 16 : 1); S *= 2) {
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

extern "C" const char* convnext_launch(Params p, void* stream_v) {
  static int grid_size = 0;
  cudaError_t e;
  if (grid_size == 0) {
    e = cudaFuncSetAttribute(convnext_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int bps = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, convnext_fused, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    if (bps < 1) bps = 1;
    grid_size = sms * bps;
  }
  const Pick k2 = pick_cfg(p.M, p.C4, p.C, grid_size, false);
  const Pick k3 = pick_cfg(p.M, p.C, p.C4, grid_size, true);
  p.cfg2 = k2.cfg; p.cfg3 = k3.cfg; p.split3 = k3.split;
  {
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
      p.p1_tiles = p.B * p.nch * p.p1_hb * p.p1_wb;
    } else {
      p.p1_mode = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_wb = 0; p.p1_hb = 0;
      int r0 = 4;
      while (r0 > 1 && ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch < grid_size) r0 >>= 1;
      p.p1_twr = r0;
      p.p1_tiles = ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch;
    }
  }
  void* args[] = {&p};
  e = cudaLaunchCooperativeKernel((const void*)convnext_fused, dim3(grid_size), dim3(NT), args, (size_t)SMEM_BYTES,
                                  (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

// =====================================================================================================
// tcgen05 swap-AB cooperative path for the wide-C small-image rows (512 threads, 16 warps)
// Phase A: plane loads, weight conversion, depthwise conv, LN partials, fp16 d, out = x + b2. Sync 1.
// Phase B: GEMM1 (warp 0 TMA, warp 1 MMA); epilogue on all 16 warps (warps 4-15 on every tile, 3 per TMEM quadrant;
//   warps 0-3 join the CTA's last tile, 4 per quadrant): 16-column chunks, LN fold, A&S erfc GELU, fp16 h,
//   per-(image,k) sumsq combined through smem. 1SM: g = fp16(gw*gf*h) stored too. PAIR: GEMM1 tiles are
//   cta_group::2 pairs (256 hidden rows: each CTA its own 128 W1 rows, half of the d tile, its own 128 TMEM lanes)
//   and only cgf[image][k] = gw*gf is stored. Sync 2.
// Phase C, 1SM: GEMM2 split-K with two accumulators (h and g), 48 KB stages.
// Phase C, PAIR (C=768): pairs with M = 256 output channels (each CTA its 128 W2 rows) x 128 tokens (each CTA lands
//   64 of them, own full barrier); warps 12-15 build the g half tile in smem from the landed h half (g = fp16(cgf*h),
//   the same rounding as the stored g), one warp per stage, and arrive on the leader's ready barrier; the leader's MMA
//   warp issues cta_group::2 MMAs into both CTAs' TMEM; warps 4-11 of each CTA drain their own 128 lanes.
// TC_STAMPS: globaltimer diagnostics (off in the shipped build; the timed path writes no stamp buffer).
// =====================================================================================================
#ifndef TC_STAMPS
#define TC_STAMPS 0
#endif
#define TC_NT 512
#define TC_RING (192 * 1024)
#define TC_STGB 2560
#define TC_OFF_STG TC_RING
#define TC_OFF_MS (TC_OFF_STG + 8 * TC_STGB)
#define TC_OFF_SQ (TC_OFF_MS + 4096)
#define TC_OFF_INV (TC_OFF_SQ + 4096)
#define TC_OFF_CB (TC_OFF_INV + 512)
#define TC_OFF_BAR (TC_OFF_CB + 512)
#define TC_NBAR 36
#define TC_SMEM (TC_OFF_BAR + TC_NBAR * 8 + 1024)
#define TC_STG2 49152
#define TC_STG2P 32768
#define TC_NST2P 6
// phase-B buffers of the all-warp epilogue inside the ring (GEMM1 stages use <= 160 KB): per-warp staging, sumsq parts
#define TC_B_STG (160 * 1024)
#define TC_B_STGW 1536
#define TC_B_SQ (TC_B_STG + 16 * TC_B_STGW)
struct TCP {
  CUtensorMap mW1, mW2, mD, mH, mG;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid, *gq;
  float *part, *uk, *vk, *wgb, *gsum, *cgf;
  unsigned long long* st;
  int B, Mcols, units, NI, T1, STG1, NST1, KS, KB2, U2;
  float eps, ln_eps;
};
__device__ __forceinline__ unsigned long long gtime() { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); return g; }
#if TC_STAMPS
#define STAMP(i) do { p.st[cta * 24 + (i)] = gtime(); } while (0)
#else
#define STAMP(i) do {} while (0)
#endif
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, uint32_t c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_arrive1(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::"r"(sa(b)), "r"(ph) : "memory"); }
// cluster-scope variants for the 2-CTA pair: acquire.cluster wait, release.cluster arrive on a (possibly remote) barrier
__device__ __forceinline__ void mb_wait_cl(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n WAITC_%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n @!p bra WAITC_%=;\n}" ::"r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ uint32_t mapa_s(uint32_t a, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r; }
__device__ __forceinline__ void mb_arrive_cl(uint32_t addr) { asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(addr) : "memory"); }
__device__ __forceinline__ void cluster_sync_all() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int x, int y, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(sa(dst)), "l"((uint64_t)m), "r"(x), "r"(y), "r"(sa(b)) : "memory"); }
// pair TMA: lands in this CTA's smem, completes bytes on the leader CTA's barrier (shared::cluster address)
__device__ __forceinline__ void tma2d_2sm(void* dst, const CUtensorMap* m, int x, int y, uint32_t bar_cl) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(sa(dst)), "l"((uint64_t)m), "r"(x), "r"(y), "r"(bar_cl) : "memory"); }
__device__ __forceinline__ bool elect1() { uint32_t r; asm volatile("{\n.reg .pred p;\n.reg .b32 t;\nelect.sync t|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(r)); return r != 0; }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFFu) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc)); }
__device__ __forceinline__ void umma2(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* b) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void umma_commit2(uint64_t* b) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(sa(b)), "h"((uint16_t)3) : "memory"); }
__device__ __forceinline__ void tfb() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tfa() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tld32(uint32_t a, uint32_t (&v)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]),
        "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
        "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]) : "r"(a));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tld16(uint32_t a, uint32_t (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(a));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void ebar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t scale_h2(uint32_t u, float s) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
  return h2u(__floats2half2_rn(f.x * s, f.y * s));
}
// exact-erf GELU v*Phi(v), Phi from Abramowitz-Stegun 7.1.26 (|erf error| <= 1.5e-7): one MUFU.RCP + one MUFU.EX2
__device__ __forceinline__ float gelu_as(float v) {
  const float a = fabsf(v) * 0.70710678118654752f;
  const float t = __fdividef(1.f, fmaf(0.3275911f, a, 1.f));
  float y = fmaf(1.061405429f, t, -1.453152027f);
  y = fmaf(y, t, 1.421413741f);
  y = fmaf(y, t, -0.284496736f);
  y = fmaf(y, t, 0.254829592f);
  const float hc = 0.5f * t * y * __expf(-a * a);
  return (v >= 0.f) ? fmaf(-v, hc, v) : v * hc;
}
// 16 fp16 columns of 32 rows (one per lane) -> per-warp smem transpose -> 2 lanes per 32-B row segment
__device__ __forceinline__ void tc_stage16(uint32_t* stg, const uint32_t (&w)[8], __half* base, int pitch, int ch, int vcols, int lane) {
  uint4* srow = reinterpret_cast<uint4*>(stg + lane * 12);
  srow[0] = make_uint4(w[0], w[1], w[2], w[3]);
  srow[1] = make_uint4(w[4], w[5], w[6], w[7]);
  __syncwarp();
#pragma unroll
  for (int i2 = 0; i2 < 2; i2++) {
    const int row = i2 * 16 + (lane >> 1), colb = ch * 16 + (lane & 1) * 8;
    const uint4 val = *reinterpret_cast<const uint4*>(stg + row * 12 + (lane & 1) * 4);
    if (colb < vcols) *reinterpret_cast<uint4*>(base + (size_t)row * pitch + colb) = val;
  }
  __syncwarp();
}

template <int C_, int HW_, bool PAIR>
__global__ void __launch_bounds__(TC_NT, 1) tc_kernel(const __grid_constant__ TCP p) {
  static_assert(!PAIR || C_ == 768, "pair path: C = 768 (3 pair M-tiles)");
  constexpr int W_ = (HW_ == 49) ? 7 : 14, H_ = W_;
  constexpr int C4 = 4 * C_, G = C_ / 32, HT = C4 / 128, CT = C_ / 128, KB1 = C_ / 64;
  constexpr int HWS = (HW_ + 7) / 8 * 8;
  constexpr int HTU = PAIR ? HT / 2 : HT;
  extern __shared__ __align__(1024) unsigned char tc_raw[];
  unsigned char* sm = tc_raw + ((1024u - (sa(tc_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  uint32_t* stg_all = reinterpret_cast<uint32_t*>(sm + TC_OFF_STG);
  float* inv_s = reinterpret_cast<float*>(sm + TC_OFF_INV);
  int* cb_s = reinterpret_cast<int*>(sm + TC_OFF_CB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + TC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *ready2 = bars + 18, *empty2 = bars + 24, *tf2 = bars + 30, *te2 = bars + 32, *cbar = bars + 34;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 35);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  uint32_t rank = 0;
  if constexpr (PAIR) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int cl = cta >> 1, NCL = P >> 1;
  cg::grid_group grid = cg::this_grid();
  if (tid == 0) {
    for (int i = 0; i < 35; i++) {
      const uint32_t cnt = (i >= 18 && i < 24) ? (PAIR ? 2u : 8u) : (i == 32 || i == 33) ? (PAIR ? 16u : 8u)
                         : (i == 10 || i == 11) ? (PAIR ? 2u : 1u) : 1u;
      mb_init(bars + i, cnt);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    if constexpr (PAIR) {
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    } else {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
  }
  tfb(); __syncthreads(); tfa();
  if constexpr (PAIR) cluster_sync_all();
  const uint32_t tbase = *tslot;
  if (tid == 0) STAMP(0);

  // ---------------- phase A: plane loads, weight conversion, depthwise conv, LN partials, fp16 d, out = x + b2
  const int nu = (cta < p.units) ? (p.units - cta + P - 1) / P : 0;
  float* xs = reinterpret_cast<float*>(ring);
  float* wsm = xs + nu * 32 * HW_;
  float* dbuf = wsm + nu * 32 * 49;
  if (tid == 0 && nu > 0) {
    mb_expect(cbar, nu * (32 * HW_ * 4 + 32 * 49 * 4));
    for (int i = 0; i < nu; i++) {
      const int u = cta + i * P, b = u / G, g = u - b * G;
      bulk_g2s(xs + i * 32 * HW_, p.x + ((size_t)b * C_ + g * 32) * HW_, 32 * HW_ * 4, cbar);
      bulk_g2s(wsm + i * 32 * 49, p.dww + (size_t)g * 32 * 49, 32 * 49 * 4, cbar);
    }
  }
  if (cta == 0 && tid < p.B) p.gsum[tid] = 0.f;
  {
    // weight units of C floats each: 4C rows of W1 (fp16(W1*lnw), u = row sum, v = W1.lnb + b1) and 4C quarter rows of
    // W2 (fp16(W2), partial W2.gb per quarter). Highest CTAs first: the low CTAs carry the extra conv units.
    constexpr int NQ = C_ / 128;
    const int nw = P * (TC_NT / 32);
    for (int wu = (P - 1 - cta) * (TC_NT / 32) + warp; wu < 8 * C_; wu += nw) {
      const bool is1 = wu < C4;
      const int row = is1 ? wu : ((wu - C4) >> 2), qtr = is1 ? 0 : ((wu - C4) & 3);
      const float4* src = reinterpret_cast<const float4*>(is1 ? p.w1 + (size_t)row * C_ : p.w2 + (size_t)row * C4 + qtr * C_);
      uint2* dst = reinterpret_cast<uint2*>(is1 ? p.w1h + (size_t)row * C_ : p.w2h + (size_t)row * C4 + qtr * C_);
      float4 v[NQ];
#pragma unroll
      for (int j = 0; j < NQ; j++) v[j] = __ldg(src + lane + 32 * j);
      if (is1) {
        float su = 0.f, sv = 0.f;
#pragma unroll
        for (int j = 0; j < NQ; j++) {
          const int c4 = lane + 32 * j;
          const float4 lw = __ldg(reinterpret_cast<const float4*>(p.lnw) + c4);
          const float4 lb = __ldg(reinterpret_cast<const float4*>(p.lnb) + c4);
          const __half2 h0 = __floats2half2_rn(v[j].x * lw.x, v[j].y * lw.y);
          const __half2 h1 = __floats2half2_rn(v[j].z * lw.z, v[j].w * lw.w);
          const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
          su += (f0.x + f0.y) + (f1.x + f1.y);
          sv += v[j].x * lb.x + v[j].y * lb.y + v[j].z * lb.z + v[j].w * lb.w;
          uint2 o; o.x = h2u(h0); o.y = h2u(h1);
          dst[c4] = o;
        }
        su = warp_sum(su); sv = warp_sum(sv);
        if (lane == 0) { p.uk[row] = su; p.vk[row] = sv + __ldg(p.b1 + row); }
      } else {
        const float4* gb4 = reinterpret_cast<const float4*>(p.gb + qtr * C_);
        float sg = 0.f;
#pragma unroll
        for (int j = 0; j < NQ; j++) {
          const int c4 = lane + 32 * j;
          const float4 g4 = __ldg(gb4 + c4);
          sg += v[j].x * g4.x + v[j].y * g4.y + v[j].z * g4.z + v[j].w * g4.w;
          uint2 o; o.x = h2u(__floats2half2_rn(v[j].x, v[j].y)); o.y = h2u(__floats2half2_rn(v[j].z, v[j].w));
          dst[c4] = o;
        }
        sg = warp_sum(sg);
        if (lane == 0) p.wgb[row * 4 + qtr] = sg;
      }
    }
  }
  if (nu > 0) {
    mb_wait(cbar, 0);
    constexpr int SEG = W_ / 7, per_plane = H_ * SEG;
    const int items = nu * 32 * per_plane;
    for (int itm = tid; itm < items; itm += TC_NT) {
      const int pl = itm / per_plane;
      const int rem = itm - pl * per_plane;
      const int h = rem / SEG, w0 = (rem - h * SEG) * 7;
      const int ui = pl >> 5, pp = pl & 31;
      const int u = cta + ui * P, g = u % G;
      const float* xp = xs + pl * HW_;
      const float* wp = wsm + pl * 49;
      float acc[7];
#pragma unroll
      for (int o = 0; o < 7; o++) acc[o] = 0.f;
#pragma unroll
      for (int i = 0; i < 7; i++) {
        const int hh = h + i - 3;
        if (hh >= 0 && hh < H_) {
          float v[13];
#pragma unroll
          for (int jj = 0; jj < 13; jj++) { const int col = w0 - 3 + jj; v[jj] = (col >= 0 && col < W_) ? xp[hh * W_ + col] : 0.f; }
          float wt[7];
#pragma unroll
          for (int jj = 0; jj < 7; jj++) wt[jj] = wp[i * 7 + jj];
#pragma unroll
          for (int o = 0; o < 7; o++)
#pragma unroll
            for (int jj = 0; jj < 7; jj++) acc[o] = fmaf(v[o + jj], wt[jj], acc[o]);
        }
      }
      const float bias = __ldg(p.dwb + g * 32 + pp);
      float* dp = dbuf + pl * HW_ + h * W_ + w0;
#pragma unroll
      for (int o = 0; o < 7; o++) dp[o] = acc[o] + bias;
    }
    __syncthreads();
    for (int e = tid; e < nu * HW_; e += TC_NT) {
      const int ui = e / HW_, hw = e - ui * HW_;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll 8
      for (int pp = 0; pp < 32; pp++) { const float v = dbuf[(ui * 32 + pp) * HW_ + hw]; s1 += v; s2 = fmaf(v, v, s2); }
      reinterpret_cast<float2*>(p.part)[((size_t)b * G + g) * HW_ + hw] = make_float2(s1, s2);
    }
    constexpr int q8n = HWS / 8;
    for (int e = tid; e < nu * 32 * q8n; e += TC_NT) {
      const int pl = e / q8n, q8 = e - pl * q8n;
      const int ui = pl >> 5, pp = pl & 31;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float v[8];
#pragma unroll
      for (int t = 0; t < 8; t++) { const int hw = q8 * 8 + t; v[t] = hw < HW_ ? dbuf[pl * HW_ + hw] : 0.f; }
      uint4 o;
      o.x = h2u(__floats2half2_rn(v[0], v[1])); o.y = h2u(__floats2half2_rn(v[2], v[3]));
      o.z = h2u(__floats2half2_rn(v[4], v[5])); o.w = h2u(__floats2half2_rn(v[6], v[7]));
      *reinterpret_cast<uint4*>(p.d16 + (size_t)(g * 32 + pp) * p.Mcols + b * HWS + q8 * 8) = o;
    }
    constexpr int n4u = 8 * HW_;
    for (int e = tid; e < nu * n4u; e += TC_NT) {
      const int ui = e / n4u, r = e - ui * n4u;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float4 v = reinterpret_cast<const float4*>(xs)[e];
      const int i0 = r * 4;
      const float* b2g = p.b2 + g * 32;
      v.x += __ldg(b2g + (i0 + 0) / HW_); v.y += __ldg(b2g + (i0 + 1) / HW_);
      v.z += __ldg(b2g + (i0 + 2) / HW_); v.w += __ldg(b2g + (i0 + 3) / HW_);
      reinterpret_cast<float4*>(p.out + ((size_t)b * C_ + g * 32) * HW_)[r] = v;
    }
  }
#if TC_STAMPS
  __syncthreads(); if (tid == 0) STAMP(1);
#endif
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); grid.sync(); tfa();
  if (tid == 0) STAMP(2);

  // ---------------- phase B: GEMM1 (hidden = W1h x d16), LN fold, GELU, fp16 hidden h (and g = gw*gf*h on 1SM)
  // PAIR: tile t = (pair hidden tile, image group) per cluster; this CTA's hidden rows ht = 2*(t%HTU) + rank
  const int tstart = PAIR ? cl : cta, tstep = PAIR ? NCL : P;
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const uint32_t nst = (uint32_t)p.NST1;
    const uint32_t fl = PAIR ? mapa_s(sa(full1), 0) : 0u;
    uint32_t it = 0;
    for (int t = tstart; t < p.T1; t += tstep) {
      const int ht = PAIR ? 2 * (t % HTU) + (int)rank : t % HTU, tt = t / HTU, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15;
      const int NH = PAIR ? N / 2 : N, nat = (NH + 63) >> 6, tokb = tok0 + (PAIR ? (int)rank * NH : 0);
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = (int)(it % nst); const uint32_t n = it / nst;
        if (it >= nst) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          if constexpr (PAIR) {
            if (rank == 0) mb_expect(full1 + s, 2u * (16384u + (uint32_t)nat * 8192u));
            tma2d_2sm(st, &p.mW1, kb * 64, m0, fl + (uint32_t)s * 8u);
            for (int a = 0; a < nat; a++) tma2d_2sm(st + 16384 + a * 8192, &p.mD, tokb + a * 64, kb * 64, fl + (uint32_t)s * 8u);
          } else {
            mb_expect(full1 + s, 16384 + nat * 8192);
            tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
            for (int a = 0; a < nat; a++) tma2d(st + 16384 + a * 8192, &p.mD, tokb + a * 64, kb * 64, full1 + s);
          }
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    if (!PAIR || rank == 0) {
      const uint32_t nst = (uint32_t)p.NST1;
      uint32_t it = 0, ti = 0;
      for (int t = tstart; t < p.T1; t += tstep, ti++) {
        const int tt = t / HTU, img0 = tt * p.NI;
        const int nimg = min(p.NI, p.B - img0), N = (nimg * HWS + 15) & ~15;
        const uint32_t buf = ti & 1;
        if (ti >= 2) {
          if constexpr (PAIR) mb_wait_cl(te1 + buf, ((ti >> 1) & 1) ^ 1); else mb_wait(te1 + buf, ((ti >> 1) & 1) ^ 1);
          tfa();
        }
        const uint32_t d = tbase + buf * 256;
        const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(N >> 3) << 17) | ((PAIR ? 16u : 8u) << 24);
        for (int kb = 0; kb < KB1; kb++, it++) {
          const int s = (int)(it % nst); const uint32_t n = it / nst;
          if constexpr (PAIR) mb_wait_cl(full1 + s, n & 1); else mb_wait(full1 + s, n & 1);
          tfa();
          if (elect1()) {
            const uint32_t a0 = sa(ring + s * p.STG1), b0 = a0 + 16384;
            if constexpr (PAIR) {
#pragma unroll
              for (int k = 0; k < 4; k++) umma2(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kb | k));
              umma_commit2(empty1 + s);
              if (kb == KB1 - 1) umma_commit2(tf1 + buf);
            } else {
#pragma unroll
              for (int k = 0; k < 4; k++) umma(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kb | k));
              umma_commit(empty1 + s);
              if (kb == KB1 - 1) umma_commit(tf1 + buf);
            }
          }
          __syncwarp();
        }
      }
    }
  }
  {
    // GEMM1 epilogue on all 16 warps. Warps 4-15 take every tile (3 per TMEM quadrant); warps 0-3 join the CTA's last
    // tile (warps 0/1 after issuing), which then runs 4 warps per quadrant. Warp j of quadrant q takes the 16-column
    // chunks j, j+nwq, ...: LN fold, A&S erfc GELU, fp16 h, per-image sumsq per 8-column group; partials combined
    // through smem; then (1SM) g = fp16(gw*gf*h) from the same registers, or (PAIR) cgf = gw*gf per (image,k).
    const int q = warp & 3, r = q * 32 + lane;
    const int ntc = (tstart < p.T1) ? (p.T1 - tstart + tstep - 1) / tstep : 0;
    uint32_t* stg = reinterpret_cast<uint32_t*>(ring + TC_B_STG) + warp * (TC_B_STGW / 4);
    float* sqb = reinterpret_cast<float*>(ring + TC_B_SQ);
#if TC_STAMPS
    unsigned long long a_pre = 0, a_wait = 0, a_post = 0;
#endif
    int ti = 0;
    for (int t = tstart; t < p.T1; t += tstep, ti++) {
      const bool last = (ti == ntc - 1);
      if (!last && warp < 4) continue;
#if TC_STAMPS
      const unsigned long long q0 = gtime();
#endif
      const int nwq = last ? 4 : 3, j = last ? (warp >> 2) : (warp >> 2) - 1;
      const int bid = last ? 1 : 2, bn = nwq * 128;
      const int ht = PAIR ? 2 * (t % HTU) + (int)rank : t % HTU, tt = t / HTU, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15, vcols = nimg * HWS;
      const int k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k), gwk = __ldg(p.gw + k);
      float4* ms = reinterpret_cast<float4*>(sm + ((ti & 1) ? TC_OFF_SQ : TC_OFF_MS));
      // per-token LN statistics (mu, rstd, -, valid): warps 2-3 for the second of two tiles (idle during the first)
      const bool early = last && ntc == 2;
      const int pw0 = early ? 2 : 4, pnw = early ? 2 : 8;
      if (warp >= pw0 && warp < pw0 + pnw) {
        for (int et = (warp - pw0) * 32 + lane; et < 256; et += pnw * 32) {
          const int i = et / HWS, hw = et - i * HWS;
          float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
          if (et < vcols && hw < HW_) {
            const float2* pp = reinterpret_cast<const float2*>(p.part) + (size_t)(img0 + i) * G * HW_ + hw;
            float2 v[G];
#pragma unroll
            for (int g = 0; g < G; g++) v[g] = __ldcg(pp + g * HW_);
            float s1 = 0.f, s2 = 0.f;
#pragma unroll
            for (int g = 0; g < G; g++) { s1 += v[g].x; s2 += v[g].y; }
            const float mu = s1 / (float)C_;
            const float var = fmaxf(s2 / (float)C_ - mu * mu, 0.f);
            m = make_float4(mu, rsqrtf(var + p.ln_eps), 0.f, 1.f);
          }
          ms[et] = m;
        }
      }
      nbar(bid, bn);
#if TC_STAMPS
      const unsigned long long q1 = gtime();
#endif
      const uint32_t buf = ti & 1;
      mb_wait(tf1 + buf, (ti >> 1) & 1);
      tfa();
#if TC_STAMPS
      const unsigned long long q2 = gtime();
#endif
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16) + buf * 256;
      const int nch = (N + 15) >> 4;
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      __half* gbase = p.gq + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      uint32_t hreg[5][8];
      float sq0 = 0.f, sq1 = 0.f, sq2 = 0.f;
#pragma unroll
      for (int c = 0; c < 5; c++) {
        const int ch = j + nwq * c;
        if (ch < nch) {
          uint32_t v[16];
          tld16(taddr + ch * 16, v);
          float s8a = 0.f, s8b = 0.f;
#pragma unroll
          for (int e = 0; e < 16; e += 2) {
            const float4 ma = ms[ch * 16 + e], mb = ms[ch * 16 + e + 1];
            const float ha = gelu_as(fmaf(ma.y, fmaf(-ma.x, uk, __uint_as_float(v[e])), vk)) * ma.w;
            const float hb = gelu_as(fmaf(mb.y, fmaf(-mb.x, uk, __uint_as_float(v[e + 1])), vk)) * mb.w;
            const __half2 hh = __floats2half2_rn(ha, hb);
            hreg[c][e >> 1] = h2u(hh);
            const float2 hr = __half22float2(hh);
            if (e < 8) s8a = fmaf(hr.x, hr.x, fmaf(hr.y, hr.y, s8a));
            else s8b = fmaf(hr.x, hr.x, fmaf(hr.y, hr.y, s8b));
          }
          const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
          sq0 += (ia == 0 ? s8a : 0.f) + (ib == 0 ? s8b : 0.f);
          sq1 += (ia == 1 ? s8a : 0.f) + (ib == 1 ? s8b : 0.f);
          sq2 += (ia == 2 ? s8a : 0.f) + (ib == 2 ? s8b : 0.f);
          tc_stage16(stg, hreg[c], hbase, p.Mcols, ch, vcols, lane);
        }
      }
      { float* w_ = sqb + (j * 128 + r) * 4; w_[0] = sq0; w_[1] = sq1; w_[2] = sq2; }
      tfb();
      nbar(bid, bn);
      if (warp == 4 && lane == 0) {
        if constexpr (PAIR) mb_arrive_cl(mapa_s(sa(te1 + buf), 0)); else mb_arrive1(te1 + buf);
      }
      float f0, f1, f2;
      {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
          if (jj < nwq) { const float* w_ = sqb + (jj * 128 + r) * 4; s0 += w_[0]; s1 += w_[1]; s2 += w_[2]; }
        const float g0 = sqrtf(s0), g1 = sqrtf(s1), g2 = sqrtf(s2);
        if (j == 0) {
          const float w0 = warp_sum(g0); if (lane == 0) atomicAdd(p.gsum + img0, w0);
          if (nimg > 1) { const float w1_ = warp_sum(g1); if (lane == 0) atomicAdd(p.gsum + img0 + 1, w1_); }
          if (nimg > 2) { const float w2_ = warp_sum(g2); if (lane == 0) atomicAdd(p.gsum + img0 + 2, w2_); }
        }
        f0 = gwk * g0; f1 = gwk * g1; f2 = gwk * g2;
      }
      if constexpr (PAIR) {
        if (j == 0) {
          float* cg_ = p.cgf + (size_t)img0 * C4 + k;
          cg_[0] = f0;
          if (nimg > 1) cg_[C4] = f1;
          if (nimg > 2) cg_[2 * C4] = f2;
        }
      } else {
#pragma unroll
        for (int c = 0; c < 5; c++) {
          const int ch = j + nwq * c;
          if (ch < nch) {
            const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
            const float fa = (ia == 0) ? f0 : (ia == 1) ? f1 : f2;
            const float fb = (ib == 0) ? f0 : (ib == 1) ? f1 : f2;
            uint32_t w[8];
#pragma unroll
            for (int x = 0; x < 4; x++) { w[x] = scale_h2(hreg[c][x], fa); w[4 + x] = scale_h2(hreg[c][4 + x], fb); }
            tc_stage16(stg, w, gbase, p.Mcols, ch, vcols, lane);
          }
        }
      }
#if TC_STAMPS
      if (warp == 4) { const unsigned long long q3 = gtime(); a_pre += q1 - q0; a_wait += q2 - q1; a_post += q3 - q2; }
#endif
    }
#if TC_STAMPS
    if (warp == 4 && lane == 0) { STAMP(4); p.st[cta * 24 + 11] = a_pre; p.st[cta * 24 + 12] = a_wait; p.st[cta * 24 + 13] = a_post; }
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); grid.sync(); tfa();
  if (tid == 0) STAMP(7);

  if constexpr (PAIR) {
    // ---------------- phase C (PAIR): cta_group::2 pairs, out += W2h x h + inv_b * (W2h x g) (+ W2.gb on k-split 0)
    constexpr int CTP = C_ / 256;
    if (warp == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      uint32_t it = 0;
      for (int u = cl; u < p.U2; u += NCL) {
        const int ks = u % p.KS, rest = u / p.KS, pm = rest % CTP, tt2 = rest / CTP;
        const int c0 = pm * 256 + (int)rank * 128, colh = tt2 * 128 + (int)rank * 64, kb0 = ks * p.KB2;
        for (int kk = 0; kk < p.KB2; kk++, it++) {
          const int s = (int)(it % TC_NST2P); const uint32_t n = it / TC_NST2P;
          if (it >= TC_NST2P) mb_wait(empty2 + s, (n & 1) ^ 1);
          if (elect1()) {
            unsigned char* st = ring + s * TC_STG2P;
            mb_expect(full2 + s, 16384 + 8192);
            tma2d(st, &p.mW2, (kb0 + kk) * 64, c0, full2 + s);
            tma2d(st + 16384, &p.mH, colh, (kb0 + kk) * 64, full2 + s);
          }
          __syncwarp();
        }
      }
    } else if (warp == 1) {
      if (rank == 0) {
        uint32_t it = 0, ui = 0;
        const uint32_t idesc = (1u << 4) | (1u << 16) | (16u << 17) | (16u << 24);
#if TC_STAMPS
        unsigned long long t_first = 0; uint32_t nstg = 0;
        if (lane == 0) STAMP(14);
#endif
        for (int u = cl; u < p.U2; u += NCL, ui++) {
          const uint32_t buf = ui & 1;
          if (ui >= 2) { mb_wait_cl(te2 + buf, ((ui >> 1) & 1) ^ 1); tfa(); }
          const uint32_t d1 = tbase + buf * 256, d2 = d1 + 128;
          for (int kk = 0; kk < p.KB2; kk++, it++) {
            const int s = (int)(it % TC_NST2P); const uint32_t n = it / TC_NST2P;
            mb_wait_cl(ready2 + s, n & 1);
            tfa();
#if TC_STAMPS
            if (it == 0) t_first = gtime();
            nstg++;
#endif
            if (elect1()) {
              const uint32_t a0 = sa(ring + s * TC_STG2P), bh = a0 + 16384, bg = a0 + 24576;
#pragma unroll
              for (int k = 0; k < 4; k++) {
                const uint64_t ad = sdesc(a0 + k * 32, 16, 1024);
                umma2(d1, ad, sdesc(bh + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
                umma2(d2, ad, sdesc(bg + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
              }
              umma_commit2(empty2 + s);
              if (kk == p.KB2 - 1) umma_commit2(tf2 + buf);
            }
            __syncwarp();
          }
        }
#if TC_STAMPS
        if (lane == 0) { p.st[cta * 24 + 15] = t_first; STAMP(16); p.st[cta * 24 + 17] = nstg; }
#endif
      }
    } else if (warp >= 12) {
      // g producers: warp pw = warp - 12 owns the stages it % 4 == pw and builds that stage's whole g half tile (lane =
      // k rows lane and lane + 32, all 8 chunks of 8 tokens; one image per chunk since HWS = 56 is a multiple of 8;
      // SW128 chunk position j ^ (r & 7) in both h and g buffers), so four stages are in production at once and the
      // per-stage fence + cluster-release arrive latencies overlap. gw*gf of the unit's K slice for the (at most 2)
      // images of this CTA's 64 tokens is staged in smem at unit start in one load round.
      const int pw = warp - 12, t = pw * 32 + lane;
      const uint32_t rdy = mapa_s(sa(ready2), 0);
      float* cbuf = reinterpret_cast<float*>(sm + TC_OFF_MS);
      const int kbn = p.KB2 * 64;
      uint32_t it = 0;
      for (int u = cl; u < p.U2; u += NCL) {
        const int ks = u % p.KS, rest = u / p.KS, tt2 = rest / CTP, kb0 = ks * p.KB2;
        const int colc = tt2 * 128 + (int)rank * 64, imgc = colc / HWS;
        uint32_t hisel = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) hisel |= (((colc + 8 * j) / HWS) != imgc) ? (1u << j) : 0u;
        nbar(3, 128);
        {
          float cv[16];
#pragma unroll
          for (int q2 = 0; q2 < 16; q2++) {
            const int i = t + q2 * 128, hi = (i >= kbn) ? 1 : 0, im = imgc + hi;
            cv[q2] = (i < 2 * kbn && im < p.B) ? __ldcg(p.cgf + (size_t)im * C4 + kb0 * 64 + (i - hi * kbn)) : 0.f;
          }
#pragma unroll
          for (int q2 = 0; q2 < 16; q2++) { const int i = t + q2 * 128; if (i < 2 * kbn) cbuf[i] = cv[q2]; }
        }
        nbar(3, 128);
        for (int kk = 0; kk < p.KB2; kk++, it++) {
          if ((int)(it & 3u) != pw) continue;
          const int s = (int)(it % TC_NST2P); const uint32_t n = it / TC_NST2P;
          mb_wait(full2 + s, n & 1);
          unsigned char* hb0 = ring + s * TC_STG2P + 16384;
#pragma unroll
          for (int h2 = 0; h2 < 2; h2++) {
            const int r = lane + 32 * h2;
            const float c_lo = cbuf[kk * 64 + r], c_hi = cbuf[kbn + kk * 64 + r];
            unsigned char* hb = hb0 + r * 128;
            uint4 w[8];
#pragma unroll
            for (int j = 0; j < 8; j++) w[j] = *reinterpret_cast<const uint4*>(hb + ((j ^ (r & 7)) << 4));
#pragma unroll
            for (int j = 0; j < 8; j++) {
              const float c = ((hisel >> j) & 1u) ? c_hi : c_lo;
              w[j].x = scale_h2(w[j].x, c); w[j].y = scale_h2(w[j].y, c); w[j].z = scale_h2(w[j].z, c); w[j].w = scale_h2(w[j].w, c);
              *reinterpret_cast<uint4*>(hb + 8192 + ((j ^ (r & 7)) << 4)) = w[j];
            }
          }
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
          if (lane == 0) mb_arrive_cl(rdy + (uint32_t)s * 8u);
        }
      }
    } else if (warp >= 4) {
      // drain warps 4-11: this CTA's 128 output channels (TMEM lanes) x 128 tokens
      const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
      float* fst = reinterpret_cast<float*>(stg_all + ew * (TC_STGB / 4));
      const uint32_t te2r = mapa_s(sa(te2), 0);
      uint32_t ui = 0;
      for (int u = cl; u < p.U2; u += NCL, ui++) {
        const int ks = u % p.KS, rest = u / p.KS, pm = rest % CTP, tt2 = rest / CTP;
        const int c0 = pm * 256 + (int)rank * 128, col0 = tt2 * 128;
        float wg = 0.f;
        if (ks == 0) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
        ebar();
        if (et < 128) {
          const int col = col0 + et, b = col / HWS, hw = col - b * HWS;
          const bool valid = (b < p.B) && (hw < HW_);
          inv_s[et] = valid ? 1.f / (__ldcg(p.gsum + b) / (float)C4 + p.eps) : 0.f;
          cb_s[et] = valid ? b * C_ * HW_ + hw : -1;
        }
        ebar();
        const uint32_t buf = ui & 1;
        mb_wait(tf2 + buf, (ui >> 1) & 1);
        tfa();
#pragma unroll
        for (int cs = 0; cs < 2; cs++) {
          const int ch = grp + 2 * cs;
          uint32_t v1[32], v2[32];
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + 128 + ch * 32, v2);
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
#pragma unroll
            for (int j = 0; j < 16; j++) { const int jc = hh * 16 + j; fst[lane * 17 + j] = fmaf(inv_s[ch * 32 + jc], __uint_as_float(v2[jc]), __uint_as_float(v1[jc]) + wg); }
            __syncwarp();
            const int cj = lane & 15;
            const int cb = cb_s[ch * 32 + hh * 16 + cj];
            if (cb >= 0) {
              float* ob = p.out + cb + (size_t)(c0 + q * 32) * HW_;
#pragma unroll 4
              for (int rr = 0; rr < 16; rr++) { const int row = 2 * rr + (lane >> 4); atomicAdd(ob + (size_t)row * HW_, fst[row * 17 + cj]); }
            }
            __syncwarp();
          }
        }
        tfb(); __syncwarp();
        if (lane == 0) mb_arrive_cl(te2r + buf * 8u);
      }
#if TC_STAMPS
      if (warp == 4 && lane == 0) STAMP(9);
#endif
    }
  } else {
    // ---------------- phase C (1SM): GEMM2 split-K, out += W2h x h + inv_b * (W2h x g) (+ W2.gb on k-split 0)
    if (warp == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      uint32_t it = 0;
      for (int u = cta; u < p.U2; u += P) {
        const int pair = u / p.KS, ks = u - pair * p.KS, ct = pair % CT, tt2 = pair / CT;
        const int c0 = ct * 128, col0 = tt2 * 128, kb0 = ks * p.KB2;
        for (int kk = 0; kk < p.KB2; kk++, it++) {
          const int kb = kb0 + kk, s = it & 3; const uint32_t n = it >> 2;
          if (it >= 4) mb_wait(empty2 + s, (n & 1) ^ 1);
          if (elect1()) {
            unsigned char* st = ring + s * TC_STG2;
            mb_expect(full2 + s, TC_STG2);
            tma2d(st, &p.mW2, kb * 64, c0, full2 + s);
            tma2d(st + 16384, &p.mH, col0, kb * 64, full2 + s);
            tma2d(st + 24576, &p.mH, col0 + 64, kb * 64, full2 + s);
            tma2d(st + 32768, &p.mG, col0, kb * 64, full2 + s);
            tma2d(st + 40960, &p.mG, col0 + 64, kb * 64, full2 + s);
          }
          __syncwarp();
        }
      }
    } else if (warp == 1) {
      uint32_t it = 0, ui = 0;
      const uint32_t idesc = (1u << 4) | (1u << 16) | (16u << 17) | (8u << 24);
      for (int u = cta; u < p.U2; u += P, ui++) {
        const uint32_t buf = ui & 1;
        if (ui >= 2) { mb_wait(te2 + buf, ((ui >> 1) & 1) ^ 1); tfa(); }
        const uint32_t d1 = tbase + buf * 256, d2 = d1 + 128;
        for (int kk = 0; kk < p.KB2; kk++, it++) {
          const int s = it & 3; const uint32_t n = it >> 2;
          mb_wait(full2 + s, n & 1);
          tfa();
          if (elect1()) {
            const uint32_t a0 = sa(ring + s * TC_STG2), bh = a0 + 16384, bg = a0 + 32768;
#pragma unroll
            for (int k = 0; k < 4; k++) {
              const uint64_t ad = sdesc(a0 + k * 32, 16, 1024);
              umma(d1, ad, sdesc(bh + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
              umma(d2, ad, sdesc(bg + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
            }
            umma_commit(empty2 + s);
            if (kk == p.KB2 - 1) umma_commit(tf2 + buf);
          }
          __syncwarp();
        }
      }
    } else if (warp >= 4 && warp < 12) {
      const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
      float* fst = reinterpret_cast<float*>(stg_all + ew * (TC_STGB / 4));
      uint32_t ui = 0;
      for (int u = cta; u < p.U2; u += P, ui++) {
        const int pair = u / p.KS, ks = u - pair * p.KS, ct = pair % CT, tt2 = pair / CT;
        const int c0 = ct * 128, col0 = tt2 * 128;
        float wg = 0.f;
        if (ks == 0) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
        ebar();
        if (et < 128) {
          const int col = col0 + et, b = col / HWS, hw = col - b * HWS;
          const bool valid = (b < p.B) && (hw < HW_);
          inv_s[et] = valid ? 1.f / (__ldcg(p.gsum + b) / (float)C4 + p.eps) : 0.f;
          cb_s[et] = valid ? b * C_ * HW_ + hw : -1;
        }
        ebar();
        const uint32_t buf = ui & 1;
        mb_wait(tf2 + buf, (ui >> 1) & 1);
        tfa();
        if constexpr (HW_ % 4 == 0) {
          // 14x14: 4 consecutive tokens of one image are 16-B aligned in NCHW, so each lane (output channel) reduces
          // straight from registers with a float4 red.add
          float* orow = p.out + (size_t)(c0 + r) * HW_;
#pragma unroll
          for (int cs = 0; cs < 2; cs++) {
            const int ch = grp + 2 * cs;
            uint32_t v1[32], v2[32];
            tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
            tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + 128 + ch * 32, v2);
#pragma unroll
            for (int g4 = 0; g4 < 8; g4++) {
              const int jc = g4 * 4;
              const int cb = cb_s[ch * 32 + jc];
              if (cb >= 0) {
                float4 o;
                o.x = fmaf(inv_s[ch * 32 + jc + 0], __uint_as_float(v2[jc + 0]), __uint_as_float(v1[jc + 0]) + wg);
                o.y = fmaf(inv_s[ch * 32 + jc + 1], __uint_as_float(v2[jc + 1]), __uint_as_float(v1[jc + 1]) + wg);
                o.z = fmaf(inv_s[ch * 32 + jc + 2], __uint_as_float(v2[jc + 2]), __uint_as_float(v1[jc + 2]) + wg);
                o.w = fmaf(inv_s[ch * 32 + jc + 3], __uint_as_float(v2[jc + 3]), __uint_as_float(v1[jc + 3]) + wg);
                atomicAdd(reinterpret_cast<float4*>(orow + cb), o);
              }
            }
          }
        } else {
          // 7x7: transpose 16 columns through shared memory so 16 lanes reduce 16 consecutive tokens of one channel
#pragma unroll
          for (int cs = 0; cs < 2; cs++) {
            const int ch = grp + 2 * cs;
            uint32_t v1[32], v2[32];
            tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
            tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + 128 + ch * 32, v2);
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
#pragma unroll
              for (int j = 0; j < 16; j++) { const int jc = hh * 16 + j; fst[lane * 17 + j] = fmaf(inv_s[ch * 32 + jc], __uint_as_float(v2[jc]), __uint_as_float(v1[jc]) + wg); }
              __syncwarp();
              const int cj = lane & 15;
              const int cb = cb_s[ch * 32 + hh * 16 + cj];
              if (cb >= 0) {
                float* ob = p.out + cb + (size_t)(c0 + q * 32) * HW_;
#pragma unroll 4
                for (int rr = 0; rr < 16; rr++) { const int row = 2 * rr + (lane >> 4); atomicAdd(ob + (size_t)row * HW_, fst[row * 17 + cj]); }
              }
              __syncwarp();
            }
          }
        }
        tfb(); __syncwarp();
        if (lane == 0) mb_arrive1(te2 + buf);
      }
#if TC_STAMPS
      if (warp == 4 && lane == 0) STAMP(9);
#endif
    }
  }
  tfb(); __syncthreads();
  if constexpr (PAIR) {
    cluster_sync_all();
    if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tbase));
  } else {
    if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
  }
}

// =====================================================================================================
// tcgen05 swap-AB cooperative path for the small-C rows (C = 96 / 160 / 192 at 28x28, 56x56, 449x7)
// Phase A: every warp converts weight units first (fp16 W1' with the LN-weight fold, row sums u, v = W1.lnb + b1,
//   fp16 W2, W2.gb partials; loads issued up front), then all 12 warps run the conv units: (image, PG-plane group,
//   row strip) from 16-B-aligned superset bulk copies; each conv thread loops over a plane subgroup, writes fp16 d and
//   out = x + b2 from registers and keeps its tokens' LN partial sums in registers. Sync 1.
// Phase B: GEMM1 tiles = 128 hidden rows x one NC1-token chunk of one image; LN fold, exact GELU (A&S erfc form, r8),
//   fp16 h, per-(image,k) sum of squares with one red.add per row per tile. Sync 2.
// Phase C: GEMM2 split-K units (128 output channels x NC2 tokens x K slice); the epilogue warps scale the hidden B
//   tile in smem by s_b[k] = 1 + gw*gf*inv_b before the MMA; coalesced red.add into the prefilled output.
// SC_STAMPS: globaltimer diagnostics (off in the shipped build).
// =====================================================================================================
#ifndef SC_STAMPS
#define SC_STAMPS 0
#endif
#define SC_NT 384
#define SC_RING (192 * 1024)
#define SC_OFF_STG SC_RING
#define SC_OFF_MS (SC_OFF_STG + 8 * TC_STGB)
#define SC_OFF_GF (SC_OFF_MS + 2048)
#define SC_OFF_SS (SC_OFF_GF + 3072)
#define SC_OFF_GW (SC_OFF_SS + 3072)
#define SC_OFF_SQ (SC_OFF_GW + 3072)
#define SC_OFF_BAR (SC_OFF_SQ + 1024)
#define SC_SMEM (SC_OFF_BAR + 40 * 8 + 1024)
#define SC_NST2 6
#define SC_STG2 32768
#define SC_MAXWU 3
struct SCP {
  CUtensorMap mW1, mW2, mD, mH;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid;
  float *part, *uk, *vk, *wgb, *gsq;
  unsigned long long* st;
  int B, H, HW, HWS, Mcols, NS, SR, SLOT, units, PSUB;
  int NC1, N1, NAT1, CPI1, T1, STG1;
  int NC2, N2, NAT2, CPI2, KS, KB2, U2;
  float eps, ln_eps;
};
#if SC_STAMPS
#define SSTAMP(i) do { p.st[cta * 24 + (i)] = gtime(); } while (0)
#else
#define SSTAMP(i) do {} while (0)
#endif

template <int C_, int W_, int PG_>
__global__ void __launch_bounds__(SC_NT, 1) sc_kernel(const __grid_constant__ SCP p) {
  constexpr int C4 = 4 * C_, G = C_ / PG_, HT = C4 / 128, CT = (C_ + 127) / 128, KB1 = (C_ + 63) / 64;
  constexpr int SEG = W_ / 7, NQ = (C_ / 4 + 31) / 32;
  static_assert(W_ % 7 == 0 && C4 % 128 == 0 && C_ % PG_ == 0 && C4 <= 768 && C_ % 4 == 0 && (PG_ * 49) % 4 == 0, "sc shape");
  extern __shared__ __align__(1024) unsigned char sc_raw[];
  unsigned char* sm = sc_raw + ((1024u - (sa(sc_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  uint32_t* stg_all = reinterpret_cast<uint32_t*>(sm + SC_OFF_STG);
  float4* ms_s = reinterpret_cast<float4*>(sm + SC_OFF_MS);
  float* gf_s = reinterpret_cast<float*>(sm + SC_OFF_GF);
  float* ss_s = reinterpret_cast<float*>(sm + SC_OFF_SS);
  float* gw_s = reinterpret_cast<float*>(sm + SC_OFF_GW);
  float* sq_s = reinterpret_cast<float*>(sm + SC_OFF_SQ);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + SC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *ready2 = bars + 18, *empty2 = bars + 24, *tf2 = bars + 30, *te2 = bars + 32, *cbar = bars + 34;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 36);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  cg::grid_group grid = cg::this_grid();
  if (tid == 0) {
    for (int i = 0; i < 35; i++) {
      const bool e8 = (i == 10 || i == 11 || (i >= 18 && i < 24) || i == 32 || i == 33);
      mb_init(bars + i, e8 ? 8u : 1u);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tfb(); __syncthreads(); tfa();
  const uint32_t tbase = *tslot;
  if (tid == 0) SSTAMP(0);

  // ---------------- phase A
  const int H = p.H, HW = p.HW, HWS = p.HWS, SR = p.SR, NS = p.NS, SLOT = p.SLOT, PSUB = p.PSUB;
  const int tps = SR * W_;
  const int nu = (cta < p.units) ? (p.units - cta + P - 1) / P : 0;
  float* xs = reinterpret_cast<float*>(ring);
  float* wsm = xs + nu * PG_ * SLOT;
  float* stp = wsm + nu * PG_ * 49;
  if (tid == 0) {
    // one transaction: gw (all CTAs), and per conv unit the PG depthwise filters and PG plane strips (with halo rows)
    uint32_t tot = C4 * 4;
    for (int pass = 0; pass < 2; pass++) {
      if (pass == 1) { mb_expect(cbar, tot); bulk_g2s(gw_s, p.gw, C4 * 4, cbar); }
      for (int i = 0; i < nu; i++) {
        const int u = cta + i * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
        const int h0 = s * SR, h1 = min(H, h0 + SR), r0 = max(0, h0 - 3), r1 = min(H, h1 + 3);
        if (pass == 0) tot += PG_ * 49 * 4;
        else bulk_g2s(wsm + i * PG_ * 49, p.dww + (size_t)g * PG_ * 49, PG_ * 49 * 4, cbar);
        for (int pp = 0; pp < PG_; pp++) {
          const size_t pb = ((size_t)b * C_ + g * PG_ + pp) * HW;
          const size_t a0 = ((pb + (size_t)r0 * W_) * 4) & ~(size_t)15;
          const size_t a1 = ((pb + (size_t)r1 * W_) * 4 + 15) & ~(size_t)15;
          if (pass == 0) tot += (uint32_t)(a1 - a0);
          else bulk_g2s(xs + (i * PG_ + pp) * SLOT, reinterpret_cast<const char*>(p.x) + a0, (uint32_t)(a1 - a0), cbar);
        }
      }
    }
  }
  for (int i = cta * SC_NT + tid; i < p.B * C4; i += P * SC_NT) p.gsq[i] = 0.f;
  {
    // weight units of C floats each, highest CTAs first; at most SC_MAXWU units per warp, all loads issued up front
    const int nw = P * (SC_NT / 32), wbase = (P - 1 - cta) * (SC_NT / 32) + warp;
    float4 v[SC_MAXWU][NQ];
#pragma unroll
    for (int k = 0; k < SC_MAXWU; k++) {
      const int wu = wbase + k * nw;
      if (wu < 8 * C_) {
        const bool is1 = wu < C4;
        const int row = is1 ? wu : ((wu - C4) >> 2), qtr = is1 ? 0 : ((wu - C4) & 3);
        const float4* src = reinterpret_cast<const float4*>(is1 ? p.w1 + (size_t)row * C_ : p.w2 + (size_t)row * C4 + qtr * C_);
#pragma unroll
        for (int j = 0; j < NQ; j++) { const int c4 = lane + 32 * j; v[k][j] = (c4 < C_ / 4) ? __ldg(src + c4) : make_float4(0.f, 0.f, 0.f, 0.f); }
      }
    }
#pragma unroll
    for (int k = 0; k < SC_MAXWU; k++) {
      const int wu = wbase + k * nw;
      if (wu < 8 * C_) {
        const bool is1 = wu < C4;
        const int row = is1 ? wu : ((wu - C4) >> 2), qtr = is1 ? 0 : ((wu - C4) & 3);
        uint2* dst = reinterpret_cast<uint2*>(is1 ? p.w1h + (size_t)row * C_ : p.w2h + (size_t)row * C4 + qtr * C_);
        if (is1) {
          float su = 0.f, sv = 0.f;
#pragma unroll
          for (int j = 0; j < NQ; j++) {
            const int c4 = lane + 32 * j;
            if (c4 < C_ / 4) {
              const float4 lw = __ldg(reinterpret_cast<const float4*>(p.lnw) + c4);
              const float4 lb = __ldg(reinterpret_cast<const float4*>(p.lnb) + c4);
              const __half2 h0 = __floats2half2_rn(v[k][j].x * lw.x, v[k][j].y * lw.y);
              const __half2 h1 = __floats2half2_rn(v[k][j].z * lw.z, v[k][j].w * lw.w);
              const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
              su += (f0.x + f0.y) + (f1.x + f1.y);
              sv += v[k][j].x * lb.x + v[k][j].y * lb.y + v[k][j].z * lb.z + v[k][j].w * lb.w;
              uint2 o; o.x = h2u(h0); o.y = h2u(h1);
              dst[c4] = o;
            }
          }
          su = warp_sum(su); sv = warp_sum(sv);
          if (lane == 0) { p.uk[row] = su; p.vk[row] = sv + __ldg(p.b1 + row); }
        } else {
          const float4* gb4 = reinterpret_cast<const float4*>(p.gb + qtr * C_);
          float sg = 0.f;
#pragma unroll
          for (int j = 0; j < NQ; j++) {
            const int c4 = lane + 32 * j;
            if (c4 < C_ / 4) {
              const float4 g4 = __ldg(gb4 + c4);
              sg += v[k][j].x * g4.x + v[k][j].y * g4.y + v[k][j].z * g4.z + v[k][j].w * g4.w;
              uint2 o; o.x = h2u(__floats2half2_rn(v[k][j].x, v[k][j].y)); o.y = h2u(__floats2half2_rn(v[k][j].z, v[k][j].w));
              dst[c4] = o;
            }
          }
          sg = warp_sum(sg);
          if (lane == 0) p.wgb[row * 4 + qtr] = sg;
        }
      }
    }
  }
  if (tid == 0) SSTAMP(6);
  mb_wait(cbar, 0);
  if (nu > 0) {
    if (tid == 0) SSTAMP(7);
    // conv thread = (unit, plane subgroup, row, 7-wide segment): loops over PG/PSUB planes, writes fp16 d and
    // out = x + b2 from registers, keeps its 7 tokens' LN partial sums in registers; rows are clamped, not branched
    const int pos = SR * SEG, per_unit = PSUB * pos;
    for (int itm = tid; itm < nu * per_unit; itm += SC_NT) {
      const int ui = itm / per_unit, r1_ = itm - ui * per_unit, ps = r1_ / pos, r2_ = r1_ - ps * pos;
      const int hl = r2_ / SEG, w0 = (r2_ - hl * SEG) * 7;
      const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
      const int h0 = s * SR, h = h0 + hl;
      if (h >= min(H, h0 + SR)) continue;
      const int r0 = max(0, h0 - 3);
      float s1[7], s2[7];
#pragma unroll
      for (int o = 0; o < 7; o++) { s1[o] = 0.f; s2[o] = 0.f; }
#pragma unroll 1
      for (int pp = ps; pp < PG_; pp += PSUB) {
        const int c = g * PG_ + pp;
        const size_t pbase = ((size_t)b * C_ + c) * HW;
        const float* xp = xs + (ui * PG_ + pp) * SLOT + (int)((pbase + (size_t)r0 * W_) & 3) - r0 * W_;
        const float* wp = wsm + (ui * PG_ + pp) * 49;
        float acc[7], xc[7];
#pragma unroll
        for (int o = 0; o < 7; o++) acc[o] = 0.f;
#pragma unroll
        for (int i = 0; i < 7; i++) {
          const int hh = h + i - 3;
          const bool rok = (hh >= 0) && (hh < H);
          const int hs = rok ? hh : h;
          float v[13];
#pragma unroll
          for (int jj = 0; jj < 13; jj++) {
            const int col = w0 - 3 + jj;
            const bool cok = (col >= 0) && (col < W_);
            const float xv = xp[hs * W_ + min(max(col, 0), W_ - 1)];
            v[jj] = (rok && cok) ? xv : 0.f;
          }
          if (i == 3) {
#pragma unroll
            for (int o = 0; o < 7; o++) xc[o] = v[o + 3];
          }
#pragma unroll
          for (int jj = 0; jj < 7; jj++) {
            const float wv = wp[i * 7 + jj];
#pragma unroll
            for (int o = 0; o < 7; o++) acc[o] = fmaf(v[o + jj], wv, acc[o]);
          }
        }
        const float bias = __ldg(p.dwb + c), b2c = __ldg(p.b2 + c);
        __half* dd = p.d16 + (size_t)c * p.Mcols + b * HWS + h * W_ + w0;
        float* oo = p.out + pbase + h * W_ + w0;
#pragma unroll
        for (int o = 0; o < 7; o++) {
          const float dv = acc[o] + bias;
          dd[o] = __float2half_rn(dv);
          oo[o] = xc[o] + b2c;
          s1[o] += dv; s2[o] = fmaf(dv, dv, s2[o]);
        }
      }
      float* sp = stp + ((size_t)(ui * PSUB + ps) * tps + hl * W_ + w0) * 2;
#pragma unroll
      for (int o = 0; o < 7; o++) { sp[2 * o] = s1[o]; sp[2 * o + 1] = s2[o]; }
    }
    __syncthreads();
    if (tid == 0) SSTAMP(8);
    for (int e = tid; e < nu * tps; e += SC_NT) {
      const int ui = e / tps, lt = e - ui * tps;
      const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
      const int h0 = s * SR;
      if (h0 + lt / W_ >= min(H, h0 + SR)) continue;
      float a1 = 0.f, a2 = 0.f;
      for (int q2 = 0; q2 < PSUB; q2++) { const float2 v = reinterpret_cast<const float2*>(stp)[(size_t)(ui * PSUB + q2) * tps + lt]; a1 += v.x; a2 += v.y; }
      reinterpret_cast<float2*>(p.part)[((size_t)b * G + g) * HW + h0 * W_ + lt] = make_float2(a1, a2);
    }
    if (HWS > HW) {
      for (int e = tid; e < nu * PG_ * 8; e += SC_NT) {
        const int pl = e >> 3, jj = e & 7, ui = pl / PG_, pp = pl - ui * PG_;
        const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
        if (min(H, s * SR + SR) == H && HW + jj < HWS)
          p.d16[(size_t)(g * PG_ + pp) * p.Mcols + b * HWS + HW + jj] = __float2half_rn(0.f);
      }
    }
  }
#if SC_STAMPS
  __syncthreads(); if (tid == 0) SSTAMP(1);
#endif
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); grid.sync(); tfa();
  if (tid == 0) SSTAMP(2);

  // ---------------- phase B: GEMM1 per (hidden tile, token chunk)
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    uint32_t it = 0;
    for (int t = cta; t < p.T1; t += P) {
      const int ht = t % HT, cj = t / HT, b = cj / p.CPI1, j = cj - b * p.CPI1;
      const int m0 = ht * 128, tok0 = b * HWS + j * p.NC1;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = it & 3; const uint32_t n = it >> 2;
        if (it >= 4) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          mb_expect(full1 + s, 16384 + p.NAT1 * 8192);
          tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
          for (int a = 0; a < p.NAT1; a++) tma2d(st + 16384 + a * 8192, &p.mD, tok0 + a * 64, kb * 64, full1 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, ti = 0;
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(p.N1 >> 3) << 17) | (8u << 24);
    for (int t = cta; t < p.T1; t += P, ti++) {
      const uint32_t buf = ti & 1;
      if (ti >= 2) { mb_wait(te1 + buf, ((ti >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = it & 3; const uint32_t n = it >> 2;
        mb_wait(full1 + s, n & 1);
        tfa();
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * p.STG1), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) umma(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kb | k));
          umma_commit(empty1 + s);
          if (kb == KB1 - 1) umma_commit(tf1 + buf);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    uint32_t* stg = stg_all + ew * (TC_STGB / 4);
    const int nch = (p.N1 + 31) >> 5;
    uint32_t ti = 0;
    for (int t = cta; t < p.T1; t += P, ti++) {
      const int ht = t % HT, cj = t / HT, b = cj / p.CPI1, j = cj - b * p.CPI1;
      const int m0 = ht * 128, tok0 = b * HWS + j * p.NC1, vcols = min(p.NC1, HW - j * p.NC1);
      const int k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k);
      ebar();
      if (et < 128) {
        // per-token LayerNorm statistics from the per-plane-group partials; one token per thread
        float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
        if (et < vcols) {
          const float2* pp = reinterpret_cast<const float2*>(p.part) + (size_t)b * G * HW + j * p.NC1 + et;
          float s1 = 0.f, s2 = 0.f;
#pragma unroll
          for (int g = 0; g < G; g++) { const float2 v = __ldcg(pp + (size_t)g * HW); s1 += v.x; s2 += v.y; }
          const float mu = s1 / (float)C_;
          const float var = fmaxf(s2 / (float)C_ - mu * mu, 0.f);
          m = make_float4(mu, rsqrtf(var + p.ln_eps), 1.f, 0.f);
        }
        ms_s[et] = m;
      }
      ebar();
      const uint32_t buf = ti & 1;
      mb_wait(tf1 + buf, (ti >> 1) & 1);
      tfa();
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16) + buf * 256;
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      float sq = 0.f;
#pragma unroll 1
      for (int ch = grp; ch < nch; ch += 2) {
        uint32_t v[32];
        tld32(taddr + ch * 32, v);
        uint32_t hw2[16];
#pragma unroll
        for (int jj = 0; jj < 32; jj += 2) {
          float hf[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const float4 m = ms_s[ch * 32 + jj + e];
            const float z = fmaf(m.y, __uint_as_float(v[jj + e]) - m.x * uk, vk);
            const float h = __half2float(__float2half_rn(gelu_as(z)));
            hf[e] = h;
            sq += (m.z != 0.f) ? h * h : 0.f;
          }
          hw2[jj >> 1] = h2u(__floats2half2_rn(hf[0], hf[1]));
        }
        uint4* srow = reinterpret_cast<uint4*>(stg + lane * 20);
#pragma unroll
        for (int qd = 0; qd < 4; qd++) srow[qd] = make_uint4(hw2[4 * qd], hw2[4 * qd + 1], hw2[4 * qd + 2], hw2[4 * qd + 3]);
        __syncwarp();
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
          const int row = i2 * 8 + (lane >> 2), colb = ch * 32 + (lane & 3) * 8;
          const uint4 val = *reinterpret_cast<const uint4*>(stg + row * 20 + (lane & 3) * 4);
          if (colb < vcols) *reinterpret_cast<uint4*>(hbase + (size_t)row * p.Mcols + colb) = val;
        }
        __syncwarp();
      }
      tfb(); __syncwarp();
      if (lane == 0) mb_arrive1(te1 + buf);
      sq_s[grp * 128 + r] = sq;
      ebar();
      if (grp == 0) atomicAdd(p.gsq + (size_t)b * C4 + k, sq_s[r] + sq_s[128 + r]);
    }
    if (warp == 4 && lane == 0) SSTAMP(3);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  tfb(); grid.sync(); tfa();

  // ---------------- phase C: GEMM2 split-K with the GRN scale applied to the hidden tile in smem
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    uint32_t it = 0;
    for (int u = cta; u < p.U2; u += P) {
      const int ks = u % p.KS, rest = u / p.KS, ct = rest % CT, cj = rest / CT, b = cj / p.CPI2, j = cj - b * p.CPI2;
      const int c0 = ct * 128, tok0 = b * HWS + j * p.NC2, kb0 = ks * p.KB2;
      for (int kk = 0; kk < p.KB2; kk++, it++) {
        const int s = it % SC_NST2; const uint32_t n = it / SC_NST2;
        if (it >= SC_NST2) mb_wait(empty2 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * SC_STG2;
          mb_expect(full2 + s, 16384 + p.NAT2 * 8192);
          tma2d(st, &p.mW2, (kb0 + kk) * 64, c0, full2 + s);
          for (int a = 0; a < p.NAT2; a++) tma2d(st + 16384 + a * 8192, &p.mH, tok0 + a * 64, (kb0 + kk) * 64, full2 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, ui = 0;
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(p.N2 >> 3) << 17) | (8u << 24);
    for (int u = cta; u < p.U2; u += P, ui++) {
      const uint32_t buf = ui & 1;
      if (ui >= 2) { mb_wait(te2 + buf, ((ui >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      for (int kk = 0; kk < p.KB2; kk++, it++) {
        const int s = it % SC_NST2; const uint32_t n = it / SC_NST2;
        mb_wait(ready2 + s, n & 1);
        tfa();
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * SC_STG2), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) umma(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
          umma_commit(empty2 + s);
          if (kk == p.KB2 - 1) umma_commit(tf2 + buf);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    if (warp == 4 && lane == 0) SSTAMP(5);
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    float* fst = reinterpret_cast<float*>(stg_all + ew * (TC_STGB / 4));
    const bool hw4 = (HW & 3) == 0;
    const int nch2 = (p.N2 + 31) >> 5;
    uint32_t ui = 0, itc = 0;
    for (int u = cta; u < p.U2; u += P, ui++) {
      const int ks = u % p.KS, rest = u / p.KS, ct = rest % CT, cj = rest / CT, b = cj / p.CPI2, j = cj - b * p.CPI2;
      const int c0 = ct * 128, kb0 = ks * p.KB2, vcols = min(p.NC2, HW - j * p.NC2);
      float wg = 0.f;
      if (ks == 0 && c0 + r < C_) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
      ebar();
      // GRN: gf = sqrt(sum of squares), inv_b = 1 / (mean_k gf + eps), s[k] = 1 + gw[k] * gf[k] * inv_b for this K slice
      {
        float a = 0.f;
        for (int kk2 = et; kk2 < C4; kk2 += 256) { const float g = sqrtf(__ldcg(p.gsq + (size_t)b * C4 + kk2)); gf_s[kk2] = g; a += g; }
        a = warp_sum(a);
        if (lane == 0) sq_s[ew] = a;
      }
      ebar();
      {
        float tot = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < 8; w8++) tot += sq_s[w8];
        const float inv = 1.f / (tot / (float)C4 + p.eps);
        for (int i = et; i < p.KB2 * 64; i += 256) { const int kx = kb0 * 64 + i; ss_s[i] = fmaf(gw_s[kx] * gf_s[kx], inv, 1.f); }
      }
      ebar();
      for (int kk = 0; kk < p.KB2; kk++, itc++) {
        const int s = itc % SC_NST2; const uint32_t n = itc / SC_NST2;
        mb_wait(full2 + s, n & 1);
        uint4* bb = reinterpret_cast<uint4*>(ring + s * SC_STG2 + 16384);
        const float* sk = ss_s + kk * 64;
        for (int ci = et; ci < p.NAT2 * 512; ci += 256) {
          const float sc = sk[(ci >> 3) & 63];
          uint4 w = bb[ci];
          w.x = scale_h2(w.x, sc); w.y = scale_h2(w.y, sc); w.z = scale_h2(w.z, sc); w.w = scale_h2(w.w, sc);
          bb[ci] = w;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive1(ready2 + s);
      }
      const uint32_t buf = ui & 1;
      mb_wait(tf2 + buf, (ui >> 1) & 1);
      tfa();
      const int crow0 = c0 + q * 32;
      float* obase = p.out + ((size_t)b * C_ + crow0) * HW + j * p.NC2;
#pragma unroll 1
      for (int ch = grp; ch < nch2; ch += 2) {
        uint32_t v[32];
        tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v);
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
#pragma unroll
          for (int jj = 0; jj < 16; jj++) fst[lane * 17 + jj] = __uint_as_float(v[hh * 16 + jj]) + wg;
          __syncwarp();
          const int colb = ch * 32 + hh * 16;
          if (hw4) {
#pragma unroll
            for (int i4 = 0; i4 < 4; i4++) {
              const int row = i4 * 8 + (lane >> 2), cq = (lane & 3) * 4;
              if (crow0 + row < C_ && colb + cq < vcols) {
                const float* f = fst + row * 17 + cq;
                atomicAdd(reinterpret_cast<float4*>(obase + (size_t)row * HW + colb + cq), make_float4(f[0], f[1], f[2], f[3]));
              }
            }
          } else {
#pragma unroll 4
            for (int rr = 0; rr < 16; rr++) {
              const int row = 2 * rr + (lane >> 4), cx = lane & 15;
              if (crow0 + row < C_ && colb + cx < vcols) atomicAdd(obase + (size_t)row * HW + colb + cx, fst[row * 17 + cx]);
            }
          }
          __syncwarp();
        }
      }
      tfb(); __syncwarp();
      if (lane == 0) mb_arrive1(te2 + buf);
    }
    if (warp == 4 && lane == 0) SSTAMP(4);
  }
  tfb(); __syncthreads();
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
}

// ---------------------------------------------------------------- host side of the tcgen05 paths
struct TCRow { int B, C, H, W, NI, KS; };
// 1SM rows: GEMM2 units = (C/128) x token tiles x KS. Pair rows (C=768, B >= 16): GEMM1 pair tiles =
// (C4/256) x image groups and GEMM2 pair units = (C/256) x token tiles x KS, spread over the 2-CTA clusters.
static const TCRow kTCRows[6] = {
    {2, 768, 7, 7, 1, 12}, {8, 768, 7, 7, 2, 6}, {16, 768, 7, 7, 3, 3},
    {32, 768, 7, 7, 3, 3}, {8, 384, 14, 14, 1, 3}, {16, 384, 14, 14, 1, 3},
};
static inline bool tc_pair(int B, int C) { return C == 768 && B >= 16; }

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};

static inline size_t tc_al(size_t v) { return (v + 255) / 256 * 256; }

struct TCPlan { TCRow r; int HW, HWS, Mcols, C4, G, T1, NAT1, STG1, U2, KB2;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_gq, o_part, o_uk, o_vk, o_wgb, o_gsum, o_cgf, bytes; };

static bool tc_plan(int B, int C, int H, int W, TCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 6; i++)
    if (kTCRows[i].B == B && kTCRows[i].C == C && kTCRows[i].H == H && kTCRows[i].W == W) idx = i;
  if (idx < 0) return false;
  const TCRow r = kTCRows[idx];
  const bool pair = tc_pair(B, C);
  pl.r = r;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / 32;
  pl.T1 = (pl.C4 / (pair ? 256 : 128)) * ((B + r.NI - 1) / r.NI);
  {
    const int N = (r.NI * pl.HWS + 15) & ~15, NH = pair ? N / 2 : N;
    pl.NAT1 = (NH + 63) / 64;
  }
  if (pl.NAT1 > 4 || r.NI > 3) return false;
  pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4) return false;
  if (pair && pl.KB2 > 16) return false;  // the g producers stage at most 2 x 16 x 64 gw*gf values per unit
  pl.U2 = (pair ? C / 256 : C / 128) * ((pl.Mcols + 127) / 128) * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_gq = o; o = tc_al(o + (pair ? 256 : (size_t)pl.C4 * pl.Mcols * 2));
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsum = o; o = tc_al(o + (size_t)B * 4);
  pl.o_cgf = o; o = tc_al(o + (size_t)B * pl.C4 * 4);
  pl.bytes = o;
  return true;
}

// fp16 2-D map over a row-major [outer][inner] tensor with row pitch inner*2 bytes, 128B swizzle
static bool enc16(CUtensorMap* m, const void* base, uint64_t inner, uint64_t outer, uint32_t box_in, uint32_t box_out) {
  const cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  const cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  const cuuint32_t box[2] = {box_in, box_out};
  const cuuint32_t estr[2] = {1u, 1u};
  CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*)base, dims, strides, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

// grid of the 2-CTA cluster launch: 2 x the resident clusters (74 on B200 with 227 KB per CTA), never above the SMs
static int tc_pair_grid(int nsm) {
  static int g = 0;
  if (g == 0) {
    cudaLaunchConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(TC_NT); cfg.dynamicSmemBytes = TC_SMEM;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    int ncl = 0;
    if (cudaOccupancyMaxActiveClusters(&ncl, (const void*)tc_kernel<768, 49, true>, &cfg) != cudaSuccess) ncl = 0;
    int gg = 2 * ncl;
    if (gg > nsm) gg = nsm & ~1;
    g = gg;
  }
  return g;
}

static const char* tc_launch(const WSArgs* a, int nsm, void* stream_v) {
  TCPlan pl;
  if (!tc_plan(a->B, a->C, a->H, a->W, pl)) return "ws_launch: unsupported shape";
  TCP p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  const bool pair = tc_pair(a->B, a->C);
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid); p.gq = (__half*)(wsb + pl.o_gq);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsum = (float*)(wsb + pl.o_gsum); p.cgf = (float*)(wsb + pl.o_cgf);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DTC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  if (!pair && !enc16(&p.mG, p.gq, pl.Mcols, C4, 64, 64)) return "ws_launch: map G";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.Mcols = pl.Mcols; p.units = a->B * pl.G; p.NI = pl.r.NI; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.NST1 = (TC_B_STG / pl.STG1 < 4) ? TC_B_STG / pl.STG1 : 4;
  p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2; p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  cudaError_t e;
  if (pair) {
    const int grid = tc_pair_grid(nsm);
    if (grid < 2) return "ws_launch: no co-resident 2-CTA clusters";
    cudaLaunchConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(TC_NT); cfg.dynamicSmemBytes = TC_SMEM; cfg.stream = (cudaStream_t)stream_v;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeCooperative; at[1].val.cooperative = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    e = cudaLaunchKernelExC(&cfg, (const void*)tc_kernel<768, 49, true>, args);
  } else {
    const void* fn = (C == 768) ? (const void*)tc_kernel<768, 49, false> : (const void*)tc_kernel<384, 196, false>;
    e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(TC_NT), args, (size_t)TC_SMEM, (cudaStream_t)stream_v);
  }
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

// small-C rows: plane group PG (template: 16 at C=192, 4 at C=96/160), row strips NS, GEMM1 token chunk NC1,
// GEMM2 token chunk NC2, GEMM2 K split KS
struct SCRow { int B, C, H, W, PG, NS, NC1, NC2, KS; };
static const SCRow kSCRows[7] = {
    {1, 192, 28, 28, 16, 10, 56, 56, 4}, {4, 192, 28, 28, 16, 3, 56, 112, 2}, {8, 192, 28, 28, 16, 3, 112, 112, 1},
    {2, 96, 56, 56, 4, 3, 64, 112, 2},   {1, 96, 56, 56, 4, 6, 64, 64, 3},    {1, 160, 449, 7, 4, 3, 112, 112, 2},
    {4, 96, 56, 56, 4, 3, 64, 112, 1},
};
struct SCPlan { SCRow r; int HW, HWS, Mcols, C4, G, SR, SLOT, units, PSUB, unitf, N1, NAT1, STG1, CPI1, T1, N2, NAT2, CPI2, KB2, U2;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_part, o_uk, o_vk, o_wgb, o_gsq, bytes; };

static bool sc_plan(int B, int C, int H, int W, int nsm, SCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 7; i++)
    if (kSCRows[i].B == B && kSCRows[i].C == C && kSCRows[i].H == H && kSCRows[i].W == W) idx = i;
  if (idx < 0 || nsm <= 0) return false;
  const SCRow r = kSCRows[idx];
  pl.r = r;
  if (!((C == 192 && W == 28 && r.PG == 16) || (C == 96 && W == 56 && r.PG == 4) || (C == 160 && W == 7 && r.PG == 4))) return false;
  if ((8 * C + (SC_NT / 32) * nsm - 1) / ((SC_NT / 32) * nsm) > SC_MAXWU) return false;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / r.PG;
  pl.SR = (H + r.NS - 1) / r.NS;
  if ((r.NS - 1) * pl.SR >= H) return false;
  pl.SLOT = ((pl.SR + 6) * W + 8 + 3) & ~3;
  pl.units = B * pl.G * r.NS;
  {
    // plane subgroups per conv unit so that the CTA's threads are used
    const int numax = (pl.units + nsm - 1) / nsm, pos = pl.SR * (W / 7);
    int target = SC_NT / (numax * pos); if (target < 1) target = 1;
    pl.PSUB = 1;
    for (int d = 1; d <= r.PG; d++) if (r.PG % d == 0 && d <= target) pl.PSUB = d;
  }
  pl.unitf = r.PG * pl.SLOT + r.PG * 49 + pl.PSUB * pl.SR * W * 2;
  pl.N1 = (r.NC1 + 15) / 16 * 16; pl.NAT1 = (pl.N1 + 63) / 64; pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.CPI1 = (pl.HW + r.NC1 - 1) / r.NC1; pl.T1 = (pl.C4 / 128) * B * pl.CPI1;
  pl.N2 = (r.NC2 + 15) / 16 * 16; pl.NAT2 = (pl.N2 + 63) / 64;
  pl.CPI2 = (pl.HW + r.NC2 - 1) / r.NC2;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4 || pl.KB2 * 64 > 768 || pl.NAT2 > 2 || pl.N1 > 128 || pl.N2 > 256) return false;
  if ((r.NC1 % 8) || (r.NC2 % 8) || 4 * pl.STG1 > SC_RING) return false;
  if ((size_t)((pl.units + nsm - 1) / nsm) * pl.unitf * 4 > (size_t)SC_RING) return false;
  pl.U2 = ((C + 127) / 128) * B * pl.CPI2 * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsq = o; o = tc_al(o + (size_t)B * pl.C4 * 4);
  pl.bytes = o;
  return true;
}

static int sm_count() {
  static int nsm = 0;
  if (nsm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
  return nsm;
}

extern "C" long long ws_workspace_bytes(int B, int C, int H, int W) {
  TCPlan pl;
  if (tc_plan(B, C, H, W, pl)) return (long long)pl.bytes;
  SCPlan sp;
  if (sc_plan(B, C, H, W, sm_count(), sp)) return (long long)sp.bytes;
  return 0;
}

static const char* sc_launch(const WSArgs* a, int nsm, void* stream_v) {
  SCPlan pl;
  if (!sc_plan(a->B, a->C, a->H, a->W, nsm, pl)) return "ws_launch: unsupported shape";
  SCP p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsq = (float*)(wsb + pl.o_gsq);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DSC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.H = a->H; p.HW = pl.HW; p.HWS = pl.HWS; p.Mcols = pl.Mcols; p.NS = pl.r.NS; p.SR = pl.SR;
  p.SLOT = pl.SLOT; p.units = pl.units; p.PSUB = pl.PSUB;
  p.NC1 = pl.r.NC1; p.N1 = pl.N1; p.NAT1 = pl.NAT1; p.CPI1 = pl.CPI1; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.NC2 = pl.r.NC2; p.N2 = pl.N2; p.NAT2 = pl.NAT2; p.CPI2 = pl.CPI2; p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2;
  p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 192) ? (const void*)sc_kernel<192, 28, 16> : (C == 96) ? (const void*)sc_kernel<96, 56, 4>
                                                                                 : (const void*)sc_kernel<160, 7, 4>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(SC_NT), args, (size_t)SC_SMEM, (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

extern "C" const char* ws_launch(const WSArgs* a, void* stream_v) {
  static bool attr_done = false;
  const int nsm = sm_count();
  if (!attr_done) {
    // one maximal value for every instantiation (never shrunk later; the portal plans all shapes first)
    cudaError_t e;
    e = cudaFuncSetAttribute(tc_kernel<768, 49, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc_kernel<768, 49, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc_kernel<384, 196, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<192, 28, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<96, 56, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<160, 7, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_done = true;
  }
  TCPlan pl;
  if (tc_plan(a->B, a->C, a->H, a->W, pl)) return tc_launch(a, nsm, stream_v);
  return sc_launch(a, nsm, stream_v);
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
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};
extern "C" const char* convnext_launch(Params p, void* stream);

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};
extern "C" long long ws_workspace_bytes(int B, int C, int H, int W);
extern "C" const char* ws_launch(const WSArgs* a, void* stream);

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
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();

  const long long wsb = ws_workspace_bytes(B, C, H, W);
  if (wsb > 0) {
    torch::Tensor ws = torch::empty({(long)wsb}, x.options().dtype(torch::kUInt8));
    WSArgs a;
    a.x = x.data_ptr<float>(); a.dww = dwconv_weight.data_ptr<float>(); a.dwb = dwconv_bias.data_ptr<float>();
    a.lnw = layernorm_weight.data_ptr<float>(); a.lnb = layernorm_bias.data_ptr<float>();
    a.w1 = pwconv1_weight.data_ptr<float>(); a.b1 = pwconv1_bias.data_ptr<float>();
    a.gw = grn_weight.data_ptr<float>(); a.gb = grn_bias.data_ptr<float>();
    a.w2 = pwconv2_weight.data_ptr<float>(); a.b2 = pwconv2_bias.data_ptr<float>();
    a.out = output.data_ptr<float>(); a.ws = ws.data_ptr(); a.stamps = nullptr;
    a.B = B; a.C = C; a.H = H; a.W = W; a.eps = (float)eps; a.ln_eps = (float)layer_norm_eps;
    const char* err = ws_launch(&a, (void*)stream);
    if (err != nullptr) throw std::runtime_error(std::string("ws_launch failed: ") + err);
    return;
  }

  const int HW = H * W, M = B * HW, C4 = 4 * C, nch = C / 32;
  const int Mp = ((M + 127) / 128) * 128;
  const size_t off_xn = 0;
  const size_t off_part = align256(off_xn + (size_t)C * Mp * 4);
  const size_t off_hidden = align256(off_part + (size_t)nch * Mp * 2 * 4);
  const size_t off_sumsq = align256(off_hidden + (size_t)C4 * Mp * 4);
  const size_t off_w1t = align256(off_sumsq + (size_t)B * C4 * 4);
  const size_t off_w2t = align256(off_w1t + (size_t)C * C4 * 4);
  const size_t total = align256(off_w2t + (size_t)C4 * C * 4);
  torch::Tensor ws = torch::empty({(long)total}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();

  Params p;
  p.x = x.data_ptr<float>(); p.dww = dwconv_weight.data_ptr<float>(); p.dwb = dwconv_bias.data_ptr<float>();
  p.lnw = layernorm_weight.data_ptr<float>(); p.lnb = layernorm_bias.data_ptr<float>();
  p.w1 = pwconv1_weight.data_ptr<float>(); p.b1 = pwconv1_bias.data_ptr<float>();
  p.gw = grn_weight.data_ptr<float>(); p.gb = grn_bias.data_ptr<float>();
  p.w2 = pwconv2_weight.data_ptr<float>(); p.b2 = pwconv2_bias.data_ptr<float>();
  p.out = output.data_ptr<float>();
  p.xn = (float*)(wp + off_xn); p.part = (float*)(wp + off_part); p.hidden = (float*)(wp + off_hidden);
  p.sumsq = (float*)(wp + off_sumsq); p.w1t = (float*)(wp + off_w1t); p.w2t = (float*)(wp + off_w2t);
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = HW; p.M = M; p.Mp = Mp; p.C4 = C4; p.nch = nch;
  p.eps = (float)eps; p.ln_eps = (float)layer_norm_eps;
  p.p1_mode = 0; p.p1_twr = 1; p.p1_tiles = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_hb = 0; p.p1_wb = 0;
  p.cfg2 = 0; p.cfg3 = 0; p.split3 = 1;
  const char* err = convnext_launch(p, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (tcgen05 swap-AB paths + fp32 anchor)"); }
```

```yaml design-card
id: r9-d-e2-pair-smemg
parents:
  - r8-d-e1-allwarp-epi
operation: structural_mutation
language: cuda_cpp
niche:
  mem: tma-tensor
  st: direct
  grid: persistent
  launch: fused
  tile: token
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  Lineage D / E2 (H33). For C=768 with B >= 16, tc_kernel runs as 2-CTA clusters (cooperative + cluster launch).
  Every tcgen05 op in that instantiation is cta_group::2, because ptxas rejects mixing cta_group::1 and ::2 in one
  kernel. GEMM1 pairs two 128-row hidden tiles over the same tokens: each CTA lands half of the d tile through
  cta_group::2 TMA onto the leader's barrier. The GEMM1 epilogue stores h only, plus gw*gf per (image, k). GEMM2 pairs
  M = 256 output channels: each CTA lands its own 128 W2 rows and 64 of the 128 tokens in 32 KB stages through a
  6-deep ring. Four producer warps build g = fp16(gw*gf*h) in shared memory from the landed h half, one warp per stage,
  and signal the leader's ready barrier with a cluster-release arrive. Measured on the rented card: 32,768 is 2-3 us
  faster than r8 and 16,768 is within noise. On 2/8,768 the pair skeleton was 1.4-2.1 us slower, so those rows and
  the 14x14 rows keep r8's 1SM code unchanged.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: medium
  L:
    pct: -1
    confidence: medium
resources_sm100a:
  regs_per_thread: 104
  smem_per_cta_bytes: 227616
  threads_per_cta: 512
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 196608
  launches_per_call: 1
knobs:
  pair_rows: "C=768 and B >= 16 (16,768 and 32,768)"
  pair_cluster: "2x1, cooperative + cluster launch, grid = 2 x cudaOccupancyMaxActiveClusters (74 -> 148)"
  gemm2_pair_stage: "32 KB per CTA (W2 16 KB + h half 8 KB + g half 8 KB), 6 stages"
  g_producers: "warps 12-15, warp (it % 4) owns stage it; gw*gf slice staged in smem per unit (<= 8 KB)"
  ready_barrier_count: "2 (one local + one remote producer warp)"
  KS: "unchanged: 2,768 12 / 8,768 6 / 16,768 3 / 32,768 3"
  TC_STAMPS: 0
  SC_STAMPS: 0
dispatch:
  - shapes: "16,768,7,7 32,768,7,7"
    kernel: "tc_kernel<768,49,true> (2-CTA clusters, all cta_group::2, smem g)"
    meta: {grid: "148 (74 clusters), cooperative + cluster"}
  - shapes: "2,768,7,7 8,768,7,7 8,384,14,14 16,384,14,14"
    kernel: "tc_kernel<768,49,false> / tc_kernel<384,196,false> (r8 code path, stored g, 1SM)"
    meta: {grid: "num SMs, cooperative"}
  - shapes: "1,192,28,28 4,192,28,28 8,192,28,28 1,96,56,56 2,96,56,56 4,96,56,56 1,160,449,7"
    kernel: "sc_kernel (unchanged from r8)"
    meta: {grid: "num SMs, cooperative"}
  - shapes: "8,320,32,32 32,128,64,64 64,96,112,112"
    kernel: "convnext_fused (r0 anchor, merge-only)"
    meta: {}
tests:
  - H33
findings:
  - "ptxas rejects mixing cta_group::1 and cta_group::2 tcgen05 ops in one kernel ('uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and that is not allowed'). A pair GEMM2 forces the whole kernel (TMEM alloc/relinquish/dealloc, GEMM1 MMAs, commits) to cta_group::2. An NVRTC resources() probe did not catch this; the full nvcc build did [compile_b200]"
  - "Cooperative launch combined with the 2x1 cluster attribute (cudaLaunchKernelExC) works. cudaOccupancyMaxActiveClusters = 74 at 512 thr / 227 KB, so the grid stays 148, and grid.sync costs 1.35 us with or without clusters [probe_b200]"
  - "cta_group::2 syntax that compiles to UTMALDG.2D.2CTA / UTCHMMA.2CTA / UTCBAR.2CTA.MULTICAST: 'cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [dst],[map,{x,y}],[leader_bar]' with leader_bar = mapa(bar,0); 'tcgen05.alloc/relinquish/dealloc.cta_group::2' issued by one warp in BOTH CTAs; idesc M field 256>>4 = 16; commit '.cta_group::2...multicast::cluster' with mask 3 [probe_b200]"
  - "GEMM1 as a pair (two consecutive 128-row hidden tiles, B split N/2 per CTA, N/2 = 32/56/88 tokens, leader expect_tx of both CTAs' bytes, te1 count 2 via remote arrive) was correct first try on all 4 C=768 rows [run_tests]"
  - "v1 (all C=768 rows paired, 4 producer warps all working on every stage, gw*gf loaded from L2 per stage): 16/16 pass but slower than r8 in the same container: 2,768 29.7 vs 26.4, 8,768 38.4 vs 32.8, 16,768 52.5 vs 43.8, 32,768 79.9 vs 64.1 us. The gw*gf L2 load sat in series in front of every stage [run_tests]"
  - "v2 (gw*gf for the unit's K slice and <= 2 images staged in smem in one unrolled load round): 29.2 / 36.5 / 47.9 / 73.3 us. Still slower than r8 [run_tests]"
  - "v2 stamps (probe, flush, globaltimer): GEMM2 per 64-k pair stage 683 / 658 / 768 / 875 ns at 2/8/16/32,768, against a two-MMA pair ideal of ~270 ns (2.5-3.2x). The MMA warp waits 331-644 ns per stage on the ready barrier, while the producers wait only 0.3-1.0 us in total on TMA landing. The producers gate: with all 4 warps on every stage, the fence.proxy.async plus the cluster-release arrive of each stage run in series [probe_b200]"
  - "GEMM1 post-accumulator per tile without the g pass (v2 stamps): 2.05 / 3.36 / 4.77 / 5.17 us at 2/8/16/32,768, against r8's 3.04 / 4.77 / 7.07 / 7.92 (-30..-35 %, short of the -40 % bar). The epilogue's accumulator wait per tile is 2.5-4.2 us in pair mode [probe_b200]"
  - "v3 (one producer warp owns a whole stage, stages round-robin over 4 warps, ready count 2): 2,768 28.5 vs 26.4, 8,768 34.2 vs 32.8, 16,768 42.7 vs 43.8, 32,768 61.1 vs 64.1 us. Pipelining the per-stage signalling is the fix; the pair only pays where a CTA runs many stages (32 at 32,768, 16 at 16,768) [run_tests]"
  - "v4 (shipped; pairs only for C=768, B >= 16): 16/16 PASS incl. portal-order re-runs. Same container vs r8-merge: 32,768 62.1 vs 64.1 (-3.1 %), 16,768 43.5 vs 43.8 (-0.5 %, -2.4 % in v3: within noise), 2,768 26.4 vs 26.4 and 8,768 33.6 vs 32.8 (r8 code, so the 0.8 us is noise). sc rows and 14x14 rows are unchanged code [run_tests]"
  - "H33 verdict: success line NOT met. Rows 32,768 61-62 vs <= 57 and 16,768 42.7-43.5 vs <= 39; v2 stage clk 2.5-3.2x ideal (v3 stages not stamped); GEMM1 post -30..-35 % vs -40 %. The refutation clause fires on 4 of 6 tc rows: 2/8,768 got slower with pairs, and the 14x14 rows were not paired (3 channel tiles cannot form full pairs). The real gain is at 32,768 (-2..-3 us rented), possibly 16,768. The pair's fixed costs (cluster coupling of GEMM1, first-ready chain of 2.3-2.8 us in phase C) dominate at B <= 8 [run_tests, probe_b200]"
  - "NOT DONE: the fallback single-accumulator form (h scaled in place by s_b); C=384 pairs (one full + one half-masked pair tile); interleaved x3 timing (each build was timed once in run_tests against r8-merge in the same container); stamps of v3/v4 stages; a fresh 10-round fp64 precision log [INFERRED]"
  - "PRECISION: roundings are unchanged from r8's tc path (d16, fp16(lnw*W1), fp16(W2), h16 once, g16 = fp16(fp32(gw*gf) * h16); the GRN sumsq order and the KS splits are unchanged). The only difference is that the GEMM accumulation is split over the CTA pair inside the tensor core. So this card cites r8's 10-round fp64 log (tc rows 59-65 % of atol, 0 beyond) plus a single-seed check of the pair path against the fp32 torch reference with harness generators (seed 200): max |err| 1.48e-3 (2,768, pair forced), 1.57e-3 (8,768), 1.68e-3 (16,768), 1.70e-3 (32,768), all below 2.9e-3 x 0.6. run_tests 16/16 x 10 rounds pass [probe_b200, run_tests]"
  - "Next levers in D: (1) 32,768/16,768: stamp v4 stages to see whether GEMM2 is now near the 270 ns pair ideal; if not, the first-ready chain (2.3-2.8 us) and the drain/red.add (3.7-8 us) are the post-sync terms. (2) The GEMM1 accumulator wait of 2.5-4.2 us per tile in pair mode suggests the pair GEMM1 mainloop is coupled to the slower CTA; a 1SM GEMM1 is impossible in the same kernel (ptxas rule), so splitting the C=768 kernel into two launches would cost more than it saves. (3) 14x14 rows: pairs need a masked half tile or channel-pair remapping [INFERRED]"
paths:
  - {max_tokens: 392, lang: cuda, width: 128, threads: 512, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "2,768 and 8,768: tc_kernel<768,49,false>, identical to r8"}
  - {max_tokens: 1568, lang: cuda, width: 128, threads: 512, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 2, note: "16,768 and 32,768: tc_kernel<768,49,true>, cooperative + 2x1 cluster launch, all tcgen05 cta_group::2, smem g producers one warp per stage"}
  - {max_tokens: 3136, lang: cuda, width: 128, threads: 512, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "8,384 and 16,384: tc_kernel<384,196,false>, identical to r8"}
  - {max_tokens: 12544, lang: cuda, width: 128, threads: 384, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "sc_kernel rows, identical to r8"}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1, note: "r0 fp32 anchor on 8,320 / 32,128 / 64,96 (merge-only)"}
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05/TMEM and cta_group::2 are sm_100a-only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "Clustered cooperative grid on the portal: tricks.md reports large clustered grids costing 2-5 % on the portal; the gain at 32,768 (~3 %) is of that size, so verify on the portal before trusting it"
  - "The pair path depends on cudaOccupancyMaxActiveClusters returning 74; if fewer clusters fit, the grid shrinks (even) and the work loops still cover all units, but timing changes"
  - "Remote mbarrier arrives (release.cluster) and the final cluster barrier must stay paired; any early exit of one CTA in a pair would hang the other"
  - "build must target B200 only (sm_100a)"
  - "16,768 gain is within noise; merge only 32,768 unless a portal run confirms 16,768"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```