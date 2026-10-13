### Rationale
The stamps showed that the r4 kernel's roughly 2× excess was not operand ingest. Its GEMM1 epilogue held `hreg[4][16]` in local memory (128 B stack, 35 LDL/STL per instance), and with 227 KB of smem the L1 was too small to hold it. That cost 45–49 µs per tile.

The fix:
- Rewrote the epilogue around registers with no spilled arrays.
- h and g are now stored through a per-warp smem transpose.
- Phase A weight conversion is split into equal C-float units, and the low-numbered CTAs, which carry the extra conv units, get the fewest of them.
- The GEMM2 epilogue at 14×14 now uses float4 `red.add`.

`run_tests` passes 16/16. On the six wide-C rows it beats `r5-merge-b` in the same container by 4–15 µs, and by at least 8 µs on 8,768, 16,768 and 8,384, which meets the success line.

The ingest probe supports H33: a cta_group::2 pair runs at 1.01× the MMA ideal at N = 224/256, while B multicast gives nothing, so the cap is on the SM side. I did not port the pair into the kernel this session. The precision log shows 0 elements beyond atol on all six rows. The other ten rows still run the r0 anchor.

```json solution-spec
{"name": "r6-d-e2-epi-fix", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "description": "D swap-AB tcgen05 cooperative kernel with the GEMM1-epilogue local-memory spill removed (staged coalesced h/g stores), balanced phase-A weight units and float4 red.add at 14x14", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN.
// Wide-C small-image rows (2/8/16/32,768,7,7 and 8/16,384,14,14): cooperative kernel tc_kernel with a tcgen05
//   kind::f16 swap-AB engine (A = fp16 weight tile, 128 rows, K-major; B = fp16 activations [channel][token],
//   MN-major; TMA 128B-swizzled 4-stage ring; whole-warp TMA producer and MMA warp; double-buffered TMEM accumulators).
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
// tcgen05 swap-AB cooperative path for the wide-C small-image rows
// =====================================================================================================
// TC_STAMPS: globaltimer diagnostics (off in the shipped build; the timed path writes no stamp buffer).
// TC_EPI_STAGED: GEMM1 epilogue stores h/g through a per-warp smem transpose (4 lanes per 64-B row segment).
#ifndef TC_STAMPS
#define TC_STAMPS 0
#endif
#ifndef TC_EPI_STAGED
#define TC_EPI_STAGED 1
#endif
#define TC_NT 384
#define TC_RING (192 * 1024)
#define TC_STGB 2560
#define TC_OFF_STG TC_RING
#define TC_OFF_MS (TC_OFF_STG + 8 * TC_STGB)
#define TC_OFF_SQ (TC_OFF_MS + 4096)
#define TC_OFF_INV (TC_OFF_SQ + 4096)
#define TC_OFF_CB (TC_OFF_INV + 512)
#define TC_OFF_BAR (TC_OFF_CB + 512)
#define TC_SMEM (TC_OFF_BAR + 27 * 8 + 1024)
#define TC_STG2 49152
struct TCP {
  CUtensorMap mW1, mW2, mD, mH, mG;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid, *gq;
  float *part, *uk, *vk, *wgb, *gsum;
  unsigned long long* st;
  int B, Mcols, units, NI, T1, STG1, KS, KB2, U2;
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
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int x, int y, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(sa(dst)), "l"((uint64_t)m), "r"(x), "r"(y), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ bool elect1() { uint32_t r; asm volatile("{\n.reg .pred p;\n.reg .b32 t;\nelect.sync t|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(r)); return r != 0; }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFFu) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* b) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void tfb() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tfa() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tld32(uint32_t a, uint32_t (&v)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]),
        "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
        "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]) : "r"(a));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void ebar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

template <int C_, int HW_>
__global__ void __launch_bounds__(TC_NT, 1) tc_kernel(const __grid_constant__ TCP p) {
  constexpr int W_ = (HW_ == 49) ? 7 : 14, H_ = W_;
  constexpr int C4 = 4 * C_, G = C_ / 32, HT = C4 / 128, CT = C_ / 128, KB1 = C_ / 64;
  constexpr int HWS = (HW_ + 7) / 8 * 8;
  extern __shared__ __align__(1024) unsigned char tc_raw[];
  unsigned char* sm = tc_raw + ((1024u - (sa(tc_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  uint32_t* stg_all = reinterpret_cast<uint32_t*>(sm + TC_OFF_STG);
  float4* ms_s = reinterpret_cast<float4*>(sm + TC_OFF_MS);
  float* sq_s = reinterpret_cast<float*>(sm + TC_OFF_SQ);
  float* inv_s = reinterpret_cast<float*>(sm + TC_OFF_INV);
  int* cb_s = reinterpret_cast<int*>(sm + TC_OFF_CB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + TC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *empty2 = bars + 16, *tf2 = bars + 20, *te2 = bars + 22, *cbar = bars + 24;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 26);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  cg::grid_group grid = cg::this_grid();
  if (tid == 0) {
    for (int i = 0; i < 25; i++) mb_init(bars + i, (i == 10 || i == 11 || i == 22 || i == 23) ? 8u : 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tfb(); __syncthreads(); tfa();
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

  // ---------------- phase B: GEMM1 (hidden = W1h x d16), LN fold, GELU, fp16 hidden h and g = gw*gf*h
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    uint32_t it = 0;
    for (int t = cta; t < p.T1; t += P) {
      const int ht = t % HT, tt = t / HT, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15, nat = (N + 63) >> 6;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = it & 3; const uint32_t n = it >> 2;
        if (it >= 4) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          mb_expect(full1 + s, 16384 + nat * 8192);
          tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
          for (int a = 0; a < nat; a++) tma2d(st + 16384 + a * 8192, &p.mD, tok0 + a * 64, kb * 64, full1 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, ti = 0;
    for (int t = cta; t < p.T1; t += P, ti++) {
      const int tt = t / HT, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), N = (nimg * HWS + 15) & ~15;
      const uint32_t buf = ti & 1;
      if (ti >= 2) { mb_wait(te1 + buf, ((ti >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(N >> 3) << 17) | (8u << 24);
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
    uint32_t ti = 0;
#if TC_STAMPS
    unsigned long long a_pre = 0, a_wait = 0, a_post = 0;
#endif
    for (int t = cta; t < p.T1; t += P, ti++) {
#if TC_STAMPS
      const unsigned long long q0 = gtime();
#endif
      const int ht = t % HT, tt = t / HT, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15, vcols = nimg * HWS;
      const int k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k), gwk = __ldg(p.gw + k);
      ebar();
      {
        // per-token LayerNorm statistics from the conv partials; one token (column) per thread
        const int i = et / HWS, hw = et - i * HWS;
        float4 m = make_float4(0.f, 0.f, -1.f, 0.f);
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
          m = make_float4(mu, rsqrtf(var + p.ln_eps), (float)i, 0.f);
        }
        ms_s[et] = m;
      }
      ebar();
#if TC_STAMPS
      const unsigned long long q1 = gtime();
#endif
      const uint32_t buf = ti & 1;
      mb_wait(tf1 + buf, (ti >> 1) & 1);
      tfa();
#if TC_STAMPS
      const unsigned long long q2 = gtime();
#endif
      // pass 1: each lane owns hidden row k; 32-column TMEM chunks alternate between the two warp groups.
      // LN fold, exact GELU, fp16 h, per-image sumsq in registers (no per-thread arrays: the r4 hreg[4][16] spilled).
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16) + buf * 256;
      const int nch = (N + 31) >> 5;
#if TC_EPI_STAGED
      uint32_t* stg = stg_all + ew * (TC_STGB / 4);
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      __half* gbase = p.gq + (size_t)(m0 + q * 32) * p.Mcols + tok0;
#else
      __half* hrow = p.hid + (size_t)k * p.Mcols + tok0;
      __half* grow = p.gq + (size_t)k * p.Mcols + tok0;
#endif
      float sq0 = 0.f, sq1 = 0.f, sq2 = 0.f;
#pragma unroll 1
      for (int ch = grp; ch < nch; ch += 2) {
        uint32_t v[32];
        tld32(taddr + ch * 32, v);
        uint32_t hw2[16];
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
          float hf[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const float4 m = ms_s[ch * 32 + j + e];
            const float z = fmaf(m.y, __uint_as_float(v[j + e]) - m.x * uk, vk);
            const float h = __half2float(__float2half_rn(gelu_f(z)));
            hf[e] = h;
            const float h2 = h * h;
            sq0 += (m.z == 0.f) ? h2 : 0.f;
            sq1 += (m.z == 1.f) ? h2 : 0.f;
            sq2 += (m.z == 2.f) ? h2 : 0.f;
          }
          hw2[j >> 1] = h2u(__floats2half2_rn(hf[0], hf[1]));
        }
#if TC_EPI_STAGED
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
#else
        uint4* dst = reinterpret_cast<uint4*>(hrow + ch * 32);
#pragma unroll
        for (int qd = 0; qd < 4; qd++)
          if (ch * 32 + qd * 8 < vcols) dst[qd] = make_uint4(hw2[4 * qd], hw2[4 * qd + 1], hw2[4 * qd + 2], hw2[4 * qd + 3]);
#endif
      }
      tfb(); __syncwarp();
      if (lane == 0) mb_arrive1(te1 + buf);
      { float* sqw = sq_s + (grp * 128 + r) * 4; sqw[0] = sq0; sqw[1] = sq1; sqw[2] = sq2; }
      ebar();
      float fsc[3];
      {
        const float* sa_ = sq_s + r * 4; const float* sb_ = sq_s + (128 + r) * 4;
#pragma unroll
        for (int i = 0; i < 3; i++) {
          const float gfi = sqrtf(sa_[i] + sb_[i]);
          if (grp == 0 && i < nimg) { const float s = warp_sum(gfi); if (lane == 0) atomicAdd(p.gsum + img0 + i, s); }
          fsc[i] = gwk * gfi;
        }
      }
      // pass 2: each thread re-reads the fp16 h it stored (L2-hot) and writes g = fp16(gw*gf*h); 8 columns never
      // straddle an image (HWS % 8 == 0)
#if TC_EPI_STAGED
      float fr[4][3];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++)
#pragma unroll
        for (int i = 0; i < 3; i++) fr[i2][i] = __shfl_sync(0xffffffffu, fsc[i], i2 * 8 + (lane >> 2));
#pragma unroll 1
      for (int cb = grp; cb < nch; cb += 4) {
        uint4 hv[2][4];
#pragma unroll
        for (int u2 = 0; u2 < 2; u2++)
#pragma unroll
          for (int i2 = 0; i2 < 4; i2++) {
            const int row = i2 * 8 + (lane >> 2), c = (cb + 2 * u2) * 32 + (lane & 3) * 8;
            hv[u2][i2] = (cb + 2 * u2 < nch && c < vcols) ? __ldcg(reinterpret_cast<const uint4*>(hbase + (size_t)row * p.Mcols + c)) : make_uint4(0u, 0u, 0u, 0u);
          }
#pragma unroll
        for (int u2 = 0; u2 < 2; u2++)
#pragma unroll
          for (int i2 = 0; i2 < 4; i2++) {
            const int row = i2 * 8 + (lane >> 2), c = (cb + 2 * u2) * 32 + (lane & 3) * 8;
            if (cb + 2 * u2 < nch && c < vcols) {
              const int im = c / HWS;
              const float f = (im == 0) ? fr[i2][0] : (im == 1) ? fr[i2][1] : fr[i2][2];
              const uint32_t wi[4] = {hv[u2][i2].x, hv[u2][i2].y, hv[u2][i2].z, hv[u2][i2].w};
              uint32_t wo[4];
#pragma unroll
              for (int w4 = 0; w4 < 4; w4++) {
                const float2 x2 = __half22float2(*reinterpret_cast<const __half2*>(&wi[w4]));
                wo[w4] = h2u(__floats2half2_rn(f * x2.x, f * x2.y));
              }
              *reinterpret_cast<uint4*>(gbase + (size_t)row * p.Mcols + c) = make_uint4(wo[0], wo[1], wo[2], wo[3]);
            }
          }
      }
#else
#pragma unroll 1
      for (int cb = grp; cb < nch; cb += 4) {
        uint4 hv[2][4];
#pragma unroll
        for (int u2 = 0; u2 < 2; u2++)
#pragma unroll
          for (int qd = 0; qd < 4; qd++) {
            const int c = (cb + 2 * u2) * 32 + qd * 8;
            hv[u2][qd] = (cb + 2 * u2 < nch && c < vcols) ? __ldcg(reinterpret_cast<const uint4*>(hrow + c)) : make_uint4(0u, 0u, 0u, 0u);
          }
#pragma unroll
        for (int u2 = 0; u2 < 2; u2++)
#pragma unroll
          for (int qd = 0; qd < 4; qd++) {
            const int c = (cb + 2 * u2) * 32 + qd * 8;
            if (cb + 2 * u2 < nch && c < vcols) {
              const int im = c / HWS;
              const float f = (im == 0) ? fsc[0] : (im == 1) ? fsc[1] : fsc[2];
              const uint32_t wi[4] = {hv[u2][qd].x, hv[u2][qd].y, hv[u2][qd].z, hv[u2][qd].w};
              uint32_t wo[4];
#pragma unroll
              for (int w4 = 0; w4 < 4; w4++) {
                const float2 x2 = __half22float2(*reinterpret_cast<const __half2*>(&wi[w4]));
                wo[w4] = h2u(__floats2half2_rn(f * x2.x, f * x2.y));
              }
              *reinterpret_cast<uint4*>(grow + c) = make_uint4(wo[0], wo[1], wo[2], wo[3]);
            }
          }
      }
#endif
#if TC_STAMPS
      const unsigned long long q3 = gtime(); a_pre += q1 - q0; a_wait += q2 - q1; a_post += q3 - q2;
#endif
    }
#if TC_STAMPS
    if (warp == 4 && lane == 0) { STAMP(4); p.st[cta * 24 + 11] = a_pre; p.st[cta * 24 + 12] = a_wait; p.st[cta * 24 + 13] = a_post; }
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  tfb(); grid.sync(); tfa();
  if (tid == 0) STAMP(7);

  // ---------------- phase C: GEMM2 split-K, out += W2h x h + inv_b * (W2h x g) (+ W2.gb on the first k-split)
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
  } else if (warp >= 4) {
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
  tfb(); __syncthreads();
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
}

// ---------------------------------------------------------------- host side of the tcgen05 path
struct TCRow { int B, C, H, W, NI, KS; };
static const TCRow kTCRows[6] = {
    {2, 768, 7, 7, 1, 12}, {8, 768, 7, 7, 2, 6}, {16, 768, 7, 7, 3, 3},
    {32, 768, 7, 7, 3, 3}, {8, 384, 14, 14, 1, 3}, {16, 384, 14, 14, 1, 3},
};

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};

static inline size_t tc_al(size_t v) { return (v + 255) / 256 * 256; }

struct TCPlan { TCRow r; int HW, HWS, Mcols, C4, G, T1, NAT1, STG1, U2, KB2;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_gq, o_part, o_uk, o_vk, o_wgb, o_gsum, bytes; };

static bool tc_plan(int B, int C, int H, int W, TCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 6; i++)
    if (kTCRows[i].B == B && kTCRows[i].C == C && kTCRows[i].H == H && kTCRows[i].W == W) idx = i;
  if (idx < 0) return false;
  const TCRow r = kTCRows[idx];
  pl.r = r;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / 32;
  pl.T1 = (pl.C4 / 128) * ((B + r.NI - 1) / r.NI);
  pl.NAT1 = (((r.NI * pl.HWS + 15) & ~15) + 63) / 64;
  if (pl.NAT1 > 4 || r.NI > 3) return false;
  pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4) return false;
  pl.U2 = (C / 128) * ((pl.Mcols + 127) / 128) * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_gq = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsum = o; o = tc_al(o + (size_t)B * 4);
  pl.bytes = o;
  return true;
}

extern "C" long long ws_workspace_bytes(int B, int C, int H, int W) {
  TCPlan pl;
  if (!tc_plan(B, C, H, W, pl)) return 0;
  return (long long)pl.bytes;
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

extern "C" const char* ws_launch(const WSArgs* a, void* stream_v) {
  static bool attr_done = false;
  static int nsm = 0;
  if (!attr_done) {
    // one maximal value for every instantiation (never shrunk later; the portal plans all shapes first)
    cudaError_t e;
    e = cudaFuncSetAttribute(tc_kernel<768, 49>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc_kernel<384, 196>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    attr_done = true;
  }
  TCPlan pl;
  if (!tc_plan(a->B, a->C, a->H, a->W, pl)) return "ws_launch: unsupported shape";
  TCP p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid); p.gq = (__half*)(wsb + pl.o_gq);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsum = (float*)(wsb + pl.o_gsum);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DTC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  if (!enc16(&p.mG, p.gq, pl.Mcols, C4, 64, 64)) return "ws_launch: map G";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.Mcols = pl.Mcols; p.units = a->B * pl.G; p.NI = pl.r.NI; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2; p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 768) ? (const void*)tc_kernel<768, 49> : (const void*)tc_kernel<384, 196>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(TC_NT), args, (size_t)TC_SMEM, (cudaStream_t)stream_v);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (wide-C tcgen05 swap-AB path + fp32 anchor)"); }
```

```yaml design-card
id: r6-d-e2-epi-fix
parents: [r4-d-tc05-swapab]
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
  Lineage D / E2 (H33). Stamping showed that the r4 tcgen05 kernel's roughly 2x excess on 32,768 and the 14x14 rows
  was the GEMM1 epilogue (45-49 us per tile), not operand ingest. Its hreg[4][16] array lived in local memory:
  128 B stack and 35-36 LDL/STL per instance, with L1 squeezed by 227 KB of smem. Changes:
  (1) a register-only epilogue (no per-thread arrays, 112-118 regs, 0 stack);
  (2) h stored through a per-warp smem transpose, so 4 lanes write each 64-B row segment, and g = fp16(gw*gf*h) made
  by each thread re-reading its own L2-hot h;
  (3) phase-A weight work split into 8C equal C-float units (W1 rows plus W2 quarter rows with partial W2.gb), handed
  out from the highest CTA down;
  (4) a float4 red.add GEMM2 epilogue at 14x14.
  The engine stays 1SM. The cta_group::2 probe reaches the MMA ideal, but the port is deferred.
expected_effect:
  S: {pct: 0, confidence: high}
  M: {pct: -30, confidence: medium}
  L: {pct: -10, confidence: medium}
resources_sm100a:
  regs_per_thread: 118
  smem_per_cta_bytes: 227544
  threads_per_cta: 384
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 196608
  launches_per_call: 1
knobs: {ring_stages: 4, stage_k: 64, threads: 384, epilogue_warps: 8, tmem_cols: 512, TC_EPI_STAGED: 1, TC_STAMPS: 0, rows: "2,768 NI1 KS12; 8,768 NI2 KS6; 16,768 NI3 KS3; 32,768 NI3 KS3; 8,384 NI1 KS3; 16,384 NI1 KS3"}
dispatch:
  - {shapes: "2,768,7,7 8,768,7,7 16,768,7,7 32,768,7,7 8,384,14,14 16,384,14,14", kernel: "tc_kernel<768,49> / tc_kernel<384,196>", meta: {grid: "num SMs, cooperative"}}
  - {shapes: "other 10 rows", kernel: convnext_fused (r0 anchor), meta: {}}
tests: [H33, H28, H31]
findings:
  - "STEP 2 ingest probe, self-written tcgen05 GEMM, 148 CTAs, warm L2, 4-stage SW128 ring, swap-AB 128xN tile, all variants validated against torch (max err 1.6e-5); clk per 64-k stage vs the 2N ideal. (a) 1SM: N=128 479 (1.87x), N=224 532 (1.19x), N=256 546 (1.07x); K=768 within 2%. (c) B TMA-multicast over a 2-CTA cluster: 489 / 548 / 555, no gain. (b) cta_group::2 pair (M=256, each CTA half the B tokens, leader-only MMA, multicast commit): 345 (1.35x) / 455 (1.01x) / 518 (1.01x) [probe_b200]"
  - "H31/H33 verdict: the per-SM cap is SM-side, not L2-side, because multicast (fewer L2 reads, same per-SM landing bytes) changes nothing while the pair (half the per-SM B bytes) reaches the MMA ideal at N >= 224. 1SM landing rate is 68-92 B/clk/SM, above r4's 55-72 estimate (r4's gate config was worse). There is also a per-stage floor of ~480 clk at N=128 in 1SM [probe_b200]"
  - "Working cta_group::2 recipe: tcgen05.alloc/relinquish/dealloc .cta_group::2 in both CTAs; cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes signalling the leader's full barrier (address from mapa ... 0), leader-only expect_tx of both CTAs' bytes; idesc M field 256>>4; tcgen05.commit.cta_group::2...multicast::cluster mask 3 for empty and accumulator-full; each CTA reads its own 128 TMEM lanes x N columns. Correct on the first try [probe_b200]"
  - "STEP 1 stamps, r4 parent (CTA median/max, us from kernel start; first container): 32,768 phase A 11.5/17.7, sync1 18.9, GEMM1 epilogue end 74.0/76.5, sync2 78.1, GEMM2 end 96.5/98.0; per-tile GEMM1 epilogue (post-accumulator) 45.6/50.6 us, wait 4.6, LN pre 3.3; GEMM2 wait 8.5, post 7.9. 8,384: G1 post 46.7 us on 1 tile; 16,384: 49.0/64.8; 16,768: 30.5; 8,768: 12.4; 2,768: 7.4. The GEMM1 mainloop finished within 3-7 us of sync1 everywhere, so ingest was not the 2x term [probe_b200]"
  - "Root cause: cuobjdump showed STACK 128 and 35-36 LDL/STL in each r4 tc_kernel instance (compile_b200's 'local 0' counts spills only, not the stack frame). hreg[4][16] lived in local memory, and with 227 KB smem the L1 is small, so each element paid L2 round trips (~370 clk per element per lane). Removing it cut G1 post from 46.7 to 12.6 us at 8,384 [probe_b200]"
  - "Interleaved x3, harness_time, same container (us), parent -> spill fix: 2,768 32.7 -> 30.2; 8,768 48.1 -> 39.1; 16,768 71.3 -> 51.4; 32,768 98.5 -> 76.5; 8,384 79.8 -> 40.8; 16,384 112.2 -> 64.7 [probe_b200]"
  - "Phase-A balance (equal C-float weight units, reversed CTA order, W2 quarter rows): -0.6 to -3.1 us across the six rows (e.g. 2,768 30.4 -> 27.3, 32,768 76.1 -> 73.3) [probe_b200]"
  - "Dead end: lighter GEMM1 epilogue arithmetic (precomputed rstd and -rstd*mu, two weighted FFMA sumsq accumulators instead of 3 compare-selects) was 0.3-0.8 us slower than the select version. The epilogue is not issue-bound [probe_b200]"
  - "Staged coalesced h/g stores (per-warp smem transpose, 4 lanes per 64-B row segment) vs one-row-per-lane 16-B stores: 2,768 27.4 -> 27.2; 8,768 37.0 -> 34.6; 16,768 49.3 -> 46.5; 32,768 72.9 -> 67.1; 8,384 39.3 -> 37.9; 16,384 63.8 -> 59.7. Per-tile G1 post is now 5.7 (8,768), 8.8 (16,768), 9 (32,768, 2 tiles = 18.1), 10.9 (8,384), 11.7 (16,384) us [probe_b200]"
  - "Remaining terms at 16,384 / 32,768: 192 / 264 GEMM1 tiles on 148 CTAs, so 44 / 116 CTAs run 2 tiles (16,384 G1 epilogue end 32.3 median vs 44.5 max); phase A 10-16 us; GEMM2 ~11-19 us (wait 3.7-8 us, post 6-8 us at 2 units) [probe_b200]"
  - "Pitfall: globaltimer stamps taken by tid 0 right after __syncthreads can be scheduled ahead of the barrier by the compiler: the pre-sync2 and end stamps read earlier than the epilogue warps' own stamps. Only use stamps from the warp that does the work [probe_b200]"
  - "PRECISION LOG (owed, delivered for D r4 and this kernel): fp64 reference, harness generator rules (randn; weight matrices /sqrt(shape[-1]); LN weight 1, bias 0; biases randn), torch.manual_seed(200) per row, 10 rounds; max|err| and count beyond own atol + 1e-5|ref|, parent / this: 2,768 (atol 2.6e-3) 1.700e-3 / 1.700e-3, 0/0; 8,768 (2.9e-3) 1.700e-3 / 1.700e-3, 0/0; 16,768 (2.9e-3) 1.701e-3 / 1.701e-3, 0/0; 32,768 (2.9e-3) 1.811e-3 / 1.810e-3, 0/0; 8,384 (2.8e-3) 1.796e-3 / 1.796e-3, 0/0; 16,384 (3.3e-3) 1.946e-3 / 1.946e-3, 0/0. Max error is 59-67% of atol, more margin than A's fp16 paths [probe_b200]"
  - "NOT DONE: the precision log for r3-ws-coop-widec at 2,768 (the merge's current 2,768 path). get_kernel returned 72 KB that could not be read in-session. This kernel is faster at 2,768 (27.6 vs 34.3 rented) and has a delivered log, so the merge can drop D r3 there [run_tests]"
  - "run_tests 16/16 PASS (portal-order re-runs included), rented, this vs r5-merge-b in the same container: 2,768 27.6 vs 34.3 (-6.7); 8,768 34.6 vs 49.4 (-14.8); 16,768 46.5 vs 55.4 (-8.9); 32,768 66.6 vs 70.9 (-4.3); 8,384 38.0 vs 51.0 (-13.0); 16,384 58.5 vs 62.4 (-3.9). The other 10 rows run the r0 anchor (merge-only kernel) [run_tests]"
  - "H33 success line: at least 8 us under A r5 on 3 of the 5 rows (8,768, 8,384, 16,768), 2,768 <= 32 kept, 0 beyond-atol: MET, but through the epilogue fix rather than the 2SM engine. The pair port (STEP 3) is still the lever for the GEMM2 wait (3.4-8 us) and the 32,768 / 16,384 mainloops [INFERRED]"
  - "Next levers: port the cta_group::2 recipe above to GEMM1 (M=256 hidden pairs: 12 pair-rows at C=768, so tile counts drop to 132 pairs at 32,768) and GEMM2; rebalance 16,384's 192 tiles (2 tiles on 44 CTAs = 12 us tail); cut phase A's 10-16 us (conv units of 32 planes are uneven at 16,384: 192 units on 148 CTAs) [INFERRED]"
paths:
  - {max_tokens: 3136, lang: cuda, width: 128, threads: 384, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "six wide-C rows only (2/8/16/32,768,7,7; 8/16,384,14,14): tc_kernel, cooperative, 1 CTA per SM, tcgen05 kind::f16 swap-AB, staged coalesced epilogue"}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1, note: "other 10 rows: r0 fp32 anchor (merge-only; use other lineages there)"}
runs_on:
  H200: {runs: false, representative: false, note: "tcgen05/TMEM are sm_100a-only"}
  A100: {runs: false, representative: false}
  L40S: {runs: false, representative: false}
risks:
  - "build must target B200 only (sm_100a); a LOCAL sm_100 target fails to assemble tcgen05"
  - "cooperative launch at 227.5 KB smem and 384 threads requires every SM free (grid = SM count)"
  - "split-K red.add and float4 red.add make summation order nondeterministic (max err stable to 1e-6 across 10 rounds)"
  - "the 10 non-target rows run the slow r0 anchor; only useful merged into a dispatch with lineages A/B on the six wide-C rows"
  - "portal ratio for D paths was ~1.32 (single cooperative launch); the gains here are SM-side cycles plus L2 traffic, so expect roughly 1.25-1.35x on the portal"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```