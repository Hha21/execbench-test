### Rationale
I built the experiment as a new kernel, `g_p1s`. Each CTA takes one 2-row tile and all 16 warps work through it in order: conv, then LayerNorm statistics, then GEMM1 with the fused epilogue.
- **4,96:** tiles are 112 tokens padded to 128, giving 112 CTAs. W1' is loaded once at kernel start and stays in shared memory. Padding lanes skip their stores and their GRN sums.
- **8,320:** W1' is streamed in 64-row blocks of 40 KB, one buffer per 4-warp group, through a `cp.async.bulk` + mbarrier ring. Each buffer is refilled as soon as its group's mma is done.

Results on the rented B200, 16/16 pass, same container as r7-merge:
- **4,96:** 37.0 vs 41.4 µs (−10.7 %).
- **8,320:** 67.2 vs 73.1 µs (−8.0 %).
- **64,96 / 32,128:** within 0.2 % of the parent.

Neither success line was met. H37 is not refuted either: pass 1 at 8,320 is 43.3 µs, under the 45 µs line, with the ring wait stamped at 0.34 µs. The stamps show what binds:
- **mma.sync pipe:** 16 warps keep it saturated, so the MMA alone costs ~14.5 µs at 8,320.
- **Epilogue:** it runs after the MMA and does not overlap it. Splitting the warps into two alternating halves does not help, because 8 warps cannot keep the pipe full.
- **Conv:** latency-bound at 9–13 µs.

The merge should take 4,96 and 8,320 from this kernel.

```json solution-spec
{"name": "r8-g-e2-singletile", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda", "-lcublasLt"]}}}
```

```cuda file=kernel.cu
// Lineage G r8 (H37): fused pass 1 for the HW >= 1024 rows.
//  64,96,112,112 and 32,128,64,64: r7 warp-specialised g_pass1 (unchanged).
//  4,96,56,56 and 8,320,32,32: g_p1s, one 2-row tile per CTA (112 tokens padded to 128 / 64 tokens), all 16 warps
//  phase-serial per tile: conv -> LN stats -> GEMM1 + epilogue. W1' arrives by cp.async.bulk issued at kernel start:
//  resident (74 KB) at C=96; streamed at C=320 in 64-hidden-row blocks (40 KB) through one buffer per 4-warp group,
//  refilled as soon as the group's mma for a block is done (the refill overlaps the GELU epilogue and other groups).
//  K0: W1' = fp16(W1*lnw) as m16n8k16 A fragments, u[k] = sum_c W1'16[k][c], v[k] = W1.lnb + b1,
//      dwconv weights padded to [C][7][8], per-(image,k) GRN sums zeroed (same call, before P1)
//  K3: per image s_b[k] = 1 + gw*gf/(mean gf + eps); W2_b = fp16(W2*s_b) [c][K2]; b2' = W2.gb + b2 as hi/lo columns
//  G2: cuBLASLt batched: out_b (col-major HW x C = NCHW) = h_b^T W2_b^T + x_b
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <stdint.h>
#include <map>
#include <tuple>

struct GParams {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* w1f; float* uv; void* h; float* gsum; void* w2b; float* dwp;
  int B, C, H, W, HW, C4, K2, TPI, ntiles;
  float eps, ln_eps;
  unsigned long long* stamp;  // only read in G_STAMPS probe builds
};

__device__ __forceinline__ float g_wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// exact-erf GELU z*Phi(z) with Phi from the A&S 7.1.26 erfc (|err| <= 1.5e-7), rcp and ex2 on the MUFU pipe
__device__ __forceinline__ float gelu_as(float z) {
  const float u = fabsf(z) * 0.70710678118654752f;
  const float t = __fdividef(1.f, fmaf(0.3275911f, u, 1.f));
  float y = fmaf(1.061405429f, t, -1.453152027f);
  y = fmaf(y, t, 1.421413741f);
  y = fmaf(y, t, -0.284496736f);
  y = fmaf(y, t, 0.254829592f);
  const float e = exp2f(z * z * -0.72134752044448170f);
  const float hq = 0.5f * (y * t) * e;
  return z * (z >= 0.f ? 1.f - hq : hq);
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* ptr) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void nbar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\t"
      "bra LAB_WAIT;\n\t"
      "DONE:\n\t}" ::"r"(a), "r"(par) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mb) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}
#ifdef G_STAMPS
__device__ __forceinline__ unsigned long long g_gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory"); return t;
}
#endif

// K0
__global__ void __launch_bounds__(256) g_prep(GParams p, int nfb, int nub, int ndb) {
  const int C = p.C, C4 = p.C4, KS = C / 16;
  int blk = blockIdx.x;
  if (blk < nfb) {
    const int i = blk * 256 + threadIdx.x;
    if (i >= (C4 / 16) * KS * 32) return;
    const int lane = i & 31, f = i >> 5, mt = f / KS, ks = f - mt * KS;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = mt * 16 + g, r1 = r0 + 8, c0 = ks * 16 + 2 * t, c2 = c0 + 8;
    auto pk = [&](int r, int c) {
      const float a = __ldg(p.w1 + (size_t)r * C + c) * __ldg(p.lnw + c);
      const float b = __ldg(p.w1 + (size_t)r * C + c + 1) * __ldg(p.lnw + c + 1);
      return h2u(__floats2half2_rn(a, b));
    };
    uint4 v; v.x = pk(r0, c0); v.y = pk(r1, c0); v.z = pk(r0, c2); v.w = pk(r1, c2);
    reinterpret_cast<uint4*>(p.w1f)[i] = v;
    return;
  }
  blk -= nfb;
  if (blk < nub) {
    const int k = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (k >= C4) return;
    const float* wr = p.w1 + (size_t)k * C;
    float u = 0.f, v = 0.f;
    for (int c = lane; c < C; c += 32) {
      const float w = __ldg(wr + c);
      u += __half2float(__float2half_rn(w * __ldg(p.lnw + c)));
      v = fmaf(w, __ldg(p.lnb + c), v);
    }
    u = g_wsum(u); v = g_wsum(v);
    if (lane == 0) { p.uv[2 * k] = u; p.uv[2 * k + 1] = v + __ldg(p.b1 + k); }
    return;
  }
  blk -= nub;
  if (blk < ndb) {
    const int i = blk * 256 + threadIdx.x;
    if (i >= C * 56) return;
    const int c = i / 56, rem = i - c * 56, ki = rem >> 3, j = rem & 7;
    p.dwp[i] = j < 7 ? __ldg(p.dww + c * 49 + ki * 7 + j) : 0.f;
    return;
  }
  blk -= ndb;
  const int i = blk * 256 + threadIdx.x;
  if (i < p.B * C4) p.gsum[i] = 0.f;
}

template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
struct P1Cfg {
  static constexpr int C4 = 4 * C, T = R * W, LDT = T + 8, QO = W / 8, NT = (NCW + NGW) * 32, NCT = NCW * 32,
                       NGT = NGW * 32, KS = C / 16, MT = C4 / 16;
  static constexpr int SM_W1F = MT * KS * 32 * 16, SM_D = C * LDT * 2, SM_ST = T * 8, SM_UV = C4 * 8, SM_PS = C4 * 4,
                       SM_DW = DWS ? C * 56 * 4 : 0;
  static constexpr int OFF_D = SM_W1F, OFF_ST = OFF_D + 2 * SM_D, OFF_UV = OFF_ST + 2 * SM_ST, OFF_PS = OFF_UV + SM_UV,
                       OFF_DW = OFF_PS + 2 * SM_PS, SMEM = OFF_DW + SM_DW;
};

// token t of a 32-token chunk sits at position n*8 + 2q + e (t = 8q + 2n + e), so that after ldmatrix.trans + mma each
// lane's accumulators cover 8 consecutive tokens of a hidden row (one 16-B store per row)
__device__ __forceinline__ int tok_pos(int t) { return (t & ~31) | (((t >> 1) & 3) << 3) | (((t >> 3) & 3) << 1) | (t & 1); }

template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
__global__ void __launch_bounds__((NCW + NGW) * 32, 1) g_pass1(GParams p) {
  using Cf = P1Cfg<C, R, W, NCW, NGW, MG, DWS>;
  constexpr int C4 = Cf::C4, T = Cf::T, LDT = Cf::LDT, QO = Cf::QO, NT = Cf::NT, NCT = Cf::NCT, NGT = Cf::NGT, KS = Cf::KS;
  constexpr int P = T / 2, S = NCT / P;
  static_assert(NCT % P == 0 && (S & (S - 1)) == 0 && S <= 32, "stats split");
  constexpr int NMG = C4 / (16 * MG), NCH = T / 32;
  static_assert(C4 % (16 * MG) == 0 && T % 32 == 0 && W % 8 == 0 && R % 2 == 0 && C % 16 == 0, "tiles");
  constexpr int PF = 3;  // conv register prefetch distance (input rows)
  extern __shared__ __align__(16) unsigned char sm[];
  uint4* w1s = reinterpret_cast<uint4*>(sm);
  __half* dsb = reinterpret_cast<__half*>(sm + Cf::OFF_D);
  float2* stb = reinterpret_cast<float2*>(sm + Cf::OFF_ST);
  float2* uvs = reinterpret_cast<float2*>(sm + Cf::OFF_UV);
  float* psb = reinterpret_cast<float*>(sm + Cf::OFF_PS);
  float* dws = reinterpret_cast<float*>(sm + Cf::OFF_DW);
  const int tid = threadIdx.x;
  const int H = p.H, HW = p.HW, K2 = p.K2, TPI = p.TPI;
  __half* hg = reinterpret_cast<__half*>(p.h);

  for (int i = tid; i < Cf::SM_W1F / 16; i += NT) w1s[i] = reinterpret_cast<const uint4*>(p.w1f)[i];
  for (int i = tid; i < C4; i += NT) { uvs[i] = reinterpret_cast<const float2*>(p.uv)[i]; psb[i] = 0.f; psb[C4 + i] = 0.f; }
  if (DWS)
    for (int i = tid; i < C * 56; i += NT) dws[i] = p.dwp[i];
  __syncthreads();
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;

  if (warp < NCW) {
    // ================= conv warps
    int i = 0;
#pragma unroll 1
    for (int tile = blockIdx.x; tile < p.ntiles; tile += gridDim.x, i++) {
      const int buf = i & 1;
      const int b = tile / TPI, rb = tile - b * TPI, h0 = rb * R, hw0 = h0 * W;
      if (i >= 2) nbar_sync(3 + buf, NT);  // GEMM warps are done with this buffer
      __half* ds = dsb + buf * (C * LDT);
      constexpr int NIT = C * (R / 2) * QO;
#pragma unroll 1
      for (int it = tid; it < NIT; it += NCT) {
        const int c = it / ((R / 2) * QO), rem = it - c * ((R / 2) * QO), rs = rem / QO, o = rem - rs * QO;
        const int w0 = 8 * o, r0 = 2 * rs;
        const float* xc = p.x + (size_t)(b * C + c) * HW;
        const float* wbase = DWS ? dws + c * 56 : p.dwp + c * 56;
        const float bias = __ldg(p.dwb + c);
        float acc[2][8];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int cc = 0; cc < 8; cc++) acc[r][cc] = 0.f;
        float4 rows[8][4];
        auto ldrow = [&](int ir) {
          const int hh = h0 + r0 - 3 + ir;
          rows[ir][0] = rows[ir][1] = rows[ir][2] = rows[ir][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (hh >= 0 && hh < H) {
            const float* rp = xc + hh * W + w0;
            if (o > 0) rows[ir][0] = __ldg(reinterpret_cast<const float4*>(rp - 4));
            rows[ir][1] = __ldg(reinterpret_cast<const float4*>(rp));
            rows[ir][2] = __ldg(reinterpret_cast<const float4*>(rp + 4));
            if (o < QO - 1) rows[ir][3] = __ldg(reinterpret_cast<const float4*>(rp + 8));
          }
        };
#pragma unroll
        for (int ir = 0; ir < PF; ir++) ldrow(ir);
#pragma unroll
        for (int ir = 0; ir < 8; ir++) {
          if (ir + PF < 8) ldrow(ir + PF);
          const float v[14] = {rows[ir][0].y, rows[ir][0].z, rows[ir][0].w, rows[ir][1].x, rows[ir][1].y,
                               rows[ir][1].z, rows[ir][1].w, rows[ir][2].x, rows[ir][2].y, rows[ir][2].z,
                               rows[ir][2].w, rows[ir][3].x, rows[ir][3].y, rows[ir][3].z};
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const int ki = ir - r;
            if (ki >= 0 && ki < 7) {
              const float4 wa = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8)
                                    : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8));
              const float4 wb = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8 + 4)
                                    : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8 + 4));
              const float w[7] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z};
#pragma unroll
              for (int j = 0; j < 7; j++)
#pragma unroll
                for (int cc = 0; cc < 8; cc++) acc[r][cc] = fmaf(v[cc + j], w[j], acc[r][cc]);
            }
          }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
          const int t0 = (r0 + r) * W + w0;  // multiple of 8: q fixed, n = 0..3
          __half* dp = ds + c * LDT + (t0 & ~31) + (((t0 >> 3) & 3) << 1);
#pragma unroll
          for (int n = 0; n < 4; n++)
            *reinterpret_cast<uint32_t*>(dp + 8 * n) = h2u(__floats2half2_rn(acc[r][2 * n] + bias, acc[r][2 * n + 1] + bias));
        }
      }
      nbar_sync(5, NCT);
      {  // per-token LayerNorm statistics of the fp16 d tile (2 tokens per thread, S channel slices)
        const int pp = tid / S, s = tid - pp * S;
        const int pos = tok_pos(2 * pp);
        float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll 8
        for (int c = s; c < C; c += S) {
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(ds + c * LDT + pos));
          s1a += f.x; s2a = fmaf(f.x, f.x, s2a); s1b += f.y; s2b = fmaf(f.y, f.y, s2b);
        }
#pragma unroll
        for (int o = 1; o < S; o <<= 1) {
          s1a += __shfl_xor_sync(0xffffffffu, s1a, o); s2a += __shfl_xor_sync(0xffffffffu, s2a, o);
          s1b += __shfl_xor_sync(0xffffffffu, s1b, o); s2b += __shfl_xor_sync(0xffffffffu, s2b, o);
        }
        if (s == 0) {
          constexpr float inv = 1.f / (float)C;
          const float m0 = s1a * inv, m1 = s1b * inv;
          const float q0 = rsqrtf(fmaxf(s2a * inv - m0 * m0, 0.f) + p.ln_eps);
          const float q1 = rsqrtf(fmaxf(s2b * inv - m1 * m1, 0.f) + p.ln_eps);
          *reinterpret_cast<float4*>(stb + buf * T + 2 * pp) = make_float4(q0, q0 * m0, q1, q1 * m1);
          __half* ha = hg + ((size_t)b * K2 + C4) * HW + hw0 + 2 * pp;
          const uint32_t one = h2u(__floats2half2_rn(1.f, 1.f));
#pragma unroll
          for (int j = 0; j < 8; j++) *reinterpret_cast<uint32_t*>(ha + (size_t)j * HW) = j < 2 ? one : 0u;
        }
      }
      nbar_arrive(1 + buf, NT);  // d tile + stats ready
    }
    for (int j = (i >= 2 ? i - 2 : 0); j < i; j++) nbar_sync(3 + (j & 1), NT);
  } else {
    // ================= GEMM + epilogue warps
    const int gw = warp - NCW, gt = tid - NCT, g = lane >> 2, t4 = lane & 3;
    int i = 0;
#pragma unroll 1
    for (int tile = blockIdx.x; tile < p.ntiles; tile += gridDim.x, i++) {
      const int buf = i & 1;
      const int b = tile / TPI, rb = tile - b * TPI, hw0 = rb * R * W;
      nbar_sync(1 + buf, NT);
      const __half* ds = dsb + buf * (C * LDT);
      const float2* sts = stb + buf * T;
      float* pss = psb + buf * C4;
#pragma unroll 1
      for (int wt = gw; wt < NMG * NCH; wt += NGW) {
        const int mg = wt / NCH, nc = wt - mg * NCH;
        float acc[MG][4][4];
#pragma unroll
        for (int m = 0; m < MG; m++)
#pragma unroll
          for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
          uint32_t bf[2][4];
#pragma unroll
          for (int pr = 0; pr < 2; pr++) {
            const int mi = lane >> 3, rr = lane & 7;
            ldsm_x4_t(bf[pr], ds + (ks * 16 + (mi & 1) * 8 + rr) * LDT + nc * 32 + pr * 16 + (mi >> 1) * 8);
          }
#pragma unroll
          for (int m = 0; m < MG; m++) {
            const uint4 a = w1s[((mg * MG + m) * KS + ks) * 32 + lane];
#pragma unroll
            for (int n = 0; n < 4; n++) mma16816(acc[m][n], a, bf[n >> 1][(n & 1) * 2], bf[n >> 1][(n & 1) * 2 + 1]);
          }
        }
        // lane holds tokens nc*32 + 8*t4 + 2n + e for hidden rows k0 (e = 0,1) and k1 = k0 + 8 (e = 2,3)
        const int tb = nc * 32 + 8 * t4;
        float4 st[4];
#pragma unroll
        for (int n = 0; n < 4; n++) st[n] = *reinterpret_cast<const float4*>(sts + tb + 2 * n);
#pragma unroll
        for (int m = 0; m < MG; m++) {
          const int k0 = (mg * MG + m) * 16 + g, k1 = k0 + 8;
          const float2 uv0 = uvs[k0], uv1 = uvs[k1];
          float ss0 = 0.f, ss1 = 0.f;
          uint32_t o0[4], o1[4];
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const float g00 = gelu_as(fmaf(st[n].x, acc[m][n][0], fmaf(-st[n].y, uv0.x, uv0.y)));
            const float g01 = gelu_as(fmaf(st[n].z, acc[m][n][1], fmaf(-st[n].w, uv0.x, uv0.y)));
            const float g10 = gelu_as(fmaf(st[n].x, acc[m][n][2], fmaf(-st[n].y, uv1.x, uv1.y)));
            const float g11 = gelu_as(fmaf(st[n].z, acc[m][n][3], fmaf(-st[n].w, uv1.x, uv1.y)));
            ss0 = fmaf(g00, g00, fmaf(g01, g01, ss0));
            ss1 = fmaf(g10, g10, fmaf(g11, g11, ss1));
            o0[n] = h2u(__floats2half2_rn(g00, g01));
            o1[n] = h2u(__floats2half2_rn(g10, g11));
          }
          __half* hp0 = hg + ((size_t)b * K2 + k0) * HW + hw0 + tb;
          *reinterpret_cast<uint4*>(hp0) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
          *reinterpret_cast<uint4*>(hp0 + (size_t)8 * HW) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
          ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1); ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
          ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1); ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
          if (t4 == 0) { atomicAdd(pss + k0, ss0); atomicAdd(pss + k1, ss1); }
        }
      }
      nbar_arrive(3 + buf, NT);  // done reading d tile + stats of this buffer
      nbar_sync(6, NGT);
      float* gs = p.gsum + (size_t)b * C4;
      for (int k = gt; k < C4; k += NGT) { atomicAdd(gs + k, pss[k]); pss[k] = 0.f; }
    }
  }
}

// ---------------------------------------------------------------- r8: single-tile-per-CTA fused pass 1 (g_p1s)
template <int C, int R, int W, int TP, int MG, bool STREAM, bool DWS>
struct SCfg {
  static constexpr int NT = 512;
  static constexpr int C4 = 4 * C, T = R * W, LDT = TP + 8, KS = C / 16, MT = C4 / 16, NCH = TP / 32;
  static constexpr int NGRP = 4, BROWS = 64, NBLK = C4 / BROWS, NBPG = STREAM ? NBLK / NGRP : 1;
  static constexpr int BUF_U4 = (BROWS / 16) * KS * 32;  // uint4 per streamed 64-row block
  static constexpr int SM_A = STREAM ? NGRP * BUF_U4 * 16 : MT * KS * 32 * 16;
  static constexpr int SM_D = C * LDT * 2, SM_ST = TP * 8, SM_DW = DWS ? C * 56 * 4 : 0;
  static constexpr int OFF_D = SM_A, OFF_ST = OFF_D + SM_D, OFF_DW = OFF_ST + SM_ST, OFF_MB = OFF_DW + SM_DW,
                       SMEM = OFF_MB + 64;
  static constexpr int P = T / 2, Q = NT / P;
  static constexpr int S = Q >= 32 ? 32 : Q >= 16 ? 16 : Q >= 8 ? 8 : Q >= 4 ? 4 : Q >= 2 ? 2 : 1;
  static constexpr int NST = P * S;
};

// 7x7 depthwise conv of a 2-row strip, 8 columns per item, all channels; writes the fp16 d tile (token-permuted)
template <int C, int R, int W, int LDT, bool DWS>
__device__ __forceinline__ void s_conv(const GParams& p, __half* ds, const float* dws, int b, int h0, int tid, int nthr) {
  constexpr int QO = W / 8, NIT = C * (R / 2) * QO, PF = 3;
  const int H = p.H, HW = p.HW;
#pragma unroll 1
  for (int it = tid; it < NIT; it += nthr) {
    const int c = it / ((R / 2) * QO), rem = it - c * ((R / 2) * QO), rs = rem / QO, o = rem - rs * QO;
    const int w0 = 8 * o, r0 = 2 * rs;
    const float* xc = p.x + (size_t)(b * C + c) * HW;
    const float* wbase = DWS ? dws + c * 56 : p.dwp + c * 56;
    const float bias = __ldg(p.dwb + c);
    float acc[2][8];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int cc = 0; cc < 8; cc++) acc[r][cc] = 0.f;
    float4 rows[8][4];
    auto ldrow = [&](int ir) {
      const int hh = h0 + r0 - 3 + ir;
      rows[ir][0] = rows[ir][1] = rows[ir][2] = rows[ir][3] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (hh >= 0 && hh < H) {
        const float* rp = xc + hh * W + w0;
        if (o > 0) rows[ir][0] = __ldg(reinterpret_cast<const float4*>(rp - 4));
        rows[ir][1] = __ldg(reinterpret_cast<const float4*>(rp));
        rows[ir][2] = __ldg(reinterpret_cast<const float4*>(rp + 4));
        if (o < QO - 1) rows[ir][3] = __ldg(reinterpret_cast<const float4*>(rp + 8));
      }
    };
#pragma unroll
    for (int ir = 0; ir < PF; ir++) ldrow(ir);
#pragma unroll
    for (int ir = 0; ir < 8; ir++) {
      if (ir + PF < 8) ldrow(ir + PF);
      const float v[14] = {rows[ir][0].y, rows[ir][0].z, rows[ir][0].w, rows[ir][1].x, rows[ir][1].y,
                           rows[ir][1].z, rows[ir][1].w, rows[ir][2].x, rows[ir][2].y, rows[ir][2].z,
                           rows[ir][2].w, rows[ir][3].x, rows[ir][3].y, rows[ir][3].z};
#pragma unroll
      for (int r = 0; r < 2; r++) {
        const int ki = ir - r;
        if (ki >= 0 && ki < 7) {
          const float4 wa = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8)
                                : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8));
          const float4 wb = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8 + 4)
                                : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8 + 4));
          const float w[7] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z};
#pragma unroll
          for (int j = 0; j < 7; j++)
#pragma unroll
            for (int cc = 0; cc < 8; cc++) acc[r][cc] = fmaf(v[cc + j], w[j], acc[r][cc]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int t0 = (r0 + r) * W + w0;
      __half* dp = ds + c * LDT + (t0 & ~31) + (((t0 >> 3) & 3) << 1);
#pragma unroll
      for (int n = 0; n < 4; n++)
        *reinterpret_cast<uint32_t*>(dp + 8 * n) = h2u(__floats2half2_rn(acc[r][2 * n] + bias, acc[r][2 * n + 1] + bias));
    }
  }
}

template <int MG, int KS, int LDT>
__device__ __forceinline__ void s_mma(float (&acc)[MG][4][4], const uint4* a0, const __half* ds, int nc, int lane) {
#pragma unroll
  for (int m = 0; m < MG; m++)
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < KS; ks++) {
    uint32_t bf[2][4];
#pragma unroll
    for (int pr = 0; pr < 2; pr++) {
      const int mi = lane >> 3, rr = lane & 7;
      ldsm_x4_t(bf[pr], ds + (ks * 16 + (mi & 1) * 8 + rr) * LDT + nc * 32 + pr * 16 + (mi >> 1) * 8);
    }
#pragma unroll
    for (int m = 0; m < MG; m++) {
      const uint4 a = a0[(m * KS + ks) * 32 + lane];
#pragma unroll
      for (int n = 0; n < 4; n++) mma16816(acc[m][n], a, bf[n >> 1][(n & 1) * 2], bf[n >> 1][(n & 1) * 2 + 1]);
    }
  }
}

// LN fold + GELU + fp16 hidden store + per-(image,k) sum of squares; lanes whose 8 tokens are padding (>= T) are masked
template <int MG, int T>
__device__ __forceinline__ void s_epi(const float (&acc)[MG][4][4], int krow, int nc, int lane, const float2* sts,
                                      const float2* uvg, __half* hb, float* gs, int HW) {
  const int g = lane >> 2, t4 = lane & 3, tb = nc * 32 + 8 * t4;
  const bool ok = tb < T;
  float4 st[4];
#pragma unroll
  for (int n = 0; n < 4; n++) st[n] = *reinterpret_cast<const float4*>(sts + tb + 2 * n);
#pragma unroll
  for (int m = 0; m < MG; m++) {
    const int k0 = krow + m * 16 + g, k1 = k0 + 8;
    const float2 uv0 = __ldg(uvg + k0), uv1 = __ldg(uvg + k1);
    float ss0 = 0.f, ss1 = 0.f;
    uint32_t o0[4], o1[4];
#pragma unroll
    for (int n = 0; n < 4; n++) {
      const float g00 = gelu_as(fmaf(st[n].x, acc[m][n][0], fmaf(-st[n].y, uv0.x, uv0.y)));
      const float g01 = gelu_as(fmaf(st[n].z, acc[m][n][1], fmaf(-st[n].w, uv0.x, uv0.y)));
      const float g10 = gelu_as(fmaf(st[n].x, acc[m][n][2], fmaf(-st[n].y, uv1.x, uv1.y)));
      const float g11 = gelu_as(fmaf(st[n].z, acc[m][n][3], fmaf(-st[n].w, uv1.x, uv1.y)));
      ss0 = fmaf(g00, g00, fmaf(g01, g01, ss0));
      ss1 = fmaf(g10, g10, fmaf(g11, g11, ss1));
      o0[n] = h2u(__floats2half2_rn(g00, g01));
      o1[n] = h2u(__floats2half2_rn(g10, g11));
    }
    if (ok) {
      __half* hp0 = hb + (size_t)k0 * HW + tb;
      *reinterpret_cast<uint4*>(hp0) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
      *reinterpret_cast<uint4*>(hp0 + (size_t)8 * HW) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
    } else {
      ss0 = 0.f; ss1 = 0.f;
    }
    ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1); ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
    ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1); ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
    if (t4 == 0) { atomicAdd(gs + k0, ss0); atomicAdd(gs + k1, ss1); }
  }
}

template <int C, int R, int W, int TP, int MG, bool STREAM, bool DWS>
__global__ void __launch_bounds__(512, 1) g_p1s(GParams p) {
  using Cf = SCfg<C, R, W, TP, MG, STREAM, DWS>;
  constexpr int NT = Cf::NT, C4 = Cf::C4, T = Cf::T, LDT = Cf::LDT, KS = Cf::KS, NCH = Cf::NCH, S = Cf::S,
                NST = Cf::NST, NBPG = Cf::NBPG, BUF_U4 = Cf::BUF_U4;
  static_assert(T <= TP && TP % 32 == 0 && T % 8 == 0 && W % 8 == 0 && R % 2 == 0 && C % 16 == 0, "tile");
  static_assert(!STREAM || (NCH == 2 && MG == 2 && Cf::NBLK % Cf::NGRP == 0), "stream layout");
  static_assert(STREAM || C4 % (16 * MG) == 0, "mg");
  static_assert(NST % 32 == 0 && S <= 32, "stats split");
  extern __shared__ __align__(128) unsigned char sm[];
  uint4* w1s = reinterpret_cast<uint4*>(sm);
  __half* ds = reinterpret_cast<__half*>(sm + Cf::OFF_D);
  float2* stb = reinterpret_cast<float2*>(sm + Cf::OFF_ST);
  float* dws = reinterpret_cast<float*>(sm + Cf::OFF_DW);
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t mb0 = sa + Cf::OFF_MB;
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int HW = p.HW, K2 = p.K2, TPI = p.TPI;
  __half* hg = reinterpret_cast<__half*>(p.h);
  const float2* uvg = reinterpret_cast<const float2*>(p.uv);
  const uint4* w1g = reinterpret_cast<const uint4*>(p.w1f);
  const int ntl = (p.ntiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int niter = ntl * NBPG;
#ifdef G_STAMPS
  unsigned long long* stp = p.stamp ? p.stamp + blockIdx.x * 8 : nullptr;
  if (stp && tid == 0) stp[0] = g_gtime();
#endif
  if (tid == 0) {
#pragma unroll
    for (int g = 0; g < 4; g++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb0 + 8 * g) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (STREAM) {
      for (int g = 0; g < Cf::NGRP; g++) bulk_g2s(sa + g * BUF_U4 * 16, w1g + g * BUF_U4, BUF_U4 * 16, mb0 + 8 * g);
    } else {
      bulk_g2s(sa, w1g, Cf::SM_A, mb0);
    }
  }
  if (DWS)
    for (int i = tid; i < C * 56; i += NT) dws[i] = p.dwp[i];
  if (T < TP)
    for (int i = tid; i < C * (TP - T); i += NT) {
      const int c = i / (TP - T), t = T + (i - c * (TP - T));
      ds[c * LDT + tok_pos(t)] = __float2half_rn(0.f);
    }
  for (int i = tid; i < TP; i += NT) stb[i] = make_float2(0.f, 0.f);
  __syncthreads();
#ifdef G_STAMPS
  if (stp && tid == 0) stp[4] = g_gtime();
#endif

  int n = 0;
#pragma unroll 1
  for (int tile = blockIdx.x; tile < p.ntiles; tile += gridDim.x) {
    const int b = tile / TPI, rb = tile - b * TPI, h0 = rb * R, hw0 = h0 * W;
    s_conv<C, R, W, LDT, DWS>(p, ds, dws, b, h0, tid, NT);
#ifdef G_STAMPS
    if (stp && tid == 0) stp[1] = g_gtime();
#endif
    __syncthreads();
    if (tid < NST) {  // per-token LayerNorm statistics of the fp16 d tile (2 tokens per thread, S channel slices)
      const int pp = tid / S, s = tid - pp * S;
      const int pos = tok_pos(2 * pp);
      float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll 4
      for (int c = s; c < C; c += S) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(ds + c * LDT + pos));
        s1a += f.x; s2a = fmaf(f.x, f.x, s2a); s1b += f.y; s2b = fmaf(f.y, f.y, s2b);
      }
#pragma unroll
      for (int o = 1; o < S; o <<= 1) {
        s1a += __shfl_xor_sync(0xffffffffu, s1a, o); s2a += __shfl_xor_sync(0xffffffffu, s2a, o);
        s1b += __shfl_xor_sync(0xffffffffu, s1b, o); s2b += __shfl_xor_sync(0xffffffffu, s2b, o);
      }
      if (s == 0) {
        constexpr float inv = 1.f / (float)C;
        const float m0 = s1a * inv, m1 = s1b * inv;
        const float q0 = rsqrtf(fmaxf(s2a * inv - m0 * m0, 0.f) + p.ln_eps);
        const float q1 = rsqrtf(fmaxf(s2b * inv - m1 * m1, 0.f) + p.ln_eps);
        *reinterpret_cast<float4*>(stb + 2 * pp) = make_float4(q0, q0 * m0, q1, q1 * m1);
        __half* ha = hg + ((size_t)b * K2 + C4) * HW + hw0 + 2 * pp;
        const uint32_t one = h2u(__floats2half2_rn(1.f, 1.f));
#pragma unroll
        for (int j = 0; j < 8; j++) *reinterpret_cast<uint32_t*>(ha + (size_t)j * HW) = j < 2 ? one : 0u;
      }
    }
    __syncthreads();
#ifdef G_STAMPS
    if (stp && tid == 0) stp[2] = g_gtime();
#endif
    __half* hb = hg + (size_t)b * K2 * HW + hw0;
    float* gs = p.gsum + (size_t)b * C4;
    if (STREAM) {
      const int grp = warp >> 2, gw = warp & 3, mg = gw >> 1, nc = gw & 1;
      const uint4* abuf = w1s + grp * BUF_U4;
      const uint32_t mbg = mb0 + 8 * grp;
#pragma unroll 1
      for (int j = 0; j < NBPG; j++, n++) {
        const int blk = grp + Cf::NGRP * j;
        mbar_wait(mbg, n & 1);
        float acc[MG][4][4];
        s_mma<MG, KS, LDT>(acc, abuf + mg * MG * KS * 32, ds, nc, lane);
        nbar_sync(1 + grp, 128);  // the group is done reading this W1' block
        if (gw == 0 && lane == 0 && n + 1 < niter) {
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          const int nb = grp + Cf::NGRP * ((j + 1) % NBPG);
          bulk_g2s(sa + grp * BUF_U4 * 16, w1g + nb * BUF_U4, BUF_U4 * 16, mbg);
        }
        s_epi<MG, T>(acc, blk * 64 + mg * MG * 16, nc, lane, stb, uvg, hb, gs, HW);
      }
    } else {
      constexpr int NMG = C4 / (16 * MG);
      if (tile == (int)blockIdx.x) mbar_wait(mb0, 0);
#pragma unroll 1
      for (int wt = warp; wt < NMG * NCH; wt += NT / 32) {
        const int mg = wt / NCH, nc = wt - mg * NCH;
        float acc[MG][4][4];
        s_mma<MG, KS, LDT>(acc, w1s + mg * MG * KS * 32, ds, nc, lane);
        s_epi<MG, T>(acc, mg * MG * 16, nc, lane, stb, uvg, hb, gs, HW);
      }
    }
#ifdef G_STAMPS
    if (stp && tid == 0) stp[3] = g_gtime();
#endif
    __syncthreads();
  }
}

// K3: grid (B, C/16): per image s_b[k] = 1 + gw*gf/(mean gf + eps); one W2_b row per warp, the row loaded up front
__global__ void __launch_bounds__(512, 2) g_grn(GParams p) {
  extern __shared__ __align__(16) float gsm[];
  constexpr int MAXI = 12;  // C4 <= 1536
  const int C = p.C, C4 = p.C4, K2 = p.K2;
  float* sc = gsm; float* red = gsm + C4;
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.y * 16 + warp;
  const float* wr = p.w2 + (size_t)(c < C ? c : 0) * C4;
  float4 wv[MAXI];
#pragma unroll
  for (int i = 0; i < MAXI; i++) {
    const int k = lane * 4 + i * 128;
    wv[i] = k < C4 ? __ldg(reinterpret_cast<const float4*>(wr + k)) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float loc = 0.f;
  for (int k = tid; k < C4; k += 512) {
    const float gf = sqrtf(__ldcg(p.gsum + (size_t)b * C4 + k));
    sc[k] = gf; loc += gf;
  }
  loc = g_wsum(loc);
  if (lane == 0) red[warp] = loc;
  __syncthreads();
  if (warp == 0) {
    float v = lane < 16 ? red[lane] : 0.f;
    v = g_wsum(v);
    if (lane == 0) red[32] = 1.f / (v / (float)C4 + p.eps);
  }
  __syncthreads();
  const float inv = red[32];
  for (int k = tid; k < C4; k += 512) sc[k] = fmaf(__ldg(p.gw + k) * sc[k], inv, 1.f);
  __syncthreads();
  if (c >= C) return;
  __half* w2b = reinterpret_cast<__half*>(p.w2b) + ((size_t)b * C + c) * K2;
  float dot = 0.f;
#pragma unroll
  for (int i = 0; i < MAXI; i++) {
    const int k = lane * 4 + i * 128;
    if (k < C4) {
      const float4 w = wv[i];
      const float4 gv = __ldg(reinterpret_cast<const float4*>(p.gb + k));
      const float4 s = *reinterpret_cast<const float4*>(sc + k);
      dot = fmaf(w.x, gv.x, dot); dot = fmaf(w.y, gv.y, dot); dot = fmaf(w.z, gv.z, dot); dot = fmaf(w.w, gv.w, dot);
      uint2 u;
      u.x = h2u(__floats2half2_rn(w.x * s.x, w.y * s.y));
      u.y = h2u(__floats2half2_rn(w.z * s.z, w.w * s.w));
      *reinterpret_cast<uint2*>(w2b + k) = u;
    }
  }
  dot = g_wsum(dot);
  if (lane == 0) {
    dot += __ldg(p.b2 + c);
    const __half hh = __float2half_rn(dot);
    const __half hl = __float2half_rn(dot - __half2float(hh));
    *reinterpret_cast<uint4*>(w2b + C4) = make_uint4(h2u(__halves2half2(hh, hl)), 0u, 0u, 0u);
  }
}

// ---------------------------------------------------------------- host
extern "C" int g_rows_per_tile(int C, int H, int W) {
  if (C == 96 && H == 112 && W == 112) return 2;
  if (C == 128 && H == 64 && W == 64) return 2;
  if (C == 96 && H == 56 && W == 56) return 2;
  if (C == 320 && H == 32 && W == 32) return 2;
  return 0;
}
static int g_num_sms() {
  static int n = 0;
  if (n == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); }
  return n;
}
template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
static const char* launch_p1(const GParams& p, cudaStream_t s) {
  using Cf = P1Cfg<C, R, W, NCW, NGW, MG, DWS>;
  static bool init = false;  // one attribute value per instantiation, never changed
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(g_pass1<C, R, W, NCW, NGW, MG, DWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    init = true;
  }
  const int grid = p.ntiles < g_num_sms() ? p.ntiles : g_num_sms();
  g_pass1<C, R, W, NCW, NGW, MG, DWS><<<grid, Cf::NT, Cf::SMEM, s>>>(p);
  return nullptr;
}
template <int C, int R, int W, int TP, int MG, bool STREAM, bool DWS>
static const char* launch_s(const GParams& p, cudaStream_t s) {
  using Cf = SCfg<C, R, W, TP, MG, STREAM, DWS>;
  static bool init = false;  // one attribute value per instantiation, never changed
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(g_p1s<C, R, W, TP, MG, STREAM, DWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    init = true;
  }
  const int grid = p.ntiles < g_num_sms() ? p.ntiles : g_num_sms();
  g_p1s<C, R, W, TP, MG, STREAM, DWS><<<grid, Cf::NT, Cf::SMEM, s>>>(p);
  return nullptr;
}
static cublasLtHandle_t g_lt() {
  static cublasLtHandle_t h = nullptr;
  if (!h) cublasLtCreate(&h);
  return h;
}
#define LTCHK(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { err = "cublasLt: " #x; goto done; } } while (0)
// descriptors, batch strides and pointers are set on every call; only the heuristic result is cached per shape
static const char* g_gemm2(const GParams& p, void* ws, size_t wsb, cudaStream_t s) {
  static std::map<std::tuple<int, int, int>, cublasLtMatmulHeuristicResult_t> cache;
  const char* err = nullptr;
  cublasLtHandle_t lt = g_lt();
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  const cublasOperation_t tn = CUBLAS_OP_N;
  const int32_t bc = p.B;
  const int64_t sa = (int64_t)p.K2 * p.HW, sb = (int64_t)p.C * p.K2, scd = (int64_t)p.C * p.HW;
  const float alpha = 1.f, beta = 1.f;
  auto key = std::make_tuple(p.B, p.C, p.HW);
  LTCHK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  LTCHK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  LTCHK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  LTCHK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, p.HW, p.K2, p.HW));
  LTCHK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, p.K2, p.C, p.K2));
  LTCHK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, p.HW, p.C, p.HW));
  LTCHK(cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &scd, sizeof(scd)));
  if (cache.find(key) == cache.end()) {
    cublasLtMatmulPreference_t pref = nullptr;
    const uint64_t wsl = wsb;
    cublasLtMatmulHeuristicResult_t res;
    int n = 0;
    LTCHK(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsl, sizeof(wsl));
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &res, &n);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || n == 0) { err = "cublasLt: no heuristic for GEMM2"; goto done; }
    cache[key] = res;
  }
  LTCHK(cublasLtMatmul(lt, op, &alpha, p.h, la, p.w2b, lb, &beta, p.x, lc, p.out, lc, &cache[key].algo, ws, wsb, s));
done:
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  if (op) cublasLtMatmulDescDestroy(op);
  return err;
}
// mask selects launches (15 = all four; lower values only for per-launch timing)
extern "C" const char* g_launch(GParams p, void* ltws, size_t ltws_bytes, int mask, void* stream_v) {
  cudaStream_t s = (cudaStream_t)stream_v;
  const int R = g_rows_per_tile(p.C, p.H, p.W);
  if (R == 0) return "unsupported shape";
  p.HW = p.H * p.W; p.C4 = 4 * p.C; p.K2 = p.C4 + 8; p.TPI = p.H / R; p.ntiles = p.B * p.TPI;
  const char* err = nullptr;
  if (mask & 1) {
    const int nfb = ((p.C4 / 16) * (p.C / 16) * 32 + 255) / 256, nub = (p.C4 + 7) / 8, ndb = (p.C * 56 + 255) / 256,
              nzb = (p.B * p.C4 + 255) / 256;
    g_prep<<<nfb + nub + ndb + nzb, 256, 0, s>>>(p, nfb, nub, ndb);
  }
  if (mask & 2) {
    if (p.C == 96 && p.W == 112) err = launch_p1<96, 2, 112, 7, 9, 3, true>(p, s);
    else if (p.C == 128 && p.W == 64) err = launch_p1<128, 2, 64, 8, 8, 2, false>(p, s);
    else if (p.C == 96 && p.W == 56) err = launch_s<96, 2, 56, 128, 3, false, false>(p, s);
    else if (p.C == 320 && p.W == 32) err = launch_s<320, 2, 32, 64, 2, true, false>(p, s);
    if (err) return err;
  }
  if (mask & 4) g_grn<<<dim3(p.B, (p.C + 15) / 16), 512, (p.C4 + 64) * 4, s>>>(p);
  if (mask & 8) { err = g_gemm2(p, ltws, ltws_bytes, s); if (err) return err; }
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
```

```cuda file=anchor.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, fp32 SIMT maths.
//  P1: depthwise 7x7 conv (+bias) -> xn [C][Mp] (K-major), per-(token, 32-channel chunk) LN partial sums;
//      zero GRN sums; transpose weights to [K][N]
//  P2: hidden[4C][Mp] = gelu(LN(xn) @ W1^T + b1); sumsq[b][k] += hidden^2 (per image)
//  P3: out = x + (hidden * (1 + gw*nf) + gb) @ W2^T + b2, written NCHW (split-K with atomics at small sizes)
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

// ---------------------------------------------------------------- phase 1
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

// MODE 1: patch tiles (TWR = TW in {4,1}); MODE 0: flattened token tiles (TWR = tokens per lane in {1,2,4})
template <int MODE, int TWR>
__device__ void phase1_conv(const Params& p, float* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nch = p.nch;
  float* st_s = smem;                     // [8][P1_SLOTS][2]
  float* wsm = smem + 8 * P1_SLOTS * 2;   // [32][49]
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg;
  const int rpt = lpr * TWR;  // slots per row (mode 1)

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

// ---------------------------------------------------------------- GEMM phases
// A = act [K][Mp] (K-major), B = Wt [K][N]. 8 warps = WM x WN spatial warps x KS K-groups. Lane grid 8 x 4.
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

    // per-thread token quad constants
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

    // ---- epilogue: partial tiles -> smem [KS][BN][CS_LD], then one token per thread
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
  if (p.split3 > 1) {  // initialise the output with residual + bias for the split-K atomics
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

// ---------------------------------------------------------------- host
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
  // phase-1 geometry
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
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <string>

// r0 anchor (cooperative fp32 kernel): every shape outside the lineage-G set
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

// lineage G (kernel.cu): K0 prep, fused pass 1, GRN scale, cuBLASLt batched GEMM2
struct GParams {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* w1f; float* uv; void* h; float* gsum; void* w2b; float* dwp;
  int B, C, H, W, HW, C4, K2, TPI, ntiles;
  float eps, ln_eps;
  unsigned long long* stamp;
};
extern "C" int g_rows_per_tile(int C, int H, int W);
extern "C" const char* g_launch(GParams p, void* ltws, size_t ltws_bytes, int mask, void* stream);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

static void run_g(torch::Tensor& x, torch::Tensor& dww, torch::Tensor& dwb, torch::Tensor& lnw, torch::Tensor& lnb,
                  torch::Tensor& w1, torch::Tensor& b1, torch::Tensor& gw, torch::Tensor& gb, torch::Tensor& w2,
                  torch::Tensor& b2, double eps, double leps, torch::Tensor& out, cudaStream_t stream) {
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  const int HW = H * W, C4 = 4 * C, K2 = C4 + 8;
  const size_t o_w1f = 0, o_uv = align256(o_w1f + (size_t)C4 * C * 2), o_dwp = align256(o_uv + (size_t)C4 * 8),
               o_gs = align256(o_dwp + (size_t)C * 56 * 4), o_w2b = align256(o_gs + (size_t)B * C4 * 4),
               o_h = align256(o_w2b + (size_t)B * C * K2 * 2), o_lt = align256(o_h + (size_t)B * K2 * HW * 2),
               ltb = (size_t)32 << 20, tot = o_lt + ltb;
  torch::Tensor ws = torch::empty({(long)tot}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();
  GParams p;
  p.x = x.data_ptr<float>(); p.dww = dww.data_ptr<float>(); p.dwb = dwb.data_ptr<float>(); p.lnw = lnw.data_ptr<float>();
  p.lnb = lnb.data_ptr<float>(); p.w1 = w1.data_ptr<float>(); p.b1 = b1.data_ptr<float>(); p.gw = gw.data_ptr<float>();
  p.gb = gb.data_ptr<float>(); p.w2 = w2.data_ptr<float>(); p.b2 = b2.data_ptr<float>(); p.out = out.data_ptr<float>();
  p.w1f = wp + o_w1f; p.uv = (float*)(wp + o_uv); p.dwp = (float*)(wp + o_dwp); p.gsum = (float*)(wp + o_gs);
  p.w2b = wp + o_w2b; p.h = wp + o_h;
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = 0; p.C4 = 0; p.K2 = 0; p.TPI = 0; p.ntiles = 0;
  p.eps = (float)eps; p.ln_eps = (float)leps;
  p.stamp = nullptr;
  const char* err = g_launch(p, wp + o_lt, ltb, 15, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("g_launch failed: ") + err);
}

static void run_anchor(torch::Tensor& x, torch::Tensor& dwconv_weight, torch::Tensor& dwconv_bias,
                       torch::Tensor& layernorm_weight, torch::Tensor& layernorm_bias, torch::Tensor& pwconv1_weight,
                       torch::Tensor& pwconv1_bias, torch::Tensor& grn_weight, torch::Tensor& grn_bias,
                       torch::Tensor& pwconv2_weight, torch::Tensor& pwconv2_bias, double eps, double layer_norm_eps,
                       torch::Tensor& output, cudaStream_t stream) {
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  TORCH_CHECK(C % 32 == 0, "dim must be a multiple of 32");
  TORCH_CHECK(H * W >= 32, "H*W must be >= 32");
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

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  if (g_rows_per_tile((int)x.size(1), (int)x.size(2), (int)x.size(3)) > 0)
    run_g(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias, grn_weight,
          grn_bias, pwconv2_weight, pwconv2_bias, eps, layer_norm_eps, output, stream);
  else
    run_anchor(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias, grn_weight,
               grn_bias, pwconv2_weight, pwconv2_bias, eps, layer_norm_eps, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (lineage G fused pass 1 + r0 fallback)"); }
```

```yaml design-card
id: r8-g-e2-singletile
parents:
  - r7-g-e1-fusedp1-mma
operation: specialisation
language: cuda_cpp
niche:
  lin: "G:E2:H37"
  mem: bulk1d
  st: direct
  grid: oneshot
  launch: split
  tile: rows2
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  The fused pass 1 (conv -> LN stats -> mma.sync GEMM1 -> LN fold + exact GELU + GRN sumsq + fp16 hidden) is now built
  for the two missing HW >= 1024 rows. Each CTA takes one 2-row tile and all 16 warps work through it in order.
  4,96 uses 112-token tiles padded to 128 (112 CTAs) with W1' resident in shared memory. 8,320 uses 64-token tiles
  (128 CTAs) and streams W1' in 64-row blocks of 40 KB, one buffer per 4-warp group, through a cp.async.bulk +
  mbarrier ring. Rented B200, same container as r7-merge, 16/16 pass: 4,96 37.0 vs 41.4 us (-10.7 %) and 8,320 67.2
  vs 73.1 us (-8.0 %); 64,96 and 32,128 within 0.2 % of the parent. Success lines (<= 36 / <= 55 us) are missed.
  H37 is not refuted: pass 1 at 8,320 is 43.3 us, under the 45 us line, with the W1' ring wait stamped at 0.34 us.
  What binds: the mma.sync pipe (16 warps saturate it, ~14.5 us of MMA at 8,320), an epilogue that does not overlap
  the MMA (8.4 us), and a latency-bound conv (9-13 us).
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -9
    confidence: medium
  L:
    pct: -7
    confidence: medium
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 210496
  threads_per_cta: 512
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 163840
  launches_per_call: 4
knobs:
  tile_4_96: "R=2, 112 tokens padded to 128, MG=3, W1' resident (73.7 KB bulk copy at entry), dw weights from global/L1 (DWS off), smem 101 KB"
  tile_8_320: "R=2, 64 tokens, 4 groups x 4 warps, MG=2 x 32 tokens per warp, W1' streamed 20 blocks x 40 KB (5 per group), smem 210 KB"
  k3: "__launch_bounds__(512,2), W2 row loaded into registers up front (MAXI=12)"
  gemm2: "cuBLASLt heuristic top-1 (algo sweep: top-1 best or tied at both rows)"
dispatch:
  - {shape: "64,96,112,112", kernel: "g_pass1<96,2,112,7,9,3,true> + g_grn + cublasLt (unchanged from r7)"}
  - {shape: "32,128,64,64", kernel: "g_pass1<128,2,64,8,8,2,false> + g_grn + cublasLt (unchanged from r7)"}
  - {shape: "4,96,56,56", kernel: "g_p1s<96,2,56,128,3,false,false> + g_grn + cublasLt", meta: {ctas: 112}}
  - {shape: "8,320,32,32", kernel: "g_p1s<320,2,32,64,2,true,false> + g_grn + cublasLt", meta: {ctas: 128}}
  - {shape: "all others", kernel: "r0 anchor convnext_fused (fallback only; do not merge)"}
tests:
  - H37
findings:
  - "run_tests 16/16 PASS (portal-order re-runs included), rented B200, this vs r7-merge in the same container (us): 4,96 37.0 vs 41.4 (-10.7 %); 8,320 67.2 vs 73.1 (-8.0 %); 64,96 867.9 vs 866.1 (+0.2 %); 32,128 227.6 vs 227.8 (-0.1 %). The other 12 rows run the r0 anchor (merge-only candidate; the emulator's 0.44 is meaningless). Timed once in run_tests plus 2-4 harness_time repeats per variant in probes, not interleaved x3 against the merge [run_tests]"
  - "H37 verdict: success line NOT met (4,96 37.0 > 36; 8,320 67.2 > 55); refutation clauses do not fire (4,96 112-tile path is 4.4 us faster than the merge path in-container; 8,320 pass 1 = 43.3 us <= 45 with the ring wait stamped at 0.34 us of 40 us). The fused single-tile structure transfers to C=320 but with mma.sync it lands at ~-8 %, not the -30 % the arithmetic assumed [run_tests, probe_b200]"
  - "Per-launch, harness_time cumulative (K0 / +P1 / +K3 / +G2, us), shipped config: 4,96 3.5 / 25.1 / 30.7 / 37.3; 8,320 4.5 / 47.8 / 56.7 / 68.0. First build (dw weights staged in smem, old K3): 4,96 3.5 / 27.4 / 32.5 / 39.1; 8,320 4.5 / 48.7 / 59.8 / 71.1 [probe_b200]"
  - "Per-tile stamps (globaltimer, flush, CTA median/max from kernel start; tid 0 / warp 0), 4,96 shipped: prologue 0.5, conv end 9.2/10.2, stats end 10.0/11.0, GEMM end 18.7/19.5 us; warp 0 GEMM side mma 3.3 us + epilogue 4.5 us for 2 warp tiles; W1' wait 0.05 us. 8,320: conv end 12.8/13.8, stats 14.1/14.9, end 39.7/41.0; warp 0 ring wait 0.34, mma 16.2, epilogue 8.4 us over 5 blocks (clock 1.78-1.88 GHz) [probe_b200]"
  - "mma.sync is pipe-bound at 16 warps: 800 mma per warp in 16.2 us = 9 clk per mma per scheduler vs the 8-clk peak (2048 FLOP/clk/SM). With 8 warps (ping-pong halves) the same work takes 11.7 us per half, i.e. 13 clk per mma per scheduler (latency-bound), so alternating mma and epilogue between warp halves gave only 1 us at 8,320 and 1.2 us at 4,96 in the stamps and nothing in harness time. Without tcgen05 the GEMM side of 8,320 cannot go below ~14.5 us MMA + the non-overlapped epilogue [probe_b200]"
  - "Dead end: cp.async.bulk.prefetch.L2 of the whole x halo footprint (one per channel, 1-1.8 KB) at kernel entry did not move the conv (11.26 vs 11.26 us at 4,96, 12.8 vs 13.3 at 8,320). The conv is not DRAM-latency-bound; it is the serial 3-row prefetch chain per item (2-3 rounds x 1.3-2.5 items per thread) plus FMA issue (8,320: 1M FMA per CTA = ~4.4 us at 128/clk) [probe_b200]"
  - "Fix kept: staging the 21.5 KB dw weights into smem with a plain load/store loop serialised (~11 dependent L2 rounds, prologue 3.3 us); reading them via __ldg from the K0-padded [C][7][8] buffer instead cut the 4,96 prologue to 0.5 us and the row by 2.5 us (40.0 -> 37.5) [probe_b200]"
  - "Fix kept: K3 at 8,320 took 11-12 us because grid (8,20) = 160 blocks of 512 threads ran as 2 waves (1 block/SM). __launch_bounds__(512,2) with the W2 row loaded up front puts it in one wave: 70.9 -> 67.9 us at 8,320, no change at 4,96 (24 blocks). Same arithmetic order as r7, so 64,96 / 32,128 are unaffected (run_tests within 0.2 %) [probe_b200]"
  - "cuBLASLt GEMM2 algo sweep (8 heuristic results, cold, us): 4,96 6.6 / 8.3 / 6.7 / 6.6 / 6.5 / 9.0 / 7.2 / 9.4 (top-1 tied best); 8,320 11.1 / 12.6 / 12.8 / 18.0 / 12.4 / 15.7 / 23.5 / 20.2 (top-1 best). At 8,320 it moves 42 MB in 11.1 us (~3.8 TB/s), below the 6.6 TB/s it reaches at HW >= 4096 [probe_b200]"
  - "PRECISION LOG (fp64 reference, harness generator rules: randn, weight matrices / sqrt(shape[-1]), LN weight 1 / bias 0, biases randn; torch.manual_seed(200), 10 rounds per row; first r8 build, whose arithmetic equals the shipped one: the shipped changes only move where the dw weights are read from and prefetch the W2 row in K3, same summation order): 4,96 max|err| 1.978e-3 vs atol 2.8e-3 (71 %), 0 beyond 2.4e-3+1e-5|ref|; 8,320 1.907e-3, 0 beyond 2.4e-3+1e-5|ref| (the smallest atol of any row, so 0 beyond its own atol; <= 79 % of it). Single-seed checks on the shipped variants: 4,96 2.01e-3, 8,320 1.90e-3 [probe_b200]"
  - "Resources (compile_b200, nvcc 13 sm_100a): g_p1s<320> 128 regs, 0 spill, 210496 B dynamic smem, 6 UBLKCP; g_p1s<96> 128 regs, 0 spill, 101 KB smem (DWS off); g_pass1 unchanged 122/128 regs; g_grn now __launch_bounds__(512,2) (<= 64 regs) [compile_b200]"
  - "Merge recommendation: take 4,96 and 8,320 from this kernel (4 launches each vs A's 5-6, so the portal ratio should be no worse than A's 1.23-1.30); keep 64,96 / 32,128 from r7 G (identical code). Rough portal: 4,96 37 x 1.27 = 47 us vs 57.4; 8,320 67 x 1.27 = 85 us vs 99.7 [INFERRED]"
  - "Next levers (in order): (1) tcgen05 GEMM1 for g_p1s: removes the 14.5 us mma.sync floor at 8,320 and lets the epilogue warps overlap the MMA (the r7 lever, now quantified); (2) conv: stage x per channel with cp.async.bulk (4,96 needs 172 KB, so two 86 KB halves after the W1' copy; 8,320 cannot fit 327 KB beside a 160 KB W1' ring); (3) GEMM2 at 8,320 runs at ~3.8 TB/s, so a custom swap-AB or a different layout is worth ~4 us; (4) the stretch STEP 4 (balanced erf, I30) was not run: stamps show the epilogue is per-warp latency/issue-bound, not MUFU-bound, at these rows [INFERRED]"
  - "Pitfalls: harness_time's CUPTI sequence check failed 3x in a row on one variant (dropped kernel); wrap it in a 6-try retry. Probe extension builds of kernel.cu + cuBLASLt take 12-21 s [probe_b200]"
paths:
  - {max_tokens: 8192, lang: cuda, width: 128, threads: 512, rows: 2, grid: oneshot, mem: tma, launches: 4, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: 12544, lang: cuda, width: 128, threads: 512, rows: 2, grid: oneshot, mem: tma, launches: 4, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: 131072, lang: cuda, width: 128, threads: 512, rows: 2, grid: persistent, mem: ldg, launches: 4, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: null, lang: cuda, width: 128, threads: 512, rows: 2, grid: persistent, mem: ldg, launches: 4, x: none, w: none, st: none, cluster: 1}
runs_on:
  H200:
    runs: false
    representative: false
    note: "built for sm_100a only (target_hardware B200)"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "Merge-only candidate: 12 rows run the slow r0 anchor; the merge must take only 4,96 / 8,320 (new) and 64,96 / 32,128 (r7 G code) from it"
  - "4 launches per call; at 4,96 (37 us) launch gaps matter on the portal more than on the bench, though A's path has 5-6"
  - "fp32 red.add GRN sums are order-nondeterministic (relative 1e-7)"
  - "cuBLASLt heuristic cached per (B,C,HW); descriptors, strides and pointers rebuilt every call"
  - "g_p1s<320> uses 210.5 KB dynamic smem; the attribute is set once per instantiation and never lowered (portal-order check passed)"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```