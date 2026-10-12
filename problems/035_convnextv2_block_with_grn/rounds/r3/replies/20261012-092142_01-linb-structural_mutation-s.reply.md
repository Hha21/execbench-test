**H24 is refuted.** Single bulk copies are faster than the parent's loads in isolation, but that doesn't make the kernel faster. The final build is the parent with bulk-row copies for GEMM2's A/B tiles on the two rows that already used the flat path (1,160,449,7 and 4,96,56,56). It passes 16/16 on the rented B200. Its times match the parent within noise, and the projected portal score is 0.5625 (the same-build control projects 0.5643).

- **Microbench (296 CTAs, 2 per SM, ~1.9 GHz):** after a grid barrier, one 32 KB bulk copy of lines just written by other CTAs takes 1591 clk (0.83 µs) median. The parent-style 16-B `cp.async` with div/mod takes 2607 clk (1.36 µs), so the bulk-rebuild branch applied. From cold DRAM, every load method takes 2.6–3.5 µs. The barrier itself costs 1.6–1.9 µs, plus 400–640 clk when stores are pending before it. The 2–4 µs "post-sync round" seen in r2 is mostly barrier release plus cold-DRAM latency; copy issue rate is a smaller part.
- **In the kernel:** I ran the parent's dispatch with and without bulk rows as a same-build control in run_tests. At 1,192 the flat path with bulk rows in both GEMMs took 31.4 µs against 29.5 µs for the ring kernel, so 1,192 stays on ring. GEMM2 bulk rows moved 1,160 from 54.0 to 53.4–53.6 µs and 4,96 from 58.4 to 57.8–58.7 µs. A row whose code did not change moved 1.5 µs between runs, so these gains are noise.
- **Success criteria missed:** the targets were 1,192 ≤ 22 µs and 2,96 ≤ 32 µs; it measured 29.7 and 42.0. The dataflow branch (per-image counters) was not run.
- **Not done:** the per-row 10-round fp64 error log. Bulk copies move the same bytes as the old loads, and the max error against an fp32 reference was identical old vs new on 7 rows for one seed.

### Rationale
E2 asked whether the 2–4 µs post-sync rounds in lineage B come from copy issue rate or from post-barrier latency. The microbench answers it. The cost is the barrier (1.6–1.9 µs) plus cold-DRAM latency (2.6–3.5 µs for any method). Per-row bulk copies help only where a CTA gets a single small tile, and there the gain is within run-to-run noise. The candidate keeps the parent's dispatch and adds GEMM2 bulk rows on the two flat-path rows. It is a negative result, recorded with numbers: B's "S rows ≤ 20 µs" thesis needs fewer barriers (H10 dataflow) or should be retired.

```json solution-spec
{"name": "r3-coop-flat-bulkrows", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, FP16 mma.sync GEMMs (fp32 accumulate).
// Two cooperative kernels, chosen per shape on the host:
//  convnext_flat (C <= 192, HW >= 128, shape classes in flat_choice): P1 weight conversion + coalesced dwconv CTA units
//   (xh token-major fp16, LN partials, out = x); P2 one full-K GEMM1 tile per CTA; P3 one split-K GEMM2 unit per CTA
//   with red.add into out. r3 (E2/H24): the post-sync A/B tiles can arrive as one cp.async.bulk per smem row on a single
//   mbarrier (FL_BULK1 for GEMM1, FL_BULK2 for GEMM2) instead of 16-byte cp.async with div/mod index maths.
//  convnext_k (all other shapes): the r1 ring kernel, unchanged.
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
#define RES_OFF (SMEM_BYTES - 4096)  // flat path: top 4 KB reserved (mbarrier)
#define PLANE_FLOATS 676
#define P1_SLOTS 512
#define FLAT_FOFF 5120
#define FL_BULK1 8
#define FL_BULK2 64

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
__device__ __forceinline__ void mbar_init(unsigned a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\nfence.mbarrier_init.release.cluster;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned a, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned a, unsigned ph) {
  asm volatile("{ .reg .pred p; LW%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra LW%=; }" ::"r"(a), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk_g2s(unsigned dst, const void* src, unsigned bytes, unsigned mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
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
  constexpr int A_STAGE = BKc * AS_LD, B_STAGE = BN * BS_LD, STAGE_H = A_STAGE + B_STAGE;
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
    for (int i = tid; i < BN; i += NT) {
      const int col = n0 + i;
      if (PHASE == 2) { cs1_s[i] = col < N ? __ldcg(p.s1 + col) : 0.f; cc_s[i] = col < N ? __ldcg(p.c1 + col) : 0.f; }
      else cc_s[i] = col < N ? __ldcg(p.c2 + col) : 0.f;
    }
    int ia_l = 0, ib_l = 0, bmask = 0; float inva = 0.f, invb = 0.f;
    const float* sqa = p.sumsq; const float* sqb = p.sumsq;
    if constexpr (PHASE == 2) {
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

__global__ void __launch_bounds__(NT, 2) convnext_k(Params p) {
  extern __shared__ float4 smem4[];
  float* smem = reinterpret_cast<float*>(smem4);
  float* aux = smem + RING_BYTES / 4;
  float* misc = aux + AUX_FLOATS;
  for (int i = threadIdx.x; i < MAXIMG * 128; i += NT) misc[i] = 0.f;
  __syncthreads();
  phase1(p, smem, aux);
  cg::this_grid().sync();
  if (p.split3 > 1) {  // residual + c2 init for the split-K atomics of phase 3
    const long total = (long)p.M * p.C;
    for (long i = (long)blockIdx.x * NT + threadIdx.x; i < total; i += (long)gridDim.x * NT) {
      const int c = (int)((i / p.HW) % p.C);
      p.out[i] = __ldg(p.x + i) + __ldcg(p.c2 + c);
    }
  }
  run_gemm<2>(p, p.cfg2, 1, smem, aux, misc);
  cg::this_grid().sync();
  run_gemm<3>(p, p.cfg3, p.split3, smem, aux, misc);
}

// ---------------------------------------------------------------- flat path (C <= 192, HW >= 128)
// the 7 input rows of one lane's 4-column segment: v[i][k] = x[h - 3 + i][w0 - 4 + k] (zero outside)
template <bool VEC>
__device__ __forceinline__ void conv_load(const float* xc, int H, int W, int h, int w0, bool act, float (&v)[7][12]) {
  if (VEC) {
#pragma unroll
    for (int i = 0; i < 7; i++) {
      const int hh = h - 3 + i;
      const bool ok = act && (unsigned)hh < (unsigned)H;
      const float* rp = xc + hh * W + w0;
      float4 L = make_float4(0.f, 0.f, 0.f, 0.f), Cq = L, R = L;
      if (ok) {
        if (w0 >= 4) L = __ldg(reinterpret_cast<const float4*>(rp - 4));
        Cq = __ldg(reinterpret_cast<const float4*>(rp));
        if (w0 + 4 < W) R = __ldg(reinterpret_cast<const float4*>(rp + 4));
      }
      v[i][0] = L.x; v[i][1] = L.y; v[i][2] = L.z; v[i][3] = L.w; v[i][4] = Cq.x; v[i][5] = Cq.y;
      v[i][6] = Cq.z; v[i][7] = Cq.w; v[i][8] = R.x; v[i][9] = R.y; v[i][10] = R.z; v[i][11] = R.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 7; i++) {
      const int hh = h - 3 + i;
      const bool ok = act && (unsigned)hh < (unsigned)H;
#pragma unroll
      for (int k = 1; k < 11; k++) {
        const int ww = w0 - 4 + k;
        v[i][k] = (ok && ww >= 0 && ww < W) ? __ldg(xc + hh * W + ww) : 0.f;
      }
      v[i][0] = 0.f; v[i][11] = 0.f;
    }
  }
}

// dwconv CTA unit: 32 channels (warp = 4 channels, lane >> 3) x 8 positions (lane & 7) = R rows x SEGW 4-column
// segments; lanes run along w so each row load is coalesced. LN partials reduced over the 32 channels in smem.
template <bool VEC>
__device__ __forceinline__ void conv_cta_unit(const Params& p, const float* __restrict__ dws, float* red, int u, int SEGW, int R) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW;
  const int nws = (W + 3) >> 2, nsb = (nws + SEGW - 1) / SEGW, nrb = (H + R - 1) / R, nch = C >> 5;
  const int sb = u % nsb; u /= nsb;
  const int rb = u % nrb; u /= nrb;
  const int ch = u % nch, b = u / nch;
  const int cl = lane >> 3, pos = lane & 7, rr = pos / SEGW, sg = pos - rr * SEGW;
  const int c = ch * 32 + warp * 4 + cl, h = rb * R + rr, w0 = (sb * SEGW + sg) * 4;
  const bool act = (rr < R) && (h < H) && (w0 < W);
  const float* xc = p.x + ((size_t)b * C + c) * HW;
  float v[7][12];
  conv_load<VEC>(xc, H, W, h, w0, act, v);
  const float bias = __ldg(p.dwb + c);
  float o[4] = {bias, bias, bias, bias};
  const float* wr = dws + c * 49;
#pragma unroll
  for (int i = 0; i < 7; i++)
#pragma unroll
    for (int j = 0; j < 7; j++) {
      const float wv = wr[i * 7 + j];
#pragma unroll
      for (int e = 0; e < 4; e++) o[e] = fmaf(v[i][e + 1 + j], wv, o[e]);
    }
  const int t0 = b * HW + h * W + w0;
  if (act) {
    // residual: out = x (GEMM2 adds y and c2 with red.add)
    float* op = p.out + ((size_t)b * C + c) * HW + h * W + w0;
    __half* xh = reinterpret_cast<__half*>(p.xh);
    if (VEC) *reinterpret_cast<float4*>(op) = make_float4(v[3][4], v[3][5], v[3][6], v[3][7]);
#pragma unroll
    for (int e = 0; e < 4; e++)
      if (w0 + e < W) {
        if (!VEC) op[e] = v[3][4 + e];
        xh[(size_t)(t0 + e) * C + c] = __float2half_rn(o[e]);
      }
  }
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const bool ok = act && (w0 + e < W);
    float s = ok ? o[e] : 0.f, q = s * s;
    s += __shfl_xor_sync(0xffffffffu, s, 8); s += __shfl_xor_sync(0xffffffffu, s, 16);
    q += __shfl_xor_sync(0xffffffffu, q, 8); q += __shfl_xor_sync(0xffffffffu, q, 16);
    if (cl == 0) { red[(warp * 32 + pos * 4 + e) * 2] = s; red[(warp * 32 + pos * 4 + e) * 2 + 1] = q; }
  }
  __syncthreads();
  if (tid < 32) {
    const int ps = tid >> 2, e = tid & 3, r2 = ps / SEGW, s2 = ps - r2 * SEGW;
    const int hh = rb * R + r2, ww = (sb * SEGW + s2) * 4 + e;
    if (r2 < R && hh < H && ww < W) {
      float S = 0.f, Q = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) { S += red[(w * 32 + tid) * 2]; Q += red[(w * 32 + tid) * 2 + 1]; }
      *reinterpret_cast<float2*>(p.part + ((size_t)ch * p.Mp + b * HW + hh * W + ww) * 2) = make_float2(S, Q);
    }
  }
  __syncthreads();
}

// A [rows][lda] and B [rows][ldb] fp16 in smem (k contiguous), full K per call; warp tile MI*16 x NI*8
template <int MI, int NI>
__device__ __forceinline__ void flat_mma(float (&acc)[MI][NI][4], unsigned a_addr, unsigned b_addr, unsigned b2_addr,
                                         int lda2, int ldb2, int nk16) {
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NI; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
#pragma unroll 2
  for (int ks = 0; ks < nk16; ks++) {
    unsigned bf[NI][2];
#pragma unroll
    for (int ni = 0; ni + 1 < NI; ni += 2)
      ldsm_x4(b_addr + ni * 8 * ldb2 + ks * 32, bf[ni][0], bf[ni][1], bf[ni + 1][0], bf[ni + 1][1]);
    if constexpr ((NI & 1) != 0) ldsm_x2(b2_addr + ks * 32, bf[NI - 1][0], bf[NI - 1][1]);
#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
      unsigned af[4];
      ldsm_x4(a_addr + mi * 16 * lda2 + ks * 32, af[0], af[1], af[2], af[3]);
#pragma unroll
      for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af, bf[ni][0], bf[ni][1]);
    }
  }
}

// warp 0 issues one cp.async.bulk per smem row: rows [0, BMr) from a (stride lda), [BMr, rows) from b (stride ldb);
// the B rows follow the A rows in smem (row pitch ld2 bytes). One mbarrier transaction covers the whole tile.
__device__ __forceinline__ void bulk_tile(unsigned dst, int ld2, const __half* a, size_t lda, const __half* b, size_t ldb,
                                          int BMr, int rows, int bytes, unsigned mbar) {
  const int lane = threadIdx.x & 31;
  if (lane == 0) { asm volatile("fence.proxy.async.global;" ::: "memory"); mbar_expect(mbar, (unsigned)(rows * bytes)); }
  __syncwarp();
  for (int r = lane; r < rows; r += 32)
    bulk_g2s(dst + r * ld2, r < BMr ? a + (size_t)r * lda : b + (size_t)(r - BMr) * ldb, bytes, mbar);
}

// GEMM1: hidden = gelu(rstd*(xh @ W1'^T - mean*s1) + c1), sumsq[b][n] += hidden^2. One round of loads per tile.
// smem: [0, FLAT_FOFF) per-row/column floats; then A/B tiles, reused for the fp32 accumulator tile Cs.
template <int BM, int BN, int WM, int WN>
__device__ void flat_gemm1(const Params& p, char* sm, bool nb, unsigned mbar, unsigned& ph) {
  constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8, NSPLIT = NT / BM;
  constexpr int CLD = BN + 4, NG = BN / 8, RS = NT / NG;
  static_assert(WM * WN == 8 && WTM % 16 == 0 && WTN % 8 == 0, "tile");
  static_assert((2 * 128 + 4 * 128 + 2 * NT) * 4 <= FLAT_FOFF, "float area");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp - (warp / WN) * WN;
  const int C = p.C, C4 = p.C4, M = p.M, Mp = p.Mp, HW = p.HW, K = C, N = C4, LD = K + 8, K8 = K >> 3;
  float* mu_s = reinterpret_cast<float*>(sm);
  float* rs_s = mu_s + BM; float* s1_s = rs_s + BM; float* c1_s = s1_s + BN; float* sq_s = c1_s + BN; float* red = sq_s + 2 * BN;
  __half* As = reinterpret_cast<__half*>(sm + FLAT_FOFF);
  __half* Bs = As + BM * LD;
  float* Cs = reinterpret_cast<float*>(sm + FLAT_FOFF);
  const __half* xh = reinterpret_cast<const __half*>(p.xh);
  const __half* w1h = reinterpret_cast<const __half*>(p.w1h);
  __half* hid = reinterpret_cast<__half*>(p.hidden);
  const int mt = (M + BM - 1) / BM, nt = N / BN, nch = C >> 5;
  const unsigned as = smem_u32(As), bs = smem_u32(Bs);
  const int gid = lane >> 2, tig = lane & 3;
  const unsigned a_addr = as + ((wm * WTM + (lane & 15)) * LD + (lane >> 4) * 8) * 2;
  const unsigned b_addr = bs + ((wn * WTN + (lane & 7) + ((lane >> 4) << 3)) * LD + ((lane >> 3) & 1) * 8) * 2;
  const unsigned b2_addr = bs + ((wn * WTN + (NI - 1) * 8 + (lane & 7)) * LD + ((lane >> 3) & 1) * 8) * 2;
  for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
    const int nti = tile % nt, mti = tile / nt, m0 = mti * BM, n0 = nti * BN;
    if (nb) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (warp == 0) bulk_tile(as, LD * 2, xh + (size_t)m0 * C, C, w1h + (size_t)n0 * C, C, BM, BM + BN, K * 2, mbar);
    } else {
      for (int i = tid; i < BM * K8; i += NT) {
        const int r = i / K8, k8 = i - r * K8;
        cp_async16(as + (r * LD + k8 * 8) * 2, xh + (size_t)(m0 + r) * C + k8 * 8, 16);
      }
      for (int i = tid; i < BN * K8; i += NT) {
        const int r = i / K8, k8 = i - r * K8;
        cp_async16(bs + (r * LD + k8 * 8) * 2, w1h + (size_t)(n0 + r) * C + k8 * 8, 16);
      }
      cp_async_commit();
    }
    for (int i = tid; i < BN; i += NT) { s1_s[i] = __ldcg(p.s1 + n0 + i); c1_s[i] = __ldcg(p.c1 + n0 + i); }
    for (int i = tid; i < 2 * BN; i += NT) sq_s[i] = 0.f;
    {
      const int tok = tid % BM, js = tid / BM;
      const int t = min(m0 + tok, M - 1);
      float S1 = 0.f, S2 = 0.f;
      for (int ch = js; ch < nch; ch += NSPLIT) {
        const float2 pp = __ldcg(reinterpret_cast<const float2*>(p.part + ((size_t)ch * Mp + t) * 2));
        S1 += pp.x; S2 += pp.y;
      }
      red[tid * 2] = S1; red[tid * 2 + 1] = S2;
    }
    __syncthreads();
    if (tid < BM) {
      float S1 = 0.f, S2 = 0.f;
#pragma unroll
      for (int js = 0; js < NSPLIT; js++) { S1 += red[(js * BM + tid) * 2]; S2 += red[(js * BM + tid) * 2 + 1]; }
      const float mean = S1 / (float)C;
      const float var = fmaxf(S2 / (float)C - mean * mean, 0.f);
      mu_s[tid] = mean; rs_s[tid] = rsqrtf(var + p.ln_eps);
    }
    if (nb) { mbar_wait(mbar, ph); ph ^= 1u; } else cp_async_wait<0>();
    __syncthreads();
    float acc[MI][NI][4];
    flat_mma<MI, NI>(acc, a_addr, b_addr, b2_addr, LD * 2, LD * 2, K >> 4);
    __syncthreads();  // every warp is done with As/Bs before Cs overwrites them
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int ni = 0; ni < NI; ni++) {
        const int r = wm * WTM + mi * 16 + gid, c = wn * WTN + ni * 8 + 2 * tig;
        *reinterpret_cast<float2*>(Cs + r * CLD + c) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<float2*>(Cs + (r + 8) * CLD + c) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
      }
    __syncthreads();
    // looped epilogue: thread = 8 consecutive columns of rows rr0, rr0 + RS, ...; 16-byte hidden stores
    const int ia = m0 / HW, bnd = (ia + 1) * HW;
    const bool two = min(m0 + BM, M) > bnd;  // CTA-uniform: tile spans two images (HW >= BM)
    const int g = tid % NG, rr0 = tid / NG;
    if (rr0 < RS) {
      float s1v[8], c1v[8], qa[8], qb[8];
#pragma unroll
      for (int e = 0; e < 8; e++) { s1v[e] = s1_s[g * 8 + e]; c1v[e] = c1_s[g * 8 + e]; qa[e] = 0.f; qb[e] = 0.f; }
#pragma unroll 1
      for (int r = rr0; r < BM; r += RS) {
        const int t = m0 + r;
        if (t >= M) break;
        const float mu = mu_s[r], rs = rs_s[r];
        const float4 x0 = *reinterpret_cast<const float4*>(Cs + r * CLD + g * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(Cs + r * CLD + g * 8 + 4);
        float v[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int e = 0; e < 8; e++) v[e] = gelu_f(fmaf(rs, v[e] - mu * s1v[e], c1v[e]));
        *reinterpret_cast<uint4*>(hid + (size_t)t * C4 + n0 + g * 8) =
            make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
        if (t < bnd) {
#pragma unroll
          for (int e = 0; e < 8; e++) qa[e] += v[e] * v[e];
        } else {
#pragma unroll
          for (int e = 0; e < 8; e++) qb[e] += v[e] * v[e];
        }
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {
        atomicAdd(&sq_s[g * 8 + e], qa[e]);
        if (two) atomicAdd(&sq_s[BN + g * 8 + e], qb[e]);
      }
    }
    __syncthreads();
    for (int i = tid; i < BN; i += NT) {
      atomicAdd(p.sumsq + (size_t)ia * C4 + n0 + i, sq_s[i]);
      if (two) atomicAdd(p.sumsq + (size_t)(ia + 1) * C4 + n0 + i, sq_s[BN + i]);
    }
    __syncthreads();
  }
}

// GEMM2 split-K unit: out += (hidden * s_b)[:, kslice] @ W2h[:, kslice]^T (+ c2 for slice 0). One round of loads.
template <int BM, int BN, int WM, int WN>
__device__ void flat_gemm2(const Params& p, char* sm, int split, bool nb, unsigned mbar, unsigned& ph) {
  constexpr int WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
  static_assert(WM * WN == 8 && WTM % 16 == 0 && WTN % 8 == 0, "tile");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WN, wn = warp - (warp / WN) * WN;
  const int C = p.C, C4 = p.C4, M = p.M, HW = p.HW, N = C, K = C4, KSL = K / split, LD = KSL + 8, K8 = KSL >> 3;
  __half* As = reinterpret_cast<__half*>(sm);
  __half* Bs = As + BM * LD;
  float* sq_s = reinterpret_cast<float*>(Bs + BN * LD);  // [2][C4] sqrt(sumsq)
  float* sc_s = sq_s + 2 * C4;                           // [2][KSL] GRN scale
  float* c2_s = sc_s + 2 * KSL;                          // [BN]
  float* red = c2_s + BN;                                // [16] + inv [2]
  const __half* hid = reinterpret_cast<const __half*>(p.hidden);
  const __half* w2h = reinterpret_cast<const __half*>(p.w2h);
  const int mt = (M + BM - 1) / BM, nt = N / BN, Q = C4 >> 2;
  const unsigned as = smem_u32(As), bs = smem_u32(Bs);
  const int gid = lane >> 2, tig = lane & 3;
  const unsigned a_addr = as + ((wm * WTM + (lane & 15)) * LD + (lane >> 4) * 8) * 2;
  const unsigned b_addr = bs + ((wn * WTN + (lane & 7) + ((lane >> 4) << 3)) * LD + ((lane >> 3) & 1) * 8) * 2;
  const unsigned b2_addr = bs + ((wn * WTN + (NI - 1) * 8 + (lane & 7)) * LD + ((lane >> 3) & 1) * 8) * 2;
  for (int unit = blockIdx.x; unit < mt * nt * split; unit += gridDim.x) {
    const int nti = unit % nt, u2 = unit / nt, s = u2 % split, mti = u2 / split;
    const int m0 = mti * BM, n0 = nti * BN, k0 = s * KSL;
    if (nb) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (warp == 0) bulk_tile(as, LD * 2, hid + (size_t)m0 * C4 + k0, C4, w2h + (size_t)n0 * C4 + k0, C4, BM, BM + BN, KSL * 2, mbar);
    } else {
      for (int i = tid; i < BM * K8; i += NT) {
        const int r = i / K8, k8 = i - r * K8;
        cp_async16(as + (r * LD + k8 * 8) * 2, hid + (size_t)(m0 + r) * C4 + k0 + k8 * 8, 16);
      }
      for (int i = tid; i < BN * K8; i += NT) {
        const int r = i / K8, k8 = i - r * K8;
        cp_async16(bs + (r * LD + k8 * 8) * 2, w2h + (size_t)(n0 + r) * C4 + k0 + k8 * 8, 16);
      }
      cp_async_commit();
    }
    const int ia = m0 / HW, ib = (min(m0 + BM, M) - 1) / HW;
    float gwr[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int i = tid + q * NT;
      gwr[q] = (i < 2 * KSL) ? __ldg(p.gw + k0 + (i >= KSL ? i - KSL : i)) : 0.f;
    }
    for (int i = tid; i < BN; i += NT) c2_s[i] = (s == 0) ? __ldcg(p.c2 + n0 + i) : 0.f;
    float a0 = 0.f, a1 = 0.f;
    for (int i = tid; i < 2 * Q; i += NT) {
      const int j = i >= Q ? 1 : 0, q = i - j * Q;
      const float4 v = __ldcg(reinterpret_cast<const float4*>(p.sumsq + (size_t)(j ? ib : ia) * C4) + q);
      const float4 r = make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
      *reinterpret_cast<float4*>(sq_s + j * C4 + 4 * q) = r;
      const float sm4 = r.x + r.y + r.z + r.w;
      if (j) a1 += sm4; else a0 += sm4;
    }
    a0 = warp_sum(a0); a1 = warp_sum(a1);
    if (lane == 0) { red[warp * 2] = a0; red[warp * 2 + 1] = a1; }
    __syncthreads();
    if (tid < 2) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) tot += red[w * 2 + tid];
      red[16 + tid] = 1.f / (tot / (float)C4 + p.eps);
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int i = tid + q * NT;
      if (i < 2 * KSL) {
        const int j = i >= KSL ? 1 : 0, kk = i - j * KSL;
        sc_s[i] = fmaf(gwr[q] * red[16 + j], sq_s[j * C4 + k0 + kk], 1.f);
      }
    }
    if (nb) { mbar_wait(mbar, ph); ph ^= 1u; } else cp_async_wait<0>();
    __syncthreads();
    const int bnd = (ia + 1) * HW;
    for (int i = tid; i < BM * K8; i += NT) {
      const int r = i / K8, k8 = i - r * K8;
      const float* sc = sc_s + ((m0 + r >= bnd) ? KSL : 0) + k8 * 8;
      uint4* ptr = reinterpret_cast<uint4*>(As + r * LD + k8 * 8);
      const uint4 v = *ptr;
      const unsigned w[4] = {v.x, v.y, v.z, v.w};
      unsigned o[4];
#pragma unroll
      for (int j = 0; j < 4; j++) { const float2 f = unpack2(w[j]); o[j] = pack2(f.x * sc[2 * j], f.y * sc[2 * j + 1]); }
      *ptr = make_uint4(o[0], o[1], o[2], o[3]);
    }
    __syncthreads();
    float acc[MI][NI][4];
    flat_mma<MI, NI>(acc, a_addr, b_addr, b2_addr, LD * 2, LD * 2, KSL >> 4);
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = wm * WTM + mi * 16 + gid + hh * 8, t = m0 + r;
        if (t < M) {
          const int b = t / HW, hw = t - b * HW;
          float* ob = p.out + (size_t)b * C * HW + hw;
#pragma unroll
          for (int ni = 0; ni < NI; ni++) {
            const int c = wn * WTN + ni * 8 + 2 * tig;
            atomicAdd(ob + (size_t)(n0 + c) * HW, acc[mi][ni][2 * hh] + c2_s[c]);
            atomicAdd(ob + (size_t)(n0 + c + 1) * HW, acc[mi][ni][2 * hh + 1] + c2_s[c + 1]);
          }
        }
      }
    __syncthreads();
  }
}

__global__ void __launch_bounds__(NT, 2) convnext_flat(Params p) {
  extern __shared__ float4 smem4[];
  char* sm = reinterpret_cast<char*>(smem4);
  const int tid = threadIdx.x, warp = tid >> 5, G = gridDim.x;
  const bool nb1 = (p.mask & FL_BULK1) != 0, nb2 = (p.mask & FL_BULK2) != 0;
  float* dws = reinterpret_cast<float*>(sm);
  float* cred = dws + p.C * 49;  // [8][32][2] conv LN partials
  const unsigned mbar = smem_u32(sm + RES_OFF + 3072);
  unsigned ph = 0;
  if ((nb1 || nb2) && tid == 0) mbar_init(mbar, 1);
  {
    const unsigned d = smem_u32(dws);
    const int n16 = p.C * 49 / 4;
    for (int i = tid; i < n16; i += NT) cp_async16(d + i * 16, p.dww + i * 4, 16);
    cp_async_commit();
  }
  for (int i = blockIdx.x * NT + tid; i < p.B * p.C4; i += G * NT) p.sumsq[i] = 0.f;
  weights_pass(p, warp * G + blockIdx.x, G * 8);
  cp_async_wait<0>();
  __syncthreads();
  {
    const int nws = (p.W + 3) >> 2, SEGW = nws >= 8 ? 8 : nws, R = nws >= 8 ? 1 : 8 / nws;
    const int nunits = p.B * (p.C >> 5) * ((p.H + R - 1) / R) * ((nws + SEGW - 1) / SEGW);
    if ((p.W & 3) == 0) { for (int u = blockIdx.x; u < nunits; u += G) conv_cta_unit<true>(p, dws, cred, u, SEGW, R); }
    else { for (int u = blockIdx.x; u < nunits; u += G) conv_cta_unit<false>(p, dws, cred, u, SEGW, R); }
  }
  // generic-proxy writes (xh, w1h, w2h) are read by the async proxy (bulk copies) after the grid sync
  if (nb1 || nb2) asm volatile("fence.proxy.async.global;" ::: "memory");
  cg::this_grid().sync();
  switch (p.cfg2) {
    case 0: flat_gemm1<64, 64, 2, 4>(p, sm, nb1, mbar, ph); break;
    case 1: flat_gemm1<128, 96, 4, 2>(p, sm, nb1, mbar, ph); break;
    default: flat_gemm1<128, 128, 4, 2>(p, sm, nb1, mbar, ph); break;
  }
  if (nb2) asm volatile("fence.proxy.async.global;" ::: "memory");
  cg::this_grid().sync();
  switch (p.cfg3) {
    case 0: flat_gemm2<64, 64, 2, 4>(p, sm, p.split3, nb2, mbar, ph); break;
    case 1: flat_gemm2<64, 96, 2, 4>(p, sm, p.split3, nb2, mbar, ph); break;
    case 2: flat_gemm2<128, 96, 4, 2>(p, sm, p.split3, nb2, mbar, ph); break;
    case 3: flat_gemm2<64, 192, 2, 4>(p, sm, p.split3, nb2, mbar, ph); break;
    default: flat_gemm2<64, 160, 2, 4>(p, sm, p.split3, nb2, mbar, ph); break;
  }
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

// flat-path variant per shape class (rented B200; r2 interleaved + r3 run_tests against a same-build control):
//  W % 4 != 0 (1,160,449,7) and C <= 96 with M > 8192 (4,96,56,56): flat path, GEMM2 tiles by bulk-row copies
//  (54.0 -> 53.4 and 58.4 -> 57.8 us vs the cp.async flat path). 1,192,28,28 stays on the ring kernel: flat with bulk
//  rows in both GEMMs measured 31.4 vs 29.5 us. Bulk rows in GEMM1 cost +0.8..+3.4 us where a CTA gets 2 tiles.
static int flat_choice(const Params& p) {
  if ((p.W & 3) != 0 || (p.C <= 96 && p.M > 8192)) return FL_BULK2;
  return -1;
}

// flat-path plan: one or two waves per phase, tiles that fit the shared memory of 2 CTAs/SM
static bool flat_plan(Params& p, int grid) {
  if (p.C % 32 != 0 || p.C > 192 || p.HW < 128) return false;
  const long M = p.M, C = p.C, C4 = p.C4;
  struct G { int id, BM, BN; };
  const G g1[3] = {{0, 64, 64}, {1, 128, 96}, {2, 128, 128}};
  const G g2[5] = {{0, 64, 64}, {1, 64, 96}, {2, 128, 96}, {3, 64, 192}, {4, 64, 160}};
  double best1 = 1e300, best2 = 1e300;
  int c1 = -1, c2 = -1, s2 = 1;
  for (const G& g : g1) {
    if (C4 % g.BN) continue;
    const long ab = (long)(g.BM + g.BN) * (C + 8) * 2, cs = (long)g.BM * (g.BN + 4) * 4;
    if (FLAT_FOFF + (ab > cs ? ab : cs) > RES_OFF) continue;
    const long tiles = ((M + g.BM - 1) / g.BM) * (C4 / g.BN), waves = (tiles + grid - 1) / grid;
    if (waves > 2) continue;
    const double cost = waves * ((double)g.BM * g.BN * C / 512.0 + (g.BM + g.BN) * C * 2 / 64.0 + 1500.0);
    if (cost < best1) { best1 = cost; c1 = g.id; }
  }
  const int splits[12] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24};
  for (const G& g : g2) {
    if (C % g.BN) continue;
    for (int s : splits) {
      if (C4 % s) continue;
      const long KSL = C4 / s;
      if (KSL % 16 || KSL > 512) continue;
      const long smem = (long)(g.BM + g.BN) * (KSL + 8) * 2 + (2 * C4 + 2 * KSL + g.BN + 32) * 4;
      if (smem > RES_OFF) continue;
      const long units = ((M + g.BM - 1) / g.BM) * (C / g.BN) * s, waves = (units + grid - 1) / grid;
      if (waves > 2) continue;
      const double cost = waves * ((double)g.BM * g.BN * KSL / 512.0 + (g.BM + g.BN) * KSL * 2 / 64.0 + g.BM * g.BN / 64.0 + 1500.0);
      if (cost < best2) { best2 = cost; c2 = g.id; s2 = s; }
    }
  }
  if (c1 < 0 || c2 < 0 || p.C * 49 * 4 + 2048 > RES_OFF) return false;
  p.p1_mode = 3; p.cfg2 = c1; p.cfg3 = c2; p.split3 = s2;
  return true;
}

// mode 0: flat path per flat_choice, else the ring kernel; mode 1: flat with p.mask wherever it applies; mode 2: ring only
extern "C" const char* convnext_launch(Params p, void* stream_v, int mode) {
  static int grid_size = 0, grid_flat = 0;
  cudaError_t e;
  cudaStream_t stream = (cudaStream_t)stream_v;
  if (grid_size == 0) {
    e = cudaFuncSetAttribute(convnext_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(convnext_flat, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int bps = 0, bpf = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, convnext_k, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpf, convnext_flat, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    bps = bps < 1 ? 1 : (bps > 2 ? 2 : bps);
    bpf = bpf < 1 ? 1 : (bpf > 2 ? 2 : bpf);
    grid_size = sms * bps; grid_flat = sms * bpf;
  }
  int fm = -1;
  if (mode == 0) fm = flat_choice(p);
  else if (mode == 1) fm = p.mask < 0 ? 0 : p.mask;
  if (fm >= 0) {
    p.mask = fm;
    if (flat_plan(p, grid_flat)) {
      void* args[] = {&p};
      e = cudaLaunchCooperativeKernel((const void*)convnext_flat, dim3(grid_flat), dim3(NT), args, (size_t)SMEM_BYTES, stream);
      if (e != cudaSuccess) return cudaGetErrorString(e);
      return nullptr;
    }
  }
  int smin = 1;
  if (p.force_split != 1) while ((long)p.C4 / smin > TABK) smin *= 2;
  const Pick k2 = pick_cfg(p.M, p.C4, p.C, grid_size, 1, 1);
  const Pick k3 = pick_cfg(p.M, p.C, p.C4, grid_size, smin, p.force_split == 1 ? 1 : 16);
  p.cfg2 = k2.cfg; p.cfg3 = k3.cfg; p.split3 = k3.split;
  // phase 1: 32-channel tiles (4 channels per warp); patch tiles drop to 8 channels when that leaves SMs idle
  if (p1_geometry(p, 4) < grid_size && p.p1_mode == 1) p1_geometry(p, 1);
  void* args[] = {&p};
  e = cudaLaunchCooperativeKernel((const void*)convnext_k, dim3(grid_size), dim3(NT), args, (size_t)SMEM_BYTES, stream);
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
  p.mask = -1; p.force_split = 0;  // mask: flat-path variant, chosen per shape in convnext_launch

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = convnext_launch(p, (void*)stream, 0);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (fused, FP16 mma.sync GEMMs)"); }
```

```yaml design-card
id: r3-coop-flat-bulkrows
parents: [r2-coop-flat-s]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: persistent
  launch: fused
  tile: rows128
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  Lineage B, E2 (H24). The hypothesis was that the 2-4 us post-grid-sync load round comes from issue rate (4-B/16-B
  cp.async with div/mod), so replacing it with cp.async.bulk copies on one mbarrier would cut 1,192 to <= 22 us. The
  microbench partly supports the mechanism: a 32 KB just-written round takes 1591 clk as one bulk copy against 2607 clk
  for 16-B cp.async+div and 4504 clk for 4-B cp.async. But the round in the kernel is dominated by the grid barrier
  (1.6-1.9 us) and by cold-DRAM latency (2.6-3.5 us for every method). Per-row bulk copies of the post-sync A/B tiles
  moved the S rows by <= 0.6 us (inside the ~1.5 us run-to-run noise) and lost 1.9 us at 1,192 against the ring kernel.
  H24 is refuted at kernel level. Shipped: the parent's dispatch, plus GEMM2 bulk-row copies on the two flat-path rows.
expected_effect:
  S: {pct: -1, confidence: medium}
  M: {pct: 0, confidence: high}
  L: {pct: 0, confidence: high}
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 107296
  threads_per_cta: 256
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 151552
  launches_per_call: 1
knobs:
  NT: 256
  GRID: 296
  FLAT_DISPATCH: "W % 4 != 0, or C <= 96 and M > 8192 (C <= 192, HW >= 128): flat path with FL_BULK2"
  RING_PATH: "unchanged r1 ring kernel for all other shapes (incl. 1,192,28,28)"
  RES_OFF: "flat-path smem plans capped at SMEM_BYTES - 4096 (mbarrier slot)"
tests: [H24]
findings:
  - "E2 step 1 microbench, cooperative-style stub (296 CTAs x 256 thr, 2/SM, ~1.9 GHz, 2xL2 flush before each launch), per-CTA load round after the grid barrier, median/max clk and B/clk/SM. JUST-WRITTEN (written by other CTAs before the barrier): 6 KB cp4+div 1876/2358 (6.6), cp16+div 1322/1747 (9.3), ldg128-all-issued 1046/1444 (11.7), one bulk 990/1433 (12.4); 12 KB 2344/2898 (10.5), 1585/2003 (15.5), 1211/1583 (20.3), 1087/1465 (22.6); 32 KB 4504/5221 (14.6), 2607/3066 (25.1), 1806/2251 (36.3), 1591/1916 (41.2) [probe_b200]"
  - "E2 step 1 microbench, COLD DRAM source: 6 KB cp4+div 3046/3888 (4.0), cp16+div 2606/3465 (4.7), ldg128 2250/3094 (5.5), bulk 2371/3304 (5.2); 12 KB 4055/5069, 3568/4536, 3134/4022, 3268/4172; 32 KB 6766/8057 (9.7), 6336/7260 (10.3), 5642/6392 (11.6), 5136/6492 (12.8). Cold rounds cost 1.2-3.5 us whatever the load method [probe_b200]"
  - "grid barrier cost alone (296 CTAs): cg-style flip barrier (atom.add.release + ld.acquire poll) waits 3007 clk median with no stores before it, 3399 with 6 KB of stores pending and 3644 with 32 KB pending; release after the last arrival is 1.5-2.0 us (globaltimer, 256 ns grain). red.release + monotone-target poll is the same (2886-3578 clk). A count+reset+generation barrier is worse: 4760-5674 clk (2.3-2.8 us) [probe_b200]"
  - "H24 verdict: the branch condition held (32 KB just-written bulk round 0.83 us median, under the 1.2 us bar), so the bulk rebuild was built. The kernel-level claim is REFUTED: the 2-4 us post-sync steps in the r2 audit are barrier release (1.6-1.9 us) plus cold-DRAM or dependent rounds. Copy issue rate is worth only ~0.5 us per 32 KB round (cp16+div 1.36 us vs bulk 0.83 us) [probe_b200, run_tests]"
  - "run_tests with a same-build control (parent dispatch, no bulk), rented, us: 1,192 ring 29.5 vs flat+bulk1+bulk2 31.4 (+1.9, so 1,192 stays on ring); 1,160,449,7 54.0 vs 53.4 / 53.6 (bulk2); 4,96,56,56 58.4 vs 57.8 / 58.7 (bulk2). Unchanged-code rows moved up to 1.5 us between runs (2,768: 46.7 vs 48.2), so the bulk2 gain is within noise. Success criteria (1,192 <= 22, 2,96 <= 32) missed: 29.7 / 42.0 [run_tests]"
  - "final run_tests (rented, 16/16 PASS, us): 1,192 29.7; 1,96 29.1; 1,160,449,7 53.6; 2,96 42.0; 4,192 46.1; 8,384 85.7; 4,96 58.7; 8,192 66.1; 16,384 113.9; 2,768 48.1; 8,768 86.7; 16,768 108.5; 8,320 174.9; 32,768 180.1; 32,128 592.9; 64,96 1783.0; projected portal 0.5625 +- 0.008 (same-session control 0.5643) [run_tests]"
  - "ablation in a ctypes probe harness (flat path only, interleaved x2, us; old flat = cp.async): bulk rows in GEMM1 -1.5 at 1,192 but +0.6..+3.4 at 2,96 / 1,160 / 4,96 / 8,192 (CTAs with 2 tiles of 224-256 rows; one warp issuing 7-8 small bulk copies per lane is slower than 256 threads of cp.async); GEMM2 bulk rows -2.0 at 1,192, -1.4 at 1,160, -1.8 at 4,96, +1.7 at 8,192 [probe_b200]"
  - "dead end: conv-first P1 ordering (x loads issued before the dw-weight cp.async wait, weight conversion moved to CTAs with one conv unit fewer, after their conv) is +1..+2 us on 6 of 7 S rows (-0.8 at 1,96) [probe_b200]"
  - "dead end: prefetching grn_weight into smem at kernel entry (to take its cold __ldg out of the GEMM2 prologue) changes nothing (+-0.4 us): that cold load was not on the critical path [probe_b200]"
  - "PITFALL (harness): a ctypes + torch.empty-in-fn probe harness timed the flat path 4-5 us slower than run_tests (1,192 flat 34.8 vs 29.5-30.5), and its relative gains did not carry over (bulk -3 us in the probe, +1.9 us vs ring in run_tests). Decide dispatch from run_tests with a same-build control, not from probe harness_time on a ctypes wrapper [probe_b200, run_tests]"
  - "PITFALL (tooling): probe_b200 does not keep /tmp between calls (a file written in one probe was gone two calls later); every probe build must carry its full source [probe_b200]"
  - "correctness: bulk copies move the same bytes as cp.async; max |err| vs an fp32 torch reference was identical old vs new (one seed): 1,192 1.73e-3, 2,96 1.92e-3, 1,96 2.21e-3, 4,192 1.96e-3, 1,160 1.91e-3, 4,96 2.29e-3, 8,192 2.06e-3. The per-row fp64 log over the harness's 10 seeded rounds was NOT run this session; the error tail is the parent's (H22) [probe_b200]"
  - "proxy ordering used: fence.proxy.async.global by all threads before each grid.sync whose consumers bulk-read generic-written global data, by the issuing lane before each expect_tx, and fence.proxy.async.shared::cta + __syncthreads before refilling smem previously written by generic stores; 16/16 x 10 rounds pass [run_tests]"
  - "lineage B verdict: with F ~4.6 us, two grid barriers at 1.6-1.9 us of release each, and cold rounds of 2.6-3.5 us, the cooperative kernel's S-row floor sits near 20 us before any SM work. The parent's S rows (29-58 us) cannot reach <= 20 us by changing load instructions. Only a barrier-free per-image dataflow (H10, not run here) could; otherwise retire the S-row thesis in r4 [INFERRED from microbench + run_tests]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: tma, wstat: false, launches: 1, prefetch: false, x: none, w: none, st: none, cluster: 1, shapes: "flat kernel with GEMM2 bulk-row copies: W%4!=0 or (C<=96 and M>8192), C<=192, HW>=128 (1,160,449,7 and 4,96,56,56)"}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: cpasync, wstat: false, launches: 1, prefetch: false, x: none, w: none, st: none, cluster: 1, shapes: "r1 ring kernel: all other shapes (unchanged)"}
runs_on:
  H200: {runs: true, representative: false, note: "cp.async.bulk/mbarrier/cooperative launch exist on sm_90; clocks, HMMA rate and barrier latency differ"}
  A100: {runs: false, representative: false, note: "cp.async.bulk and mbarrier expect_tx need sm_90+"}
  L40S: {runs: false, representative: false, note: "no bulk copies; 100 KB smem < 107 KB dynamic smem"}
risks:
  - "Gain is within noise (<= 0.6 us on two rows); the portal score should match the parent within +-0.008"
  - "Bulk reads of data written through the generic proxy rely on fence.proxy.async.global around the grid syncs; passes 16/16 x 10 rounds, but a missing fence would show up only as rare stale reads"
  - "Split-K red.add into out (pre-filled with x by the conv) is order-nondeterministic, within tolerance (same as parent)"
  - "Both kernels set MaxDynamicSharedMemorySize once to the same value (portal-order safe)"
measure_first: [H200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```