### Rationale
E1 tested H17, and on the bench the idea does not hold up. The depth-2 kernel (2 heads × 2 positions per 16 lanes) only keeps all its x loads in flight when it is given 80 registers, and then it fits 6 CTAs/SM. Capping registers at 64 for 8 CTAs/SM made ptxas push half the x loads behind the first FMAs. That happened with nc loads, plain loads, and table loads through `cp.async` to shared memory, and cost +3 to +24%. So 36 MB cannot become a single wave without stalling the loads, and at 6 CTAs/SM the depth-2 kernel is +6% there. The candidate therefore uses depth-2 only at 57–134 MB, where it is roughly level with r2 on the bench (−0.7 to +1.1%) and has 15–20% fewer waves. r2 code runs everywhere else. It is a cheap portal probe of whether fewer wave boundaries pay off at 1.5 GHz. 61 MB is +1.1%, just over the +1% bar, which is within bench noise. This is a portal test, not a speed claim.

```json solution-spec
{"name": "r3-rope-quad-d2-m", "definition": "088_rotary_position_embedding_application", "author": "solx-loop", "description": "Depth-2 quad kernel (heads 2h,2h+1 x positions 2s2,2s2+1 per 16 lanes, all 16 loads up front, 80 regs, 6 CTAs/SM) at 50k-140k rows; r2 pair/row paths elsewhere. Portal probe of H17/H15.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// RoPE application (Q and K), fp32, head_dim 128. One fused one-shot launch.
// Layout: 16 lanes per row; lane j owns columns [4j, 4j+4) and [64+4j, 64+4j+4) (one 128-bit load each),
// so the rotate_half partner is already in the same thread and no shuffles are needed.
// out[i]    = x[i]*cos[i]    - x[i+64]*sin[i]        (i < 64)
// out[i+64] = x[i+64]*cos[i+64] + x[i]*sin[i+64]
// Three kernels: rope_row (one row per 16 lanes), rope_pair (heads 2h, 2h+1 at the same s per 16 lanes,
// sharing one cos/sin load) and rope_quad (heads 2h, 2h+1 at positions 2s2, 2s2+1: four rows and two table
// rows per 16 lanes, all 16 loads issued before any FMA). The sequence position uses a host-computed magic
// divisor (q = (umulhi(n, m) + n) >> l, exact for n < 2^31) instead of a runtime modulo.
#include <cuda_runtime.h>
#include <stdint.h>

#define LD4(p, v) asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define LDX4(p, v) asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define ST4EL(p, v, pol) asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]), "l"(pol) : "memory")
// Predicated store without a branch: a branch let ptxas sink the second position's loads into it.
#define ST4ELP(p, v, pol, pr) asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %6, 0;\n" \
    " @q st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;\n}" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]), "l"(pol), "r"(pr) : "memory")

// The pair kernel compiles to 40 regs (12 CTAs/SM = 96 KB of x in flight), which measured 2-8% slower than
// 10 CTAs/SM. Unused dynamic shared memory with a 132 KB carve-out caps it at 10 CTAs/SM and leaves
// 124 KB of L1 for the cos/sin rows.
#define PAIR_PAD_SMEM 11500
#define PAIR_CARVEOUT_PCT 50

__device__ __forceinline__ unsigned div_magic(unsigned n, unsigned m, unsigned l)
{
    return (__umulhi(n, m) + n) >> l;               // n < 2^31, so the add cannot overflow
}

__device__ __forceinline__ void rot4(const float* xl, const float* xh, const float* cl, const float* ch,
                                     const float* sl, const float* sh, float* ol, float* oh)
{
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        ol[e] = xl[e] * cl[e] - xh[e] * sl[e];
        oh[e] = xh[e] * ch[e] + xl[e] * sh[e];
    }
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
rope_row(const float* __restrict__ q, const float* __restrict__ k,
         const float* __restrict__ cs, const float* __restrict__ sn,
         float* __restrict__ qo, float* __restrict__ ko,
         unsigned Rq, unsigned R, unsigned S, unsigned m, unsigned l)
{
    constexpr int ROWS = THREADS / 16;
    const unsigned j = threadIdx.x & 15;
    const unsigned r = blockIdx.x * ROWS + (threadIdx.x >> 4);
    if (r >= R) return;
    const float* x; float* y; unsigned rl;
    if (r < Rq) { x = q; y = qo; rl = r; } else { x = k; y = ko; rl = r - Rq; }
    const unsigned s = rl - div_magic(rl, m, l) * S;    // sequence position of this row
    const float* xp = x + (size_t)rl * 128 + j * 4;
    const float* cp = cs + (size_t)s * 128 + j * 4;
    const float* sp = sn + (size_t)s * 128 + j * 4;
    y += (size_t)rl * 128 + j * 4;
    float xl[4], xh[4], cl[4], ch[4], sl[4], sh[4], ol[4], oh[4];
    LDX4(xp, xl); LDX4(xp + 64, xh);
    LD4(cp, cl);  LD4(cp + 64, ch);
    LD4(sp, sl);  LD4(sp + 64, sh);
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    rot4(xl, xh, cl, ch, sl, sh, ol, oh);
    ST4EL(y, ol, pol);
    ST4EL(y + 64, oh, pol);
}

// Pair p of a stream covers heads 2*hp and 2*hp+1 of batch b at position s, p = (b*H/2 + hp)*S + s.
// Its first row is 2*(p - s) + s and the second is S rows further. Needs H and Hkv even.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
rope_pair(const float* __restrict__ q, const float* __restrict__ k,
          const float* __restrict__ cs, const float* __restrict__ sn,
          float* __restrict__ qo, float* __restrict__ ko,
          unsigned Pq, unsigned P, unsigned S, unsigned m, unsigned l)
{
    constexpr int PAIRS = THREADS / 16;
    const unsigned j = threadIdx.x & 15;
    const unsigned p = blockIdx.x * PAIRS + (threadIdx.x >> 4);
    if (p >= P) return;
    const float* x; float* y; unsigned pl;
    if (p < Pq) { x = q; y = qo; pl = p; } else { x = k; y = ko; pl = p - Pq; }
    const unsigned qd = div_magic(pl, m, l);
    const unsigned s = pl - qd * S;
    const size_t row0 = (size_t)(2 * qd) * S + s;
    const size_t hs = (size_t)S * 128;               // one head further
    const float* xp = x + row0 * 128 + j * 4;
    const float* cp = cs + (size_t)s * 128 + j * 4;
    const float* sp = sn + (size_t)s * 128 + j * 4;
    y += row0 * 128 + j * 4;
    float xl0[4], xh0[4], xl1[4], xh1[4], cl[4], ch[4], sl[4], sh[4], ol[4], oh[4];
    LDX4(xp, xl0);      LDX4(xp + 64, xh0);
    LDX4(xp + hs, xl1); LDX4(xp + hs + 64, xh1);
    LD4(cp, cl);  LD4(cp + 64, ch);
    LD4(sp, sl);  LD4(sp + 64, sh);
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    rot4(xl0, xh0, cl, ch, sl, sh, ol, oh);
    ST4EL(y, ol, pol);
    ST4EL(y + 64, oh, pol);
    rot4(xl1, xh1, cl, ch, sl, sh, ol, oh);
    ST4EL(y + hs, ol, pol);
    ST4EL(y + hs + 64, oh, pol);
}

// Group g of a stream covers heads 2*hp, 2*hp+1 at positions s = 2*s2 and s+1, g = (b*H/2 + hp)*S2 + s2,
// S2 = ceil(S/2). Rows row0, row0+1, row0+S, row0+S+1 with row0 = 2*(g - s2)*... = 2*qd*S + s.
// For odd S the last group's second position is off: its loads re-read position s and its stores are
// predicated off. All 16 loads are issued before any FMA (80 regs, 6 CTAs/SM at 128 threads); capping
// registers at 64 made ptxas delay half of the x loads behind the first FMAs (+3..+19% on the bench).
__global__ void __launch_bounds__(128, 6)
rope_quad(const float* __restrict__ q, const float* __restrict__ k,
          const float* __restrict__ cs, const float* __restrict__ sn,
          float* __restrict__ qo, float* __restrict__ ko,
          unsigned Gq, unsigned G, unsigned S, unsigned S2, unsigned m, unsigned l)
{
    const unsigned j = threadIdx.x & 15;
    const unsigned g = blockIdx.x * 8 + (threadIdx.x >> 4);
    if (g >= G) return;
    const float* x; float* y; unsigned gl;
    if (g < Gq) { x = q; y = qo; gl = g; } else { x = k; y = ko; gl = g - Gq; }
    const unsigned qd = div_magic(gl, m, l);
    const unsigned s = 2 * (gl - qd * S2);
    const unsigned two = s + 1 < S;
    const unsigned d1 = two ? 128u : 0u;
    const size_t row0 = (size_t)(2 * qd) * S + s;
    const size_t hs = (size_t)S * 128;
    const float* xp = x + row0 * 128 + j * 4;
    const float* cp = cs + (size_t)s * 128 + j * 4;
    const float* sp = sn + (size_t)s * 128 + j * 4;
    y += row0 * 128 + j * 4;
    float a0l[4], a0h[4], a1l[4], a1h[4], b0l[4], b0h[4], b1l[4], b1h[4];
    float cl[4], ch[4], sl[4], sh[4], cl1[4], ch1[4], sl1[4], sh1[4], ol[4], oh[4];
    LDX4(xp, a0l);           LDX4(xp + 64, a0h);
    LDX4(xp + hs, a1l);      LDX4(xp + hs + 64, a1h);
    LDX4(xp + d1, b0l);      LDX4(xp + d1 + 64, b0h);
    LDX4(xp + hs + d1, b1l); LDX4(xp + hs + d1 + 64, b1h);
    LD4(cp, cl);       LD4(cp + 64, ch);       LD4(sp, sl);       LD4(sp + 64, sh);
    LD4(cp + d1, cl1); LD4(cp + d1 + 64, ch1); LD4(sp + d1, sl1); LD4(sp + d1 + 64, sh1);
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    rot4(a0l, a0h, cl, ch, sl, sh, ol, oh);
    ST4EL(y, ol, pol);             ST4EL(y + 64, oh, pol);
    rot4(a1l, a1h, cl, ch, sl, sh, ol, oh);
    ST4EL(y + hs, ol, pol);        ST4EL(y + hs + 64, oh, pol);
    rot4(b0l, b0h, cl1, ch1, sl1, sh1, ol, oh);
    ST4ELP(y + 128, ol, pol, two); ST4ELP(y + 192, oh, pol, two);
    rot4(b1l, b1h, cl1, ch1, sl1, sh1, ol, oh);
    ST4ELP(y + hs + 128, ol, pol, two); ST4ELP(y + hs + 192, oh, pol, two);
}

static void magic_div(unsigned d, unsigned& m, unsigned& l)
{
    // l = ceil(log2 d), m = floor(2^32 (2^l - d) / d) + 1
    l = 0;
    while ((1ull << l) < d) ++l;
    m = (unsigned)((((1ull << 32) * ((1ull << l) - d)) / d) + 1);
}

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            unsigned H, unsigned Hkv, cudaStream_t stream)
{
    unsigned m, l;
    const bool even = H % 2 == 0 && Hkv % 2 == 0;
    if (even && R >= 50000u && R <= 140000u && S <= 1536u) {
        // 57-134 MB workloads: four rows (2 heads x 2 positions) per 16 lanes, 32 rows per 128-thread CTA
        const unsigned S2 = (S + 1) / 2;
        magic_div(S2, m, l);
        const unsigned B = Rq / (H * S);
        const unsigned Gq = B * (H / 2) * S2;
        const unsigned G = Gq + B * (Hkv / 2) * S2;
        rope_quad<<<(G + 7) / 8, 128, 0, stream>>>(q, k, cs, sn, qo, ko, Gq, G, S, S2, m, l);
        return;
    }
    magic_div(S, m, l);
    if (even && R > 4096u && S <= 1536u) {
        // 6.6-326 MB workloads: two heads per 16 lanes, 8 pairs per 128-thread CTA, 10 CTAs/SM
        static bool attr_set = false;            // kernel attribute, independent of the call's data
        if (!attr_set) {
            cudaFuncSetAttribute(rope_pair<128>, cudaFuncAttributePreferredSharedMemoryCarveout, PAIR_CARVEOUT_PCT);
            attr_set = true;
        }
        const unsigned P = R / 2;
        const unsigned grid = (P + 7) / 8;
        rope_pair<128><<<grid, 128, PAIR_PAD_SMEM, stream>>>(q, k, cs, sn, qo, ko, Rq / 2, P, S, m, l);
    } else if (R <= 262144u) {     // 8 rows per 128-thread CTA
        const unsigned grid = (R + 7) / 8;
        rope_row<128><<<grid, 128, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S, m, l);
    } else {                       // 16 rows per 256-thread CTA
        const unsigned grid = (R + 15) / 16;
        rope_row<256><<<grid, 256, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S, m, l);
    }
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            unsigned H, unsigned Hkv, cudaStream_t stream);

void run(torch::Tensor query, torch::Tensor key, torch::Tensor cos, torch::Tensor sin,
         torch::Tensor query_rotated, torch::Tensor key_rotated)
{
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && cos.is_cuda() && sin.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
                cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && cos.is_contiguous() && sin.is_contiguous() &&
                query_rotated.is_contiguous() && key_rotated.is_contiguous(), "contiguous only");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && query.size(3) == 128 && key.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == query.size(2) && cos.size(1) == 128, "cos shape");
    TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape");
    TORCH_CHECK(key.size(2) == query.size(2) && key.size(0) == query.size(0), "batch/seq_len mismatch");
    TORCH_CHECK(query_rotated.sizes() == query.sizes() && key_rotated.sizes() == key.sizes(), "output shapes");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(query.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(key.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(cos.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(sin.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(query_rotated.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(key_rotated.data_ptr()) & 15) == 0, "16-byte alignment required");

    const int64_t S = query.size(2);
    const int64_t Rq = query.numel() / 128;
    const int64_t R = Rq + key.numel() / 128;
    TORCH_CHECK(R < (1LL << 31) && S > 0, "too many rows");
    if (R == 0) return;

    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_launch(query.data_ptr<float>(), key.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
                query_rotated.data_ptr<float>(), key_rotated.data_ptr<float>(),
                (unsigned)Rq, (unsigned)R, (unsigned)S, (unsigned)query.size(1), (unsigned)key.size(1), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "RoPE application (DPS)"); }
```

```yaml design-card
id: r3-rope-quad-d2-m
parents:
  - r2-rope-pair2-magic-occ10
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: fused
  tile: rows32
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  E1/H17: a depth-2 group (heads 2h, 2h+1 at positions 2s2, 2s2+1; 4 rows, 2 table rows, all 16 loads
  issued before any FMA) cuts the number of waves at 57-134 MB by 15-20% (6 CTAs/SM x 32 rows = 192 rows
  per SM, against r2's 160). On the bench it lands within -0.7..+1.1% of r2 there. If the portal penalty
  comes from wave boundaries at the locked 1.5 GHz clock (H15/H17), these four workloads should gain >= 2%
  on the portal. The single-wave 36 MB case planned in E1 cannot be built without ptxas serialising the
  loads, so 36 MB stays on r2's pair path.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: medium
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 80
  smem_per_cta_bytes: 0
  threads_per_cta: 128
  ctas_per_sm: 6
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  quad_threads: 128
  quad_launch_bounds_min_blocks: 6
  quad_rule: "H,Hkv even and 50000 <= R <= 140000 and S <= 1536"
  pair_pad_smem: 11500
  pair_carveout_pct: 50
  row_threshold: 262144
  store_hint: evict_last
dispatch:
  - {rule: "H%2==0 && Hkv%2==0 && 50000<=R<=140000 && S<=1536", kernel: rope_quad, meta: {threads: 128, groups: 8, rows: 32, ctas_per_sm: 6}}
  - {rule: "H%2==0 && Hkv%2==0 && R>4096 && S<=1536", kernel: rope_pair<128>, meta: {threads: 128, pairs: 8, ctas_per_sm: 10}}
  - {max_rows: 262144, kernel: rope_row<128>, meta: {threads: 128, rows: 8}}
  - {max_rows: null, kernel: rope_row<256>, meta: {threads: 256, rows: 16}}
tests:
  - H17
  - H15
findings:
  - "H17 core assumption fails on the bench: a single wave at 36 MB needs >= 238 resident rows/SM (8 CTAs x 32 rows at <= 64 regs), but at 64 regs ptxas always delays part of the 8 x loads until after the first FMAs. Seen with nc x loads (quad_8: 64 regs, +3.1% at 36 MB, +9.4% at 57, +6.3% at 61, +12..+19% at 6.6-16 MB), with table(s+1) loads moved after the first stores (56 regs, 9 CTAs/SM: +0.9..+18%), and with plain aliasing x loads (still split) [probe_b200 SASS + harness_time, 3 interleaved repeats]"
  - "dead end: table rows via cp.async.ca into per-thread smem slots (x only in registers): nvcc/NVRTC still allocate 64 regs, and ptxas puts the cp.async wait before the last x load: +5..+24% at 6.6-134 MB (cpa128 8 CTAs/SM; cpa256 4 CTAs/SM is worse, +11..+19%) [probe_b200]"
  - "trap: an if (two) branch around the second position's stores let ptxas sink that position's 8 loads into the branch (two serial round trips). Use predicated asm stores (setp + @p st) so the code has no branch [probe_b200 SASS]"
  - "best depth-2 variant: all 16 loads up front, 80 regs (nvcc and NVRTC agree), 6 CTAs/SM = 96 KB of x per SM: vs r2 pair +0.9% at 6.6 MB, +2.5% at 15.8, +6.1% at 36 (1.24 waves vs pair's 1.49), +1.4% at 57, +0.9% at 61 [probe_b200, 3 interleaved repeats]"
  - "run_tests with depth-2 routed to 50k-140k rows: 16/16 pass; 57 MB -0.7%, 61 MB +1.1%, 102 MB +0.3%, 134 MB -0.2% vs r2 on the same B200; other workloads run r2's code (-3.4..+1.1%, noise); band S -1.3 / M +0.4 / L -0.2%; predicted portal 0.6835 +- 0.0115 [run_tests]"
  - "waves at 57/61/102/134 MB: quad 1.92/2.10/3.49/4.61 vs pair 2.31/2.52/4.19/5.54 (CTAs / (148 x resident CTAs per SM)) [reasoning]"
  - "consistent with H4: the bench optimum is about 128-192 rows in flight per SM; one wave at 36 MB needs about 238, so depth and wave count are tied to an occupancy the bench penalises [probe_b200, r2 findings]"
paths:
  - {max_tokens: null, max_rows: 4096, lang: cuda, width: 128, threads: 128, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, min_rows: 4097, max_rows: 49999, max_seq: 1536, lang: cuda, width: 128, threads: 128, rows: 16, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el, rows_per_thread: 2, table_shared_heads: 2, ctas_per_sm_cap: 10, pad_smem: 11500}
  - {max_tokens: null, min_rows: 50000, max_rows: 140000, max_seq: 1536, lang: cuda, width: 128, threads: 128, rows: 32, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el, rows_per_thread: 4, table_shared_heads: 2, positions_per_thread: 2, regs: 80, ctas_per_sm_cap: 6}
  - {max_tokens: null, min_rows: 140001, max_rows: 330000, max_seq: 1536, lang: cuda, width: 128, threads: 128, rows: 16, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el, rows_per_thread: 2, table_shared_heads: 2, ctas_per_sm_cap: 10, pad_smem: 11500}
  - {max_tokens: null, max_rows: null, lang: cuda, width: 128, threads: 256, rows: 16, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
runs_on:
  H200:
    runs: false
    representative: false
    note: "not measured; B200 bench only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "Bench-neutral at 57-134 MB (61 MB +1.1%, just over the +1% submission bar but within noise); any portal gain depends on H17 holding, and a null portal result closes wave count as a lever at M"
  - "36 MB, the workload where H17 predicted the largest gain, is not covered: depth-2 is +6% there on the bench at 6 CTAs/SM"
  - "rope_quad depends on ptxas keeping all 16 loads before the first FMA at 80 regs; a different ptxas version could reschedule them (check the SASS order)"
  - "cudaFuncSetAttribute (carve-out preference) on rope_pair, inherited from r2; flag for review"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```