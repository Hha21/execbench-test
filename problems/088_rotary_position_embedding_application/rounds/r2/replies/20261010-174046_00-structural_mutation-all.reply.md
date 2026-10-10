I tested the E1 idea. It does not reach the 3% success bar, but my best version of it is 1.7–2.6% faster than r1 at S/M on the bench, ties at L, and passes 16/16 workloads.

**Result.** Against r1 on the same B200, by band: S −2.6%, M −1.7%, L −0.1%. No workload is more than 0.2% slower. The emulator predicts a portal score of 0.681 ± 0.012, against 0.677 for r1.

**What the bench showed:**
- **Two heads sharing one cos/sin load (H5, 48 regs, no spills) is within ±1.5% of r1 at every size, so H5 looks refuted on the bench.** Four heads per thread is 2–44% slower.
- **Table traffic is small.** The twin with tables replaced by constants is only 1.5–2.6% faster, and only at S ≥ 4096.
- **More bytes in flight hurts here (new finding).** The nvcc build gives the pair kernel 40 regs, so 12 CTAs/SM fit (96 KB of x in flight per SM). Its machine code is identical to the 48-reg build at 10 CTAs/SM (80 KB), yet it is 2–8% slower. Cutting the row kernel's occupancy also hurts. The optimum seems to sit around 64–80 KB per SM.
- **The magic divisor (H8) helps a little:** −0.5 to −2.5% at S/M, neutral at L.

**What the candidate does.** It uses the pair kernel for the 6.6–326 MB workloads. That kernel gets 11,500 B of unused dynamic shared memory and a 132 KB carve-out, which caps it at 10 CTAs/SM. The row kernel, now with the magic divisor, handles the rest.

**Before submitting:**
- The shared-memory padding sets a `cudaFuncSetAttribute` carve-out on our own kernel once. I believe that is normal occupancy control, not environment manipulation, but you may want to confirm.
- The portal slot is still the H15 reading. If the portal gain at M beats this bench gain (about 1.7%) by 3 points or more, H15 is supported.
- The 10 CTAs/SM cap assumes the driver rounds the 50% carve-out hint up to 132 KB, as it did in the probe.

I used 3 of the 8 GPU calls.

```json solution-spec
{"name": "r2-rope-pair2-magic-occ10", "definition": "088_rotary_position_embedding_application", "author": "solx-loop", "description": "Two heads per 16-lane group sharing one cos/sin load (occupancy capped at 10 CTAs/SM) at 6.6-326 MB, r1 rows elsewhere; magic-divisor sequence index everywhere.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// RoPE application (Q and K), fp32, head_dim 128. One fused one-shot launch.
// Layout: 16 lanes per row; lane j owns columns [4j, 4j+4) and [64+4j, 64+4j+4) (one 128-bit load each),
// so the rotate_half partner is already in the same thread and no shuffles are needed.
// out[i]    = x[i]*cos[i]    - x[i+64]*sin[i]        (i < 64)
// out[i+64] = x[i+64]*cos[i+64] + x[i]*sin[i+64]
// Two kernels: rope_row (one row per 16 lanes) and rope_pair (rows of heads 2h and 2h+1 at the same s per
// 16 lanes, sharing one cos/sin load). The sequence position uses a host-computed magic divisor
// (q = (umulhi(n, m) + n) >> l, exact for n < 2^31) instead of a runtime modulo.
#include <cuda_runtime.h>
#include <stdint.h>

#define LD4(p, v) asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define LDX4(p, v) asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define ST4EL(p, v, pol) asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]), "l"(pol) : "memory")

// The pair kernel compiles to 40 regs (12 CTAs/SM = 96 KB of x in flight), which measured 2-8% slower than
// 10 CTAs/SM. Unused dynamic shared memory with a 132 KB carve-out caps it at 10 CTAs/SM and leaves
// 124 KB of L1 for the cos/sin rows.
#define PAIR_PAD_SMEM 11500
#define PAIR_CARVEOUT_PCT 50

__device__ __forceinline__ unsigned div_magic(unsigned n, unsigned m, unsigned l)
{
    return (__umulhi(n, m) + n) >> l;               // n < 2^31, so the add cannot overflow
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
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        ol[e] = xl[e] * cl[e] - xh[e] * sl[e];
        oh[e] = xh[e] * ch[e] + xl[e] * sh[e];
    }
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
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        ol[e] = xl0[e] * cl[e] - xh0[e] * sl[e];
        oh[e] = xh0[e] * ch[e] + xl0[e] * sh[e];
    }
    ST4EL(y, ol, pol);
    ST4EL(y + 64, oh, pol);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        ol[e] = xl1[e] * cl[e] - xh1[e] * sl[e];
        oh[e] = xh1[e] * ch[e] + xl1[e] * sh[e];
    }
    ST4EL(y + hs, ol, pol);
    ST4EL(y + hs + 64, oh, pol);
}

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            unsigned H, unsigned Hkv, cudaStream_t stream)
{
    // magic divisor for S: l = ceil(log2 S), m = floor(2^32 (2^l - S) / S) + 1
    unsigned l = 0;
    while ((1ull << l) < S) ++l;
    const unsigned m = (unsigned)((((1ull << 32) * ((1ull << l) - S)) / S) + 1);
    if (H % 2 == 0 && Hkv % 2 == 0 && R > 4096u && S <= 1536u) {
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
id: r2-rope-pair2-magic-occ10
parents:
  - r1-rope-v4nc-s128
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: fused
  tile: rows16
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  E1/H5: two heads at the same s per 16-lane group share one cos/sin load and double the x bytes per thread
  (16 rows per 128-thread CTA, half the CTAs). On the bench it is neutral (within +-1.5% of r1). Bytes in flight
  beyond about 80 KB per SM hurt, so the pair path is capped at 10 CTAs/SM with unused dynamic smem. A magic
  divisor replaces rl % S on all paths (H8, -0.5..-2.5% at S/M). The portal slot reads H15: if the locked clock
  starves latency, the half-CTA-count pair path should gain more on the portal at M than on the bench (~-1.7%).
expected_effect:
  S:
    pct: -2.6
    confidence: medium
  M:
    pct: -1.7
    confidence: medium
  L:
    pct: 0
    confidence: medium
resources_sm100a:
  regs_per_thread: 40
  smem_per_cta_bytes: 11500
  threads_per_cta: 128
  ctas_per_sm: 10
  bytes_in_flight_per_sm: 81920
  launches_per_call: 1
knobs:
  pair_threads: 128
  pair_HG: 2
  pair_pad_smem: 11500
  pair_carveout_pct: 50
  pair_rule: "H,Hkv even and R > 4096 and S <= 1536"
  row_threshold: 262144
  store_hint: evict_last
dispatch:
  - {rule: "H%2==0 && Hkv%2==0 && R>4096 && S<=1536", kernel: rope_pair<128>, meta: {threads: 128, pairs: 8, ctas_per_sm: 10}}
  - {max_rows: 262144, kernel: rope_row<128>, meta: {threads: 128, rows: 8}}
  - {max_rows: null, kernel: rope_row<256>, meta: {threads: 256, rows: 16}}
tests:
  - H5
  - H8
  - H3
  - H15
  - H11
findings:
  - "H5 refuted on the bench: pair HG=2 (NVRTC 48 regs, 10 CTAs/SM, 0 spills) vs r1+magic within +-1.5% at every size: 6.6 MB -2.4..-4.5%, 15.8-61 MB +-1%, 102 MB -1.3..-2.2%, 134-326 MB -0.5..-1%, 354-612 MB +0.4..+1.4%, 2689 MB +1.1..+1.3% [probe_b200, 3 interleaved repeats, 2 sessions]"
  - "dead end: HG=4 (48 regs) +44% at 2.2 MB, +12..+22% at 6.6-16 MB, +7% at 57-61 MB, +2..+3% at L [probe_b200]"
  - "H3: no-table twin of the pair kernel (constants for cos/sin) vs pair with tables: -2.6% at 612 MB, -1.5% at 407, +-1% at 269-354, +1% at 2689; table L2 traffic is <= 2.6% on the bench even at S=8192 [probe_b200]"
  - "NEW: more bytes in flight hurts. The pair kernel at 40 regs (nvcc; 12 CTAs/SM = 96 KB of x per SM) has SASS identical to the 48-reg build (10 CTAs/SM = 80 KB) but is +2..+8% slower at 6.6-134 MB and +2..+3% at L [probe_b200: p2_128b vs p2_128; run_tests first draft +1.3..+2.6% at 36-102 MB]"
  - "occupancy sweep with smem padding (132 KB carve-out): row kernel 16 CTAs/SM best; 13 CTAs +2..+4.5% at 36-102 MB; 10 CTAs +5..+10% everywhere. Pair kernel 10 vs 7 CTAs/SM within 1-3%. Rows in flight per SM: optimum about 128-160 rows (64-80 KB) [probe_b200]"
  - "padding smem alone at unchanged occupancy (row kernel 16 CTAs/SM with 7.4 KB smem) costs +0.4..+3.5% at 57-102 MB (smaller L1 for tables); the pair kernel at 10 CTAs/SM with 11.5 KB padding ties no padding [probe_b200]"
  - "carve-out: PreferredSharedMemoryCarveout 45-50% gives the 132 KB config; CTAs = floor(135168/(smem+1024)) matched cuOccupancy for 7400/9000/11500/16000 B (16/13/10/7) [probe_b200]"
  - "H8 supported (small): magic divisor (umulhi + add + shift, host-computed) vs rl % S in r1: -2.5% at 2.2 MB, -1.0..-2.1% at 6.6-16 MB, -0.5..-1.1% at 36-134 MB, 0 at L [probe_b200]"
  - "run_tests final: 16/16 pass; S -2.6%, M -1.7%, L -0.1% vs r1 on the same B200; 2.2 MB 2.5 vs 2.7, 6.6 MB -3.8%, 57 MB -2.4%, 269 MB (8,24,8) -4.1% / (8,28,4) -1.1%, no workload worse than +0.2%; predicted portal 0.6813 +- 0.0116 [run_tests]"
  - "first draft without occupancy cap (pair at 12 CTAs/SM): S -1.0%, M -0.3%, L -0.1%; 36 MB +2.6%, 57 MB +1.6%, 102 MB +1.3% [run_tests]"
  - "toolchain: nvcc (portal build) and NVRTC (probe) allocate different register counts for identical SASS (40 vs 48); occupancy-sensitive designs must be checked with compile_b200, not probe resources [compile_b200, probe_b200]"
paths:
  - {max_tokens: null, max_rows: 4096, lang: cuda, width: 128, threads: 128, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, max_rows: 330000, max_seq: 1536, lang: cuda, width: 128, threads: 128, rows: 16, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el, rows_per_thread: 2, table_shared_heads: 2, ctas_per_sm_cap: 10, pad_smem: 11500}
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
  - "Bench gains are 1-3%, near noise at S; the portal may show nothing beyond the magic-divisor gain"
  - "The 10 CTAs/SM cap relies on the driver choosing the 132 KB carve-out from a 50% hint; if it picks 228 KB the pair path drops to 18-smem-limited... (actually the reg limit of 12 applies) and loses L1 for tables"
  - "cudaFuncSetAttribute (carve-out preference) is set once on our own kernel; it is ordinary occupancy control, but flag it for review"
  - "Pair path dispatch keys on S <= 1536 and R > 4096, chosen from bench data; shape-only, correct for all shapes (odd H or Hkv fall back to the row kernel)"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```