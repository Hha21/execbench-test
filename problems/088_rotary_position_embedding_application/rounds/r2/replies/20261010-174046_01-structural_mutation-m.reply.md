H13 is refuted on the B200 test bench. None of the register-pipelined persistent configurations beat r1 by 3% at 36–134 MB, and the best one loses 1.4–4.4% at the four sizes it serves in run_tests. As instructed, I'm returning that best version with the evidence; it should not go to the portal unless a second reading of H15 (bench vs portal penalty) is wanted.

### Rationale
I tested E2/H13 on the rented B200, all runs interleaved with 3 repeats:

- **Static persistent loses, and more so with size.** Every grid of 148·k CTAs (linear or s-stationary, pipeline depth 0, 1 or 2, 64–256 threads, k = occupancy or occupancy − 1) is slower than r1 at 134–612 MB, by 3–14%. That includes one at r1's own occupancy (32 regs, 16 CTAs/SM, no prefetch).
- **It's a bandwidth-rate loss, not a fixed cost.** The gap grows with size. My explanation is that a static per-SM split can't adapt to uneven per-SM bandwidth, while the hardware's CTA scheduler can; I haven't proven this.
- **Work stealing doesn't fix it.** Taking chunks from an atomic ticket counter adds about 15–20 µs of fixed cost from same-address atomic contention.

The best variant (s-stationary, depth 1, 48 regs, k = occupancy − 1) ties r1 within about 2% at 36–102 MB, so it is routed only there.

```json solution-spec
{"name": "r2-rope-psst-m", "definition": "088_rotary_position_embedding_application", "author": "solx-loop", "description": "r1 dispatcher plus an s-stationary register-pipelined persistent path for 32k-110k rows (E2/H13 test; bench says H13 refuted)", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// RoPE application (Q and K), fp32, head_dim 128. One fused launch per call.
// Layout: 16 lanes per row; lane j owns columns [4j, 4j+4) and [64+4j, 64+4j+4) (one 128-bit load each),
// so the rotate_half partner is already in the same thread and no shuffles are needed.
// out[i]    = x[i]*cos[i]    - x[i+64]*sin[i]        (i < 64)
// out[i+64] = x[i+64]*cos[i+64] + x[i]*sin[i+64]
// Paths: one-shot (r1) everywhere except 32k-110k rows, which use an s-stationary persistent grid:
// each 16-lane group keeps one sequence position s (cos/sin row in registers, loaded once) and walks the
// rows s + S*h, prefetching the next row's x into a second register set before computing the current row.
#include <cuda_runtime.h>
#include <stdint.h>

#define LD4(p, v) asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define LDX4(p, v) asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "l"(p))
#define ST4EL(p, v, pol) asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]), "l"(pol) : "memory")

struct Row { float xl[4], xh[4], cl[4], ch[4], sl[4], sh[4]; };

__device__ __forceinline__ void ld_x(const float* q, const float* k, unsigned r, unsigned Rq, unsigned j, Row& w) {
    const float* xp = (r < Rq ? q + (size_t)r * 128 : k + (size_t)(r - Rq) * 128) + j * 4;
    LDX4(xp, w.xl); LDX4(xp + 64, w.xh);
}
__device__ __forceinline__ void ld_tab(const float* cs, const float* sn, unsigned s, unsigned j, Row& w) {
    const float* cp = cs + (size_t)s * 128 + j * 4;
    const float* sp = sn + (size_t)s * 128 + j * 4;
    LD4(cp, w.cl); LD4(cp + 64, w.ch); LD4(sp, w.sl); LD4(sp + 64, w.sh);
}
__device__ __forceinline__ void st_row(float* qo, float* ko, unsigned r, unsigned Rq, unsigned j, const Row& w,
                                       unsigned long long pol) {
    float* y = (r < Rq ? qo + (size_t)r * 128 : ko + (size_t)(r - Rq) * 128) + j * 4;
    float ol[4], oh[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        ol[e] = w.xl[e] * w.cl[e] - w.xh[e] * w.sl[e];
        oh[e] = w.xh[e] * w.ch[e] + w.xl[e] * w.sh[e];
    }
    ST4EL(y, ol, pol);
    ST4EL(y + 64, oh, pol);
}

#define ARGS const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ cs, \
             const float* __restrict__ sn, float* __restrict__ qo, float* __restrict__ ko

// r1 one-shot: one row per 16-lane group.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
rope_kernel(ARGS, unsigned Rq, unsigned R, unsigned S)
{
    const unsigned j = threadIdx.x & 15;
    const unsigned r = blockIdx.x * (THREADS / 16) + (threadIdx.x >> 4);
    if (r >= R) return;
    Row w;
    ld_x(q, k, r, Rq, j, w);
    ld_tab(cs, sn, r % S, j, w);        // Rq is a multiple of S, so s = r % S for Q and K rows alike
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    st_row(qo, ko, r, Rq, j, w, pol);
}

// s-stationary persistent: group g -> s = g % S, c = g / S; rows s + S*(c + C*i), i = 0, 1, ...
// Pipeline depth 1 in registers (two x sets, one table set), no shared memory, no barriers.
__global__ void __launch_bounds__(128)
rope_psst(ARGS, unsigned Rq, unsigned R, unsigned S, unsigned C)
{
    const unsigned j = threadIdx.x & 15;
    const unsigned g = blockIdx.x * 8 + (threadIdx.x >> 4);
    if (g >= S * C) return;
    const unsigned s = g % S, c = g / S;
    const unsigned step = C * S;
    unsigned r = s + c * S;
    if (r >= R) return;
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    Row A, B;
    ld_x(q, k, r, Rq, j, A);
    ld_tab(cs, sn, s, j, A);
#pragma unroll
    for (int e = 0; e < 4; ++e) { B.cl[e] = A.cl[e]; B.ch[e] = A.ch[e]; B.sl[e] = A.sl[e]; B.sh[e] = A.sh[e]; }
    for (;;) {
        const unsigned r1 = r + step;
        if (r1 < R) ld_x(q, k, r1, Rq, j, B);
        st_row(qo, ko, r, Rq, j, A, pol);
        if (r1 >= R) break;
        const unsigned r2 = r1 + step;
        if (r2 < R) ld_x(q, k, r2, Rq, j, A);
        st_row(qo, ko, r1, Rq, j, B, pol);
        if (r2 >= R) break;
        r = r2;
    }
}

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            cudaStream_t stream)
{
    static int n_sm = 0, occ = 0;        // device properties, cached once (shape-independent)
    if (n_sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, rope_psst, 128, 0);
        if (occ < 1) occ = 1;
    }
    if (R > 32768u && R <= 110000u) {    // 36-102 MB: persistent, k = occupancy - 1 CTAs per SM
        const unsigned NH = R / S;       // rows per sequence position (B*(H+Hkv))
        const unsigned k_sm = occ > 1 ? (unsigned)occ - 1u : 1u;
        unsigned Ct = (unsigned)n_sm * k_sm * 8u / S;
        if (Ct < 1u) Ct = 1u;
        if (Ct > NH) Ct = NH;
        const unsigned per = (NH + Ct - 1) / Ct;     // rows per group, balanced
        const unsigned C = (NH + per - 1) / per;
        const unsigned grid = (S * C + 7) / 8;
        rope_psst<<<grid, 128, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S, C);
    } else if (R <= 262144u) {           // S/M band: 8 rows per 128-thread CTA
        const unsigned grid = (R + 7) / 8;
        rope_kernel<128><<<grid, 128, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
    } else {                             // L band: 16 rows per 256-thread CTA
        const unsigned grid = (R + 15) / 16;
        rope_kernel<256><<<grid, 256, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
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
                            cudaStream_t stream);

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
    TORCH_CHECK(key.size(2) == query.size(2), "seq_len mismatch");
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
                (unsigned)Rq, (unsigned)R, (unsigned)S, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "RoPE application (DPS)"); }
```

```yaml design-card
id: r2-rope-psst-m
parents:
  - r1-rope-v4nc-s128
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: persistent-wstat
  launch: fused
  tile: rows8
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  H13 test. A register-pipelined persistent grid (148 x k CTAs, next row's loads issued before the current row's
  FMAs, no smem, no barriers) should remove the per-CTA replacement gap and beat r1's one-shot grid at 36-134 MB.
  Bench result: refuted. Static persistent grids lose a share of bandwidth that grows with size (+3..+14% at the
  same occupancy as r1), likely because a fixed per-SM split cannot follow uneven per-SM bandwidth the way dynamic
  CTA dispatch does. The best variant (s-stationary, depth 1, 48 regs) ties r1 within about 2% at 36-102 MB and is
  routed only there.
expected_effect:
  S:
    pct: 2
    confidence: medium
  M:
    pct: 1
    confidence: medium
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 48
  smem_per_cta_bytes: 0
  threads_per_cta: 128
  ctas_per_sm: 9
  bytes_in_flight_per_sm: 73728
  launches_per_call: 1
knobs:
  persistent_rows_min: 32769
  persistent_rows_max: 110000
  k_sm: occupancy_minus_1
  pipeline_depth: 1
  THREADS_P: 128
  THREADS_S: 128
  THREADS_L: 256
  row_threshold: 262144
  store_hint: evict_last
dispatch:
  - {max_rows: 32768, kernel: rope_kernel<128>, meta: {threads: 128, rows: 8}}
  - {max_rows: 110000, kernel: rope_psst, meta: {threads: 128, k: occupancy-1, depth: 1}}
  - {max_rows: 262144, kernel: rope_kernel<128>, meta: {threads: 128, rows: 8}}
  - {max_rows: null, kernel: rope_kernel<256>, meta: {threads: 256, rows: 16}}
tests:
  - H13
findings:
  - "H13 refuted on the bench: no register-only persistent configuration at <= 64 regs beats r1 by >= 3% at 36-134 MB; at 134-612 MB all are +3..+14% slower [probe_b200, 3 sessions]"
  - "registers/occupancy (sm_100a): linear 2-set pipeline 72 regs (128 thr, 7 CTAs/SM) / 77 (256 thr, 3); with __launch_bounds__(128,8) 64 regs; s-stationary depth 1 48 regs (10 CTAs/SM at 128 thr, 20 at 64), depth 2 56 regs (9); all 0 spills, 0 smem [probe_b200 resources + cuOccupancy]"
  - "linear persistent vs r1 (36/61/102/134/269 MB): plin128 k7 +4.9/+5.2/+6.1/+4.7/+10.1%; plin128 64-reg k8 +1.5/+3.5/+4.8/+4.1/+10.3%; plin256 k3 +7.3/+9.0/+7.6/+5.3/+8.4% [probe_b200]"
  - "s-stationary vs r1: psst128 k9 6.6 MB +1.9%, 15.8 +1.4, 36 +1.8 (-3.5 in another session), 57 -0.4, 61 +0.8, 102 +1.2, 134 +4.1, 269 +8.3, 612 +3.5%; k10 is worse than k9; depth 2 (k7/k8) and 64-thread k20 are no better [probe_b200]"
  - "dead end, key result: persistent with NO prefetch at r1's own occupancy (32 regs, 16 CTAs/SM, grid 148x16) is +3.6% at 6.6 MB, +7.8 at 15.8, +7 at 61, +6.5 at 102, +7.2 at 134, +10.6 at 269, +13.8% at 612 MB. Removing CTA replacement costs bandwidth; the loss grows with size, so it is a rate deficit and not a fixed cost [probe_b200]"
  - "replacement gap: empty one-shot grids (R/8 CTAs of 128 thr) span 1.6 us at 6.6 MB, 3.2 us at 36 MB, 9.4 us at 134 MB, 18 us at 269 MB and 39 us at 612 MB; an empty persistent grid (148x16) spans 2.2 us at every size. Dispatch overlaps memory work: r1 at 269 MB (39 us) is not dispatch-bound, and r1 time minus persistent time is negative at every size [probe_b200]"
  - "dead end: dynamic work-stealing persistent (warp tickets on one global atomic counter, last warp resets it; chunks of 8 or 32 rows, with or without row prefetch) adds about 15-20 us fixed cost: +250..+390% at 15.8 MB, +37..+56% at 269 MB, +17..+34% at 612 MB. Same-address atomics from 2-4.7k warps serialise [probe_b200]"
  - "run_tests (final, 16/16 pass) vs r1 on the same B200: persistent-path workloads 36 MB 6.8 vs 6.6 (+4.4%), 57 MB 9.0 vs 8.9 (+1.4%), 61 MB 9.6 vs 9.4 (+2.3%), 102 MB 15.1 vs 14.8 (+2.2%); other workloads use identical code (-0.3..+0.2% at M/L, noise at 2.2/6.6 MB); predicted portal 0.676 vs 0.677 [run_tests]"
  - "implication for later rounds: on this B200, hardware CTA dispatch acts as a free dynamic load balancer for 128-float row streams; persistent designs (static or atomic-dynamic) should be parked for #88 unless they raise bytes in flight per SM well beyond r1's ~64 KB of x at no occupancy cost"
paths:
  - {max_tokens: null, max_rows: 32768, lang: cuda, width: 128, threads: 128, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, max_rows: 110000, lang: cuda, width: 128, threads: 128, rows: 8, grid: persistent, mem: ldg, wstat: true, launches: 1, prefetch: true, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, max_rows: 262144, lang: cuda, width: 128, threads: 128, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
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
  - "Bench says the persistent path is 1-4% slower at 36-102 MB; a portal slot would only be a second reading of H15 (whether a design with fewer CTA replacements loses less under the locked clock), with an expected score at or below r1's 0.677"
  - "k is read from cudaOccupancyMaxActiveBlocksPerMultiprocessor on the running device (48 regs -> 10 at 128 threads, k = 9); the static cache is a device property, not per-call state"
  - "the s-stationary mapping needs Rq and R to be multiples of S, which holds for every [B,H,S,128] shape"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```