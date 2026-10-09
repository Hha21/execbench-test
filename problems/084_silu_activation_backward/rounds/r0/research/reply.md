All measurements are in, so I'm writing up the problem card, the ledger and the candidate now. The block-size repeat showed the 512-thread outlier was noise, and torch.compile of the reference runs at 2.1 µs at S and 46.6 µs at 268 MB, which is my best estimate of the hidden baseline.

### Rationale

SiLU backward is a pure elementwise 3-read/1-write fp32 stream (16 B and 5 FLOPs per element), so every workload is memory- or launch-bound. On the rented B200 a float4 one-shot kernel sits at the 2.1–2.4 µs floor for S (empty-kernel span 1.6 µs) and reaches 7.3 TB/s marginal at L once loads carry L2 evict_first and stores evict_last (together worth 10 % at ≥ 34 MB; neutral at S). 128- vs 256-bit, unroll depth and block size all tie. The candidate is that kernel, with the reference's exact op order in IEEE-rounded fp32 so results are bit-identical. Niche-tag note: this problem has no reduction and no rows, so the card uses `red: none` and `tile: elems1024`; the vocabulary needs those two tags for elementwise problems.

```markdown card
# Problem card: L1 #084 `084_silu_activation_backward`

Measurements are from the rented B200 (unlocked clocks; the portal locks SM at 1500 MHz) via probe_b200 and run_tests,
9 October 2026. No portal result yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim maths, `definition.json`, from stable-diffusion-xl):

```python
@torch.no_grad()
def run(grad_output, x, sigmoid_x):
    one_minus_sigmoid = 1.0 - sigmoid_x
    x_times_one_minus_sigmoid = x * one_minus_sigmoid
    bracket_term = 1.0 + x_times_one_minus_sigmoid
    local_grad = sigmoid_x * bracket_term
    return grad_output * local_grad
```

- DPS signature: `run(grad_output, x, sigmoid_x, grad_input)`. All four tensors are `[num_elements]` fp32, contiguous.
- Pure elementwise: `o[i] = g[i] * (s[i] * (1 + x[i] * (1 - s[i])))`. No reduction, no reuse, no broadcast operand.
  Every byte is touched exactly once; 16 B and 5 FLOPs per element.
- `num_elements` ranges from 131 to 16,777,216. Three sizes (131, 2053, 3089) are not multiples of 4, so a vector
  kernel needs a scalar tail (at most 3 elements). The harness keeps pointers 256-B aligned, so 16/32-B vector
  accesses are always legal on the main body.
- Traffic per call: `16 · num_elements` bytes (3 reads + 1 write), 2 KB at the smallest and 268 MB at the largest.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|` for ≥ 99 % of elements, no NaN/Inf, not all zeros [PAPER].
- All three inputs are `randn` (so `sigmoid_x` is NOT in [0,1]; it is N(0,1) too). Outputs are products of up to
  four N(0,1) values and reach magnitude > 100, so rtol dominates; near cancellation in `1 + x(1−s)` the output is
  small and atol covers it.
- Our kernel uses `__fmul_rn/__fadd_rn/__fsub_rn` in the reference's exact order, so `--use_fast_math` cannot
  contract to FMA and the result is **bit-identical to PyTorch eager** (verified: 0 mismatching elements at every
  size tested, max |err| = 0). Any fp32 reordering or FMA would also pass (≤ 2–3 ulp), but exactness is free here.
- What would fail: bf16/fp16/TF32 anywhere (forbidden anyway); leaving the tail elements unwritten (outputs arrive
  zero-filled, 3 of 131 elements = 2.3 % > the 1 % allowance).

## 3. The 16 workloads (sorted by size)

`floor8` = 16n / 8 TB/s. "stream" = plain read+write stream of the same bytes measured on the rented B200 (briefing).
"ref" = PyTorch reference timed like the harness (5 eager kernels). "ours" = run_tests of the first kernel.
"compile" = `torch.compile` of the reference (one Triton kernel), our estimate of what the hidden PyTorch-only
baseline could be.

| n | MB | floor8 µs | stream µs | ref µs | ours µs | ours TB/s | compile µs | band |
|---|---|---|---|---|---|---|---|---|
| 131 | 0.002 | 0.0003 | 2.02 | 17.9 | 2.1 | – | 2.11 | S |
| 2053 | 0.033 | 0.004 | 2.05 | 17.8 | 2.4 | – | – | S |
| 3089 | 0.049 | 0.006 | 2.05 | 17.9 | 2.5 | – | – | S |
| 4096 | 0.066 | 0.008 | 2.05 | 17.8 | 2.4 | – | – | S |
| 16384 | 0.26 | 0.03 | 2.13 | 18.0 | 2.3 | 0.11 | 2.16 | S |
| 40960 | 0.66 | 0.08 | 2.17 | 18.1 | 2.4 | 0.27 | – | M |
| 163840 | 2.62 | 0.33 | 2.59 | 18.2 | 2.8 | 0.94 | 2.72 | M |
| 262144 | 4.19 | 0.52 | 3.14 | 18.5 | 3.1 | 1.35 | – | M |
| 655360 | 10.49 | 1.31 | 4.35 | 21.0 | 4.2 | 2.50 | – | M |
| 786432 | 12.58 | 1.57 | 4.77 | 21.3 | 4.6 | 2.73 | 4.67 | M |
| 2097152 | 33.55 | 4.19 | 8.80 | 27.7 | 7.3 | 4.60 | 8.13 | L |
| 4194304 | 67.11 | 8.39 | 15.83 | 41.4 | 11.8 | 5.69 | – | L |
| 5242880 | 83.89 | 10.49 | 19.55 | 47.0 | 13.8 | 6.08 | – | L |
| 8388608 | 134.22 | 16.78 | 30.00 | 65.4 | 20.0 | 6.71 | 25.8 | L |
| 10485760 | 167.77 | 20.97 | 37.01 | 81.8 | 25.1 | 6.68 | – | L |
| 16777216 | 268.44 | 33.55 | 55.50 | 131.8 | 38.5 | 6.97 | 46.6 | L |

Bands (loop convention): S ≤ 0.3 MB (5 workloads), M 0.66–12.6 MB (5), L 33.5–268 MB (6). Only `num_elements`
matters, so dispatch (if ever needed) keys on it.

## 4. Bounds and what dominates per band

- Arithmetic intensity 5/16 FLOP/B. At 57 TFLOP/s fp32 SIMT (1.5 GHz) the largest workload needs 1.5 µs of compute
  against 33.6 µs of DRAM time: **memory-bound at every size, launch-bound below about 1 MB**.
- Floor at 8 TB/s: 0.0003–33.6 µs; geometric mean of floor8 over the 16 workloads ≈ 0.86 µs, which is meaningless
  for S where the real floor is the launch span. SOLAR likely sits even lower (for #38 it was 0.53× the byte floor);
  **score 1.0 is unreachable, and S-band scores depend almost entirely on the hidden Tb**.
- Measured fixed cost: empty kernel 1.57–1.66 µs CUPTI span; our kernel at 131 elements 2.1–2.3 µs. So the S band
  is one DRAM round trip above the empty-kernel floor and has essentially no headroom.
- Our kernel fits `t ≈ 2.1 µs + bytes / 7.3 TB/s` at L (marginal 134→268 MB: 7.25 TB/s); at 33.5 MB the effective
  rate is only 4.6 TB/s because the fixed 2.1 µs and the dirty-L2 write-back cost (briefing H7) are a large share.
- run_tests compares against a plain copy of the same bytes: ours is 1.05–1.2× the copy at S/M (noise and the tail
  thread) and **0.67–0.83× the copy at L** (the cache hints beat a hint-less copy).

## 5. Where the reference loses and what a fast kernel must do

- Eager PyTorch launches 5 kernels (rsub, mul, add, mul, mul) and materialises 4 intermediates: 17.9 µs at S
  (5 launches × ~3.5 µs with gaps) and 3.4× the bytes at L (131.8 µs at 268 MB ≈ 2.4× our time).
- `torch.compile` fuses it into one Triton kernel: 2.11 µs at S (equal to ours) and 46.6 µs at 268 MB (21 % slower
  than ours, because it uses default cache policies and hits the dirty-L2 write-back cost). If NVIDIA's agentic
  baseline found this, Tb ≈ our time at S and ≈ 1.2× ours at L.
- A fast kernel must: launch once; issue all loads before any use; use L2 evict_first on the three input streams
  and evict_last on the output (measured −10 % at ≥ 34 MB, neutral at S); keep registers ≤ 32 (ours: 24) so 2048
  threads per SM are resident; write the ≤ 3 tail elements.

## 6. Design priorities

1. **M/L bandwidth under the dirty-L2 regime** is the only lever with headroom: 6.7–7.0 TB/s now vs the 8 TB/s
   floor. Candidates: compare EF+EL against `.cs` loads (tied in probes), 256-bit loads with EF (`ld.global.nc.L2::evict_first`
   qualifier form), and whether partial/fractional policies shift the balance. Expect ≤ 3 %.
2. **S band: nothing to gain** (2.1–2.4 µs vs 1.6 µs empty). Keep the S path minimal; do not add prologues,
   persistent loops or TMA. Check on the portal whether the tail-thread path costs anything at 2053/3089 (run_tests
   shows 2.4–2.5 vs 2.1 at 131, probably noise).
3. No per-size dispatch is needed: one config (256 threads, 1 float4 per thread) ties the best at every size.
4. Avoid: TMA/bulk rings (+1 µs fixed, refuted on #38), persistent grids, clusters, 256-bit accesses at tiny sizes
   (+0.8 µs at 131 elements, cause unknown).

## 7. Results so far

First kernel `silu-bw-cuda-ldg128-os-efel` (CUDA C++, two files): 16/16 PASSED on the rented B200 (run_tests,
20.8 s). Times in §3 "ours". sm_100a: 24 regs, 0 smem, 0 spills, 3 × LDG.128 + 1 × STG.128 per thread.
Probe ladder at 268 MB: plain 43.0 µs → EL stores 40.2 → EF loads + EL stores 38.8 (= `.cs` loads 38.9;
`nc`+EF 38.9; `L2::256B` prefetch hint 40.2, no gain; `nc` alone 40.2). Block size 128/256/512/1024 and unroll
1/2/4: all within ±1 % (3 repeats), so noise. Portal submission pending for Tb/Tsol anchors.

## 8. Open questions

- What is the hidden Tb per workload? If it is torch.compile-like (2.1 µs at S), S scores are capped near 0.5 and
  the 6 L workloads decide the problem. If it is eager-like (18 µs), S scores approach 0.9.
- Does the portal's 1500 MHz SM lock change the S floor (empty span 1.6 µs here at unlocked clocks)?
- Why do 256-bit loads cost +0.8 µs at 131 elements (one CTA, 16 active lanes)? Irrelevant for the design but
  worth understanding before any 256-bit variant.
- Can anything reduce the dirty-L2 write-back cost further than EF loads + EL stores (e.g. ordering the output
  stores before loads of later tiles, or fractional evict_last)? The remaining L gap to 8 TB/s is about 5 µs at
  268 MB.
```

```yaml ledger
# Hypothesis ledger for #084 silu_activation_backward, seeded 9 October 2026 by the research session.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
hypotheses:
- id: H1
  statement: L2 evict_last on the output stores cuts in-window dirty write-backs, gaining 5-7% at L and 3% at 12.6 MB.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB plain 43.0 us vs EL stores 40.2; 134 MB 23.65 vs 21.5; 33.5 MB 7.83 vs 7.41; 12.6 MB 4.68 vs 4.51; S ties'
- id: H2
  statement: L2 evict_first on the three input streams adds a further 3-4% at L on top of EL stores, because reads are 75% of
    the traffic here (unlike #38's 50%), so our own clean lines become the victims instead of the flush's dirty lines.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB EL 40.2 -> EF+EL 38.8; 134 MB 21.5 -> 20.1; 83.9 MB 14.8 -> 13.8; 33.5 MB 7.41 -> 7.10; S ties'
  - 'probe_b200 r0: ld.global.cs and nc+EF and L1::no_allocate+EF all equal EF (within 0.1 us); L2::256B prefetch hint = no gain'
- id: H3
  statement: 256-bit loads/stores do not beat 128-bit for this stream, and cost +0.8 us at 131 elements.
  status: supported
  evidence:
  - 'probe_b200 r0: k8 vs k4 tie at 12.6-268 MB (within 1%); at 131 elements k8 3.06 us vs k4 2.29, repeatable across block sizes; cause unknown'
- id: H4
  statement: The S band (<= 0.3 MB) is at the launch floor; no kernel-side change can gain more than ~0.2 us there.
  status: supported
  evidence:
  - 'probe_b200 r0: empty kernel 1.57-1.66 us; float4/scalar kernels 2.2-2.4 us at 131-16384 elements for every block size, unroll and hint tried'
- id: H5
  statement: The hidden baseline Tb is a torch.compile-fused kernel (~2.1 us at S, ~1.2x our time at L), so S workloads score
    about 0.5 regardless and the problem score is decided by the 6 L workloads and 5 M workloads.
  status: open
  evidence:
  - 'probe_b200 r0: torch.compile(reference) 2.11 us at 131, 2.72 at 163840, 4.67 at 786432, 8.13 at 2.1M, 25.8 at 8.4M, 46.6 at 16.8M; eager reference 17.9-131.8 us'
  - 'needs the first portal submission (Tb per workload)'
- id: H6
  statement: More bytes in flight per thread (unroll 2-4, 512-1024 threads) raises L-band bandwidth.
  status: refuted
  evidence:
  - 'probe_b200 r0: with EF+EL hints, block 128/256/512/1024 x unroll 1/2/4 all within +-1% at 67-268 MB over 3 repeats; the 512x2 37.4 us reading did not reproduce (38.4-38.5)'
- id: H7
  statement: The marginal bandwidth ceiling for this 3:1 read:write stream under the dirty-L2 flush is ~7.3 TB/s; the remaining
    ~5 us at 268 MB (vs 8 TB/s) is dirty write-back traffic that no load/store hint removes.
  status: open
  evidence:
  - 'run_tests r0: 134 MB 20.0 us, 268 MB 38.5 us -> marginal 7.25 TB/s; fit t = 2.1 us + bytes/7.3 TB/s'
  - 'untested: a clean-flush comparison (as in #38 r12 E1) to size the dirty cost for this traffic mix'
- id: H8
  statement: Per-size dispatch is unnecessary; one config (256 threads, one float4 per thread, EF+EL) is within noise of the
    best at every size.
  status: supported
  evidence:
  - 'probe_b200 r0: all configs tie at S/M; H6 ties at L; run_tests r0 16/16 with one config'
- id: H9
  statement: TMA/bulk rings, persistent grids, clusters or weight-style prefetch could help this problem.
  status: parked
  evidence:
  - '#38 evidence: TMA +1-1.5 us fixed, persistent slower at every size, clusters lost 2-5%; nothing in this problem (no reduction, no reuse) changes that reasoning'
```

```json solution-spec
{"name": "silu-bw-cuda-ldg128-os-efel", "definition": "084_silu_activation_backward", "author": "solx-loop",
 "description": "Fused fp32 SiLU backward: one float4 per thread, one-shot grid, L2 evict_first loads and evict_last stores to minimise in-window dirty write-backs after the harness's zero-fill flush.",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// SiLU backward: grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x)), fp32 elementwise.
// One fused launch. 128-bit loads/stores, one float4 per thread, one-shot grid.
// Loads carry L2 evict_first, stores L2 evict_last: the harness's zero-fill flush leaves L2 full of dirty lines,
// and this pairing minimises the write-backs that land inside the timed window (measured on B200: -10% at >= 34 MB).
#include <cuda_runtime.h>
#include <stdint.h>

// Same operation order as the PyTorch reference, with IEEE-rounded ops (no FMA contraction), so results are bit-identical.
__device__ __forceinline__ float silu_bw(float g, float x, float s) {
    return __fmul_rn(g, __fmul_rn(s, __fadd_rn(1.0f, __fmul_rn(x, __fsub_rn(1.0f, s)))));
}

__device__ __forceinline__ float4 ld4_hint(const float* p, unsigned long long pol) {
    float4 v;
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void st4_hint(float* p, float4 v, unsigned long long pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol) : "memory");
}

static constexpr int THREADS = 256;

// Vector path: thread i handles floats [4i, 4i+4). The thread with i == nvec handles the < 4 tail elements.
__global__ void __launch_bounds__(THREADS)
silu_bw_vec4(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
             float* __restrict__ o, long long n) {
    const long long nvec = n >> 2;
    const long long i = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (i < nvec) {
        unsigned long long pol_ld, pol_st;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol_ld));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol_st));
        const float4 a = ld4_hint(g + (i << 2), pol_ld);
        const float4 b = ld4_hint(x + (i << 2), pol_ld);
        const float4 c = ld4_hint(s + (i << 2), pol_ld);
        float4 r;
        r.x = silu_bw(a.x, b.x, c.x);
        r.y = silu_bw(a.y, b.y, c.y);
        r.z = silu_bw(a.z, b.z, c.z);
        r.w = silu_bw(a.w, b.w, c.w);
        st4_hint(o + (i << 2), r, pol_st);
    } else if (i == nvec) {
        for (long long e = nvec << 2; e < n; ++e) o[e] = silu_bw(g[e], x[e], s[e]);
    }
}

// Scalar fallback for pointers that are not 16-B aligned (never the case under the harness).
__global__ void __launch_bounds__(THREADS)
silu_bw_scalar(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
               float* __restrict__ o, long long n) {
    const long long i = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (i < n) o[i] = silu_bw(g[i], x[i], s[i]);
}

extern "C" void silu_bw_launch(const float* g, const float* x, const float* s, float* o, long long n,
                               cudaStream_t stream) {
    if (n <= 0) return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(x) |
                          reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(o)) & 15u) == 0;
    if (aligned) {
        const long long nthreads = (n >> 2) + ((n & 3) ? 1 : 0);
        const unsigned grid = (unsigned)((nthreads + THREADS - 1) / THREADS);
        silu_bw_vec4<<<grid, THREADS, 0, stream>>>(g, x, s, o, n);
    } else {
        const unsigned grid = (unsigned)((n + THREADS - 1) / THREADS);
        silu_bw_scalar<<<grid, THREADS, 0, stream>>>(g, x, s, o, n);
    }
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" void silu_bw_launch(const float* g, const float* x, const float* s, float* o, long long n,
                               cudaStream_t stream);

// DPS: run(grad_output, x, sigmoid_x, grad_input) writes grad_input in place.
void run(torch::Tensor grad_output, torch::Tensor x, torch::Tensor sigmoid_x, torch::Tensor grad_input) {
    TORCH_CHECK(grad_output.is_cuda() && x.is_cuda() && sigmoid_x.is_cuda() && grad_input.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32 &&
                sigmoid_x.scalar_type() == torch::kFloat32 && grad_input.scalar_type() == torch::kFloat32, "fp32 expected");
    TORCH_CHECK(grad_output.is_contiguous() && x.is_contiguous() && sigmoid_x.is_contiguous() && grad_input.is_contiguous(),
                "contiguous tensors expected");
    const long long n = grad_output.numel();
    TORCH_CHECK(x.numel() == n && sigmoid_x.numel() == n && grad_input.numel() == n, "numel mismatch");
    const c10::cuda::CUDAGuard guard(grad_output.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_bw_launch(grad_output.data_ptr<float>(), x.data_ptr<float>(), sigmoid_x.data_ptr<float>(),
                   grad_input.data_ptr<float>(), n, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "SiLU backward (DPS)"); }
```

```yaml design-card
id: silu-bw-cuda-ldg128-os-efel
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: fused
  tile: elems1024
  red: none
  cache: stream
  spec: all
hypothesis: >-
  A pure 3-read/1-write fp32 stream is at the launch floor for S and bandwidth-bound for M/L; on B200 the harness's
  dirty L2 makes cache policy the only lever, and evict_first loads plus evict_last stores cut in-window write-backs
  by 10% at >= 34 MB while costing nothing at S. One float4 per thread, 256-thread CTAs, one-shot grid, exact
  reference op order in IEEE fp32 (bit-identical output).
expected_effect:
  S: {pct: 0, confidence: high}
  M: {pct: -3, confidence: medium}
  L: {pct: -10, confidence: high}
resources_sm100a:
  regs_per_thread: 24
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs: {THREADS: 256, FLOATS_PER_THREAD: 4, ELEMS_PER_CTA: 1024, LOAD_HINT: evict_first, STORE_HINT: evict_last}
tests: [H1, H2, H4, H8]
findings:
  - "probe: empty-kernel CUPTI span 1.57-1.66 us; float4/scalar one-shot kernels 2.2-2.4 us at 131-16384 elements for every block size, unroll and hint: S band is at the floor [probe_b200]"
  - "probe: 256-bit loads tie 128-bit at 12.6-268 MB but cost +0.8 us at 131 elements (3.06 vs 2.29 us), repeatable across block sizes, cause unknown [probe_b200]"
  - "probe: at 268 MB plain 43.0 us -> EL stores 40.2 -> EF loads + EL stores 38.8; 134 MB 23.65 -> 21.5 -> 20.1; 33.5 MB 7.83 -> 7.41 -> 7.10; S unchanged [probe_b200]"
  - "probe: .cs loads, nc+EF, L1::no_allocate+EF all equal EF+EL within 0.1 us; L2::256B prefetch hint and nc alone give no gain over EL-only [probe_b200]"
  - "probe: block 128/256/512/1024 x unroll 1/2/4 with EF+EL all within +-1% at 67-268 MB over 3 repeats; a 512x2 outlier (37.4 us) did not reproduce [probe_b200]"
  - "probe: torch.compile(reference) is a plausible hidden baseline: 2.11 us at 131 elements (equal to ours), 8.13 at 33.5 MB, 25.8 at 134 MB, 46.6 at 268 MB (ours 7.3/20.0/38.5); eager reference 17.9-131.8 us [probe_b200]"
  - "run_tests: 16/16 passed, bit-identical to eager; 2.1-2.5 us at S, 2.4-4.6 at M, 7.3-38.5 at L; 0.67-0.83x a plain copy at L, 1.05-1.2x at S/M [run_tests]"
  - "compile: 24 regs, 0 smem, 0 spills, 3x LDG.128 + 1x STG.128 per thread on sm_100a [compile_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 1, grid: oneshot, mem: ldg, launches: 1, x: ef, st: el}
runs_on:
  H200: {runs: true, representative: true, note: "createpolicy/L2::cache_hint exist on sm_90a; dirty-L2 effect size differs with the 50 MB L2"}
  A100: {runs: true, representative: false, note: "createpolicy needs sm_80+; L2::cache_hint on ld/st is sm_80+ too, but 40 MB L2 changes the write-back regime"}
  L40S: {runs: true, representative: false}
risks:
  - "If the hidden baseline is torch.compile-like, S workloads score ~0.5 no matter what; the problem score rides on M/L"
  - "The tail thread for n % 4 != 0 (131, 2053, 3089) runs a 1-3 element scalar loop; measured cost is within noise but unverified at the portal's 1500 MHz"
  - "The scalar fallback kernel is never launched under the harness (256-B aligned pointers); dispatch on alignment is pointer-derived, not value- or count-derived"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```

The first kernel passed all 16 workloads on the rented B200 and is ready for the portal submission that will reveal the hidden baseline and SOL anchors. The key research results: the S band sits at the launch floor with nothing left to gain, evict_first loads plus evict_last stores are worth 10 percent at large sizes, and torch.compile of the reference matches our S-band time, so the problem score will most likely be decided by the medium and large workloads.