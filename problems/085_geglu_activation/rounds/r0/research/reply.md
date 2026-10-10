All 16 workloads pass, and the kernel beats the plain copy at every size (0.69 to 0.95 of the copy's time). The results below give the problem card, the hypothesis ledger, and the first kernel.

```markdown card
# Problem card: L1 #85 `085_geglu_activation`

Definition and workloads from the task prompt (SOL-ExecBench v1.1). Measurements are from the rented B200 (unlocked
clocks, SM up to 1965 MHz; the portal locks 1500 MHz) through `probe_b200` and `run_tests`, 10 October 2026. No portal
submission yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim, `definition.json`), from stable-diffusion-xl-refiner-1.0:

```python
@torch.no_grad()
def run(x):
    x_gate, x_linear = x.chunk(2, dim=-1)
    return F.gelu(x_gate, approximate='tanh') * x_linear
```

- DPS signature: `run(x, output)`. `x`: `[B, S, 10240]` fp32 contiguous; `output`: `[B, S, 5120]` fp32.
- View `x` as `[R, 10240]` with `R = B·S` rows (128 to 8192). Row r: gate half `x[r, 0:5120]`, linear half
  `x[r, 5120:10240]`, 20 KB apart. `out[r, j] = gelu_tanh(x[r, j]) · x[r, 5120 + j]`.
- Pure elementwise: no reduction, no reuse, every byte touched once. 12 B of traffic per output element (8 read, 4
  written). Traffic per call = `B·S·61,440` bytes.
- Identity used by the kernel: `0.5·a·(1 + tanh(u)) = a · sigmoid(2u)`, so gelu·b = `a·b / (1 + exp(−2u))` with
  `u = sqrt(2/π)·(a + 0.044715·a³)`. One exp and one reciprocal per element, no cancellation near a = 0.
- 5120 = 2^10·5, so every row is 640 chunks of 8 floats, or 5 CTAs of 128 threads × 8 floats. No masks, no tails,
  no row with a size that needs a tail check for any B·S.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|`, ≥ 99% of elements, no NaN/Inf [PAPER: workload.jsonl]. Inputs are
  `randn`, so |out| reaches about 25 and the relative budget is about 84 ulp.
- The reference computes tanhf (≤ 2 ulp) in fp32 and materialises gelu, then multiplies. Our sigmoid form with
  `__expf` (ex2.approx) and `__fdividef` under `--use_fast_math` measured: max abs error 1.9e-6, 0 elements out of
  tolerance at 128 and 2048 rows [probe]. The `tanhf` form measured 9.5e-7 max abs error and was 3 to 10% slower
  (34 regs, more MUFU work) [probe]. Both have ≥ 5× margin on the absolute term.
- For u < −44, exp(−2u) overflows to inf and the result is exactly 0; the reference also gives 0 there
  (tanh = −1 exactly). No NaN path for finite inputs.
- `--use_fast_math` does not change the accuracy of either form (measured with and without) [probe].
- What would fail: bf16/fp16/TF32 anywhere; `tanh.approx.f32` (2^-11 relative, about 2000 ulp); unwritten rows.

## 3. The 16 workloads (sorted by size)

Bytes = read 8·B·S·5120 + write 4·B·S·5120. `floor8` = bytes / 8 TB/s. `copy` = the loop's plain copy of the same
bytes on the rented B200 (from `run_tests`). `ours` = r0-geglu-ldg256-2d-stel (`run_tests`, rented B200).
`ref` = PyTorch reference timed like the harness (task prompt).

| B,S | B·S rows | MB | band | floor8 µs | copy µs | ours µs | ours TB/s | ref µs | ref/ours |
|---|---|---|---|---|---|---|---|---|---|
| 1,128 | 128 | 7.86 | S | 0.98 | 3.9 | 3.6 | 2.2 | 11.98 | 3.3 |
| 1,131 | 131 | 8.05 | S | 1.01 | 3.9 | 3.7 | 2.2 | 11.93 | 3.2 |
| 2,211 | 422 | 25.93 | S | 3.24 | 7.4 | 6.2 | 4.2 | 20.49 | 3.3 |
| 1,512 | 512 | 31.46 | S | 3.93 | 8.3 | 7.0 | 4.5 | 21.89 | 3.1 |
| 4,128 | 512 | 31.46 | S | 3.93 | 8.2 | 6.9 | 4.6 | 22.06 | 3.2 |
| 1,1024 | 1024 | 62.91 | M | 7.86 | 14.4 | 10.8 | 5.8 | 37.77 | 3.5 |
| 2,512 | 1024 | 62.91 | M | 7.86 | 14.4 | 10.9 | 5.8 | 37.76 | 3.5 |
| 1,2048 | 2048 | 125.83 | M | 15.73 | 26.8 | 18.6 | 6.8 | 70.93 | 3.8 |
| 8,256 | 2048 | 125.83 | M | 15.73 | 27.0 | 18.7 | 6.7 | 70.86 | 3.8 |
| 4,541 | 2164 | 132.96 | M | 16.62 | 28.2 | 19.6 | 6.8 | 75.11 | 3.8 |
| 16,256 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.2 | 7.1 | 134.88 | 3.8 |
| 4,1024 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.1 | 7.2 | 135.20 | 3.9 |
| 32,128 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.1 | 7.2 | 135.26 | 3.9 |
| 8,613 | 4904 | 301.30 | L | 37.66 | 59.8 | 42.5 | 7.1 | 160.01 | 3.8 |
| 16,449 | 7184 | 441.38 | L | 55.17 | 83.6 | 62.8 | 7.0 | 229.75 | 3.7 |
| 8,1024 | 8192 | 503.32 | L | 62.92 | 93.8 | 71.7 | 7.0 | 261.78 | 3.7 |

Only B·S matters (the three 4096-row shapes time identically), so 16 workloads are 11 sizes. Geometric mean of
floor8: 11.9 µs. Geometric mean of ours: 18.8 µs.

Bands (loop convention): S ≤ 31.5 MB (5 workloads, 128–512 rows), M ≤ 133 MB (5 workloads, 1024–2164 rows),
L above (6 workloads, 4096–8192 rows).

## 4. Bounds and what dominates per band

- **Memory-bound everywhere.** FLOPs per element about 12 (cubic, exp, reciprocal, 3 muls) → 0.67e12 elements/s at
  8 TB/s needs about 8 TFLOP/s against 57 TFLOP/s FP32 SIMT at 1.5 GHz. MUFU: 2 ops per element → 1.3e12/s against
  3.55e12/s at 1.5 GHz (37% of MUFU capacity; 28% at the rented clock). A copy-only variant (no GELU maths) timed
  identically to the full kernel at every size on the rented B200 [probe], so compute is hidden at 1965 MHz. At the
  portal's 1500 MHz the SM-side budget shrinks by 24%; MUFU at 37% should still hide, but it is the one compute risk
  (H5).
- **S band (7.9–31 MB):** fixed cost dominates. Empty kernel CUPTI span 1.73 µs; read-only kernel 3.5 µs and full
  kernel 3.6–3.8 µs at 7.9 MB, against a 0.98 µs byte floor [probe]. About 2.7 µs of fixed cost per call.
- **M band (63–133 MB):** mixed. 5.8–6.8 TB/s achieved; the dirty L2 left by the harness's flush and the store
  policy decide (evict_last stores gained 8–13% here).
- **L band (252–503 MB):** sustained bandwidth. 7.0–7.2 TB/s with evict_last stores, matching the 7.05–7.1 TB/s
  public ceiling for 1:1 read+write streams on B200 (`b200_sota.md` §2). Reads alone reach only 5.3 TB/s at 503 MB
  after the dirty flush [probe], the same effect #38 recorded as H7.
- **SOL (Tsol):** unknown until the first portal result. If SOLAR uses bytes/8 TB/s, our geomean is 1.58× it. On #38
  Tsol sat at about 0.53× the byte floor at M/L, so expect a similar unreachable anchor here.

## 5. Where the reference loses and what a fast kernel must do

- The reference runs two kernels: `gelu` on the strided gate view (read 4 B, write 4 B per element to a temporary),
  then `mul` (read 8 B, write 4 B). That is 20 B per element instead of 12, plus a temporary allocation, a second
  launch and a gap. Measured 3.1–3.9× our time.
- A fast kernel: one launch; each thread loads 8 gate floats and 8 linear floats with 256-bit loads, computes, stores
  8 floats with one 256-bit store under `L2::evict_last`; a one-shot grid with every load issued before any use.

## 6. Design priorities

1. **Keep evict_last on stores.** Measured gain at M/L (unlocked B200): 63 MB −8%, 126 MB −13%, 252 MB −12%,
   503 MB −6%; 0 at S [probe]. Same mechanism as #38 H1 (fewer in-window write-backs of the flush's dirty lines).
2. **One chunk of 8 floats per thread, small CTAs.** 2 chunks/thread +1–5%, 4 chunks/thread +3–10% (82 regs) [probe].
   CTA size 128/256/512 within 1% [probe]. 2-D grid (5 CTAs per row, no division) −2% at S versus a 1-D grid with
   div-by-640, equal at L [probe].
3. **S-band fixed cost (about 2.7 µs above the byte floor at 7.9 MB).** Largest score headroom; the lever set from #38
   applies (no prologue, all loads up front). The empty-kernel span of 1.73 µs is 45% of the S time.
4. **Compute at 1.5 GHz.** If H5 shows the maths costs time at the locked clock, cut MUFU work: a single-MUFU
   sigmoid (rcp of 1+exp via FMA Newton from a cheap seed), or `ex2` on a pre-scaled argument to drop the extra mul.
5. **Load hints:** `.nc` vs plain loads made no difference [probe]. evict_first on loads is untested here (#38: hurt L
   by 2%).

## 7. Results so far

r0-geglu-ldg256-2d-stel (CUDA C++, 32 regs, 0 smem, one launch): 16/16 pass; times in §3. Faster than the loop's
plain copy at every size (0.69–0.95× copy). Sustained 7.0–7.2 TB/s at L, 5.8–6.8 TB/s at M, 2.2–4.6 TB/s at S.
Fit on M/L (`t = fixed + bytes/BW`): about 2.5 µs + 7.25 TB/s. Awaiting the first portal submission for Tb and Tsol.

## 8. Open questions

- Tb and Tsol per workload (portal). Whether Tsol is bytes/8 TB/s or lower, as on #38.
- Does the GELU maths show at 1500 MHz (H5)? Needs a portal pair (copy-only is wrong, so compare a 1-MUFU variant).
- Can the S-band fixed cost drop below 2.5 µs? The empty-kernel span alone is 1.73 µs on the rented B200.
- Does the harness's dirty-flush write-back cost land the same way here (read-only 5.3 TB/s at L)? Any legitimate
  kernel-side mitigation remains unknown (#38 H7/H8).
- Does the portal's CUPTI window differ for a 2-D grid of 5×8192 CTAs versus a 1-D grid (CTA dispatch rate at L)?
```

```yaml ledger
# Hypothesis ledger for #85 geglu_activation. Seeded 10 October 2026 by the research phase.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.

constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
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
  statement: evict_last on output stores cuts in-window write-backs of the flush's dirty L2 lines, so M and L sizes gain 6-13%.
  status: supported
  evidence:
  - 'probe r0 (rented B200): EL vs plain stores: 63 MB 10.98 vs 11.97 us, 126 MB 18.74 vs 21.64, 252 MB 35.35 vs 40.00, 503 MB 71.55 vs 76.07; S unchanged'
  - 'same mechanism as #38 H1; portal confirmation pending'
- id: H2
  statement: One 8-float chunk per thread in small one-shot CTAs is faster than fatter threads; the one-shot grid needs no persistence.
  status: supported
  evidence:
  - 'probe r0: 2 chunks/thread +1..5%, 4 chunks/thread +3..10% (82 regs); block 128/256/512 within 1%'
- id: H3
  statement: The sigmoid form (a*b/(1+exp(-2u)), __expf + __fdividef) is both within tolerance and faster than tanhf.
  status: supported
  evidence:
  - 'probe r0: max abs err 1.9e-6, zero elements out of tolerance at 128/2048 rows; tanhf variant 3-10% slower'
- id: H4
  statement: The 2-D grid (5 CTAs x rows, no integer division, no bounds checks) beats the 1-D grid at S by about 2% and ties at L.
  status: supported
  evidence:
  - 'probe r0: 7.9 MB 3.73 vs 3.81 us, 31 MB 6.79 vs 6.94, 126 MB 18.66 vs 18.69, 503 MB 71.56 vs 71.60'
- id: H5
  statement: At the portal's locked 1500 MHz the GELU maths (2 MUFU per element, ~37% of MUFU capacity) becomes visible; a 1-MUFU formulation would gain 1-3% at M/L.
  status: open
  evidence:
  - 'probe r0 (1965 MHz): copy-only kernel (no GELU) times equal to the full kernel at every size, so compute is hidden at the rented clock'
  - 'needs a portal pair: r0 vs a 1-MUFU variant'
- id: H6
  statement: The S-band fixed cost (~2.7 us above the byte floor at 7.9 MB) can be reduced below the current level by a dedicated small-size path.
  status: open
  evidence:
  - 'probe r0: empty kernel 1.73 us; read-only 3.52 us; full 3.6-3.8 us at 7.9 MB'
  - '#38 experience: launch params moved the empty span < 0.03 us; most of the remainder is DRAM latency + dirty flush cost'
- id: H7
  statement: The harness's dirty L2 (zero-fill flush) costs reads: a read-only kernel reaches only 5.3 TB/s at 503 MB while read+write reaches 7.0 TB/s.
  status: supported
  evidence:
  - 'probe r0: read-only 62.95 us vs full 71.6 us at 503 MB (335 MB read)'
  - 'no legitimate kernel-side mitigation known (#38 H7/H8)'
- id: H8
  statement: Load hints (.nc, evict_first, L1::no_allocate) do not matter for this once-read stream.
  status: open
  evidence:
  - 'probe r0: ld.global.nc vs plain ld.global: identical at every size'
  - 'evict_first on loads untested here; #38 H2: hurt L by ~2%'
- id: H9
  statement: Tsol for this problem sits below the bytes/8 TB/s floor at M/L, as on #38, so score 1.0 is unreachable and relative speed-ups are worth the same in every band.
  status: open
  evidence:
  - 'awaiting first portal result'
```

### Rationale

Pure elementwise fp32 stream, 12 B per output element. The first kernel is a one-shot, one-launch CUDA C++ design: a 2-D grid of 5 CTAs per row, 128 threads, each thread two 256-bit loads and one 256-bit store under L2 evict_last. The GELU uses the exact identity gelu_tanh(a) = a·sigmoid(2u), one exp and one reciprocal. Measured 16/16 pass and 0.69–0.95× the plain-copy time on the rented B200.

```json solution-spec
{"name": "r0-geglu-ldg256-2d-stel", "definition": "085_geglu_activation", "author": "solx-loop",
 "description": "GEGLU fused one-shot: 2-D grid (5 CTAs x rows), 128 threads, 8 floats per thread via 256-bit loads/stores, evict_last stores, sigmoid form of tanh-GELU",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// GEGLU: out[r, j] = gelu_tanh(x[r, j]) * x[r, INNER + j], fp32, rows of 2*INNER.
// One fused one-shot launch. Each thread handles 8 consecutive outputs: two 256-bit loads
// (gate half, linear half) and one 256-bit store with an L2 evict_last policy.
// gelu_tanh(a) = 0.5*a*(1+tanh(u)) = a * sigmoid(2u), u = sqrt(2/pi)*(a + 0.044715 a^3).
#include <cuda_runtime.h>
#include <stdint.h>

#define INNER 5120
#define THREADS 128
#define CTAS_PER_ROW (INNER / (THREADS * 8))  // 5

__device__ __forceinline__ void ld8(const float* p, float* v) {
  asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "l"(p));
}

__device__ __forceinline__ void st8_el(float* q, const float* v, unsigned long long pol) {
  asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;" ::"l"(q), "f"(v[0]), "f"(v[1]),
               "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol)
               : "memory");
}

__device__ __forceinline__ float geglu(float a, float b) {
  // 2*sqrt(2/pi) = 1.5957691216057308
  float u2 = 1.5957691216057308f * (a + 0.044715f * a * a * a);
  float e = __expf(-u2);                     // ex2.approx based, ~2 ulp
  return a * b * __fdividef(1.f, 1.f + e);   // a * b * sigmoid(2u)
}

// Fast path: inner == 5120. grid = (5, rows), 128 threads, 8 floats per thread.
__global__ void __launch_bounds__(THREADS) geglu_5120_kernel(const float* __restrict__ x, float* __restrict__ y) {
  const long long row = blockIdx.y;
  const int col = (blockIdx.x * THREADS + threadIdx.x) * 8;
  const float* g = x + row * (2 * INNER) + col;
  float gv[8], lv[8], ov[8];
  ld8(g, gv);
  ld8(g + INNER, lv);
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#pragma unroll
  for (int i = 0; i < 8; ++i) ov[i] = geglu(gv[i], lv[i]);
  st8_el(y + row * INNER + col, ov, pol);
}

// Generic fallback (any inner, any alignment, rows > 65535): one output element per thread.
__global__ void geglu_generic_kernel(const float* __restrict__ x, float* __restrict__ y, long long rows, int inner) {
  long long n = rows * inner;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    long long r = i / inner;
    int c = (int)(i - r * inner);
    float a = x[r * (2LL * inner) + c];
    float b = x[r * (2LL * inner) + inner + c];
    float u = 0.7978845608028654f * (a + 0.044715f * a * a * a);
    y[i] = 0.5f * a * (1.f + tanhf(u)) * b;
  }
}

extern "C" void geglu_launch(const float* x, float* y, long long rows, int inner, cudaStream_t stream) {
  bool fast = (inner == INNER) && rows <= 65535 && ((((uintptr_t)x) & 31) == 0) && ((((uintptr_t)y) & 31) == 0);
  if (fast) {
    dim3 grid(CTAS_PER_ROW, (unsigned)rows);
    geglu_5120_kernel<<<grid, THREADS, 0, stream>>>(x, y);
  } else {
    long long n = rows * inner;
    long long blocks = (n + 255) / 256;
    if (blocks > 1048576) blocks = 1048576;
    geglu_generic_kernel<<<(unsigned)blocks, 256, 0, stream>>>(x, y, rows, inner);
  }
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" void geglu_launch(const float* x, float* y, long long rows, int inner, cudaStream_t stream);

void run(torch::Tensor x, torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && output.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && output.scalar_type() == torch::kFloat32, "fp32 only");
  TORCH_CHECK(x.is_contiguous() && output.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dim");
  const int64_t in_dim = x.size(-1);
  TORCH_CHECK(in_dim % 2 == 0, "last dim must be even");
  const int64_t inner = in_dim / 2;
  TORCH_CHECK(output.size(-1) == inner, "output last dim mismatch");
  const int64_t rows = x.numel() / in_dim;
  TORCH_CHECK(output.numel() == rows * inner, "output numel mismatch");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  geglu_launch(x.data_ptr<float>(), output.data_ptr<float>(), (long long)rows, (int)inner, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "GEGLU fused (DPS)"); }
```

```yaml design-card
id: r0-geglu-ldg256-2d-stel
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows1
  red: warp
  cache: default
  spec: all
hypothesis: >-
  A pure one-pass elementwise stream: one launch, a 2-D grid of 5 CTAs per row with 128 threads, each thread two
  256-bit loads (gate, linear) and one 256-bit store under L2 evict_last, GELU computed as a*sigmoid(2u) with one
  exp and one reciprocal. Evict_last stores cut the in-window write-back of the harness's dirty L2 lines, which is
  what separates this from a plain copy at M/L.
expected_effect:
  S: {pct: 0, confidence: medium}
  M: {pct: 0, confidence: medium}
  L: {pct: 0, confidence: medium}
resources_sm100a:
  regs_per_thread: 32
  smem_per_cta_bytes: 0
  threads_per_cta: 128
  ctas_per_sm: 16
  bytes_in_flight_per_sm: 131072
  launches_per_call: 1
knobs: {THREADS: 128, CTAS_PER_ROW: 5, FLOATS_PER_THREAD: 8, STORE_HINT: evict_last}
tests: [H1, H2, H3, H4]
findings:
  - "evict_last stores vs plain: -8% at 63 MB, -13% at 126 MB, -12% at 252 MB, -6% at 503 MB, 0 at S [probe_b200]"
  - "1 chunk (8 floats) per thread best; 2 chunks +1..5%, 4 chunks +3..10% with 82 regs [probe_b200]"
  - "CTA size 128/256/512 within 1%; 2-D grid without division -2% at S vs 1-D grid, equal at L [probe_b200]"
  - "sigmoid form (__expf + __fdividef) max abs err 1.9e-6, zero elements out of 1e-5 tolerance; tanhf form 3-10% slower [probe_b200]"
  - "copy-only kernel (no GELU) equals the full kernel at every size at 1965 MHz: compute hidden on the rented clock [probe_b200]"
  - "ld.global.nc vs plain loads: identical [probe_b200]"
  - "empty kernel 1.73 us; read-only 3.52 us; full 3.6-3.8 us at 7.9 MB; read-only reaches only 5.3 TB/s at 503 MB vs 7.0 read+write [probe_b200]"
  - "run_tests: 16/16 pass, 0.69-0.95x plain copy time; 7.0-7.2 TB/s at L, 5.8-6.8 at M, 2.2-4.6 at S [run_tests]"
paths:
  - {max_tokens: null, lang: cuda, width: 256, threads: 128, rows: 1, grid: oneshot, mem: ldg, x: none, st: el, launches: 1}
runs_on:
  H200: {runs: false, representative: false, note: "256-bit ld/st and L2::cache_hint on st need sm_100; would need a 128-bit fallback"}
  A100: {runs: false, representative: false, note: "sm_100-only PTX"}
  L40S: {runs: false, representative: false, note: "sm_100-only PTX"}
risks:
  - "At the portal's 1500 MHz the 2 MUFU per element (37% of MUFU capacity) may become visible at M/L (H5)"
  - "gridDim.y limit 65535 rows: larger inputs fall back to the scalar generic kernel (never hit by the 16 workloads)"
  - "Tb and Tsol unknown until the first portal result"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```