I ran experiment E1 on H14. **H14 is refuted, and L is at the hardware ceiling for this traffic.** The "6.2 TB/s read cap" was the write-back of the harness flush's dirty lines plus ramp-up, not a limit on the read path. With a clean L2, plain loads reach about 7.4–7.5 TB/s marginal. The full kernel already runs at 86–88% of DRAM peak, and no access order helps.

Neither clock lock is possible on the rented box: `nvidia-smi -lgc 1500,1500` is refused for lack of permission, and `-lmc` is unsupported (memory already runs at 3996 MHz). ncu's `--clock-control base` also fails to lock.

**Step 1 – read-only ceilings** (CUPTI span, 167.8 MB / 335.5 MB of reads):

| Kernel | Clean TB/s | Dirty TB/s | Clean marginal |
|---|---|---|---|
| (a) 256-bit loads | 6.10 / 6.69 | 5.05 / 5.33 | 7.42 |
| (b) 128-bit loads | 6.19 / 6.79 | 5.07 / 5.40 | 7.52 |
| (d) bulk copy into shared memory | 6.04 / 6.58 | – | 7.22 |
| 256-bit + L2 prefetch 64 rows ahead | 6.47 / 6.92 | 5.00 / 5.40 | 7.43 |

- **(c) prefetch-only:** it can't be timed. The kernel finishes issuing in 16 µs (about 21 TB/s) because prefetches are fire-and-forget. Plain loads already exceed 6.8 TB/s, so the SM return path is not the limit.
- **(e) data already in L2:** about 22 TB/s.
- **ncu at 503 MB:** the full kernel reads 335.5 MB and writes 155 MB (dirty) or 99 MB (clean), at 86.6% / 88.4% of DRAM peak. The read-only kernel runs at 79–82%.

**Step 2 – access order on the full kernel** (21 interleaved reps, medians, vs the r1 large-size path at 252 / 441 / 503 MB):

| Variant | Change |
|---|---|
| Row swizzle, G = 8 / 16 / 64 | +2.8% to +6.3% (slower) |
| L2 prefetch 64 / 512 / 2048 rows ahead | +0.1% up to +46% (slower) |
| Reversed row order | −0.2% to +0.2% |
| Two rows per thread (control) | +1.3% to +1.6% |
| One 640-thread CTA per row | −0.1% to −0.2% |

I skipped variant (iv): a CTA holding the gate of row r and the linear half of row r+1 cannot produce any output row on its own.

**Step 3 – result.** No bulk-copy path won Step 1, so I didn't build one. The only new finding is about CTA dispatch: an empty kernel with 5×8192 CTAs of 128 threads takes 22.2 µs, against 7.2 µs with one 640-thread CTA per row. That cost is hidden at the rented clock. It might show at the portal's locked 1500 MHz, but I couldn't test that. I'm returning that variant as the candidate:
- 16/16 pass.
- Against r1 on the same GPU: S −0.1%, M −0.9%, L −0.1%, all within noise.
- 32 registers, no spills, 256-bit loads and stores in the SASS.
- Emulator prediction: 0.678 ± 0.011, with a 59% chance of beating the current 0.6755.

### Rationale
The success threshold (≥ 3% at 252, 441 and 503 MB) was not met. The candidate (one CTA per row, otherwise r1) keeps everything that is known to work and only reduces CTA dispatch, which might matter at 1500 MHz. Treat it as neutral, with a small upside on the portal.

```json solution-spec
{"name": "r2-geglu-row640-efm", "definition": "085_geglu_activation", "author": "solx-loop", "description": "One 640-thread CTA per row (5x fewer CTAs than the (5, rows) grid), 256-bit loads, evict_last 256-bit stores, evict_first loads up to 3072 rows. E1 found L at the DRAM ceiling; this variant ties r1 on the rented B200 and may gain on the portal if CTA dispatch is clock-bound.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// GEGLU: out[r, j] = gelu_tanh(x[r, j]) * x[r, INNER + j], fp32, rows of 2*INNER.
// One fused one-shot launch, one CTA per row: 640 threads x 8 consecutive outputs = 5120.
// Each thread: two 256-bit loads (gate half, linear half), one 256-bit store with an L2
// evict_last policy. One CTA per row cuts the CTA count 5x versus the (5, rows) grid of
// 128-thread CTAs (empty-kernel span at 8192 rows 22.2 -> 7.2 us on the rented B200), with
// the same per-thread work and no integer division.
// gelu_tanh(a) = 0.5*a*(1+tanh(u)) = a * sigmoid(2u), u = sqrt(2/pi)*(a + 0.044715 a^3).
// Load policy by size (shape-only dispatch): up to EF_MAX_ROWS rows the input loads carry
// L2::evict_first, so read misses evict our own clean input lines instead of the harness
// flush's dirty lines (fewer in-window write-backs). Above that default loads.
#include <cuda_runtime.h>
#include <stdint.h>

#define INNER 5120
#define THREADS (INNER / 8)  // 640
#define EF_MAX_ROWS 3072

#define V8 "{%0,%1,%2,%3,%4,%5,%6,%7}"
#define O8 "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])

template <bool EF>
__device__ __forceinline__ void ld8(const float* p, float* v) {
  if (EF)
    asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v8.f32 " V8 ", [%8];" : O8 : "l"(p));
  else
    asm volatile("ld.global.nc.v8.f32 " V8 ", [%8];" : O8 : "l"(p));
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

// Fast path: inner == 5120. grid = rows, 640 threads, 8 floats per thread.
template <bool EF>
__global__ void __launch_bounds__(THREADS) geglu_row_kernel(const float* __restrict__ x, float* __restrict__ y) {
  const long long row = blockIdx.x;
  const int col = threadIdx.x * 8;
  const float* g = x + row * (2 * INNER) + col;
  float gv[8], lv[8], ov[8];
  ld8<EF>(g, gv);
  ld8<EF>(g + INNER, lv);
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#pragma unroll
  for (int i = 0; i < 8; ++i) ov[i] = geglu(gv[i], lv[i]);
  st8_el(y + row * INNER + col, ov, pol);
}

// Generic fallback (any inner, any alignment): one output element per thread.
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
  bool fast = (inner == INNER) && rows <= 2147483647LL && ((((uintptr_t)x) & 31) == 0) && ((((uintptr_t)y) & 31) == 0);
  if (fast) {
    if (rows <= EF_MAX_ROWS)
      geglu_row_kernel<true><<<(unsigned)rows, THREADS, 0, stream>>>(x, y);
    else
      geglu_row_kernel<false><<<(unsigned)rows, THREADS, 0, stream>>>(x, y);
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
id: r2-geglu-row640-efm
parents: [r1-geglu-efld-m]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows1
  red: warp
  cache: stream
  spec: dispatch:size
hypothesis: >-
  E1 (H14) finds no read-path cap. Clean-L2 plain loads stream at 7.4-7.5 TB/s marginal. The full
  kernel runs at 86-88% of DRAM peak (ncu). The dirty-flush write-back (155 MB at 503 MB) sets the
  L time, and no access order changes it. Swizzles, prefetch-ahead, reversed order and two rows
  per thread all tie or lose. The only structural lever left is the CTA count. One 640-thread CTA
  per row dispatches 5x fewer CTAs: the empty-kernel span at 8192 rows drops from 22.2 to 7.2 us.
  On the rented B200 this ties r1, because dispatch is hidden at 1965 MHz. If CTA dispatch scales
  with the portal's 1500 MHz SM clock, it could gain slightly at M/L on the portal only.
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: -1
    confidence: low
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 32
  smem_per_cta_bytes: 0
  threads_per_cta: 640
  ctas_per_sm: 3
  bytes_in_flight_per_sm: 122880
  launches_per_call: 1
knobs:
  THREADS: 640
  CTAS_PER_ROW: 1
  FLOATS_PER_THREAD: 8
  EF_MAX_ROWS: 3072
  LOAD_HINT_SMALL: "ld.global.nc.L1::no_allocate.L2::evict_first.v8.f32"
  LOAD_HINT_LARGE: "ld.global.nc.v8.f32"
  STORE_HINT: evict_last
dispatch:
  - {max_tokens: 3072, kernel: "geglu_row_kernel<true>", meta: {x: ef, st: el, threads: 640}}
  - {max_tokens: null, kernel: "geglu_row_kernel<false>", meta: {x: none, st: el, threads: 640}}
tests: [H14]
findings:
  - "nvidia-smi -lgc 1500,1500 is not permitted on the rented B200 (no permission); -lmc is unsupported (memory already at 3996 MHz); ncu --clock-control base also fails to lock clocks, so H15 cannot be measured on this box [probe_b200]"
  - "bench method: torch CUDA events around a ctypes launch include CPU launch gaps (empty kernel read 16-26 us); use sol_execbench.core.bench.cupti_utils.collect_cupti_activities and per-kernel end-start instead [probe_b200]"
  - "step 1 read-only, CUPTI, 167.8/335.5 MB of reads (252/503 MB workloads): 256-bit DIRTY 33.25/62.99 us (5.05/5.33 TB/s), CLEAN 27.50/50.12 (6.10/6.69); 128-bit (512 B per warp instruction) DIRTY 33.06/62.16, CLEAN 27.12/49.42 (6.19/6.79); bulk copy into smem DIRTY 33.60/63.35, CLEAN 27.76/51.01 [probe_b200]"
  - "step 1: CLEAN marginal read bandwidth (4096 -> 8192 rows) 7.42 TB/s (256-bit), 7.52 (128-bit), 7.22 (bulk copy into smem); DIRTY marginal 5.8 TB/s. The 6.2 TB/s 'cap' in H14 was write-back plus ramp, not a read-path limit; H14 refuted [probe_b200]"
  - "step 1: 128-bit and 256-bit loads tie on reads (128-bit 1-2% faster CLEAN); the TMA/bulk path into smem is 1-3% slower than LDG; the SM return path is not the limit [probe_b200]"
  - "step 1 (c): a prefetch.L2-only kernel finishes issuing 335 MB in 16 us (21 TB/s): fire-and-forget, so it cannot measure the DRAM fill rate. One-shot pf_l2 (2 x 4 KB per CTA) is 22.8 us, the same as the empty kernel [probe_b200]"
  - "step 1 (e): data already in L2 (32 MB x 8 passes, persistent 592-2368 CTAs): 21.6-22.7 TB/s [probe_b200]"
  - "step 1: read-only with L2 prefetch of row+64: CLEAN 25.92/48.49 us (-6%/-3% vs plain 256-bit), DIRTY 33.54/62.10 (+1/-1%) [probe_b200]"
  - "ncu (rented clock, single samples) at 503 MB: full kernel DIRTY read 335.5 MB, write 155 MB, 86.1-86.7% of DRAM peak; CLEAN write 99 MB, 87.5-88.4%; read-only DIRTY write 61.7 MB, 79%; read-only CLEAN 81-82%. lts read sectors 15.7M (= input bytes / 32 B for every variant). At 63 MB: full DIRTY write 15.8-18.4 MB, EF loads 8.1 MB [probe_b200 ncu]"
  - "CTA dispatch: empty kernel span with (5, rows) x 128-thread CTAs = 1.47/2.31/3.62/6.31/11.59/22.18 us at 128/512/1024/2048/4096/8192 rows (~0.54 ns per CTA GPU-wide); with rows x 640-thread CTAs 1.47/1.54/1.76/2.53/4.07/7.23 us [probe_b200]"
  - "step 2 (full kernel, EL stores, default loads, 21 interleaved reps, harness sequence) vs base at 252/441/503 MB: reversed rows -0.2/-0.1/+0.2%; swizzle G=8 +4.7/+2.9/+2.8%; G=16 +6.3/+4.1/+3.3%; G=64 +5.5/-/+4.3%; L2 prefetch row+64 +0.1/+1.0/+3.9%; +512 +8.3/+10.8/+12.0%; +2048 +35/+45/+46%; 2 rows per thread (all loads first) +1.6/+1.3/+1.3%; one 640-thread CTA per row -0.1/-0.2/-0.2% [probe_b200]"
  - "dead end: spreading concurrent CTAs over 1-4 MB apart regions (row swizzle) costs 3-6% at L; DRAM likes the natural sequential order [probe_b200]"
  - "dead end: L2 prefetch ahead of the read front hurts the full kernel (prefetched lines evicted, or extra dirty evictions before use); worse the further ahead [probe_b200]"
  - "variant (iv), a CTA holding gate of row r and linear of row r+1, cannot compute any output row alone; not built [INFERRED]"
  - "640-thread one-CTA-per-row vs (5, rows) 128-thread at 128/512/1024/2048/4096/8192 rows: default loads 0.0/0.0/0.0/-0.3/-0.4/-0.1%; with EF loads vs base+EF: 0.0/-1.4/-1.4/-0.4%. 2 x 320-thread CTAs per row: -2.5% at 128 rows, +0.9% at 512, about 0 at L (single bench, noise-level) [probe_b200]"
  - "run_tests 16/16 pass vs r1-geglu-efld-m on the same B200: S -0.1%, M -0.9% (1,1024 -1.2%, 2,512 -0.8%, 8,256 -1.4%, 1,2048 -0.7%, 4,541 -0.4%), L -0.1% (16,256 +2.0%, others -0.4..-0.8%); emulator 0.678 +- 0.011 [run_tests]"
  - "conclusion for H14: L is at the DRAM ceiling for this traffic (86-88% of peak with 490 MB in-window at 503 MB); the only L lever left would be fewer in-window write-back bytes, which H17 says no policy changes [probe_b200; INFERRED]"
paths:
  - {max_tokens: 3072, lang: cuda, width: 256, threads: 640, rows: 1, grid: oneshot, mem: ldg, x: ef, st: el, launches: 1}
  - {max_tokens: null, lang: cuda, width: 256, threads: 640, rows: 1, grid: oneshot, mem: ldg, x: none, st: el, launches: 1}
runs_on:
  H200:
    runs: false
    representative: false
    note: "256-bit ld/st and .L2::evict_first on 256-bit loads need sm_100"
  A100:
    runs: false
    representative: false
    note: "sm_100-only PTX"
  L40S:
    runs: false
    representative: false
    note: "sm_100-only PTX"
risks:
  - "Ties r1 on the rented B200; any portal gain depends on CTA dispatch being SM-clock-bound at 1500 MHz, which could not be measured (clock locks not permitted)"
  - "At 128/131 rows the grid has fewer CTAs (128/131) than SMs (148); measured tie on the rented B200, but a few SMs stay idle"
  - "3 CTAs/SM x 640 threads = 1920 resident threads (94% occupancy) vs 2048 for 128-thread CTAs; slightly fewer bytes in flight per SM (123 vs 131 KB)"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```