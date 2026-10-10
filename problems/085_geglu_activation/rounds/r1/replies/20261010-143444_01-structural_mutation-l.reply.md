### Rationale
H12 is refuted on the rented B200. Giving the read side more bytes in flight does not speed up L. Persistent 1-D bulk rings were 7–10% slower than r0 on the full kernel and 3–5% slower read-only, at every depth tried: 2 to 12 stages, 64 to 192 KB in flight per SM, strided or contiguous tile order. A persistent kernel that prefetches the next tile into registers was 4–8% slower. Fat threads (2 or 4 chunks per thread, all loads issued before any use, `__launch_bounds__(128,8)`) tied r0 within 1%, on the full kernel and read-only alike. The read-only cap of about 5.3 TB/s at 503 MB stays the same whatever the in-flight depth, so the L headroom rests on H11, not H12.

As the task requires, the candidate below is the best ring variant (S3, k=8, strided), with r0 kept for rows ≤ 2164. run_tests: 16/16 pass; S +0.1%, M −0.5%, L +8.1% (slower than r0). It is a negative result for the notebook and should not get a portal slot.

```json solution-spec
{"name": "r1-geglu-bulkring-s3k8-l", "definition": "085_geglu_activation", "author": "solx-loop", "description": "Persistent 1-D bulk ring (3 stages x 8 KB, 148x8 CTAs) for rows > 2164, r0 one-shot below; tests H12 (read-side bytes in flight). Refuted: L +8% vs r0.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// GEGLU: out[r, j] = gelu_tanh(x[r, j]) * x[r, INNER + j], fp32, rows of 2*INNER.
// One launch per call, dispatched on rows only:
//  rows <= RING_MIN_ROWS-1: r0's 2-D one-shot kernel (two 256-bit loads, one 256-bit evict_last store per thread).
//  larger: persistent 1-D bulk ring (E2/H12): 148*8 CTAs of 128 threads, each looping over (row, 1024-float segment)
//  tiles; thread 0 refills a 3-stage shared ring (4 KB gate + 4 KB linear per stage) with cp.async.bulk + mbarrier,
//  consumers read 8+8 floats from shared memory and store 8 floats with the evict_last 256-bit store.
// gelu_tanh(a) = 0.5*a*(1+tanh(u)) = a * sigmoid(2u), u = sqrt(2/pi)*(a + 0.044715 a^3).
#include <cuda_runtime.h>
#include <stdint.h>

#define INNER 5120
#define THREADS 128
#define CTAS_PER_ROW (INNER / (THREADS * 8))  // 5
#define SEG 1024                              // floats per tile segment (THREADS * 8)
#define SEGS_PER_ROW (INNER / SEG)            // 5
#define STAGES 3
#define CTAS_PER_SM 8
#define RING_MIN_ROWS 2165
#define RING_SMEM (STAGES * 2 * SEG * 4 + 128)

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

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

// Bulk ring: queue tile t (gate + linear segment, 8 KB) into one stage.
__device__ __forceinline__ void ring_issue(const float* x, float* dst, uint64_t* bar, int t) {
  const int row = t / SEGS_PER_ROW, seg = t - row * SEGS_PER_ROW;
  const float* src = x + (long long)row * (2 * INNER) + seg * SEG;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(bar)), "r"(2 * SEG * 4) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)),
               "l"(src), "r"(SEG * 4), "r"(sa(bar))
               : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   sa(dst + SEG)),
               "l"(src + INNER), "r"(SEG * 4), "r"(sa(bar))
               : "memory");
}

__device__ __forceinline__ void ring_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t@P1 bra "
      "DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}" ::"r"(bar),
      "r"(phase)
      : "memory");
}

// Persistent ring: CTA b handles tiles b, b+G, b+2G, ... (G = gridDim.x <= ntiles).
__global__ void __launch_bounds__(THREADS) geglu_ring_kernel(const float* __restrict__ x, float* __restrict__ y,
                                                             int ntiles) {
  extern __shared__ __align__(128) unsigned char smem[];
  float* buf = (float*)smem;
  uint64_t* bar = (uint64_t*)(smem + STAGES * 2 * SEG * 4);
  const int tid = threadIdx.x, G = gridDim.x, t0 = blockIdx.x;
  const int n = (ntiles - t0 + G - 1) / G;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(&bar[s])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int s = 0; s < STAGES && s < n; ++s) ring_issue(x, buf + s * 2 * SEG, &bar[s], t0 + s * G);
  }
  __syncthreads();
  for (int i = 0; i < n; ++i) {
    const int s = i % STAGES;
    ring_wait(sa(&bar[s]), (i / STAGES) & 1);
    const float* gp = buf + s * 2 * SEG + tid * 8;
    float g[8], l[8], o[8];
    *(float4*)&g[0] = ((const float4*)gp)[0];
    *(float4*)&g[4] = ((const float4*)gp)[1];
    *(float4*)&l[0] = ((const float4*)(gp + SEG))[0];
    *(float4*)&l[4] = ((const float4*)(gp + SEG))[1];
    __syncthreads();  // every consumer has read stage s: refill it
    if (tid == 0 && i + STAGES < n) ring_issue(x, buf + s * 2 * SEG, &bar[s], t0 + (i + STAGES) * G);
    const int t = t0 + i * G;
    const int row = t / SEGS_PER_ROW, seg = t - row * SEGS_PER_ROW;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = geglu(g[k], l[k]);
    st8_el(y + (long long)row * INNER + seg * SEG + tid * 8, o, pol);
  }
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

static int num_sms() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

extern "C" void geglu_launch(const float* x, float* y, long long rows, int inner, cudaStream_t stream) {
  bool fast = (inner == INNER) && rows <= 65535 && ((((uintptr_t)x) & 31) == 0) && ((((uintptr_t)y) & 31) == 0);
  if (fast && rows >= RING_MIN_ROWS) {
    const int ntiles = (int)rows * SEGS_PER_ROW;
    int grid = num_sms() * CTAS_PER_SM;
    if (grid > ntiles) grid = ntiles;
    geglu_ring_kernel<<<grid, THREADS, RING_SMEM, stream>>>(x, y, ntiles);
  } else if (fast) {
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
id: r1-geglu-bulkring-s3k8-l
parents: [r0-geglu-ldg256-2d-stel]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: persistent
  launch: fused
  tile: rows1
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  H12: the L-band read cap (about 5.3 TB/s read-only after the dirty flush) comes from a bytes-in-flight limit
  on the read side. A persistent 1-D bulk ring (148x8 CTAs, 3 stages of 4 KB gate + 4 KB linear, 192 KB in
  flight per SM, one instruction per 4 KB) should lift L by at least 3%. Measured: refuted. On the rented B200
  the ring is 7-10% slower than r0 at L (full kernel) and 3-5% slower read-only, at every depth tried. Fat threads
  with all loads issued first tie r0. In-flight bytes do not limit the stream.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: high
  L:
    pct: 8
    confidence: high
resources_sm100a:
  regs_per_thread: 32
  smem_per_cta_bytes: 24704
  threads_per_cta: 128
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 196608
  launches_per_call: 1
knobs:
  STAGES: 3
  CTAS_PER_SM: 8
  SEG_FLOATS: 1024
  THREADS: 128
  RING_MIN_ROWS: 2165
  STORE_HINT: evict_last
dispatch:
  - max_tokens: 2164
    kernel: geglu_5120_kernel
    meta:
      grid: "(5, rows)"
      threads: 128
  - max_tokens: null
    kernel: geglu_ring_kernel
    meta:
      grid: "148*8"
      threads: 128
      STAGES: 3
tests: [H12]
findings:
  - "bulk ring, full kernel, strided tiles, 252/503 MB (r0 35.1/70.3-71.5 us): S4k6 38.3/77.6, S6k4 37.9-38.6/75.6-77.2, S8k3 38.5/77.8, S3k8 38.2/77.1, S2k12 38.2/77.4 us: 7-10% slower at every depth from 64 to 192 KB per SM [probe_b200]"
  - "contiguous (blocked) tile assignment per CTA is worse than strided: S6k4 39.0/78.2, S3k8 38.9/78.6, S12k2 40.1/80.8 us at 252/503 MB [probe_b200]"
  - "read-only ladder at 503 MB: r0 one-shot 63.5-63.7 us (5.27 TB/s of reads), ring S6k4 65.6-66.0, ring S3k8 66.1, 2 chunks/thread 63.9, 4 chunks/thread 63.9 us; no variant lifts the read cap [probe_b200]"
  - "read-only at 252 MB: r0 33.4-33.5, ring 35.2-35.7, 2 chunks/thread 33.8, 4 chunks/thread 34.2 us [probe_b200]"
  - "fat one-shot threads with all loads issued first and __launch_bounds__(128,8) (64 regs at 4 chunks), full kernel: 2 chunks 35.1/71.3 us, 4 chunks 35.6-36.2/71.1-72.2 vs r0 35.2/71.4 at 252/503 MB: tie within 1% [probe_b200]"
  - "persistent register-prefetch kernel (148x8 CTAs, next tile's 256-bit loads issued before the current compute): 36.5/75.9 us vs r0 35.1/70.3, 4-8% slower [probe_b200]"
  - "EF policy on the bulk copies (.L2::cache_hint evict_first) with EL stores: 38.5/76.1 vs 37.9/77.1 us without; no gain [probe_b200]"
  - "at 126 MB (2048 rows) the ring is 8% slower too (20.2-20.5 vs 18.8 us); read-only ring 19.0-19.5 vs r0 read-only 18.1 [probe_b200]"
  - "the ring costs about 2 us over r0 read-only but about 6 us on the full kernel at 503 MB: persistence hurts the interplay of evict_last stores and reads more than the reads themselves [probe_b200]"
  - "run_tests: 16/16 pass; S +0.1%, M -0.5% (unchanged r0 path), L +7.6..+8.5% (ring); emulator 0.661 vs 0.674 [run_tests]"
  - "H12 refuted on the rented B200: the read cap is not an in-flight-bytes limit; L headroom rests on H11 (dirty-flush write-backs) [probe_b200]"
paths:
  - max_tokens: 2164
    lang: cuda
    width: 256
    threads: 128
    rows: 1
    grid: oneshot
    mem: ldg
    x: none
    st: el
    launches: 1
  - max_tokens: null
    lang: cuda
    width: 256
    threads: 128
    rows: 1
    grid: persistent
    mem: tma
    prefetch: true
    x: none
    st: el
    launches: 1
runs_on:
  H200:
    runs: false
    representative: false
    note: "256-bit ld/st and st.L2::cache_hint .v8 need sm_100"
  A100:
    runs: false
    representative: false
    note: "sm_100-only PTX; no bulk copies"
  L40S:
    runs: false
    representative: false
    note: "sm_100-only PTX"
risks:
  - "Measured 8% slower than r0 at L on the rented B200; not worth a portal slot except as a deliberate negative control"
  - "Persistent grid sized from cudaDevAttrMultiProcessorCount (148 on B200); 8 CTAs/SM fits by smem (8 x 25.7 KB) and regs (32)"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```