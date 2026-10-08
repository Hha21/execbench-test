### Rationale
H13 is **refuted** for fp32 direct loads on B200. I ran interleaved harness-timed probes (5–7 repeats) of FlashInfer's 4-lane layout (32 rows per 128-thread CTA) and of 8-lane variants (128 and 256 threads) against c1's 16-lane layout with the same cache hints. All of them land within about ±3% of c1 at every size, and that is the same spread as the noise: two equal-size shapes (2×128 and 1×256) disagreed by 4%. At 586 tokens they are 1–4% slower. Plain copies with the same layouts are no faster than the 16-lane copy, and they are slower at 586, 1024 and 8192 tokens. Every variant already runs at copy speed, so the lane layout has nothing to win.

The candidate is the best version of the idea: the 4-lane layout with evict_first x loads for 601–1536 tokens, and c1 elsewhere. It passes 16/16. On the rented B200 it is S +0.7%, M −1.4% and L 0% against r10-rtok-m-disp, with a predicted portal score of 0.6106 ± 0.006. That is a tie, and it misses the 0.615 success bar.

```json solution-spec
{"name": "r11-fi4lane-m-disp", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "FlashInfer sm_100 4-lanes-per-row layout (H13) at 601-1536 tokens, c1-control-r6 elsewhere; refuted as a lever, ties the best", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r11-fi4lane-m-disp: per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// Tests H13 (FlashInfer sm_100 QK-norm layout). LANES lanes share one 128-float row; lane t of a row owns
// columns [8t, 8t+8) + 8*LANES*j, so each ld.global.v8.f32 warp instruction covers 32/LANES rows with
// 8*LANES*4 contiguous bytes each. LANES = 16 is the parent c1/r6 layout (half-warp per row, 1 load per
// thread); LANES = 4 is FlashInfer's sm_100 layout (4 loads of x per thread, 32 rows per 128-thread CTA).
//   B*S <= 600   : LANES 16, 256 threads, x loads L2 evict_first, weights nc   (c1-control-r6 S path)
//   601..1536    : LANES 4, 128 threads, x loads L2 evict_first, weights nc    (new)
//   above        : LANES 16, 256 threads, default x loads, weights nc          (c1 M/L path)
// Output stores always carry an L2 evict_last policy (r5/r6).
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
typedef unsigned long long u64;

__device__ __forceinline__ u64 pol_evict_last() {
    u64 p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float* v) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_ef(const float* p, float* v) {
    asm volatile("ld.global.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], 0x12F0000000000000;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void st8_h(float* p, const float* v, u64 pol) {
    asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol) : "memory");
}
#else
// sm_80/sm_90 local-test path: two 128-bit accesses with the same cache policies.
__device__ __forceinline__ void ld4(const float* p, float* v) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_ef(const float* p, float* v) {
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], 0x12F0000000000000;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void st4_h(float* p, const float* v, u64 pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "l"(pol) : "memory");
}
__device__ __forceinline__ void ld8(const float* p, float* v) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void ld8_ef(const float* p, float* v) { ld4_ef(p, v); ld4_ef(p + 4, v + 4); }
__device__ __forceinline__ void ld8_nc(const float* p, float* v) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void st8_h(float* p, const float* v, u64 pol) { st4_h(p, v, pol); st4_h(p + 4, v + 4, pol); }
#endif

template <int LANES, int THREADS, bool EF, int MINB>
__global__ void __launch_bounds__(THREADS, MINB)
qk_rms_kernel(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_rows, int n_tiles, float eps) {
    constexpr int NL = 16 / LANES;                  // 8-float chunks per thread
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int t = threadIdx.x % LANES;
    int row = tile * (THREADS / LANES) + threadIdx.x / LANES;
    const bool valid = row < n_rows;                // only a 32-row tile over an odd token count masks
    if (!valid) row = n_rows - 1;
    const int head = row % H;
    const float* xr = x + (size_t)row * D + 8 * t;
    const float* wr = w + (size_t)head * D + 8 * t;

    float xv[NL][8], wv[NL][8];
#pragma unroll
    for (int j = 0; j < NL; ++j) {                  // every load issued before any use
        if (EF) ld8_ef(xr + 8 * LANES * j, xv[j]);
        else    ld8(xr + 8 * LANES * j, xv[j]);
    }
#pragma unroll
    for (int j = 0; j < NL; ++j) ld8_nc(wr + 8 * LANES * j, wv[j]);
    const u64 pol_out = pol_evict_last();

    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < NL; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) s = fmaf(xv[j][i], xv[j][i], s);
#pragma unroll
    for (int o = LANES / 2; o >= 1; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float inv = rsqrtf(s * (1.0f / (float)D) + eps);   // * (1/128) is exact
#pragma unroll
    for (int j = 0; j < NL; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[j][i] = (xv[j][i] * inv) * wv[j][i];
    if (valid) {
        float* yr = y + (size_t)row * D + 8 * t;
#pragma unroll
        for (int j = 0; j < NL; ++j) st8_h(yr + 8 * LANES * j, xv[j], pol_out);
    }
}

template <int LANES, int THREADS, bool EF, int MINB>
cudaError_t launch(const float* q, const float* k, const float* wq, const float* wk, float* qo, float* ko,
                   int64_t n_rows, float eps, cudaStream_t stream) {
    constexpr int ROWS = THREADS / LANES;
    const int64_t n_tiles = (n_rows + ROWS - 1) / ROWS;
    if (2 * n_tiles > (int64_t)0x7fffffff) return cudaErrorInvalidValue;
    qk_rms_kernel<LANES, THREADS, EF, MINB><<<(unsigned)(2 * n_tiles), THREADS, 0, stream>>>(
        q, k, wq, wk, qo, ko, (int)n_rows, (int)n_tiles, eps);
    return cudaGetLastError();
}
}  // namespace

extern "C" cudaError_t qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                                     float* qo, float* ko, int64_t n_rows, float eps, cudaStream_t stream) {
    const int64_t tokens = n_rows / H;              // shape-only dispatch
    if (tokens <= 600)  return launch<16, 256, true, 8>(q, k, wq, wk, qo, ko, n_rows, eps, stream);
    if (tokens <= 1536) return launch<4, 128, true, 1>(q, k, wq, wk, qo, ko, n_rows, eps, stream);
    return launch<16, 256, false, 8>(q, k, wq, wk, qo, ko, n_rows, eps, stream);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" cudaError_t qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                                     float* qo, float* ko, int64_t n_rows, float eps, cudaStream_t stream);

static void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
                " must be a contiguous fp32 CUDA tensor");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 31) == 0, name, " must be 32-byte aligned");
}

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm) {
    check_tensor(query, "query"); check_tensor(key, "key");
    check_tensor(weight_q, "weight_q"); check_tensor(weight_k, "weight_k");
    check_tensor(query_norm, "query_norm"); check_tensor(key_norm, "key_norm");
    TORCH_CHECK(weight_q.numel() == 48 * 128 && weight_k.numel() == 48 * 128, "weights must be [48,128]");
    TORCH_CHECK(query.numel() % (48 * 128) == 0 && key.numel() == query.numel() &&
                query_norm.numel() == query.numel() && key_norm.numel() == query.numel(), "shape mismatch");
    const int64_t n_rows = query.numel() / 128;
    if (n_rows == 0) return;
    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const cudaError_t err = qk_rms_launch(query.data_ptr<float>(), key.data_ptr<float>(),
                                          weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
                                          query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
                                          n_rows, (float)eps, stream);
    TORCH_CHECK(err == cudaSuccess, "qk_rms_launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); FlashInfer 4-lane layout at 601-1536 tokens");
}
```

```yaml design-card
id: r11-fi4lane-m-disp
parents:
  - c1-control-r6
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows32
  red: warp
  cache: stream
  spec: dispatch:size
hypothesis: >-
  H13: FlashInfer's sm_100 QK-norm layout gives each thread 4 x 32 B of one row (4 lanes per row), so each
  warp instruction covers 8 rows. It was expected to beat our half-warp-per-row layout on B200. Interleaved
  harness-timed probes refute this for fp32 direct loads. Compared with the same hints (evict_first x at <= 1024
  tokens, evict_last stores), 4-lane and 8-lane layouts are within +-3% of c1 at 128-8192 tokens (the noise level)
  and 1-4% slower at 586. Every variant already runs at plain-copy speed, and copies with the 4/8-lane layout are
  no faster than 16-lane copies. FlashInfer's bf16 gain comes from fixing too few bytes per thread (16 B in bf16),
  a problem our 32 B-per-thread fp32 layout does not have. The candidate keeps the 4-lane path only at 601-1536
  tokens, where it ties r10-rtok-m-disp's R=3 path.
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: -0.5
    confidence: low
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 75
  smem_per_cta_bytes: 0
  threads_per_cta: 128
  ctas_per_sm: 6
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  S_MAX_TOKENS: 600
  M4_MAX_TOKENS: 1536
  LANES_M: 4
  THREADS_M: 128
  ROWS_M: 32
  other_paths: "LANES 16, 256 threads, 16 rows (c1); 31 regs"
dispatch:
  - max_tokens: 600
    kernel: qk_rms_kernel<16,256,EF=true>
    meta:
      ROWS: 16
      THREADS: 256
  - max_tokens: 1536
    kernel: qk_rms_kernel<4,128,EF=true>
    meta:
      ROWS: 32
      THREADS: 128
  - max_tokens: 1000000000
    kernel: qk_rms_kernel<16,256,EF=false>
    meta:
      ROWS: 16
      THREADS: 256
tests:
  - H13
findings:
  - "H13 refuted: FlashInfer's 4-lane layout gives no gain for fp32 direct loads on B200; within +-3% of c1 with the same hints at every size and 1-4% slower at 586 tokens. Close H13 [probe_b200]"
  - "probe (default x loads, 5 reps, vs c1): 4-lane/128 thr 128 tok +2.3%, 256 -1.2%, 1024 -1.5%, 2048 -0.3%, 8192 +0.3%; 8-lane/128 thr -0.0/-2.3/-1.2/+0.0/+0.2%; 4-lane/192 thr (48-row token tile) +0.7/-1.7/-1.0/-0.1/+0.3% [probe_b200]"
  - "probe (evict_first x matched, 7 reps, vs r6ef at S / r6 at M): 2x128 tok 4-lane +2.4%, 8-lane +1.2%; 1x256 tok 4-lane -1.1%, 8-lane -2.6% (the same size disagrees by about 4%, so this is noise); 586 tok 4-lane +4.0%, 8-lane/128 +1.4%, 8-lane/256 +2.5%; 1024 tok 4-lane -4.3% vs r6, -1.8% vs r6ef; 2048 tok all within +-1.4% [probe_b200]"
  - "copy lag: plain copies with evict_last stores run 2-5% under the kernels at S and within 0-2% at M/L. 4/8-lane copies are no faster than the 16-lane copy (586 tok +7..15% slower, 1024 +1..3%, 8192 +1.2..1.7%). The lane layout does not change DRAM efficiency [probe_b200]"
  - "evict_first x loads at 586 tokens: -8% (r6 9.44 -> r6ef 8.65 us) again, and evict_first helps 1024 by -2.5% in this run; it also helps the 4/8-lane layouts at 586 (8-lane 9.20 -> 8.77) [probe_b200]"
  - "regs: 4-lane path 72-75 regs (32 x + 32 w floats), no spills, 6 CTAs/SM at 128 threads; 8-lane 44 regs; copies cp4 40 / cp8 26 / cp16 16 [probe_b200, compile_b200]"
  - "run_tests vs r10-rtok-m-disp: 16/16 pass; S +0.7%, M -1.4%, L -0.0%; 1024-token rows -1.8/-2.0/+0.3%, so the 4-lane path ties R=3. The 2048-token rows run identical code and still moved -3.6%/0.0%, so band differences of about 1% are noise. Predicted portal 0.6106 +- 0.006 [run_tests]"
  - "tooling: b200probe.cuda_kernel/resources take opts as a tuple (a bare string is split into characters); NVRTC has no <cstdint>; harness_time can raise 'No timing results' at random, so wrap it in a retry [probe_b200]"
paths:
  - max_tokens: 600
    lang: cuda
    width: 256
    threads: 256
    rows: 16
    grid: oneshot
    x: ef
    w: nc
    st: el
  - max_tokens: 1536
    lang: cuda
    width: 256
    threads: 128
    rows: 32
    grid: oneshot
    x: ef
    w: nc
    st: el
  - max_tokens: null
    lang: cuda
    width: 256
    threads: 256
    rows: 16
    grid: oneshot
    w: nc
    st: el
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback; 50 MB L2 changes the evict_last/dirty-flush interplay"
  A100:
    runs: true
    representative: false
  L40S:
    runs: true
    representative: false
risks:
  - "On the portal the 75-register 4-lane path might lose a little more than on the bench at 1024 tokens (more SM instructions per row at 1500 MHz). It ties r10-rtok-m-disp's 48-register R=3 path, which is also unproven on the portal."
  - "Expected portal result is a tie with the current best; not worth a slot unless a slot is spare."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```