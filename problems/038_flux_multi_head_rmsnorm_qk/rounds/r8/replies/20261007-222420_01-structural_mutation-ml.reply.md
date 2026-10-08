The candidate is r8-ldg256-l2mode-disp. On the rented B200 it is about 1% faster than r6-s-ldef-nc-disp on medium and large inputs, against the 5% the task asked for. It passes all 16 workloads, the harness projects a portal score of 0.613 (best so far 0.610), and that gain is close to noise. My probes suggest the parent is already as fast as a plain copy for these sizes, so I don't think any single change can cut 5% here.

**What the probes found (rented B200, harness timing):**
- **The parent matches a plain copy.** A 256-bit copy with evict_last stores and no maths times the same as the parent at 1024–8192 tokens (14.2–115.8 µs). Medium and large inputs are limited by DRAM bandwidth at about 7.05 TB/s for reads plus writes, not by the kernel. The evict_last effect does reproduce: plain stores are 4–21% slower.
- **The harness's L2 flush leaves dirty lines.** Its zero-fill fills the L2 with about 120 MB of dirty lines, and our kernel pays to write them back. A 400 MB read-only kernel takes 80.9 µs after a zero flush and 64.6 µs after a clean one. Evict_first loads recover part of that (70.7 µs) in a read-only kernel.
- **Why the L2 end state can't fix it.** Whatever dirty output stays in L2 at the end displaced a dirty flush line earlier, so the write traffic inside the timed window is at least the output size. Changing what sits in L2 at the end cannot get below reads plus writes, and the tests agree. Several L2-policy variants all landed within ±1% of the parent:
  - evict_last only on the last 64/96/112 MB of output;
  - evict_first or normal priority on the earlier output;
  - evict_last weights;
  - interleaving Q and K tiles;
  - 128- or 1024-thread CTAs.
- **The two small gains that repeated,** measured three times each:
  - evict_first on x loads at 1024 tokens: −1.3%;
  - 512-thread CTAs with 32 rows each at 2048 tokens and above: −0.3% to −0.9%.

The candidate dispatches between those two by token count. Small inputs run r6's path unchanged. I did not run `compile_b200` on the final files, so registers and spills are not checked. The design-card figures are from the parent (31 registers), and I expect the 512-thread variant to stay under 32.

### Rationale
r8-ldg256-l2mode-disp keeps r6's half-warp-per-row, 256-bit, evict_last-store structure and changes only, by token count, the load cache policy and the CTA size. Given the copy ceiling above, it takes the only two changes that repeated in probes. Full test: medium −1.2%, large −1.2%, small −0.1%.

```json solution-spec
{"languages": ["cuda_cpp"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}
```

```cuda file=kernel.cu
// r8-ldg256-l2mode-disp: per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// Structure of r5/r6: half-warp per 128-float row, 256-bit ld/st on sm_100a, output stores L2 evict_last.
// Dispatch (shape only):
//   B*S <= 600          : 16 rows / 256-thread CTA, x loads L2 evict_first, w ld.global.nc (r6 S path)
//   600 < B*S <= 1536   : 16 rows / 256-thread CTA, x loads L2 evict_first, w default
//   B*S > 1536          : 32 rows / 512-thread CTA, default loads (fewer, fatter CTAs)
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
constexpr uint64_t POL_EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t POL_EVICT_LAST  = 0x14F0000000000000ULL;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) {
    asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol) : "memory");
}
#else
__device__ __forceinline__ void ld4(const float* p, float* v) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_h(const float* p, float* v, uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st4_h(float* p, const float* v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "l"(pol) : "memory");
}
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    ld4_h(p, v, pol); ld4_h(p + 4, v + 4, pol);
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    st4_h(p, v, pol); st4_h(p + 4, v + 4, pol);
}
#endif

// XMODE: 0 default x loads, 1 evict_first x loads. WNC: weights via ld.global.nc.
template <int THREADS, int XMODE, bool WNC>
__global__ void __launch_bounds__(THREADS, 2048 / THREADS)
qk_rms_kernel(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_tiles, float eps) {
    constexpr int ROWS = THREADS / 16;
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int r = threadIdx.x >> 4;
    const int c = (threadIdx.x & 15) * 8;
    const size_t row = (size_t)tile * ROWS + r;
    const int head = (int)(row % H);

    float xv[8], wv[8], yv[8];
    if (XMODE == 1) ld8_h(x + row * D + c, xv, POL_EVICT_FIRST);
    else            ld8(x + row * D + c, xv);
    if (WNC) ld8_nc(w + (size_t)head * D + c, wv);
    else     ld8(w + (size_t)head * D + c, wv);

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = fmaf(xv[i], xv[i], s);
    s += __shfl_xor_sync(0xffffffffu, s, 8);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    const float inv = rsqrtf(s * (1.0f / (float)D) + eps);   // * (1/128) is exact
#pragma unroll
    for (int i = 0; i < 8; ++i) yv[i] = (xv[i] * inv) * wv[i];
    st8_h(y + row * D + c, yv, POL_EVICT_LAST);
}
}  // namespace

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_rows, float eps, cudaStream_t stream) {
    if (n_rows <= 0) return 0;
    const long long tokens = n_rows / H;
    if (tokens <= 600) {
        const long long nt = n_rows / 16;
        qk_rms_kernel<256, 1, true><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    } else if (tokens <= 1536 || (n_rows % 32) != 0) {
        const long long nt = n_rows / 16;
        if (tokens <= 1536)
            qk_rms_kernel<256, 1, false><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
        else
            qk_rms_kernel<256, 0, false><<<(unsigned)(2 * nt), 256, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    } else {
        const long long nt = n_rows / 32;
        qk_rms_kernel<512, 0, false><<<(unsigned)(2 * nt), 512, 0, stream>>>(q, k, wq, wk, qo, ko, (int)nt, eps);
    }
    return (int)cudaGetLastError();
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_rows, float eps, cudaStream_t stream);

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
    TORCH_CHECK(n_rows / 16 * 2 <= (int64_t)0x7fffffff, "grid too large");
    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int err = qk_rms_launch(query.data_ptr<float>(), key.data_ptr<float>(), weight_q.data_ptr<float>(),
                            weight_k.data_ptr<float>(), query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
                            (long long)n_rows, (float)eps, stream);
    TORCH_CHECK(err == 0, "qk_rms launch failed: ", cudaGetErrorString((cudaError_t)err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32), size-dispatched L2 policy / CTA size");
}
```

```yaml design-card
id: r8-ldg256-l2mode-disp
parents:
  - r6-s-ldef-nc-disp
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows32
  red: halfwarp
  cache: stream
  spec: dispatch:size
hypothesis: >-
  Probes on the rented B200 (harness CUPTI timing, cold L2) show the parent is at the ceiling for medium and
  large inputs. A 256-bit copy with evict_last stores and no maths times the same as the parent
  (14.2/28.6/57.8/115.8 us at 1024/2048/4096/8192 tokens). The harness zero-flush leaves about 120 MB of dirty
  L2 lines; a 400 MB read-only kernel takes 80.9 us after a zero flush against 64.6 us after a clean flush.
  Any dirty output left in L2 at kernel end displaced a dirty flush line earlier, so in-window DRAM traffic
  cannot drop below reads + writes. That explains why these L2-policy variants were all within +-1%:
  evict_last only on the last 64/96/112 MB of output, evict_first or normal stores for the early output,
  evict_last weights, Q/K tile interleave, and 128- or 1024-thread CTAs. Two small effects repeated (3x each)
  and are dispatched here: evict_first x loads at 1024 tokens (-1.3%), and 512-thread CTAs with 32 rows for
  more than 1536 tokens (-0.3 to -0.9%). Small inputs keep r6's path. Full run_tests vs the parent on the
  rented B200: S -0.1%, M -1.2%, L -1.2%, 16/16 pass, projected portal score 0.6131 (parent 0.6099).
  The gain is B200-specific and close to noise.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -0.8
    confidence: low
  L:
    pct: -1.0
    confidence: low
resources_sm100a:
  regs_per_thread: 31
  smem_per_cta_bytes: 0
  threads_per_cta: 512
  ctas_per_sm: 4
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  S_MAX_TOKENS: 600
  EF_MAX_TOKENS: 1536
  L_THREADS: 512
  L_ROWS: 32
  store_policy: "L2 evict_last constant 0x14F0000000000000 (all sizes)"
dispatch:
  - max_tokens: 600
    kernel: qk_rms_kernel<256,1,true>
    meta:
      ROWS: 16
      loads: "x evict_first, w nc"
  - max_tokens: 1536
    kernel: qk_rms_kernel<256,1,false>
    meta:
      ROWS: 16
      loads: "x evict_first, w default"
  - max_tokens: 1000000000
    kernel: qk_rms_kernel<512,0,false>
    meta:
      ROWS: 32
      loads: "default"
      note: "falls back to the 256-thread kernel if n_rows % 32 != 0"
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback, 50 MB L2"
  A100:
    runs: true
    representative: false
  L40S:
    runs: true
    representative: false
risks:
  - "Gains of about 1% are at the edge of portal noise (about 5%); the portal may show no change."
  - "Medium and large sizes appear bound by reads+writes at about 7.05 TB/s; further structural changes to this kernel are unlikely to help."
  - "compile_b200 was not run on the final files; regs_per_thread is the parent's value, not a measured one."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```