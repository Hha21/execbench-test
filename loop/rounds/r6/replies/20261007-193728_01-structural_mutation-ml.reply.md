I'm submitting **r6-ldg256-qkpair-r16**. For token counts B·S > 600, each half-warp now normalises one Q row and the matching K row together. That doubles each thread's DRAM reads in flight (64 B instead of 32 B), raising the estimate per SM from 64 KB to 96 KB, and halves the number of CTAs. Its only measured result is a tie with the parent on A100, so the gain on B200 is a prediction, not a measurement.

### Rationale
- **Cache-hint probe (rejected).** I first tested my L2 idea: marking input loads `evict_first` so used inputs leave L2 before the harness's dirty flush lines. On A100 it was clearly worse: S +0.5%, M +2.0%, L +8.5%. I dropped it.
- **The change I kept.** The parent keeps only about 64 KB of reads in flight per SM, below what B200 seems to need (about 96–128 KB under load). The new kernel issues both rows' 256-bit loads first, then does both reductions. The weights are loaded only after the reductions; they are L1/L2 hits, so they don't hold registers while the DRAM reads are outstanding.
- **Resources on sm_100a.** 39 registers, no spills, 4 × LDG.256 and 2 × STG.256 in the new kernel, so 6 CTAs fit per SM. Stores keep the parent's `evict_last` hint. Inputs of 600 tokens or fewer run the parent kernel unchanged.
- **A100 result.** All 16 workloads pass, and timing is neutral: S +0.1%, M +0.2%, L −0.1%. That is expected, because A100 already has enough reads in flight.
- **Variant I didn't submit.** Forcing 32 registers (8 CTAs/SM, 128 KB in flight) compiles without spills on sm_100a, but was 3.5% slower than the parent on A100 M. I submit the 39-register version that I actually tested.

```json solution-spec
{"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}
```

```cuda file=kernel.cu
// r6-ldg256-qkpair-r16: per-head RMSNorm of Q and K (fp32), one fused one-shot launch.
// Parent r5-ldg256-os-r16-stel (16 rows per 256-thread CTA, half-warp per 128-float row, 256-bit ld/st,
// output stores with an L2 evict_last policy, default-policy loads).
// Change: for B*S > 600 each half-warp handles row r of Q AND row r of K (same head) in one CTA, so each
// thread has 2 x 32 B of DRAM loads in flight (64 B/thread) and the grid has half as many CTAs.
// Weights are loaded after the reductions (L1/L2 hits), so they hold no registers across the DRAM latency.
// The S band (B*S <= 600) runs the parent kernel unchanged.
// All maths fp32; rsqrtf -> rsqrt.approx.f32; 1/128 is an exact multiply.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
constexpr int ROWS = 16;
constexpr int THREADS = 256;
constexpr int GROUPS = H / ROWS;
constexpr int64_t PAIR_MIN_TOKENS = 601;

__device__ __forceinline__ uint64_t pol_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
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
__device__ __forceinline__ void st4_h(float* p, const float* v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "l"(pol) : "memory");
}
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    st4_h(p, v, pol); st4_h(p + 4, v + 4, pol);
}
#endif

__device__ __forceinline__ float half_warp_sum(float s) {
    s += __shfl_xor_sync(0xffffffffu, s, 8);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    return s;
}

// Parent kernel (S band): one row per half-warp, Q tiles then K tiles.
__global__ void __launch_bounds__(THREADS, 8)
qk_rms_single(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_tiles, float eps) {
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;
    const int r = threadIdx.x >> 4;
    const int c = (threadIdx.x & 15) * 8;
    const size_t row = (size_t)tile * ROWS + r;
    const int head = (tile % GROUPS) * ROWS + r;
    float xv[8], wv[8], yv[8];
    ld8(x + row * D + c, xv);
    ld8(w + (size_t)head * D + c, wv);
    const uint64_t pol_out = pol_evict_last();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = fmaf(xv[i], xv[i], s);
    const float inv = rsqrtf(half_warp_sum(s) * (1.0f / (float)D) + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) yv[i] = (xv[i] * inv) * wv[i];
    st8_h(y + row * D + c, yv, pol_out);
}

// M/L bands: each half-warp does row r of Q and row r of K (same head). 39 regs on sm_100a -> 6 CTAs/SM.
__global__ void __launch_bounds__(THREADS, 6)
qk_rms_pair(const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ wq, const float* __restrict__ wk,
            float* __restrict__ qo, float* __restrict__ ko, float eps) {
    const int tile = blockIdx.x;
    const int r = threadIdx.x >> 4;
    const int c = (threadIdx.x & 15) * 8;
    const size_t off = ((size_t)tile * ROWS + r) * D + c;
    const int hoff = ((tile % GROUPS) * ROWS + r) * D + c;
    float xq[8], xk[8], w[8], y[8];
    ld8(q + off, xq);                 // both DRAM loads in flight together
    ld8(k + off, xk);
    const uint64_t pol_out = pol_evict_last();
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { sq = fmaf(xq[i], xq[i], sq); sk = fmaf(xk[i], xk[i], sk); }
    sq += __shfl_xor_sync(0xffffffffu, sq, 8); sk += __shfl_xor_sync(0xffffffffu, sk, 8);
    sq += __shfl_xor_sync(0xffffffffu, sq, 4); sk += __shfl_xor_sync(0xffffffffu, sk, 4);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2); sk += __shfl_xor_sync(0xffffffffu, sk, 2);
    sq += __shfl_xor_sync(0xffffffffu, sq, 1); sk += __shfl_xor_sync(0xffffffffu, sk, 1);
    const float iq = rsqrtf(sq * (1.0f / (float)D) + eps);
    const float ik = rsqrtf(sk * (1.0f / (float)D) + eps);
    ld8(wq + hoff, w);                // L1/L2 hit
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = (xq[i] * iq) * w[i];
    st8_h(qo + off, y, pol_out);
    ld8(wk + hoff, w);
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = (xk[i] * ik) * w[i];
    st8_h(ko + off, y, pol_out);
}

void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
                " must be a contiguous fp32 CUDA tensor");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 31) == 0, name, " must be 32-byte aligned");
}
}  // namespace

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm) {
    check_tensor(query, "query"); check_tensor(key, "key");
    check_tensor(weight_q, "weight_q"); check_tensor(weight_k, "weight_k");
    check_tensor(query_norm, "query_norm"); check_tensor(key_norm, "key_norm");
    TORCH_CHECK(weight_q.numel() == H * D && weight_k.numel() == H * D, "weights must be [48,128]");
    TORCH_CHECK(query.numel() % ((int64_t)H * D) == 0 && key.numel() == query.numel() &&
                query_norm.numel() == query.numel() && key_norm.numel() == query.numel(), "shape mismatch");
    const int64_t n_rows = query.numel() / D;
    if (n_rows == 0) return;
    const int64_t n_tiles = n_rows / ROWS;          // exact: n_rows is a multiple of 48
    const int64_t tokens = n_rows / H;
    TORCH_CHECK(2 * n_tiles <= (int64_t)0x7fffffff, "grid too large");
    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (tokens >= PAIR_MIN_TOKENS) {
        qk_rms_pair<<<dim3((unsigned)n_tiles), THREADS, 0, stream>>>(
            query.data_ptr<float>(), key.data_ptr<float>(), weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
            query_norm.data_ptr<float>(), key_norm.data_ptr<float>(), (float)eps);
    } else {
        qk_rms_single<<<dim3((unsigned)(2 * n_tiles)), THREADS, 0, stream>>>(
            query.data_ptr<float>(), key.data_ptr<float>(), weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
            query_norm.data_ptr<float>(), key_norm.data_ptr<float>(), (int)n_tiles, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32, 256-bit, Q/K row pairs, EL stores)");
}
```

```yaml design-card
id: r6-ldg256-qkpair-r16
parents:
  - r5-ldg256-os-r16-stel
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows16
  red: halfwarp
  cache: stream
  spec: dispatch:size
hypothesis: >-
  For B*S > 600, one CTA processes the same 16 rows of Q and of K. Each half-warp issues both 256-bit
  row loads before either reduction, so each thread has 64 B of DRAM reads in flight instead of 32 B.
  That raises the estimate per SM from 64 KB to about 96 KB at 6 CTAs/SM (39 regs on sm_100a). It also
  halves the CTA count, which cuts per-CTA launch and exit overhead in the 40-plus-wave one-shot grid.
  Weights are loaded only after the reductions; they are L1/L2 hits and hold no registers across the
  DRAM latency. Stores keep the parent's L2 evict_last policy, so the outputs left in L2 at kernel end are
  unchanged. On A100, which already has enough reads in flight, the change is neutral
  (S +0.1, M +0.2, L -0.1%). The expected gain is B200-only, from its larger in-flight requirement.
  Measured and rejected in this session on A100: evict_first x loads (S +0.5, M +2.0, L +8.5%). Also
  rejected: a 32-register variant forced to 8 CTAs/SM (no spills on sm_100a), which was 3.5% slower on A100 M.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -3
    confidence: low
  L:
    pct: -3
    confidence: low
resources_sm100a:
  regs_per_thread: 39
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 6
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  ROWS: 16
  THREADS: 256
  PAIR_MIN_TOKENS: 601
  pair_min_blocks: 6
  store_policy: "L2 evict_last, fraction 1.0"
  load_policy: default
dispatch:
  - max_tokens: 600
    kernel: qk_rms_single
    meta:
      ROWS: 16
      grid: "2*n_tiles"
  - max_tokens: 9223372036854775807
    kernel: qk_rms_pair
    meta:
      ROWS: 16
      grid: "n_tiles"
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback; needs fewer bytes in flight per SM than B200"
  A100:
    runs: true
    representative: false
    note: "measured: 16/16 pass, S +0.1 / M +0.2 / L -0.1% vs parent (128-bit fallback)"
  L40S:
    runs: true
    representative: false
risks:
  - "6 CTAs/SM (39 regs) instead of 8 means 1536 resident threads; if B200 is limited by CTA turnover rather than bytes in flight, the gain may be zero."
  - "Q and K reads and writes now interleave inside each CTA (4 concurrent address streams); DRAM page locality could get slightly worse."
  - "Weight loads after the reduction add an L1-hit latency (about 26 ns) per CTA; small, but present on every tile."
  - "The predicted gain is from reasoning only; A100 cannot show it."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```