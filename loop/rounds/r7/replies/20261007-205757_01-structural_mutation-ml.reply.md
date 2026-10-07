### Rationale
I tested six variants on the rented B200 against r5. Changing which lines stay in L2 barely moved M or L. Evict_first on x loads gave S −1.8%, M −1.1%, L +0.2%. Keeping evict_last only for the last 96 MB of output gave nothing more. The only real gain came from one `cp.async.bulk.prefetch.L2` per CTA for the tile one resident wave ahead (148×8 CTAs). That cut S by about 3–7% and the 1024-token size by 4.5%, but it made L 4–5% slower. The likely cause is that once evict_last outputs fill the L2, prefetched lines are evicted before they are used. Prefetching with an evict_last hint was worse everywhere (M +4%, L +6%). Prefetching 2 waves ahead up to 2100 tokens gave less (S −1.1%, M −0.6%). So the candidate turns the prefetch on only for B·S ≤ 1536. Measured result: S −2.9%, M −2.6%, L 0.0%; projected portal score 0.6125 against 0.6089. L stays at about 7.0 TB/s marginal, which looks like the ceiling for a mixed read and write stream.

```json solution-spec
{"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}
```

```cuda file=kernel.cu
// r7-ldg256-l2pf-mdisp: per-head RMSNorm of Q and K (fp32), one fused one-shot launch.
// Structure of r5 (16 rows per 256-thread CTA, half-warp per row, 256-bit ld/st on sm_100a, output stores
// L2 evict_last). Changes: x loads carry L2 evict_first (policy constant), weights use ld.global.nc, and
// for B*S <= 1536 (total footprint <= ~150 MB) thread 0 of each CTA issues one cp.async.bulk.prefetch.L2 of
// the input tile that will be processed one resident wave (8 CTAs x #SMs) later. Above that size the
// prefetch is off (measured slower: prefetched lines compete with the evict_last outputs in a full L2).
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
constexpr int CTAS_PER_SM = 8;
constexpr int64_t PF_MAX_TOKENS = 1536;
constexpr uint64_t POL_EVICT_FIRST = 0x12F0000000000000ULL;   // CUTLASS CacheHintSm90::EVICT_FIRST

__device__ __forceinline__ uint64_t pol_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void l2_prefetch(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
// sm_80/sm_90 local-test path: two 128-bit accesses with the same cache policies.
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
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    ld4_h(p, v, pol); ld4_h(p + 4, v + 4, pol);
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    st4_h(p, v, pol); st4_h(p + 4, v + 4, pol);
}
#endif

template <bool PF>
__global__ void __launch_bounds__(THREADS, CTAS_PER_SM)
qk_rms_kernel(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_tiles, int pf_dist, float eps) {
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int r = threadIdx.x >> 4;
    const int c = (threadIdx.x & 15) * 8;
    const size_t row = (size_t)tile * ROWS + r;
    const int head = (tile % GROUPS) * ROWS + r;   // == row % 48

    float xv[8], wv[8], yv[8];
    ld8_h(x + row * D + c, xv, POL_EVICT_FIRST);    // both loads issued before any use
    ld8_nc(w + (size_t)head * D + c, wv);
    if (PF && threadIdx.x == 0) {                   // warm L2 for the CTA one resident wave later
        const int pb = bid + pf_dist;
        if (pb < 2 * n_tiles) {
            const bool pk = pb >= n_tiles;
            const int pt = pk ? pb - n_tiles : pb;
            l2_prefetch((pk ? k : q) + (size_t)pt * ROWS * D, ROWS * D * 4);
        }
    }
    const uint64_t pol_out = pol_evict_last();

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
    st8_h(y + row * D + c, yv, pol_out);
}

int g_num_sms = 0;   // device property cache (not data-dependent)

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
    if (g_num_sms == 0) {
        cudaDeviceProp prop;
        C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, query.device().index()));
        g_num_sms = prop.multiProcessorCount;
    }
    const int pf_dist = g_num_sms * CTAS_PER_SM;    // one resident wave ahead
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)(2 * n_tiles));
    const float* qp = query.data_ptr<float>(); const float* kp = key.data_ptr<float>();
    const float* wqp = weight_q.data_ptr<float>(); const float* wkp = weight_k.data_ptr<float>();
    float* qop = query_norm.data_ptr<float>(); float* kop = key_norm.data_ptr<float>();
    if (tokens <= PF_MAX_TOKENS)                     // shape-only dispatch
        qk_rms_kernel<true><<<grid, THREADS, 0, stream>>>(qp, kp, wqp, wkp, qop, kop, (int)n_tiles, pf_dist, (float)eps);
    else
        qk_rms_kernel<false><<<grid, THREADS, 0, stream>>>(qp, kp, wqp, wkp, qop, kop, (int)n_tiles, pf_dist, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "QK RMSNorm fp32: EF x loads, EL stores, L2 bulk prefetch one wave ahead for B*S<=1536");
}
```

```yaml design-card
id: r7-ldg256-l2pf-mdisp
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
  For footprints that fit in L2 (B*S <= 1536), one cp.async.bulk.prefetch.L2 per CTA brings in the input tile
  for the CTA one resident wave (148x8) later. That deepens the DRAM queue and turns most demand loads into
  L2 hits, which cuts the latency-bound ramp and tail. Above that size the prefetched (normal-priority) lines
  are evicted by evict_last outputs filling the L2: L was +4-5% slower with prefetch, and an evict_last hint
  on the prefetch was worse at every size. So large sizes run r5 plus evict_first x loads and nc weights,
  which is neutral. Measured on a rented B200 (unlocked clocks), all against r5:
  - evict_first loads alone: S -1.8 / M -1.1 / L +0.2%.
  - evict_last stores only on the last 96 MB of output: -2.3 / -1.0 / -0.3%.
  - prefetch at all sizes: -2.7 / -2.5 / +4.3%.
  - evict_last prefetch: -2.4 / +4.2 / +6.0%.
  - 2-wave prefetch up to 2100 tokens: -1.1 / -0.6 / 0.0%.
  - this dispatched version: S -2.9 / M -2.6 / L 0.0%, 16/16 pass, projected 0.6125.
  Outputs left in L2 are not the limit any more: L stays at about 7.0 TB/s marginal under every cache policy
  tried.
expected_effect:
  S:
    pct: -2
    confidence: medium
  M:
    pct: -2
    confidence: low
  L:
    pct: 0
    confidence: medium
resources_sm100a:
  regs_per_thread: 28
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  ROWS: 16
  THREADS: 256
  PF_MAX_TOKENS: 1536
  PF_DISTANCE: "1 resident wave = num_SMs x 8 CTAs (8 KB per prefetch)"
  load_policy: "x L2::cache_hint EVICT_FIRST constant; w ld.global.nc"
  store_policy: "L2 evict_last, fraction 1.0"
dispatch:
  - max_tokens: 1536
    kernel: qk_rms_kernel<true>
    meta:
      prefetch: "cp.async.bulk.prefetch.L2 one wave ahead"
  - max_tokens: 1000000000
    kernel: qk_rms_kernel<false>
    meta:
      prefetch: none
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback; the 50 MB L2 changes where prefetch stops paying"
  A100:
    runs: false
    representative: false
    note: "cp.async.bulk.prefetch needs sm_90+"
  L40S:
    runs: false
    representative: false
risks:
  - "The gain was measured on a rented B200 at unlocked clocks. Portal S/M changes are usually smaller (projection exponents 0.45 / 0.63)."
  - "The 1536-token threshold sits between measured sizes 1024 (gain) and 2048 (neutral). Sizes in between are not tested."
  - "The 4,1024 and 16,256 L sizes moved by about ±0.4% across runs; that is noise."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```