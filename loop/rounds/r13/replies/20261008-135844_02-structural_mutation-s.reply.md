### Rationale
Bulk stores lost everywhere in the probes, so dispatch never uses them. The candidate below is c2 with the bulk-store path added behind a compile-time flag that is off for every size. I could not get `run_tests` to run it: the tool returned "Error executing tool run_tests" on both attempts, so 16/16 correctness and the projected portal score are unverified. `compile_b200` shows the same registers, shared memory and SASS mix as c2, which is why I expect it to behave like c2 (an expectation, not a measurement).

```json solution-spec
{"name": "r13-e3-bulkstore-off", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "c2 plus an unused cp.async.bulk store path (BULK flag off everywhere); E3/H23 probes show bulk stores lose", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r13-e3-bulkstore-off: c2-cluster2-s-only with an optional cp.async.bulk store path (BULK template flag).
// E3 / H23 probes showed the bulk store loses at every size (S +4.6..+8.7%, 586 tok +7%, 1024/2048 tok +5%) and pays the
// same dirty-flush cost as STG with evict_last, so dispatch never enables it; the path stays compiled for later rounds.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
constexpr int THREADS = 256;
constexpr int RPC = THREADS / 16;       // heads per CTA per token = 16
constexpr int NG = H / RPC;             // 3 head groups
constexpr uint64_t POL_EVICT_FIRST = 0x12F0000000000000ULL;

__device__ __forceinline__ uint64_t pol_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float* v) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
__device__ __forceinline__ void ld8_h(const float* p, float* v, uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st8_h(float* p, const float* v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol) : "memory");
}
#else
__device__ __forceinline__ void ld4(const float* p, float* v) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_nc(const float* p, float* v) {
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
__device__ __forceinline__ void ld4_h(const float* p, float* v, uint64_t pol) {
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p), "l"(pol));
}
__device__ __forceinline__ void st4_h(float* p, const float* v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "l"(pol) : "memory");
}
__device__ __forceinline__ void ld8(const float* p, float* v) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void ld8_nc(const float* p, float* v) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void ld8_h(const float* p, float* v, uint64_t pol) { ld4_h(p, v, pol); ld4_h(p + 4, v + 4, pol); }
__device__ __forceinline__ void st8_h(float* p, const float* v, uint64_t pol) { st4_h(p, v, pol); st4_h(p + 4, v + 4, pol); }
#endif

template <int R, bool LOAD_S, int MINB, bool BULK>
__global__ void __launch_bounds__(THREADS, MINB)
qk_rms_rtok(const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ wq, const float* __restrict__ wk,
            float* __restrict__ qo, float* __restrict__ ko, int n_cta, int n_tok, float eps) {
    __shared__ __align__(128) float tile[BULK ? 16 * D : 1];   // BULK only (R == 1): 8 KB output tile
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_cta;
    const int b = is_k ? bid - n_cta : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int g = b % NG;
    const int tok0 = (b / NG) * R;
    const int head = g * RPC + (threadIdx.x >> 4);
    const int c = (threadIdx.x & 15) * 8;

    float xv[R][8], wv[8];
#pragma unroll
    for (int j = 0; j < R; ++j) {                    // all x loads first
        if (R == 1 || tok0 + j < n_tok) {
            const size_t row = (size_t)(tok0 + j) * H + head;
            if (LOAD_S) ld8_h(x + row * D + c, xv[j], POL_EVICT_FIRST);
            else        ld8(x + row * D + c, xv[j]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) xv[j][i] = 0.0f;
        }
    }
    if (LOAD_S) ld8_nc(w + (size_t)head * D + c, wv);   // one weight load serves R rows
    else        ld8(w + (size_t)head * D + c, wv);
    const uint64_t pol_out = pol_evict_last();

#pragma unroll
    for (int j = 0; j < R; ++j) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) s = fmaf(xv[j][i], xv[j][i], s);
        s += __shfl_xor_sync(0xffffffffu, s, 8);
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        const float inv = rsqrtf(s * (1.0f / (float)D) + eps);   // * (1/128) is exact
        float yv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) yv[i] = (xv[j][i] * inv) * wv[i];
        if (BULK) {
            // R == 1: the CTA's 16 rows are contiguous in the output (8192 B)
            float4* t = reinterpret_cast<float4*>(tile + (threadIdx.x >> 4) * D + c);
            t[0] = make_float4(yv[0], yv[1], yv[2], yv[3]);
            t[1] = make_float4(yv[4], yv[5], yv[6], yv[7]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (threadIdx.x == 0) {
                float* dst = y + ((size_t)tok0 * H + g * RPC) * D;
                const unsigned sa = (unsigned)__cvta_generic_to_shared(tile);
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], 8192, %2;"
                             :: "l"(dst), "r"(sa), "l"(pol_out) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            }
        } else if (R == 1 || tok0 + j < n_tok) {
            const size_t row = (size_t)(tok0 + j) * H + head;
            st8_h(y + row * D + c, yv, pol_out);
        }
    }
}

template <int R, bool LOAD_S, int MINB, bool BULK = false>
cudaError_t launch(const float* q, const float* k, const float* wq, const float* wk, float* qo, float* ko,
                   int n_tok, float eps, cudaStream_t st, bool cluster2) {
    const int n_cta = NG * ((n_tok + R - 1) / R);
    const unsigned grid = 2u * (unsigned)n_cta;      // always even (6 * tokens for R=1)
    if (!cluster2) {
        qk_rms_rtok<R, LOAD_S, MINB, BULK><<<dim3(grid), THREADS, 0, st>>>(q, k, wq, wk, qo, ko, n_cta, n_tok, eps);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, qk_rms_rtok<R, LOAD_S, MINB, BULK>, q, k, wq, wk, qo, ko, n_cta, n_tok, eps);
}
}  // namespace

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_tok, float eps, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (n_tok <= 0) return 0;
    cudaError_t err;
    if (n_tok <= 600)        err = launch<1, true, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, n_tok <= 300);   // S
    else if (n_tok <= 1100)  err = launch<3, false, 5>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);       // 1024 tok
    else                     err = launch<1, false, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);       // M/L
    return (int)err;
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_tok, float eps, void* stream);

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
    const long long n_tok = query.numel() / (48 * 128);
    if (n_tok == 0) return;
    TORCH_CHECK(n_tok * 6 <= 0x7fffffffLL, "grid too large");
    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int err = qk_rms_launch(query.data_ptr<float>(), key.data_ptr<float>(), weight_q.data_ptr<float>(),
                            weight_k.data_ptr<float>(), query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
                            n_tok, (float)eps, (void*)stream);
    TORCH_CHECK(err == 0, "qk_rms launch failed: ", cudaGetErrorString((cudaError_t)err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); cluster-of-2 launch at small sizes");
}
```

```yaml design-card
id: r13-e3-bulkstore-off
parents:
  - c2-cluster2-s-only
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
  H23 test: a cp.async.bulk store (8 KB smem tile per CTA, fence.proxy.async, one elected thread, wait_group.read 0)
  would shorten the 0.67 us store/exit tail at 128-256 tokens, or pay less of the dirty-flush cost at 586-2048
  tokens. Result: refuted. Bulk stores are slower than STG.256 with evict_last at every size, and pay the same
  dirty-flush cost. The shipped kernel keeps the bulk path behind a template flag that is off everywhere, so it is
  functionally c2 (identical regs and SASS in compile_b200).
tests: [H23]
expected_effect:
  S: {pct: 0, confidence: medium}
  M: {pct: 0, confidence: medium}
  L: {pct: 0, confidence: medium}
resources_sm100a:
  regs_per_thread: 30
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs: {THREADS: 256, R_S: 1, R_1024: 3, BULK: false, cluster_max_tokens: 300}
dispatch:
  - {max_tokens: 300, kernel: qk_rms_rtok, meta: {R: 1, LOAD_S: true, cluster: 2}}
  - {max_tokens: 600, kernel: qk_rms_rtok, meta: {R: 1, LOAD_S: true}}
  - {max_tokens: 1100, kernel: qk_rms_rtok, meta: {R: 3}}
  - {max_tokens: null, kernel: qk_rms_rtok, meta: {R: 1}}
findings:
  - "probe (R=1 S path, cluster 2 at <=300 tok, 10 interleaved reps, harness CUPTI, medians, us), bulk_EL = bulk store with evict_last policy: 128 tok STG-EL 4.031, bulk_EL 4.382 (+8.7%), bulk no hint +8.0%, bulk EF +8.8%, bulk EL with full wait_group 0 +8.7%. 131 tok 4.161: bulk_EL +4.6%, no hint +5.3%, EF +5.4%. 256 tok 5.246: bulk_EL +7.4%, no hint +20.0%, EF +20.1%, wg0 +9.5%. 586 tok 8.721: bulk_EL +7.0%, no hint +24.2%, EF +24.5%, wg0 +9.7%. The 0.67 us store tail does not get shorter; the smem staging, fence and barrier cost more than they save [probe_b200]"
  - "probe, 1024 and 2048 tok, R=1 default loads, 8 reps, dirty (harness zero flush) vs clean (zero then read of the same buffer), us: 1024 STG-EL 15.32/12.86 (d-c 2.47); bulk_EL 16.12/13.62 (d-c 2.50); bulk no hint 17.40/13.63 (d-c 3.77); bulk_EL wg0 16.51/14.19 (d-c 2.32). 2048 STG-EL 28.53/25.59 (2.93); bulk_EL 30.00/27.08 (2.92); bulk no hint 31.33/26.98 (4.35); bulk_EL wg0 30.80/28.09 (2.72). Bulk EL pays the same dirty cost as STG EL and is 0.8-1.5 us slower in absolute terms [probe_b200]"
  - "H23 store half closed: the async proxy honours the evict_last policy the same way STG does. Without evict_last the bulk store avoids some of STG's dirty cost (3.77 vs about 5.0 us at 1024, 4.35 vs about 6.4-7.7 at 2048, the STG numbers being from the r12 E1 R=1 curve, not a same-run control), but it is still 2-3 us worse than STG with evict_last. No bulk variant wins at any size [probe_b200]"
  - "LDG/STG add-ons for E1's sweep (S path, same sizes): st.global.L2::evict_last qualifier form equals the createpolicy evict_last store within noise (-0.2/0.0/+0.5/-0.4% at 128/131/256/586 tok). st.global.wt is worse (+3.2/+6.9/+15.9/+24.0%), so the write-through qualifier is not an alternative to evict_last [probe_b200]"
  - "dead end: a bulk store of the whole 8 KB CTA tile needs __syncthreads plus a single issuing thread; the S path is latency-bound, so this serialises the store behind the slowest warp (and wait_group.read keeps the CTA resident until the smem has been read). Do not revisit bulk stores for this problem [probe_b200]"
  - "tooling: harness_time raised 'No timing results' once in the first draft, probably the known CUPTI flake or a variant that failed to launch; retrying each timing up to 3 times fixed it. b200probe.cuda_kernel accepts __cluster_dims__(CL,1,1) compiled into the kernel, so cluster-2 variants need no launch attribute [probe_b200]"
  - "run_tests returned 'Error executing tool run_tests' on both attempts (before any result), so 16/16 correctness and the projected portal score for this exact file are NOT verified. compile_b200 gave 30/48/30 regs, 0 B smem, 0 B local and the same LDG.256 x7, LDG.CONSTANT.256 x1, STG.256 x5 mix as c2 [compile_b200]"
paths:
  - {max_tokens: 300, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el, cluster: 2}
  - {max_tokens: 600, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: 1100, lang: cuda, width: 256, threads: 256, rows: 48, grid: oneshot, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
runs_on:
  H200: {runs: true, representative: false, note: "256-bit loads fall back to 2x128-bit on sm_90"}
  A100: {runs: false, representative: false, note: "cluster launch needs sm_90+"}
  L40S: {runs: false, representative: false}
risks:
  - "run_tests did not complete, so correctness on all 16 workloads is untested for this exact build; it relies on the generated code being identical to c2"
  - "Dead BULK code could be flagged on review; it is compiled out for every instantiation"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```