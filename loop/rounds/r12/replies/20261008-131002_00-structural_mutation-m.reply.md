### Rationale
E1 tested H18, and H18 is refuted at 1024 and 2048 tokens. Measured as asked (harness CUPTI timing, dirty zero-flush vs clean zero+read flush, interleaved reps, medians), the evict_first-load gain has a sharp edge. It is −6.5…−7.4% from 500 to 800 tokens, −3.9% at 900, −0.8% at 1024 and +1% at 1536–2048 tokens. This explains the −8% at 586 tokens. But no store hybrid helps at 1024 or 2048: tail-EL, head-EL and fractional evict_last all tie c2 or lose. With a clean flush, every store policy ties at 586–1024 tokens. So evict_last stores win only by avoiding the dirty-flush write-backs *during* the kernel, not by leaving output in L2 after it.

The candidate is c2 with the EF-load path extended to ≤ 900 tokens (the measured EF window). No workload falls in 601–900 tokens, so it behaves exactly like c2. run_tests passed 16/16; every band is within ±0.3% of c2 (two single workloads moved −2.5% and +2.0%, which is noise).

```json solution-spec
{"name": "r12-e1-h18-efcurve-disp", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "c2-cluster2-s-only with the evict_first-load R=1 path extended to <= 900 tokens (the measured EF-gain window); H18 hybrids refuted.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r12-e1-h18-efcurve-disp: c2-cluster2-s-only with the EF-load R=1 path extended from <= 600 to <= 900 tokens.
// Per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// A thread may handle R tokens of the SAME head (rows h, h+48, h+96, ...), so one weight load serves R rows.
// Dispatch: <= 900 tok R=1 + evict_first x loads + nc weights (cluster of 2 at <= 300 tok);
// 900 < tok <= 1100: R=3, 5 CTAs/SM; above: R=1. Output stores L2 evict_last everywhere.
// r12 E1 (H18): with evict_last stores, evict_first x loads cut the harness time by 6.5-7.4% at 500-800 tokens,
// 3.9% at 900, 0.8% at 1024 and cost +1% at 1536-2048 (dirty-flush effect; clean-flush gain ~0 below 1024).
// Tail/head-only evict_last and fractional evict_last policies were all slower at 1024-2048 tokens.
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

template <int R, bool LOAD_S, int MINB>
__global__ void __launch_bounds__(THREADS, MINB)
qk_rms_rtok(const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ wq, const float* __restrict__ wk,
            float* __restrict__ qo, float* __restrict__ ko, int n_cta, int n_tok, float eps) {
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
        if (R == 1 || tok0 + j < n_tok) {
            const size_t row = (size_t)(tok0 + j) * H + head;
            st8_h(y + row * D + c, yv, pol_out);
        }
    }
}

template <int R, bool LOAD_S, int MINB>
cudaError_t launch(const float* q, const float* k, const float* wq, const float* wk, float* qo, float* ko,
                   int n_tok, float eps, cudaStream_t st, bool cluster2) {
    const int n_cta = NG * ((n_tok + R - 1) / R);
    const unsigned grid = 2u * (unsigned)n_cta;      // always even (6 * tokens for R=1)
    if (!cluster2) {
        qk_rms_rtok<R, LOAD_S, MINB><<<dim3(grid), THREADS, 0, st>>>(q, k, wq, wk, qo, ko, n_cta, n_tok, eps);
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
    return cudaLaunchKernelEx(&cfg, qk_rms_rtok<R, LOAD_S, MINB>, q, k, wq, wk, qo, ko, n_cta, n_tok, eps);
}
}  // namespace

extern "C" int qk_rms_launch(const float* q, const float* k, const float* wq, const float* wk,
                             float* qo, float* ko, long long n_tok, float eps, void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    if (n_tok <= 0) return 0;
    cudaError_t err;
    if (n_tok <= 900)        err = launch<1, true, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, n_tok <= 300);   // S + EF window
    else if (n_tok <= 1100)  err = launch<3, false, 5>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);       // 1024 tok
    else                     err = launch<1, false, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);       // M/L: plain
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
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); cluster-of-2 launch at <= 300 tokens");
}
```

```yaml design-card
id: r12-e1-h18-efcurve-disp
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
tests:
  - H18
hypothesis: >-
  H18 (refuted at 1024-2048 tokens): footprint-capped store policies (evict_last only for the last T MiB of output,
  EF or EN stores earlier, EF loads) would bring the 586-token evict_first gain to 1024-2048 tokens. Measured: the EF
  gain has a sharp edge between 800 and 1024 tokens (39-50 MB of evict_last output), and every hybrid ties or loses
  to all-evict_last. The candidate is c2 with the EF-load R=1 path widened to <= 900 tokens (the measured window).
  No workload falls in 601-900 tokens, so it behaves exactly like c2.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: high
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 30
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  EF_MAX_TOKENS: 900
  R3_MAX_TOKENS: 1100
  CLUSTER2_MAX_TOKENS: 300
dispatch:
  - max_tokens: 300
    kernel: qk_rms_rtok
    meta: {R: 1, LOAD_S: true, MINB: 8, cluster: 2}
  - max_tokens: 900
    kernel: qk_rms_rtok
    meta: {R: 1, LOAD_S: true, MINB: 8}
  - max_tokens: 1100
    kernel: qk_rms_rtok
    meta: {R: 3, LOAD_S: false, MINB: 5}
  - max_tokens: null
    kernel: qk_rms_rtok
    meta: {R: 1, LOAD_S: false, MINB: 8}
paths:
  - {max_tokens: 300, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el, cluster: 2}
  - {max_tokens: 900, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: 1100, lang: cuda, width: 256, threads: 256, rows: 48, grid: oneshot, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
findings:
  - "probe E1 curve (R=1, 1xN, harness CUPTI, 5 interleaved reps, medians us; dirty = harness zero flush, clean = zero flush then a read of the same 2xL2 buffer). Columns: tok: (a) default x + EL stores dirty/clean/d-c | (b) EF x + EL dirty/clean/d-c | EF gain dirty/clean. 300: 6.11/5.33/0.78 | 5.95/5.31/0.64 | -2.6/-0.4%. 400: 7.15/6.35/0.80 | 6.87/6.34/0.54 | -3.9/-0.3. 500: 8.42/7.31/1.11 | 7.87/7.30/0.58 | -6.5/-0.2. 586: 9.41/8.19/1.21 | 8.74/8.19/0.54 | -7.1/-0.0. 700: 10.88/9.38/1.51 | 10.08/9.34/0.74 | -7.4/-0.3. 800: 12.11/10.35/1.76 | 11.29/10.34/0.96 | -6.7/-0.1. 900: 13.09/11.44/1.65 | 12.58/11.39/1.19 | -3.9/-0.4. 1024: 14.69/12.67/2.01 | 14.56/12.48/2.08 | -0.8/-1.5. 1200: 16.85/14.60/2.25 | 16.72/14.37/2.35 | -0.8/-1.6. 1536: 21.27/18.48/2.79 | 21.57/17.70/3.87 | +1.4/-4.2. 2048: 28.81/24.18/4.63 | 29.13/23.37/5.76 | +1.1/-3.3 [probe_b200]"
  - "H18 threshold: with EF loads the dirty cost stays flat at ~0.55 us up to 586-700 tokens (<= ~34 MB of EL output) and then grows. The EF gain falls from -6.7% at 800 tokens (39 MB) to -0.8% at 1024 (50 MB). The edge is sharp at about 40-45 MB of evict_last output, which explains the 586-token -8%. No production workload falls in 600-1000 tokens, so the effect cannot be extended to a scored size [probe_b200]"
  - "probe (c)/(d) and cross terms, dirty/clean/d-c us. 586: L0+EL 9.49/8.27/1.23; EF+EL 8.67/8.24/0.42; EF+EFst 10.95/8.29/2.66; L0+plain 10.95/8.23/2.72; L0+EFst 11.17/8.26/2.92; EF+plain 10.91/8.21/2.70; EF+ENst 10.93/8.23/2.70. 1024: 14.60/12.71/1.89; 14.27/12.58/1.69; 17.66/12.67/4.98; 17.67/12.68/4.99; 17.93/12.74/5.19; 17.57/12.58/4.99; 17.66/12.59/5.07. 2048: 28.51/24.06/4.45; 28.93/23.27/5.66; 32.63/26.07/6.56; 32.41/24.74/7.67; 32.64/25.38/7.26; 32.38/25.95/6.43; 32.32/25.97/6.36 [probe_b200]"
  - "key mechanism finding: after a clean flush all store policies tie at 586-1024 tokens (8.21-8.29 and 12.58-12.74 us). The evict_last-store gain (H1) is therefore entirely a dirty-flush effect. Non-EL stores (plain/EF/EN) pay 2.7 us of dirty cost at 586 and 5.0 at 1024; EL stores cut it to 1.2/1.9. EF loads cut it further only while the EL footprint stays under ~40 MB. Load policy alone (any store policy except EL) changes nothing dirty [probe_b200]"
  - "dead end (H18 step 2), 7 interleaved reps vs c2 paths (1024: R=3 14.14 us; 2048: R=1 28.69 us), EF loads, R=1, tail = last-dispatched K CTAs. 1024: all-EL +0.5%; tailEL20 EFst +11.9 / ENst +11.0; tailEL30 +3.7/+2.9; tailEL40 -0.6/-0.7; headEL (EL first, EF after) 20/30/40: +3.9/+0.7/+0.0%. 2048: all-EL +0.7; tailEL20 +6.0/+5.3; tailEL30 +3.0/+2.1; tailEL40 +1.3/+0.5; headEL20/30/40 +3.5/+1.7/+1.6%. Every hybrid ties or loses, and EL on the FIRST output beats EL on the last. Evict_last avoids in-kernel dirty write-backs; it is not output parked past the window [probe_b200]"
  - "dead end: fractional evict_last (0.75 or 0.5 with evict_first secondary; 0.75 with unchanged secondary), default or EF loads, vs c2 paths, 5 reps. 900 tok: best L1 EL.75/unch -0.8% (all-EL+EF -1.5%); 1024: +0.8..+9.8% (all-EL+EF -0.5%); 1200: -0.4..+5.8%; 2048: +0.5..+3.7%. Full evict_last (1.0) is the best store policy everywhere. Extends H15 to 900-1200 tokens [probe_b200]"
  - "EF loads after a CLEAN flush gain 3-4% at 1536-2048 tokens but lose 1% after the dirty flush; this cannot be used under the harness [probe_b200]"
  - "run_tests: 16/16 pass; vs c2-cluster2-s-only S -0.0%, M -0.1%, L +0.3% (same code on every workload; 4,256 -2.5% and 8,256 +2.0% are noise); predicted portal 0.6129 +- 0.006 [run_tests]"
  - "tooling: b200probe.harness_time has no clean-flush option; monkey-patching sol_execbench.core.bench.timing._clear_cache inside a probe (zero_ then int64 max over the same buffer) gives one. /tmp persists between probes, so a shared helper module (/tmp/e1lib.py) saves re-sending code. One harness_time call (rep 50) takes ~0.07 s [probe_b200]"
runs_on:
  H200:
    runs: true
    representative: false
    note: "sm_90 fallback uses 128-bit loads; the L2 dirty-flush effects are B200-specific"
  A100:
    runs: true
    representative: false
  L40S:
    runs: true
    representative: false
risks:
  - "None new: same code as c2 on all 16 workloads. The EF window to 900 tokens only matters for unscored shapes."
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```