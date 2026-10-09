H21 is refuted, so this candidate is c2 with only the comments changed. It does not need a portal slot. `run_tests` passed 16/16 and is a tie with c2: S +0.6%, M +0.1%, L −0.1%, predicted 0.6123 ± 0.006.

### Rationale
**The test (step 2).** I copied the harness's setup for each call: shifted input copy, output zero, `cudaCtxResetPersistingL2Cache`, then the 265 MB zero flush. The measured call was always the same c2 kernel. Only the previous call's store policy alternated, evict_last vs plain, for 40 interleaved repeats per size. The difference was 0.00 to −0.15% at 128, 1024, 2048 and 8192 tokens.

A control shows the test could detect the effect. When the reset was skipped before the measured call, the previous call's evict_last outputs were still in L2. They cost +2.05, +4.1 and +6.1 µs at 1024, 2048 and 8192 tokens (35–40 of 40 repeats). So the harness reset clears them, outside the timed window.

**Steps 3 and 4.** Step 3 (demoting our own outputs at the end of the kernel) was skipped because H21 failed. For step 4 I retested H4, "fatter CTAs don't help", with 7 interleaved repeats. c2's dispatch is already the best of R = 1–4 tokens per thread at every size.

```json solution-spec
{"name": "r14-a1-h21-audit-c2", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "Assumption audit A1: H21 refuted directly (previous call's store policy moves the measured call < 0.2% under the harness reset+flush); code identical to c2-cluster2-s-only", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r14-a1-h21-audit-c2: code identical to c2-cluster2-s-only (comments only). Assumption audit A1 (H21).
// r14 probes: with the harness's cudaCtxResetPersistingL2Cache + zero flush, the PREVIOUS call's store policy
// (evict_last vs plain) changes the measured call by <= 0.15% at 128/1024/2048/8192 tokens (40 interleaved reps).
// Without the reset, the previous call's surviving evict_last outputs cost +2.0/+4.1/+6.1 us at 1024/2048/8192,
// so the test was sensitive and the reset is what removes them. H21 refuted; no end-of-kernel demotion is needed.
// Per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// A thread may handle R tokens of the SAME head (rows h, h+48, h+96, ...), so one weight load serves R rows.
// Dispatch: S (<=600 tok) R=1 + EF x loads + nc weights (cluster of 2 at <= 300 tok);
// 600 < tok <= 1100: R=3, 5 CTAs/SM; above: R=1. Output stores L2 evict_last everywhere.
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
    if (n_tok <= 600)        err = launch<1, true, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, n_tok <= 300);   // S
    else if (n_tok <= 1100)  err = launch<3, false, 5>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);       // 1024 tok
    else                     err = launch<1, false, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);           // M/L: plain, as r10-rtok
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
id: r14-a1-h21-audit-c2
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
  - H21
  - H4
hypothesis: >-
  H21 (the previous call's evict_last outputs survive the harness reset plus zero flush and are written back in the next
  timed window) tested directly and refuted: with the harness reset, the previous call's store policy moves the measured
  call by <= 0.15% at 128-8192 tokens, while the same test without the reset shows +2/+4/+6 us, so the test was
  sensitive. No end-of-kernel demotion is needed; the candidate is c2's code with comments changed and needs no portal slot.
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
  S_threshold_tokens: 600
  cluster2_max_tokens: 300
  R_1024band: 3
paths:
  - {max_tokens: 300, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el, cluster: 2}
  - {max_tokens: 600, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: 1100, lang: cuda, width: 256, threads: 256, rows: 48, grid: oneshot, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
findings:
  - "A1 settled-belief audit, weakest first (strength 1-5): (1) H24 'the leader halves the dirty cost' - arithmetic only, no measurement, strength 1; (2) H11 'L ceiling is DRAM-side, no SM change lifts L' - r10/r11 read-only probes, not interleaved, and r13 itself notes clean-flush copies run at 8+ TB/s apparent, so the ceiling is a dirty-L2 ceiling as stated, strength 2; (3) H2 at L 'evict_first x loads hurt' - one non-interleaved portal pair (r7 vs c1 +2.4%) against bench +0.3..+0.5%, strength 2; (4) H9 'portal penalises SM work ~5% per 20% lag' - emulator fit on 18-22 kernels plus one pair (r8/r9), with r10-rtok R=3 as a counterexample, strength 3; (5) H4 'fatter CTAs refuted' - r10 one-wave probe plus portal r8 (confounded with L2 mode), yet R=3 at 1024 won on the portal (-2.4%), so the label was too broad; retested here, strength now 4; (6) H15 'about 25-30 MB evict_last cap, store end-state final' - r10/r12 probes at 900-8192 tokens, interleaved in r12, strength 3; (7) 'no legitimate mechanism for the dirty-flush cost' (H7/H22) - r13 E1 sweep with 5 interleaved reps per cell, and H23 bulk paths closed, but the mechanism is inferred, strength 4; (8) H6 'persistent/stealing loses' - r10 probes plus 3 portal results (Triton persistent, no evict_last) and bench r10-ws-l-steal, strength 4; (9) H10 'S at the copy floor' - medians of 105 (r11 E3) plus timelines, though r11-consol reports a 3-5% copy lag at S, strength 4; (10) H17/H19/H20 (S launch, L2 prefetch, tile order) - 5-15 interleaved reps each, strength 5 [ledger review]"
  - "H21 direct test (step 2) [probe_b200]: harness-like sequence per call (shifted input copy, output zero, cudaCtxResetPersistingL2Cache, 265 MB zero flush), CUDA events around the measured c2 path (evict_last stores), previous call alternating evict_last vs plain store policy, 40 interleaved reps, 4 arms rotated. With reset, previous EL minus previous plain (paired median): 128 tok +0.03 us (+0.35%, 22/40 slower); 1024 -0.03 us (-0.15%, 9/40); 2048 0.00 (14/40); 8192 +0.03 us (+0.03%, 23/40). All under 1%, so H21 is refuted by its own criterion"
  - "H21 positive control [probe_b200]: if the measured call's setup skips the reset, the previous call's evict_last outputs cost +2.05 us (+9.5%, 35/40) at 1024, +4.10 us (+12.1%, 39/40) at 2048, +6.10 us (+5.0%, 38/40) at 8192, and 0 at 128 (6 MB fits the clean room). With a plain previous call, skipping the reset costs 0. The harness reset demotes them and the flush then evicts them outside our window, which agrees with the r13-e2 latency probe. So the full evict_last gain (H1) is genuinely free and nothing is paid back across calls"
  - "first H21 probe design flaw (fixed in the rerun): without a reset before the previous call, older measured calls' evict_last lines piled up and masked the previous call's policy; also, events must be preceded by torch.cuda._sleep so the CPU queues ahead [probe_b200]"
  - "event timing caveat: CUDA-event spans include a constant ~5-7 us over the harness CUPTI times (128 tok 9.2 vs 4.0 us), fine for paired differences but not absolute; at 128 tok it may hide sub-0.1 us effects [probe_b200]"
  - "step 3 skipped: H21 does not hold, so no demotion candidate; r13-e2 already measured applypriority demotion as +2..+30% [ledger]"
  - "step 4, H4 retest with harness_time CUPTI, 7 interleaved reps, R tokens of the same head per thread, all with evict_last stores [probe_b200]: 586 tok R1-EF 8.64 us (base), R2-EF +13.7%, R3-EF +7.2%, R1-plain +9.3%. 1024: R3 14.19 (base), R1 +2.7%, R2 +6.2%, R4 +0.5%, R3-EF +0.1%. 2048: R1 28.55, R2 +3.8%, R3 +0.4%, R4 +0.4%. 4096: R1 58.02, R2 +1.7%, R3 +0.4%. 8192: R1 115.96, R2 +1.4%, R3 +0.8%. c2's dispatch is the best R at every size; R=2 is consistently the worst (cause unknown; 6 CTAs/SM by launch bounds). Fatter CTAs help only at 1024 tokens; H4 stands everywhere else"
  - "dead end: nothing related to evict_last remains to be recovered across calls (H21 closed by two independent methods, latency in r13-e2 and timing here) [probe_b200]"
  - "run_tests (c2 code, comments only): 16/16 pass; vs c2 S +0.6%, M +0.1%, L -0.1% (noise); predicted portal 0.6123 +- 0.0059 [run_tests]"
runs_on:
  H200:
    runs: true
    representative: false
    note: "falls back to 128-bit accesses; no cluster/evict_last behaviour equivalent to B200's L2"
  A100:
    runs: false
    representative: false
    note: "cluster launch needs sm_90+"
  L40S:
    runs: false
    representative: false
risks:
  - "None new: code identical to c2 (portal 0.6125). Do not spend a portal slot on it"
measure_first: []
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```