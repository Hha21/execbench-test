### Rationale
E2 placement test for H17. A cluster-of-2 launch gives each SM the same number of CTAs as a plain launch (5 or 6 at 128 tokens) and the same TPC pairing. Only the dispatch order differs, and cluster dispatch is slower. Inside the CTAs the run takes the same time: 3024 ns for both at 128 tokens. Even an empty 768-CTA kernel is 0.15 µs faster with cluster 2. So the gain comes from the launch front-end or teardown, outside the CTAs, not from where the CTAs land. Nothing tried at the program level moved the time in the cluster's favour. A plain launch that reorders tiles SM by SM gained nothing; staggered load issue cost +3 to +14%; reversed order and other cluster CTA sizes did not help either. No variant beats c2 by 2% or more, so H17 is refuted as stated and should be parked. The candidate is c2's code unchanged, with comments recording the mechanism. run_tests: 16/16 pass, S +0.1%, M −0.4%, L 0.0% against c2, predicted portal score 0.6122.

```json solution-spec
{"name": "r12-e2-h17-cl2-frontend", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "c2-cluster2-s-only code unchanged; E2 shows the cluster-2 gain is a ~0.15 us launch front-end/teardown saving, not CTA placement", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r12-e2-h17-cl2-frontend: c2-cluster2-s-only, code unchanged; comments record experiment E2 (H17).
// Per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// A thread may handle R tokens of the SAME head (rows h, h+48, h+96, ...), so one weight load serves R rows.
// Dispatch: S (<=600 tok) R=1 + EF x loads + nc weights; 600 < tok <= 1100: R=3, 5 CTAs/SM; above: R=1.
// Output stores L2 evict_last everywhere. Cluster-of-2 launch for tok <= 300 only.
// E2 (r12): why cluster 2 helps. Same per-SM CTA histogram and TPC pairing as a plain launch; only the dispatch
// order differs (cluster: SMs in index order, strict period-148 round robin; plain: round robin over GPCs) and
// cluster dispatch is slower (start p50 384 vs 64 ns). In-kernel globaltimer span is identical (3024 vs 3024 ns at
// 128 tok); the ~0.15 us saving is outside the CTAs (launch front-end/teardown) and also shows on an empty
// 768-CTA kernel. Plain-launch permutations, staggered issue, reversed order, and other cluster CTA shapes do not
// beat it. Slower cluster dispatch costs at multi-wave grids (empty 6144 CTAs: +1.4 us), hence <= 300 tok only.
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
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); cluster-of-2 launch at <= 300 tokens");
}
```

```yaml design-card
id: r12-e2-h17-cl2-frontend
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
  - H17
hypothesis: >-
  E2 tested whether c2's cluster-of-2 gain at <= 256 tokens comes from CTA placement or dispatch order. It does not.
  Per-SM CTA counts, TPC pairing and die split are the same as a plain launch, and the in-kernel span is the same.
  The ~0.15 us saving is a fixed launch front-end/teardown cost outside the CTAs; an empty 768-CTA kernel shows it
  too. No permutation, staggered issue or cluster CTA shape enlarges it, so the code is c2's, unchanged.
  H17 is refuted; park it and stop S-band launch work.
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
  THREADS: 256
  R_S: 1
  R_1024: 3
  cluster_max_tokens: 300
paths:
  - {max_tokens: 300, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el, cluster: 2}
  - {max_tokens: 600, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: 1100, lang: cuda, width: 256, threads: 256, rows: 48, grid: oneshot, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
findings:
  - "probe E2 step1 (%smid log, 5 cold-L2 reps): 128 tok, both launches give 5 CTAs on 120 SMs and 6 on 28; both CTAs of every blockIdx pair (2b,2b+1) land on one TPC (100%); 372/768 CTAs on die A for both. Placement is identical [probe_b200]"
  - "probe E2 step1: dispatch order differs only in SM order. Cluster-2 is a strict period-148 round robin in SM index order (142..147,0,1,2,3,...; SM142 gets bids 0,148,296,...). Plain round-robins over GPCs in TPC pairs (142..147,0,1,16,17,32,33,...) and its later rounds are irregular (SM142 gets 0,108,236,366,...) [probe_b200]"
  - "probe E2 step1: cluster dispatch is slower. CTA start spread p50/p90/max 384/544/768 ns (cluster) vs 64/480/640 ns (plain) at 128 tok; first-load-return and end distributions are the same (firstld p50 1760 ns, end p50 2432 ns for both) [probe_b200]"
  - "probe E2: in-kernel globaltimer span (first CTA start to last CTA end, 40 cold reps) 128 tok 3024 vs 3024 ns, 256 tok 4416 (plain) vs 4480 (cluster), while CUPTI is 4.19 vs 4.03 and 5.60 vs 5.39 us. The gain is entirely outside the CTAs [probe_b200]"
  - "probe E2: the cluster saving appears in every traffic mode at 128 tok (768 CTAs): full -0.161 us, read-only -0.161, write-only -0.128, EMPTY kernel -0.153 (1.616 -> 1.463). Mechanism: a cheaper launch front-end/teardown for a one-wave cluster grid. This contradicts r11's 'empty kernel unchanged by cluster dims' (r11 used the launch attribute; here __cluster_dims__ was compiled in, 8 params) [probe_b200]"
  - "probe E2: the empty kernel loses with cluster 2 once the grid exceeds one wave: 1536 CTAs +0.24 us, 6144 CTAs +1.43 us (4.11 -> 5.55). Slow cluster CTA dispatch is hidden behind memory work at 256 tok (full -0.17 us) and roughly breaks even at 1024 tok (full -0.03). This is a likely mechanism for H16's portal losses at M/L at the 1500 MHz clock [probe_b200]"
  - "dead end (step 2i): a plain launch with an SM-major contiguous tile permutation is +4.9% at 128 tok and +5.6% at 131 vs cluster-2, the same as plain identity order (+4.1/+5.0%). Reversed blockIdx order: plain -0.2/-0.1/+0.6%, cluster -4.0/-3.7/-2.3% vs plain at 128/131/256. Order is not the mechanism [probe_b200]"
  - "dead end: staggering load issue in a plain launch (bid-proportional spin of 0-300/600/1000 ns) costs +3/+4.5/+8% at 128 tok and +3/+5/+14% at 256; cluster + 300 ns stagger loses its gain. Smoothing the DRAM burst does not help [probe_b200]"
  - "dead end (step 2ii): cluster pairs taking Q tile i and K tile i (half the grid apart) instead of adjacent tiles: -0.8% at 128 tok, +2.8% at 131 vs c2. Pairing a working CTA with an idle partner: +12% (128) / +24% (256) [probe_b200]"
  - "dead end (step 2iii, 10 interleaved reps, vs c2 S path): cluster 2 of 128-thread CTAs +9.9/+10.5% (128/131 tok); 384 threads -0.6/+0.6/+0.3% (128/131/256, a tie); 512 threads with masks +1.2/+2.0%; 768 threads (one token per CTA) +1.6/+1.7/+13%; R=2 rows per thread (half the threads) cluster +3.2/+5.6/+7.5%, plain +5.6/+7.5/+10.6%. Step 2iv (DSMEM weight sharing) skipped: nothing pointed at L2 request count [probe_b200]"
  - "H17 verdict: refuted. Placement and pairing are identical; the order differs, but reproducing or changing it moves nothing; the gain is a ~0.13-0.16 us fixed launch-path saving at one-wave grids. Park H17 and stop S-band launch work [inferred from probes]"
  - "run_tests: 16/16 pass; vs c2 on the same GPU S +0.1%, M -0.4%, L -0.0% (identical code); predicted portal 0.6122 +- 0.006 [run_tests]"
  - "tooling: NVRTC in b200probe has no <cstdint> (typedef uint64_t yourself); files written to /tmp do not persist between probes; harness_time with warmup=5, rep=40 takes about 0.1 s per call, so 8-10 interleaved reps of 8 variants at 3 sizes fit in one probe [probe_b200]"
runs_on:
  H200:
    runs: true
    representative: false
    note: "no 256-bit path; cluster launch behaviour differs"
  A100:
    runs: false
    representative: false
    note: "cluster launch needs sm_90+"
  L40S:
    runs: false
    representative: false
risks:
  - "Identical to c2: no expected score change; it exists only to document E2"
  - "The front-end saving was measured on the rented B200 at unlocked clocks; its portal size at 1500 MHz is as c2 measured (-4% at 128 tok)"
measure_first: []
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```