### Rationale
I tested H23's load half with interleaved harness-CUPTI probes (7–9 reps, dirty zero-fill flush and a clean flush). The bulk-load kernels used no L2 hint, a createpolicy evict_first hint or an evict_last hint, with 1, 2 or 4 tiles of 8 KB per CTA. **H23's load half is refuted.**
- **Clean flush:** bulk loads are 4–13% slower at 128–1024 tokens.
- **Dirty-flush cost:** a variant's dirty cost (dirty-flush time minus clean-flush time) only fell when its clean-flush time rose. The slower kernel hides some write-backs; it removes none.
- **Hints:** evict_first on the bulk copy gives about the same recycling as evict_first LDG and is no better. evict_last on the bulk copy is the worst.
- **2 tiles per CTA, no hint:** ties c2 at 2048–8192 tokens, within ±0.6%.

The candidate is c2 with that 2-tile path at 1100 < B·S ≤ 2100 tokens. It passes 16/16, with a predicted portal score of 0.6125 ± 0.006, a tie with c2. It is an exploration record and does not need a portal slot.

```json solution-spec
{"name": "r13-e4-bulkld-t2-disp", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "c2 dispatch plus a one-shot cp.async.bulk load path (2 x 8 KB tiles per CTA, no L2 hint, STG.256 evict_last) at 1100 < B*S <= 2100; E4/H23 exploration record, ties c2", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r13-e4-bulkld-t2-disp: c2-cluster2-s-only plus a one-shot cp.async.bulk load path at 1100 < B*S <= 2100 tokens
// (experiment E4 / H23). Per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// c2 paths: <=300 tok R=1 + EF x loads + nc weights + cluster-2; <=600 the same without cluster; <=1100 R=3;
// above 2100 R=1 plain. Output stores L2 evict_last everywhere.
// Bulk path: each 256-thread CTA owns two adjacent 16-row (8 KB) tiles; thread 0 inits two mbarriers and issues two
// 8 KB cp.async.bulk loads (no L2 hint) at kernel start; each half-warp reads its row's 8 floats from shared memory,
// reduces, multiplies by the weight (ld.global.nc) and stores with STG.256 evict_last. Bench: ties c2 at 2048-8192
// tokens (+-0.6%), so it is an exploration record, not a speed-up.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int D = 128;
constexpr int H = 48;
constexpr int THREADS = 256;
constexpr int RPC = THREADS / 16;       // heads per CTA per token = 16
constexpr int NG = H / RPC;             // 3 head groups
constexpr int TILE = 16 * D;            // floats per 16-row tile (8 KB)
constexpr int TPC = 2;                  // tiles per CTA on the bulk path
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

__device__ __forceinline__ uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// Bulk-load path: tiles t0 = 2*blockIdx.x and t0+1 over the concatenated [Q tiles | K tiles] list (n_tiles per tensor).
__global__ void __launch_bounds__(THREADS)
qk_rms_bulk2(const float* __restrict__ q, const float* __restrict__ k,
             const float* __restrict__ wq, const float* __restrict__ wk,
             float* __restrict__ qo, float* __restrict__ ko, int n_tiles, float eps) {
    const int tid = threadIdx.x;
    const int t0 = blockIdx.x * TPC;
    const int r = tid >> 4, c = (tid & 15) * 8;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __shared__ __align__(128) float sm[TPC * TILE];       // 16 KB
    __shared__ __align__(8) uint64_t bar[TPC];
    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < TPC; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(saddr(&bar[j])));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
        for (int j = 0; j < TPC; ++j) {
            const int t = t0 + j;
            const float* src = t < n_tiles ? q + (size_t)t * TILE : k + (size_t)(t - n_tiles) * TILE;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         :: "r"(saddr(&bar[j])), "r"((unsigned)(TILE * 4)) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(saddr(sm + j * TILE)), "l"(src), "r"((unsigned)(TILE * 4)), "r"(saddr(&bar[j])) : "memory");
        }
    }
#endif
    float wv[TPC][8];
#pragma unroll
    for (int j = 0; j < TPC; ++j) {
        const int t = t0 + j;
        const bool is_k = t >= n_tiles;
        const int lt = is_k ? t - n_tiles : t;
        const int head = ((lt * 16) % H) + r;
        ld8_nc((is_k ? wk : wq) + (size_t)head * D + c, wv[j]);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __syncthreads();                                      // barrier init visible to every thread
#endif
    const uint64_t pol_out = pol_evict_last();
#pragma unroll
    for (int j = 0; j < TPC; ++j) {
        const int t = t0 + j;
        const bool is_k = t >= n_tiles;
        const int lt = is_k ? t - n_tiles : t;
        float* y = (is_k ? ko : qo) + (size_t)lt * TILE + r * D + c;
        float xv[8];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("{\n.reg .pred P;\nWAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}"
                     :: "r"(saddr(&bar[j])), "r"(0u) : "memory");
        const float* p = sm + j * TILE + r * D + c;
        const float4 a = *reinterpret_cast<const float4*>(p);
        const float4 a2 = *reinterpret_cast<const float4*>(p + 4);
        xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w; xv[4] = a2.x; xv[5] = a2.y; xv[6] = a2.z; xv[7] = a2.w;
#else
        ld8((is_k ? k : q) + (size_t)lt * TILE + r * D + c, xv);
#endif
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) s = fmaf(xv[i], xv[i], s);
        s += __shfl_xor_sync(0xffffffffu, s, 8);
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        const float inv = rsqrtf(s * (1.0f / (float)D) + eps);
        float yv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) yv[i] = (xv[i] * inv) * wv[j][i];
        st8_h(y, yv, pol_out);
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
    else if (n_tok <= 2100) {                                                                                  // 2048 tok: bulk loads
        const int n_tiles = 3 * (int)n_tok;          // 16-row tiles per tensor; 2*n_tiles is even, so TPC=2 divides it
        qk_rms_bulk2<<<dim3((unsigned)(2 * n_tiles / TPC)), THREADS, 0, st>>>(q, k, wq, wk, qo, ko, n_tiles, eps);
        err = cudaGetLastError();
    }
    else                     err = launch<1, false, 8>(q, k, wq, wk, qo, ko, (int)n_tok, eps, st, false);           // L
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
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); c2 dispatch plus a bulk-load path at 2048 tokens");
}
```

```yaml design-card
id: r13-e4-bulkld-t2-disp
parents:
  - c2-cluster2-s-only
  - r10-bulk1d-ring-stel
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: oneshot
  launch: fused
  tile: rows16
  red: halfwarp
  cache: stream
  spec: dispatch:size
tests:
  - H23
hypothesis: >-
  E4 / H23 (load half): async-proxy bulk loads (cp.async.bulk with no hint, evict_first or evict_last L2 hint) allocate
  in L2 differently from LDG.256 and pay less of the dirty-flush cost at 586-2048 tokens. REFUTED on the rented B200.
  Bulk loads are 4-13% slower than LDG.256 after a clean flush at 128-1024 tokens. Their dirty-minus-clean cost drops
  only when the clean time rises (a slower kernel hides some write-backs; none are avoided). The evict_first bulk hint
  recycles no better than evict_first LDG, and evict_last is the worst. The only competitive variant (2 x 8 KB tiles
  per CTA, no hint) ties c2 at 2048-8192 tokens. This candidate puts it on the 1100-2100 token band (the two 2048-token
  workloads) and keeps c2 everywhere else, as an exploration record. Expected portal result: a tie with c2.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: medium
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 38
  smem_per_cta_bytes: 17424
  threads_per_cta: 256
  ctas_per_sm: 6
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  TPC: 2
  TILE_BYTES: 8192
  bulk_policy: none
  bulk_band_tokens: "1100 < B*S <= 2100"
dispatch:
  - {max_tokens: 300, kernel: qk_rms_rtok, meta: {R: 1, LOAD_S: true, cluster: 2}}
  - {max_tokens: 600, kernel: qk_rms_rtok, meta: {R: 1, LOAD_S: true}}
  - {max_tokens: 1100, kernel: qk_rms_rtok, meta: {R: 3}}
  - {max_tokens: 2100, kernel: qk_rms_bulk2, meta: {TPC: 2, policy: none}}
  - {max_tokens: null, kernel: qk_rms_rtok, meta: {R: 1}}
findings:
  - "probe E4, 586 tok, 7 interleaved reps, harness CUPTI, dirty/clean/d-c us: c2 R=1 EF-LDG 8.74/8.23/0.51; R=1 plain LDG 9.40/8.24/1.16; bulk T=1 none/EF/EL 10.44/8.93/1.51, 9.71/8.91/0.80, 10.50/8.91/1.59; bulk T=2 none/EF/EL 9.92/8.81/1.12, 9.58/8.83/0.74, 9.79/8.80/0.99. Bulk loads +7-8% clean; the EF bulk hint recycles about like EF LDG (0.74-0.80 vs 0.51 us) and is not better [probe_b200]"
  - "probe E4, 1024 tok (base c2 R=3 14.34/12.74/1.61): c2 R=1 +2.4% dirty (d-c 2.11); R=1 EF-LDG +0.0% (1.80); bulk T=1 none/EF/EL +13.5/+6.6/+14.9% dirty, +5-6% clean; bulk T=2 none/EF/EL +1.7/+2.2/+6.4% dirty, +3.8% clean, d-c 1.36/1.43/2.03. Repeat (9 reps): T=2 none +1.8% dirty, +3.5% clean, d-c 1.33 vs 1.52; T=4 none/EF +10.5/+7.8% [probe_b200]"
  - "probe E4, 2048 tok (base c2 R=1 28.77/24.07/4.70): R=1 EF-LDG +1.3% dirty, -2.1% clean (d-c 5.57); bulk T=1 none/EF/EL +4.2/+3.7/+5.1% dirty, +11.8/+4.5/+11.9% clean (d-c 3.05/4.70/3.29); bulk T=2 none/EF/EL -0.6/+1.2/+4.3% dirty, -0.5/-2.3/-0.4% clean (d-c 4.66/5.60/6.02). Repeat (9 reps): T=2 none +0.6% dirty, -0.6% clean; T=4 none +1.7%, T=4 EF +4.8% [probe_b200]"
  - "probe E4, 4096 tok (9 reps; c2 R=1 58.08/50.29/7.79): bulk T=2 none +0.3%/+0.9% (d-c 7.54); T=4 none +0.7/+1.5% (7.41); T=4 EF +1.6/+5.5% (5.93). 8192 tok dirty only: c2 116.11, T=2 116.56 (+0.4%), T=4 116.86 (+0.6%) [probe_b200]"
  - "probe E4, 128 tok (5 reps; c2 S path 4.19/3.55/0.64): every bulk variant is +5.4..+7.7% dirty and +12.6..+13.5% clean, which confirms the ~0.45 us mbarrier/bulk setup penalty at one wave; keep c2's S path [probe_b200]"
  - "H23 load-half verdict: refuted. No bulk variant cuts the dirty cost by >= 3% of total at 1024 or 2048 tokens without a larger clean-time loss. Lower d-c values appear only in variants that are slower after a clean flush: slowness hides write-backs, it does not avoid them. The evict_first bulk hint behaves like evict_first LDG (good at 586 tokens, a loss at 2048 dirty), and the evict_last bulk hint is the worst at every size. Close the load half of H23 and do not revisit async-proxy loads for #38 [probe_b200]"
  - "more tiles per CTA help the bulk path (T=1 -> T=2 cuts clean time by 4-11% at 1024-2048 tokens), but T=4 (32 KB) is worse again at every size. T=2 no-hint is the best bulk configuration and only ties LDG.256 at >= 2048 tokens [probe_b200]"
  - "compile sm_100a: bulk2 kernel 38 regs, 17,424 B static smem, 0 local, UBLKCP x2; c2 paths unchanged (30/48/30 regs) [compile_b200]"
  - "run_tests: first run 14/16 (1x131 and 1x256 hit the known CUPTI 'No timing results' flake on unchanged c2 paths); rerun 16/16. vs c2 on the same GPU: S +0.3%, M -0.3%, L +0.3%; 2048-token workloads -0.1% / -0.7%; predicted portal 0.6125 +- 0.006, P(beats c2) = 50%: a tie [run_tests]"
  - "tooling: a clean flush in probes = monkey-patch sol_execbench.core.bench.timing._clear_cache with zero_() then .view(torch.int64).max() over the same buffer; /tmp/e4lib.py (kernels plus a bench/report helper) persisted between probes in this session [probe_b200]"
paths:
  - {max_tokens: 300, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el, cluster: 2}
  - {max_tokens: 600, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: 1100, lang: cuda, width: 256, threads: 256, rows: 48, grid: oneshot, st: el}
  - {max_tokens: 2100, lang: cuda, width: 256, threads: 256, rows: 32, grid: oneshot, mem: tma, w: nc, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
runs_on:
  H200:
    runs: true
    representative: false
    note: "bulk path runs on sm_90 but STG.256 falls back to 128-bit; smaller L2 changes the dirty-flush behaviour"
  A100:
    runs: true
    representative: false
    note: "sm_80 falls back to plain loads on the bulk path; correctness only"
  L40S:
    runs: true
    representative: false
risks:
  - "Ties c2 on the bench; it does not deserve a portal slot except as an A/B of the bulk path at 2048 tokens"
  - "At the portal's 1500 MHz SM clock the mbarrier setup (~0.45 us at one wave) could cost slightly more at 2048 tokens"
  - "38 regs on the bulk path limits it to 6 CTAs/SM (96 KB in flight); still enough at 2048 tokens on the bench"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```