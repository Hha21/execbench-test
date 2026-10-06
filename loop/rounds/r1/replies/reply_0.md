### Rationale

v039 has no prefetch. Each persistent program loads a tile, reduces it, stores it, and only then issues its next loads. Its 8 × 8 KB of nominal in-flight data is therefore only partly outstanding at any moment, and that is probably why it stops at 5.77 TB/s.

This candidate separates load issue from compute:

- **Producer.** One CTA per SM, with one producer thread, keeps a 16-stage ring of 8 KB 1-D bulk copies (`cp.async.bulk`, L2 evict-first). That holds about 128 KB of reads outstanding per SM, about 3× the Little's-law need, using one instruction per 8 KB.
- **Consumers.** Sixteen consumer warps each take one row per tile. A warp releases its stage right after its reduction, then stores directly (`st.global.cs`).
- **Weights.** Both weight matrices (48 KB) are bulk-loaded into shared memory once per CTA.

Other properties:

- 16-row tiles divide 48, so there are no partial tiles on any shape.
- On the S band all of a CTA's 2–6 tiles are issued up front.
- Sub-sm_90 GPUs fall back to a plain per-warp loop, for correctness only.

```json solution-spec
{"name": "r1-bulk1d-ring-r16s16", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "Persistent 1-CTA/SM kernel: a producer thread keeps a 16-stage ring of 8 KB cp.async.bulk loads (128 KB/SM in flight); 16 consumer warps normalise one row each against smem-resident weights and store with st.global.cs.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// SOL-ExecBench L1/038 flux_multi_head_rmsnorm_qk: per-head RMSNorm of Q and K, fp32, [B,S,48,128].
//
// sm_90+/sm_100a path: persistent grid, one CTA per SM.
//   - warp NWC (producer, lane 0) bulk-loads both weight matrices (48 KB) into smem once, then keeps a
//     STAGES-deep ring of TR-row tiles (TR*512 B, contiguous) in flight with cp.async.bulk + mbarrier tx counts.
//   - warps 0..NWC-1 (consumers) each handle one row of every tile: LDS.128, warp-shuffle sum of squares,
//     release the stage (empty mbarrier), rsqrt, multiply by the smem weight row, st.global.cs.
// Older GPUs (sm_80/sm_89 local testing) run a plain warp-per-row fallback: correct but not representative.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int D = 128;
constexpr int H = 48;
constexpr int TR = 16;                       // rows per tile; divides 48, so no partial tiles
constexpr int NWC = TR;                      // consumer warps, one row each
constexpr int STAGES = 16;                   // ring depth: 16 x 8 KB = 128 KB of reads in flight per SM
constexpr int THREADS = (NWC + 1) * 32;      // + 1 producer warp
constexpr int TILE_FLOATS = TR * D;
constexpr int TILE_BYTES = TILE_FLOATS * 4;  // 8192
constexpr int WMAT_FLOATS = H * D;           // one weight matrix
constexpr int WMAT_BYTES = WMAT_FLOATS * 4;  // 24576
constexpr int W_BYTES = 2 * WMAT_BYTES;      // 49152
constexpr int RING_BYTES = STAGES * TILE_BYTES;
constexpr int BAR_BYTES = (2 * STAGES + 1) * 8;
constexpr int SMEM_BYTES = W_BYTES + RING_BYTES + BAR_BYTES;  // 180,488 B

static_assert(H % TR == 0, "TR must divide 48");
static_assert(NWC == TR, "one consumer warp per tile row");
static_assert(SMEM_BYTES <= 227 * 1024, "smem budget");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t done;
    do {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t}"
            : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    } while (!done);
}
__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar,
                                         uint64_t policy) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(policy) : "memory");
}
#endif

__global__ void __launch_bounds__(THREADS, 1)
qk_rms_bulk(const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ wq, const float* __restrict__ wk,
            float* __restrict__ qo, float* __restrict__ ko,
            int n_tiles, float eps)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) unsigned char smem[];
    float* ws = reinterpret_cast<float*>(smem);                       // [2][48][128]: Q weights, then K weights
    float* ring = reinterpret_cast<float*>(smem + W_BYTES);           // [STAGES][TR][128]
    const uint32_t bar0 = smem_u32(smem + W_BYTES + RING_BYTES);      // full[s] = bar0 + 8s
    const uint32_t empty0 = bar0 + 8 * STAGES;                        // empty[s] = empty0 + 8s
    const uint32_t wbar = bar0 + 16 * STAGES;

    const int total = 2 * n_tiles;           // Q tiles [0, n_tiles), K tiles [n_tiles, 2 n_tiles)
    const int G = gridDim.x;
    const int b = blockIdx.x;
    const int n_my = (b < total) ? (total - 1 - b) / G + 1 : 0;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(bar0 + 8 * s, 1);
            mbar_init(empty0 + 8 * s, NWC);
        }
        mbar_init(wbar, 1);
        fence_mbar_init();
    }
    __syncthreads();

    if (warp == NWC) {                        // producer warp
        if (lane == 0) {
            const uint64_t pol_stream = policy_evict_first();
            const uint64_t pol_keep = policy_evict_last();
            mbar_expect_tx(wbar, W_BYTES);
            bulk_g2s(smem_u32(ws), wq, WMAT_BYTES, wbar, pol_keep);
            bulk_g2s(smem_u32(ws + WMAT_FLOATS), wk, WMAT_BYTES, wbar, pol_keep);
            const uint32_t ring_s = smem_u32(ring);
            for (int i = 0; i < n_my; ++i) {
                const int s = i % STAGES;
                if (i >= STAGES) mbar_wait(empty0 + 8 * s, ((i / STAGES) - 1) & 1);
                const int t = b + i * G;
                const float* src = (t < n_tiles) ? q + (size_t)t * TILE_FLOATS
                                                 : k + (size_t)(t - n_tiles) * TILE_FLOATS;
                const uint32_t full = bar0 + 8 * s;
                mbar_expect_tx(full, TILE_BYTES);
                bulk_g2s(ring_s + s * TILE_BYTES, src, TILE_BYTES, full, pol_stream);
            }
        }
        return;
    }

    // consumer warps
    mbar_wait(wbar, 0);
    const float4* ws4 = reinterpret_cast<const float4*>(ws);
    const float4* ring4 = reinterpret_cast<const float4*>(ring);
    for (int i = 0; i < n_my; ++i) {
        const int s = i % STAGES;
        const int t = b + i * G;
        const bool is_k = t >= n_tiles;
        const int tile = is_k ? t - n_tiles : t;
        mbar_wait(bar0 + 8 * s, (i / STAGES) & 1);
        const float4 x = ring4[(s * TILE_FLOATS + warp * D) / 4 + lane];
        float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        // every lane's x has been consumed by the shuffles: release the stage early
        if (lane == 0) mbar_arrive(empty0 + 8 * s);
        const int h = (tile % (H / TR)) * TR + warp;
        const float4 w = ws4[((is_k ? H : 0) + h) * (D / 4) + lane];
        const float inv = rsqrtf(ss * (1.0f / D) + eps);
        float4 y;
        y.x = (x.x * inv) * w.x;
        y.y = (x.y * inv) * w.y;
        y.z = (x.z * inv) * w.z;
        y.w = (x.w * inv) * w.w;
        float* dst = (is_k ? ko : qo) + (size_t)tile * TILE_FLOATS + warp * D;
        __stcs(reinterpret_cast<float4*>(dst) + lane, y);
    }
#else
    // Fallback for GPUs without cp.async.bulk (local A100/L40S correctness runs): warp per row, grid-stride.
    const int wpb = blockDim.x >> 5;
    const long long n_rows = (long long)n_tiles * TR;     // rows per stream
    const long long total_rows = 2 * n_rows;
    const long long nw = (long long)gridDim.x * wpb;
    for (long long r = (long long)blockIdx.x * wpb + warp; r < total_rows; r += nw) {
        const bool is_k = r >= n_rows;
        const long long row = is_k ? r - n_rows : r;
        const float4* xp = reinterpret_cast<const float4*>((is_k ? k : q) + row * D);
        const float4* wp = reinterpret_cast<const float4*>((is_k ? wk : wq) + (row % H) * D);
        const float4 x = __ldg(xp + lane);
        const float4 w = __ldg(wp + lane);
        float ss = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        const float inv = rsqrtf(ss * (1.0f / D) + eps);
        float4 y;
        y.x = (x.x * inv) * w.x;
        y.y = (x.y * inv) * w.y;
        y.z = (x.z * inv) * w.z;
        y.w = (x.w * inv) * w.w;
        __stcs(reinterpret_cast<float4*>((is_k ? ko : qo) + row * D) + lane, y);
    }
#endif
}

int g_sms = -1;
int g_major = 0;
bool g_attr_set = false;

void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, name, " must be 16-byte aligned");
}

}  // namespace

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k, double eps,
         torch::Tensor query_norm, torch::Tensor key_norm)
{
    check_tensor(query, "query");
    check_tensor(key, "key");
    check_tensor(weight_q, "weight_q");
    check_tensor(weight_k, "weight_k");
    check_tensor(query_norm, "query_norm");
    check_tensor(key_norm, "key_norm");
    TORCH_CHECK(weight_q.numel() == H * D && weight_k.numel() == H * D, "weights must be [48,128]");
    TORCH_CHECK(key.numel() == query.numel() && query_norm.numel() == query.numel() &&
                key_norm.numel() == query.numel(), "shape mismatch");
    const int64_t n_rows = query.numel() / D;
    TORCH_CHECK(n_rows % H == 0, "rows must be a multiple of 48");
    const int64_t n_tiles64 = n_rows / TR;
    TORCH_CHECK(n_tiles64 * 2 < (int64_t)1 << 31, "too many tiles");
    const int n_tiles = (int)n_tiles64;
    if (n_tiles == 0) return;

    if (g_sms < 0) {
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&g_major, cudaDevAttrComputeCapabilityMajor, dev));
    }
    const bool bulk = g_major >= 9;
    if (bulk && !g_attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(qk_rms_bulk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        g_attr_set = true;
    }
    const int total = 2 * n_tiles;
    int grid = bulk ? g_sms : g_sms * 4;
    if (grid > total) grid = total;
    const size_t smem = bulk ? (size_t)SMEM_BYTES : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    qk_rms_bulk<<<grid, THREADS, smem, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(),
        weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
        query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
        n_tiles, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Per-head RMSNorm of Q and K (bulk-copy ring, DPS)");
}
```

```yaml design-card
id: r1-bulk1d-ring-r16s16
parents:
  - v039
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: persistent
  launch: fused
  tile: rows16
  red: warp
  cache: stream
  spec: L
hypothesis: >-
  v039 serialises load, reduce and store inside each persistent program and never prefetches, so its outstanding
  reads per SM fall well below the ~43-64 KB B200 needs. A dedicated producer thread keeps 16 x 8 KB cp.async.bulk
  loads in flight per SM. Consumer warps release stages right after their reduction. Weights sit in shared memory.
  Read issue is therefore decoupled from compute latency at one instruction per 8 KB, which should lift
  sustained bandwidth from 5.8 to about 7 TB/s.
expected_effect:
  S:
    pct: -5
    confidence: low
  M:
    pct: -12
    confidence: low
  L:
    pct: -17
    confidence: medium
resources_sm100a:
  regs_per_thread: 32
  smem_per_cta_bytes: 180488
  threads_per_cta: 544
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 131072
  launches_per_call: 1
knobs:
  TR: 16
  STAGES: 16
  consumer_warps: 16
  producer_warps: 1
  grid: num_SMs
runs_on:
  H200:
    runs: true
    representative: true
    note: "same bulk/mbarrier path (sm_90); H200 needs less in flight, so expect it to show correctness and a moderate gain, not the full B200 gain"
  A100:
    runs: true
    representative: false
    note: "runs the plain warp-per-row fallback (no cp.async.bulk on sm_80); correctness only"
  L40S:
    runs: true
    representative: false
    note: "fallback path; correctness only"
risks:
  - "One CTA per SM with 16 consumer warps: if per-tile consumer latency (wait, LDS, 5 shuffles, rsqrt, STG) exceeds about 520 clk at 1.5 GHz, the consumers become the bottleneck; then try 2 rows per warp with TR=32 or 8 KB tiles over 2 CTAs/SM."
  - "Direct stores from only 16 warps per SM may cap write throughput; st:bulk via smem staging is the follow-up axis."
  - "A single producer thread issues all bulk copies; issue rate is about 1 per 520 clk needed, so it should be fine, but it is unmeasured on B200."
  - "Prologue (mbarrier init and 48 KB weight bulk load, an L2 hit after the first CTA) adds to S-band fixed cost; S-band loads are all issued up front, which may offset it."
  - "180 KB of dynamic smem needs the cudaFuncSetAttribute opt-in (done once on the host); it compiles for sm_100a only via the inline-PTX paths guarded by __CUDA_ARCH__ >= 900."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```