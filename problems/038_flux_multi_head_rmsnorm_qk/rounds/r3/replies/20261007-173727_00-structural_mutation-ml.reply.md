### Rationale
Persistent and ring designs (TMA, cp.async, wstat) have all lost to the one-shot parent on B200, at the same or higher bytes in flight. So the parent's remaining losses are probably CTA turnover and the cold-DRAM ramp and tail that every CTA's own load exposes, not loop structure.

This candidate keeps the parent's one-shot schedule. Each CTA still owns 8 rows, uses 4 warps and ldg128 with evict hints, and the hardware still balances CTAs dynamically. The one structural addition: thread 0 of every CTA issues a single `cp.async.bulk.prefetch.L2` (4 KB) for the tile two resident waves ahead.

DRAM concurrency then no longer depends on how many CTAs are resident. About 10 MB of reads stay queued at all times, and most demand loads become L2 hits. That should cut both the steady-state turnover loss and the M-band tail.

When the grid fits in one or two waves (most of band S), no prefetch target exists. The kernel then behaves like the parent, ported to CUDA C++.

```json solution-spec
{"name": "r3-os-l2pf-r8w4", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "One-shot ROWS=8 ldg128 kernel (parent g2-os-r8w4 ported to CUDA C++) plus one cp.async.bulk.prefetch.L2 per CTA for the tile two resident waves ahead, decoupling DRAM concurrency from CTA residency on M/L inputs.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// SOL-ExecBench #38: per-head RMSNorm of Q and K, fp32, [B, S, 48, 128].
// One-shot fused kernel: one CTA (4 warps) per 8-row tile; Q tiles first, then K tiles.
// Each warp handles 2 rows, each lane one float4 per row; warp-shuffle reduction.
// Addition over the parent: thread 0 of every CTA issues one 4 KB L2 bulk prefetch for the
// tile `lookahead` CTAs ahead, so DRAM reads run ahead of CTA residency (sm_90+ only).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int D = 128;
constexpr int H = 48;
constexpr int ROWS = 8;                       // divides 48: no masks, fixed head pattern
constexpr int THREADS = 128;                  // 4 warps x 2 rows
constexpr int TILE_BYTES = ROWS * D * 4;      // 4096 B, one contiguous chunk
constexpr int PF_WAVES = 2;                   // prefetch distance in resident waves

__device__ __forceinline__ uint64_t policy_evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ float4 ld_stream(const float* p, uint64_t pol) {
    float4 v;
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void st_stream(float* p, float4 v, uint64_t pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol)
                 : "memory");
}

__global__ void __launch_bounds__(THREADS, 16)
qk_rms_pf_kernel(const float* __restrict__ q, const float* __restrict__ k,
                 const float* __restrict__ wq, const float* __restrict__ wk,
                 float* __restrict__ qo, float* __restrict__ ko,
                 int n_tiles, int lookahead, float eps)
{
    const int pid = blockIdx.x;
    const bool is_k = pid >= n_tiles;
    const int tile = is_k ? pid - n_tiles : pid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r0 = tile * ROWS + warp * 2;            // even; rows r0, r0+1
    const int h0 = r0 % H;                            // even, so h0+1 <= 47

    const uint64_t pol = policy_evict_first();
    const size_t off = (size_t)r0 * D + (size_t)lane * 4;

    // Demand loads first (critical path).
    const float4 a = ld_stream(x + off, pol);
    const float4 b = ld_stream(x + off + D, pol);
    const float4 wa = __ldg(reinterpret_cast<const float4*>(w + h0 * D + lane * 4));
    const float4 wb = __ldg(reinterpret_cast<const float4*>(w + (h0 + 1) * D + lane * 4));

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // One fire-and-forget L2 prefetch per CTA for a tile a later CTA will consume.
    if (threadIdx.x == 0 && lookahead > 0) {
        const int p = pid + lookahead;
        if (p < 2 * n_tiles) {
            const bool pk = p >= n_tiles;
            const int pt = pk ? p - n_tiles : p;
            const float* src = (pk ? k : q) + (size_t)pt * ROWS * D;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                         :: "l"(src), "r"(TILE_BYTES) : "memory");
        }
    }
#endif

    float sa = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    float sb = b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sa += __shfl_xor_sync(0xffffffffu, sa, o);
        sb += __shfl_xor_sync(0xffffffffu, sb, o);
    }
    const float ia = rsqrtf(sa * (1.0f / D) + eps);
    const float ib = rsqrtf(sb * (1.0f / D) + eps);

    float4 ya, yb;
    ya.x = (a.x * ia) * wa.x; ya.y = (a.y * ia) * wa.y; ya.z = (a.z * ia) * wa.z; ya.w = (a.w * ia) * wa.w;
    yb.x = (b.x * ib) * wb.x; yb.y = (b.y * ib) * wb.y; yb.z = (b.z * ib) * wb.z; yb.w = (b.w * ib) * wb.w;

    st_stream(y + off, ya, pol);
    st_stream(y + off + D, yb, pol);
}

// Device-only facts (no tensors): SM count and resident CTAs of this kernel.
int g_sms = 0;
int g_occ = 0;

}  // namespace

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm)
{
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && query_norm.is_cuda() && key_norm.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
                weight_q.scalar_type() == torch::kFloat32 && weight_k.scalar_type() == torch::kFloat32 &&
                query_norm.scalar_type() == torch::kFloat32 && key_norm.scalar_type() == torch::kFloat32,
                "fp32 required");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && query_norm.is_contiguous() &&
                key_norm.is_contiguous() && weight_q.is_contiguous() && weight_k.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(query.numel() == key.numel() && query.numel() == query_norm.numel() &&
                key.numel() == key_norm.numel(), "shape mismatch");
    TORCH_CHECK(query.numel() % (H * D) == 0, "last dims must be [48, 128]");
    TORCH_CHECK(weight_q.numel() == H * D && weight_k.numel() == H * D, "weights must be [48, 128]");

    const int64_t n_rows = query.numel() / D;
    const int n_tiles = (int)(n_rows / ROWS);
    const int total = 2 * n_tiles;
    if (total == 0) return;

    if (g_sms == 0) {
        g_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, qk_rms_pf_kernel, THREADS, 0);
        g_occ = occ > 0 ? occ : 1;
    }
    const int resident = g_sms * g_occ;
    const int lookahead = (total > resident) ? PF_WAVES * resident : 0;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qk_rms_pf_kernel<<<total, THREADS, 0, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(),
        weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
        query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
        n_tiles, lookahead, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "flux multi-head RMSNorm QK (one-shot + L2 bulk prefetch)");
}
```

```yaml design-card
id: r3-os-l2pf-r8w4
parents:
  - g2-os-r8w4
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: fused
  tile: rows8
  red: warp
  cache: stream
  spec: L
hypothesis: >-
  The one-shot parent's in-flight reads are tied to resident CTAs. CTA turnover and each CTA's own cold-DRAM wait
  therefore cost bandwidth in steady state and in the M-band tail. One cp.async.bulk.prefetch.L2 per CTA, two
  resident waves ahead, keeps about 10 MB of reads queued at DRAM regardless of residency. Demand loads become L2
  hits. Dynamic CTA scheduling is kept, unlike the persistent designs that all lost on B200. H200 cannot show the
  B200-specific size of this effect (36 B/clk/SM at 1.5 GHz, higher loaded latency).
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: -6
    confidence: low
  L:
    pct: -4
    confidence: low
resources_sm100a:
  regs_per_thread: 30
  smem_per_cta_bytes: 0
  threads_per_cta: 128
  ctas_per_sm: 16
  bytes_in_flight_per_sm: 131072
  launches_per_call: 1
knobs:
  ROWS: 8
  num_warps: 4
  PF_WAVES: 2
  PF_BYTES: 4096
runs_on:
  H200:
    runs: true
    representative: true
    note: "has cp.async.bulk.prefetch.L2; closest proxy for the prefetch effect"
  A100:
    runs: true
    representative: false
    note: "prefetch compiled out below sm_90; measures only the plain CUDA port"
  L40S:
    runs: true
    representative: false
    note: "prefetch compiled out below sm_90; correctness smoke test"
risks:
  - "L2 bulk prefetches may be throttled or dropped under load, giving no gain (the cost is only 1 instruction per CTA)"
  - "If DRAM efficiency (read/write mix, write-backs of the flush's dirty L2 lines) is the true limit at about 82%, more queued reads cannot help"
  - "Prefetched lines (about 10 MB in flight) could be evicted before use; then only extra request overhead, no extra DRAM bytes, since misses merge"
  - "Assumes CTAs are dispatched in roughly increasing blockIdx order; if not, part of the prefetch lead is lost"
  - "The CUDA C++ port may differ from the Triton parent's codegen (registers, scheduling); compare the A100 timing against the parent to isolate the port effect"
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```