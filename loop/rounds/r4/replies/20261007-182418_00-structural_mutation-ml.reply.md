### Rationale
The parent is a one-shot kernel. Each thread has one 32 B load outstanding, so the SM is capped at 64 KB in flight, and that cap drops every time a CTA finishes and the next one has not yet issued its loads. Adding more 256-bit loads per thread in a one-shot grid would cost registers and occupancy.

This candidate fills the empty niche **ldg256/persistent-wstat/all**:

- **Grid sizing.** It launches exactly one balanced wave (6 CTAs per SM, so 888 CTAs on B200), sized from the sm_100a occupancy query.
- **Stationary weights.** Each CTA owns one (stream, 16-head group) class, so its weights are loaded into registers once.
- **Register prefetch ring.** Each thread keeps two 256-bit input tiles in flight (depth P=2) and refills a slot as soon as it has stored that slot's output.

That gives about 96 KB of loads outstanding per SM continuously, with no CTA-turnover gaps and no shared-memory round trip.

Earlier persistent designs on B200 lost for different reasons: serial latency (v039), Triton's cp.async/TMA overheads, or unbalanced tails. Tile counts per CTA here differ by at most one, and the tail loss is ≤1.2%.

```json solution-spec
{"name": "r4-ldg256-wstat-pf2", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop",
 "description": "Persistent weight-stationary 256-bit kernel: one balanced wave, register prefetch depth 2, about 96 KB of loads in flight per SM with no CTA-turnover gaps.",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// r4-ldg256-wstat-pf2: SOL-ExecBench #38 (per-head RMSNorm of Q and K, fp32), CUDA C++.
//
// Each tensor is viewed as [n_tok, 48, 128]. A "tile" is 16 consecutive heads (16 rows x 128 floats = 8 KB) of one
// token. There are 6 tile classes: (stream Q/K) x (head group 0..2). CTA b handles class b % 6, and the tokens
// j, j+nj, j+2nj, ... with j = b / 6. Within a class every tile uses the same weight rows, so each thread loads its
// 8 weights once (weight-stationary).
//
// Thread layout: 256 threads = 16 rows x 16 lanes. A half-warp covers one 128-float row with one 256-bit access
// per lane. The reduction is a half-warp butterfly (offsets 8, 4, 2, 1).
//
// Pipeline: a register ring of P=2 tiles per thread. After a slot's tile is normalised and stored, the slot is
// refilled with the tile P steps ahead. This keeps about 2 x 32 B outstanding per thread for the whole kernel.
//
// Grid: one wave. CTAs per class = ceil(n_tok / T), where T = ceil(n_tok / cap) and cap = SMs * occupancy / 6.
// So per-CTA tile counts differ by at most one. The grid depends only on the shape and the device.
// All maths is fp32. The kernel is launched on the current PyTorch stream.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

namespace {

constexpr int D = 128;
constexpr int H = 48;
constexpr int TOK = H * D;        // floats per token per tensor
constexpr int THREADS = 256;
constexpr int ROWS = 16;          // heads per tile (divides 48)
constexpr int GROUPS = H / ROWS;  // 3
constexpr int CLASSES = 2 * GROUPS;
constexpr int P = 2;              // register prefetch depth
constexpr int MIN_BLOCKS = 6;     // caps registers at 40 -> 6 x 256 threads per SM

__device__ __forceinline__ void ld8(float (&v)[8], const float* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("ld.global.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
                 : "l"(p));
#else
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#endif
}

__device__ __forceinline__ void st8(float* p, const float (&v)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7])
                 : "memory");
#else
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
#endif
}

__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
qk_rms_wstat_ldg256(const float* __restrict__ q, const float* __restrict__ k,
                    const float* __restrict__ wq, const float* __restrict__ wk,
                    float* __restrict__ qo, float* __restrict__ ko,
                    int n_tok, int nj, float eps)
{
    const int cls = blockIdx.x % CLASSES;
    const int j = blockIdx.x / CLASSES;          // j < nj <= n_tok
    const bool is_k = cls >= GROUPS;
    const int g = is_k ? cls - GROUPS : cls;
    const float* X = is_k ? k : q;
    const float* W = is_k ? wk : wq;
    float* Y = is_k ? ko : qo;

    const int t = threadIdx.x;
    const int head = g * ROWS + (t >> 4);
    const int in_tok = head * D + (t & 15) * 8;  // 32-B aligned offset inside a token

    float w[8];
    ld8(w, W + in_tok);                          // stationary for the whole kernel

    const float* xp = X + (size_t)j * TOK + in_tok;
    float* yp = Y + (size_t)j * TOK + in_tok;
    const size_t step = (size_t)nj * TOK;
    const int ntiles = (n_tok - j + nj - 1) / nj;   // >= 1, uniform per CTA

    float b[P][8];
#pragma unroll
    for (int p = 0; p < P; ++p)
        if (p < ntiles) ld8(b[p], xp + (size_t)p * step);   // all first loads issued up front

    for (int base = 0; base < ntiles; base += P) {
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const int i = base + p;
            if (i < ntiles) {
                float ss = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) ss = fmaf(b[p][e], b[p][e], ss);
                ss += __shfl_xor_sync(0xffffffffu, ss, 8);
                ss += __shfl_xor_sync(0xffffffffu, ss, 4);
                ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                ss += __shfl_xor_sync(0xffffffffu, ss, 1);
                const float r = rsqrtf(ss * (1.0f / 128.0f) + eps);   // * (1/128) is exact
#pragma unroll
                for (int e = 0; e < 8; ++e) b[p][e] = (b[p][e] * r) * w[e];
                st8(yp + (size_t)i * step, b[p]);
                if (i + P < ntiles) ld8(b[p], xp + (size_t)(i + P) * step);   // refill this slot
            }
        }
    }
}

int g_cap = 0;   // CTAs per class that fit in one wave (device/compile property, not data)

int wave_cap() {
    if (g_cap == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, qk_rms_wstat_ldg256, THREADS, 0);
        if (occ < 1) occ = 1;
        g_cap = std::max(1, (sms * occ) / CLASSES);
    }
    return g_cap;
}

}  // namespace

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm)
{
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && query_norm.is_cuda() && key_norm.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
                weight_q.scalar_type() == torch::kFloat32 && weight_k.scalar_type() == torch::kFloat32 &&
                query_norm.scalar_type() == torch::kFloat32 && key_norm.scalar_type() == torch::kFloat32,
                "fp32 required");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && weight_q.is_contiguous() && weight_k.is_contiguous() &&
                query_norm.is_contiguous() && key_norm.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(query.numel() % TOK == 0 && key.numel() == query.numel(), "shape must be [B,S,48,128]");

    const int64_t n_tok = query.numel() / TOK;
    if (n_tok == 0) return;

    const int64_t cap = wave_cap();
    const int64_t T = (n_tok + cap - 1) / cap;          // tiles per CTA (max)
    const int nj = (int)((n_tok + T - 1) / T);          // CTAs per class, <= cap
    const dim3 grid((unsigned)(CLASSES * nj));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    qk_rms_wstat_ldg256<<<grid, THREADS, 0, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(),
        weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
        query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
        (int)n_tok, nj, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "flux multi-head RMSNorm QK (fp32, DPS)");
}
```

```yaml design-card
id: r4-ldg256-wstat-pf2
parents:
  - r3-cute-ldg256-os-r16
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: persistent-wstat
  launch: fused
  tile: rows16
  red: halfwarp
  cache: default
  spec: all
hypothesis: >-
  The one-shot parent caps each SM at about 64 KB in flight (one 32 B load per thread), and that cap falls further
  while CTAs retire and new ones ramp up. At 36 B/clk/SM and roughly 1-2 us loaded latency on B200, this limits it
  to about 6.7 TB/s. A single balanced wave of weight-stationary CTAs fixes this. Each CTA keeps a depth-2 register
  ring of 256-bit loads, refilled right after each store, which sustains about 96 KB per SM with no turnover gaps and
  no shared-memory round trip. It should move M/L toward 7.1-7.3 TB/s. For n_tok <= 148 it degenerates to a one-shot
  grid with the same CTA count as the parent, so S should be unchanged.
expected_effect:
  S:
    pct: 1
    confidence: low
  M:
    pct: -4
    confidence: low
  L:
    pct: -6
    confidence: low
resources_sm100a:
  regs_per_thread: 40
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 6
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  P: 2
  ROWS: 16
  THREADS: 256
  MIN_BLOCKS: 6
  CLASSES: 6
  grid_rule: "nj = ceil(n_tok / ceil(n_tok / cap)), cap = SMs * occupancy / 6 (148 on B200 at 6 CTAs/SM); grid = 6 * nj"
dispatch: []
runs_on:
  H200:
    runs: true
    representative: false
    note: "sm_90 path uses 2 x float4 instead of 256-bit; useful as a check of the persistence/prefetch structure and for correctness"
  A100:
    runs: true
    representative: false
    note: "128-bit fallback; correctness and gross-regression check only"
  L40S:
    runs: true
    representative: false
risks:
  - "Earlier persistent designs (v039, g2-ws-*, g2-tma-*) were slower than one-shot on B200. If the cause was not serial latency or tail imbalance but something about the strided per-class token order (DRAM page or partition locality), this design inherits the problem."
  - "If ptxas needs more than 40 registers, __launch_bounds__(256,6) may spill (check LOCAL=0 in the sm_100a compile). The occupancy query keeps the grid in one wave either way."
  - "With only P=2 tiles outstanding, the depth may still be short of the latency-bandwidth product. P=3 at 5 CTAs/SM (about 120 KB) is the next knob, but it breaks the 148-per-class balance."
  - "Inline-PTX v8.f32 loads and stores need PTX ISA >= 8.8 (CUDA 13.1 on the portal). Older archs compile the float4 fallback under the __CUDA_ARCH__ guard."
  - "The static occupancy cap is computed once per process from the device and the compiled kernel only (no data); it assumes a single GPU."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```