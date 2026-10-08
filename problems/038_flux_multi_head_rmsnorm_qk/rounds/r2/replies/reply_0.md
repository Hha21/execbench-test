### Rationale

This ports the one-shot fused design of `g2-os-r16w8` to CUDA C++. The tile, grid, resident CTAs per SM and bytes in flight per SM all stay the same. One thing changes: each row of 128 fp32 values is handled by a half-warp, and each lane does one 256-bit `ld.global.v8.f32` and one `st.global.v8.f32`.

Per byte, this halves the load and store instructions, the address arithmetic and the LSU request count. The parent spent 4 LDG + 2 STG per thread; this kernel spends 2 LDG + 1 STG. At B200's 36 B per clock per SM at 1.5 GHz, issue overhead is the likeliest limit keeping the parent at 6.4 TB/s on the L band.

Cache hints:
- Streamed inputs use `L1::no_allocate` with an L2 `evict_first` policy.
- Weights use an `evict_last` policy.
- Outputs use L2 `evict_first`.

`__launch_bounds__(256, 8)` pins the kernel at ≤32 registers, so 8 CTAs fit per SM. Below sm_100 the same kernel compiles to a float4 path that is correct but not representative of B200 timing.

```json solution-spec
{"name": "r2-ldg256-os-r16", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "CUDA C++ port of g2-os-r16w8: one-shot fused Q+K launch, 16 rows per 256-thread CTA, half-warp per row with 256-bit ld/st.global.v8.f32 and L2 evict_first/evict_last cache policies on sm_100; float4 fallback below sm_100.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.cu::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// SOL-ExecBench #38 flux_multi_head_rmsnorm_qk, per-head RMSNorm of Q and K, fp32, [B, S, 48, 128].
//
// One-shot, fused launch (port of Triton g2-os-r16w8):
//   grid = 2 * n_tiles CTAs; CTA b < n_tiles handles Q tile b, otherwise K tile (b - n_tiles).
//   Each CTA owns ROWS = 16 consecutive rows (16 divides 48, so no masks and a fixed head block).
//   256 threads: each half-warp (16 lanes x 8 floats) covers one 128-float row.
//   Reduction: 8 in-thread FMAs, then __shfl_xor 8/4/2/1 inside the half-warp.
//
// sm_100+: 256-bit global accesses (PTX ISA 8.8: ld/st.global.v8.f32 -> LDG/STG.E.*.256)
//          streamed x: L1::no_allocate + L2 evict_first policy; weights: L2 evict_last policy;
//          stores: L2 evict_first policy.
// < sm_100 (A100/H200/L40S local testing): float4 path with __ldcs/__ldg/__stcs. Correct, but its timing does not
//          represent the 256-bit path.
//
// All maths fp32; rsqrtf -> rsqrt.approx.f32 (rel err <= 2^-22.9); 1/128 is an exact multiply.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

// If ptxas ever rejects the L2::cache_hint form on .v8, set this to 0. The kernel then uses the
// compile-verified plain forms (ld.global.L1::no_allocate.v8.f32 / st.global.v8.f32).
#ifndef QK_USE_L2_HINT
#define QK_USE_L2_HINT 1
#endif

namespace {

constexpr int D = 128;
constexpr int H = 48;
constexpr int ROWS = 16;              // rows per CTA; divides 48
constexpr int THREADS = ROWS * 16;    // 16 lanes per row, 8 floats per lane
constexpr int GROUPS = H / ROWS;      // head blocks per token
static_assert(H % ROWS == 0, "ROWS must divide 48");
static_assert(THREADS == 256, "launch bounds assume 256 threads");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define QK_V8 1
#else
#define QK_V8 0
#endif

#if QK_V8
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

// Streamed input: no L1 allocation, L2 evict_first.
__device__ __forceinline__ void ld8_stream(const float* p, float (&v)[8], uint64_t pol) {
#if QK_USE_L2_HINT
    asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
                 : "l"(p), "l"(pol));
#else
    (void)pol;
    asm volatile("ld.global.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
                 : "l"(p));
#endif
}

// Reused weights: normal L1 allocation (other CTAs on the SM reuse them), L2 evict_last.
__device__ __forceinline__ void ld8_keep(const float* p, float (&v)[8], uint64_t pol) {
#if QK_USE_L2_HINT
    asm volatile("ld.global.L2::cache_hint.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
                 : "l"(p), "l"(pol));
#else
    (void)pol;
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
                 : "l"(p));
#endif
}

__device__ __forceinline__ void st8_stream(float* p, const float (&v)[8], uint64_t pol) {
#if QK_USE_L2_HINT
    asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]), "l"(pol)
                 : "memory");
#else
    (void)pol;
    asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7])
                 : "memory");
#endif
}
#endif  // QK_V8

__global__ void __launch_bounds__(THREADS, 8)
qk_rms_ldg256_kernel(const float* __restrict__ q, const float* __restrict__ k,
                     const float* __restrict__ wq, const float* __restrict__ wk,
                     float* __restrict__ qo, float* __restrict__ ko,
                     int n_tiles, float eps) {
    const int bid = blockIdx.x;
    const bool is_k = bid >= n_tiles;
    const int tile = is_k ? bid - n_tiles : bid;
    const float* __restrict__ x = is_k ? k : q;
    const float* __restrict__ w = is_k ? wk : wq;
    float* __restrict__ y = is_k ? ko : qo;

    const int r = threadIdx.x >> 4;           // row within the tile (0..15)
    const int c = (threadIdx.x & 15) * 8;     // first column of this lane
    const size_t row = (size_t)tile * ROWS + r;
    const int head = (tile % GROUPS) * ROWS + r;   // == row % 48 because ROWS divides 48

    const float* xp = x + row * D + c;
    const float* wp = w + (size_t)head * D + c;
    float* yp = y + row * D + c;

    float xv[8], wv[8], yv[8];

#if QK_V8
    const uint64_t pol_first = policy_evict_first();
    const uint64_t pol_last = policy_evict_last();
    ld8_stream(xp, xv, pol_first);   // both loads issued before any use
    ld8_keep(wp, wv, pol_last);
#else
    {
        const float4 a = __ldcs(reinterpret_cast<const float4*>(xp));
        const float4 b = __ldcs(reinterpret_cast<const float4*>(xp) + 1);
        const float4 wa = __ldg(reinterpret_cast<const float4*>(wp));
        const float4 wb = __ldg(reinterpret_cast<const float4*>(wp) + 1);
        xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
        xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
        wv[0] = wa.x; wv[1] = wa.y; wv[2] = wa.z; wv[3] = wa.w;
        wv[4] = wb.x; wv[5] = wb.y; wv[6] = wb.z; wv[7] = wb.w;
    }
#endif

    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = fmaf(xv[i], xv[i], s);
    // Reduce inside each half-warp (lanes 0-15: one row, 16-31: the next row). All 32 lanes are active.
    s += __shfl_xor_sync(0xffffffffu, s, 8);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);

    const float inv = rsqrtf(s * (1.0f / (float)D) + eps);

#pragma unroll
    for (int i = 0; i < 8; ++i) yv[i] = (xv[i] * inv) * wv[i];

#if QK_V8
    st8_stream(yp, yv, pol_first);
#else
    __stcs(reinterpret_cast<float4*>(yp), make_float4(yv[0], yv[1], yv[2], yv[3]));
    __stcs(reinterpret_cast<float4*>(yp) + 1, make_float4(yv[4], yv[5], yv[6], yv[7]));
#endif
}

void check_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    // 256-bit accesses need 32-B alignment; the harness gives 256-B aligned buffers and rows are 512 B.
    TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 31) == 0, name, " must be 32-byte aligned");
}

}  // namespace

void run(torch::Tensor query, torch::Tensor key, torch::Tensor weight_q, torch::Tensor weight_k,
         double eps, torch::Tensor query_norm, torch::Tensor key_norm) {
    check_tensor(query, "query");
    check_tensor(key, "key");
    check_tensor(weight_q, "weight_q");
    check_tensor(weight_k, "weight_k");
    check_tensor(query_norm, "query_norm");
    check_tensor(key_norm, "key_norm");
    TORCH_CHECK(query.numel() == key.numel() && query.numel() == query_norm.numel() &&
                key.numel() == key_norm.numel(), "Q/K/output sizes must match");
    TORCH_CHECK(weight_q.numel() == H * D && weight_k.numel() == H * D, "weights must be [48,128]");
    TORCH_CHECK(query.size(-1) == D && query.numel() % ((int64_t)H * D) == 0, "last dims must be [48,128]");

    const int64_t n_rows = query.numel() / D;
    if (n_rows == 0) return;
    const int64_t n_tiles = n_rows / ROWS;   // exact: n_rows is a multiple of 48
    TORCH_CHECK(2 * n_tiles <= (int64_t)0x7fffffff, "grid too large");

    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)(2 * n_tiles));
    qk_rms_ldg256_kernel<<<grid, THREADS, 0, stream>>>(
        query.data_ptr<float>(), key.data_ptr<float>(),
        weight_q.data_ptr<float>(), weight_k.data_ptr<float>(),
        query_norm.data_ptr<float>(), key_norm.data_ptr<float>(),
        (int)n_tiles, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32, 256-bit ld/st on sm_100)");
}
```

```yaml design-card
id: r2-ldg256-os-r16
parents:
  - g2-os-r16w8
operation: port
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows16
  red: halfwarp
  cache: stream
  spec: all
hypothesis: >-
  This keeps g2-os-r16w8's one-shot fused grid unchanged: 16 rows per 256-thread CTA, 8 CTAs per SM and 64 KB of
  loads in flight per SM. Each half-warp lane moves 32 B per ld/st.global.v8.f32, which halves load/store
  instructions and LSU requests per byte. At B200's 36 B/clk/SM under the 1.5 GHz lock, issue overhead is the likely
  cap on the parent's 6.4 TB/s (L) and 5.7 TB/s (M). H200, A100 and L40S cannot see this: they run the float4
  fallback.
expected_effect:
  S:
    pct: -3
    confidence: low
  M:
    pct: -5
    confidence: low
  L:
    pct: -6
    confidence: low
resources_sm100a:
  regs_per_thread: 30
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  ROWS: 16
  THREADS: 256
  min_blocks_per_sm: 8
  QK_USE_L2_HINT: 1
runs_on:
  H200:
    runs: true
    representative: false
    note: "sm_90 compiles the float4 fallback (__ldcs/__stcs); use for correctness only, timing does not represent the 256-bit path"
  A100:
    runs: true
    representative: false
    note: "float4 fallback; correctness check only"
  L40S:
    runs: true
    representative: false
    note: "float4 fallback; correctness smoke test only"
risks:
  - "ptxas may reject .L2::cache_hint combined with .v8.f32; the compile-verified forms (L1::no_allocate.v8 / st.global.v8) are available by setting QK_USE_L2_HINT=0"
  - "__launch_bounds__(256,8) caps the kernel at 32 regs; if ptxas needs more it will spill (check LOCAL in the sm_100a compile)"
  - "per-SM bytes in flight are unchanged from the parent (64 KB), so if B200 is latency/occupancy-bound rather than issue-bound the gain may be ~0"
  - "the alignment TORCH_CHECK requires 32-B aligned buffers; the harness gives 256-B aligned buffers, but any other caller with misaligned views would error rather than run"
  - "uneven wave on the smallest workloads (768 CTAs over 148 SMs, 5-6 each), the same as the parent"
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```