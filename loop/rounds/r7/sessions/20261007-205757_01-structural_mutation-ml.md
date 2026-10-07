# Design session 20261007-205757_01-structural_mutation-ml

Task: {"operation": "structural_mutation", "parents": ["r5-ldg256-os-r16-stel"], "band": "M,L", "gain": 0.01576610161101355, "instructions": "Goal: cut medium and large input latency (B*S > 600) on B200. The parent reaches 6.45 TB/s (M) and 6.93 TB/s (L) effective; a further 5% would add about +0.016 to the score. Its fit is 7.5 TB/s, above the physical read+write rate: outputs still in L2 when the kernel ends are written back after the timed window (see problem_038.md section 7), so think about what occupies L2 at the end, not only bandwidth. Propose the single change most likely to help, correct for every shape, one launch. Slower on B200 than the parent in these bands already: d1-os-r8w4-nohint (ldg128/oneshot): +7%; g2-os-r16w8 (ldg128/oneshot): +9%; g2-os-r8w4 (ldg128/oneshot): +9%; g2-tma-r8s3 (tma-tensor/persistent): +26%; g2-tmaws-hb8s4 (tma-tensor/persistent-wstat): +20%; g2-ws-hb16w8s3 (cpasync/persistent-wstat): +24%; r2-ldg256-os-r16 (ldg256/oneshot): +9%; r3-cute-ldg256-os-r16 (ldg256/oneshot): +5%; v028 (ldg128/persistent): +59%; v039 (ldg128/persistent): +22%; v040 (ldg128/oneshot): +143%; v050 (ldg128/oneshot): +38%. Ideas already proposed but not yet measured on B200 (do not repeat them): r1-bulk1d-ring-r16s16 (bulk1d/persistent, cuda_cpp); r2-ldg256-tokblk (ldg256/oneshot, cuda_cpp); r3-os-l2pf-r8w4 (ldg128/oneshot, cuda_cpp); r3-tokblk-s-disp (ldg128/oneshot, triton); r4-cuda-ldg256-token-r2 (ldg256/oneshot, cuda_cpp); r4-cute-ldg256-sw2-disp (ldg256/persistent, cute_dsl); r4-ldg256-wstat-pf2 (ldg256/persistent-wstat, cuda_cpp); r6-gluon-cpasync-os-r16w8-stel (cpasync/oneshot, gluon); r6-ldg256-qkpair-r16 (ldg256/oneshot, cuda_cpp); r6-s-ldef-nc-disp (ldg256/oneshot, cuda_cpp)."}


## Turn 0 (11943 tokens out, $0.3250248)



**get_kernel** →
```
--- parent r6-s-ldef-nc-disp ---
design card:
id: r6-s-ldef-nc-disp
parents:
  - r5-ldg256-os-r16-stel
operation: specialisation
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
  My estimate of what the parent's ~3.2 us fixed cost on small inputs is made of:
  (a) CUPTI launch-to-end span of a near-empty kernel, about 1-1.5 us;
  (b) loaded DRAM latency plus TLB misses on first touch of fresh pool addresses, about 1 us;
  (c) store drain and grid exit, about 0.3-0.5 us;
  (d) possibly extra DRAM write-backs: every line we allocate in an L2 full of the flush's dirty zero lines
  may evict one of them.
  (a)-(c) are mostly outside a kernel's control. r4-cute-ldg256-sw2-disp tests warp and CTA dispatch.
  This candidate tests (d) only. For B*S <= 600, x loads carry L2 evict_first and weights use ld.global.nc,
  the load policy CAKE kept after paired cold-L2 B200 tests. Allocations for input lines then recycle our
  own clean lines instead of evicting dirty flush lines, and output stores (still evict_last) can take the
  place of those clean input lines. Larger sizes run the parent unchanged, so the B200 result isolates the
  effect on S.
  Measured on A100 (128-bit fallback), against the parent: S +0.3% (noise; -2.2%, -1.1% at 25 MB, +1.5 to +1.8%
  at 12.6 MB), M/L 0%. The near-peak marginal bandwidth on both GPUs already argues that (d) is small. So
  this is mainly a cheap B200 check of the mechanism: an expected gain of 0 to 3% on S, B200-only if it
  appears at all.
  Also tried and rejected: 2-4 tiles per CTA. On A100 that was -6% at 128 tokens and +14% at 256 tokens. The
  128-token gain is an A100 wave effect (the parent runs 1.8 waves on 108 SMs, but one wave on B200's 148).
  It duplicates r4-cute, and the 2-tile kernel needs 48 registers on sm_100a, so 5 CTAs/SM would spill
  256 tokens into a second wave.
expected_effect:
  S:
    pct: -1.5
    confidence: low
  M:
    pct: 0
    confidence: high
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 31
  smem_per_cta_bytes: 0
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  ROWS: 16
  THREADS: 256
  S_MAX_TOKENS: 600
  S_load_policy: "x: L2::cache_hint 0x12F0000000000000 (evict_first); w: ld.global.nc"
  store_policy: "L2 evict_last, fraction 1.0 (all sizes)"
dispatch:
  - max_tokens: 600
    kernel: qk_rms_kernel<true>
    meta:
      ROWS: 16
      THREADS: 256
      loads: "x evict_first, w nc"
  - max_tokens: 1000000000
    kernel: qk_rms_kernel<false>
    meta:
      ROWS: 16
      THREADS: 256
      note: "identical to parent r5-ldg256-os-r16-stel"
runs_on:
  H200:
    runs: true
    representative: false
    note: "128-bit fallback; the L2 is 50 MB instead of 126 MB, so the dirty-flush interaction differs"
  A100:
    runs: true
    representative: false
    note: "measured S +0.3%, M 0.0%, L -0.1% vs parent (noise); 16/16 pass"
  L40S:
    runs: true
    representative: false
risks:
  - "Term (d) may be negligible on B200, since marginal bandwidth between 12.6 and 25 MB is already about 7.9 TB/s. The expected result is then about 0 on S."
  - "On A100, evict_first loads hurt L by 4% (r5 session). That is why they apply only to B*S <= 600; the 586-token workload (57.6 MB) is the one most exposed if B200 behaves the same way."
  - "S-band noise on the portal is about 5%, so one submission cannot resolve an effect of 1-3%."
measure_first:
  - H200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true

measurements (geomean µs per band S/M/L): B200r 5.4/19.0/75.4
B200 portal: score not submitted, geomean - µs

sm_100a static features: [{"kernel": "_ZN41_GLOBAL__N__b40f901f_9_kernel_cu_a02f06f813qk_rms_kernelILb0EEEvPKfS2_S2_S2_PfS3_if", "num_warps": null, "regs": 31, "smem": 0, "local": 0, "resident_per_sm": null, "ops": {"LDG.256": 3, "STG.256": 2, "LDG.CONSTANT.256": 1}}, {"kernel": "_ZN41_GLOBAL__N__b40f901f_9_kernel_cu_a02f06f813qk_rms_kernelILb1EEEvPKfS2_S2_S2_PfS3_if", "num_warps": null, "regs": 31, "smem": 0, "local": 0, "resident_per_sm": null, "ops": {"LDG.256": 3, "STG.256": 2, "LDG.CONSTANT.256": 1}}]
source:
--- file kernel.cu ---
// r6-s-ldef-nc-disp: per-head RMSNorm of Q and K (fp32), one fused one-shot launch, size dispatch.
// Every size uses the parent r5 structure (16 rows per 256-thread CTA, half-warp per row, 256-bit ld/st on
// sm_100a, output stores L2 evict_last). M/L (B*S > 600) run the parent kernel exactly (LOAD_S = false).
// S band (B*S <= 600) changes only the load cache policy, copying the policy CAKE kept after paired B200
// CUPTI cold-L2 tests: streamed x loads carry L2::cache_hint EVICT_FIRST (policy constant), weights load
// through the non-coherent read-only path (ld.global.nc).
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
constexpr int64_t S_MAX_TOKENS = 600;
constexpr uint64_t POL_EVICT_FIRST = 0x12F0000000000000ULL;   // CUTLASS CacheHintSm90::EVICT_FIRST

__device__ __forceinline__ uint64_t pol_evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
    return p;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]) : "l"(p));
}
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
__device__ __forceinline__ void ld4(const float* p, float* v) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p));
}
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
__device__ __forceinline__ void ld8(const float* p, float (&v)[8]) { ld4(p, v); ld4(p + 4, v + 4); }
__device__ __forceinline__ void ld8_nc(const float* p, float (&v)[8]) { ld4_nc(p, v); ld4_nc(p + 4, v + 4); }
__device__ __forceinline__ void ld8_h(const float* p, float (&v)[8], uint64_t pol) {
    ld4_h(p, v, pol); ld4_h(p + 4, v + 4, pol);
}
__device__ __forceinline__ void st8_h(float* p, const float (&v)[8], uint64_t pol) {
    st4_h(p, v, pol); st4_h(p + 4, v + 4, pol);
}
#endif

template <bool LOAD_S>
__global__ void __launch_bounds__(THREADS, 8)
qk_rms_kernel(const float* __restrict__ q, const float* __restrict__ k,
              const float* __restrict__ wq, const float* __restrict__ wk,
              float* __restrict__ qo, float* __restrict__ ko, int n_tiles, float eps) {
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
    if (LOAD_S) {                                   // both loads issued before any use
        ld8_h(x + row * D + c, xv, POL_EVICT_FIRST);
        ld8_nc(w + (size_t)head * D + c, wv);
    } else {
        ld8(x + row * D + c, xv);
        ld8(w + (size_t)head * D + c, wv);
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)(2 * n_tiles));
    const float* qp = query.data_ptr<float>(); const float* kp = key.data_ptr<float>();
    const float* wqp = weight_q.data_ptr<float>(); const float* wkp = weight_k.data_ptr<float>();
    float* qop = query_norm.data_ptr<float>(); float* kop = key_norm.data_ptr<float>();
    if (tokens <= S_MAX_TOKENS)   // shape-only dispatch
        qk_rms_kernel<true><<<grid, THREADS, 0, stream>>>(qp, kp, wqp, wkp, qop, kop, (int)n_tiles, (float)eps);
    else
        qk_rms_kernel<false><<<grid, THREADS, 0, stream>>>(qp, kp, wqp, wkp, qop, kop, (int)n_tiles, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Fused per-head RMSNorm of Q and K (fp32); S band: EF x loads, nc weights");
}


```


## Turn 1 (3998 tokens out, $0.2377798)




### run_tests (draft 1): r5 + evict_first x loads + nc weights at all sizes (tests L2 occupancy by inputs at M/L)

**run_tests** →
```
run_tests on NVIDIA B200 (81.8 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.5 vs 5.6 (-1.3%); 4,1657 93.5 vs 93.5 (-0.0%); 4,1024 58.3 vs 58.0 (+0.6%); 8,773 86.9 vs 87.0 (-0.0%); 8,128 13.9 vs 14.5 (-4.4%); 2,293 8.7 vs 9.3 (-6.6%); 16,256 58.1 vs 58.1 (+0.0%); 1,1024 14.6 vs 14.7 (-0.5%); 4,256 14.2 vs 14.5 (-2.0%); 1,128 4.2 vs 4.2 (+0.0%); 32,128 58.2 vs 57.9 (+0.4%); 4,512 28.8 vs 28.6 (+0.7%); 1,131 4.1 vs 4.1 (-0.7%); 8,256 28.9 vs 28.7 (+0.7%); 1,8192 115.7 vs 115.7 (+0.0%); 1,256 5.5 vs 5.6 (-0.5%)
change by band (geomean): S -1.8%, M -1.1%, L +0.2%
projected portal score: 0.6104 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 2 (8368 tokens out, $0.219987)




### run_tests (draft 2): EF loads + evict_last stores only on the last 96 MB of output (normal before)

**run_tests** →
```
run_tests on NVIDIA B200 (42.4 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.4 vs 5.6 (-2.5%); 4,1657 93.2 vs 93.5 (-0.4%); 4,1024 57.8 vs 58.0 (-0.4%); 8,773 87.0 vs 87.0 (+0.0%); 8,128 14.2 vs 14.5 (-2.2%); 2,293 8.7 vs 9.3 (-7.0%); 16,256 57.8 vs 58.1 (-0.5%); 1,1024 14.5 vs 14.7 (-1.4%); 4,256 14.2 vs 14.5 (-2.2%); 1,128 4.2 vs 4.2 (+0.0%); 32,128 57.9 vs 57.9 (+0.0%); 4,512 28.7 vs 28.6 (+0.5%); 1,131 4.1 vs 4.1 (-0.7%); 8,256 28.7 vs 28.7 (+0.2%); 1,8192 115.3 vs 115.7 (-0.4%); 1,256 5.5 vs 5.6 (-1.1%)
change by band (geomean): S -2.3%, M -1.0%, L -0.3%
projected portal score: 0.6114 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 3 (6607 tokens out, $0.2205894)




### run_tests (draft 3): EF loads + EL stores + cp.async.bulk.prefetch.L2 of the tile one resident wave ahead

**run_tests** →
```
run_tests on NVIDIA B200 (81.8 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.4 vs 5.6 (-3.6%); 4,1657 98.1 vs 93.5 (+4.8%); 4,1024 60.0 vs 58.0 (+3.5%); 8,773 91.6 vs 87.0 (+5.4%); 8,128 13.9 vs 14.5 (-4.6%); 2,293 8.8 vs 9.3 (-5.6%); 16,256 60.6 vs 58.1 (+4.3%); 1,1024 14.0 vs 14.7 (-4.6%); 4,256 13.9 vs 14.5 (-4.4%); 1,128 4.1 vs 4.2 (-0.7%); 32,128 59.7 vs 57.9 (+3.1%); 4,512 28.8 vs 28.6 (+0.8%); 1,131 4.0 vs 4.1 (-1.7%); 8,256 28.8 vs 28.7 (+0.2%); 1,8192 121.3 vs 115.7 (+4.8%); 1,256 5.5 vs 5.6 (-1.8%)
change by band (geomean): S -2.7%, M -2.5%, L +4.3%
projected portal score: 0.6045 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 4 (4841 tokens out, $0.1729724)




### run_tests (draft 4): prefetch one wave ahead with evict_last hint so prefetched lines survive the EL output pressure at L

**run_tests** →
```
run_tests on NVIDIA B200 (88.1 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.3 vs 5.6 (-4.3%); 4,1657 99.6 vs 93.5 (+6.5%); 4,1024 61.3 vs 58.0 (+5.8%); 8,773 92.5 vs 87.0 (+6.4%); 8,128 15.0 vs 14.5 (+3.2%); 2,293 9.0 vs 9.3 (-3.5%); 16,256 61.2 vs 58.1 (+5.2%); 1,1024 15.2 vs 14.7 (+3.4%); 4,256 15.0 vs 14.5 (+3.3%); 1,128 4.1 vs 4.2 (-0.5%); 32,128 61.3 vs 57.9 (+5.9%); 4,512 30.2 vs 28.6 (+5.8%); 1,131 4.1 vs 4.1 (-1.0%); 8,256 30.2 vs 28.7 (+5.4%); 1,8192 122.7 vs 115.7 (+6.1%); 1,256 5.4 vs 5.6 (-2.9%)
change by band (geomean): S -2.4%, M +4.2%, L +6.0%
projected portal score: 0.5960 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 5 (5214 tokens out, $0.1677432)




### run_tests (draft 5): size dispatch: L2 bulk prefetch one wave ahead for B*S<=1536, EF loads + EL stores everywhere

**run_tests** →
```
run_tests on NVIDIA B200 (83.3 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.3 vs 5.6 (-4.3%); 4,1657 93.3 vs 93.5 (-0.3%); 4,1024 58.2 vs 58.0 (+0.4%); 8,773 87.1 vs 87.0 (+0.1%); 8,128 13.8 vs 14.5 (-4.9%); 2,293 8.8 vs 9.3 (-5.7%); 16,256 58.0 vs 58.1 (-0.1%); 1,1024 14.1 vs 14.7 (-4.2%); 4,256 13.9 vs 14.5 (-4.2%); 1,128 4.1 vs 4.2 (-0.7%); 32,128 58.1 vs 57.9 (+0.2%); 4,512 28.6 vs 28.6 (+0.3%); 1,131 4.0 vs 4.1 (-1.7%); 8,256 28.7 vs 28.7 (-0.1%); 1,8192 115.3 vs 115.7 (-0.3%); 1,256 5.5 vs 5.6 (-1.8%)
change by band (geomean): S -2.9%, M -2.6%, L +0.0%
projected portal score: 0.6125 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 6 (4573 tokens out, $0.1589928)




### run_tests (draft 6): probe: prefetch distance 2 waves, enabled up to 2100 tokens

**run_tests** →
```
run_tests on NVIDIA B200 (42.7 s): 16/16 workloads PASSED
latency vs reference r5-ldg256-os-r16-stel on the same GPU (µs, this vs reference):
  2,128 5.4 vs 5.6 (-2.5%); 4,1657 93.2 vs 93.5 (-0.4%); 4,1024 58.1 vs 58.0 (+0.3%); 8,773 87.0 vs 87.0 (+0.0%); 8,128 14.2 vs 14.5 (-2.2%); 2,293 9.4 vs 9.3 (+0.8%); 16,256 58.1 vs 58.1 (-0.0%); 1,1024 14.4 vs 14.7 (-2.1%); 4,256 14.4 vs 14.5 (-0.8%); 1,128 4.1 vs 4.2 (-0.7%); 32,128 58.1 vs 57.9 (+0.4%); 4,512 28.8 vs 28.6 (+0.8%); 1,131 4.0 vs 4.1 (-1.7%); 8,256 29.1 vs 28.7 (+1.3%); 1,8192 115.3 vs 115.7 (-0.3%); 1,256 5.5 vs 5.6 (-1.4%)
change by band (geomean): S -1.1%, M -0.6%, L -0.0%
projected portal score: 0.6099 (current best 0.6089). This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, so the projection shrinks them (portal change ~ measured change ** k, k = S 0.45, M 0.63, L 1.09, fitted on kernels measured on both).
```


## Turn 7 (4832 tokens out, $0.160205)




### compile_b200 (draft 7): final candidate static check

**compile_b200** →
```
compile_b200 for sm_100a (9.5 s): 2 kernel(s)
- _ZN41_GLOBAL__N__f332ecb9_9_kernel_cu_a02f06f813qk_rms_kerne: 28 regs/thread, 0 B shared, 0 B local (spills), CTAs/SM unknown; memory ops in SASS: LDG.256 x2, LDG.CONSTANT.256 x2, STG.256 x2
- _ZN41_GLOBAL__N__f332ecb9_9_kernel_cu_a02f06f813qk_rms_kerne: 28 regs/thread, 0 B shared, 0 B local (spills), CTAs/SM unknown; memory ops in SASS: LDG.256 x2, LDG.CONSTANT.256 x2, STG.256 x2
```


## Turn 8 (6214 tokens out, $0.188533)

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


---
session total: 9 turns, 7 GPU calls, $1.85
