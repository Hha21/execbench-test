### Rationale
H16 is refuted, and nothing beats the parent at S. I split the parent's time into its parts on the rented B200 (ABAB, 7 reps, harness timing, cluster-2 in every variant). At every size from 131 to 40960, about 0.7 µs above the empty kernel is the load round trip, even with a clean flush. Stores add 0.03–0.08 µs. The dirty flush adds 0.03–0.14 µs.

Every alternative ties the parent within ±0.05 µs: other store types, other load types, and 32/64-thread CTAs. The one-shot bulk copy is slower: +0.1–0.35 µs as a CTA-level copy and +0.16–0.20 µs as a warp-level copy.

The only thing that moved the time is how many input streams are read. One stream, or three loads of the same address, is 0.1–0.2 µs faster than three streams, even though the buffers sit 21 KB apart in the same page. So the extra cost comes from three separate DRAM locations, which no access mechanism can avoid.

The candidate below is the best version of the idea (warp-level bulk copy at n ≤ 16384). It passed 16/16, but it is +6.8% slower at S, so it is a negative control and should **not** go to the portal.

```json solution-spec
{"name": "r2-silu-bw-s-wbulk-e2", "definition": "084_silu_activation_backward", "author": "solx-loop", "description": "E2 negative control: warp-level one-shot 1-D bulk loads at n<=16384, parent cluster-2 path elsewhere", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// SiLU backward: grad_input = grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x)), fp32 elementwise.
// One fused launch. Parent r1-silu-bw-cluster2-small everywhere except n <= 16384, where each warp loads its
// 3 x 512 B through one-shot 1-D bulk copies (cp.async.bulk, one mbarrier per warp) into shared memory.
// E2 (H16) experiment: the bulk path is 0.16-0.20 us SLOWER than the parent at S on the rented B200; it is kept
// as the best version of the idea for the archive, not as a portal candidate.
#include <cuda_runtime.h>
#include <stdint.h>

// Same operation order as the PyTorch reference, with IEEE-rounded ops (no FMA contraction), so results are bit-identical.
__device__ __forceinline__ float silu_bw(float g, float x, float s) {
    return __fmul_rn(g, __fmul_rn(s, __fadd_rn(1.0f, __fmul_rn(x, __fsub_rn(1.0f, s)))));
}

__device__ __forceinline__ float4 ld4_hint(const float* p, unsigned long long pol) {
    float4 v;
    asm volatile("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p), "l"(pol));
    return v;
}

__device__ __forceinline__ void st4_hint(float* p, float4 v, unsigned long long pol) {
    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1,%2,%3,%4}, %5;"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "l"(pol) : "memory");
}

static constexpr int THREADS = 256;
static constexpr unsigned long long POL_EF = 0x12F0000000000000ull;
static constexpr unsigned long long POL_EL = 0x14F0000000000000ull;
static constexpr long long BULK_MAX_N = 16384;

// Thread i handles floats [4i, 4i+4). The thread with i == nvec handles the < 4 tail elements, all loads first.
template <typename IT>
__device__ __forceinline__ void silu_body(const float* __restrict__ g, const float* __restrict__ x,
                                          const float* __restrict__ s, float* __restrict__ o, IT n) {
    const IT nvec = n >> 2;
    const IT i = (IT)blockIdx.x * THREADS + threadIdx.x;
    if (i < nvec) {
        const float4 a = ld4_hint(g + 4 * i, POL_EF);
        const float4 b = ld4_hint(x + 4 * i, POL_EF);
        const float4 c = ld4_hint(s + 4 * i, POL_EF);
        float4 r;
        r.x = silu_bw(a.x, b.x, c.x);
        r.y = silu_bw(a.y, b.y, c.y);
        r.z = silu_bw(a.z, b.z, c.z);
        r.w = silu_bw(a.w, b.w, c.w);
        st4_hint(o + 4 * i, r, POL_EL);
    } else if (i == nvec) {
        const IT e = nvec << 2;
        float ta[3], tb[3], tc[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (e + k < n) { ta[k] = g[e + k]; tb[k] = x[e + k]; tc[k] = s[e + k]; }
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (e + k < n) o[e + k] = silu_bw(ta[k], tb[k], tc[k]);
        }
    }
}

__global__ void __launch_bounds__(THREADS)
silu_bw_vec4(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
             float* __restrict__ o, unsigned n) { silu_body<unsigned>(g, x, s, o, n); }

// Same body, launched as clusters of 2 (grid padded to an even CTA count; the extra CTA exits on i < nvec).
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS)
silu_bw_vec4_c2(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
                float* __restrict__ o, unsigned n) { silu_body<unsigned>(g, x, s, o, n); }

// Small-n bulk path (clusters of 2): lane 0 of each warp issues three 1-D bulk copies (EF policy) of the warp's
// <= 128 floats per stream into shared memory and arms the warp's mbarrier; the warp waits on it, computes from
// shared memory and stores with EL. The <= 3 tail elements use plain loads. Sources are 16-B aligned (checked on host).
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS)
silu_bw_wbulk_c2(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
                 float* __restrict__ o, unsigned n) {
    __shared__ __align__(128) float sm[THREADS / 32][3][128];
    __shared__ __align__(8) unsigned long long bars[THREADS / 32];
    const unsigned nvec = n >> 2;
    const unsigned i = blockIdx.x * THREADS + threadIdx.x;
    const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned base = (blockIdx.x * THREADS + w * 32) * 4;
    const unsigned nb = 4 * nvec;
    const unsigned cnt = base < nb ? min(128u, nb - base) : 0u;
    if (i == nvec) {
        const unsigned e = nvec << 2;
        for (int k = 0; k < 3; ++k)
            if (e + k < n) o[e + k] = silu_bw(g[e + k], x[e + k], s[e + k]);
    }
    if (!cnt) return;
    const unsigned b = (unsigned)__cvta_generic_to_shared(&bars[w]);
    if (lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(cnt * 12u) : "memory");
        const float* src[3] = {g + base, x + base, s + base};
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const unsigned d = (unsigned)__cvta_generic_to_shared(&sm[w][k][0]);
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                         :: "r"(d), "l"(src[k]), "r"(cnt * 4u), "r"(b), "l"(POL_EF) : "memory");
        }
    }
    __syncwarp();
    asm volatile("{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; @!p bra WAIT_%=; }"
                 :: "r"(b) : "memory");
    const unsigned j = lane * 4;
    if (j < cnt) {
        const float4 a = *reinterpret_cast<const float4*>(&sm[w][0][j]);
        const float4 bb = *reinterpret_cast<const float4*>(&sm[w][1][j]);
        const float4 c = *reinterpret_cast<const float4*>(&sm[w][2][j]);
        float4 r;
        r.x = silu_bw(a.x, bb.x, c.x);
        r.y = silu_bw(a.y, bb.y, c.y);
        r.z = silu_bw(a.z, bb.z, c.z);
        r.w = silu_bw(a.w, bb.w, c.w);
        st4_hint(o + base + j, r, POL_EL);
    }
}

// 64-bit index variant for n >= 2^31.
__global__ void __launch_bounds__(THREADS)
silu_bw_vec4_64(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
                float* __restrict__ o, unsigned long long n) { silu_body<unsigned long long>(g, x, s, o, n); }

// Scalar fallback for pointers that are not 16-B aligned (never the case under the harness).
__global__ void __launch_bounds__(THREADS)
silu_bw_scalar(const float* __restrict__ g, const float* __restrict__ x, const float* __restrict__ s,
               float* __restrict__ o, long long n) {
    const long long i = (long long)blockIdx.x * THREADS + threadIdx.x;
    if (i < n) o[i] = silu_bw(g[i], x[i], s[i]);
}

extern "C" void silu_bw_launch(const float* g, const float* x, const float* s, float* o, long long n,
                               cudaStream_t stream) {
    if (n <= 0) return;
    const bool aligned = ((reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(x) |
                          reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(o)) & 15u) == 0;
    if (aligned) {
        const long long nthreads = (n >> 2) + ((n & 3) ? 1 : 0);
        const unsigned long long grid = (nthreads + THREADS - 1) / THREADS;
        if (n >= (1ll << 31)) {
            silu_bw_vec4_64<<<(unsigned)grid, THREADS, 0, stream>>>(g, x, s, o, (unsigned long long)n);
        } else if (n <= BULK_MAX_N) {
            silu_bw_wbulk_c2<<<(unsigned)((grid + 1) & ~1ull), THREADS, 0, stream>>>(g, x, s, o, (unsigned)n);
        } else if (grid <= 1184) {
            silu_bw_vec4_c2<<<(unsigned)((grid + 1) & ~1ull), THREADS, 0, stream>>>(g, x, s, o, (unsigned)n);
        } else {
            silu_bw_vec4<<<(unsigned)grid, THREADS, 0, stream>>>(g, x, s, o, (unsigned)n);
        }
    } else {
        const unsigned grid = (unsigned)((n + THREADS - 1) / THREADS);
        silu_bw_scalar<<<grid, THREADS, 0, stream>>>(g, x, s, o, n);
    }
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" void silu_bw_launch(const float* g, const float* x, const float* s, float* o, long long n,
                               cudaStream_t stream);

// DPS: run(grad_output, x, sigmoid_x, grad_input) writes grad_input in place.
void run(torch::Tensor grad_output, torch::Tensor x, torch::Tensor sigmoid_x, torch::Tensor grad_input) {
    TORCH_CHECK(grad_output.is_cuda() && x.is_cuda() && sigmoid_x.is_cuda() && grad_input.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32 &&
                sigmoid_x.scalar_type() == torch::kFloat32 && grad_input.scalar_type() == torch::kFloat32, "fp32 expected");
    TORCH_CHECK(grad_output.is_contiguous() && x.is_contiguous() && sigmoid_x.is_contiguous() && grad_input.is_contiguous(),
                "contiguous tensors expected");
    const long long n = grad_output.numel();
    TORCH_CHECK(x.numel() == n && sigmoid_x.numel() == n && grad_input.numel() == n, "numel mismatch");
    const c10::cuda::CUDAGuard guard(grad_output.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    silu_bw_launch(grad_output.data_ptr<float>(), x.data_ptr<float>(), sigmoid_x.data_ptr<float>(),
                   grad_input.data_ptr<float>(), n, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "SiLU backward (DPS)"); }
```

```yaml design-card
id: r2-silu-bw-s-wbulk-e2
parents: [r1-silu-bw-cluster2-small]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: oneshot
  launch: fused
  tile: elems1024
  red: none
  cache: stream
  spec: dispatch:size
hypothesis: >-
  H16: the ~0.3-0.5 us that the S-band kernel spends above an empty launch is a DRAM round trip plus store
  acknowledgement under the dirty post-memset L2, and a different access mechanism (one-shot 1-D bulk copy) shortens it.
  Result: REFUTED. The S-band excess is ~0.7 us of load round trip (the same under a clean flush); stores add <= 0.08 us.
  The warp-level bulk copy is 0.16-0.20 us slower, so this kernel is a negative control: do NOT submit to the portal.
expected_effect:
  S: {pct: +7, confidence: high}
  M: {pct: 0, confidence: high}
  L: {pct: 0, confidence: high}
resources_sm100a:
  regs_per_thread: 30
  smem_per_cta_bytes: 12352
  threads_per_cta: 256
  ctas_per_sm: 8
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs: {THREADS: 256, FLOATS_PER_THREAD: 4, CLUSTER: 2, CLUSTER_MAX_GRID: 1184, BULK_MAX_N: 16384, BULK_GRANULE: warp_512B_per_stream, LOAD_HINT: evict_first_const, STORE_HINT: evict_last_const}
dispatch:
  - {max_n: 16384, kernel: silu_bw_wbulk_c2, meta: {cluster: 2, bulk: warp}}
  - {max_n: 1212416, kernel: silu_bw_vec4_c2, meta: {cluster: 2}}
  - {max_n: null, kernel: silu_bw_vec4, meta: {}}
tests: [H16, H9, H4]
findings:
  - "DECOMPOSITION (rented B200, harness_time, cluster-2 everywhere, ABAB 7 reps, median us; sizes 131/2053/4096/16384/40960). DIRTY flush: empty 1.41/1.38/1.41/1.38/1.39; loads-only (3 EF loads, store predicated off) 2.14/2.16/2.15/2.13/2.19; parent 2.17/2.18/2.21/2.21/2.24; stores-only EL 1.55/1.63/1.63/1.63/1.63 [probe_b200]"
  - "DECOMPOSITION CLEAN read-sweep flush (monkeypatched timing._clear_cache to a sum over the buffer): empty 1.36/1.36/1.38/1.35/1.35; loads-only 2.11/2.10/2.05/2.09/2.10; parent 2.14/2.13/2.07/2.09/2.10; stores-only 1.44/1.55/1.55/1.52/1.60 [probe_b200]"
  - "=> the load leg holds ~0.73 us above the empty kernel at every S size, dirty or clean; the store leg adds only 0.03-0.08 us; the dirty flush costs 0.03-0.14 us in total. The S excess is the DRAM read latency itself, not dirty write-backs or store acks [probe_b200]"
  - "cluster-2 empty kernel 1.41 us vs non-cluster empty 1.51 vs empty with 1 predicated store 1.54 (4096 grid, one run): the cluster gain is also visible on an empty kernel [probe_b200]"
  - "dead end (store leg, 7 reps): st.global.wt, st.global.cs, default st, st.L1::no_allocate+EL policy all within +-0.05 us of the parent EL store at 131-40960 [probe_b200]"
  - "dead end (load leg, 7 reps): ld.global.nc.L1::no_allocate+EF policy, ld.global.cv, plain ld.global, 64-thread and 32-thread CTAs (cluster 2) all within +-0.05 us of the parent [probe_b200]"
  - "dead end: one-shot 1-D bulk copy, CTA level (thread 0: mbarrier.init + expect_tx + 3 cp.async.bulk EF into 12 KB smem, __syncthreads, try_wait): +0.10/+0.22/+0.21/+0.27/+0.27 us; with a bulk store (fence.proxy.async + bulk_group + wait_group.read 0) +0.18/+0.37/+0.30/+0.42/+0.34 us [probe_b200]"
  - "dead end: warp-level one-shot bulk (lane 0 per warp, own mbarrier, __syncwarp only) +0.16/+0.20/+0.16/+0.18/+0.15 us (11 reps); bit-exact. TMA/bulk engine latency exceeds LDG latency for a single round trip: H9 closes for this problem [probe_b200]"
  - "KEY diagnostic: loading 1 stream instead of 3 is 0.06-0.22 us faster (s1 1.99/1.99/2.02/2.02 vs s3 2.21/2.05/2.19/2.24 at 131/4096/16384/40960, 9 reps); 3 loads of the SAME address = 1 stream (2.00/2.02/2.02/2.05); 2 streams in between; reversing load order changes nothing [probe_b200]"
  - "harness pool layout: the shifted g, x, s, o buffers are 21 KB apart (one contiguous pool region, same 2 MB page), so the multi-stream cost is not a TLB miss; most likely DRAM bank/row or channel conflicts between three distinct locations, fixed by the harness layout [probe_b200; cause INFERRED]"
  - "run_tests: 16/16 passed; vs the parent on the same GPU S +6.8% (131 2.3 vs 2.1, 2053-4096 +7-8%), M +0.1%, L -0.3%; predicted portal 0.5110 vs 0.5166 [run_tests]"
  - "conclusion: H16 refuted (every store type, load type, CTA size and bulk variant ties or loses). The remaining S excess is ~0.55 us of single-access DRAM latency plus ~0.15 us from three distinct input locations; nothing kernel-side reaches it. Stop S memory-path work; only launch-path ideas (E1/H15) remain [INFERRED]"
paths:
  - {max_tokens: 16384, lang: cuda, width: 128, threads: 256, rows: 1, grid: oneshot, mem: tma, launches: 1, x: ef, st: el, cluster: 2}
  - {max_tokens: 1212416, lang: cuda, width: 128, threads: 256, rows: 1, grid: oneshot, mem: ldg, launches: 1, x: ef, st: el, cluster: 2}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 1, grid: oneshot, mem: ldg, launches: 1, x: ef, st: el}
runs_on:
  H200: {runs: true, representative: false, note: "bulk copies and clusters exist on sm_90a, but the launch and latency mix differs; measured on B200 only"}
  A100: {runs: false, representative: false, note: "no cp.async.bulk or clusters on sm_80"}
  L40S: {runs: false, representative: false}
risks:
  - "Slower than the parent at S by ~0.15 us (+7%): a negative control, not a portal candidate"
  - "Bulk sources need 16-B alignment: the host checks it, and the harness gives 256-B aligned pointers"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```