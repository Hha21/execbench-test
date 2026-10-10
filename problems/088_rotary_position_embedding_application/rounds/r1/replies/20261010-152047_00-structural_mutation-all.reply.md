### Rationale
I couldn't lock the clocks on the rented card (`nvidia-smi -lgc` is not permitted), so the portal's 1.5 GHz regime can't be reproduced. The probes found that most of r0's L-band loss comes from its 256-thread path, not from the cos/sin tables. With 64-thread CTAs, r0 runs 6–8% faster at 322–407 MB and 2–3% faster at 612/2689 MB; three probe sessions and run_tests agree. Measured against that 64-thread kernel, forcing every table read to hit L1 gains only 0.5–1.6% at 269–2689 MB, except 612 MB (5.5%). None of the table-sharing layouts beat the plain 64-thread kernel: L1 remap (A), shared memory (B), register loop (C), and 4 heads per warp (W). An `.L2::evict_last` hint on the table loads recovers about 2.7 points at 612 MB.

The candidate is r0's kernel body unchanged, with 64-thread CTAs up to 450k rows and 128-thread CTAs plus evict_last tables above. run_tests: 16/16 pass, L −5.2%. S/M show +1.2%, but their code is identical to r0's 64-thread path, so I read that as bench bias.

```json solution-spec
{"name": "r1-rope-ldg256-os64-tabel", "definition": "088_rotary_position_embedding_application", "author": "solx-loop", "description": "r0 body with small CTAs at every size (64 threads up to 450k rows, 128 above) and L2::evict_last table loads on the largest path; r0's 256-thread L path lost 2-9% on the bench", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// RoPE application (Q and K), fp32, head_dim 128. One fused one-shot launch.
// Layout: 8 lanes per row; lane j owns columns [8j, 8j+8) and [64+8j, 64+8j+8) (one 256-bit load each),
// so the rotate_half partner is already in the same thread and no shuffles are needed.
// out[i]    = x[i]*cos[i]    - x[i+64]*sin[i]        (i < 64)
// out[i+64] = x[i+64]*cos[i+64] + x[i]*sin[i+64]
// r1: small CTAs at every size (64 threads up to 450k rows, 128 above; r0's 256-thread CTAs lost 2-9% at L on
// the bench), and on the largest path the cos/sin loads are marked L2::evict_last so the tables (4-8 MB there)
// are not evicted by the stream of evict_last output lines.
#include <cuda_runtime.h>
#include <stdint.h>

#define LD8(p, v) asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p))
#define LDT8(p, v) asm volatile("ld.global.nc.L2::evict_last.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p))
#define LDX8(p, v) asm volatile("ld.global.nc.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p))
#define ST8EL(p, v, pol) asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]), "l"(pol) : "memory")

template <int THREADS, bool TAB_EL>
__global__ void __launch_bounds__(THREADS)
rope_kernel(const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ cs, const float* __restrict__ sn,
            float* __restrict__ qo, float* __restrict__ ko,
            unsigned Rq, unsigned R, unsigned S)
{
    constexpr int ROWS = THREADS / 8;
    const unsigned j = threadIdx.x & 7;
    const unsigned r = blockIdx.x * ROWS + (threadIdx.x >> 3);
    if (r >= R) return;
    const float* x; float* y; unsigned rl;
    if (r < Rq) { x = q; y = qo; rl = r; } else { x = k; y = ko; rl = r - Rq; }
    const unsigned s = rl % S;                      // sequence position of this row
    const float* xp = x + (size_t)rl * 128 + j * 8;
    const float* cp = cs + (size_t)s * 128 + j * 8;
    const float* sp = sn + (size_t)s * 128 + j * 8;
    y += (size_t)rl * 128 + j * 8;
    float xl[8], xh[8], cl[8], ch[8], sl[8], sh[8], ol[8], oh[8];
    LDX8(xp, xl); LDX8(xp + 64, xh);
    if (TAB_EL) { LDT8(cp, cl); LDT8(cp + 64, ch); LDT8(sp, sl); LDT8(sp + 64, sh); }
    else        { LD8(cp, cl);  LD8(cp + 64, ch);  LD8(sp, sl);  LD8(sp + 64, sh); }
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        ol[e] = xl[e] * cl[e] - xh[e] * sl[e];
        oh[e] = xh[e] * ch[e] + xl[e] * sh[e];
    }
    ST8EL(y, ol, pol);
    ST8EL(y + 64, oh, pol);
}

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            cudaStream_t stream)
{
    if (R <= 450000u) {            // up to 407 MB: 8 rows per 64-thread CTA (r0's S/M path, now also 322-407 MB)
        const unsigned grid = (R + 7) / 8;
        rope_kernel<64, false><<<grid, 64, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
    } else {                       // 612 MB and above: 16 rows per 128-thread CTA, evict_last tables
        const unsigned grid = (R + 15) / 16;
        rope_kernel<128, true><<<grid, 128, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
    }
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

extern "C" void rope_launch(const float* q, const float* k, const float* cs, const float* sn,
                            float* qo, float* ko, unsigned Rq, unsigned R, unsigned S,
                            cudaStream_t stream);

void run(torch::Tensor query, torch::Tensor key, torch::Tensor cos, torch::Tensor sin,
         torch::Tensor query_rotated, torch::Tensor key_rotated)
{
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && cos.is_cuda() && sin.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && key.scalar_type() == torch::kFloat32 &&
                cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && cos.is_contiguous() && sin.is_contiguous() &&
                query_rotated.is_contiguous() && key_rotated.is_contiguous(), "contiguous only");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && query.size(3) == 128 && key.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == query.size(2) && cos.size(1) == 128, "cos shape");
    TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape");
    TORCH_CHECK(key.size(2) == query.size(2), "seq_len mismatch");
    TORCH_CHECK(query_rotated.sizes() == query.sizes() && key_rotated.sizes() == key.sizes(), "output shapes");
    TORCH_CHECK(((query.data_ptr<float>() == nullptr) || (reinterpret_cast<uintptr_t>(query.data_ptr()) & 31) == 0) &&
                (reinterpret_cast<uintptr_t>(key.data_ptr()) & 31) == 0 &&
                (reinterpret_cast<uintptr_t>(cos.data_ptr()) & 31) == 0 &&
                (reinterpret_cast<uintptr_t>(sin.data_ptr()) & 31) == 0 &&
                (reinterpret_cast<uintptr_t>(query_rotated.data_ptr()) & 31) == 0 &&
                (reinterpret_cast<uintptr_t>(key_rotated.data_ptr()) & 31) == 0, "32-byte alignment required");

    const int64_t S = query.size(2);
    const int64_t Rq = query.numel() / 128;
    const int64_t R = Rq + key.numel() / 128;
    TORCH_CHECK(R < (1LL << 31) && S > 0, "too many rows");
    if (R == 0) return;

    const c10::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rope_launch(query.data_ptr<float>(), key.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
                query_rotated.data_ptr<float>(), key_rotated.data_ptr<float>(),
                (unsigned)Rq, (unsigned)R, (unsigned)S, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "RoPE application (DPS)"); }
```

```yaml design-card
id: r1-rope-ldg256-os64-tabel
parents:
  - r0-rope-ldg256-os-stel
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows8
  red: warp
  cache: default
  spec: dispatch:size
hypothesis: >-
  Most of r0's L-band loss is its 256-thread CTA path, not the cos/sin L2 traffic. 50 regs give 4x256 = 1024
  threads per SM against 18x64 = 1152, with coarser CTA retirement, and on the bench 64-thread CTAs beat it by 6-8%
  at 322-407 MB and 2-3% at 612-2689 MB. Table-sharing layouts recover nothing beyond that. An L2::evict_last hint
  on the table loads helps where the tables are 4-8 MB and are re-read across a multi-MB output stream that is also
  evict_last. The portal locks the SM clock at 1.5 GHz, which may scale CTA dispatch cost, so the largest path uses
  128-thread CTAs to halve the CTA count (2689 MB: 164k CTAs instead of 328k).
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: high
  L:
    pct: -5
    confidence: medium
resources_sm100a:
  regs_per_thread: 50
  smem_per_cta_bytes: 0
  threads_per_cta: 64
  ctas_per_sm: 18
  bytes_in_flight_per_sm: 73728
  launches_per_call: 1
knobs:
  THREADS_SM: 64
  THREADS_L: 128
  row_threshold: 450000
  table_hint_large: L2::evict_last
  store_hint: evict_last
dispatch:
  - {max_rows: 450000, kernel: "rope_kernel<64,false>", meta: {threads: 64, rows: 8, table: nc}}
  - {max_rows: null, kernel: "rope_kernel<128,true>", meta: {threads: 128, rows: 16, table: nc_L2_evict_last}}
tests:
  - H11
  - H2
  - H3
findings:
  - "clock lock: nvidia-smi -lgc 1500,1500 refused on the rented card (no permission); -lmc unsupported (only deferred). The portal's 1.5 GHz regime cannot be reproduced; the H11 locked-clock test is impossible here [probe_b200]"
  - "no-table probe (cos/sin replaced by constants, 30 regs) vs r0 dispatcher: -10..-12% at 36 MB, -8..-10% at 61, -3% at 134, -1..-2% at 269, -7% at 322, -8% at 407, -7% at 612, -2% at 2689 [probe_b200, 3 sessions]"
  - "s0 probe (same 50 regs, table row forced to s&1 so every table load hits L1) matches no-table at L (-7..-8% at 322-612) but not at 36-61 MB (-2.5..-5% vs -10%): at S/M part of the no-table gain is occupancy (30 vs 50 regs) [probe_b200]"
  - "MAIN: r0's 256-thread L path is the loss. r0 with 64-thread CTAs at every size: -8.0% at 322, -7.2% at 326, -7.3% at 354, -6.1% at 407, -1.8% at 612, -2.5% at 2689 vs r0's dispatcher; 128-thread: -6.8/-5.4/-6.3/-5.3/-3.0/-3.1%. Seen in 3 probe sessions; contradicts r0's finding that 256 beat 64 by 5% at 407 MB. 49k-327k 64-thread CTAs are not dispatch-bound on the bench [probe_b200]"
  - "vs 64-thread r0, the L1-hit table (s0) gains only -0.5..-1.6% at 269-2689 MB, but -5.5% at 612 MB (S=8192, 8 MB tables) and -11% at 2.2 MB: table traffic is a small cost on the bench once CTAs are small [probe_b200]"
  - "table loads with L2::evict_last (.L2::evict_last qualifier or cache_hint policy; same SASS LDG.E.ELL2.256): -2.7 points at 612 MB, -0.9 at 2689, +0.5..+0.7 (noise) at 322-407, neutral at S/M; kept on the >450k-row path only [probe_b200]"
  - "dead end, variant A (s-tile x head-group CTAs, heads slowest; 256 thr TS x HG = 8x4, 4x8, 16x2; 64 thr 4x2, 2x4): +2..+12% at 36-269 MB, ties at 2689; A_4_2's -5% at 407 MB is the 64-thread effect, not table sharing [probe_b200]"
  - "dead end, variant B (table slice staged in static smem, __syncthreads): +7..+25% at every size (B_8_4: 426 vs 399 us at 2689 MB) [probe_b200]"
  - "dead end, variant C (table in registers, loop over HG heads; HG=2 with x prefetch 80 regs, HG=4/8 with or without prefetch 64-80 regs; C2 unrolled 72 regs at 64/128 thr): best C2p -4% at 407 vs 256-thread r0 but +2..+3% vs 64-thread r0 at 407-2689 MB [probe_b200]"
  - "dead end, variant W (4 heads of one s per warp, so each table instruction hits 256 B): +3..+4% vs 64-thread r0 at 407-2689 MB; its no-table twin W0 ties r0, so W's in-warp table merging costs ~3.5% instead of saving [probe_b200]"
  - "trap: padding CTAs with dynamic smem to control occupancy (carve-out 100%) makes r0 +40..+77% slower because the table loads lose L1; r0 depends on L1 for the tables, and smem-padding occupancy tests are invalid for this kernel [probe_b200]"
  - "run_tests (final): 16/16 pass; L band -5.2% (322 MB -7.2%, 326 -8.1%, 354 -7.2%, 407 -4.3%, 612 -1.1%, 2689 -3.0%); S/M +1.2% although the <=450k-row code is identical to r0's 64-thread path: bench bias or noise; the same +1.1% appeared with the hint on all paths [run_tests]"
paths:
  - {max_tokens: null, max_rows: 450000, lang: cuda, width: 256, threads: 64, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, max_rows: null, lang: cuda, width: 256, threads: 128, rows: 16, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: el, st: el}
runs_on:
  H200:
    runs: false
    representative: false
    note: "256-bit ld/st.global.v8.f32 and createpolicy need sm_100a"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "r0's card measured 256-thread beating 64-thread by 5% at 407 MB; three sessions here say the opposite. If the portal (1.5 GHz SM clock) is CTA-dispatch-bound, 49k 64-thread CTAs at 407 MB could lose; the 128-thread path for >450k rows hedges the largest sizes only"
  - "S/M are unchanged code; any S/M move on the portal is noise"
  - "H11 is still untested at 1.5 GHz: if the portal's +15% gap is table traffic at the locked L2 clock, this kernel recovers only the CTA-shape part"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```