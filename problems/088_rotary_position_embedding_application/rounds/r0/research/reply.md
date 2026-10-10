Everything needed is in hand: the kernel passes all 16 workloads, and the probes settled the first-round questions (store policy, CTA size, layout alternatives, dispatch cost). Below are the problem card, the ledger and the first kernel.

```markdown card
# Problem card: L1 #88 `088_rotary_position_embedding_application`

Research round r0, 10 October 2026. No portal result yet; Tb and Tsol per workload are unknown until the first
submission. All timings below are from the rented B200 (unlocked clocks, harness CUPTI methodology, cold L2).

## 1. Semantics

Reference (from HunyuanImage-2.1): `query_rotated = query * cos + rotate_half(query) * sin`, same for key, with
`rotate_half(x) = cat(-x[..., 64:], x[..., :64])`. Everything fp32.

- DPS signature: `run(query, key, cos, sin, query_rotated, key_rotated)`.
- `query`: `[B, H, S, 128]`, `key`: `[B, Hkv, S, 128]`, `cos`/`sin`: `[S, 128]`, all fp32 contiguous. Outputs have
  the input shapes. No scalars.
- Per element, for column i < 64 of a row at sequence position s:
  `out[i] = x[i]·cos[s,i] − x[i+64]·sin[s,i]` and `out[i+64] = x[i+64]·cos[s,i+64] + x[i]·sin[s,i+64]`.
  Each 512 B row is independent; the only cross-element dependence is between columns i and i+64 of the same row.
- Row view: `R = B·(H+Hkv)·S` rows of 128 floats, Q rows first then K rows. Row `rl` of a stream sits at sequence
  position `s = rl % S`, so cos/sin row `s` is shared by `B·H` (or `B·Hkv`) rows that are `S·512` B apart.
- `cos` and `sin` are generated independently as `cos(randn.clamp(-2,2))` and `sin(randn.clamp(-2,2))`
  [PAPER: io.py `_is_rope_cos_sin`]: values in [−1, 1], **no** cos²+sin²=1 and no symmetry between the halves. The
  kernel must read all 128 columns of both tables.
- Traffic per call: read Q and K once, write both outputs once, plus the two tables:
  `bytes = 2·R·512 + 2·S·512`. The tables are at most 8 MB (S = 8192) and are re-read from L2: 2 B of table per
  1 B of x read.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|` on ≥ 99% of elements, no NaN/Inf, not all zero
  [PAPER: workload.jsonl, correctness.py]. One fp32 ulp is 1.19e-7 relative, so the budget is about 84 ulp.
- The reference does two rounded products and one rounded add per element. Our kernel contracts into
  `fma(x, cos, −x2·sin)` under `--use_fast_math`: the difference is ≤ 2 ulp of the larger product. Measured max
  abs difference against the reference: 2.4e-7 [probe_b200].
- Cancellation is harmless: when the two products nearly cancel, the absolute error stays at the ulp of the products
  (≤ about 5e-7 for |x| ≤ 5), well under the 1e-5 absolute term.
- Nothing may be stored or computed below fp32. There are no reductions, rsqrt or divisions.

## 3. The 16 workloads (sorted by bytes)

`floor8` = bytes / 8 TB/s. "copy" = the plain read+write stream of the same bytes on the rented B200 (from the
briefing). "ref" = the PyTorch reference timed like the harness. "r0" = our first kernel (run_tests, same B200).
CTAs: 64-thread path has 8 rows per CTA, 256-thread path 32 rows.

| B,H,Hkv,S | R rows | Q rows | MB | FLOP M | floor8 µs | copy µs | ref µs | ref/copy | r0 µs | r0/copy | band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,8,8,128 | 2,048 | 1,024 | 2.23 | 0.8 | 0.28 | 2.47 | 44.1 | 17.9 | 3.0 | 1.23 | S |
| 1,40,8,131 | 6,288 | 5,240 | 6.57 | 2.4 | 0.82 | 3.46 | 49.0 | 14.2 | 3.6 | 1.04 | S |
| 2,32,4,211 | 15,192 | 13,504 | 15.77 | 5.8 | 1.97 | 5.00 | 65.0 | 13.0 | 4.9 | 0.98 | S |
| 4,24,6,293 | 35,160 | 28,128 | 36.30 | 13.5 | 4.54 | 8.13 | 90.4 | 11.1 | 7.5 | 0.93 | S |
| 1,56,8,853 | 54,592 | 47,768 | 56.78 | 21.0 | 7.10 | 11.46 | 120.9 | 10.6 | 10.4 | 0.91 | S |
| 8,16,4,373 | 59,680 | 47,744 | 61.49 | 22.9 | 7.69 | 12.31 | 131.7 | 10.7 | 11.3 | 0.92 | M |
| 2,48,6,919 | 99,252 | 88,224 | 102.58 | 38.1 | 12.82 | 18.83 | 204.0 | 10.8 | 16.6 | 0.88 | M |
| 64,8,8,128 | 131,072 | 65,536 | 134.35 | 50.3 | 16.79 | 23.96 | 248.7 | 10.4 | 20.0 | 0.83 | M |
| 8,24,8,1024 | 262,144 | 196,608 | 269.48 | 100.7 | 33.69 | 44.94 | 489.4 | 10.9 | 38.8 | 0.86 | M |
| 8,28,4,1024 | 262,144 | 229,376 | 269.48 | 100.7 | 33.69 | 44.89 | 489.5 | 10.9 | 39.6 | 0.88 | M |
| 8,32,4,1087 | 313,056 | 278,272 | 321.68 | 120.2 | 40.21 | 52.93 | 579.9 | 11.0 | 51.3 | 0.97 | L |
| 4,48,12,1321 | 317,040 | 253,632 | 326.00 | 121.7 | 40.75 | 53.72 | 582.3 | 10.8 | 51.6 | 0.96 | L |
| 4,36,6,2048 | 344,064 | 294,912 | 354.42 | 132.1 | 44.30 | 57.93 | 630.8 | 10.9 | 56.3 | 0.97 | L |
| 2,40,8,4096 | 393,216 | 327,680 | 406.85 | 151.0 | 50.86 | 65.54 | 717.0 | 10.9 | 63.2 | 0.96 | L |
| 1,64,8,8192 | 589,824 | 524,288 | 612.37 | 226.5 | 76.55 | 97.95 | 1055.7 | 10.8 | 95.8 | 0.98 | L |
| 32,16,4,4096 | 2,621,440 | 2,097,152 | 2688.55 | 1006.6 | 336.07 | 413.22 | 4564.8 | 11.0 | 400.0 | 0.97 | L |

Geometric means over the 16 workloads: floor8 14.2 µs, plain copy about 21 µs, reference 284 µs, r0 23.5 µs.
Unlike #38, the shapes do not collapse into a few sizes: `R` differs for all 16, and the Q/K split varies
(K is 11–50% of the rows). Dispatch keys on `R` (and `S` for the table size).

## 4. Bounds and what dominates per band

- Arithmetic intensity is 3 FLOP per 8 B of DRAM traffic (0.375 FLOP/B). At 57 TFLOP/s fp32 SIMT the biggest
  workload needs 18 µs of maths against 336 µs of bytes. Every workload is memory-bound; the SOL time is almost
  surely `bytes / peak BW`, so Tsol ≈ floor8 (maybe without the cos/sin bytes, which are < 1.5% of traffic except at
  2.2 MB where they are 6%).
- Extra on-chip traffic: cos/sin rows come from L2 at 2 B per 1 B of x read, about 4 TB/s of L2 reads when DRAM runs
  at 7 TB/s. L2 serves it (21 TB/s), and the probe where two heads share one table load was not faster, so it is not
  on the critical path yet [probe_b200].
- S band (2–57 MB): fixed cost. An empty grid of the same CTA count spans 1.76 µs at 2.2 MB; our kernel takes 2.6–3.0
  µs there against a 0.28 µs byte floor. The smallest workload has only 2,048 rows: 256 CTAs of 64 threads, under two
  per SM, so the first DRAM round trip and the tail are the whole time.
- M band (61–269 MB): mixed. r0 is already 12–17% faster than the plain copy because evict_last stores keep part of the
  output write-back outside the window (the #38 H1 effect carries over). This is where the copy baseline loses most.
- L band (322–2689 MB): sustained bandwidth, 6.3–6.7 TB/s in r0 (unlocked clocks). CTA dispatch is a real cost here:
  an empty grid of 64-thread CTAs takes 0.55 µs per 1,000 CTAs, 26 µs for the 49k CTAs of the 407 MB workload
  [probe_b200], so the L path uses 256-thread CTAs (32 rows each). 512- and 1024-thread CTAs are 3–8% slower again.

## 5. Where the reference loses and what a fast kernel must do

- The reference is 10–18× the plain copy. It materialises `-x2`, the `cat` (a full extra tensor), `x*cos`,
  `rotate_half(x)*sin` and the sum, for Q and for K: about 5 full passes of read+write per stream, plus about ten
  kernel launches whose gaps dominate at small sizes (44 µs for 2.2 MB).
- A fast kernel: one fused launch over Q and K; each thread holds both halves of its column pair (lane j owns
  columns 8j..8j+7 and 64+8j..64+8j+7), so no shuffles and no second pass; 256-bit loads and stores; per-row
  cos/sin fetched from L2; evict_last on the output stores; CTA size chosen per row count.
- Measured on the rented B200 (within one probe session; sessions differ by up to 10%, compare only within a line):

| design knob | 36 MB | 61 MB | 269 MB | 407 MB |
|---|---|---|---|---|
| 256 thr, 1 row/thread, plain stores | 8.02 | 11.48 | 42.10 | 61.93 |
| 256 thr, 1 row/thread, evict_last stores | 7.11 | 10.49 | 39.05 | 56.73 |
| 64 thr, 1 row/thread, evict_last | 6.94 | 10.18 | 38.77 | 59.75 |
| 128 thr, 1 row/thread, evict_last | 7.07 | 10.13 | 38.82 | 59.66 |
| 256 thr, 2 rows/thread (108 regs) | 7.75 | 11.49 | 43.43 | 65.09 |
| float4 layout, 16 lanes/row (32 regs), 128 thr | 6.85 | – | 38.85 | 59.83 |
| 2 heads per thread sharing cos/sin (80 regs), 256 thr | 7.04 | – | 38.39 | 60.96 |

## 6. Design priorities

1. Keep: one launch, evict_last output stores (−7 to −11%), 256-bit accesses, dispatch on `R`
   (64-thread CTAs up to 262,144 rows, 256-thread above).
2. S band fixed cost: 2.6–3.0 µs against a 1.76 µs empty-grid span and a 0.28 µs byte floor. Try fewer, fatter
   CTAs only if they issue all loads up front; test x-load hints (`.nc`/`L1::no_allocate` vs plain) and whether
   the `rl % S` modulo and the 60-register body cost anything at the locked 1.5 GHz.
3. L band bandwidth: 6.3–6.7 TB/s. Levers: register count (60 → 4 CTAs/SM at 256 threads; the float4 layout at 32
   regs ties, so occupancy is not the limit yet), the DRAM page pattern of the Q/K split, and a grid that avoids the
   dispatch-rate ceiling without going to 512-thread CTAs.
4. M band: already 12–17% below the plain copy. Check whether evict_last on only the first output or a fraction is
   better (#38 H15 said no; verify here since bytes differ).
5. Do not spend slots on: TMA/bulk rings, clusters, multi-row-per-thread register tiles (108 regs lost everywhere),
   or sharing cos/sin across heads (neutral).

## 7. Results so far

- `r0-rope-ldg256-os-stel` (CUDA C++, 60 regs, 0 smem, no spills): 16/16 workloads pass on the B200 test bench.
  Times in §3: geomean 23.5 µs, 0.83–0.98× the plain copy except 1.23× at 2.2 MB and 1.04× at 6.6 MB.
- First submission to the portal will give Tb and Tsol per workload and unlock the score model.

## 8. Open questions

- Does SOLAR count the cos/sin bytes and the real 8.18 TB/s, or something lower like #38's 0.53× floor?
- Why do probe sessions differ by 10% at M/L (39.05 vs 42.8 µs at 269 MB for the same binary)? Clock state on
  the rented card; the portal locks clocks, so only within-session comparisons are trusted.
- Is the smallest workload (2.2 MB, 2,048 rows) better served by 4-row or 1-row CTAs, or by fewer CTAs with
  several rows in flight per thread?
- Does the 32-byte alignment of the shifted pointers always hold (the harness shifts by multiples of 256 B; the
  binding checks it and raises otherwise)?
```

```yaml ledger
# Hypothesis ledger for #88 (RoPE application), seeded 10 October 2026 by the r0 research session.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
hypotheses:
- id: H1
  statement: evict_last on the output stores cuts in-window dirty write-backs, as in #38, so M sizes gain most.
  status: supported
  evidence:
  - 'probe r0: EL vs plain stores, same kernel: 36 MB 7.11 vs 8.02 us, 61 MB 10.49 vs 11.48, 269 MB 39.05 vs 42.10, 407 MB 56.73 vs 61.93 [probe_b200]'
  - 'run_tests r0: 0.83-0.88x the plain copy at 61-269 MB'
- id: H2
  statement: CTA dispatch rate (about 0.55-0.65 us per 1000 CTAs regardless of 64 or 256 threads) caps 64-thread one-shot grids at L, so 256-thread CTAs win there while 64-thread CTAs win at S/M.
  status: supported
  evidence:
  - 'probe r0: empty grids 64x32768 18.0 us, 64x49152 26.4 us, 256x8192 5.3 us, 256x12288 7.4 us, 256x81920 43.5 us'
  - 'probe r0: 64-thr vs 256-thr: 36 MB 6.94 vs 7.11, 269 MB 38.77 vs 39.05, 407 MB 59.75 vs 56.73'
  - 'probe r0: 512 and 1024 threads per CTA are 3-8% slower than 256 at 269-2689 MB'
- id: H3
  statement: Sharing one cos/sin load across several heads (rows S*512 B apart) cuts L2 traffic and helps L.
  status: refuted
  evidence:
  - 'probe r0: 2-head variant (80 regs): 269 MB 38.39 vs 39.05 (noise), 407 MB 60.96 vs 56.73; the strided row pairs seem to cost more than the saved L2 reads'
- id: H4
  statement: Occupancy (60 regs -> 4 CTAs/SM at 256 threads) limits L-band bandwidth.
  status: refuted
  evidence:
  - 'probe r0: float4 layout with 32 regs (full occupancy) ties the 60-reg v8 layout at 36 and 269 MB and is 5% slower at 407 MB'
- id: H5
  statement: More rows per thread (register tiles) raise bytes in flight and help L.
  status: refuted
  evidence:
  - 'probe r0: 2 rows/thread at 256 threads (108 regs): slower at every size (+9% at 36 MB, +15% at 407 MB)'
- id: H6
  statement: The S-band fixed cost (2.6-3.0 us at 2.2 MB vs 1.76 us for an empty grid) can be cut by issuing all of a CTA's loads before any use with fewer CTAs, or by 1-4 row CTAs.
  status: open
  evidence:
  - 'probe r0: 64/128/256-thread one-row-per-thread variants are within 0.15 us of each other at 2.2 MB (2.62-2.77 us)'
- id: H7
  statement: Load hints on x (.nc, L1::no_allocate) and on cos/sin (.nc) change nothing measurable; plain loads would do.
  status: open
  evidence:
  - 'r0 uses .nc + L1::no_allocate on x and .nc on tables without an A/B'
- id: H8
  statement: The per-row integer modulo (rl % S) and the 60-register body are cheap at 1.5 GHz; a (stream, bh, s-tile) grid that avoids the modulo gains nothing.
  status: open
  evidence:
  - 'probe r0: the 2-head variant uses a modulo-free grid and was not faster'
- id: H9
  statement: Partial or first-output-only evict_last is no better than evict_last on all stores (as #38 H15), since the gain saturates at about 25-30 MB of parked write-back.
  status: open
  evidence: []
- id: H10
  statement: SOLAR for this problem is bytes/8 TB/s including cos/sin, so scores at the floor would be about 0.75-0.8 as in #38.
  status: open
  evidence:
  - 'unknown until the first portal submission; floor8 geomean 14.2 us, r0 geomean 23.5 us on the unlocked rented card'
```

### Rationale

New problem, empty archive. The first kernel is the design the probes ranked best: a fused one-shot CUDA C++ kernel, eight lanes per row so the rotate-half partner is in the same thread, 256-bit loads and stores, evict_last stores, 64-thread CTAs up to 262,144 rows and 256-thread above. It passes all 16 workloads on the B200 bench. It is submitted to obtain the hidden baseline and SOL anchors.

```json solution-spec
{"name": "r0-rope-ldg256-os-stel", "definition": "088_rotary_position_embedding_application", "author": "solx-loop", "description": "One-shot fused RoPE for Q and K: 8 lanes per 128-float row, 256-bit loads/stores, cos/sin from L2 per row, evict_last output stores, 64-thread CTAs for small row counts and 256-thread CTAs above.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// RoPE application (Q and K), fp32, head_dim 128. One fused one-shot launch.
// Layout: 8 lanes per row; lane j owns columns [8j, 8j+8) and [64+8j, 64+8j+8) (one 256-bit load each),
// so the rotate_half partner is already in the same thread and no shuffles are needed.
// out[i]    = x[i]*cos[i]    - x[i+64]*sin[i]        (i < 64)
// out[i+64] = x[i+64]*cos[i+64] + x[i]*sin[i+64]
#include <cuda_runtime.h>
#include <stdint.h>

#define LD8(p, v) asm volatile("ld.global.nc.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p))
#define LDX8(p, v) asm volatile("ld.global.nc.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p))
#define ST8EL(p, v, pol) asm volatile("st.global.L2::cache_hint.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;" \
    :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]), "l"(pol) : "memory")

template <int THREADS>
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
    LD8(cp, cl);  LD8(cp + 64, ch);
    LD8(sp, sl);  LD8(sp + 64, sh);
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
    if (R <= 262144u) {            // S/M band: 8 rows per 64-thread CTA (many small CTAs)
        const unsigned grid = (R + 7) / 8;
        rope_kernel<64><<<grid, 64, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
    } else {                       // L band: 32 rows per 256-thread CTA
        const unsigned grid = (R + 31) / 32;
        rope_kernel<256><<<grid, 256, 0, stream>>>(q, k, cs, sn, qo, ko, Rq, R, S);
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
id: r0-rope-ldg256-os-stel
parents: []
operation: new_design
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
  RoPE is a pure stream with a per-row L2-resident cos/sin table; a fused one-shot kernel where each thread owns
  both halves of its column pair (no shuffles, one pass), with 256-bit accesses and evict_last stores, runs at the
  copy ceiling, and the CTA size must follow the row count because CTA dispatch (0.55 us per 1000 CTAs) caps
  64-thread grids on the L band.
expected_effect:
  S: {pct: 0, confidence: medium}
  M: {pct: 0, confidence: medium}
  L: {pct: 0, confidence: medium}
resources_sm100a:
  regs_per_thread: 60
  smem_per_cta_bytes: 0
  threads_per_cta: 64
  ctas_per_sm: 16
  bytes_in_flight_per_sm: 131072
  launches_per_call: 1
knobs: {THREADS_S: 64, THREADS_L: 256, ROWS_S: 8, ROWS_L: 32, row_threshold: 262144, store_hint: evict_last}
dispatch:
  - {max_rows: 262144, kernel: rope_kernel<64>, meta: {threads: 64, rows: 8}}
  - {max_rows: null, kernel: rope_kernel<256>, meta: {threads: 256, rows: 32}}
tests: [H1, H2]
findings:
  - "run_tests: 16/16 pass; 3.0 us at 2.2 MB (1.23x copy), 0.83-0.88x copy at 61-269 MB, 0.96-0.98x copy at 322-2689 MB [run_tests]"
  - "evict_last stores vs plain, same kernel: -11% at 36 MB, -9% at 61 MB, -7% at 269 MB, -8% at 407 MB [probe_b200]"
  - "empty grid spans: 1.76 us for 256 CTAs; 0.55-0.65 us per 1000 CTAs at 64 or 256 threads (18 us for 32768x64, 43.5 us for 81920x256); CTA dispatch caps 64-thread grids at L [probe_b200]"
  - "64-thread CTAs beat 256-thread by 2-3% up to 269 MB; 256-thread beats 64 by 5% at 407 MB; 512/1024-thread CTAs 3-8% slower than 256 at L [probe_b200]"
  - "dead end: 2 rows per thread (108 regs) +9..+15% everywhere; sharing cos/sin across 2 heads (80 regs) neutral at 269 MB, +7% at 407 MB [probe_b200]"
  - "float4 layout with 16 lanes/row (32 regs) ties the 60-reg 256-bit layout at 36 and 269 MB, 5% slower at 407 MB: occupancy is not the limiter [probe_b200]"
  - "bug found: indexing K rows with the global row index read/wrote past the key tensors; fixed by using the in-stream row index [probe_b200]"
  - "cos/sin are independent cos(randn)/sin(randn) tables in [-1,1], all 128 columns needed; harness timing sessions on the rented card drift up to 10% at M/L between probes, compare within a session only [probe_b200]"
paths:
  - {max_tokens: null, max_rows: 262144, lang: cuda, width: 256, threads: 64, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
  - {max_tokens: null, max_rows: null, lang: cuda, width: 256, threads: 256, rows: 32, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: nc_noalloc, w: nc, st: el}
runs_on:
  H200: {runs: false, representative: false, note: "256-bit ld/st.global.v8.f32 and createpolicy need sm_100a; does not compile for sm_90a"}
  A100: {runs: false, representative: false}
  L40S: {runs: false, representative: false}
risks:
  - "60 registers: 4 CTAs/SM on the 256-thread path (1024 threads); fine so far but limits any multi-row extension"
  - "binding raises if a pointer is not 32-byte aligned (harness shifts are multiples of 256 B, so this should not trigger)"
  - "the 2.2 MB workload is 1.23x the plain copy: fixed cost, not bandwidth; S-band specialist still to be found"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```