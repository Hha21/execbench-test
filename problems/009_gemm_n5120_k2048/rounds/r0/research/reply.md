Everything I need is measured. I'm writing up the problem card, the ledger and the first kernel now.

```markdown card
# Problem card: FlashInfer-Bench `009_gemm_n5120_k2048`

Written 2026-10-08 from the definition, the harness run on the rented B200 (unlocked clocks, 1965 MHz) and probes.
No portal result yet, so Tb and Tsol per workload are unknown; the first submission (the kernel below) anchors them.

## 1. Semantics

Reference (verbatim): `C = torch.matmul(A, B.T)`, captured from Qwen3-30B-A3B `attn.qkv_proj`.

- DPS signature: `run(A, B, C)`. `A` is `[M, 2048]` fp16, `B` is `[5120, 2048]` fp16, `C` is `[M, 5120]` fp16, all
  contiguous (row-major, K innermost for both inputs: a "TN" GEMM, the ideal layout for every tensor-core path).
- `C[m, n] = sum_k A[m, k] * B[n, k]`, fp32 accumulation (cuBLAS default; `allow_fp16_reduced_precision_reduction`
  is True in this torch, but the single nvjet kernels seen here do not split K).
- Only M varies (1 to 16294). N = 5120 = 40 x 128 = 20 x 256; K = 2048 = 32 x 64. Neither N nor K needs masking;
  M does (16294 = 127 x 128 + 38).
- Traffic per call: `20,971,520 + M x 14,336` bytes (B is 20.97 MB and is read once; A and C add 14 KB per row).
  FLOPs: `M x 20,971,520` (2 x M x N x K).

## 2. Numerics and tolerance

- The harness check is `|out - ref| <= max_atol + max_rtol * |ref|` on >= 99% of elements (`correctness.py`); the
  exact atol/rtol for this definition were not visible from the B200 box, but cuBLAS's own output and our mma.sync
  kernel (fp32 accumulate, different summation order, max |diff| = 1 fp16 ulp at |C| ~ 100-180) both pass all 25
  workloads over 10 random rounds [run_tests].
- Inputs are `randn` fp16, so |C| reaches ~200 (sqrt(2048) x 4); the fp16 ulp there is 0.125. Any fp16-input,
  fp32-accumulate path (mma.sync, tcgen05, cuBLAS) is within tolerance. Accumulating in fp16 would not be; FP8/FP4
  or TF32-style truncation of the inputs is a precision downgrade and forbidden by the rules.
- Every element of C must be written (outputs arrive zeroed).

## 3. The 25 workloads

`bytes` = A + B + C. `floor8` = bytes / 8 TB/s. `tc1.5` = FLOPs / 1.82 PFLOP/s (148 SMs x 8192 FP16 FLOP/clk at the
portal's 1500 MHz lock; the datasheet's 2.25 PFLOP/s would give 0.81x these). `cuBLAS` and `r0` are harness
(CUPTI, cold L2, shifted pointers) times on the rented B200 at 1965 MHz [run_tests]. `copy` is the harness's plain
copy of the same bytes.

| M | band | bytes MB | GFLOP | floor8 us | tc1.5 us | copy us | cuBLAS us | r0 us | cuBLAS kernel (nvjet) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S | 20.99 | 0.02 | 2.62 | 0.01 | 7.3 | 7.6 | **6.9** | (gemv, not captured) |
| 2 | S | 21.00 | 0.04 | 2.62 | 0.02 | 7.2 | 7.8 | **6.9** | 64x8_64x16 |
| 4 | S | 21.03 | 0.08 | 2.63 | 0.05 | 7.2 | 7.8 | **7.0** | 64x8 |
| 5 | S | 21.04 | 0.10 | 2.63 | 0.06 | 7.2 | 8.0 | **6.9** | 64x8 |
| 6 | S | 21.06 | 0.13 | 2.63 | 0.07 | 7.2 | 7.9 | **7.0** | 64x8 |
| 8 | S | 21.09 | 0.17 | 2.64 | 0.09 | 7.2 | 7.8 | **7.0** | 64x8 |
| 16 | S | 21.20 | 0.34 | 2.65 | 0.18 | 7.2 | 7.9 | **7.2** | 64x24 |
| 17 | S | 21.22 | 0.36 | 2.65 | 0.20 | 7.2 | 7.9 | 7.8 | 64x24_64x16 |
| 25 | M | 21.33 | 0.52 | 2.67 | 0.29 | 7.2 | 8.5 | 8.5 | 128x16_64x11 |
| 32 | M | 21.43 | 0.67 | 2.68 | 0.37 | 7.0 | 7.9 | 7.9 | 64x32_64x16 |
| 34 | M | 21.46 | 0.71 | 2.68 | 0.39 | 7.0 | 8.6 | 8.6 | 128x24_64x11 |
| 63 | M | 21.87 | 1.32 | 2.73 | 0.73 | 7.3 | 7.9 | 8.0 | 40x64_64x16 |
| 64 | M | 21.89 | 1.34 | 2.74 | 0.74 | 7.2 | 8.0 | 8.0 | 40x64_64x16 |
| 93 | M | 22.30 | 1.95 | 2.79 | 1.07 | 7.4 | 8.9 | 8.7 | 80x64 2cta |
| 128 | M | 22.81 | 2.68 | 2.85 | 1.47 | 7.4 | 8.3 | 8.3 | 80x64 2cta |
| 172 | M | 23.44 | 3.61 | 2.93 | 1.98 | 7.5 | 10.3 | 10.3 | 128x64_64x8 |
| 289 | L | 25.11 | 6.06 | 3.14 | 3.33 | 7.8 | 11.2 | 11.1 | 128x104_64x7 |
| 492 | L | 28.02 | 10.3 | 3.50 | 5.67 | 8.2 | 11.8 | 11.7 | 144x128 2cta |
| 952 | L | 34.62 | 20.0 | 4.33 | 11.0 | 8.9 | 17.9 | 17.9 | 256x136_64x4 |
| 8828 | L | 147.5 | 185 | 18.4 | 101.7 | 24.3 | 108.6 | 111.2 | 128x256_64x6 2cta |
| 11006 | L | 178.8 | 231 | 22.3 | 126.8 | 28.7 | 130.6 | 131.2 | 128x256 2cta |
| 12251 | L | 196.6 | 257 | 24.6 | 141.2 | 31.6 | 142.2 | 145.7 | 128x256 2cta |
| 12853 | L | 205.2 | 270 | 25.7 | 148.1 | 33.2 | 151.9 | 155.3 | 128x256 2cta |
| 14915 | L | 234.8 | 313 | 29.3 | 171.9 | 37.1 | 174.8 | 178.7 | 128x256 2cta |
| 16294 | L | 254.6 | 342 | 31.8 | 187.8 | 40.0 | 194.9 | 196.0 | 128x256 2cta |

The loop's S/M/L bands (by bytes) do not follow the physics. The useful bands are:

| Regime | M | Workloads | Character |
|---|---|---|---|
| GEMV-like | 1-16 | 7 | stream B once; compute is < 0.2 us even on mma.sync |
| skinny | 17-172 | 9 | still B-stream bound (compute <= 2 us at 1.5 GHz) but A re-use and tile shape decide |
| mixed | 289-952 | 3 | compute floor 3-11 us vs memory floor 3-4 us: tensor cores must overlap the stream |
| compute | 8828-16294 | 6 | cuBLAS at ~93% of the 1.82 PF floor on the rented box; tcgen05 is mandatory |

## 4. Bounds and what dominates

- **Memory-bound up to M ~ 300** (crossover of `tc1.5` and `floor8` at M ~ 270; at the datasheet peak ~ 350).
  19 of 25 workloads are in the memory/latency regime where the whole job is "read 21 MB of B as fast as the
  harness allows".
- **The harness read floor for 21 MB is 6.4 us, not 2.6 us.** A 256-bit read-only kernel takes 6.4-6.7 us for any grid
  shape (320 x 64 KB to 2560 x 8 KB CTAs) [probe]. The empty-kernel CUPTI floor is 1.5 us. CUDA-event timing with a
  read-only flush instead of the harness's zero-fill flush is 2.1 us faster, so about 2 us of the 6.4 is write-back of
  the dirty L2 residue left by the flush (same mechanism as #38's H7). `L2::evict_first` on the loads changes nothing
  [probe]. So the practical S-band floor is ~6.4 us, ~4.3 us if the residue could ever be dodged.
- **Compute-bound at M >= 8828.** cuBLAS runs 1.75 PFLOP/s at 1965 MHz (73% of the 2.38 PF peak at that clock). The
  CuTe DSL `dense_gemm_persistent` example (256x256 tile, cluster 2x1, 2-CTA MMA) reaches 195 us at M=16294 vs
  cuBLAS 199 us: the same ceiling [probe]. At the portal's 1500 MHz the FP16 peak is 1.82 PF, so these 6 workloads
  cannot go below ~102-188 us; whether cuBLAS slows by the full 1.31x there is unknown.
- **SOLAR Tsol guess**: `max(FLOPs / 2.25 PF, bytes / 8 TB/s)`: ~2.6 us for M <= 172, 3.1-8.9 us for 289-952, and
  82-152 us for the compute band. If Tb is cuBLAS, the S/M-band score ceiling at the 6.4 us floor is about
  `5.2 / (3.8 + 5.2) = 0.58`, and 0.55 at 6.9 us; large-M scores stay near 0.5 unless cuBLAS is beaten.

## 5. Where the reference loses, and what a fast kernel must do

- At M <= 16 cuBLAS picks small-tile nvjet kernels (64x8, 64x24) that are latency-limited: 7.8 us, 1.4 us above the
  read floor. A 640-CTA stream of 256-bit loads straight into `mma.sync` fragments gets 6.9-7.2 us (r0 kernel).
- At M = 17-172 cuBLAS changes tiles every few rows (64x24, 128x16, 40x64, 80x64 2cta, 128x64) and loses up to 2.8 us
  (M=172: 10.3 us). Our naive extension (each warp re-reads A tiles from L2 per 16-row block) is far worse
  (10.8 us at M=17, 19.5 us at M=64): with 640 CTAs the A traffic is `640 x M x 4 KB` (164 MB at M=64) through L2
  [probe]. A fast kernel here needs wider N tiles (32-64 rows of B per CTA, 80-160 CTAs) with A staged once per
  k-chunk in shared memory, and split-K across a 2-4 CTA cluster (DSMEM reduction) to get back to >= 300 CTAs and
  >= 43 KB in flight per SM.
- At M = 289-952 cuBLAS is 2-4x above the memory floor and 1.6-3.4x above the compute floor: the tile is too large
  for the grid (e.g. 256x136 at M=952 gives 4 x 38 = 152 CTAs, one wave with a long K loop). A persistent tcgen05
  kernel with 128x128 or 128x64 tiles that keeps B streaming should land near max(memory, compute) + fixed cost.
- At M >= 8828 the job is to match cuBLAS/CUTLASS (same ceiling) and shave the wave quantization: 40 x 128 or
  20 x 256 N-tiles times ceil(M/128) M-tiles over 148 SMs.
- Triton `tl.dot` kernels are 1.5-2.4x slower than cuBLAS at every M tried (pointer loads, 128-512 x 16-256 tiles)
  [probe]; use CUDA C++ (mma.sync for M <= 64, raw tcgen05 or CUTLASS/CuTe DSL above).
- mma.sync FP16 throughput on B200 is only ~550 TFLOP/s (1/4 of tcgen05) [probe]: fine up to M ~ 64 (2.5 us of MMA
  overlapped with the 6 us stream), marginal at 93-172, hopeless above.

## 6. Design priorities

1. **Skinny band M = 17-172 (9 workloads)**: a CUDA kernel with BN = 32-64 B-rows per CTA, cluster split-K (2-4) with
   DSMEM reduction, A chunk in smem, mma.sync (M <= 64) or tcgen05 (M > 64). Target 7.0-7.5 us vs cuBLAS 7.9-10.3.
2. **GEMV band M <= 16 (7 workloads)**: trim the 0.6 us tail of r0 (reduction after the last load, 640/148 = 4.3
   CTAs per SM imbalance): try 1184-CTA grids with cluster split-K, or 148 x k persistent with all loads up front.
3. **Mixed band 289-952 (3 workloads)**: a tcgen05 persistent kernel with 128-row tiles; expected 1.5-2x over cuBLAS.
4. **Compute band (6 workloads)**: CUTLASS/CuTe DSL 2-CTA kernel with raster/tile tuned to N=5120; expect +-3%.
5. The dirty-L2 residue (~2 us of every small workload) is the biggest single pool but no legitimate kernel-side
   mechanism is known (#38 ledger H7/H8); only revisit if a new idea appears.

## 7. Results so far

r0 (`r0-skinny16-cublas`): M <= 16 served by the mma.sync skinny kernel (46 regs, no spills, 12 x LDG.256 per lane,
640 CTAs x 256 threads), everything else by cuBLAS through `at::matmul_out`. 25/25 pass. Times on the rented B200
[run_tests]: M <= 16 6.9-7.2 us (cuBLAS 7.6-7.9, -10%); M = 17..952 identical to cuBLAS (7.8-17.9 us); large M
111-196 us (cuBLAS 109-195, within noise). Skinny-kernel knobs tried: 8 warps x 4 loads per lane best; 4 warps
(8.9 us), 16 warps (7.7-8.4), BN=16 (7.5-7.7), evict_first loads (no change), A loads before B loads (+0.3 us).

## 8. Open questions

- The exact atol/rtol of this definition (both cuBLAS and mma.sync pass; relevant only if a reduction order gets
  sloppy).
- Tb per workload: is the hidden baseline exactly cuBLAS, and how much slower is cuBLAS at the 1500 MHz lock on the
  compute band? (first portal submission)
- Tsol per workload: FLOPs/2.25 PF or bytes/8 TB/s? (derive from the submission page)
- Whether tcgen05 can be driven from raw PTX in a small kernel for M = 64-952 without the CUTLASS machinery, and the
  CuTe DSL compile time budget (20 s per kernel, 300 s limit).
- Whether a 2-CTA cluster DSMEM split-K reduction costs less than the 0.3-0.6 us it is meant to save at M <= 16.
```

```yaml ledger
# Hypothesis ledger for FlashInfer-Bench/009_gemm_n5120_k2048, seeded 2026-10-08 by the research phase (r0).
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
- 'fp16 inputs with fp32 accumulation only: no FP8/FP4/TF32-style truncation of A or B, no fp16 accumulation.'
hypotheses:
- id: H1
  statement: For M <= 172 the time is the cost of streaming B (21 MB) under the harness, about 6.4 us (read floor incl. ~2 us dirty-L2 write-back), plus a kernel-specific tail; cuBLAS sits 1.4-3.9 us above that.
  status: supported
  evidence:
  - 'probe r0: 256-bit read-only kernel 6.4-6.7 us for every grid shape; empty kernel 1.5 us; read-only flush is 2.1 us faster than the zero flush (event timing)'
  - 'run_tests r0: cuBLAS 7.6-10.3 us at M <= 172'
- id: H2
  statement: At M <= 16 a one-shot 640-CTA mma.sync kernel with 256-bit loads straight into fragments beats cuBLAS by ~10%; the residual 0.6 us is the post-load tail plus 640/148 CTA imbalance.
  status: supported
  evidence:
  - 'run_tests r0: 6.9-7.2 us vs cuBLAS 7.6-7.9'
  - 'probe r0: 8 warps x 4 loads/lane best; 4 warps 8.9 us, 16 warps 7.7-8.4 us, BN=16 7.5-7.7 us; tail/imbalance split not yet measured'
- id: H3
  statement: For 17 <= M <= 172 the winning structure is BN = 32-64 B-rows per CTA with A staged once per k-chunk in shared memory and split-K across a 2-4 CTA cluster (DSMEM reduction), reaching ~7-7.5 us; re-reading A per warp from L2 does not work.
  status: open
  evidence:
  - 'probe r0: per-warp A re-read (640 CTAs) 10.8 us at M=17, 19.5 us at M=64, 33.7 us at M=128 (A L2 traffic = 640 x M x 4 KB)'
- id: H4
  statement: mma.sync (~550 TFLOP/s on B200) is enough up to M ~ 64 when overlapped with the B stream; from M ~ 93 upward tcgen05 is required to stay memory-bound.
  status: open
  evidence:
  - 'probe r0: mma.sync m16n8k16 fp16 peaks at 553 TFLOP/s (148-1184 CTAs); 2.5 us of MMA at M=64, 6.5 us at M=172'
- id: H5
  statement: Triton tl.dot GEMMs cannot compete at any M on B200; CUDA C++ (mma.sync / tcgen05 / CUTLASS) is the only route.
  status: supported
  evidence:
  - 'probe r0: Triton 8.9-13 us at M <= 64 vs cuBLAS 7.8-8.0; 309 us vs 199 us at M=16294 (best of 5 configs)'
- id: H6
  statement: At M >= 8828 cuBLAS nvjet 128x256 2-CTA kernels are at the same ceiling as CUTLASS (CuTe DSL persistent 256x256 2cta: 195 vs 199 us at M=16294); only wave-quantization and raster tuning for N=5120 can gain 1-3%.
  status: open
  evidence:
  - 'probe r0: CuTe DSL dense_gemm_persistent 195.5 us (256x256, 2x1), 216.7 us (128x256), 252.7 us (128x128 no 2cta); cuBLAS 198.7 us harness'
- id: H7
  statement: On the portal's 1500 MHz lock the compute-band workloads slow by up to 1.31x for both cuBLAS and us (peak 1.82 PF), while M <= 952 barely moves; Tb for large M will be ~1.2-1.3x the rented-box cuBLAS time.
  status: open
  evidence:
  - 'rented box runs 1965 MHz under GEMM load; cuBLAS 1.75 PFLOP/s = 73% of 2.38 PF at that clock (probe r0)'
- id: H8
  statement: SOLAR Tsol = max(FLOPs/2.25 PF, bytes/8 TB/s) ~ 2.6 us for M <= 172, so with Tb = cuBLAS the small-M score ceiling at the 6.4 us floor is ~0.58 per workload (0.55 at 6.9 us), and the compute band is capped near 0.5.
  status: open
  evidence:
  - 'arithmetic from the card; needs the first portal page to confirm Tb and Tsol'
- id: H9
  statement: evict_first on the streamed B loads does not reduce the dirty-L2 write-back cost for a read-dominated 21 MB kernel.
  status: refuted
  evidence:
  - 'probe r0: rd4e 6.46 us vs rd4 6.43 us; skinny k8w8m1e 7.07/7.28 vs 7.07/7.33'
- id: H10
  statement: cuBLAS's small-M nvjet kernels are latency-limited by their small grids (64x8 tiles over N=5120 = 80 CTAs), which is why a 640-CTA stream wins and why 1184-1280 CTAs with cluster split-K may win more.
  status: open
  evidence:
  - 'kernel names from torch.profiler (probe r0); grid sizes inferred, not measured'
- id: H11
  statement: For M = 289-952 a persistent tcgen05 kernel with 128-row tiles that keeps B streaming lands near max(memory, compute floor) + ~3 us, i.e. 1.5-2x faster than cuBLAS (11-18 us).
  status: open
  evidence:
  - 'floors from the card; cuBLAS tiles (128x104, 144x128, 256x136) give one under-filled wave'
```

### Rationale

First kernel for the research phase: correct everywhere, one launch per call, and a 10% win over cuBLAS on the seven M ≤ 16 workloads from a register-resident mma.sync stream of B with 256-bit loads. Every other shape goes to cuBLAS through ATen inside the same C++ extension, so the submission anchors Tb and Tsol per workload while already carrying the skinny path later rounds will extend to M ≤ 172.

```json solution-spec
{"name": "r0-skinny16-cublas", "definition": "009_gemm_n5120_k2048", "author": "solx-loop",
 "description": "M<=16: one-shot 640-CTA mma.sync skinny GEMM streaming B with 256-bit loads straight into fragments (fp32 accumulate); M>16: cuBLAS via at::matmul_out",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// 009_gemm_n5120_k2048: C[M,5120] = A[M,2048] @ B[5120,2048]^T, fp16 in/out, fp32 accumulate.
// Skinny path (M <= 16): one CTA per 8 rows of B (640 CTAs), 8 warps split K into 256-wide slices,
// each lane streams 4 x 32 B of B straight into mma.sync fragments (k permuted consistently for A and B),
// warps' partial sums are reduced through shared memory and written as fp16.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {

constexpr int N_DIM = 5120;
constexpr int K_DIM = 2048;

template <int BN, int WARPS>
__device__ __forceinline__ void skinny_body(const __half* __restrict__ A, const __half* __restrict__ B,
                                            __half* __restrict__ C, int M) {
  constexpr int KS = K_DIM / WARPS;   // k-slice per warp
  constexpr int CH = KS / 64;         // 64-wide k chunks per warp (one 256-bit load per lane each)
  constexpr int NT = BN / 8;          // n8 tiles per CTA
  __shared__ float red[WARPS][16][BN];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * BN;
  const int kw = warp * KS;

  // B fragments: lane (g,t) holds row n0+8*nt+g, k = kw + 64*c + 16*t .. +15
  unsigned bq[CH][NT][8];
#pragma unroll
  for (int c = 0; c < CH; ++c)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const __half* p = B + (long)(n0 + 8 * nt + g) * K_DIM + kw + 64 * c + 16 * t;
      asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(bq[c][nt][0]), "=r"(bq[c][nt][1]), "=r"(bq[c][nt][2]), "=r"(bq[c][nt][3]),
                     "=r"(bq[c][nt][4]), "=r"(bq[c][nt][5]), "=r"(bq[c][nt][6]), "=r"(bq[c][nt][7])
                   : "l"(p));
    }

  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[nt][j] = 0.f;

  const bool r0 = g < M, r1 = (g + 8) < M;
#pragma unroll
  for (int c = 0; c < CH; ++c) {
    unsigned a0[8], a1[8];
    const __half* pa0 = A + (long)g * K_DIM + kw + 64 * c + 16 * t;
    const __half* pa1 = pa0 + 8l * K_DIM;
    if (r0)
      asm volatile("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3]), "=r"(a0[4]), "=r"(a0[5]), "=r"(a0[6]), "=r"(a0[7])
                   : "l"(pa0));
    else {
#pragma unroll
      for (int j = 0; j < 8; ++j) a0[j] = 0u;
    }
    if (r1)
      asm volatile("ld.global.nc.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                   : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3]), "=r"(a1[4]), "=r"(a1[5]), "=r"(a1[6]), "=r"(a1[7])
                   : "l"(pa1));
    else {
#pragma unroll
      for (int j = 0; j < 8; ++j) a1[j] = 0u;
    }
    // virtual k-step s uses elements 4s..4s+3 of each lane's 16-element chunk (same permutation for A and B)
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
            : "r"(a0[2 * s]), "r"(a1[2 * s]), "r"(a0[2 * s + 1]), "r"(a1[2 * s + 1]), "r"(bq[c][nt][2 * s]),
              "r"(bq[c][nt][2 * s + 1]));
  }

  // cross-warp reduction of the [16, BN] fp32 partials
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    red[warp][g][8 * nt + 2 * t] = acc[nt][0];
    red[warp][g][8 * nt + 2 * t + 1] = acc[nt][1];
    red[warp][g + 8][8 * nt + 2 * t] = acc[nt][2];
    red[warp][g + 8][8 * nt + 2 * t + 1] = acc[nt][3];
  }
  __syncthreads();
  for (int e = threadIdx.x; e < 16 * BN; e += WARPS * 32) {
    const int m = e / BN, n = e % BN;
    if (m < M) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS; ++w) s += red[w][m][n];
      C[(long)m * N_DIM + n0 + n] = __float2half_rn(s);
    }
  }
}

__global__ void __launch_bounds__(256) skinny_m16_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                                         __half* __restrict__ C, int M) {
  skinny_body<8, 8>(A, B, C, M);
}

}  // namespace

extern "C" void launch_skinny_m16(const void* A, const void* B, void* C, int M, cudaStream_t stream) {
  skinny_m16_kernel<<<N_DIM / 8, 256, 0, stream>>>(reinterpret_cast<const __half*>(A),
                                                   reinterpret_cast<const __half*>(B),
                                                   reinterpret_cast<__half*>(C), M);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" void launch_skinny_m16(const void* A, const void* B, void* C, int M, cudaStream_t stream);

namespace {
constexpr int64_t N_DIM = 5120;
constexpr int64_t K_DIM = 2048;

bool aligned32(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 31) == 0; }
}  // namespace

// DPS entry: C = A @ B.T, A [M,2048] fp16, B [5120,2048] fp16, C [M,5120] fp16.
void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == torch::kHalf && B.scalar_type() == torch::kHalf && C.scalar_type() == torch::kHalf,
              "fp16 expected");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "2-D tensors expected");
  const int64_t M = A.size(0);
  TORCH_CHECK(A.size(1) == K_DIM && B.size(0) == N_DIM && B.size(1) == K_DIM && C.size(0) == M && C.size(1) == N_DIM,
              "unexpected shape");
  const at::cuda::OptionalCUDAGuard guard(A.device());

  const bool skinny_ok = M >= 1 && M <= 16 && A.is_contiguous() && B.is_contiguous() && C.is_contiguous() &&
                         aligned32(A.data_ptr()) && aligned32(B.data_ptr());
  if (skinny_ok) {
    launch_skinny_m16(A.data_ptr(), B.data_ptr(), C.data_ptr(), static_cast<int>(M),
                      at::cuda::getCurrentCUDAStream().stream());
  } else {
    at::matmul_out(C, A, B.t());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "GEMM C = A @ B.T (DPS)"); }
```

```yaml design-card
id: r0-skinny16-cublas
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: ldg256
  st: direct
  grid: oneshot
  launch: fused
  tile: rows8
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  For M <= 16 the GEMM is a 21 MB stream of B; 640 CTAs that load 32 KB each with 256-bit loads straight into
  mma.sync fragments (k permuted identically for A and B) beat cuBLAS's latency-limited small-tile kernels by ~10%.
  Every other M is served by cuBLAS (at::matmul_out) so the submission anchors Tb and Tsol per workload.
expected_effect:
  S: {pct: -10, confidence: medium}
  M: {pct: 0, confidence: high}
  L: {pct: 0, confidence: high}
resources_sm100a:
  regs_per_thread: 46
  smem_per_cta_bytes: 5120
  threads_per_cta: 256
  ctas_per_sm: 5
  bytes_in_flight_per_sm: 139264
  launches_per_call: 1
knobs: {BN: 8, WARPS: 8, CH: 4, M_MAX_SKINNY: 16}
dispatch:
  - {max_tokens: 16, kernel: skinny_m16_kernel, meta: {BN: 8, WARPS: 8}}
  - {max_tokens: null, kernel: at::matmul_out (cuBLAS nvjet), meta: {}}
tests: [H1, H2, H8]
findings:
  - "cuBLAS launches one nvjet kernel per call at every M; 7.6-10.3 us for M <= 172, 11-18 us for 289-952, 109-195 us for M >= 8828 on the rented B200 at 1965 MHz [run_tests, probe]"
  - "256-bit read-only kernel over the 21 MB of B: 6.4-6.7 us for any grid shape (320-2560 CTAs); empty kernel 1.5 us; read-only flush instead of the zero flush saves 2.1 us (event timing), so ~2 us is dirty-L2 write-back [probe_b200]"
  - "evict_first on B loads: no change (6.46 vs 6.43 us read-only; 7.07 vs 7.07 skinny) [probe_b200]"
  - "mma.sync m16n8k16 fp16 peaks at ~553 TFLOP/s on B200 (1/4 of tcgen05) [probe_b200]"
  - "skinny kernel for M<=16: 8 warps x 4 x 32 B per lane best (6.9-7.2 us); 4 warps 8.9 us, 16 warps 7.7-8.4 us, BN=16 7.5-7.7 us, A loads issued before B +0.3 us [probe_b200]"
  - "dead end: extending the skinny kernel to M > 16 by re-reading 16-row A tiles per warp from L2: 10.8 us at M=17, 19.5 us at M=64, 33.7 us at M=128 (A traffic 640 x M x 4 KB) [probe_b200]"
  - "Triton tl.dot pointer GEMMs 8.9-13 us at M <= 64 and 309 us at M=16294 vs cuBLAS 7.8 / 199 us [probe_b200]"
  - "CuTe DSL dense_gemm_persistent (256x256, cluster 2x1, 2-CTA MMA) 195.5 us at M=16294 vs cuBLAS 198.7 us: same ceiling, 73% of the 2.38 PF peak at 1965 MHz [probe_b200]"
  - "r0 passes 25/25; M<=16 6.9-7.2 us (-10% vs cuBLAS), all other sizes equal to cuBLAS within noise [run_tests]"
paths:
  - {max_tokens: 16, lang: cuda, width: 256, threads: 256, rows: 8, grid: oneshot, mem: ldg, wstat: false, launches: 1, prefetch: false, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: tma, launches: 1, note: "cuBLAS nvjet via at::matmul_out; tile and kernel chosen by cuBLAS heuristics per M"}
runs_on:
  H200: {runs: false, representative: false, note: "ld.global.v8 (256-bit) is sm_100 only; the cuBLAS path runs anywhere"}
  A100: {runs: false, representative: false, note: "256-bit loads unsupported"}
  L40S: {runs: false, representative: false, note: "256-bit loads unsupported"}
risks:
  - "Tolerance of this definition not read directly; cuBLAS-equivalent fp32-accumulate error (1 fp16 ulp) passed 10 rounds x 25 workloads"
  - "640 CTAs over 148 SMs (4.3 per SM) leave a 1-CTA imbalance; the 0.6 us above the read floor is unexplained in detail"
  - "Relies on 32-B alignment of A and B (harness pointers are 256-B aligned; falls back to cuBLAS otherwise)"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```