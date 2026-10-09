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
- **Portal Tsol (measured, r0's page)**: `max(bytes / 6.75 TB/s, FLOPs / 1.81 PFLOP/s)` reproduces all 25 reported
  scores to within rounding (rms error 0.0014; `sol.yaml`, `planner.sol_model`). That is 3.1-3.7 us for M <= 289,
  5.7 us at 492, 11.1 us at 952 and 102-189 us for the compute band. The compute term already binds from M ~ 450.
- **Score leverage per workload.** Small/mid M: Tb 8.7-21 us against Tsol 3-11 us, so 1 us is worth ~0.04 of S
  there; reaching the ~8 us portal read floor at M = 17-128 would give S ~ 0.56-0.57. Large M: Tsol is 85% of Tb
  (189 vs 223 us at M=16294), so **each 1% of time is ~0.02-0.03 of S** on those 6 workloads: matching the
  reference is +0.08-0.11 each, beating it by 3% about +0.15.

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

**Portal, submission 62765: score 0.4870** (B200, v1.1, AC). Per workload (portal us / Tb us / S):
M <= 16 7.7-8.2 / 8.7-9.3 / 0.54-0.55 (our kernel, -10% on both machines); M = 17-952 9.1-22.0 / 8.9-21.3 /
0.46-0.52; M >= 8828 140.6-242.6 / 130.6-223.1 / **0.39-0.42**. The same `at::matmul_out` cuBLAS call is 3%
slower than the reference on the rented B200 at large M and **6-9% slower on the portal** (mid sizes: 0.95-1.11x
Tb, while r0's own portal times are smooth).
**Tb is a stored constant, not re-measured per submission**: #38's 23 pages show identical Tb for every workload.
So the uneven mid-size Tb is a fixed target (some sizes are easy, some hard), and the 6-9% large-M gap is fixed:
NVIDIA's baseline run was faster than any run of the same cuBLAS kernel we get. Matching it needs a kernel ~7%
faster than cuBLAS under the portal's conditions. Portal/rented time
ratio: 1.07-1.13 for M <= 16 (memory-bound), 1.18-1.23 for 93-952, 1.22-1.26 for the compute band (clock ratio
1965/1500 = 1.31). The reference's portal/rented ratio is 1.06-1.23 (noisy at small M, 1.16-1.21 at large M).

**r1 portal (62881) r1-cutlass256-lt-tiles: 0.4847.** cublasLt tile 314 -7/-10% at M=25/34 (transferred), tile 183
-1% at 172 (rented said -7%), CUTLASS 256x256 2-CTA +1.1-2.5% slower than cuBLAS at M >= 8828 (rented said 1-3%
faster: the power-capped rented box cannot rank large-M kernels). `r1b-lt-tiles` = r0 + the tiles, no CUTLASS:
25/25 rented, expected ~0.489 from the two pages.

**r2 (all negative):**
- Pipelined cp.async ring + mma.sync for M = 17-64: 1.8-2.4x slower than cuBLAS. Streaming B and A through smem with
  *no* MMA already takes 7.7-8.9 us vs cuBLAS 7.8-8.0; a B-only ring reaches 6.85-7.2 us. mma.sync needs >= 5
  independent accumulator chains per warp to reach ~520 TF/s; per-chunk barriers + ldmatrix leave it at ~30 clk per
  HMMA. Split-K epilogues cost +1-5.6 us. cuBLAS is within ~0.3-0.9 us of the read floor at M = 17-64.
- cublasLt beyond the heuristic: nothing new >= 4% at any mid size (heuristic indices 0-2 with a 32 MB workspace;
  split-K 2 is +48-141%). Not run: the full template enumeration (634 tiles x custom option 0-7 x cluster ids;
  start from a heuristic-returned algo, since a bare AlgoInit never passes AlgoCheck).
- CUTLASS 1-SM whole-wave tiles at M = 289-952: 17-62% slower than cuBLAS with any BK or A multicast. Per-SM ingress
  is capped at ~150-180 GB/s whatever the L2 aggregate, so single-wave tiles follow
  `t = bytes received per SM / ~90 GB/s + 1.5 us`; cuBLAS's 2-CTA tiles win by halving the B bytes each SM receives.

**r1b portal (62899) r1b-lt-tiles: 0.4818**, below r0 although its large-M code is r0's exact cuBLAS call: the
compute band drifted slower with each submission (r0 -> r1 +2%, -> r1b +2.6%), while M <= 172 repeated within
0.1 us. **Portal run-to-run variance is ~2-3% for compute-bound sizes**, so r1's "CUTLASS 1-2.5% slower" is
within noise, and a large-M comparison needs both kernels in one submission or repeated pages.

**Status: parked at r2.** Realistic remaining upside ~+0.01 (cublasLt enumeration, a 2-SM tcgen05 kernel at
289-952 with L2 prefetch of B). Large M is blocked by the stored Tb.

## 8. Open questions

- The exact atol/rtol of this definition (both cuBLAS and mma.sync pass; relevant only if a reduction order gets
  sloppy).
- Answered in r1 (H12 refuted): `at::matmul_out`, `mm_out`, `matmul(out=)`, `F.linear` and the reference launch the
  identical nvjet kernel (128x256_64x6 2cta at M >= 8828); no other cublasLt algorithm is faster; C alignment is
  irrelevant. The residual ~1% on the rented box is the harness zero-filling the DPS output before each call.
  The portal gap is the stored Tb (above).
- r1 also found: CUTLASS 256x256x64 2-CTA persistent = cuBLAS at large M (every other tile, raster, StreamK is
  2-17% slower); stock CUTLASS is 1.2-3x slower than cuBLAS at M = 93-952; forced cublasLt tiles beat cuBLAS's pick
  at M = 25/34 (tile 314) and 172 (tile 183) by 4-10%; the rented box power-caps to ~1117 MHz under sustained GEMM
  load; up-front-load split-K with mma.sync for M = 17-64 is 1.3-2.6x slower than cuBLAS (all MMA runs after the
  stream ends: a mid-band kernel must pipeline loads and tensor work).
- Answered by r0's page: Tb is the reference (cuBLAS) timed on the portal, 1.16-1.21x the rented time at large M
  (not the full 1.31x clock ratio); Tsol is the roofline above.
- Whether tcgen05 can be driven from raw PTX in a small kernel for M = 64-952 without the CUTLASS machinery, and the
  CuTe DSL compile time budget (20 s per kernel, 300 s limit).
- Whether a 2-CTA cluster DSMEM split-K reduction costs less than the 0.3-0.6 us it is meant to save at M <= 16.
