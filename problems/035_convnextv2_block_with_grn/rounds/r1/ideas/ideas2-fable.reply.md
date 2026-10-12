### Notes
Score arithmetic drives everything: Tsol is 0.6–9 µs while Tb is 68–220 µs on 14 of 16 workloads, so S ≈ Tb/(t+Tb). A workload at 10 µs scores ~0.88, at 20 µs ~0.78, at 40 µs ~0.65. The two largest sizes (Tb 0.9/4.1 ms) are near-saturated by any tensor-core design (0.85–0.9) and are worth ≤ +0.007 total. The lever is therefore **critical-path latency at small sizes**: one launch, ≤ 1 grid barrier, weights read once (fp32 via TMA, converted in smem, no pre-pass), everything else L2-resident. predict_score: −85/−90/−95 % per band = 0.83; the leader's 0.69 needs only ~0.45·Tb. Shared technique used by several ideas: fold LayerNorm into GEMM1's epilogue (A = fp16(d⊙lnw); a = r_t·(acc − μ_t·u[n]) + v[n], with u = W1·lnw, v = W1·lnb + b1), so the dwconv producer needs no second pass.

```json ideas
[
{"title": "One cooperative FP16 tcgen05 dataflow kernel with tile flags and image counters",
 "family": "tiled tcgen05 dataflow",
 "design": "Single cudaLaunchCooperativeKernel, 148-296 persistent warp-specialised CTAs (TMA warp, MMA warp, epilogue warps), one ordered work list, one grid.sync only after phase 0 (zero counters, u/v GEMVs). Phase 1 items: 128-token x 64-channel dwconv tiles; x plane rows (+halo) via cp.async.bulk into smem, SIMT 7x7 stencil, emit A1 = fp16(d*lnw) [M][C] (token-major = K-major UMMA A) plus per-token (sum, sumsq) partials; bump a per-128-token-tile flag. Phase 2 items: GEMM1 tiles 128x256 (2SM 256xN pairs for C=768 so each SM ingests half of W1) spin on their tile flag, A1 and fp32 W1 via TMA (W converted to fp16 in smem by a converter warp: no weight pre-pass, weights read once from DRAM), LN fold in the epilogue, exact erf GELU, h fp16 [M][4C] store, segmented per-image column sumsq via warp reduce + few atomics, bump image counter. Phase 3 items: GEMM2 tiles spin on the counters of the images in their M tile, compute s_b[k] (4C sqrt + mean) in smem, A = h via TMA scaled per row in smem by a scaler warp (half2), fp32 W2 via TMA + smem convert, epilogue + (W2*gb + b2) + x residual, coalesced NCHW store (column n -> 128 contiguous hw). cp.async.bulk.prefetch.L2 of W2 and x issued by phase-2 CTAs. Deadlock-free because items are claimed in list order and all producers are co-resident. Small-M/large-K sizes use idea 10's cluster split-K. Dispatch tile tables per (M, C).",
 "p_expert": 0.5,
 "effect": "S: 127 -> 8-12 us (launch 1.8 + dwconv ~1 + GEMM1 ~1.5 + counter wait ~0.5 + GEMM2 ~1.5 + tail); e.g. 1,192,28,28 GEMM1 21 CTAs x 3 k-blocks x 512 clk = 1 us, GEMM2 split-K 4 -> 0.8 us. M: 270 -> 10-18 us (8,384,14,14: 78 GEMM1 CTAs x 6 kb x 512 clk = 2 us). L: 1317 -> ~45 us geomean (8,768: 12; 32,768: 20-25 with weight L2 ingest 66 MB per pair wave = 3 us; 8,320: ~25; 32,128: ~150; 64,96: ~500). Predicted score ~0.83-0.86.",
 "risk": "tcgen05/TMEM/mbarrier complexity and the ELECT-waterfall lesson; cooperative launch forces co-residency at 1 CTA/SM with ~200 KB smem; spin-waits on counters add latency if producers are scheduled late in the list; counter zeroing needs the single grid.sync (1.5 us).",
 "test": "10-min probe on the rented B200: cooperative 148-CTA kernel measuring producer->consumer flag handoff latency and grid.sync cost at 148 vs 296 CTAs, against the 3-launch gap (~2 us each) under harness timing."},

{"title": "Three launches with programmatic dependent launch, prologues overlapped",
 "family": "tiled tcgen05 dataflow",
 "design": "Same phases as idea 1 as three plain kernels on the current stream, launched with the PDL attribute (allowed between our own kernels): K1 dwconv+LN-fold producer (+ u/v GEMVs); K2 GEMM1 whose prologue (mbarrier init, TMEM alloc, fp32 W1 TMA + smem fp16 convert, prefetch) runs before griddepcontrol.wait; K3 GEMM2 whose prologue prefetches W2 and the x residual rows into smem/L2 before waiting. No counters, no cooperative launch, no spin loops; K3 uses idea 10's cluster split-K for M<=400. Per-size tile tables.",
 "p_expert": 0.7,
 "effect": "S: 127 -> 12-16 us (idea 1 + ~2 launch gaps minus PDL overlap; if PDL hides ~1 us per boundary the gap to idea 1 is ~2 us). M: 270 -> 14-20 us. L: 1317 -> ~48 us geomean. Score ~0.80-0.83; lower risk and faster to build, so the natural control lineage.",
 "risk": "Launch-gap savings shrink on the portal (tricks.md 2); 2 extra launches = 20-30% of the S-band time; PDL semantics: K2's writes must not race K1's reads (only prologue work before the wait).",
 "test": "Three near-empty kernels with and without the PDL attribute under harness timing on the rented B200: CUPTI span of the sequence (10 min) tells the true per-boundary cost."},

{"title": "Cluster per image: token-tiled CTAs keep h in shared memory, GRN reduced through DSMEM",
 "family": "in-SM fusion",
 "design": "For C<=192 and HW<=2048 (1,192; 4,192; 8,192 at M=64 per CTA; 32,128,64,64 at 2SM M=256; others routed to idea 1/2): one thread-block cluster of G = ceil(HW/rows) CTAs per image (G<=16, non-portable attribute). Each CTA: TMA the x rows+halo of its tokens per channel, SIMT dwconv to fp16 LN-folded A1 in smem (24-32 KB), GEMM1 from smem in N chunks of <=256 (TMEM), epilogue GELU -> h fp16 in smem (<=128 KB), column sumsq [4C] to smem; cluster barrier; every CTA sums the G sumsq vectors via ld.shared::cluster, forms s_b[k] (identical across the cluster, so it scales the h columns in place, cheap); GEMM2 with A = own h, B = fp32 W2 via TMA multicast to the cluster + smem convert (each W tile crosses DRAM/L2 once per cluster), epilogue + x residual -> NCHW. No global intermediates, no counters, no grid barrier, no atomics.",
 "p_expert": 0.18,
 "effect": "Removes the h/A1 L2 round trips and all cross-CTA waits: 1,192,28,28 on 13 SMs: launch 1.8 + dwconv 7.4M FMA/(13x128) = 3 us + GEMMs 0.46 GFLOP at 13x8192 FLOP/clk = 2.9 us + tail ~1 -> ~9 us (same as idea 1 because B=1 pins it to <=16 SMs); 4,192 and 8,192 ~8 us; 32,128,64,64 ~90 us (no 536 MB h traffic). Net vs idea 1: -1 to -3 us at S/M on 4 workloads, -40% at 32,128. Score +0.01 to +0.02 over idea 1 if all four land.",
 "risk": "Covers only 4 workloads as sized (C=96 56x56 needs 25 tiles > 16; C>=384 h does not fit smem); B=1 sizes starve at 13-16 SMs for the dwconv; 16-CTA clusters need a whole GPC and may slow following launches (ledger constraint allows clusters for GEMM tiles).",
 "test": "Occupancy/placement probe: launch a 16-CTA cluster kernel with 200 KB smem per CTA on the rented B200, measure cluster co-scheduling latency and DSMEM reduce of 16 x 768 floats (10 min)."},

{"title": "h never leaves the SM: GEMM2 A operand read from TMEM, image-aligned tiles, B-side GRN scaling",
 "family": "in-SM fusion",
 "design": "Variant of idea 1 for C<=192 (9 workloads incl. 1,160: 4C/2 = 320 cols): the CTA that computes a GEMM1 tile writes GELU(h) as packed fp16 into a TMEM A region (4C/2 columns, tcgen05.st) instead of global memory, reusing one 128-column accumulator region for GEMM1 chunks and later GEMM2 chunks (N<=128; 384+128 = 512 columns at C=192). M tiles are padded per image (49->64, 196->256, 784->896) so every tile holds one image; the CTA spins on that image's counter, then runs tcgen05.mma with A from TMEM and B = W2 k-blocks scaled in smem per image by s_b[k] (a scaler warp multiplies the N x 64 half2 tile before the MMA; 64 scales per k-block). h traffic to L2 is zero and GEMM2's A ingest costs no smem bandwidth.",
 "p_expert": 0.08,
 "effect": "Versus idea 1 at S/M: saves the h store/load chain and the per-row A scaling (~1-2 us of the ~8-10 us path, i.e. -10..-20% on 9 workloads, ~+0.015 score); at 64,96,112,112 saves 1.23 GB of h traffic (~150 us of ~500). Costs: GEMM2 per CTA sees full K with N=C (no split-K since A is local): 1,192: 12 kb x 384 clk = 3 us, mitigated by N=96 halves (1.5 us).",
 "risk": "tcgen05 A-from-TMEM fp16 packing/layout for kind::f16 is poorly documented and easy to get silently wrong; 1 CTA/SM TMEM full-allocation; padded tiles waste 23% MMA at 7x7 (negligible) but make tile counts small at B=1.",
 "test": "CuTe/PTX probe: a 128x128x64 kind::f16 MMA with A written to TMEM via tcgen05.st from registers, checked against fp32 reference on the rented B200 (10 min); if the layout works, the rest is idea 1 plumbing."},

{"title": "Library path: cuBLASLt FP16 GEMMs with strided-batch NCHW output, custom glue kernels",
 "family": "library",
 "design": "5 launches: K1 custom (dwconv + LN -> fp16 A1 [M][C], W1/W2 -> fp16 workspace, out = x + b2 as the beta=1 C operand); cublasLtMatmul GEMM1 fp16 in / fp32 out with BIAS epilogue (not its GELU: cuBLASLt's GELU is the tanh form, which fails the budget); K2 custom: exact erf GELU + per-image column sumsq -> h fp16; K3 custom: s_b[k] and in-place scale of h rows (or skip by scaling per image into W2 copies when B*C*4C*2 B is small); GEMM2 as a strided-batched cublasLt call computing Dᵀ_b = W2 (C x 4C) x h_bᵀ so D lands directly in NCHW [C][HW] per image with beta=1 residual (strideA = 0, strideB = HW*4C, strideD = C*HW). Per-size algo table from cublasLtMatmulAlgoGetHeuristic.",
 "p_expert": 0.6,
 "effect": "S: 127 -> ~25-30 us (1.8 + 4 x 2 us launches + two small cuBLAS GEMMs at 3-6 us each + glue); M: ~30-40 us; L: 8,768..32,768 ~35-60 us, 32,128 ~250 us, 64,96 ~900 us (cuBLAS at ~1.2 PF/s plus 2.5 GB of h traffic). Score ~0.70-0.73: roughly the public leader, built in a day; a calibration anchor for Tb/Tsol per workload and for the portal/rented ratio of tensor-core code.",
 "risk": "cuBLASLt picks poor kernels for M=98..784 and K=3072 (split-K heuristics); 5 launches cap S at ~0.7; the h fp32 output of GEMM1 doubles traffic at the largest.",
 "test": "Time the two cublasLt fp16 GEMM shapes for all 16 workloads with the heuristic's top-3 algos on the rented B200 (10 min); if GEMM1+GEMM2 < 8 us at S/M the path is viable as the control lineage."},

{"title": "Legacy mma.sync FP16 megakernel: everything in registers and smem, no TMEM/TMA",
 "family": "legacy mma",
 "design": "Idea 1's work list and counters, but GEMMs via mma.sync.m16n8k16 fp16/fp32 with ldmatrix from padded smem tiles filled by cp.async; dwconv warps write fp16 A fragments straight into the GEMM1 smem tiles (LN fold in the accumulator epilogue), GELU/sumsq in registers, h fp16 to L2, GEMM2 A fragments scaled per row by s_b at ldmatrix time (one half2 multiply per fragment), split-K via idea 10. 4 warps x 64x64 warp tiles, 2 CTAs/SM. No warp specialisation, no descriptors: ~600 lines, two days instead of two weeks.",
 "p_expert": 0.3,
 "effect": "Bounded by the unknown HMMA.16816 rate on sm_100 (if 1/2-1/4 of tcgen05 = 0.45-0.9 PF/s): S 8-12 us unchanged (GEMMs <= 2 GFLOP = 2-4 us), M 10-18 us, 32,768,7,7 ~30 us, 32,128 ~200 us, 64,96 ~600 us. Score ~0.80-0.84 with far lower build risk than idea 1.",
 "risk": "If legacy MMA on B200 is throttled to <0.3 PF/s the L band loses 0.02; register pressure with fused dwconv + fragments; smem bank conflicts on fp16 A written by the stencil (pad rows).",
 "test": "Microbench mma.sync m16n8k16 fp16 throughput on the rented B200 in SM cycles with clock64 (10 min): >= 0.6 PF/s at 1.5 GHz makes this a lineage."},

{"title": "dwconv as the in-CTA producer of GEMM1 with LayerNorm folded into the epilogue (large M)",
 "family": "big-size fusion",
 "design": "For sizes with >= 1000 token tiles (32,128,64,64; 64,96,112,112; optionally 8,320): one CTA per 128-token tile covering all N = 4C (<= 512 TMEM columns): x channel-chunk planes (+halo) arrive via TMA; two dwconv warp groups compute 8x4 register patches from smem (no LDGs in flight at the arrive) and write fp16(d*lnw) 64-channel k-blocks into the A ring the MMA warp consumes; per-token sum/sumsq accumulate in smem; epilogue a = r_t(acc - mu_t u[n]) + v[n], GELU, h fp16 store + sumsq. The A1 tensor (308 MB write + 154 MB read at the largest) and one phase boundary disappear; the dwconv FMA work (134 us at the largest) overlaps the MMA (33 us) and the 616 MB x stream (77 us).",
 "p_expert": 0.35,
 "effect": "64,96,112,112: ~500 -> ~400 us (removes ~58 us of traffic and one phase tail); 32,128,64,64: ~150 -> ~120 us; S/M unchanged (routed to idea 1). Score ~+0.002 to +0.003 (these sizes are near-saturated by Tb).",
 "risk": "Register pressure: 32 accumulators + 10-row patch + ring bookkeeping per thread; SIMT dwconv still FMA-bound at 134 us, so the kernel is compute-bound at the largest no matter what; if ingest of x limits, little gain.",
 "test": "Measure a smem-staged 128-token x 96-channel dwconv tile (8x4 patches) in SM cycles on the rented B200: >= 80% of 128 FMA/clk makes the fused producer worthwhile (10 min)."},

{"title": "Dual-GEMM tile with GEMM1 recompute: no hidden tensor at C <= 128",
 "family": "big-size fusion",
 "design": "Phase 2 computes GEMM1 + GELU only for the sumsq statistics (nothing stored). Phase 3, per 128-token tile: GEMM1 again (acc 4C <= 384 columns, A1 fp16 via TMA), GELU * s_b[k] applied in registers (per-row image index), written as fp16 into a smem A2 buffer (128 x 4C x 2 = 96 KB), then GEMM2 from A2 (acc C columns; 480 of 512 TMEM at C=96; C=128 done as two 256-column halves accumulating into the same 128-column acc) + residual -> NCHW. Hidden-tensor traffic (write+read 1.23 GB at the largest, 536 MB at 32,128) disappears for +50% MMA (33 us / 9 us).",
 "p_expert": 0.3,
 "effect": "64,96,112,112: ~500 -> ~350 us; 32,128,64,64: ~150 -> ~100 us; also usable at 1,96/2,96/4,96 (h round trip 2-10 MB, saves ~1 us). Score ~+0.003 to +0.004; combines with idea 7 only if A1 is kept (dwconv once).",
 "risk": "TMEM budget at C=128 forces two GEMM1 halves and a serialized pipeline inside the tile; the epilogue (GELU + cvt + smem write of 128x384) sits on the MMA critical path unless double-buffered across tiles.",
 "test": "CuTe DSL prototype of one tile (GEMM1 128x384x96 -> GELU -> smem -> GEMM2 128x96x384) timed in cycles vs the two GEMMs separately on the rented B200 (10 min); viable if the fused tile is < 1.4x the sum."},

{"title": "Toeplitz depthwise conv on tensor cores for W in {112, 64, 56, 32}",
 "family": "tensor-core dwconv",
 "design": "Per channel plane (fp32 via TMA, converted to fp16 in smem by a cvt warp; optionally hi/lo split planes for fp32-equivalent accuracy at 3 MMAs), output block = 128 image rows x 16 columns: A = 7 row-offset windows of 32 input columns starting at the 16-B aligned column w0-8 (K = 7 x 32 = 224, 14 k16 steps; UMMA K-major descriptor, no swizzle, SBO = W x 2 B, which is a multiple of 16 B only for those widths); B = banded Toeplitz T_c [224 x 16] fp16 built from the 49 weights by one warp (7 KB per channel, L2-resident). MMA per block 14 x 32 = 448 clk; FLOP inflation 4.57x. Output + bias -> LN stats -> fp16 A1 (or into idea 7's ring).",
 "p_expert": 0.15,
 "effect": "64,96,112,112: dwconv 134 us FMA-bound -> 19 us ideal single-fp16 (57 us with the hi/lo split), realistically 40-80 us: total ~500 -> ~430 us; 32,128,64,64: 29 -> ~8 us. Score ~+0.002. Nothing at S/M (dwconv < 1 us there).",
 "risk": "Precision: fp16-rounded x adds ~7e-4 rms end-to-end (3.4 sigma against atol 2.4e-3) so single-fp16 passes on the 98% rule but not with 5-sigma margin; the overlapping-row descriptor trick may be bank-conflicted or disallowed; applies to 4 workloads only.",
 "test": "fp64 numpy simulation of fp16 dwconv inputs on 3 workloads x 2 seeds (count elements beyond tol) plus a single-channel Toeplitz MMA correctness probe on the rented B200 (10 min)."},

{"title": "Cluster split-K with DSMEM reduce-scatter for small-M / large-K GEMMs",
 "family": "tiled tcgen05 dataflow",
 "design": "For GEMM2 at M <= 400 (2,768; 8,768; 1,192 at K=768) and GEMM1 at 2,768,7,7: a cluster of G = 4-8 CTAs along K (allowed: a GEMM tile needs it), each computing a 128 x N partial over K/G in TMEM; after tcgen05.ld to registers and st.shared, a cluster barrier, then CTA j owns N/G columns and reads the other G-1 partial slices through ld.shared::cluster (128 x N/G x 4 B = 8 KB each at N=128), adds x + (W2 gb + b2) and stores NCHW. Replaces fp32 global atomics (anchor's P3 pattern, ~1.2-4.8 M atomics) and the x+b2 output pre-initialisation pass, and removes the atomic-order nondeterminism.",
 "p_expert": 0.3,
 "effect": "2,768,7,7 GEMM2: 6 N-tiles x G=8 = 48 CTAs, 6 kb x 256 clk = 1 us + 0.3 us reduce (vs 48 kb serial = 8 us or atomics); 8,768: same with 4 M-tiles -> 192 CTAs; 1,192: 7 x 2 x 4. Shaves 2-6 us from the ~10 us S/M path on 4-5 workloads: ~+0.01 score on top of idea 1/2.",
 "risk": "Clustered launches cost 2-5% on the portal (#38) and seemed to slow later launches; partial tiles need 64 KB smem each beside the operand ring; cluster scheduling adds dispatch latency on a one-wave grid.",
 "test": "Microbench on the rented B200 (10 min): cluster-8 reduce-scatter of 128x128 fp32 partials vs red.global.add.v4.f32 into a pre-initialised output, both under harness timing with the L2 flush."}
]
```

```json leader-stories
[
{"story": "A PyTorch-level solution: channels_last + torch.compile(max-autotune) with TF32 matmul enabled. Inductor leaves cuDNN depthwise conv and two cuBLAS TF32 GEMMs, and fuses LN, bias+GELU, the GRN reductions/affine and the residual into 3-4 Triton kernels: ~8 launches.",
 "consistent_with": "Implied S-band times 30-46 us are ~10x the compute/DRAM time, i.e. pure launch chain (8 x ~4 us); L-band 85 us at 32,768,7,7 = 174 TF/s needs TF32 cuBLAS; 1889 us at the largest fits cuDNN dwconv (~500 us) + TF32 GEMMs (~150 us) + ~1.2 GB of fused elementwise traffic; uniform ~0.69 across sizes.",
 "test": "Run exactly this (compile + allow_tf32 + channels_last) on the rented B200 at 1,192; 8,384; 32,768; 64,96 and compare with 30.6 / 47.2 / 85.9 / 1888.7 us; a match within 20% confirms the story and a 3-4x headroom for a single-launch design."},
{"story": "Hand-written CUDA with library GEMMs: a custom fused dwconv+LN kernel (faster than cuDNN's generic depthwise kernel), cuBLASLt fp16/TF32 GEMMs, a fused GELU+GRN-statistics kernel, and GEMM2 with the GRN scale and residual folded in: 5-6 launches, fp16 hidden tensor materialised.",
 "consistent_with": "At 32,128,64,64 the leader's 424 us total is below cuDNN's depthwise conv alone (492 us), so the dwconv must be custom; 85 us at 32,768,7,7 = weights 19 MB + x/out 28 MB (7 us) + two cuBLAS GEMMs at small M (~20 us each) + 3-4 launch gaps; S band 30-45 us = 6 launches x ~3 us + two inefficient small-M cuBLAS calls.",
 "test": "Build idea 5 (cuBLASLt path) in a day and compare its 16 times with the implied ones; if the per-size pattern (especially the C=768 sizes) matches within 25%, the leader is launch-bound at S and beatable by ~3x there."},
{"story": "Non-uniform scores: a tensor-core design (CUTLASS/CuTe sm100 GEMMs, 2SM tiles) that scores ~0.9 on the two largest sizes and 8,320, but is synchronisation-bound at small sizes (~0.6, i.e. 45-70 us) because it keeps 6-10 launches or grid barriers and converts weights in separate passes.",
 "consistent_with": "0.69 x 16 = 11.0; if 3 large sizes score 0.9 the other 13 average 0.64, i.e. t ~ 0.55 Tb = 40-80 us, which is what a 10-stage pipeline with ~2.5 us gaps plus a 19 MB weight-conversion pass gives at C=768; the uniform-S assumption in leader_gap.py would then overstate their S-band speed and understate their L-band speed.",
 "test": "Fit t = F + bytes/BW + FLOP/R to the implied per-workload times: a fixed term F of 25-40 us with R near 1 PF/s supports this story, F < 10 us refutes it. Independently, our first single-launch submission gives real per-workload S values to compare against the 0.69 mean."}
]
```