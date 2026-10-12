### Notes
Score arithmetic from the leader table and lineage A's rented rows ×1.28: the 7×7/14×14 rows with C=384/768 are our worst (S 0.48–0.60; weight streaming plus dependent chains, not FLOPs) and are worth about +0.09 if they land at 20–40 µs rented. The seven C≤192 rows are worth +0.05–0.065 at 15–20 µs rented. The top two rows are already ahead of the leader; fusing K2 and the conv there adds only about +0.02. Every measured floor says the same thing: ~2 µs per dependent round, 1.5 µs per sync, 4.6 µs F, cp.async post-sync rounds at 4–8 B/clk/SM but cp.async.bulk at 120 GB/s/SM. The ideas below attack bytes-per-CTA and round count per row class, with three long shots on TMEM, a tensor-core Toeplitz conv and full recompute. Portal/rented ratio 1.28 is assumed throughout.

```json ideas
[
{"title": "Weight-split cooperative kernel for the 7x7/14x14 rows (C=384/768), TF32 straight from fp32 weights",
 "family": "coop weight-split",
 "design": "One cooperative launch for the six rows with M<=3136 and C>=384. Phase 0: 148-296 CTAs do the conv on 4-channel plane groups via 1-D cp.async.bulk (784 B / 3136 B copies, 16 B multiples), write d fp16 plus per-(chunk,token) LN partials. Sync 1. Phase 1: hidden split into NS slices of width 4C/NS (NS=48 at 2,768; 24 at 8,768; 16 at 16,768; 12 at 32,768 with 2-3 token tiles per slice); each CTA streams its W1 slice (18.9 MB/NS fp32: 390 KB at NS=48) through a TMA TFLOAT32 2-D map (H14: round-to-nearest, no conversion pass) into a 4-stage ring, reads d for its tokens from L2, applies LN per token at fragment load (cvt.rna or f16 split if needed), computes h for its slice, exact GELU, keeps the slice fp16 in smem (M*width*2 <= 75 KB), writes per-image sum-of-squares for its own channels (no atomics) and a per-image partial sum of gf. Sync 2. Phase 2: every CTA reads B*NS floats, forms s_b for its own channels locally, scales its smem slice, adds gb, streams its W2 slice (C x width fp32) and red.global.add's the [M x C] partial into out pre-filled with x+b2 in phase 0. Weight bytes per CTA = 18.9 MB/NS at ~120 GB/s/SM; red.add bytes = M*C*4*NS at 4 TB/s; the NS minimising the sum gives 7 us at 2,768, 14 at 8,768, 19 at 16,768, 27 at 32,768. C=384 rows have 4.7 MB of weights and are faster still.",
 "p_expert": 0.3,
 "effect": "Rented targets 20/25/30/40 us at 2,768/8,768/16,768/32,768 (A today 64/63/78/97) and 25/35 at 8,384/16,384 (78/89): per-row S 0.48-0.60 -> 0.76-0.80, about +0.09 of mean; half the gain still gives +0.045.",
 "risk": "L2 ingest of 148 CTAs each pulling 400 KB-1.2 MB at once may fall below 120 GB/s/SM (the E3 probe used 148 CTAs reading mostly shared slices); red.add contention from NS writers per address; TF32 accumulation error must be re-simulated (round-to-nearest TF32 passed at 0.000% beyond tol).",
 "test": "148-CTA probe after the 2xL2 flush: each CTA streams a distinct 590 KB TFLOAT32 TMA slice of W1 into a ring and red.adds a 98x768 partial; time vs CTA count and NS. Go if the sum is <= 10 us at NS=32."},

{"title": "Token-stationary single-sync kernel: BN = 4C, 8-16 token tiles, weights streamed not resident",
 "family": "single-sync token-stationary",
 "design": "C lineage E3 with the hidden block widened to all 4C channels so the conv is computed once per token (no 4C/BN redundancy) and the only cross-CTA dependency is the GRN sum. CTA = 8-16 tokens x all C channels: x halo via one 3-D TMA box (or bulk planes for W=7/14), conv with lanes along w, LN in-CTA, a tile fp16 in smem. W1 and W2 arrive as fp32 through cp.async.bulk rings started at kernel entry (295 KB each at C=192, 74 KB at C=96), converted with cvt.rn.f16x2 at fragment load (the E3 finding: converting inside the ring was 2.5x slower). GEMM1 (mma.sync) -> GELU -> hidden tile T x 4C fp16 in smem (T=16: 24 KB at C=192) -> red.add of per-image sumsq (B*4C floats). One grid.sync. Read gf (B*4C), scale in smem, GEMM2 from the W2 ring already in flight, residual from the x tile centre already in smem, NCHW store with lanes along hw. CTAs: 98 at 1,192 (T=8), 196-392 at the other S rows.",
 "p_expert": 0.35,
 "effect": "Chain: F 4.6 + x round 2 + conv 1 + GEMM1/GELU 1.5 + sync 1.5 + gf round 2 + GEMM2/store 2 = ~15 us rented at 1,192 (E3 v2 30.8, B 29.8), ~20 at 2,96/4,192: S rows 0.60-0.65 -> 0.75-0.78, about +0.05 of mean over seven rows.",
 "risk": "Aggregate L2 weight traffic: 148-392 CTAs x 590 KB = 90-230 MB at 16 TB/s = 6-14 us at C=192, which would eat the gain; mitigation is a 2-CTA cluster with TMA multicast of the weight tiles (allowed: the GEMM tile shares B), or fp16 weight scratch written by role CTAs with a per-slice ready counter.",
 "test": "Stub kernel: 148 and 392 CTAs each stream 590 KB of W1+W2 (fp32, bulk ring) after the flush, with and without 2-CTA multicast; harness_time minus the 1.9 us empty kernel. Go if <= 4 us at 392 CTAs."},

{"title": "Two-sync tiled coop kernel with every independent operand prefetched by bulk copies before the sync it precedes",
 "family": "coop two-sync",
 "design": "Lineage B structure, rebuilt around the measured pathology: post-sync cp.async/LDG rounds run at 4-8 B/clk/SM while cp.async.bulk runs at 120 GB/s/SM. Tiles shrink to 16 tokens x 64 hidden (588 CTAs at 1,192, 2 CTAs/SM resident) so no CTA has a long serial chain. Before sync 1 each CTA already holds its W1 slice (64x192 fp16 = 24 KB) and dwconv weights in smem; after sync 1 the only load is the a tile (16x192 fp16 = 6 KB) by one cp.async.bulk. Before sync 2 it holds its W2 slice (64 out-ch x 768 fp16 = 96 KB) and the 16-token x residual; after sync 2 it bulk-loads gf (3 KB) and its 16x768 fp16 hidden rows (24 KB). Conv CTAs cover all channels of a token group so LN is in-CTA and d is written already normalised (a) in fp16. Weight conversion for C<=384 done by the conv CTAs' spare warps during phase 1 (1.2 MB at C=192), published through a counter.",
 "p_expert": 0.4,
 "effect": "F 4.6 + conv 3 + sync 1.5 + 1 round 1.5 + MMA/GELU 1 + sync 1.5 + 1 round 1.5 + MMA/store 1.5 = ~17 us rented at 1,192 (B 29.8); S rows +0.1-0.15 each, ~+0.05 of mean; M rows with B>=4 gain 8-12 us each.",
 "risk": "The r2 flat kernel collapsed rounds and did not move 1,192, which suggests cycles (1.28x clock scaling) rather than rounds dominate; if a 6 KB bulk round after a sync still costs 2-4 us this idea is dead.",
 "test": "Instrumented stub: 588 CTAs, grid.sync, then one cp.async.bulk of 6 KB (L2-resident, just written by other SMs) vs one cp.async round of the same bytes; globaltimer stamps. Go if the bulk round is <= 1.2 us."},

{"title": "Per-image dataflow: no grid barrier, image counters and an ordered tile queue",
 "family": "dataflow",
 "design": "Persistent plain launch (<= resident CTAs) for rows with B>=4. CTA 0 zeroes the counters, then one initial grid.sync (or a 1 KB cudaMemsetAsync, one extra activity). A global atomic queue hands out tiles in the order: all conv tiles by image, then GEMM1 tiles by image, then GEMM2 tiles by image. A GEMM1 tile of image b spins on conv_done[b]; a GEMM2 tile spins on gemm1_done[b] (its sumsq is complete then). Deadlock-free because every tile a CTA can wait on was dequeued earlier by a resident CTA that never waits on later tiles. GEMM2 of image 0 overlaps GEMM1 of images 1..B-1, so the two barrier waits and their post-sync ramps disappear and the weight passes of phase 2 hide under phase 1 work. Any GEMM engine (mma.sync now, tcgen05 later) plugs in.",
 "p_expert": 0.3,
 "effect": "Removes 2 x (1.5 us sync + 2-4 us post-sync round) and overlaps weight streams: 7-11 us per row on 60-120 us rows (8,384; 4,96; 8,192; 16,384; 8,768; 16,768; 8,320; 32,768), about +0.02-0.03 of mean; nothing at B=1-2.",
 "risk": "Spinning CTAs steal issue slots and L2 bandwidth from working ones; queue atomics add ~1 us; the counter initialisation needs either the extra sync or a memset activity.",
 "test": "Take lineage B, replace grid.sync with counters at 8,384,14,14 and 16,768,7,7 only; interleaved x3. Go if >= 6 us faster per row."},

{"title": "Two-pass streaming tcgen05 kernel for the three big rows (8,320; 32,128; 64,96) with K2 fused",
 "family": "tcgen05 fused",
 "design": "Pass 1 (one launch, 1 CTA/SM, 2SM pairs where N>=256): TMA 3-D x boxes -> SIMT conv -> LN in-CTA -> fp16 a tile in swizzled smem -> tcgen05 kind::f16 GEMM1 (N = 4C in two TMEM accumulators, 384-512 columns) -> epilogue warps: tcgen05.ld, exact GELU, per-image sumsq red.add, fp16 hidden store with evict_last. Pass 2: GEMM2 2SM with A = hidden via TMA, B = per-image W2*diag(s_b) fp16 built by the first CTA of each image (4-6.5 MB total at these rows), residual x and NCHW store in the epilogue with lanes along hw. Weight conversion W1 in pass 1 role CTAs. Hidden traffic 616 MB write + read at the largest = 176 us; conv at the current 3x-off rate ~200 us overlaps the hidden write.",
 "p_expert": 0.5,
 "effect": "64,96: 1151 -> ~400 us rented (S 0.743 -> 0.92); 32,128: 298 -> ~130 (0.713 -> 0.85); 8,320: 95 -> ~55 (0.65 -> 0.78): about +0.025 of mean.",
 "risk": "tcgen05 descriptors/swizzle silently wrong (validate on all sizes); the conv remains at 315 us at the largest and bounds pass 1 unless idea 8 or 7 lands; 2 launches cost ~2.5 us extra at 8,320.",
 "test": "Pass-1-only kernel at 32,128 with the GEMM1 epilogue stubbed to a store: compare against K1+GEMM1+K2 of lineage A (81+44+113 us). Go if <= 120 us."},

{"title": "Library pipeline cut from 5-6 launches to 3: custom GEMM1 and GEMM2 with fused epilogues around the TMA conv",
 "family": "library pipeline",
 "design": "Keep lineage A's K1 (TMA conv, d channel-major, stats, weight prep). Replace cuBLASLt GEMM1 + K2 with one mma.sync (tcgen05 later) GEMM1 whose epilogue applies z = rstd*(Y - mean*u) + cst, exact GELU, per-image sumsq (4 copies), fp16 hidden store. Replace K3 + cuBLASLt GEMM2 + K4 with one GEMM2 kernel: prologue reduces the B x 4C sumsq from L2 into s_b in smem (the k3a chain moves inside), A tile scaled by s_b at fragment load, swapped operands (M = C rows, N = token tile) so each thread's TMEM/accumulator row is contiguous along hw: coalesced NCHW residual read and output store, no transpose kernel. Per-shape templates for N tiles of 96/128/160/192 (no runtime guards around HMMA).",
 "p_expert": 0.6,
 "effect": "Removes K2 (8-14 us), K3 (6-17 us) and K4 (4-8 us) at S/M rows: 1,192 34.8 -> ~24 us rented, 2,768 64 -> ~40; about +0.04 of mean. Largest row: K2's 447 us becomes part of GEMM1's store.",
 "risk": "Each custom launch has cost 6-17 us in this lineage because of its own dependent chains; the new kernels must issue all operand loads up front or the saving is halved.",
 "test": "GEMM1-with-fused-epilogue kernel alone at 1,192 and 8,320 vs cuBLASLt GEMM1 + K2 (5.1+9.8 and 11+37.8 us). Go if <= 8 and <= 30 us."},

{"title": "Long shot: tensor-core Toeplitz depthwise conv with an fp16 hi/lo split of x",
 "family": "conv",
 "design": "Per channel and 16-column block, out[h, w0:w0+16] = sum_i X_i[h, w0-3 .. w0+28] . T_i where T_i is the 32x16 banded Toeplitz of tap row i: a GEMM with M = output rows, K = 7 x 32 = 224, N = 16, A = fp16 plane rows read at (row h+i, column offset w0-3) from smem, B = 49 weights expanded to a 224 x 16 fp16 matrix once per channel. x is split x = x_hi + x_lo (both fp16, 2 MMAs) so the conv is fp32-class (the single-pass fp16 conv is only ~3.5 sigma). FLOP inflation 4.6 x 2 = 9.2x: 70 GFLOP at the largest = 38 us on tcgen05 kind::f16, 153 us on mma.sync (no better than the SIMT bound, so only tcgen05 pays). A operand: mma.sync fragments via LDS from the plane (K-blocks never cross i); for tcgen05 the plane is stored once per row phase i (7 copies, 7x smem) so each k-block is a canonical no-swizzle K-major slab.",
 "p_expert": 0.15,
 "effect": "Conv at the largest 315 -> ~60 us, 32,128 81 -> ~15, 8,320 23 -> ~6; combined with idea 5 the largest row reaches ~250 us rented (S 0.95): about +0.015 of mean, more if conv also bounds the M rows.",
 "risk": "The 7x replicated smem plane limits tile height and occupancy; tcgen05 A-descriptor gymnastics; hi/lo doubles everything and the LN'd result still rounds to fp16 for GEMM1 (already accepted).",
 "test": "mma.sync version of one 112x112 plane with the hi/lo split: check max |d - fp32 d| <= 1e-5 relative and time per plane; then the smem-replication cost of the tcgen05 layout."},

{"title": "SIMT conv restructure: warp-private TMA stages, 8-row strips, no CTA-wide syncs",
 "family": "conv",
 "design": "Keep the 3-D TMA box ingest (the only staging measured above 15 GB/s/SM) but make each warp own its stage ring with its own mbarriers (no __syncthreads per stage), give each thread a 4-column x 8-row strip (42 LDS.128 per 1568 FMA, 1:37 instead of the 1:12 that made R=2 LDS-bound), tile height RT = 16 with the halo 22/16 = 1.4x instead of 1.75x, 3 CTAs/SM x 256 threads at <= 80 regs, and start all STG + TMA issues for the next channel group before computing the current one. For W=7/14 rows, planes of 4 channels via 1-D bulk copies in the same ring. The measured 40%-busy FMA pipe at 64,96 should rise toward 80%.",
 "p_expert": 0.6,
 "effect": "K1 at 64,96 315 -> ~170 us rented, 32,128 81 -> ~45, 8,320 23 -> ~13, S rows 5.6 -> ~4: about +0.01 of mean alone, +0.02 with idea 5 where the conv bounds pass 1.",
 "risk": "R=4 at 128 regs was already 'no better', so the limiter may be the TMA engine's narrow-box throughput (34 GB/s/SM at the largest) rather than the SM side; then only wider boxes (2 channels per box row) or fewer, taller boxes help.",
 "test": "Stage-only TMA ring at 64,96 with box {120, 22, 2} vs {120, 14, 2} vs a 4-D box carrying 2 channels per row: ingest GB/s per SM. If none exceeds 60 GB/s/SM, the conv is TMA-bound and idea 7 is the only route."},

{"title": "Long shot: TMEM-resident hidden for the C=96 rows, kind::tf32 with zero conversion passes and one sync",
 "family": "tcgen05 on-chip",
 "design": "For 1,96 / 2,96 / 4,96 (and 1,160,449,7 with 4C = 640 columns via two passes). CTA = 128 tokens (or 64 with M=64 MMAs for 1,96: 49 CTAs). Conv + LN in-CTA from a TMA x box into an fp32 a tile; W1 (96 x 384 fp32 = 147 KB) and W2 streamed from the fp32 inputs through TFLOAT32 TMA maps (round-to-nearest, H14), kind::tf32 GEMM1 accumulating D1 [128 x 384] fp32 in TMEM (384 of 512 columns). Epilogue warps tcgen05.ld, exact GELU, sumsq red.add, tcgen05.st the hidden back into TMEM as the tf32 A operand of GEMM2 (A-from-TMEM is legal for tcgen05.mma). One grid.sync. Read gf (B x 384), scale the TMEM columns in place (ld/st), GEMM2 with B = W2 TFLOAT32 from smem, epilogue adds x from the conv tile centre and stores NCHW. The hidden never touches smem or L2 and no weight is ever converted.",
 "p_expert": 0.1,
 "effect": "Chain ~16 us rented at 1,96 (A 35, B 28): three rows from ~0.65 to ~0.77, +0.02 of mean; extends to 1,160 (205 KB weights) for +0.01.",
 "risk": "25-98 CTAs only, so any per-CTA serial step shows fully; the TMEM ld/st passes for GELU and scaling cost ~4 x 128 x 384 x 4 B through registers per CTA (fine); tcgen05 A-from-TMEM layout rules for tf32 are intricate and untested here; TF32 accumulation error needs the fp64 simulation for these rows.",
 "test": "A 49-CTA kernel at 1,96 doing only GEMM1 -> TMEM -> GELU -> tcgen05.st -> GEMM2 from TMEM with random data, validated against torch; time it. Go if <= 6 us for the two GEMMs plus epilogues."},

{"title": "Long shot: full recompute at the top two rows, no hidden in memory at all",
 "family": "recompute",
 "design": "Two passes over x, both the same fused tile kernel (idea 5's pass 1): pass A computes conv -> LN -> GEMM1 -> GELU and emits only the per-image sumsq (red.add, B x 4C floats); pass B recomputes the same tile, scales by s_b (read once per CTA), runs GEMM2 (tcgen05 kind::f16, W2 fp16 scratch) and adds the residual from the x tile already in smem, storing NCHW. DRAM traffic 410 + 410 + 206 MB = 146 us at 7 TB/s instead of 410 + 616 + 616 + 410 + 206 = 320 us; GEMM 1.5x = 98 us on tensor cores; conv twice. Wins over materialising only if the conv runs within ~1.5x of its 134 us bound (ideas 7 or 8), when the total is ~300 us rented vs ~460 for the materialised design; at 32,128 the hidden is only 134 MB so the two tie.",
 "p_expert": 0.3,
 "effect": "64,96: 1151 -> ~300 us rented (S 0.743 -> 0.94), +0.012 of mean; 32,128 ~0 unless the conv is also cheap.",
 "risk": "Pays the conv twice: at today's 315 us conv it loses to idea 5; the sumsq pass has no useful output to overlap its tail with.",
 "test": "Time idea 5's pass 1 with the hidden store removed at 64,96; if it is <= 220 us, recompute beats materialising on paper and the pass-B variant is worth building."}
]
```

```json leader-stories
[
{"story": "One cooperative or persistent fused kernel (SIMT dwconv, LN, two fp16/TF32 mma.sync GEMMs, fp16 hidden in L2, two grid syncs), written well but at the same floor our lineage B hits: F ~4.6 us plus two syncs plus ~6 post-sync rounds of 2-2.5 us gives 22-25 us rented = 28-32 us portal, exactly the 30.7 us at 1,192. Their L rows (424 and 1889 us portal) are a conv ~3x off its FMA bound plus an unfused fp16 hidden round trip, which also matches our A lineage within 20%.",
 "consistent_with": "Leader S rows cluster at 31-47 us with a bytes slope of ~4 us/MB (conv staging), the C=768 rows cost only 35-86 us (weights read once from a fused kernel, no 5-launch chain), and the largest row is slower than our A kernel.",
 "test": "Fit leader rows to a + b*MB + c*GFLOP: a ~ 25 us portal and c ~ 1/(0.4 PF/s) support this story. Then run lineage B with ideas 3 or 4: if 1,192 drops below 22 us rented, we pass this leader on the S rows."},
{"story": "A 3-4 launch library pipeline: cuDNN or a custom TMA depthwise conv, cuBLASLt TF32 GEMMs reading the fp32 weights directly (no conversion pass, which is why 2,768,7,7 costs only 35 us portal = 27 rented, close to the 19 MB weight read plus two GEMMs at 7+12 us), and one fused elementwise kernel for GELU/GRN. Each launch is a 6-10 us chain as in our A lineage; the S rows are 3-4 launches x 8-10 us.",
 "consistent_with": "Our r1 TF32 cuBLASLt timings (98x3072x768: 11.8 + 7.3 us cold) plus a 6 us conv and a 6 us elementwise launch sum to 31-35 us rented at 2,768 = 40-45 portal, within 25% of the leader; S rows at 3 launches x ~10 us = 30 rented = 38 portal, slightly above the leader.",
 "test": "Build lineage A with TF32 cuBLASLt (no w1h/w2h conversion in K1) and the K2+K3 fusion of idea 6 at 2,768 and 8,768: if the rows land at 30-40 us rented, the leader's C=768 times are a plain library pipeline and idea 1 beats them by 2x."},
{"story": "A Triton (or CuTe DSL) persistent kernel using tl.dot in TF32 on fp32 operands with the hidden materialised in fp32: no conversion passes, few hand-tuned rounds, but a 2x hidden footprint. The whole-call TF/s saturates at 130-175 for the 7-15 GFLOP rows (TF32 tensor cores at 20-40% of the 0.91 PF/s peak, typical of Triton short-K tiles) and the largest row runs at ~0.33 TB/s effective because the 2.5 GB fp32 hidden round trip (350 us) plus a cuDNN-class conv (~500 us) plus two GEMMs fill 1889 us.",
 "consistent_with": "Leader L rows: 64,96 at 1889 us and 32,128 at 424 us are within 10% of 'our A pipeline with an fp32 hidden'; the M rows' implied 60-130 TF/s match Triton tl.dot TF32 at short K (#009 saw Triton far below cuBLAS on B200).",
 "test": "Replace lineage A's fp16 scratch with fp32 (hidden and d) and TF32 cuBLASLt at 32,128 and 64,96: if the times land within 15% of 424/1889 portal (331/1475 rented), the leader is this family and our tcgen05 pass (idea 5) is worth +0.05 on those two rows alone against it."}
]
```