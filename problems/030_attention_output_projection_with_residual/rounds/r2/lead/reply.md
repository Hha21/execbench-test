### Assessment

Score 0.453; top 5 needs 0.491. Back-solved Tsol is 15.4/30.1/60.2 µs at 2048/4096/8192, and Tb equals a plain cuBLAS GEMM at the lock (flat 1.22 × rented matmul, r1 E1). So every workload sits at the same wall: our fused kernels pay for the residual read, and they pay more at the portal (ratio 1.37) than plain GEMMs do (1.22). If the fused kernel scaled at 1.22 it would land at 79.5 µs at 8192 (S 0.57 instead of 0.48). The 4096 group (four workloads) and 2048 (two) are the weakest: still on cublasLt, 1.14 × Tb, and r1 showed cublasLt's C read is intrinsic there. A whole-wave transposed tile exists for both (TN=192 at 4096 and TN=96 at 2048 both give 220 pair tiles, 2.97 waves) but r1's TN=192 lost to cublasLt with the costly C epilogue, so the tile and the epilogue fixes are coupled: predict_score gives +0.047 for M −12% plus L −7%.

This round tests three things: E1 builds the M-band whole-wave transposed kernels with the best epilogue settings and packages a 4096-group portal A/B (H13). E2 attacks the C read inside the existing CUTLASS epilogue: deeper C stages, smaller subtiles, and an L2 prefetch of the C tile at tile start, packaged as an 8192-group A/B (H14, which also settles H17 on the portal). E3 is exploratory: fold the residual into the TMEM accumulator before the K loop so the epilogue has no C at all (H15). S band is parked: it is worth +0.013 per 10% and the split-K route is refuted.

```yaml ledger
# Hypothesis ledger for #30 (030_attention_output_projection_with_residual). Seeded 2026-10-09; updated r2 2026-10-10.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids
  lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
- 'Any cublasLt algorithm forced per size must be chosen from shape only, verified 16/16 on 10 random rounds, and must launch the
  same activity sequence every call (split-K with a reduce kernel is allowed only if the count is constant per shape).'
- 'The loop''s lint rejects the literal bf16 type name in CUTLASS sources; CUTLASS candidates go through run_tests and the
  operator whitelists the type name for this problem (dtype is bf16 by definition).'
- 'Portal A/B packaging (r2): a candidate may dispatch by (batch_size, seq_len) among same-M shapes (4096 group: 4,1024 / 16,256 /
  8,512 plus 2,2053; 8192 group: 16,512 / 8,1024 / 64,128 / 32,256). Shape-only dispatch is legitimate, but every variant must be a
  real candidate (16/16 correct, one launch); no diagnostic or deliberately wrong variant. The card must list which shape runs
  which variant and all 10 rented per-size times per variant, measured interleaved in one probe.'
- 'Rented differences under about 4% between fused variants are unproven at the portal (fused kernels scale 1.36-1.40 portal/rented,
  plain GEMMs 1.22); interleave variants in the same probe, 3 rounds, and only promote gains of 5% or more.'
hypotheses:
- id: H1
  statement: Fusing the residual as the GEMM's C operand (C != D) removes the reference's separate add kernel at no cost.
  status: refuted
  evidence:
  - 'run_tests: cublasLt fused 8.7..77.5 us vs reference 12.3..90.2 us (matmul alone 8.2..72 us), 16/16 pass'
  - 'probe: torch.addmm(out=) launches a DtoD memcpy of the residual plus the badd nvjet kernel; same time as the reference'
  - 'r0 rented: fused cublasLt 77.5 us at 8192 vs plain nvjet matmul 65-72 us: the add is removed but not for free (see H10)'
  - 'r0 portal: 106 us at 8192 vs Tb 86 and eager ~107; on the portal the fused call is no faster than matmul+add'
  - 'r1 attribution (rented, interleaved): plain matmul 20.6/35.7/70.6 us, fused C!=D 23.2/38.5/76.5, diagnostic C=D 21.2/36.4/71.5
    at 2048/4096/8192: the cost is reading the residual as a separate C tensor, not the algorithm [r1-oproj-cublaslt-forced-algo]'
- id: H2
  statement: The L band is tensor-core bound and loses about 13% to wave quantisation (320 pair tiles of 256x256 over 74 SM pairs
    = 4.32 waves); stream-K or a tile count that divides evenly recovers most of it.
  status: supported
  evidence:
  - 'probe: cuBLAS 1.64 PFLOP/s at M=8192 vs 1.88 at M=9472 (5 full waves) and 1.93 at M=18944 (rented B200, 1965 MHz)'
  - 'probe: M=9472 (exactly 5 waves of 256x256) takes 78.4 us vs 76.1 at 8192: 16% more FLOPs for +3% time [r1-oproj-cutlass-t224-dispatch]'
  - 'r1: transposed 2SM 256x224x64 (370 tiles = 5.00 waves at 8192), epi 128x32, 6 stages, AlongM: rented 64.6-65.0 vs cublasLt
    75.5-76.2 (-14%); portal L 80.5 vs 94.8 (-15%), 8192 at 88.5 vs Tb 86.0, S 0.477 [r1-oproj-cutlass-t224-dispatch, portal r1]'
  - 'dead end: CUTLASS StreamKScheduler 2SM 256x256 84-85 us at 8192 (25 MB workspace), SplitK=2 185 us, 256x128 stream-K 79-82 us;
    untransposed 256x128/256x160 tiles divide better but lose per-tile efficiency [r1-oproj-cutlass-t224-dispatch]'
  - 'remaining L gap is the residual epilogue (H14/H15) and the portal ratio (H17), not quantisation'
- id: H3
  statement: The S band (M <= 1571) is a latency chain of 40 serial K-steps on few tiles; splitting K over all 148 SMs with an
    in-kernel reduction approaches the 6-8 us harness copy floor.
  status: parked
  evidence:
  - 'probe: cuBLAS at M=256 takes 4.9/6.2/8.1 us for K=640/1280/2560; a 17 MB copy takes 6.1-8.2 us under harness timing'
  - 'run_tests: CUTLASS 128x256 (20 CTAs) 25 us and 2SM 256x256 (20 clusters) 18.9 us at M=256 vs cublasLt 8.7 us'
  - 'r1 portal: S band t/Tb = 1.07 (10.6 vs 9.9 at 256 tokens), S 0.47-0.48; a 10% S cut is worth +0.013 (predict_score r2)'
  - 'parked r2: cublasLt split-K is refuted (H12) and a custom split-K kernel is a full session for the smallest lever; revisit
    after the residual epilogue is fixed (the fix applies to S too)'
- id: H4
  statement: 2-CTA tcgen05 MMA (cta_group::2, 256x256x64 tiles) beats 1SM 128x256x64 at every size.
  status: supported
  evidence:
  - 'run_tests: CUTLASS 2SM 18.9/32.9/48.0/73.7 us vs 1SM 25.1/43.0/61.0/95.0 us at M=256/2048/4096/8192'
  - 'cluster 4x1 (untransposed) and 2x2 (transposed 224) are slower: 76.3/84.6 and 76.8 us at 8192 [r1-oproj-cutlass-t224-dispatch]'
- id: H5
  statement: A per-size dispatcher (cublasLt or split-K specialist at S/M, CUTLASS 2SM stream-K at L) beats cublasLt alone.
  status: supported
  evidence:
  - 'r1 portal: dispatcher (cublasLt below 6144 tokens, CUTLASS transposed 224 above) 0.4529 vs cublasLt alone 0.4024; all of the gain is L'
  - 'r2 predict_score: a 10% cut is worth S +0.013, M +0.020, L +0.033; M -12% alone gives 0.478, M -12% plus L -7% gives 0.500'
- id: H6
  statement: SOLAR's compute term uses the theoretical 1.5 GHz tensor peak of about 1.8 PFLOP/s (not 2.25); the practical lock
    peak is about 1.47 PFLOP/s, so the L-band score is capped near 0.66 even at cuBLAS's best rate, and S/M hold the rest.
  status: supported
  evidence:
  - 'probe: best cuBLAS rate 1.93 PFLOP/s at 1965 MHz with no power throttling (300 W of 1000 W); scales to 1.47 at 1500 MHz'
  - 'r0/r1 portal, Tsol back-solved from (t, Tb, S): 60.2 us at 8192, 30.1 at 4096, 15.4 at 2048 = FLOPs / 1.75-1.80 PFLOP/s;
    at 256 tokens Tsol ~ 1.9-2.1 us (ill-conditioned)'
  - 'inferred: t = 73 us at 8192 (practical lock peak, full waves) would score (86-60)/((73-60)+(86-60)) = 0.67'
- id: H7
  statement: The hidden baseline Tb is close to the reference time (PyTorch-only code cannot fuse the residual into cuBLAS).
  status: refuted
  evidence:
  - 'probe: matmul+add and addmm(out=) are within 3% of each other at every size'
  - 'r0 portal: Tb is 7-23% faster than r0 at every size and ~25% faster than eager on the leaderboard'
  - 'r1 E1: Tb / eager rented = 0.81..0.95 (spread 17%), Tb / addmm rented 0.65..0.97; neither is flat. Tb / plain-matmul rented
    is a flat 1.22 (spread 7%), see H16 [r1-oproj-e1-baseline-id]'
- id: H8
  statement: evict_last on the output stores (TMA store cache hint) recovers part of the dirty-L2 flush cost at S/M, as in #38.
  status: open
  evidence:
  - '#38 portal: r5 0.609 vs r3 0.588 (M -8%); untested here'
  - 'blocked cheaply: CUTLASS 4.4 SM90_TMA_STORE has no L2 cache-hint operand; needs a patched epilogue copy op
    [r1-oproj-cutlass-t224-dispatch]. Low priority while H14/H15 are open'
- id: H9
  statement: cuBLAS heuristics pick the same algorithms at the portal's locked clock; the portal/rented ratio follows the
    compute/memory split (compute-bound parts slow by 1965/1500 = 1.31x, memory-bound parts hardly at all).
  status: open
  evidence:
  - 'r0 portal vs rented: ratio 1.22 at S (10.6/8.7), 1.32 at M (30.8/23.3), 1.37 at L (106.2/77.5)'
  - 'r1 portal vs rented: cublasLt-fused M 1.35 (40.7/30.2), CUTLASS-fused L 1.36 (80.5/59.0); Tb / plain-matmul rented 1.22 flat.
    Both fused kernels lose 10-15% more than plain GEMMs at the lock, whatever the algorithm (sharpened as H17)'
  - 'clock lock on the rented box failed (nvidia-smi -lgc: no permission; the GPU runs 1965 MHz under load) [r1-oproj-e1-baseline-id];
    the ratio can only be measured through portal A/B pairs'
- id: H10
  statement: The fused cublasLt call (beta=1, C != D) picks a worse kernel or pays the residual read unhidden, so it runs 7-15%
    slower than torch.matmul's plain nvjet kernel at M/L; forcing the matmul's algorithm (or another heuristic candidate) with
    the residual as C recovers plain-GEMM speed with the add for free. This is the largest lever (about +0.06 to +0.08).
  status: supported
  evidence:
  - 'r0 rented: fused 77.5 us vs matmul alone 65-72 us at 8192, 39.3 vs ~36 at 4096'
  - 'r1 (partly): forcing heuristic algos recovers L to ~2% over plain matmul (71.4-71.9 vs 70.1) and 512/586 by 8%; at M no
    algorithm helps, the residual read (~3 us) is intrinsic to cublasLt''s fused epilogue [r1-oproj-cublaslt-forced-algo]'
  - 'the mechanism (unhidden C read) is confirmed; the fix is a custom epilogue, carried forward as H14/H15'
- id: H11
  statement: Tb is torch.compile of the reference (max-autotune or default), which emits one GEMM with the residual fused in the
    epilogue at cuBLAS-class speed; its per-size rented times scale to Tb by a flat ratio of about 1.2-1.4.
  status: refuted
  evidence:
  - 'r1 E1: torch.compile default and max-autotune-no-cudagraphs lower to aten addmm (DtoD copy + badd nvjet), same time as eager,
    Tb/t 0.81-0.96 not flat; max-autotune with cudagraphs is 2-3x slower under harness timing; Inductor ranks aten addmm above its
    Triton templates (1.4x slower at 8192) [r1-oproj-e1-baseline-id]'
- id: H12
  statement: cublasLt split-K heuristic candidates (SPLITK_NUM > 1, in-kernel or reduce-kernel reduction) beat the default
    algorithm by >= 4% at M <= 1571, a cheap stand-in for a custom split-K kernel (H3).
  status: refuted
  evidence:
  - 'r1: cublasLt returns 8 heuristics per shape, all nvjet with splitK 1; forced split-K 2/4 with fp32 reduction at M=256 takes
    17-19 us vs 8.7 (nvjet splitK kernel + splitKreduce_kernel) and CUPTI matching failed intermittently [r1-oproj-cublaslt-forced-algo]'
- id: H13
  statement: At 2048 and 4096 tokens the transposed 2SM kernel with a whole-wave tile (TN=192 at 4096 and 4106, TN=96 at 2048;
    both 220 pair tiles = 2.97 waves of 74) plus the r1 epilogue settings (128x32 epi tile, 6 stages, AlongM) beats cublasLt by
    5% or more on the rented box, and the same C-read fix as L brings it to plain-GEMM speed (rented 35.6 at 4096, 20.7 at 2048).
  status: open
  evidence:
  - 'r1: transposed TN=192 at 4096 40.2 vs lt 38.4; TN=160 at 2048 25.6 vs lt 22.9 (rented, config not tuned per size; both carry
    the costly C epilogue) [r1-oproj-cutlass-t224-dispatch]'
  - 'tile table (10 feature tiles x token tiles / 74 pairs): 4096: TN=192 220 tiles 2.97 waves, TN=224 2.57, TN=256 2.16, TN=160 3.51,
    TN=128 4.32; 2048: TN=96 2.97, TN=128 2.16, TN=192 1.49, TN=224 1.35; 1SM 128x96 at 2048: 440/148 = 2.97 [inferred]'
  - 'portal r1: 4096 group at 50.4 vs Tb 44.2 (S 0.41), 2048 at 29.5 vs 26.0 (S 0.43); Tsol 30.1 and 15.4, so 43.4/25.3 (plain-GEMM
    speed at the lock) would score 0.515/0.517 and 39.5/21 would score 0.60/0.65 [portal r1, predict_score]'
- id: H14
  statement: The residual C cost in the CUTLASS sm100 epilogue (~6 us = 8% at 8192 rented) is a latency chain, not bandwidth; the
    epilogue issues C subtile TMA loads only a few stages ahead, each a cold-L2 DRAM round trip of about 1 us, and the chain is
    not fully overlapped with the next tile's mainloop. A deeper C pipeline (explicit Sm100TmaWarpSpecialized2Sm StagesC/StagesD,
    smaller epilogue subtiles) or an L2 prefetch of the whole C tile at tile start (cp.async.bulk.prefetch.tensor) cuts the C cost
    to about 2%.
  status: open
  evidence:
  - 'r1: 256x256 with C 76.1 us vs C=void 70.1 at 8192; with stages forced equal (5) noC is 70.9, so the cost is the C load itself;
    EpilogueTileAuto with C wastes smem (67.6 KB epilogue, 5 stages) and 128x32 gives 6 stages (73.5 vs 76.4) [r1-oproj-cutlass-t224-dispatch]'
  - 'inferred: a 256x224 pair tile needs 115 KB of C per 12-15 us mainloop, i.e. under 10 GB/s per pair, so bandwidth cannot be the
    limit; only latency on the critical path explains 1.2 us per tile'
- id: H15
  statement: Folding the residual into the TMEM accumulator before the K loop (epilogue warps load R for the tile two ahead after
    draining the accumulator stage, convert bf16 to fp32, tcgen05.st it into the stage, and the mainloop always runs with
    accumulate=1) removes the C operand from the epilogue entirely; the R load overlaps the previous tile's mainloop, the epilogue
    becomes the C=void epilogue, and the freed smem buys mainloop stages. Numerics are unchanged (fp32 add, one bf16 rounding).
  status: open
  evidence:
  - 'none yet; r1''s C=void numbers (70.1 at 8192 for 256x256; transposed t224 C=void not yet measured) are the target'
- id: H16
  statement: Tb equals the plain cuBLAS GEMM (no add) run at the portal's locked clock; per size it is 1.22 x the rented plain
    matmul time, flat across M.
  status: supported
  evidence:
  - 'r1 E1: Tb / rented matmul-only = 1.20, 1.20, 1.19, 1.25, 1.18, 1.26, 1.24, 1.23, 1.22, 1.23 at the 10 sizes (mean 1.22, spread 7%);
    no whole PyTorch program tested (eager, addmm, compile, F.linear+copy) is flat [r1-oproj-e1-baseline-id]'
  - 'consequence: a fused kernel at plain-GEMM speed scores about 0.5 everywhere; beating 0.5 needs tile efficiency above cuBLAS
    (the transposed whole-wave tiles) with the residual hidden'
- id: H17
  statement: The extra 10-15% that fused kernels lose at the portal (portal/rented 1.36-1.40 vs 1.22 for plain GEMMs) is the exposed
    residual read scaling with the locked SM clock (L2/NoC domain and epilogue instruction time); a kernel whose C read is hidden
    under the mainloop scales at about 1.22-1.27 and lands near 79-82 us at 8192 (S 0.55-0.57) and 44-46 at 4096 (S 0.46-0.50).
  status: open
  evidence:
  - 'portal vs rented: r0 cublasLt-fused L 1.37-1.40, r1 CUTLASS-fused L 1.36, cublasLt-fused M 1.35; Tb / plain rented 1.22
    [portal r0, r1; r1-oproj-e1-baseline-id]'
  - 'cannot be measured on the rented box (no clock lock); tested only by portal A/B pairs on same-M shapes (E2, E3 packaging)'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H13",
    "title": "Whole-wave transposed 2SM tiles for 2048 and 4096 tokens, packaged as a 4096-group portal A/B",
    "operation": "specialisation",
    "parents": ["r1-oproj-cutlass-t224-dispatch"],
    "band": "M",
    "instructions": "Start from the parent's Tn<TN> template (transposed problem D^T = W A^T + R^T, 2SM cluster 2x1, epi tile 128x32, StageCountAutoCarveout, AlongM, max_swizzle 1). Step 1: build Tn<192> and Tn<224> for M=4096/4106 and Tn<96>, Tn<128> for M=2048 (tile counts: TN=192 at 4096 and TN=96 at 2048 are 220 pair tiles = 2.97 waves of 74; TN=224 at 4096 is 2.57 waves, TN=128 at 2048 is 2.16). Also build a 1SM KernelTmaWarpSpecialized1SmSm100 128x96 variant for 2048 (440 tiles = 2.97 waves of 148) as a control. Step 2: time all variants plus cublasLt-fused and plain torch.matmul interleaved in one probe, 3 rounds, at M=2048, 4096, 4106 (rented, harness_time); 8 CUTLASS variants per 120 s probe fit. Step 3: for the best tile at each size, sweep the epilogue: explicit dispatch policy cutlass::epilogue::Sm100TmaWarpSpecialized2Sm<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore> with StagesC 2/4/6 and epi tiles 128x32 and 128x64, raster AlongM vs AlongN, and record mainloop stages from the carveout (print sizeof SharedStorage). Note which settings beat the r1 defaults by >= 4%. Step 4: also time each best variant with C = void (diagnostic only, never submitted) to report the C cost per size. Step 5: deliver one candidate: dispatch by M (cublasLt below 1600 tokens, best 2048 kernel for 1600-3000, best 4096 kernel for 3000-6143, parent Tn<224> above), and inside the 4096 group dispatch by (batch_size, seq_len): 4,1024 -> best 4096 variant; 16,256 -> second-best variant (e.g. the other TN or epilogue setting); 8,512 -> cublasLt-fused (the r1 path, as the control); 2,2053 -> best 4096 variant. Run run_tests (16/16) and put all 10 rented per-size times for every variant and the shape-to-variant map in the design card. Keep every variant a legitimate candidate: one launch, correct, no diagnostic in the submission.",
    "success": "Rented, interleaved, 3 rounds: best transposed kernel <= 36.5 us at 4096 and 4106 (cublasLt-fused 38.4) and <= 21.8 us at 2048 (cublasLt 22.9), i.e. 5% or more, 16/16 correct, one launch; run_tests predicts >= 0.465. The card lists per-size rented times and the A/B map.",
    "refuted_if": "No whole-wave transposed tile, with any epilogue setting, is within 4% of cublasLt-fused at both 2048 and 4096, and the C=void diagnostic of the same kernel is not faster than plain torch.matmul either: then tile quantisation is not the M-band lever and M waits for H14/H15.",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H14",
    "title": "Hide the residual C read in the CUTLASS 2SM epilogue: deeper C stages and an L2 prefetch of the C tile; 8192-group portal A/B",
    "operation": "structural_mutation",
    "parents": ["r1-oproj-cutlass-t224-dispatch"],
    "band": "L",
    "instructions": "Step 1 (diagnose, ~20% of the session): with the parent's Tn<224> kernel at M=8192 and 4096, time C=residual vs C=void interleaved (3 rounds) to fix the C cost for this tile. Read from $CUTLASS_DIR/include: cutlass/epilogue/collective/builders/sm100_builder.inl (how StagesC, StagesD, ReuseSmemC and DelayTmaStore are chosen for TmaWarpSpecialized2Sm with a C operand and a 128x32 epi tile), cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp (the load warp's C pipeline: how many subtiles ahead it issues TMA loads, and where it waits), and cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp (AccumulatorPipelineStageCount / IsOverlappingAccum for a 256x224 fp32 accumulator: 1 or 2 TMEM stages). Report these numbers as findings. Step 2 (pipeline depth): replace the builder's schedule tag with an explicit cutlass::epilogue::Sm100TmaWarpSpecialized2Sm<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore> and sweep StagesC 3/4/7 (7 = the whole 128x224 CTA tile in 128x32 subtiles), StagesD 2/4, ReuseSmemC on/off, DelayTmaStore on/off, epi tiles 128x16/128x32/128x64; keep mainloop stages >= 4 (check the carveout) and record the trade. Step 3 (L2 prefetch): copy sm100_gemm_tma_warpspecialized.hpp into the solution as a local kernel (own namespace), and in the mainloop producer warp, at the start of each tile before the A/B loads, issue cp.async.bulk.prefetch.tensor.2d.L2.global [tma_load_c descriptor, {coords}] for every C subtile of the tile via inline PTX (the epilogue's TMA C descriptor is in the epilogue params; prefetch the descriptor first). Time against step 2's best. Optionally also try prefetching one tile ahead. Step 4: pick the two best configurations (e.g. deepest C pipeline, and L2 prefetch) and deliver one candidate identical to the parent except that the 8192 group is dispatched by (batch_size, seq_len): 16,512 -> parent Tn<224> (control); 8,1024 -> variant A; 64,128 -> variant B; 32,256 -> variant C or the best of A/B repeated. Apply the best variant also to 8,997 if it wins there. run_tests 16/16; put all 10 rented per-size times per variant and the shape-to-variant map in the card. The portal result then gives the portal/rented ratio per variant (H17): report in findings what the ratio would need to be for the 8192 score to pass 0.55.",
    "success": "Rented, interleaved, 3 rounds: the C-vs-void gap of Tn<224> at 8192 drops from the measured value (expected ~5-6 us) to <= 2 us, i.e. fused time <= 61 us at 8192 and <= 60.5 at 7976, 16/16 correct, one launch, no workspace kernel; run_tests predicts >= 0.47.",
    "refuted_if": "Neither a C pipeline of 7 stages nor the L2 prefetch (which makes every C load an L2 hit) shrinks the gap below 4 us: then the cost is epilogue serialization (accumulator stages or epilogue issue time), not load latency, and H15 (E3) is the route.",
    "model": "opus"
  },
  {
    "id": "E3",
    "hypothesis": "H15",
    "title": "Exploratory: fold the residual into the TMEM accumulator before the K loop (no C operand in the epilogue)",
    "operation": "structural_mutation",
    "parents": ["r1-oproj-cutlass-t224-dispatch"],
    "band": "all",
    "instructions": "Goal: a 2SM transposed CUTLASS kernel whose epilogue is the C=void epilogue, with the residual added by initialising the accumulator. Step 1 (baseline, 10 min): time Tn<224> with C=void at 8192 and Tn<192>/Tn<96> with C=void at 4096/2048 (rented, interleaved with the fused parent): these are the targets. Step 2 (read): cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp (warp roles, accumulator_pipeline producer_acquire in the MMA warp, consumer_release in the epilogue warps, the tile scheduler loop each role runs), cutlass/gemm/collective/sm100_mma_warpspecialized.hpp (tiled_mma.accumulate_ = UMMA::ScaleOut::Zero on the first k-block, One after), cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp (the tcgen05.ld partition: tiled_t2r / SM100_TMEM_LOAD_32dp32b, the lane-to-warp mapping, and where the accumulator stage is released). Step 3 (build): copy those three headers into the solution under a local namespace and patch: (a) mainloop always uses ScaleOut::One; (b) add one mbarrier per accumulator stage ('acc_init') in shared storage; the MMA warp waits on it (parity per use) after producer_acquire and before the first MMA of each tile; (c) the 4 epilogue store warps, after the last tcgen05.ld of tile i on stage s and before consumer_release, compute the tile that will next use stage s (tile i + AccumulatorPipelineStageCount in this CTA's scheduler sequence; skip if none), load its R subtiles straight from global with 16-byte LDG (bf16x8; the transposed C/D layout is column-major [N,M], so a CTA's R block is 128 features x TN tokens with features contiguous; mask token rows >= M with zeros), convert to fp32, tcgen05.st into stage s using the mirror of the tcgen05.ld partition (SM100_TMEM_STORE_32dp32b, same lane/column mapping), tcgen05.wait::st, fence, then arrive on acc_init[s] and consumer_release; (d) before the main loop, the same warps pre-fill stages 0..S-1 with R of this CTA's first S tiles and arrive. If AccumulatorPipelineStageCount is 1 for 256x224 (overlapping accumulator), use TN=192 or restructure so two stages fit (2 x 192 = 384 columns), and note the finding. Step 4: validate against the reference on random inputs including the residue shapes (4106, 7976, 586), then run_tests 16/16. Step 5: time at all 10 sizes interleaved with the parent and the C=void targets; if it wins at L and M, deliver a candidate dispatched by M (cublasLt below 1600, R-init kernels above) and, if two configurations are close, use the 8192 group for an A/B (16,512 -> parent control; others -> R-init variants), listing all per-size rented times in the card. Budget: if step 3 is not correct by 60% of the session, deliver the diagnosis (which barrier or partition failed, measured cost of the tcgen05.st path) as findings instead of a candidate.",
    "success": "R-init kernel correct 16/16 and within 2% of the C=void kernel's rented time at 8192 (expected ~59-61 us vs 64.8 fused) and at 4096 with Tn<192> (expected <= 36 us vs cublasLt 38.4); one launch; run_tests predicts >= 0.48.",
    "refuted_if": "The R-init path (LDG + cvt + tcgen05.st per tile) costs as much as the C epilogue it replaces, or the acc_init handoff stalls the MMA warp so the kernel is no faster than the fused parent; or the accumulator has only one TMEM stage at every usable TN so the R fill cannot overlap the mainloop.",
    "model": "opus"
  }
]
```