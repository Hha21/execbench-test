### Assessment

The gap to the live first place is +0.008 of mean. The score model says M −8 % with L −6 % reaches 0.746, so two mid-sized row-group wins close it; S rows stay as they are (no family has a credible −4 µs there in one session).

What the lineages taught in r9: D's phase A is bound by the W1 DRAM stream, not a copy chain, and barriers are worth ≤ 0.5 µs; D's pair GEMM2 pays only at 32,768; G's tcgen05 engine has a per-instruction throughput floor below N≈208 and its 8,320 / 4,96 rows still spend 20 / 12 µs in K3 + cuBLASLt G2; the tile tails at 16,384 / 32,768 / 16,768 (44–116 CTAs with two whole-image tiles) are the largest untouched term on the lowest-S rows.

Audit answers: (1) TF32 zero-conversion: the critic's arithmetic stands (393 KB fp32 per tile owner at 40–94 GB/s/SM is 4–10 µs against the 2.5 µs grid-wide pass); I1/I20 parked. (2) The ~105 clk mechanism is unsupported; the measured throughput number is what the tile-N choices use, and E2's N=256 MMAs will show whether 1SM reaches the ideal there. (3) H39 refuted on mechanism, H2/H18 and D's conclusions corrected below. (4) S-row worth corrected (S −20 % = +0.0128); no S experiment this round for lack of a p > 0.2 family. (5) Flat barrier: stress loop + protocol check + STACK scheduled as STEP 0 of E1, with cg grid.sync as the fallback.

This round: E1 (D) balanced image-straddling token-range tiles with no g and a scale-in-smem GEMM2 (five tc rows, −8..−15 %); E2 (G) a custom tcgen05 GEMM2 that retires K3 and cuBLASLt G2 at 8,320 / 4,96 (−17..−19 %, one launch fewer); E3 (new lineage I) the top-row pass 1 rebuilt at its pipe bounds, gated by a conv-only probe. H is retired (incubation over, ceiling argued below D).

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (bulk/TMA-staged dwconv+LN, cuBLASLt FP16 GEMMs, lean one-round elementwise kernels, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: retired
    started: r1
    best: r5-a-e2-ringk2
    conclusions:
      - Retired r9. Holds no row after r8-merge; a 5-6 launch pipeline runs at portal ratio 1.40-1.55 on S/M rows against 1.20-1.32 for single launches, so its ceiling is below D on every row.
      - Keep. Each custom launch is a 2-3 dependent-round chain (6-17 us), not a 2 us launch; launch count is a first-order axis on S/M rows.
      - Keep. K2 is not stream-bound (59 % of copy rate at 64,96); the ~100 us left is FMA/INT issue of the exact erf, so only a GEMM-epilogue GELU removes it.
      - Keep. cuBLASLt FP16 is 1.4-2.1x faster than TF32 everywhere; its GELU epilogue is the tanh form (rejected); swap-AB batched NCHW GEMM2 is right at HW >= 784 and 20-190 us slower at HW = 49/196.
      - Keep. Bulk-plane conv and the TMA 3-D box ring for W%4==0 rows are the conv carriers every lineage reuses; cross-host rented drift is 5-12 us, so every pick needs an interleaved same-container control.
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), aiming at S rows <= 24 us rented by cutting the per-phase chain between the two grid barriers.
    status: retired
    started: r1
    best: r3-coop-flat-bulkrows
    conclusions:
      - Retired r8. D's sc_kernel (same single-launch structure, tcgen05 engine) runs 1,192 at 19.4 and 1,96 at 19.1 us rented against B's 29.5-30.6.
      - Keep. Single-launch coop beats a launch pipeline on the portal even where the bench says otherwise (ratio 1.25-1.31 vs 1.40-1.55).
      - Keep. Barrier microbench (r4): flat atomic-counter barrier 0.86-1.07 us vs 1.6-1.9 us for the cg flip barrier; a two-level tree is worse (3.9 us); one flag polled by 295 CTAs is observed in 0.64 us.
      - Keep. Per-tile dataflow (H27) lost 1.4-2x for skeleton reasons, and the image-level version (H38) lost on per-item cost; counters are cheap but do not pay.
      - Pitfalls. runtime guards around mma.sync cost ~470 clk per k-block; register-direct erff epilogues take 16 us per tile; a release arrive with loads in flight stalls.
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - Retired r3. H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) refuted; no edge over a materialised fp16 hidden.
      - Keep. One-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM; converting inside the ring is 2-2.5x slower.
      - Keep. TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs.
      - Keep. Cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry.
      - Conv redundancy 4C/BN and per-CTA serial rounds dominated; the all-N token-tile family is dead for C >= 384.
  - id: D
    thesis: Cooperative single-launch specialist (bulk-plane conv, per-tile GRN statistics, split-K GEMM2 with coalesced red.add) whose GEMM engine is hand-written tcgen05 kind::f16 in the swap-AB orientation (weights as M, tokens as N), so the output tile is NCHW; the six wide-C rows (tc_kernel) and the C <= 192 rows, 1,160 and 4,96 (sc_kernel).
    status: active
    started: r3
    best: r9-d-e1-asynca
    conclusions:
      - Holds 12 of 16 rows in r9-merge (0.7359); portal ratios 1.20-1.32; M/L projections within +-2 %; S rows move +-1.2 us between submissions with unchanged code.
      - Remaining terms per wide-C row (r9 stamps, corrected): phase A 5.5-7 us at C=768 bound by the W1 fp32 DRAM stream (2.5 us of phase A, 3.2 us at sync 1; the conv ends 0.9 us after its planes land, it is not 15x its bound); barrier releases 1.0-1.5 us each (flat counter gains 0.1-1.3 us on tc rows, loses 1-2 us on sc rows); GEMM1 post-accumulator 3.0-8.1 us per tile bound by h+g store bytes; GEMM2 ~0.7 us per 48 KB stage (1SM) and 2.5-3.2x the pair ideal before per-stage signalling was pipelined; whole-image tile tails (44 / 116 CTAs with 2 tiles at 16,384 / 32,768) are the largest untouched term.
      - Validated and reusable: MN-major SW128 B descriptor (LBO 8192, SBO 1024, +2048 per k16), K-major SW128 A box (SBO 1024, +32 B per k16), idesc bits, cta_group::2 pair recipe (whole kernel must be one cta_group; coop + cluster 2x1 launch works; pairs pay only where a CTA runs many stages), one producer warp owning a whole stage with stages round-robin over warps (pipelines fence + arrive), all-warp epilogue at 512 threads, staged coalesced h stores, target_hardware [B200] only, STACK check in cuobjdump.
      - Dead ends: weight warps split from conv warps; lighter epilogue arithmetic; 2-deep pipelined tcgen05.ld; g built in smem inside the 4-stage 48 KB ring; L2 prefetch of W2 in the sync-1 window (moves cost, no net); pairs at B <= 8 (fixed coupling costs); reordering W1 loads vs conv (ties, phase A is bandwidth-bound).
      - Precision: tc paths 59-65 % of atol (hidden rounded once), sc paths 79-89 % (hidden rounded twice via fp16(h*s)); a no-g tc path moves to the sc class and must be logged over 10 rounds before a slot.
  - id: E
    thesis: Library tcgen05 with fused epilogues (CUTLASS 4.4 sm100 builder GEMMs whose EVT epilogues carry the LN fold, exact erf GELU, hidden store and per-image GRN sum of squares; TF32 operands straight from fp32 memory where the MMA is cheap).
    status: retired
    started: r5
    best: null
    conclusions:
      - Retired r8 (H30 refuted r5). Per-image batching re-reads W1 once per image and TF32 doubles the bytes; G1 with the full EVT was 17.0 us at 8,768 against D's whole row at 33.
      - Keep. EVT nodes that instantiate on sm100 TmaWarpSpecialized1Sm: Sm90ColBroadcast, Sm90RowBroadcast, Sm90Compute<multiply_add>, Sm90Compute<GELU> (erf form), Sm90RowReduction with FinalReduction=false; the erf GELU costs +2.4-8.5 us when the epilogue does not overlap a one-tile-per-CTA mainloop.
      - Keep. CUTLASS tfloat32_t on fp32 memory rounds to nearest (H14); split-K as the batch mode works on sm100; three .cu files build in 26-50 s wall.
      - Unresolved. run_tests 14/16 twice from CUPTI sequence-matching errors on changing rows; any multi-kernel library path must clear this before a portal slot.
  - id: F
    thesis: Library-lean four-launch pipeline for the HW<=784 rows (LayerNorm finished inside the conv launch via a cluster-of-8 DSMEM statistics exchange, cuBLASLt FP16 GEMM1, a one-launch GRN statistics-and-scale kernel with per-image counters, cuBLASLt swap-AB batched GEMM2 writing NCHW) that wins by launch count.
    status: retired
    started: r6
    best: r6-f-e1-libpipe
    conclusions:
      - Retired r8 (H32 refuted): D r6 beat F by 4-10 us on every wide-C row and D r7 sc_kernel beat F's best S row (1,192 19.4 vs 26.6 rented, single launch vs four).
      - Keep. cuBLASLt's GELU epilogue is the tanh form and fails the own-atol bar; folding gb into the fp16 hidden also fails. The exact erf on an fp32 pre-activation and gb kept in b2' = W2.gb + b2 are required.
      - Keep. One-launch GRN kernel with a plain launch when grid <= residency: 64-bit red.add, ld.relaxed poll, departure counter resetting the cell; plain launch saves 1.8 us over cooperative.
      - Keep. Cluster-of-8 LN exchange costs 0.4-2.4 us; L2 weight prefetch at K1 entry had no measurable effect; conv and weight conversion serialise inside a CTA.
      - Precision log delivered for all 9 F paths (0 beyond atol, 59-76 % of atol).
  - id: G
    thesis: Fused pass-1 persistent kernel for the HW >= 1024 rows (64,96; 32,128; 8,320; 4,96): CUDA-core warps stage x and run the 7x7 conv plus LN statistics, writing the fp16 d tile straight into shared memory as the GEMM1 B operand (swap-AB, W1' as A; tcgen05 at 8,320 / 4,96, mma.sync still at the top two rows); epilogue warps apply the LN fold, an exact erf GELU, per-(image,k) sumsq and the fp16 hidden store; the GRN scale and GEMM2 finish the row in as few launches as possible.
    status: active
    started: r7
    best: r9-g-e3-tc05-p1s
    conclusions:
      - Holds 4 rows in r9-merge: 64,96 (portal 1079, S 0.801) and 32,128 (288, 0.771) from r7 g_pass1 (mma.sync); 8,320 (74.5, 0.763) from the r9 tcgen05 g_tc (59.0 rented, -17.5 %); 4,96 (46.7, 0.744) from g_tc (37.2 rented); 4-launch pipeline at portal ratio 1.26-1.27.
      - tcgen05 engine validated at C=96/320 (writer address functions in the notebook); 1SM kind::f16 128xNx16 throughput ~105 clk per instruction for N <= 128 (ideal only at N >= ~208); conv-overlapped MMA in channel rounds and per-warp smem-transposed h stores are the shipped levers; ring depth 9 vs 10 no difference.
      - Remaining per-launch at 8,320 (rented): K0 ~4 / P1 ~39 (epilogue ~16 us, ~2x its issue bound; per-MMA floor at N=64) / K3 8.9 / cuBLASLt G2 11.3 (3.8 TB/s, 8-algo sweep found nothing better). At 4,96: K0 3.5 / P1 ~23 / K3 5.6 / G2 6.6.
      - Top rows: pass 1 at 64,96 is 668 us (conv side alone 472 vs a 102 us FMA bound, 3-row prefetch chain through L1/L2; mma.sync 100-130 us serialised in the epilogue warps; A&S GELU 2 MUFU per element = 133 us floor); x cannot be staged beside a resident W1' at R=2, so W1' must stream.
      - Precision logged for all 4 shipped rows (61-82 % of atol, 0 beyond). Pitfalls: harness_time CUPTI drops (retry wrapper); pass every file's content to run_tests; NVRTC probes cannot include <stdint.h>.
  - id: H
    thesis: Image-counter dataflow in one persistent tcgen05 kernel for the B >= 8 wide-C rows: a static work queue claimed through an atomic head, each item gated on per-image readiness counters instead of grid barriers.
    status: retired
    started: r8
    best: null
    conclusions:
      - Retired r10 (incubation over). H38 refuted: +58-98 % vs D on all five rows; summed item CTA-time / 148 = 47 / 111 / 72 us at 8,768 / 32,768 / 16,384 against D's whole rows 34 / 64 / 58, so perfect packing cannot win; the ceiling is below D's current rows.
      - Keep. Queue mechanics are cheap (claim-to-start 1-2 us, inter-item gap 1.5-3.3 us, dependency spin 5-15 % of item time); counters self-reset through a departure counter; the lint rejects __nanosleep in spin loops.
      - Keep. Engine recipe validated on the first build and recorded in the notebook (idesc, A/B descriptors, N values, tcgen05.ld form); CUTLASS-style release (bar.sync then red.release.gpu) plus consumer ld.acquire + fence.proxy.async is correct for TMA readers in other CTAs.
      - Keep. LN-statistics prestage must be a fully unrolled predicated load array; weight conversion must load JB=8 float4 into registers before any shuffle; scalar red.add volume scales with split-K (16K per item ~4 us).
      - Revival only through intra-CTA overlap of item i's epilogue with item i+1's mainloop, which is D's engine territory.
  - id: I
    thesis: Top-row specialist (64,96 and 32,128): a pass-1 kernel with fixed warp roles that puts the conv on the FMA pipe from bulk-staged x row bands (register-blocked patches, 49 taps in registers), the GEMM on tcgen05 against a streamed W1', and the LN fold / A&S GELU / sumsq / h stores on epilogue warps, so pass 1 runs within 1.5x of the SM pipe bound (~200 us at 1.95 GHz) instead of 668 us; the GRN scale and GEMM2 follow the notebook recipe (red.add sums, per-image fp16 W2_b, cuBLASLt batched swap-AB writing NCHW).
    status: active
    started: r10
    best: null
    conclusions:
      - Seeded r10 from pool I18 (with I3 as the fallback step). Ceiling argument: 64,96 at ~560 us rented is ~700 portal (S 0.85 vs 0.80) and 32,128 at ~150 rented is ~190 portal (S 0.83 vs 0.77): about +0.006 of mean, +0.01 at the pipe bounds; the two rows have the largest absolute S headroom on the board.
      - The FMA pipe is the binding resource at 64,96 (conv 102 + GELU FMA ~80 + stats/stores ~20 = ~200 us at 1.95 GHz) ahead of MUFU (133) and DRAM (~140), so the balanced erf (I30) must not move MUFU work onto FMA.
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r10 2026-10-14.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window. No programmatic dependent launch at all, even between our own kernels: the lint rejects griddepcontrol and the PDL launch attributes, so an idea that relies on PDL must be built without it.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot.'
- 'Precision: FP16 and TF32 operands with fp32 accumulation pass every workload with ~5 sigma margin (simulated with the harness generators); BF16 fails. Any reduced-precision candidate must be re-simulated or run_tests 16/16 before a slot; the erf GELU must stay exact (no tanh approximation).'
- 'Weights are converted/transposed inside the call (no cross-call cache of fp16 weights or descriptors; pointers shift every call).'
- 'Workspaces come from torch.empty inside run() (no kernel launched); a cooperative launch counts as one kernel. Grid barriers via cg::grid_group or a self-resetting counter only.'
- 'Host state shared across shapes (function attributes such as max dynamic shared memory, cached plans, handles) must be right for every shape in any order: the portal sets up every workload before it times them (#65570: per-shape cudaFuncSetAttribute shrank the limit for earlier shapes, code 101 on 8 workloads). Set such attributes to the maximum any shape needs, or only ever raise them. run_tests re-runs every workload after all have run once and reports a PORTAL-ORDER CHECK failure.'
notes:
- 'Report effects per workload (or at least C=768 rows vs others), not per MB band: the MB bands mix weight-bound (C=768 7x7), latency-bound and stream-bound rows. Card fix pending: the MB cell for 64,96,112,112 is 616.9 MB, not "118.0 G".'
- 'Where the score is (r10; r9-merge #66880 = 0.7359, projection 0.7351; live first place 0.7435, so +0.008 of mean = +0.12 summed S is needed). Per-row S, lowest first (portal us, path): 16,384 0.689 (67.5, D tc r8); 16,768 0.697 (56.7, D tc r8); 8,768 0.702 (42.0, D tc r8); 8,384 0.704 (44.4, D tc r8); 32,768 0.710 (79.0, D r9 pair); 2,768 0.720 (30.7, D r9 asynca); 8,192 0.728 (54.7, D sc r8); 4,192 0.735 (37.1, D sc r8); 2,96 0.738 (33.3); 1,160 0.739 (33.2); 4,96 0.744 (46.7, G r8); 1,192 0.747 (23.3, D sc r7); 8,320 0.763 (74.5, G r9); 32,128 0.771 (288.0, G r7); 1,96 0.788 (23.8); 64,96 0.801 (1079.0, G r7). predict_score r10: M -8 % and L -6 % = 0.7461; M -3 % and L -12 % = 0.7476; M -12 % and L -10 % = 0.7522; a 10 % cut is worth +0.0062 (S) / +0.0069 (M) / +0.0080 (L). Audit r10 correction: S rows are NOT worth less than M/L rows (S -20 % = +0.0128, more than the whole gap); what is true is that changes of ~1 us there are invisible (+-1.2 us between unchanged submissions) and that no family has p > 0.2 of -4 us at an S row in one session, so none is scheduled in r10. The six wide-C rows remain the cheapest lever: lifting all six to S = 0.74 is worth about +0.017 of mean.'
- 'Portal/rented ratio (audit r4, extended r6-r9, replaces the uniform 1.28): per path, per row AND per launch structure. On S/M rows the 5-6 launch A pipeline runs at 1.40-1.55x its rented time, the single-launch cooperative B/D kernels at 1.20-1.32x (D sc_kernel 1.20-1.27, D tc_kernel 1.20-1.32, G''s 4-launch pipeline 1.26-1.27). Rented hosts differ in launch speed and the portal behaves like the slow host. Rule: each launch removed on an S/M row is worth about 1-1.5 us more on the portal than the bench shows; judge S/M designs by launch count as well as rented time; project each path with its own lineage ratio (r7-merge projected 0.7208 vs 0.7207 measured; r8-merge 0.7333 vs 0.7313; r9-merge 0.7351 vs 0.7359).'
- 'Audit r10 answers. (1) Zero-conversion TF32 for the C=768 rows (I1/I20, H14): the critic''s arithmetic is accepted and the ledger''s r9 "next lever" line is withdrawn: only the 24 / 96 tile owners would read W1, 393 KB fp32 per tile at 40-94 GB/s/SM = 4-10 us against the 2.5 us grid-wide pass, and TF32 doubles the MMA count against a per-instruction floor; parked without a slot (H14 evidence updated). (2) The ~105 clk per-instruction floor at N <= 128 (G r9): the throughput number is measured and stands for 1SM operands in smem; the A-slice-fetch mechanism is unsupported (the pair probe did 128x128x16 of work per SM in 86 clk). Tile-N choices rest on the measured number; E2 (H42) issues N=256 MMAs in 1SM mode and reports clk per instruction, which tests the "ideal at N >= 208" claim. (3) Phase A at C=768 as a serial copy chain with the conv at 15x its bound: wrong, corrected in H2, H18, H39 and lineage D (r9 stamps: copies issued at entry, conv ends 0.9 us after landing, W1 stream binds). (4) S-row worth: corrected in the "Where the score is" note. (5) Flat self-resetting barrier safety on tc_kernel: scheduled as STEP 0 of E1 (protocol check of release/acquire plus fence.proxy.async on the waiting side before any TMA of other CTAs'' generic writes, cuobjdump STACK on the shipped build, 1500-call stress loop at 8,768 and 16,384 against the torch reference); if anything fails, tc_kernel reverts to cg grid.sync (cost 0.1-1.3 us per tc row).'
- 'Pool families (dedup r10; built or refuted ones are not re-run): balanced image-straddling tiles (I5, I9: built as E1 = H41); G fusion / custom GEMM2 (I6, I8: built as E2 = H42, 3 launches first, the single-coop form later); fused top rows at pipe bounds (I18, with I3 as its fallback step: built as E3 = H43, lineage I); one-MMA [h|g] GEMM2 (I19: superseded by the no-g form of H41; fallback for the C=768 rows if H41 fails at 7x7); zero-conversion TF32 (I1, I20: parked, audit r10); cluster split-K (I2, I13: p 0.15, exchange cost ~1.2 us per CTA at 48-77 KB on measured DSMEM/L2 rates, not scheduled); TMEM-sourced A operand (I4: unbuilt, 10-min microbench optional in E2 if >= 3 B200 min remain, kill if N=64 stays >= 80 clk); cache hints (I7, I10: cheap, optional STEP in E1 if time remains: W1/W2 loads evict_first interleaved x3 at 8,768 / 32,768, ship if >= 0.7 us); bulk-reduce epilogue (I11: unbuilt, probe deferred); right-sized S grids (I12: noise-level, not scheduled); in-ring weight conversion (I14, I16: unbuilt, p 0.25-0.3, -1..-2 us per tc row, after H41); tensor-core Toeplitz conv (I15: long shot, kept); owner-converted sc weights (I17: unbuilt, S rows, next round if any); balanced erf (I30: refuted by arithmetic for 64,96, the FMA pipe binds there, see H43); dataflow/counters (H27, H38 refuted, lineage H retired r10); library EVT (E retired); library-lean (F retired); on-chip hidden (C retired).'
- 'Precision logging (audit r2, tightened r3, delivered r5-r9 for every shipped path): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the rms. Every run_tests records per workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded rounds (seed 200) against an fp64 reference, for every path the lineage ships. A row with any beyond-atol element switches the cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M). Passing through the 98 % ratio is not acceptable. A path whose arithmetic (roundings, summation order of a reduction) is unchanged may cite the previous log plus a single-seed check. A path that rounds the hidden twice (fp16 h then fp16(h*s), the sc_kernel class at 79-89 %) must be logged afresh on every row it ships; the cuBLASLt GELU epilogue is the tanh form and fails this bar; the A&S erfc form (1.2e-6 in h) is accepted as exact.'
- 'Cross-session rented timings drift by ~6 us per call and across Modal hosts by 5-20 us on unchanged rows. Every experiment times its parent and the merge interleaved x3 in its own container. Rented within-noise picks lose on the portal (r3 merge at 8,768); the merge step times candidates side by side in one container and projects with per-lineage ratios.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65; 8,192 0.63; 32,128 0.62; 64,96 0.65.'
- 'Audit r3/r4 on tcgen05 (H23): the ideal GEMM savings on the four GEMM-heavy rows are worth about +0.010 of mean, +0.005 against cuBLASLt; but tcgen05 is also the enabler for fused epilogues, because the custom mma.sync engine lost 4x to cuBLASLt (H26). Routes: D''s hand-written engine (H28, H34, H36 supported; H33 partly), G''s fused pass (H35 open, H40 partly supported), E (H30 refuted), F (H32 refuted), H (H38 refuted).'
- 'Cross-problem code: a design session may read another problem''s archive kernel as a structure reference (the #35 tc_ws tcgen05 ring, tricks s5); within this problem lineages still share conclusions only. get_kernel on D''s kernels returns 75 KB, above the tool limit; sessions should request the kernel''s card and sources in parts or rely on the notebook recipes (H''s r8 engine recipe and G''s r9 writer address functions are complete). A lineage that builds a tcgen05 engine validates it against torch before anything else.'
- 'Per-SM operand ingest (H31, corrected r6, extended r9): a tcgen05 tile fed from L2 through a TMA ring lands 68-92 B/clk/SM in 1SM mode, the cap is SM-side, and a cta_group::2 pair halves per-SM B bytes and reaches the MMA ideal at N >= 224. In 1SM mode a kind::f16 128xNx16 instruction has a measured ~105 clk throughput floor for N <= 128 (ideal at N=256), so small-N tiles are instruction-bound, not ingest-bound. r7 sc_kernel: the GEMM2 B-tile scale costs 0.3-0.45 us per 64-k block (TMA landing, not arithmetic). r8 D: a 4-stage ring of 48 KB stages at 7x7 is latency-bound (~0.7 us per stage for two N-tile MMAs); work inserted between landing and MMA extends every stage unless the per-stage fence + arrive are pipelined (r9 E2 v3: one producer warp owns a whole stage, stages round-robin over warps).'
- 'r6-r9 engineering lessons: (a) a per-thread array the compiler cannot keep in registers becomes a local stack frame (cuobjdump STACK); a stamp build alone can push a 512-thread epilogue over the 128-reg cap, so stamp numbers are relative. (b) globaltimer stamps from the warp that does the work. (c) Coalesce 16-B hidden stores through a per-warp smem transpose (D r6, G r9: 8,320 row -4.2 us). (d) target_hardware [B200] only for tcgen05 solutions. (e) torch extension builds define __CUDA_NO_HALF_CONVERSIONS__: use __half2float. (f) Weight conversion first on all warps, then the conv; reordering loads vs conv ties because phase A is bandwidth-bound. (g) Register stats over plane subgroups beat per-element post-conv passes. (h) A token-permuted d layout lets each lane store 8 consecutive tokens per hidden row with one STG.128. (i) Small vectors via __ldg from a K0-padded buffer or by bulk copy with the planes. (j) Fully unrolled predicated load arrays, never runtime-bounded unrolled loops. (k) ptxas forbids mixing cta_group::1 and ::2 tcgen05 ops in one kernel. (l) Pass every file''s content to compile_b200/run_tests (the paths field does not pull archive files).'
- 'r4/r8/r9 barrier and dataflow lesson: cheap release-to-observe (0.64 us for one flag polled by 295 CTAs; flat atomic barrier 0.86-1.07 us; cg grid.sync release maxima 1.0-1.5 us in situ) does not translate into kernel time; barriers are worth at most ~0.5 us per sync on any wide-C row, and the flat counter loses 1-2 us on the skewed sc rows. The per-tile and image-level dataflows lost for per-item serial-chain reasons.'
- 'r5 K2 lesson (H29): a register-prefetch or smem-ring K2 reaches only 47-59 % of the plain copy rate because the per-element math is FMA/INT issue, not MUFU; the A&S erf form is 1.16e-6 from erff. SIMT bounds at 64,96 (rented 1.95 GHz): conv 102 us, exact GELU at 20 instr/element 167 us, A&S erfc 2 MUFU per element 133 us; conv + GELU FMA + stats share the FMA pipe (~200 us), which is the binding pipe for a fused pass (H43).'
- 'Cluster note (r6, r9): the constraint allows clusters where a GEMM tile or split-K cell needs them; cooperative launch combined with the 2x1 cluster attribute works (cudaLaunchKernelExC), cudaOccupancyMaxActiveClusters = 74 at 512 thr / 227 KB, grid.sync costs 1.35 us with or without clusters.'
- 'Retirements and pauses: r3 C; r8 B, E, F; r9 A; r10 H (incubation over, ceiling below D). Active: D (one experiment, E1), G (one, E2), I (new, E3). The half rule holds: D has 1 of 3.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: FP16 is 1.4-2.1x faster everywhere'
  - 'r5 A precision log (fp64 reference, 10 seeded rounds, 16 rows): 0 beyond-own-atol elements on every row, max |err| 2.427e-3 vs atol 2.7e-3 at 1,96 (tightest); r5 E TF32 paths 1.67-1.94e-3; r6 D 1.70-1.95e-3 (59-67 %); r6 F 1.61-2.07e-3 (59-76 %)'
  - 'r6 F: the tanh GELU (cuBLASLt epilogue) fails, and folding gb into the fp16 hidden fails: the exact erf on an fp32 pre-activation and gb kept in b2'' are required'
  - 'r7-r9: D sc_kernel 79-89 % of atol (hidden rounded twice), D tc 59-65 %, G 61-82 %, H 58-64 %, D r9 pair path single-seed 1.48-1.70e-3. Every shipped path is logged'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at the top two rows because the implementations are staging/load-path-bound, not FMA-bound (G's conv side 472 us at 64,96 against a 102 us FMA bound); at the wide-C small-image rows the bulk-plane conv is cheap (ends 0.9 us after its planes land) and is not a lever.
  status: supported
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); ~8 GB/s per SM through LDG/LDGSTS staging'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96. r3 D: bulk-plane conv 1.5 us at 2,768, 6 at 8,768, 10 at 32,768'
  - 'r4 A (E1): bulk-plane K1 interleaved x3: 2,768 18.1 -> 9.7; 8,768 23.2 -> 12.2; 32,768 33.4 -> 14.2; 16,384 28.9 -> 16.7; 1,160 17.9 -> 5.2 us'
  - 'r7/r8 G: inside the fused pass the conv side alone is 472 us at 64,96 (L1/L2 load path, serial 3-row prefetch chain, 4x halo redundancy at R=2); an L2 prefetch of the x halo changed nothing'
  - 'r9 D stamps at 8,768 (audit r10 correction): plane copies issued at entry, landed at 1.5 us without W1 traffic (2.7 us behind the W1 stream), conv end 0.9 us after landing; the earlier "15x its FMA bound" reading measured the wait for the W1 stream, not the conv. At C=768 the conv is not a lever; H43 attacks the top rows'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes; elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal at 134 MB; at the largest row K2 alone took 447 us'
  - 'r7 G materialises the fp16 hidden once from the fused pass and reads it once in GEMM2 (186 us DRAM-bound at 64,96); the pass itself is SIMT-bound at 668 us, so recompute stays parked until the conv and GELU reach their bounds (H43)'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192 and 46.7 vs 50.2 at 2,768; stubbed floors 4.6 vs 9.9 us'
  - 'portal r6: the coop kernel''s portal ratio is 1.20-1.32 against 1.40-1.55 for the 5-6 launch pipeline on S/M rows, so the portal difference is larger than the bench difference'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7) split-K is required to use the SMs; at 1,192,28,28 it is not needed.
  status: supported
  evidence:
  - 'r1 B ablation, split-K auto vs forced 1: 2,768 46.5 vs 64.6 us, 8,768 86.1 vs 120.6; 1,192 29.9 vs 30.6'
  - 'r4 D gate: 768x1568x3072 without split-K 20 us warm (48 CTAs). r8 H: scalar red.add volume scales with split-K (16K per item ~4 us), so KS=2 beat KS=4 by 7-10 us per C=768 row in a design with per-item red'
- id: H6
  statement: The fp32 SIMT GEMM core runs at 46-50 TF/s on the rented card, i.e. 35-38 TF/s at the portal lock; the fp32 SIMT ceiling mean (0.680) is below the leader, so SIMT GEMMs cannot reach the top.
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s, 57-60 TF/s without global loads; ablations no-epilogue 34-35 TF/s, no-affine 33-35'
  - 'portal #65377 vs rented: 1.30-1.32'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes, so per-size tile tables (or image-straddling token ranges) matter as much as the core rate.
  status: supported
  evidence:
  - 'r1 B: 1,160,449,7 P3 ~26 us with N=160 padded to 2x128 tiles. r4 D gate: 3072x1568x768 = 168 CTAs = 1.14 waves (4.3 us per CTA)'
  - 'r6 D stamps: 192 / 264 whole-image GEMM1 tiles on 148 CTAs at 16,384 / 32,768 leave 44 / 116 CTAs with 2 tiles; at 16,384 the GEMM1 epilogue ends at 32.3 us median vs 44.5 max'
  - 'r7 D sc_kernel stamps: 3 GEMM1 chunks per CTA on 40 CTAs at 4,192 / 8,192 (3.9 us each); r8 G: 112-token tiles (112 CTAs) took 4,96. r8 D: the all-warp epilogue cut the per-tile term 16-28 %, so the 2-tile tails shrank with it'
  - 'r9: pairs (H33) did not remove the tails (2 units per CTA at 32,768 remain); H38 dynamic claiming refuted. r10 arithmetic: balancing to <= 148 tiles of ~1.3 images each gives a critical path of ~1.3 x 10.7 us instead of 2 x 10.7 at 16,384 (-7 us), 5.2 instead of 6 images per CTA at 32,768, 1.3 instead of 2 at 8,768 / 16,768, 0.65 instead of 1 at 8,384: built as H41'
- id: H8
  statement: Applying the GRN scale per image to W2 (or scaling the hidden tile in smem while converting to fp16) removes the per-element affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row); r3 D folds fp16(s_img*W2) per image; r4-r9 D tc uses two accumulators (h and g = gw*gf*h) at the price of doubled GEMM2 B ingest and a g store in the GEMM1 epilogue (~1.2 us of the 2.3 us store term); r6 F''s one-launch K3 costs 7-20 us'
  - 'r7 D sc_kernel scales the fp16 B tile in smem (0.3-0.45 us per 64-k block); r7-r9 G uses per-image fp16 W2_b built in a 5.6-8.9 us K3. r8 D: g produced in smem inside GEMM2 gated every stage of the 4-stage 48 KB ring; r9 E2: pairs with pipelined per-stage signalling (one producer warp per stage) fixed the gating but the g pass stayed; the untested fallback (single accumulator, h*s in place) is what H41 and H42 build, with the stage-owner warp scaling >= 2 stages ahead'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager, so even the fp32 anchor scores > 0.5 on the S/M band.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload; the anchor scores 0.10-0.51'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images' GEMM1 work at B >= 8 and removes two barrier waits.
  status: refuted
  evidence:
  - 'r4 B (E2) microbench: one red.release flag polled by 295 CTAs is observed in 0.64 us median; flat atomic-counter barrier 0.86-1.07 us; the per-tile dataflow kernel (H27) was 1.4-2x slower on every S row'
  - 'r8 H (H38): the image-level version ran +58-98 % vs D on all five B >= 8 wide-C rows; summed item CTA-time / 148 exceeds D''s whole rows. Closed for the wide-C rows; lineage H retired r10'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches; what sets the S rows is the SM work and dependent rounds inside each phase. Cutting the per-phase chain to ~3-4 us gives S rows of 15-20 us rented (S ~0.78-0.82).
  status: supported
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7'
  - 'r2 B audit at 1,192: ~12 dependent steps of 1.5-2.5 us between P1 end and the end at 28.2 us'
  - 'r3/r4 B microbench: barrier release 1.6-1.9 us per 296-CTA flip barrier, flat atomic-counter barrier 0.86-1.07 us, cold rounds 1.2-3.5 us'
  - 'r7 D (E2) SUPPORTED: sc_kernel runs 1,192 at 19.4 us and 1,96 at 19.1 us rented in one launch (portal 23.3 / 23.8 in r9-merge). Stamped chain at 1,192: phase A 5.4 (weights 2.6-3.1 then conv 2.5-3), sync 1 ~1.2, GEMM1 single tile 0.6 + 1.0 + 3.7 post-accumulator, sync 2, GEMM2 1.4 + 1.0 + 1.6. r9: the flat barrier loses 1-2 us on sc rows (skewed arrivals). Audit r10: the S rows are worth +0.0062 per 10 % and remain a lever (I17 / I20 owner-converted or zero-conversion weights, ~2.5-3 us of the 5.4 us phase A); unchanged-code portal spread there is +-1.2 us'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz.
  status: supported
  evidence:
  - 'r1 probes: 2018-2048 FLOP/clk/SM = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16'
  - 'r3 A/D: real tiles 160 TF/s; cuBLASLt ~4x faster. r7 G: mma.sync costs ~100-130 us of the 668 us at 64,96 serialised in the epilogue warps. r8 G at 8,320: pipe-bound at 16 warps (9 clk per mma per scheduler); r9 G replaced it with tcgen05 at 8,320 / 4,96 (H40); the top two rows still run it (H43 replaces it)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, hidden slices in smem, statistics and GEMM2 partials reduced through DSMEM) runs the 7x7/14x14 workloads in <= 30 us.
  status: refuted
  evidence:
  - 'r1 C: DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM; v1 at 2,768 129 us'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map; r5 E: CUTLASS tfloat32_t A/B on fp32 memory gives max |err| 1.1-1.4e-5 vs an fp64 reference with RNE-rounded operands'
  - 'But the zero-conversion route does not pay on the C=768 rows (audit r10): only the GEMM1 tile owners would read W1 (24 CTAs at 2,768, 96 at 8,768), 393 KB fp32 per tile at 40-94 GB/s/SM = 4-10 us against 2.5 us for the grid-wide fp16 pass plus 197 KB fp16 per tile, and kind::tf32 doubles the instruction count against the ~105 clk floor; the library route lost on time too (H30). Parked for the C=768 rows; at the S rows (W1+W2 0.3-1.2 MB) the owner-converted or TF32 form (I17 / I20) remains unbuilt'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd'
  - 'r6 F: GELU_BIAS epilogue is the tanh form (rejected); swap-AB batched NCHW GEMM2 is 2-4 us faster than channels-last + K4 at HW=784 and 20-190 us slower at HW=49/196. r7-r9 G: the batched swap-AB GEMM2 runs ~6.6 TB/s at HW >= 4096 but 3.8 TB/s at 8,320 (11.1-11.3 us for 42 MB, 8-algo sweep found nothing better) and 6.6 us at 4,96: H42 replaces it with a custom tcgen05 GEMM2 on those two rows'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little, so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r8 D: the h+g stores (~15.7 MB per tile wave at 8,384) cost ~2.3 us of the GEMM1 epilogue at ~6-7 TB/s under the dirty flush; the bytes are cheap but they bind the epilogue, so halving them (no g) is worth ~1.2 us per tile wave (H41 drops g)'
- id: H17
  statement: The anchor's phase 1 at C=768 is block-level weight transposes plus the 4-tokens-per-lane conv; warp-level transposes and zero-padded smem planes cut it to ~45 us.
  status: supported
  evidence:
  - 'r0 probes: 42 vs 11.5 us transposes; 306 vs 59 us conv; smem planes 35.5 us at 32,768,7,7'
- id: H18
  statement: Staging the dwconv input through TMA boxes (3-D map for W%4==0 rows; 1-D cp.async.bulk plane copies for 7x7, 14x14 and 449x7 rows) in a multi-stage mbarrier ring lifts conv ingest to >= 40 GB/s per SM; at the small-image rows the conv then lands within 1.5x of its FMA bound, at the top rows it does not.
  status: supported
  evidence:
  - 'r2 A (E1): 15-34 GB/s/SM, 11x the old staging; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96 (3x the FMA bound at the largest). Box inner start must be a multiple of 16 B'
  - 'r3 D: bulk-plane copies for W=7/14 (conv phase 1.5-10 us on the six wide-C rows); r4 A: 449x7 with per-lane 1-D bulk copies; r7 D sc_kernel: plane copies of 3136 / 12544 / 12572 B work'
  - 'r9 D (audit r10 correction): at 8,768 the plane copies land at 1.5 us when W1 traffic is absent and the conv ends 0.9 us later (FMA bound 0.4 us): within ~2x of the bound and no longer a lever. r7/r8 G: the fused pass stages x through L1/L2 at the top rows (conv 4.6x its bound); H43 stages x row bands by 1-D bulk copies into a ring instead'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be overlapped across the grid syncs.
  status: refuted
  evidence:
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile in shared memory across the grid sync runs the seven C <= 192 rows in <= 15-20 us rented.
  status: refuted
  evidence:
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3 (kill line 25 us); r7 D''s sc_kernel reached 19.4 us at 1,192 with a materialised hidden'
- id: H21
  statement: The portal/rented ratio is a per-path, per-row and per-launch-structure quantity, not a constant 1.28. Multi-launch pipelines pay more on the portal than single-launch cooperative kernels on S/M rows; each launch removed on an S/M row is worth 1-1.5 us more on the portal than the bench shows.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented: 1.27-1.28; fp32 anchor 1.30-1.32; #65878/#65879: B paths 1.24-1.26, D paths ~1.32'
  - 'operator r6: moving 1,96 and 1,192 from B to A looked 5-8 % faster rented and was 6-8 % slower on the portal; r6-merge: D at 1.20-1.32, A at 1.40-1.55 on S/M rows'
  - 'r7-r9 merges: projected 0.7208 / 0.7333 / 0.7351 vs measured 0.7207 / 0.7313 / 0.7359; D sc 1.20-1.27, D tc 1.20-1.32, G 1.26-1.27; M/L rows within +-2 %, S rows +-1.2 us between submissions with unchanged code'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol on the 77M-element rows; a hi/lo fp16 split of one operand restores fp32-class accuracy at no cost where MMA work is <= 2 us, so per-row error logging decides where to spend it.
  status: supported
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r5 A, 10 seeded rounds, 16 rows: 0 beyond-own-atol everywhere; max up to 2.427e-3 (1,96, 90 %). r6-r9: D tc 59-65 %, D sc 79-89 % (tightest 4,192 at 89 %), G 61-82 %, H 58-64 %, F 59-76 %, E TF32 1.67-1.94e-3. No hi/lo split needed on any shipped path; H41 (no-g tc path) and H42 (scaled B at 8,320, atol 2.4e-3) move to the twice-rounded class and must be logged; the hi/lo B split or scaling the A tile from fp32 W2 are the fallbacks'
- id: H23
  statement: tcgen05 kind::f16 (TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the GEMM-heavy rows and is the lever for the L band once the S/M rows are near the fused floor.
  status: supported
  evidence:
  - 'r6: D''s tcgen05 engine took five rows (+0.016); r7: six more (+0.030); r8: the all-warp epilogue took 11 rows (+0.011); r9: G''s tcgen05 g_tc took 8,320 (-17.5 % rented) and 4,96. The top two rows are the last on mma.sync (H43)'
- id: H24
  statement: The 2-4 us cost of a 12-32 KB per-CTA load round right after a grid sync is an issue-rate artefact of 4-byte copies; one cp.async.bulk per operand brings the round to <= 1.2 us.
  status: refuted
  evidence:
  - 'r3 B microbench: just-written 32 KB round cp4+div 2.4 us / cp16 1.36 / ldg128 0.95 / one bulk 0.83 us median; cold DRAM source 2.7-3.5 us whatever the method; run_tests bulk rows 1,192 31.4 vs ring 29.5'
- id: H25
  statement: A weight-stationary cooperative kernel for the six wide-C small-image rows with an mma.sync engine runs 2,768 <= 35 us, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 rented.
  status: refuted
  evidence:
  - 'r3 D v3 run_tests: 2,768 34.1 (met); 8,768 63.9; 16,768 93.6; 32,768 169.6; 8,384 53.5; 16,384 93.7. Binding term: mma.sync 0.64 us per 32-k chunk plus a sync-heavy ring; engine swap in H28'
- id: H26
  statement: Replacing cuBLASLt GEMM1 + K2 and K3 + cuBLASLt GEMM2 + K4 in lineage A with two custom mma.sync FP16 GEMMs with fused epilogues cuts the pipeline to 3 launches and the S/M rows by 10-14 us rented.
  status: refuted
  evidence:
  - 'r3 A (E1): 1,192 36.3 us (target <= 26), 2,768 60.1 (<= 48), 8,384 97.5 (<= 60); at 8,320 the custom GEMMs run at 160 TF/s vs cuBLASLt ~4x faster. The single-launch tcgen05 form (H34) reached 19.4 us'
- id: H27
  statement: In the cooperative kernel, replacing the grid syncs with per-token-tile and per-image counters removes the two barrier releases and the conv tail wait, cutting the S rows by 4-8 us each.
  status: refuted
  evidence:
  - 'r4 B (E2): the dataflow kernel is slower on every S row (1,192 42.8 vs 29.8; 8,192 137.7 vs 66.5); the image-level version (H38) also lost'
- id: H28
  statement: 'A tcgen05 kind::f16 engine in the swap-AB orientation inside D''s cooperative skeleton makes the six wide-C rows MMA- and byte-bound: thread-local LN fold and GRN sumsq per (image, k), hidden stored [k][token] fp16, GRN scale via two accumulators, output tile [c][tokens] = NCHW with the residual in the epilogue.'
  status: supported
  evidence:
  - 'r4 D (E3) first build 16/16: 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us; per 64-k stage 524-683 clk vs 224-448 ideal'
  - 'r6 D (E2): the 2x term was a 128-B local stack frame in the epilogue; fixed: 27.6 / 34.6 / 46.5 / 66.6 / 38.0 / 58.5. r8 (H36): 26.5 / 33.3 / 44.0 / 64.2 / 34.2 / 51.7. r9 (H39 / H33): 24.0 / 32.4 / 43.5 / 62.1 / 33.4 / 50.9. Remaining: W1 stream in phase A, whole-image tile tails (H41), GEMM2 stage time, GEMM1 h+g store bytes'
- id: H29
  statement: Rewriting lineage A's K2 as a stream-rate kernel cuts 447 -> ~200 us at 64,96 and a tcgen05 GEMM1 with the same epilogue would remove K2 entirely.
  status: supported
  evidence:
  - 'r5 A (E2): ring K2 444 -> 319 us at 64,96, 117.8 -> 79.5 at 32,128; the remaining ~100 us is FMA/INT issue of the GELU. r7 G: K2 removed by the fused pass; the GELU is MUFU-bound inside the fused warps (133 us floor at 64,96) and the FMA pipe is shared with the conv (H43)'
- id: H30
  statement: 'A CUTLASS 4.4 sm100 builder GEMM1 (batched per image) with the LN fold, erf GELU, hidden store and per-batch column reduction in its EVT removes K2, K3 and K4 from the wide-C rows.'
  status: refuted
  evidence:
  - 'r5 E (E1): G1 with EVT 17.0 us at 8,768 (kill 14) and 20.0 at 8,384 (kill 12); per-image batching re-reads W1 and TF32 doubles the bytes; CUPTI sequence-matching failures 14/16 twice. Lineage E retired r8'
- id: H31
  statement: 'On B200 a CTA fed from L2 through a TMA or 1-D bulk ring lands at most ~68-92 B/clk/SM in 1SM mode, the cap is SM-side, so designs must halve per-SM operand bytes (cta_group::2 pairs, or operands produced in smem by the CTA itself) rather than multicast; in 1SM mode a kind::f16 128xNx16 instruction also has a ~105 clk throughput floor for N <= 128.'
  status: supported
  evidence:
  - 'r4 D gate: per 64-k stage 683 clk at N=224 (ideal 448), 587 at N=112 (224), 524 at N=128 (256); stage depth 2-3 vs 8 unchanged'
  - 'r6 D probe (148 CTAs, warm L2, 4-stage SW128 ring): 1SM 479 / 532 / 546 clk at N = 128 / 224 / 256; B multicast over a 2-CTA cluster no gain; cta_group::2 pair 345 / 455 / 518 (1.35x / 1.01x / 1.01x of ideal)'
  - 'r9 G: 1SM 128xNx16 with smem-resident operands 106 / 107 / 128 clk per instruction at N = 64 / 128 / 256 (ideal 64 / 128 / 256... measured ideal at 256), independent of accumulator dependence; 10-stage bulk ring cold: 94 GB/s/SM from L2, 40 GB/s/SM distinct-per-CTA from DRAM. Audit r10: the A-slice-fetch mechanism is unsupported (pair probe 86 clk per 128x128x16 per SM); the number stands'
  - 'r9 D E2 v2 stamps: pair GEMM2 stages 658-875 ns vs a ~270 ns ideal with the producers gating (fence + cluster arrive in series per stage); v3 (one producer warp per stage) recovered it at 32,768 (-3 %). Pairs pay only where a CTA runs many stages'
- id: H32
  statement: 'A library-lean four-launch pipeline (cluster-of-8 LN exchange, cuBLASLt GEMM1 with GELU epilogue, one-launch GRN, cuBLASLt swap-AB GEMM2) removes K2, K4 and the K3 chain from every HW <= 784 row.'
  status: refuted
  evidence:
  - 'r6 F (E1): refutation clause fired on 5 of 6 wide-C rows; D r6/r7 then beat F everywhere. Lineage F retired r8'
- id: H33
  statement: 'In D''s swap-AB tcgen05 engine, GEMM2 as cta_group::2 pairs (M = 256 output channels across the pair, each CTA loading half of the B token tile) halves the per-SM landing and allows a deeper ring in which g is produced in smem by look-ahead warps, so g leaves the GEMM1 epilogue; rows run 16,384 <= 46, 8,384 <= 30, 32,768 <= 57, 16,768 <= 39, 8,768 <= 29, 2,768 <= 24 us rented.'
  status: refuted
  evidence:
  - 'r6 D probe: the pair reaches 1.35x / 1.01x / 1.01x of the MMA ideal at N = 128 / 224 / 256; the pair recipe worked first try'
  - 'r9 D (E2): ptxas forbids mixing cta_group::1 and ::2 in one kernel, so GEMM1 paired too (coupled accumulator wait 2.5-4.2 us per tile). v1/v2 slower than r8 on all four C=768 rows (gw*gf load or producer fence+arrive in series per stage; stages 2.5-3.2x ideal); v3 (one producer warp owns a stage) 32,768 61.1 vs 64.1, 16,768 42.7 vs 43.8, 2/8,768 slower. Shipped v4 pairs only C=768 at B >= 16: 32,768 62.1 vs 64.1 (-3.1 %). Success line missed on every row; the smem-g pass was never re-enabled. The real content (pipelined per-stage signalling, pairs at many stages per CTA) is kept; the g production by look-ahead warps is replaced by the no-g single-accumulator form in H41'
- id: H34
  statement: 'D''s single-launch tcgen05 skeleton extended to the C <= 192 rows and 1,160 runs 1,192 <= 24 us rented, 4,192 <= 30, 8,192 <= 42, 2,96 <= 32, 1,160 <= 34.'
  status: supported
  evidence:
  - 'r7 D (E2) sc_kernel 16/16: 1,192 19.4; 1,96 19.1; 1,160 26.9; 2,96 27.0; 4,192 30.5; 8,192 44.8; r8: A&S GELU and 4,96 added (1,96 18.4, 2,96 25.5, 4,192 29.4, 8,192 43.4, 1,160 26.0, 1,192 18.6, 4,96 38.8); r9: 18.2 / 25.0 / 29.3 / 41.0 / 25.3 / 18.2 / 37.6. 8,320 is not instantiable (W % 7, 4C <= 768)'
- id: H35
  statement: 'A fused persistent pass-1 kernel for the HW >= 1024 rows replaces K1 + G1 + K2 with one kernel bounded by the SIMT work of conv + GELU (~200-270 us at 64,96 rented): pass 1 <= 300 / 75 / 25 us; with a GRN-scale launch and a cuBLASLt batched swap-AB GEMM2 the rows run <= 550 / 140 / 55 us rented.'
  status: open
  evidence:
  - 'r7 G (E1), mma.sync engine: pass 1 668 us at 64,96 (2.5x the tile bound), 171 at 32,128; rows 857.5 / 229.8 rented (-16.5 % / -10.6 % vs A). Ablations: conv side alone 472 us, GEMM side 522 (mma.sync 100-130 serialised, 2-MUFU GELU floor 133); phases add because 1 CTA of 16 warps fits per SM'
  - 'r9 G (H40): tcgen05 at 8,320 / 4,96 gave -17.5 % / -10 %; the top two rows still run the r7 mma.sync pass. H43 (lineage I) rebuilds pass 1 at the pipe bounds; the direction stands, the targets are open'
- id: H36
  statement: 'In D''s tcgen05 kernels the post-accumulator GEMM1 epilogue is latency- and issue-bound on 8 epilogue warps; running it on all warps cuts the per-tile epilogue by >= 40 %.'
  status: supported
  evidence:
  - 'r8 D (E1): all-warp epilogue at 512 threads: per-tile post-accumulator -16..-28 %; bar missed but 11 rows won (+0.011). Ablation at 8,384: post 7.6 us = pass 1 4.3 + g write 3.0; GELU ~1.4, h+g stores ~2.3; the epilogue is bound by the h+g store bytes, so removing g (H41) is the next lever'
- id: H37
  statement: 'G''s fused pass-1 pipeline instantiated for 8,320 and 4,96 with mma.sync runs 8,320 <= 50 us and 4,96 <= 34 us rented.'
  status: refuted
  evidence:
  - 'r8 G (E2): 4,96 37.0 vs 41.4 (-10.7 %); 8,320 67.2 vs 73.1 (-8.0 %); mma.sync pipe-bound at 16 warps (16.2 us) with the epilogue serialised behind it; replaced by tcgen05 in H40'
- id: H38
  statement: 'One persistent tcgen05 kernel that claims items from a static queue and gates them on per-image counters runs the B >= 8 wide-C rows without grid barriers.'
  status: refuted
  evidence:
  - 'r8 H (E3): +58-98 % vs D on all five rows; summed item CTA-time / 148 = 47.0 / 111.5 / 71.9 us at 8,768 / 32,768 / 16,384 above D''s whole rows. Lineage H retired r10'
- id: H39
  statement: 'Phase A of D''s kernels is a serial chain, not a bandwidth term: issuing every conv plane copy at kernel entry, converting W1 while they land, and deferring the W2 conversion and the out prefill into the GEMM1 phase cuts phase A to <= 5 us at C=768 and <= 3.5 us at 1,192: -2..-5 us on every D row (tc rows 8,768 <= 30.5, 2,768 <= 23.5, 16,768 <= 41, 32,768 <= 61, 8,384 <= 31.5, 16,384 <= 49 rented). Stretch: flat self-resetting counter barriers for another -1..-1.5 us per row.'
  status: refuted
  evidence:
  - 'r9 D (E1): refutation clause fired on 4 of 6 rows after the reorder. Stamps at 8,768: the plane copies were already issued at entry in one transaction; issuing the 12 LDG.128 per lane of W1 takes 2.2 us (~4 TB/s over the grid), the planes land at 2.7 us instead of 1.5 because they queue behind the W1 stream, conv ends 4.0, W1 conversion 5.6, phase A end 5.7 / 7.1; without W1: 3.2 / 4.1 and sync 1 at 5.3-5.9. Phase A is bound by the W1 fp32 DRAM stream (2.5 us of phase A, 3.2 us at sync 1), not by latency'
  - 'The deferred W2 pass still wins 1-3 us per row (it leaves the critical path; it finishes ~5 us after sync 1 and costs 0.5-0.9 us of GEMM1 end at 8,768, 2.2 us at 32,768); an L2 prefetch of W2 in the sync-1 window only moves the cost. Flat barrier: -0.1..-1.3 us on tc rows, +1..+2.1 us on sc rows (shipped on tc_kernel only). Final: 2,768 24.0 (merge path r6 tc 28.7), 8,768 32.4, 16,768 43.5, 16,384 50.9, 8,384 33.4, 32,768 65.1; targets met only at 2,768 within 0.5 us. Shipped build''s STACK and the single-seed precision check were not run (E1 r10 STEP 0)'
- id: H40
  statement: 'Replacing mma.sync by tcgen05 kind::f16 in G''s single-tile kernel removes the 16.2 us pipe-bound mma.sync term at 8,320 and the 3.3 us at 4,96, so pass 1 runs <= 30 us at 8,320 and <= 19 us at 4,96 and the rows <= 55 / <= 33 us rented.'
  status: supported
  evidence:
  - 'r9 G (E3) run_tests 16/16, same container vs r8-merge: 8,320 59.0 vs 71.5 (-17.5 %); 4,96 37.2 vs 41.4 (-10.0 %); taken by r9-merge (portal 74.5 / 46.7, S 0.763 / 0.744). Targets missed (59.0 > 55; 37.2 > 33; pass-1 ends 34.3 / 19.2 us); refutation clause did not fire. Engine gate passed first try; conv-overlapped MMA in channel rounds (MMA end 34.8 -> 23.6 us) and per-warp smem-transposed h stores (row -4.2 us) are the shipped levers; at 4,96 the engine alone gave nothing (not MMA-bound under mma.sync either). Binding terms now: the epilogue (~16 us at 8,320, ~2x its issue bound), the per-MMA floor at N=64, and K3 + cuBLASLt G2 (20.2 us at 8,320, 12.2 at 4,96: H42)'
- id: H41
  statement: 'In D''s tc_kernel, balanced image-straddling token-range tiles (flat [C][B*HW] d and [4C][B*HW] h layouts, range starts at multiples of 8 tokens, hidden tiles x ranges <= 148, N = range padded to 16 with two MMAs when N > 256) with the per-(image,k) GRN sumsq split at image boundaries in the epilogue and red.add-ed into a [B][4C] gsum zeroed in phase A, no g (h stores only), and a single-accumulator GEMM2 whose landed h tile is scaled in place by s[img(token)][k] by the stage-owner producer warp running >= 2 stages ahead (r9 E2 v3 signalling), remove the 2-tile tails and halve the GEMM1 store bytes and GEMM2 B landing: 16,384 <= 44, 32,768 <= 56, 8,384 <= 30.5, 16,768 <= 39.5, 8,768 <= 30 us rented (r9: 50.9 / 62.1 / 33.4 / 43.5 / 32.4), about +0.005-0.007 of mean. The hidden is rounded twice (sc_kernel class) and must be logged per row.'
  status: open
  evidence:
  - 'arithmetic (r6/r8 stamps): per-tile GEMM1 = ~2.6 mainloop + 8.1 epilogue at 16,384 with 44 of 148 CTAs running 2 tiles; balanced ranges of 264 tokens give every CTA ~1.35 images of work: critical path ~14 vs 21.4 us. 32,768: 264 whole-image tiles (116 CTAs with 2) vs 144 tiles of 5.4 images. 8,768 / 16,768: 96 / 192 tiles of 2 images vs 144 of ~1.3. 8,384: 96 tiles idle 52 SMs. H16: the g store is ~1.2 us per tile wave; r7 sc_kernel: the B-tile scale costs 0.3-0.45 us per 64-k block when it sits between landing and MMA, which the stage-owner warp >= 2 stages ahead avoids. Built as E1 (r10)'
- id: H42
  statement: 'A custom tcgen05 kind::f16 GEMM2 launch g_g2 for G''s HW >= 1024 rows (swap-AB: A = fp16 W2 128-channel tiles written by K0, B = the pass-1 h [k][token] fp16 tile in 64-token MN-major chunks scaled in place by s_b[k] by the stage-owner producer warp, s_b computed in the prologue from the red.add gsum array, b2'' = W2.gb + b2 in fp32 from K0, epilogue out = x + acc + b2'' staged through smem for coalesced NCHW rows, N = 256 at 8,320 (96 units) and 224 or 112 at 4,96) replaces K3 and cuBLASLt G2 (4 launches -> 3): g_g2 <= 9 us cold at 8,320 (K3 8.9 + G2 11.3 now) and <= 4.5 us at 4,96 (5.6 + 6.6), rows <= 49 / <= 31 us rented (59.0 / 37.2), portal ~60 / ~38 (S 0.763 -> ~0.81, 0.744 -> ~0.79), about +0.005-0.006 of mean. It also reports clk per 128x256x16 instruction in 1SM mode (audit r10 item 2).'
  status: open
  evidence:
  - 'arithmetic: 8,320 moves 42 MB (h 21 + x 10.5 + out 10.5) = ~6.5 us at 6.5 TB/s; per-unit ingest at N=256 is A 327 KB + B 655 KB = ~11k clk = 5.7 us at 1.9 GHz on 96 of 148 SMs; MMA at N=256 runs at the ideal (r9 G: 128 clk per instruction). r8 G: cuBLASLt G2 at 8,320 runs 3.8 TB/s (11.1-11.3 us) and K3 is a 7-9 us floor. Precision: h rounded twice (sc class, 79-89 %) against 8,320''s atol 2.4e-3 (tightest row): logged over 10 rounds; fallbacks are the hi/lo B split (2 MMAs at the N=256 ideal, ~+2 us) or scaling the A tile per unit from fp32 W2 (keeps h once-rounded). Built as E2 (r10)'
- id: H43
  statement: 'A pass-1 kernel for 64,96 / 32,128 with fixed warp roles (one TMA warp streaming x channel-group row bands by 1-D bulk copies into a 2-stage ring, 8 conv warps computing register-blocked 2x8 output patches with the 49 taps in registers and writing fp16 d straight into the MN-major SW128 B tile with LN partials, one MMA warp running kind::f16 against W1'' streamed per tile from the K0 fp16 scratch, 6 epilogue warps doing the LN fold, A&S erfc GELU, per-(image,k) sumsq and staged coalesced fp16 h stores) runs within 1.5x of the SM pipe bound: conv-only <= 180 us and pass 1 <= 350 us rented at 64,96 (G r7: 472 / 668), so with the notebook''s GRN-scale + cuBLASLt batched swap-AB GEMM2 tail the rows run 64,96 <= 560 us rented (857) and 32,128 <= 150 (230): S 0.80 -> ~0.85 and 0.77 -> ~0.83, about +0.006 of mean (+0.01 at the bounds). The FMA pipe (conv 102 + GELU FMA ~80 + stats/stores ~20 = ~200 us at 1.95 GHz) is the binding resource ahead of MUFU (133) and DRAM (~140), so the balanced erf (I30) must not move MUFU work onto FMA.'
  status: open
  evidence:
  - 'r7 G ablations at 64,96: conv side alone 472 us (L1/L2 load path, serial 3-row prefetch chain, 4x halo redundancy at R=2), GEMM side 522 (mma.sync 100-130 serialised in the epilogue warps, 2-MUFU GELU floor 133), phases add at 1 CTA of 16 warps per SM; an L2 prefetch of the halo changed nothing. r9 G: tcgen05 engine validated with d written by plain st.shared into the MN-major SW128 layout (writer address functions in the notebook); 1SM MMA ideal at N >= ~208 (224-token tiles at 64,96). Smem at R=2, 12-channel x stages: x ring 2 x 43 KB + d tile 2 x 43 KB + A ring 2 x 16 KB + h staging ~16 KB = ~204 KB. Built as E3 (r10, lineage I)'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "D",
    "hypothesis": "H41",
    "title": "Balanced image-straddling token-range tiles, no g, scale-in-smem GEMM2 for the six tc rows",
    "operation": "structural_mutation",
    "parents": ["r9-d-e1-asynca"],
    "band": "all",
    "instructions": "Budget: 18 B200 minutes, 90 min wall. Parent = r9-d-e1-asynca (the default tc path in r9-merge); r9-d-e2-pair-smemg is in your lineage too (reuse its 'one producer warp owns a whole stage, stages round-robin over warps' signalling). Keep every sc row and the sc_kernel untouched; keep the old tc path in the file so that any tc row where the new path loses stays on the r9 code (per-row dispatch by shape).\n\nSTEP 0 (<= 2 B200 min, answers the premise audit): (a) read tc_kernel's flat self-resetting barrier: the arriving side must be __syncthreads then one thread red.release.gpu (or __threadfence + atomicAdd); the waiting side must spin with ld.acquire.gpu, then execute fence.proxy.async (so the CTA's later TMA reads of d/h/scratch written by other CTAs' generic stores are ordered), then __syncthreads. Add whatever is missing. (b) cuobjdump -res-usage on the shipped parent build: STACK must be 0 for tc_kernel and sc_kernel; record registers. (c) 1500-call stress loop at 8,768 and 16,384 against the fp32 torch reference (max |err| must stay <= the r8 log + 1e-4 on every call). If (a)-(c) show anything, revert tc_kernel to cg grid.sync and report; otherwise record 'barrier verified'.\n\nSTEP 1: flat token layouts. d scratch becomes [C][B*HW] fp16 and h becomes [4C][B*HW] fp16 with no per-image padding (the conv unit's stores and the TMA maps change; token = b*HW + hw). TMA box starts along the token axis must be multiples of 8 tokens (16 B).\n\nSTEP 2: tile tables per tc row (host, shape-only): GEMM1 tiles = hidden tiles x token ranges, ranges = ceil(ceil(B*HW / R) / 8) * 8 tokens with (hidden tiles x R) <= 148: 16,384 12 x 12 (264 tokens, N = 272 as two MMAs 128 + 144 into adjacent TMEM columns); 32,768 24 x 6 (264); 16,768 24 x 6 (136, N = 144); 8,768 24 x 6 (72, N = 80); 8,384 12 x 12 (136, N = 144); 2,768 only if time remains (24 x 5 ranges of 24 tokens, N = 32), otherwise it stays on the parent path (24.0 us). Epilogue per tile: LN fold + A&S GELU as now; the sumsq per (image, k) is accumulated in registers per image segment (image = token / HW, at most 2 segments at 14x14 and 6-7 at 7x7 per 264-token range; the 4-way image select exists) and red.add-ed into gsum[B][4C] (fp32, zeroed in phase A before sync 1); h stores only (no g, no gf in the epilogue). GEMM2 units = out-channel tiles x token ranges x KS with the same balancing (<= 148 units per pass, or 2 per CTA where unavoidable).\n\nSTEP 3: GEMM2 with one accumulator (N = range, two MMAs when > 256). After sync 2 each CTA computes, for the <= 7 images its units touch, gf[k] = sqrt(gsum[b][k]) and mean_k gf (one block reduction per image over 4C values), then s[b][k] = 1 + gw[k] * gf[k] / (mean + eps) for its K slices into smem (<= 7 x 768 fp32). Ring: A 16 KB + B (64 k x N x 2 B) per 64-k stage, as many stages as fit (>= 4). The stage-owner producer warp (round-robin over 4 producer warps, >= 2 stages ahead of the MMA warp) waits for the stage's TMA landing, multiplies the B tile in place by s[img(token)][k] (each k row is 128 contiguous bytes per 64-token chunk in the SW128 layout; 16-B vectors of 8 tokens can straddle an image boundary at HW = 49 or 196, so select per element), then fence.proxy.async.shared::cta and arrives on the ready barrier. Epilogue: out[b][c][hw] via the existing per-token addressing (red.add for KS > 1 into the x + b2' prefill).\n\nSTEP 4: run_tests 16/16 (portal-order re-runs); interleave x3 in one container vs r9-merge on the six tc rows; globaltimer stamps of GEMM1 end (median / max) at 16,384 and GEMM2 per-stage time at 8,768; cuobjdump STACK 0; PRECISION LOG over 10 seeded rounds (fp64 reference, harness generators) for every tc row the new path ships (the hidden is now rounded twice). If any row has a beyond-atol element, switch that row's B operand to the hi/lo fp16 split and re-log. Ship the new path only on rows where it wins; report per-row rented times and the stamps.\n\nOptional if >= 3 B200 min remain: createpolicy evict_first on the W1/W2 fp32 loads in phase A, interleaved x3 at 8,768 / 32,768 (ship if >= 0.7 us).",
    "success": "In-container vs r9-merge: 16,384 <= 44 us and at least one of 32,768 <= 56, 8,384 <= 30.5, 16,768 <= 39.5, 8,768 <= 30; GEMM1 end (max over CTAs) at 16,384 >= 4 us earlier than the parent's; run_tests 16/16; precision log 0 beyond own atol on every shipped row.",
    "refuted_if": "At 16,384 the GEMM1 end (max over CTAs) is less than 2 us earlier than the parent's despite one tile per CTA, or the in-place B scale makes GEMM2 stages > 1.3x the parent's stage time at 8,768 with >= 2 stages of look-ahead, or the new path wins no tc row by >= 1.5 us in-container.",
    "model": "opus"
  },
  {
    "id": "E2",
    "lineage": "G",
    "hypothesis": "H42",
    "title": "Custom tcgen05 GEMM2 launch replaces K3 + cuBLASLt G2 at 8,320 and 4,96 (4 launches -> 3)",
    "operation": "structural_mutation",
    "parents": ["r9-g-e3-tc05-p1s"],
    "band": "L",
    "instructions": "Budget: 18 B200 minutes, 90 min wall. Keep g_tc (pass 1), K0 and the 64,96 / 32,128 path (g_pass1 + K3 + cuBLASLt G2) unchanged; the new kernel is dispatched for 8,320 and 4,96 only.\n\nSTEP 0 (host/K0, no probe): K0 additionally writes fp16 W2 as [C][4C] row-major (K-major for the swap-AB A operand; TMA SW128 boxes of 128 rows x 64 k) and b2'[c] = sum_k W2[c][k] * gb[k] + b2[c] in fp32 (320 dot products of length 1280; one warp per c). Keep K0 at its current 3.5-4.5 us (check with harness_time).\n\nSTEP 1: g_g2 kernel, 1SM tcgen05 kind::f16, swap-AB, using your validated recipe (idesc, K-major SW128 A box SBO 1024 +32 B per k16, MN-major B 64-token chunks LBO = 64 k x 128 B, SBO 1024, +2048 per k16, tcgen05.ld 32x32b). Units = (c-tile, image, token chunk): 8,320 c-tiles 128 / 128 / 64 (the last via M=64 or a masked 128-row tile with TMA zero fill), 8 images x 4 chunks of 256 tokens = 96 units, N = 256 (one MMA per k16, 256 TMEM columns); 4,96 one c-tile of 96 rows (masked), chunks of 224 tokens (56 units) or 112 (112 units): time both. B = the pass-1 h_b array [K2 = 4C + 8][HW] per image (ignore the 8 augmentation rows: K = 4C exactly). Prologue per CTA: load gsum[b][0..4C) (the red.add sums pass 1 already writes), gf = sqrt, block-reduce mean_k, s[k] = 1 + gw[k] * gf[k] / (mean + eps) into smem (4C fp32), plus b2' for the c-tile. Ring: 4-6 stages of A 16 KB + B 32 KB (N = 256). Four producer warps, each owning a whole stage round-robin (>= 2 stages ahead of the MMA warp): wait for the TMA landing, scale the B tile in place by s[k] (each k row is 128 contiguous bytes per 64-token chunk; cvt to f32, multiply, cvt back, 16-B vectors), fence.proxy.async.shared::cta, arrive. One MMA warp with elect.sync, commit to the accumulator barrier. Stamp (globaltimer + clock64, from the MMA warp) the clk per 128x256x16 instruction at 8,320 and report it (audit item: does 1SM reach the ~256 clk ideal at N = 256 with TMA-fed operands?). Epilogue on all non-producer warps: tcgen05.ld, out[b][c][hw] = x[b][c][hw] + acc + b2'[c]; stage each 128-row x 256-token tile through the (now free) ring smem so that global reads of x and stores of out are 128-B-coalesced along hw (NCHW rows are contiguous for HW = 1024 / 3136).\n\nSTEP 2: validate g_g2 standalone against torch (fp32 reference of x + W2 (h * s) + b2') at both rows; then harness_time of the 3-launch pipeline vs the 4-launch parent interleaved x3 in one container at 8,320 and 4,96, plus g_g2 standalone cold (after the 2xL2 flush).\n\nSTEP 3: run_tests 16/16 (portal-order re-runs; the other 12 rows run the r0 anchor as a merge-only candidate), cuobjdump STACK 0 and <= 128 regs, PRECISION LOG over 10 seeded rounds (fp64 reference, harness generators) at 8,320 (atol 2.4e-3, the tightest row) and 4,96: the hidden is now rounded twice (fp16 h then fp16(h * s)). If any element is beyond atol + 1e-5|ref|, switch to the hi/lo B split (B_hi = fp16(h*s), B_lo = fp16(h*s - B_hi), two MMAs per k16) or scale the A tile per unit from fp32 W2 instead, re-log, and report the cost.\n\nStretch if >= 5 B200 min remain: dispatch g_g2 at 32,128 too (32 images x 16 chunks, persistent over 512 units) and compare with K3 + cuBLASLt (7.5 + 48.4 us); ship there only if faster in-container.",
    "success": "g_g2 standalone cold <= 9 us at 8,320 and <= 4.5 us at 4,96; rows in-container vs the parent 8,320 <= 49 us and 4,96 <= 31 us rented; run_tests 16/16; precision log 0 beyond own atol at both rows.",
    "refuted_if": "g_g2 standalone cold > 14 us at 8,320 after the ring and epilogue are tuned (no better than K3 + cuBLASLt minus a launch), or the in-place B scale holds stages > 1.5x the N=256 MMA ideal with 2 stages of look-ahead, or the twice-rounded hidden fails the 10-round log at 8,320 and both fallbacks cost more than the K3 + G2 pair they replace.",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "I",
    "hypothesis": "H43",
    "title": "New design: top-row pass 1 at the pipe bounds (bulk-staged register-blocked conv on FMA, tcgen05 GEMM1, GELU epilogue warps) for 64,96 and 32,128",
    "operation": "new_design",
    "parents": [],
    "band": "L",
    "instructions": "Budget: 18 B200 minutes, 90 min wall. You start a new lineage (I) and may read only the r0 baseline, the lab notebook and the lineage conclusions; the notebook holds everything you need: the tcgen05 recipe (H r8: idesc = (1<<4)|(1<<16)|(N>>3)<<17|(128>>4)<<24, K-major SW128 A box SBO 1024 +32 B per k16, MN-major B [channel][token] 64-token chunks LBO = C*128, SBO 1024, +2048 per k16, tcgen05.ld 32x32b.x32), G r9's writer address functions for d written by plain st.shared into the SW128 B layout (A image byte = r*128 + (((c>>3) ^ (r&7))<<4) + (c&7)*2; B byte = (t>>6)*LBO + c*128 + ((((t&63)>>3) ^ (c&7))<<4) + (t&7)*2), the GRN tail (red.add per tile into a [B][4C] gsum zeroed by K0; a GRN-scale launch building per-image fp16 W2_b = fp16(W2 * s_b) and b2' hi/lo columns, grid (B, C/16), one row per warp, __launch_bounds__(512,2); cuBLASLt batched swap-AB GEMM2: A = h_b [K2 = 4C + 8][HW] fp16 per image with two constant 1,1 rows, B = W2_b [C][K2], C = x_b, D = out, beta 1, writes NCHW; ~6.6 TB/s at these rows), and the precision rules (d16, fp16(lnw*W1), h16 once, A&S erfc GELU; log 10 seeded rounds vs fp64). Target rows: 64,96,112,112 and 32,128,64,64 only; every other row runs the r0 anchor (merge-only candidate).\n\nGATE (STEP 1, <= 5 B200 min): conv-only kernel at 64,96. Persistent CTAs of 16 warps, 1 CTA/SM, tile = R=2 image rows x W tokens x all C channels (224 tokens at 64,96; use R=3, 192 tokens at 32,128); each CTA takes a contiguous run of strips so halo rows hit L2. One TMA warp streams x in channel groups: for each group of 12 (64,96) or 16 (32,128) channels, (R+6) rows x W x 4 B per channel by 1-D cp.async.bulk (3.6 KB per channel at W=112) into a 2-stage ring (~43 KB per stage) with mbarrier expect_tx; zero rows at the image top/bottom are handled by predication, left/right zero padding by the tap loop. Eight conv warps compute register-blocked 2x8 output patches per lane (49 taps per channel in registers via __ldg from a K0-padded [C][7][8] buffer, inputs from LDS, ~7 FMA per LDS) and write fp16 d straight into the MN-major SW128 B tile (double-buffered, 43 KB each) plus per-token sum / sumsq partials in smem (fp32 of the fp16-rounded d). Measure with globaltimer and harness_time after the 2xL2 flush: conv-only pass over the whole 64,96 tensor (writing d tiles but no MMA / epilogue). Bound: 102 us at 1.95 GHz. Report rented us and the clock.\n\nSTEP 2 (only if STEP 1 <= 250 us; otherwise spend the remaining time on diagnosing the conv and report): add the GEMM: K0 writes W1' = fp16(lnw * W1) [4C][C] and u/v fold vectors; one MMA warp issues kind::f16 128 x N x 16 against A = W1' boxes 128 hidden x 64 k (K=96: 64 + 32) streamed per tile by TMA into a 2 x 16 KB ring (per-SM ingest ~74 KB per tile, negligible), 3 hidden blocks (4 at C=128) into two 224-column TMEM slots ping-ponged (slot reuse as G did). Six epilogue warps per finished block: tcgen05.ld, LN fold z = rs*(acc - mu*u) + v, A&S erfc GELU (do NOT add an FMA-polynomial erf: the FMA pipe is the binding one here), per-(image,k) sumsq partials red.add-ed into gsum, fp16 h stores staged through a per-warp 2 KB smem transpose so each STG.128 writes 64 B rows (h_b layout [K2][HW] per image with the two constant rows). Validate pass 1 against a torch reference of the hidden (fp16 tolerance) on a small shape before timing.\n\nSTEP 3: the tail (GRN-scale launch + cuBLASLt batched swap-AB GEMM2 exactly as in the notebook recipe), run_tests 16/16 (portal-order re-runs), cuobjdump STACK 0 at <= 128 regs x 512 threads, in-container interleaved x3 vs r9-merge at 64,96 and 32,128, PRECISION LOG over 10 seeded rounds at both rows. Report per-phase stamps (conv-only, pass 1, K3, G2) whatever the outcome: the top-row story needs the numbers even if nothing ships.",
    "success": "Conv-only <= 180 us and pass 1 <= 350 us rented at 64,96; rows in-container vs r9-merge 64,96 <= 560 us and 32,128 <= 150 us; run_tests 16/16; precision log 0 beyond own atol on both rows.",
    "refuted_if": "The conv-only build with bulk-staged row bands and register-blocked patches is > 300 us rented at 64,96 (3x the FMA bound, no better than G's 472 by a margin that pays), or pass 1 with the MMA and GELU warps is > 550 us (the warp roles do not overlap on one SM), in which case the fused top-row family is closed and the remaining top-row lever is the tensor-core Toeplitz conv (I15).",
    "model": "opus"
  }
]
```