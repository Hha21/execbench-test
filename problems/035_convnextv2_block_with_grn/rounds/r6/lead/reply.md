### Assessment

The remaining score sits in two places. The six wide-C small-image rows are about 0.03 of mean (8,384 64 vs 47 us portal; 8,768 60 vs 44; 16,384 82 vs 67; 16,768 68 vs 58; 2,768 45 vs 35), and predict_score prices a leader-level wide-C result at about +0.018 (M −14 %, L −8 % gives 0.692). The seven small rows are about 0.015 more, and the new portal evidence says their cost is launch structure: A's 5-6 launches run at 1.40-1.55x rented on S rows, single-launch coop kernels at 1.25-1.31x, so each launch removed is worth 1-1.5 us more on the portal than the bench shows.

What the lineages taught us: A wins 13 rows by per-phase efficiency but pays six dependent-round launches; its K2 is not stream-bound (math issue, r5). B's single launch holds the B=1 rows but its mma.sync engine and conv stop it elsewhere. D's tcgen05 engine is operand-ingest-bound (55-72 B/clk/SM) with an undiagnosed 2x at 32,768 and the 14x14 rows. E's CUTLASS batched EVT re-reads W1 per image and cannot beat A's chain; it is paused (incubation forbids retirement).

This round attacks the wide-C rows by two different mechanisms. E1 starts lineage F, the unbuilt library-lean family (I27 plus I8's NCHW GEMM2 and I26's prefetch): LayerNorm finished inside K1, GELU in the cuBLASLt epilogue, a one-launch GRN scale, GEMM2 writing NCHW with the residual as C, so four launches instead of six. E2 gives D its ingest fix: phase stamps, then cta_group::2 pairs or B multicast measured against the MMA ideal, ported into the cooperative kernel. A, the leader, rests this round; its cheap follow-ups (W1-row split, L2 weight prefetch) wait for r7 or are absorbed by F if F wins.

Audit answers are in the ledger notes; the owed precision log for the merge's B and D r3 paths is an operator action on r5-merge-b.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (bulk/TMA-staged dwconv+LN, cuBLASLt FP16 GEMMs, elementwise steps as lean one-round kernels, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: active
    started: r1
    best: r5-a-e2-ringk2
    conclusions:
      - r5 ring K2 (8 uint4 in flight, register sumsq, A&S erf) cut K2 by 28-33 % at 64,96 / 32,128 and now wins 13 of 16 rows in r5-merge-b; K2 is NOT stream-bound (59 % of copy rate), the rest is FMA issue (~100 us at 64,96), so only a GEMM-epilogue GELU removes it.
      - Each custom launch is a 2-3 dependent-round chain and the portal charges A's 5-6 launches at 1.40-1.55x rented on S rows (coop kernels 1.25-1.31x): launch count is now a first-order design axis on S/M rows; unbuilt levers are GELU in the library epilogue (K2 gone), a one-launch GRN scale, NCHW GEMM2 (K4 gone), L2 weight prefetch, W1-row split in the weight pass.
      - cuBLASLt FP16 is 1.4-2.1x faster than TF32 everywhere; at C=768 GEMM1+GEMM2 cost 15.6-24 us cold against a 2.7 us weight stream; NCHW strided-batched GEMM2 with m=HW is slow (swap-AB m=C, n=HW batched form untested).
      - Precision log delivered r5 for all 16 shipped paths: 0 beyond-own-atol elements over 10 seeded rounds, max |err| 2.427e-3 vs atol 2.7e-3 at 1,96 (tightest); no hi/lo split needed.
      - Cross-run and cross-host rented drift is 5-12 us on unchanged rows (A's S rows 26 vs 33 us on two Modal hosts; the portal behaves like the slow host): every pick needs an interleaved same-container control.
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), aiming at S rows <= 24 us rented by cutting the per-phase chain between the two grid barriers.
    status: paused
    started: r1
    best: r3-coop-flat-bulkrows
    conclusions:
      - Single-launch coop beats A's launch pipeline at 1,192 and 1,96 on the portal (37.8 / 35.8 us) even where A looked 5-8 % faster rented, because its portal ratio is 1.25-1.31 against A's 1.40-1.55 on those rows (#66310 vs #66402).
      - H27 refuted (r4): per-tile counters release in 0.64 us but the dataflow kernel was 1.4-2x slower on every S row (per-tile weight re-conversion, static lists); any dataflow must keep a one-pass weight conversion and claim tiles dynamically.
      - Barrier microbench (r4): a flat atomic-counter barrier releases in 0.86-1.07 us vs 1.6-1.9 us for the cg flip barrier; a two-level tree is worse (3.9 us). Unbuilt knob: swap the two grid.syncs for the flat counter (about -1.5 us per S row).
      - Floor with two barriers is 10-13 us, so 15-18 us of SM work and conv-tail wait remain at 1,192 (29.5 us); three rounds of load-path changes did not move it; the mma.sync engine and the conv lose every row with B >= 2 or C >= 384.
      - Pitfalls: runtime guards around mma.sync cost ~470 clk per k-block; register-direct erff epilogues take 16 us per tile; a release arrive with loads in flight stalls; precision log for B's shipped paths (1,192, 1,96) still owed (operator action on r5-merge-b).
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - Retired r3: H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) refuted; no edge over a materialised fp16 hidden and no plan for 8,192 / C>=320.
      - Keep: one-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM; converting inside the ring is 2-2.5x slower, so keep slices fp32-resident and cvt.rn.f16x2 at fragment load.
      - Keep: TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs.
      - Keep: cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry.
      - Conv redundancy 4C/BN (2-6x) and per-CTA serial rounds dominated, not bytes; pool I17/I21 (tcgen05 all-N token tiles) belong to this family and stay in the pool.
  - id: D
    thesis: Cooperative wide-C specialist for the six C>=384, HW<=196 rows (bulk-plane conv, CTA-local GRN statistics, split-K GEMM2 with coalesced red.add) whose GEMM engine is hand-written tcgen05 kind::f16 in the swap-AB orientation (weights as M, tokens as N), so the output tile is NCHW and the stats are thread-local.
    status: active
    started: r3
    best: r4-d-tc05-swapab
    conclusions:
      - r4 tcgen05 swap-AB build passes 16/16 (168 regs, 227.5 KB smem, 148 CTAs x 384 thr): 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us rented; wins only 2,768 (the merge still ships D r3 there, 34.1 rented / 44.8 portal); single-launch portal ratio ~1.32.
      - Validated and reusable: MN-major SW128 B descriptor (LBO 8192, SBO 1024, +2048 per k16), instruction descriptor bits, 2-accumulator GEMM2 (h and g = gw*gf*h) that avoids a B transform; target_hardware must be [B200] only for tcgen05.
      - Mainloop is bound by L2->SM operand ingest at ~55-72 B/clk/SM (same ~135 GB/s/SM as the bulk ring): per 64-k stage 524-683 clk vs a 224-448 ideal, unchanged by stage depth; swap-AB 1SM tiles need (128+N)*64/N B/clk, 1.5-2.6x the cap, so cta_group::2 pairs (per-SM B halves) or B multicast are the r6 test (H33).
      - 32,768 (99 us) and the 14x14 rows (80/112 us) sit ~2x above a bytes-and-ingest estimate of 40-50 us: undiagnosed (no stamps). Suspects: GEMM1 epilogue issue rate (~35-40 instr/element on 8 warps), doubled GEMM2 B ingest from g, scalar red volume, 2 tiles per CTA.
      - Bulk-plane conv 1.5-10 us; coalesced red.global.add.v4 2.4-4.2 us for 2.4-9.6 MB; latency-bound smem loops with runtime trip counts cost 5-17 us each (use compile-time counts with all loads issued first); precision log for D r3's fp16(s*W2) path and D r4 still owed.
  - id: E
    thesis: Library tcgen05 with fused epilogues: CUTLASS 4.4 sm100 builder GEMMs whose EVT epilogues carry the LN fold, exact erf GELU, hidden store and per-image GRN sum of squares (GEMM1) and the residual plus NCHW store (batched split-K GEMM2), fed TF32 operands straight from fp32 memory through TMA where the MMA is cheap and fp16 where it is not.
    status: paused
    started: r5
    best: none
    conclusions:
      - H30 refuted as specified (r5): per-image batching re-reads W1 (9.4 MB at C=768) once per image and TF32 doubles the bytes, so G1 with the full EVT is 17.0 us at 8,768 and 20.0 at 8,384 (kill lines 14 / 12); the pipeline (K1/G1/K3/G2/Kred) beat A r4 only at 2,768 (40.1 vs 48.5) and was 3-23 us slower on the other five rows.
      - EVT nodes that instantiate on sm100 TmaWarpSpecialized1Sm: Sm90ColBroadcast (batch stride), Sm90RowBroadcast, Sm90Compute<multiply_add>, Sm90Compute<GELU> (erf form), Sm90RowReduction with FinalReduction=false (compact partials, no atomics); broadcasts and the reduction cost +0.3-0.8 us, the erf GELU +2.4-8.5 us because the epilogue does not overlap a one-tile-per-CTA mainloop.
      - CUTLASS tfloat32_t on fp32 memory rounds to nearest (H14 confirmed, max |err| 1.1-1.4e-5 vs RNE); TF32 G1+G2 precision log: max |err| 1.67-1.94e-3, 0 beyond atol on the six wide-C rows; split-K as the batch mode (overlapping batch strides) works on sm100.
      - Three .cu files build in 26-50 s wall (36.6 s single-threaded for G1), so the 300 s portal limit is not a risk; -DV= as a macro name breaks the CUTLASS headers.
      - Unresolved: run_tests 14/16 twice from CUPTI sequence-matching errors on changing rows (first kernels of the call fall before the CPU timestamp, no PDL involved); this path must not go to the portal until the cause is known. Paused r6: a non-batched M=B*HW GEMM1 would need a custom segmented per-image reduction node, and even cuBLAS-level TF32 GEMM1 plus GELU leaves no room under A's chain.
  - id: F
    thesis: Library-lean four-launch pipeline for the HW<=784 rows: LayerNorm finished inside the conv launch (cluster-of-8 DSMEM token statistics), cuBLASLt FP16 GEMM1 with the GELU+bias epilogue (no K2), a one-launch GRN statistics-and-scale kernel with per-image counters, and cuBLASLt GEMM2 in the swap-AB batched form writing NCHW with x as the C operand (no K4), with fp32 weights L2-prefetched at entry; wins by launch count, which the portal charges at 1-1.5 us per launch above the bench.
    status: active
    started: r6
    best: null
    conclusions:
      - Seeded r6 from the unbuilt library-lean family (pool I27 plus I8's NCHW GEMM2 and I26's prefetch), aimed at the six wide-C rows first (8,384; 8,768; 16,384; 16,768; 2,768; 32,768), then the HW=784 rows (1,192; 4,192; 8,192) and 1,160.
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r6 2026-10-13.
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
- 'Where the score is (r6; r5-merge-b #66402 = 0.6747 vs leader 0.690 uniform): the six wide-C small-image rows are still the largest gap, about 0.03 of mean (portal ours vs leader-implied: 8,384 64 vs 47 us; 8,768 60 vs 44; 16,384 82 vs 67; 16,768 68 vs 58; 2,768 45 vs 35; 32,768 89 vs 86); predict_score: M -14 % and L -8 % (those rows at leader level) = 0.692, M -15 % / L -12 % = 0.697, plus S -10 % = 0.700. The seven C<=192 rows and 1,160 are about 0.015 (1,192 38 vs 31; 1,160 49 vs 42; 2,96 46 vs 42; 4,192 49 vs 46); 1,96 / 4,96 / 8,320 / 32,128 / 64,96 lead the leader. Per-row S of r5-merge-b (from the leader-gap table): 1,192 0.643; 1,96 0.709; 1,160 0.654; 2,96 0.668; 4,192 0.673; 8,384 0.618; 4,96 0.701; 8,192 0.685; 16,384 0.643; 2,768 0.634; 8,768 0.615; 16,768 0.653; 8,320 0.697; 32,768 0.683; 32,128 0.748; 64,96 0.770.'
- 'Portal/rented ratio (audit r4, extended r6, replaces the uniform 1.28): per path, per row AND per launch structure. Operator evidence #66310 vs #65972 vs #66402: on S/M rows the 5-6 launch A pipeline runs at 1.40-1.55x its rented time, the single-launch cooperative B/D kernels at 1.25-1.31x; moving 1,96 and 1,192 from B to A looked 5-8 % faster rented and was 6-8 % slower on the portal. Rented hosts differ in launch speed (A''s S rows 26 vs 33 us on two Modal containers, B unchanged) and the portal behaves like the slow host. Rule: each launch removed on an S/M row is worth about 1-1.5 us more on the portal than the bench shows; judge S/M designs by launch count as well as rented time; project each path with its own lineage ratio (the merge step now does this). Older per-row figures: A r2 (#65878) 1.16-1.32; B paths 1.24-1.26; D paths ~1.32; fp32 anchor 1.30-1.32.'
- 'Audit r6 answers. (1) Uniform 1.27-1.28 ratio: wrong, corrected in H21 (per-path) and extended with the launch-structure evidence above. (2) Top two rows low priority: corrected in r5 (H29 built: K2 -28/-33 % rented, 64,96 now S 0.770 and 32,128 0.748); the next +0.01 there needs the GELU inside a tensor-core GEMM1 epilogue (lineage F or D engines), K2 is not stream-bound. (3) "E1 lost because of mma.sync": the H26 evidence already carries the correction (MMA share 3-5 us at 1,192; the rest is dependent rounds), stands. (4) Coop S-row floor ~20 us: corrected in H11 to 10-13 us; the flat-counter barrier knob (-1.5 us per S row) is unbuilt and waits for B to resume. (5) Precision margin: delivered in r5 for all 16 of A''s shipped paths (0 beyond-own-atol over 10 seeded rounds, max 2.427e-3 vs atol 2.7e-3 at 1,96) and for E''s TF32 paths; still owed for the merge''s B paths (1,192; 1,96) and D r3''s fp16(s*W2) path (2,768): operator action, run the fp64 10-round log on r5-merge-b as shipped; D''s r6 session logs its own paths. (6) Merge picks: resolved by r5-merge-b (portal-measured paths, 8,768 on A r5 at 60.5 us vs 83.8 / 80.1 before).'
- 'Precision logging (audit r2, tightened r3, delivered r5 for A): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the rms. Every run_tests records per workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded rounds (seed 200) against an fp64 reference, for every path the lineage ships. A row with any beyond-atol element switches the cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M). Passing through the 98 % ratio is not acceptable. A session that ends without the log returns no candidate. Status r6: A 16/16 rows logged (0 beyond atol); E TF32 six rows logged; B and D paths owed (see audit answer 5).'
- 'Cross-session rented timings drift by ~6 us per call and across Modal hosts by 5-12 us on unchanged rows. Every experiment times its parent and the r0 anchor interleaved x3 in its own container. Rented within-noise picks lose on the portal (r3 merge at 8,768); the merge step now times candidates side by side in one container and projects with per-lineage ratios.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65; 8,192 0.63; 32,128 0.62; 64,96 0.65. predict_score r6 vs r5-merge-b: a 10 % cut is worth +0.0073 (S) / +0.0076 (M) / +0.0087 (L).'
- 'Audit r3/r4 on tcgen05 (H23): the ideal GEMM savings on the four GEMM-heavy rows are worth about +0.010 of mean, +0.005 against cuBLASLt; but tcgen05 is also the enabler for fused epilogues (K2 removal at the top rows, K2/K3/K4 removal on wide-C rows), because the custom mma.sync engine lost 4x to cuBLASLt (H26). Routes: D''s hand-written engine (H28, ingest-bound, H33 tests the 2SM fix); E''s CUTLASS builder (H30 refuted as specified, paused); F''s library-lean pipeline removes K2/K4 without a custom engine (H32).'
- 'Cross-problem code: a design session may read another problem''s archive kernel as a structure reference (the #35 tc_ws tcgen05 ring, tricks s5); within this problem lineages still share conclusions only. r4: get_kernel refused #35 tc_ws from lineage D, so the gate ran on a self-written probe. Flagged to the operator.'
- 'Per-SM operand ingest (r4 D gate, H31): a tcgen05 tile fed from L2 through a TMA ring ingests ~55-72 B/clk/SM (~135 GB/s/SM at 1.9 GHz), the same as a 1-D bulk ring (lineage C/D stubs), independent of stage depth. At the portal lock that is ~96-108 GB/s/SM. Any wide-C design must size tiles so that (A tile + B tile) bytes per k-step stay under ~64 B/clk of MMA time, i.e. 2SM pairs, multicast, or weight tiles shared across the N dimension; CTA counts below ~100 leave DRAM/L2 idle and make the row a per-SM ingest chain.'
- 'r4 dataflow lesson (H10/H27): a cheap release-to-observe (0.64 us for one flag polled by 295 CTAs; flat atomic barrier 0.86-1.07 us) does not translate into kernel time; the barrier version wins because each phase''s per-CTA work is uniform and the weight conversion is spread over all warps once per call. Any future dataflow must keep a one-pass weight conversion (or a weights-ready counter) and claim tiles dynamically.'
- 'r5 K2 lesson (H29): a register-prefetch or smem-ring K2 reaches only 47-59 % of the plain copy rate (6.0-6.6 TB/s r+w) because the per-element math (13 instr erf + 7 other) is FMA/INT issue, not MUFU (removing both MUFU ops saves 29 of 319 us at 64,96); the A&S erf form is 1.16e-6 from erff (outside the 2e-7 gate) but changes the end-to-end max error by <= 6e-6; the in-gate NR 10-coefficient form costs ~30 us more at 64,96.'
- 'Cluster note for lineage F (r6): the constraint allows clusters where a GEMM tile or split-K cell needs them; F''s K1 uses a cluster of 8 CTAs per (image, row strip) to exchange per-token LayerNorm statistics through DSMEM (2 floats per token), a one-wave grid of <= 256 CTAs. The #38 penalty (2-5 %) is far below one launch (4-5 us portal); the operator checks legitimacy before a slot, and the session reports a no-cluster control (K1 writing d16 + stats, LN fold in a K2) to isolate the cluster cost.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: 3.3/5.4 us at 784x192x768, 7.2/11.8 at 98x3072x768, 129/270 at the largest, so FP16 is 1.4-2.1x faster everywhere. TF32 only where the weight conversion cannot be overlapped (candidate: 2,768,7,7)'
  - 'r5 A precision log (fp64 reference, 10 seeded rounds, 16 rows): 0 beyond-own-atol elements on every row, max |err| 2.427e-3 vs atol 2.7e-3 at 1,96 (tightest), 2.389e-3 vs 2.9e-3 at 32,768; r5 E TF32 paths 1.67-1.94e-3 on the six wide-C rows. The margin is a max-error statement now, not only rms'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at every size because the implementations are staging-bound, not FMA-bound (K1 315 us in A at the largest against a 134 us FMA bound; 81 us at 32,128 against 29; 18.6-33.5 us of K1 on the W=7/14 rows against 1.5-10 us for a bulk-plane conv); a conv at ~1.5x its FMA bound is worth ~+0.01 at the top two rows, and the bulk-plane conv in A's K1 is worth 8-20 us on seven W=7/14/449x7 rows.
  status: supported
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); ~8 GB/s per SM through LDG/LDGSTS staging'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 us at 1,192, 23 at 8,320, 81 at 32,128, 315 at the largest (3x the FMA bound: LDS about equal to FMA, per-stage __syncthreads, ~40% FMA busy)'
  - 'r3 D: bulk-plane conv (1-D cp.async.bulk of 32-plane groups, thread per 7-output row segment) 1.5 us at 2,768, 6 at 8,768, 10 at 32,768, 8-9 at 8/16,384, correct 16/16'
  - 'r4 A (E1) SUPPORTED for the W=7/14/449x7 part: K1 interleaved x3: 2,768 18.1 -> 9.7; 8,768 23.2 -> 12.2; 32,768 33.4 -> 14.2; 16,384 28.9 -> 16.7; 1,160 17.9 -> 5.2 us; conv alone 3.9-13.2 us; the weight pass (8.9-11 us at C=768, W2 split r5 saves 1.0-1.5 us) now bounds K1 at B<=8. The top-two-row part (conv at 1.5x FMA, 315 -> ~150 us, worth about +0.004 of mean after r5) is still open (pool I24/I28)'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and 0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32'
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal (57 vs 38 us) at 134 MB; at the largest row K2 alone took 447 us (616 MB write + read)'
  - 'audit r4/r6: S is absolute per row, but after the r5 ring K2 the rows sit at 0.770 / 0.748 and the remaining K2 cost is FMA issue of the GELU (H29), which recompute would double. Parked until a tensor-core GEMM1 with a fused GELU epilogue exists (F or D engines)'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192,28,28 and 46.7 vs 50.2 at 2,768,7,7; stubbed floors 4.6 vs 9.9 us'
  - 'portal r6 (#66310 vs #66402): the coop kernel''s portal ratio is 1.25-1.31 against 1.40-1.55 for the 5-6 launch pipeline on S rows, so the portal difference is larger than the bench difference, not within noise'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7) split-K is required to use the SMs; at 1,192,28,28 it is not needed.
  status: supported
  evidence:
  - 'r1 B ablation, split-K auto vs forced 1: 2,768,7,7 46.5 vs 64.6 us, 8,768,7,7 86.1 vs 120.6 (28-29%); 1,192,28,28 29.9 vs 30.6 (none)'
  - 'r4 D gate: 768x1568x3072 without split-K 20 us warm (48 CTAs); H31 explains why: per-SM ingest, not MMA, sets the time at low CTA counts'
- id: H6
  statement: The fp32 SIMT GEMM core runs at 46-50 TF/s on the rented card at 1965 MHz, i.e. 35-38 TF/s at the 1500 MHz portal lock; the anchor loses to 17-25 TF/s through the epilogue and affine. The fp32 SIMT ceiling mean (0.680) is below the leader (0.690), so SIMT GEMMs cannot reach the top.
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s, 57-60 TF/s without global loads; ablations no-epilogue 34-35 TF/s, no-affine 33-35'
  - 'portal #65377 vs rented: 127.2/97.5, 270.0/204.7, 1317.4/1010.5 = 1.30-1.32'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes (416 tiles = 1.4 waves at 32,768,7,7; N=96/160 padded to 128), so per-size tile tables or stream-K matter as much as the core rate.
  status: parked
  evidence:
  - 'r0 cost model picked 128x96 tiles for C=768; r1 B: 1,160,449,7 P3 ~26 us with N=160 padded to 2x128 tiles is the one measured padding loss. r4 A: 1,160 fell to 34.9 us with the bulk-plane conv; r4 D gate: 3072x1568x768 = 168 CTAs = 1.14 waves (4.3 us per CTA)'
- id: H8
  statement: Applying the GRN scale per image to W2 (or scaling the hidden tile in smem while converting to fp16) removes the per-element affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'probe r0: noaff phase 3: 495 vs 678 us at 32,128,64,64, 1816 vs 2684 us at the largest'
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row, a 3-round chain); r3 D folds fp16(s_img*W2) per image; r4 D uses two accumulators (h and g = gw*gf*h, scale inv_b in the epilogue) at the price of doubled GEMM2 B ingest. E1 (r5) used a one-round in-place scale kernel between the two library GEMMs (K3 3.4-7.8 us cold); F (r6) merges the statistics and the in-place scale into one launch with per-image counters (H32); none has been isolated as a win yet'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager (157 us .. 11.5 ms), so even the fp32 anchor scores > 0.5 on the S/M band.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload; the anchor scores 0.10-0.51. Reaching S=0.5 needs t <= 68-222 us on 14 rows'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images' GEMM1 work at B >= 8 and removes two barrier waits.
  status: open
  evidence:
  - 'r4 B (E2) microbench: one red.release flag polled by 295 CTAs is observed in 0.64 us median; a flat atomic-counter barrier releases in 0.86-1.07 us; a two-level tree is worse (3.9 us). The per-tile dataflow kernel (H27) was 1.4-2x SLOWER on every S row for skeleton reasons (per-tile weight re-conversion, static lists). The B >= 8 image-overlap claim is untested at kernel level; any retry must keep a one-pass weight conversion and dynamic tile claiming. Low priority while the launch-pipeline lineages (A, F) and D hold the wide-C rows'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches; what sets the S rows is the SM work and dependent rounds inside each phase (8-10 us per phase at 1,192,28,28 in B; 6-17 us per custom launch in A). Cutting the per-phase chain to ~3-4 us gives S rows of 15-20 us rented (S ~0.78-0.82).
  status: open
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7 (P1 13.2 / P2 +7 / P3 +9.6)'
  - 'r1 A: cold dependent load 0.44 us per step, ~1 us per round once loaded; K1/K2/K3 are 2-3 round chains each'
  - 'r2 B audit at 1,192 (globaltimer): ~12 dependent steps of 1.5-2.5 us between P1 end (1.8 us median, 8.6 max on 48 conv CTAs) and the end at 28.2 us'
  - 'r3 B microbench + audit r4: barrier release 1.6-1.9 us per 296-CTA flip barrier, cold rounds 1.2-3.5 us, just-written 32 KB bulk round 0.83 us. r4 B microbench: a flat atomic-counter barrier releases in 0.86-1.07 us, so the two-barrier coop floor is 10-13 us and 15-18 us of SM work and tail waits remain at 1,192. Portal r6: on S rows the portal charges launches harder than the bench (A 1.40-1.55x vs coop 1.25-1.31x), so per-launch dependent rounds cost 1-1.5 us more each on the portal; launch count is the S-row lever (H32)'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz.
  status: supported
  evidence:
  - 'r1 probes (B and C): 2018-2048 FLOP/clk/SM = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16'
  - 'r3 A (E1) and D: real tiles reach 160 TF/s (64x128, 4-stage cp.async) and 0.64 us per 32-k chunk vs 0.40 ideal (D); cuBLASLt does the same GEMMs ~4x faster, so mma.sync loses on every MMA-heavy row (8,320; 16/32,768; 32,128; 64,96)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, hidden slices in smem, statistics and GEMM2 partials reduced through DSMEM) runs the 7x7/14x14 workloads in <= 30 us.
  status: refuted
  evidence:
  - 'r1 C: DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM; v1 at 2,768,7,7 129 us. Floor ~50-60 us per cluster wave at C=768. Note: a 2-float-per-token statistics exchange (F''s K1) is three orders of magnitude smaller than the bulk DSMEM traffic refuted here'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map; FLOAT32 maps leave bits unchanged'
  - 'r5 E: CUTLASS tfloat32_t A/B on fp32 memory gives max |err| 1.1-1.4e-5 vs an fp64 reference with RNE-rounded operands (4.6-6.0e-3 vs truncated), so CUTLASS''s TFLOAT32 maps round to nearest too. But the library TF32 route lost on time (H30): TF32 doubles operand bytes on an ingest-bound engine and cuBLAS TF32 is 1.4-2.1x slower than fp16, so the zero-conversion idea only pays inside a custom engine that is not ingest-bound'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; 8 heuristic algos per shape, top-3 within 0.3% (checked at 1,192 only)'
  - 'r1 A: NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd; channels-last GEMM2 + 32x32 transpose-add (4-8 us) instead. The swap-AB batched form (m=C, n=HW, batch=images, C operand = x) was never timed: F step 0b (r6)'
  - 'audit r3: supported at 1,192 (and C<=192); at C=768 the two GEMMs cost 15.6-24 us against a 2.7 us weight stream, so the library IS a first-order term on four rows. r5 E: cuBLAS TF32 on the flat shape 7.5/10.7/20.5/9.5 us cold at 2,768/8,768/32,768/8,384 without epilogue. The GELU_BIAS epilogue cost in cuBLASLt is untested: F step 0a (r6)'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little (<= 4 us over the launch floor at 20-42 MB; ~1.4x ideal at 134 MB), so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r2 C (E3): an on-chip hidden with one barrier was no faster than B''s materialised hidden at 1,192 / 2,96'
  - 'r5 A: store hints on the hidden (auto/normal/evict_last/evict_first) are within 1 us except evict_first -4 us at 32,128 (already auto above 42 MB)'
- id: H17
  statement: The anchor's phase 1 at C=768 is block-level weight transposes plus the 4-tokens-per-lane conv; warp-level transposes and zero-padded smem planes cut it to ~45 us.
  status: supported
  evidence:
  - 'r0 probes: 42 vs 11.5 us transposes; 306 vs 59 us conv; smem planes 35.5 us at 32,768,7,7'
- id: H18
  statement: Staging the dwconv input through TMA boxes (3-D map over x for W%4==0 rows; 1-D cp.async.bulk plane copies for 7x7, 14x14 and 449x7 rows) in a multi-stage mbarrier ring lifts conv ingest from ~8 GB/s per SM to >= 40 GB/s per SM, so the conv phase lands within 1.5x of its FMA bound.
  status: open
  evidence:
  - 'r2 A (E1): ingest part SUPPORTED for W%4==0 rows: 15-34 GB/s/SM, 11x the old staging; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96. The 1.5x-of-FMA-bound part is NOT met (3x at the largest). Box inner start must be a multiple of 16 B (start at w=-4, box W+8)'
  - 'r3 D: bulk-plane copies SUPPORTED for W=7/14 (conv phase 1.5-10 us on the six wide-C rows). r4 A: 449x7 SUPPORTED with per-lane 1-D bulk copies (K1 5.2 us at 1,160; conv-only 5.0); bulk issue rate never bound; plane slots must be 16-B aligned in smem (bank skew by channel mod 4 for odd HW). Plane copies also apply to 28x28 (3136 B per plane) and 56x56 (12544 B) images'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be overlapped across the grid syncs; together they cut 1,192,28,28 from 29.7 to <= 22 us rented.
  status: refuted
  evidence:
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring); the remaining cost is barrier release plus cold rounds plus the conv tail (H11, H24)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile in shared memory across the grid sync and reduces GEMM2 split-K partials with red.global.add runs the seven C <= 192 rows in <= 15-20 us rented.
  status: refuted
  evidence:
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3 (kill line 25 us); conv redundancy 4C/BN (2-6x) and serial dependent rounds dominate. Lineage C retired r3; pool I17/I21 (all-N token tiles on tcgen05) are the same family and remain unbuilt'
- id: H21
  statement: The portal/rented ratio is a per-path, per-row and per-launch-structure quantity, not a constant 1.28. Multi-launch pipelines pay more on the portal than single-launch cooperative kernels on S/M rows (A's 5-6 launches 1.40-1.55x, B/D coop 1.25-1.31x), and the portal behaves like the slow rented hosts; each launch removed on an S/M row is worth 1-1.5 us more on the portal than the bench shows. Project every path with its own lineage ratio.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented on 8 rows: 1.27-1.28 each; fp32 anchor 1.30-1.32'
  - 'portal #65878 (A r2, 16 rows): 1.16 at 1,96, 1.20 at 1,192, 1.22 at 2,96 and 2,768, 1.23-1.32 elsewhere. #65879 (merge): B paths 1.24-1.26 (1,192 37.3/29.7), D paths ~1.32 (2,768 45.1/34.1; 8,384 70.7/53.5). r4 merge #65972: bands 1.38 S / 1.36 M / 1.34 L against same-container rented numbers'
  - 'operator r6 (#66310 vs #65972 vs #66402): moving 1,96 and 1,192 from B to A looked 5-8 % faster rented and was 6-8 % slower on the portal; A''s S rows measured 26 vs 33 us on two Modal hosts with B unchanged, and the portal matches the slow host. r5-merge-b (B at 1,96 / 1,192, D r3 at 2,768, A r5 elsewhere) = 0.6747'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol (2.4e-3) on the 77M-element rows; a hi/lo fp16 split of one operand (2 MMAs) restores fp32-class accuracy at no cost where MMA work is <= 2 us (all S/M rows), so per-row error logging decides where to spend it.
  status: supported
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <= 1e-7 fraction beyond 2.4e-3 with randomized LN affine; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r2 A: fp64 reference, 1-3 rounds per row: 0 beyond 2.4e-3 + 1e-5|ref| on all 16 rows, max 2.39e-3 at 32,128. r2 B: 1 element beyond at 4,96 and 8,192 (randomized LN, seed 1). r3 A E1 max 2.35e-3 (1 round); r3 D max 1.52-1.84e-3 (1 seed)'
  - 'r5 A (E2 step 0), fp64 reference, 10 seeded rounds, all 16 rows, parent and ring-K2 kernel: 0 beyond-own-atol elements everywhere; max |err| per row 1.871e-3 (2,768) .. 2.427e-3 (1,96, atol 2.7e-3, 90 % of the budget); the new K2/erf moves the max by <= 6e-6. r5 E TF32: 1.67-1.94e-3 on six rows. Verdict: no hi/lo split needed on A''s paths; the logging rule stays for every new path (B and D r3 paths still owed)'
- id: H23
  statement: tcgen05 kind::f16 (TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the four GEMM-heavy rows and is the lever for the L band once the S/M rows are near the fused floor.
  status: parked
  evidence:
  - 'arithmetic: mma.sync ideal 29/33/76/260 us vs tcgen05 7/8/19/65 us at those rows'
  - 'audit r3: with the score formula the ideal savings give +0.010 of mean, ~+0.005 against cuBLASLt. Audit r4: tcgen05 is the enabler for fused epilogues (H26 refuted on engine grounds), so it is built inside lineage D as H28 (ingest-bound, H33 is the r6 fix) and was tried through the CUTLASS builder in lineage E (H30 refuted as specified)'
- id: H24
  statement: The 2-4 us cost of a 12-32 KB per-CTA load round right after a grid sync is an issue-rate artefact of 4-byte copies, integer div/mod and serialised waits, not post-barrier L2 latency; one cp.async.bulk per operand brings the round to <= 1.2 us, which cuts 1,192 to <= 22 us rented.
  status: refuted
  evidence:
  - 'r3 B microbench (296 CTAs, after the flush): just-written 32 KB round cp4+div 2.4 us / cp16 1.36 / ldg128 0.95 / one bulk 0.83 us median; cold DRAM source 2.7-3.5 us whatever the method; barrier release 1.5-2.0 us after the last arrival. Issue rate is worth ~0.5 us per round'
  - 'r3 B run_tests with a same-build control: bulk rows 1,192 31.4 vs ring 29.5 (+1.9); 1,160 / 4,96 within noise. The 2-4 us post-sync steps are barrier release plus cold or dependent rounds'
- id: H25
  statement: A weight-stationary cooperative kernel for the six wide-C small-image rows (CTA owns a hidden slice and whole-image token tiles, fp32 W slices ingested once through bulk rings, CTA-local GRN statistics, bulk-plane conv, split-K red.add GEMM2) runs 2,768 <= 35 us, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 rented.
  status: refuted
  evidence:
  - 'r3 D stub with real bytes (2 syncs, plane conv, 1.4 MB/CTA W ring at ~135 GB/s/SM, coalesced red): 16-29 us on the six rows, so ingest and post-sync rounds are NOT the binding term'
  - 'r3 D v3 run_tests: 2,768 34.1 (met); 8,768 63.9; 16,768 93.6; 32,768 169.6; 8,384 53.5; 16,384 93.7 (all missed). Binding term: mma.sync 0.64 us per 32-k chunk (ideal 0.40) plus 0.26 us per chunk of bare wait+sync at 16 warps/SM, 96 chunks at 32,768. Fully resident fp32 W fits only at M <= ~400. The skeleton, not the thesis''s ingest, failed: engine swap in H28'
- id: H26
  statement: Replacing cuBLASLt GEMM1 + K2 and K3 + cuBLASLt GEMM2 + K4 in lineage A with two custom mma.sync FP16 GEMMs whose epilogues carry the LN fold, GELU, sumsq (GEMM1) and the s_b reduction, A-scale, residual and NCHW store (GEMM2) cuts the pipeline to 3 launches and the S/M rows by 10-14 us rented.
  status: refuted
  evidence:
  - 'r3 A (E1): 1,192 36.3 us (target <= 26; library 34.3 in-container), 2,768 60.1 (<= 48), 8,384 97.5 (<= 60); GEMM1f alone 12-14 us cold at 1,192 vs cuBLASLt GEMM1 + K2 5.1 + 9.8; at 8,320 the custom GEMMs run at 160 TF/s vs cuBLASLt ~4x faster'
  - 'audit r4 (overstated claim corrected): at 1,192 the MMA share of GEMM1f is 3-5 us, so a tcgen05 engine would save 2-4 us per S row; the rest is 2-3 dependent rounds per launch (H11). The engine matters on M-heavy rows (8,320; 16/32,768), where it is 4x. r6: the launch-count part of the thesis is retried with library GEMMs in F (H32)'
- id: H27
  statement: In the cooperative kernel, replacing grid sync 1 with per-token-tile ready counters (conv CTAs red.release on the tiles they complete; GEMM1 tiles poll with ld.acquire for their own channel groups) and sync 2 with per-image counters, plus spreading the conv over all CTAs, removes the two barrier releases (1.6-1.9 us each) and the conv tail wait (1,192 P1 median 1.8 us vs max 8.6), cutting 1,192 to <= 24 us rented and 2,96 / 4,192 / 1,160 / 4,96 / 8,192 by 4-8 us each (about +0.01-0.015 of mean); for B >= 2 the GEMM2 tiles of early images overlap GEMM1 of later ones (H10).
  status: refuted
  evidence:
  - 'for: r2 B stamps at 1,192: sync1 at 9.9 us with P1 median 1.8 us (conv on 48 CTAs); r4 microbench: one flag released and observed by 295 CTAs in 0.64 us median; a flat atomic barrier releases in 0.86-1.07 us; a monotone counter with a generation number is a legal self-resetting workspace (run_tests 16/16 incl. portal-order re-runs)'
  - 'against (r4 B E2, run_tests): the dataflow kernel is slower on every S row: 1,192 42.8 vs 29.8 (parent) / 30.1 (merge); 1,96 37.5 vs 28.4; 1,160 84.3 vs 51.8; 2,96 50.5 vs 43.4; 4,192 79.4 vs 46.3; 4,96 71.7 vs 56.5; 8,192 137.7 vs 66.5. Suspected (not stamped): every tile re-converts its fp32 weight slice with smem float atomics; static round-robin lists serialise conv -> GEMM1 -> GEMM2 on the first CTAs; at B=1 GEMM2 still waits for the last GEMM1 tile. Both the success line and the refutation clause fired'
- id: H28
  statement: 'A tcgen05 kind::f16 engine (TMA 128B-swizzled ring, warp-uniform TMA/MMA/epilogue roles, TMEM accumulators) in the swap-AB orientation (A = weight tile of 128 hidden or output channels, B = activations [channel][token] MN-major with tokens of one image or image group as N) inside D''s cooperative skeleton makes the six wide-C rows MMA- and byte-bound: thread-local LN fold and GRN sumsq per (image, k) with no atomics, hidden stored [k][token] fp16, GRN scale applied per K-row to the GEMM2 B stage (or per column to the W2 A tile) by a transform warp, output tile [c][tokens] = NCHW for HW=49/196 with the residual in the epilogue. Targets rented: 2,768 <= 32, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 (about +0.03-0.04 of mean at the per-path ratio).'
  status: open
  evidence:
  - 'arithmetic: kind::f16 ideal 0.5/2/4/8 us at 2/8/16/32,768 and 2/4 us at 8/16,384; D stub with real bytes 16-29 us; the per-stage floor (~565 clk) requires 128-k stages at N <= 256 (2N clk per 64-k); #35 tc_ws ring ran at 770 clk per stage vs 768 ideal with uniform roles'
  - 'r4 D (E3) first build, run_tests 16/16: 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us rented; targets met 0 of 6; beats A only at 2,768. Gate GEMM: per 64-k stage 524-683 clk vs 224-448 ideal, independent of stage depth, i.e. bound by L2->SM operand ingest at ~55-72 B/clk/SM (H31), not by an ELECT waterfall. 32,768 and the 14x14 rows are ~2x above a bytes-and-ingest estimate (40-50 us) for undiagnosed reasons (no stamps). Validated descriptors (MN-major SW128 B: LBO 8192, SBO 1024) are reusable. r6 (E2): globaltimer stamps, then cta_group::2 pairs or B multicast (H33), then the biggest stamped term. Comparison bar is now A r5: 48.1 / 56.1 / 70.2 / 50.2 / 62.1 rented at 8,768 / 16,768 / 32,768 / 8,384 / 16,384'
- id: H29
  statement: Rewriting lineage A's K2 (GELU + GRN stats over the fp16 hidden) as a stream-rate kernel (long token runs per CTA, 8 uint4 loads in flight per thread, register-resident sumsq, one atomic per 8 channels per CTA, exact erf, evict_last hidden stores) cuts 447 -> ~200 us at 64,96, 113 -> ~50 at 32,128 and 38 -> ~15 at 8,320 (about +0.01 of mean), and a tcgen05 GEMM1 with the same epilogue (H28/H30 engine) would remove K2 entirely.
  status: supported
  evidence:
  - 'r1 A per-launch: K2 447 us at 64,96 (616 MB write + read, ideal ~175), 113 at 32,128 (ideal ~40), 37.8 at 8,320; 196-thread blocks with 4 loads in flight and smem reductions'
  - 'r5 A (E2): copy probe after the flush 6.6 / 6.0 / 4.2 TB/s r+w at 617 / 134 / 21 MB, old K2 at 42 / 38 / 27 % of it. Ring K2 (smem bulk ring, A&S erf): 444 -> 319 us at 64,96, 117.8 -> 79.5 at 32,128, 37-39 -> 21-23 at 8,320 (targets 260 / 65 / 22: one met); full pipeline 1155 -> 1029, 301 -> 260, 96 -> 80 rented, also -4 to -14 us at 16,384 / 8,192 / 4,96 / 8,384 / 16,768; shipped above 1 MB of hidden. Partially supported: the direction and about +0.01 of mean hold (r5-merge-b 0.6747), the stream-rate targets do not, because the no-math ring floor is 212 us and the remaining ~100 us is FMA/INT issue of the GELU (not MUFU: removing both MUFU ops saves 29 us). The K2-removal half waits for a tensor-core GEMM1 epilogue (F step 2 uses the cuBLASLt GELU epilogue; D''s engine carries it on wide-C rows)'
- id: H30
  statement: 'A CUTLASS 4.4 sm100 builder GEMM1 (batched per image, L = images, M = HW tokens padded to the tile, N = 4C, K = C) whose EVT epilogue carries the LN fold z = rstd[m]*(acc - mu[m]*u[n]) + v[n], the exact erf GELU, the hidden store and a per-batch column reduction of z*z into gf2[b][k], followed by a one-round in-place GRN scale kernel and a batched split-K GEMM2 (A = W2, B = scaled hidden, D [c][hw] per image = NCHW, residual x as the source, bias b2'' = W2*gb + b2) removes K2, K3 and K4 from the wide-C rows; with TF32 operands read straight from fp32 memory through TMA (round-to-nearest, H14) the weight pass disappears on the M <= 400 rows. Targets rented: 2,768 <= 32, 8,768 <= 38, 8,384 <= 42, 16,384 <= 58, 16,768 <= 50, 32,768 <= 68.'
  status: refuted
  evidence:
  - 'r5 E (E1): G1 with the full EVT 17.0 us at 8,768 (kill line 14) and 20.0 at 8,384 (kill 12); broadcasts and the reduction +0.3-0.8 us, erf GELU +2.4-8.5 us (no mainloop overlap at ~1 tile per CTA); binding term is per-image batching re-reading W1 (9.4 MB at C=768) once per image, with TF32 doubling the bytes. Pipeline cumulative cold (K1/G1/K3/G2/Kred): 2,768 40.1; 8,768 47.7; 16,768 61.5; 32,768 95.5; 8,384 62.3; 16,384 94.2 vs A r4 48.5 / 44.4 / 53.2 / 70.1 / 50.4 / 71.5: wins only 2,768, targets 0/6. Plus unresolved CUPTI sequence-matching failures (14/16 twice). Refuted as specified; a non-batched M=B*HW GEMM1 would need a custom segmented per-image reduction node and fp16 operands with a one-pass conversion, and cuBLAS-level TF32 GEMM1 (10.7 us at 8,768) plus GELU leaves no room under A''s chain. Lineage E paused'
- id: H31
  statement: 'On B200 a CTA fed from L2 through a TMA or 1-D bulk ring ingests at most ~55-72 B/clk/SM (~135 GB/s/SM at 1.9 GHz, ~100 GB/s at the portal lock) whatever the stage depth, so a tcgen05 tile whose (A + B) stage bytes exceed ~64 B per clk of MMA time is ingest-bound, and a GEMM with fewer than ~100 CTAs is a per-SM ingest chain. Designs must halve per-SM operand bytes (2SM pairs, multicast of the shared operand across a cluster, weight tiles reused across all tokens in smem) or raise the CTA count with split-K.'
  status: open
  evidence:
  - 'r4 D gate: per 64-k stage 683 clk at N=224 (ideal 448), 587 at N=112 (224), 524 at N=128 (256); stage depth 2-3 vs 8 unchanged; 874-1003 clk at 48 CTAs. r3 C/D stubs: 1-D bulk rings ingest ~120-135 GB/s/SM. The 148-SM aggregate (~20 TB/s at 1.9 GHz) matches the L2 bandwidth figure, so it is unknown whether the cap is the SM port or the L2 slice; the r6 D probe (1SM vs cta_group::2 pair vs B multicast on the same tile, H33) separates the two: a pair halves both per-SM and aggregate bytes, multicast halves only the aggregate L2 reads'
- id: H32
  statement: 'A library-lean four-launch pipeline (K1: bulk-plane conv with the full LayerNorm finished in-CTA through a cluster-of-8 DSMEM exchange of per-token sum/sumsq, token-major fp16 a16, fp16 W1/W2 conversion and b2'' = W2*gb + b2 by role CTAs, fp32 weights L2-prefetched at entry; G1: cuBLASLt FP16 with the GELU+bias epilogue writing the fp16 hidden; K3'': one launch of (image, k-slice) CTAs that take column sumsq over all tokens of the image, exchange sum_k gf through a per-image red.add plus self-resetting arrival counter, and rescale the slice in place; G2: cuBLASLt swap-AB batched per image, D_b[C x HW] = W2 * h_b^T with C = x_b (beta 1) and bias b2'', so the output is NCHW and there is no K4) removes K2 and K4 and the K3 chain from every HW <= 784 row. Targets rented: 8,768 <= 40, 8,384 <= 42, 16,384 <= 54, 16,768 <= 48, 32,768 <= 64, 2,768 <= 34 (A r5: 48.1 / 50.2 / 62.1 / 56.1 / 70.2 / 54.5), later 1,192 <= 28, 4,192 <= 32, 8,192 <= 44 (A r5 34.4 / 38.9 / 52.7); with two launches fewer, the portal gain is 2-3 us per row above the rented gain (H21).'
  status: open
  evidence:
  - 'arithmetic: A r5 at 8,768 is K1 10 + G1 ~6 + K2 ~8 + K3 ~10 + G2 ~9.6 + K4 ~4; the new chain is K1 ~11 + G1 ~7 + K3'' ~5 + G2 ~10 = ~33-37 us; leader-implied rented 34. Risks: the cuBLASLt GELU_BIAS epilogue may pick a slower kernel (#30 C != D precedent); the swap-AB batched G2 with N = 49/196 is untested (the m=HW form was 20-200 us in r1); K3'' needs co-residency for its counter spin (grid <= resident capacity or a cooperative launch); fp16 rounding of a16 ~N(0,1) is the same 2^-11 as d16 (log it). Pool sources I27, I8 (G2), I26 (prefetch). Built in F (E1, r6)'
- id: H33
  statement: 'In D''s swap-AB tcgen05 engine, cta_group::2 pairs (cluster 2x1, M = 256 hidden or output channels across the pair, each CTA loading half of the B token tile, multicast commit to both CTAs'' barriers) or TMA multicast of the B tile across a cluster of two CTAs working on different weight tiles bring the per-64-k stage from 524-683 clk to <= 1.3x the MMA ideal (224-448 clk), and with the biggest stamped term fixed the kernel runs 8,768 <= 40, 16,768 <= 48, 32,768 <= 62, 8,384 <= 42, 16,384 <= 54, 2,768 <= 30 us rented (at least 8 us under A r5 on three or more rows, about +0.015-0.02 of mean at D''s 1.32 ratio).'
  status: open
  evidence:
  - 'arithmetic: 1SM swap-AB stage bytes (128+N)*128 B per 2N clk = 100 B/clk at N=224 vs the 55-72 B/clk cap (H31); a pair needs (128+N/2)*128 B per 2N clk = 69 B/clk at N=224 and 57 at N=128, i.e. at the cap; multicast leaves per-SM bytes unchanged and halves L2 reads, so the pair-vs-multicast probe tells whether the cap is SM-side or L2-side. The 2x unexplained at 32,768 / 14x14 rows (99 / 80 / 112 us vs 40-50 estimate) must be stamped before the engine change is credited. Built in D (E2, r6)'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "F",
    "hypothesis": "H32",
    "title": "New design: library-lean four-launch pipeline (LN in K1 via cluster DSMEM stats, cuBLASLt GELU epilogue, one-launch GRN scale, swap-AB NCHW GEMM2 with residual, L2 weight prefetch)",
    "operation": "new_design",
    "parents": [],
    "band": "all",
    "instructions": "Build from scratch (you may read the r0 baseline r0-convnext-coop-fp32-anchor for host plumbing and the per-lineage conclusions for recipes; no other lineage code). Target rows in order: the six wide-C rows (2,768; 8,768; 16,768; 32,768; 8,384; 16,384), then the HW=784 rows (1,192; 4,192; 8,192), then 1,160,449,7, then 1,96 / 2,96 if time allows. Rows you do not reach run a simple correct fallback (the r0 baseline path is acceptable); the merge keeps A there. CUDA C++ two-file layout (kernel.cu without torch headers, binding.cpp), cuBLASLt from the binding, one handle and per-shape plans set up on first use and valid in any shape order. STEP 0, kill gates, cold after a 2xL2 zero-fill, harness_time x3, interleave the r0 anchor: (a) cuBLASLt fp16 GEMM1 [M x C]x[C x 4C] with CUBLASLT_EPILOGUE_GELU_BIAS (fp16 out, fp32 compute) vs EPILOGUE_BIAS at 392x3072x768, 1568x1536x384, 784x768x192: the GELU delta must be <= 1.5 us per shape (kill at > 3 us on two shapes). (b) GEMM2 in the swap-AB batched form: per image b, D_b[C x HW] = W2[C x 4C] * h_b^T (h_b fp16 [HW x 4C] row-major, so op(B) = T), beta = 1 with C_b = x_b (fp32 [C x HW], the NCHW slab) and a per-row bias b2' = W2*gb + b2 through EPILOGUE_BIAS, strided batch over images (B = 2..32, N = 49 / 196 / 784): time it against channels-last GEMM2 plus a 32x32 transpose-add kernel at 8,768, 8,384, 1,192; adopt swap-AB where it is <= channels-last + 3 us, else keep the transpose-add (5 launches on those rows). (c) A 4-launch stub (each kernel one dependent L2 round of the real bytes) to record the launch floor on this host. STEP 1, K1: 1-D cp.async.bulk plane copies into zero-padded smem (plane groups 16-B aligned; per-lane copies rounded outward to 16 B for 49-float planes; bank-skew slots for odd HW), one thread per 7-output row segment, 49 taps and bias in registers, per-token (sum, sumsq) partials; grid = (image, row strip) x cluster of 8 CTAs along channels (C/8 channels each: 96 at C=768, 48 at C=384, 24 at C=192, 20 at C=160); exchange the 2 floats per token through DSMEM (cluster barrier, each CTA reads the 7 peers' partials), then apply LayerNorm with lnw/lnb in-CTA and write a16 token-major [M][C] fp16 (or the channel-major TT layout if cuBLASLt's heuristic prefers it; test both at 8,768). Role CTAs (or a first round over all CTAs, all loads issued before use) convert W1 and W2 to fp16 with 16-B loads/stores, compute b2' in fp32 (hi/lo fp16 columns not needed since the bias is applied in fp32 by the epilogue), and issue cp.async.bulk.prefetch.L2 of the fp32 W1/W2 at entry so the conversion reads hit L2 (I26). Also build a no-cluster control of K1 (d16 + stats, no LN) to isolate the cluster cost; report both. STEP 2, G1 per step 0a. STEP 3, K3': grid = B x (4C/KS) CTAs with KS = 256 (128 at C<=192), each reads its HW x KS fp16 slice (all tokens of its image) from L2 with every load issued first, column sumsq -> gf[k] = sqrt, red.add of its sum_k gf into a per-image fp32 cell plus a red.release arrival counter in a torch.empty int32 workspace with a generation word (self-resetting: the last arriver clears the cell and bumps the generation; correct for every call and shape order), spins with ld.acquire until the image's 4C/KS CTAs have arrived, computes s_b[k] = 1 + gw[k]*gf[k]/(mean + eps), rescales its slice in place (fp16) and adds gb[k] where the gb term is not folded into b2' (fold it: h' = h*s_b, bias b2' = W2*gb + b2). Co-residency: check with cudaOccupancyMaxActiveBlocksPerMultiprocessor that the grid fits in one wave (max 32x12 = 384 CTAs at 32,768) else launch cooperatively. STEP 4, G2 per step 0b. STEP 5, precision log, mandatory: fp64 reference, 10 seeded rounds from seed 200 with the harness generators, per row max |err| and the count beyond that row's own atol + 1e-5|ref|, for every path you ship (a16 is one fp16 rounding of a ~N(0,1) value, hidden one rounding, W1/W2 one rounding each). STEP 6, run_tests 16/16 with the portal-order check, interleaved with the r0 anchor; report per row rented time and launch count, and the step 0 probe numbers. Report what was not done.",
    "success": "Four launches on the wide-C rows (five if the transpose-add stays) and rented times at least 8 us under A r5 (8,768 48.1; 8,384 50.2; 16,384 62.1; 16,768 56.1; 32,768 70.2; 2,768 54.5) on three or more of the six rows, with 0 beyond-own-atol elements over the 10 seeded rounds on every shipped row; stretch: 1,192 <= 28, 4,192 <= 32, 8,192 <= 44 rented.",
    "refuted_if": "The cuBLASLt GELU_BIAS epilogue costs > 3 us over BIAS at both 8,768 and 8,384 shapes (then the library epilogue route is dead and the finding goes to the pool), or the full four-launch pipeline is within 3 us of A r5 on all six wide-C rows in the same container.",
    "model": "opus"
  },
  {
    "id": "E2",
    "lineage": "D",
    "hypothesis": "H33",
    "title": "D: stamp the 2x at 32,768 / 14x14 rows, then cta_group::2 pairs or B multicast to lift the operand-ingest bound of the swap-AB tcgen05 engine",
    "operation": "structural_mutation",
    "parents": ["r4-d-tc05-swapab"],
    "band": "all",
    "instructions": "STEP 0: run_tests r4-d-tc05-swapab and the r0 anchor interleaved x3 in this container to set the session baseline (cross-host drift is 5-12 us). Deliver the owed precision log: fp64 reference, 10 seeded rounds from seed 200 with the harness generators, per row max |err| and beyond-own-atol count, for the r4 paths on the six wide-C rows and for the r3-ws-coop-widec path at 2,768 (the merge ships it). STEP 1: globaltimer stamps (CTA medians and max) at phase A end, sync 1, GEMM1 mainloop end, GEMM1 epilogue end, sync 2, GEMM2 mainloop end, red end, at 32,768, 16,384, 8,384 and 8,768; identify which term carries the ~2x above the 40-50 us bytes-and-ingest estimate (suspects: GEMM1 epilogue issue rate, doubled GEMM2 B ingest from g, scalar red volume, 2 tiles per CTA). STEP 2, ingest probe on the r4 gate GEMM (128xNT swap-AB tile, SW128 TMA ring, warm L2, clk per 64-k stage, N = 112 / 128 / 224, K = 768 and 3072, 148 CTAs): (a) 1SM as r4; (b) cta_group::2 pair: cluster 2x1, M = 256 across the pair (each CTA its 128 rows of the weight tile), each CTA loads half of the B token tile into its own smem, the even CTA issues the MMA with the pair descriptors, tcgen05.commit with the multicast form to both CTAs' barriers, consumer release through the cluster-scope barrier; (c) 1SM with the B tile TMA-multicast across a cluster of two CTAs that hold different weight tiles. Report clk per stage vs the MMA ideal (2N clk) for each; go with the variant at <= 1.3x the ideal; also record which of (b) and (c) wins, because that tells whether the H31 cap is SM-side or L2-side. STEP 3: port the winner into GEMM1 and GEMM2 of the cooperative kernel, keeping the cooperative launch, the validated descriptors (MN-major SW128 B: LBO 8192, SBO 1024; +2048 per k16), the bulk-plane conv and the CTA-local statistics; pair tiles must keep tile counts a multiple of 74 pairs per row (pick N and split-K per row). Fix the largest stamped term from step 1 (epilogue: 2 rows x 16 columns per lane with every tcgen05.ld issued first; red: red.global.add.v4 instead of scalar; B ingest: per-K-row s_b transform on the B stage instead of the g accumulator if the doubled ingest is the term). STEP 4: run_tests 16/16 with the portal-order check, interleaved with the parent and the r0 anchor; report per row rented times, the stamps before and after, and the per-stage clk. Reminder: target_hardware [B200] only; a release arrive from a warp with loads in flight stalls (tricks s5); clusters are allowed here because the GEMM tile needs them.",
    "success": "Per-64-k stage <= 1.3x the MMA ideal in the probe, and the full kernel at least 8 us under A r5 on three or more of 8,768 (48.1), 16,768 (56.1), 32,768 (70.2), 8,384 (50.2), 16,384 (62.1), with 2,768 <= 32 kept and 0 beyond-own-atol elements over 10 seeded rounds on the six rows.",
    "refuted_if": "Both the pair and the multicast variant stay above 1.5x the MMA ideal per stage (then the ingest cap is not operand-bytes-per-SM and a wide-C single-launch design must reuse weight tiles from smem across all tokens instead), or the ported kernel is not >= 8 us faster than A r5 on any of the five rows.",
    "model": "opus"
  }
]
```

Operator actions outside the two sessions: run the fp64 10-round precision log on r5-merge-b as shipped to cover the B paths at 1,192 and 1,96 (audit item 5), and check the legitimacy of lineage F's cluster-of-8 K1 (a one-wave grid of at most 256 CTAs used only for a 2-float-per-token DSMEM exchange) before it gets a portal slot.