### Assessment

The merge sits at 0.6553. The remaining score is in two places. First, the six wide-C small-image rows (8,768 at 62 µs portal vs the leader's 44, 16,384 at 100 vs 67, 8,384 at 70 vs 47): about 0.04 of mean, and the cost there is lineage A's K2/K3/K4 elementwise chain (about 30 µs per row) rather than the GEMMs. Second, the top two rows plus 8,320, where K2 alone runs at 2.8 TB/s on 616 MB: about +0.01 of mean for a stream-rate rewrite.

Round 4 taught us: the bulk-plane conv in A's K1 works (five rows gained 12-24 µs; the fp32-to-fp16 weight pass now binds K1 at C=768); counter dataflow in B is refuted at kernel level even though a release is observed in 0.64 µs; D's hand-written tcgen05 swap-AB passes 16/16 but is bound by L2-to-SM operand ingest at ~64-72 B/clk/SM and by an undiagnosed epilogue, so it only wins 2,768.

This round runs two experiments. E1 starts lineage E, a new design from the unbuilt library-fusion family: CUTLASS 4.4 sm100 builder GEMMs with EVT epilogues (LN fold, exact erf GELU, fp16/fp32 hidden store, per-image column sum of squares), fed TF32 operands straight from fp32 memory through TMA (round-to-nearest, H14), so there is no weight conversion at all on the M<=400 rows, plus a one-round GRN scale kernel and a batched split-K GEMM2 that writes NCHW with the residual. It attacks the same rows as D by a different route, and it is a second bet on the tcgen05 enabler for fused epilogues. E2 is lineage A's lean-pass repair: K2 at stream rate (H29, including the critic's copy probe), the W2 conversion split over warps, and the owed 10-round precision log. B is paused (its S thesis has not moved in three rounds); D keeps its slot for r6 (stamps, then 2SM or B multicast) unless E lands first.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (bulk/TMA-staged dwconv+LN, cuBLASLt FP16 GEMMs, elementwise steps as lean one-round kernels, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: active
    started: r1
    best: r4-a-e1-bulkplane-wprep
    conclusions:
      - "Bulk-plane conv in K1 (r4) cut K1 from 18-33 us to 10-17 us on the W=7/14/449x7 rows (5 rows gained 12-24 us rented); the conv itself is 4-9 us at C=768 and the fp32->fp16 weight pass (8.9-11 us, W2 rows one warp with 4 serial rounds) now bounds K1 at C=768 with B<=8."
      - "Each custom launch costs 6-17 us at S because it is a 2-3 dependent-round chain; K2 (447 us at 64,96 for 616 MB, 113 at 32,128, 38 at 8,320) and K3 (6-17 us, 3-round chain) are the levers left on this path; a tcgen05/library epilogue would remove them outright (lineage E tests that)."
      - "cuBLASLt FP16 is 1.4-2.1x faster than TF32 everywhere; at C=768 GEMM1+GEMM2 cost 15.6-24 us cold against a 2.7 us weight stream; NCHW strided-batched GEMM2 is slow in cuBLASLt (K4 transpose-add instead)."
      - "Portal/rented ratio for this lineage is 1.16-1.22 on the small multi-launch rows and 1.23-1.32 elsewhere; cross-run drift of 5-12 us on unchanged rows means every pick needs an interleaved same-container control."
      - "Precision max |err| 2.37-2.39e-3 against atol 2.4e-3 on three rows; the 10-seeded-round log is still owed (hard deliverable of E2 r5)."
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), aiming at S rows <= 24 us rented by cutting the per-phase chain between the two grid barriers.
    status: paused
    started: r1
    best: r2-coop-flat-s
    conclusions:
      - "Coop beats three plain launches by 3-7 us at S (F 4.6 vs 9.9 us); mma.sync f16 is enough on every S row; still the merge's best at 1,192 / 1,96 / 2,96."
      - "H27 refuted (r4): per-tile counters release in 0.64 us but the dataflow kernel was 1.4-2x slower on every S row because each tile re-converted its weight slice and the static lists serialised conv -> GEMM1 -> GEMM2 on the first CTAs; a dataflow design must keep the one-pass weight conversion and claim tiles dynamically."
      - "Barrier microbench (r4): a flat atomic-counter barrier releases in 0.86-1.07 us median vs 1.6-1.9 us for the cg flip barrier; a two-level tree is worse (3.9 us). Cheap knob for r6: swap the two grid.syncs for the flat counter barrier (about -1.5 us per S row)."
      - "Floor with two barriers is 10-13 us (F 4.6 + 2 x 1 us releases + one cold round + two just-written rounds), so 15-18 us of SM work and conv-tail wait remain at 1,192 (29.5 us); three rounds of load-path changes did not move it."
      - "Pitfalls: runtime guards around mma.sync cost ~470 clk per k-block; register-direct erff epilogues take 16 us per tile; a release arrive with loads in flight stalls; decide dispatch from run_tests with a same-build control, never from a ctypes probe harness."
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - "Retired r3: H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) refuted; no edge over a materialised fp16 hidden and no plan for 8,192 / C>=320."
      - "Keep: one-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM; converting inside the ring is 2-2.5x slower, so keep slices fp32-resident and cvt.rn.f16x2 at fragment load."
      - "Keep: TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs."
      - "Keep: cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry."
      - "Conv redundancy 4C/BN (2-6x) and per-CTA serial rounds dominated, not bytes."
  - id: D
    thesis: Cooperative wide-C specialist for the six C>=384, HW<=196 rows (bulk-plane conv, CTA-local GRN statistics, split-K GEMM2 with coalesced red.add) whose GEMM engine is hand-written tcgen05 kind::f16 in the swap-AB orientation (weights as M, tokens as N), so the output tile is NCHW and the stats are thread-local.
    status: active
    started: r3
    best: r4-d-tc05-swapab
    conclusions:
      - "r4 tcgen05 swap-AB build passes 16/16 (168 regs, 227.5 KB smem, 148 CTAs x 384 thr): 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us rented; wins only 2,768 against A r4 (48.5 / 44.4 / 53.2 / 70.1 / 50.4 / 71.5). H28 targets missed 0 of 6."
      - "Validated and reusable: MN-major SW128 B descriptor (LBO 8192, SBO 1024, +2048 per k16), instruction descriptor bits, 2-accumulator GEMM2 (h and g = gw*gf*h) that avoids a B transform; target_hardware must be [B200] only for tcgen05."
      - "Mainloop is bound by L2->SM operand ingest at ~55-72 B/clk/SM (same ~135 GB/s/SM as the bulk ring): per 64-k stage 524-683 clk vs a 224-448 ideal, unchanged by stage depth, so neither latency nor an ELECT waterfall. Fix candidates: cta_group::2 pairs (halve per-SM B ingest) or TMA multicast of B across hidden tiles."
      - "32,768 (99 us) and the 14x14 rows (80/112 us) sit ~2x above a bytes-and-ingest estimate of 40-50 us: undiagnosed (no stamps). Suspects: GEMM1 epilogue issue rate (~35-40 instr/element on 8 warps), doubled GEMM2 B ingest from g, scalar red volume, 2 tiles per CTA. r6 step: globaltimer stamps first, then the biggest term."
      - "Bulk-plane conv 1.5-10 us; coalesced red.global.add.v4 2.4-4.2 us for 2.4-9.6 MB; latency-bound smem loops with runtime trip counts cost 5-17 us each (use compile-time counts with all loads issued first)."
  - id: E
    thesis: Library tcgen05 with fused epilogues: CUTLASS 4.4 sm100 builder GEMMs whose EVT epilogues carry the LN fold, exact erf GELU, hidden store and per-image GRN sum of squares (GEMM1) and the residual plus NCHW store (batched split-K GEMM2), fed TF32 operands straight from fp32 memory through TMA where the MMA is cheap (no weight conversion) and fp16 where it is not, with a one-round GRN scale kernel between them.
    status: active
    started: r5
    best: null
    conclusions:
      - "Seeded r5 from the unbuilt library-fusion family (pool I6/I8/I16 plus the zero-conversion TF32 idea I2/I20); aimed at the six wide-C small-image rows first, HW=784 rows next."
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r5 2026-10-13.
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
- 'Where the score is (r5; merge #65972 = 0.6553 vs leader 0.690 uniform): the six wide-C small-image rows are still the largest gap, about 0.04 of mean (portal 8,768 ~62 us vs leader 44; 16,384 ~100 vs 67; 8,384 ~70 vs 47; 16,768 ~69 vs 58; 32,768 ~91 vs 86; 2,768 ~45 vs 35), and on lineage A the cost there is the K2/K3/K4 elementwise chain (about 30 us per row), not the GEMMs. The seven C<=192 rows are about 0.015 (1,160 ~45 vs 42 after r4; 1,192 37 vs 31). The top rows lead the leader (32,128 0.72, 64,96 0.75) but S is absolute: K2 at stream rate is worth about +0.01 (H29). r3-merge per-row S for reference: 1,192 0.646; 1,96 0.707; 1,160 0.572; 2,96 0.654; 4,192 0.651; 8,384 0.593; 4,96 0.682; 8,192 0.661; 16,384 0.550; 2,768 0.632; 8,768 0.533; 16,768 0.560; 8,320 0.655; 32,768 0.604; 32,128 0.720; 64,96 0.748.'
- 'Portal/rented ratio (audit r4, replaces the uniform 1.28): per path and per row. Lineage A r2 (#65878, 16 rows): 1.16 at 1,96, 1.20 at 1,192, 1.22 at 2,96 and 2,768, 1.23-1.32 elsewhere (bands 1.22 S / 1.25 M / 1.26 L). Lineage B paths in the merge (#65879): 1.24-1.26 (1,192 37.3/29.7). Lineage D paths: about 1.32 (2,768 45.1/34.1; 8,384 70.7/53.5). fp32 anchor 1.30-1.32. Project each path with its own ratio; rented gains of cooperative kernels over A''s small-row launches are overstated by 5-10% when a uniform ratio is used. Normalise every rented run by its in-session anchor.'
- 'Precision logging (audit r2, tightened r3, owed r4 and r5): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the rms. Measured single-pass FP16 max |err| 2.37-2.39e-3 at 32,128 / 8,768 / 32,768 (A r2, 1-3 rounds) and 1.9-2.8e-3 (B, randomized LN) against the smallest atol 2.4e-3; atol varies 2.4e-3..3.8e-3 per row (workload.jsonl). Six sessions have now skipped the log, including D''s fp16(s*W2) path and A''s r4 paths that ship in the merge. It is a hard deliverable: every run_tests records per workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded rounds (seed 200) against an fp64 reference, for every path the lineage ships. In r5 it is step 0 of E2 (A''s shipped paths, which cover 11 of the merge''s 16 rows) and a gate of E1 (TF32 paths). A row with any beyond-atol element switches the cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M). Passing through the 98% ratio is not acceptable. A session that ends without the log returns no candidate.'
- 'Cross-session rented timings drift by ~6 us per call (v3fix 50.4/72.6/192.3 vs v3 44.4/66.4/182.8 from a host-only change; r4 A saw 5-12 us drift on unchanged TMA rows between two run_tests in one session). Every experiment times its parent and the r0 anchor interleaved x3 in its own container. Rented within-noise picks lose on the portal: the r3 merge took 8,768 from a 62.4-vs-63.1 rented difference and read 83.8 portal vs 80.1 for r2-lib; r4 re-dispatched 8,768 to A r4 (44.4 rented) and the row now reads ~62 portal.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65; 8,192 0.63; 32,128 0.62; 64,96 0.65. predict_score r5 vs the merge: M -20% and L -15% -> 0.685; M -12% and L -8% (wide-C rows only, at D''s r4 targets) -> 0.672; S -3 / M -2 / L -7 (K2 at stream rate plus the W2 pass) -> 0.665; a 10% cut is worth +0.0074 (S) / +0.0078 (M) / +0.0090 (L).'
- 'Audit r3/r4 on tcgen05 (H23): the ideal GEMM savings on the four GEMM-heavy rows are worth about +0.010 of mean, +0.005 against cuBLASLt; but tcgen05 is also the enabler for fused epilogues (K2 removal at the top rows, K2/K3/K4 removal on wide-C rows), because the custom mma.sync engine lost 4x to cuBLASLt (H26). Two routes to the enabler now run in parallel: D''s hand-written engine (H28, ingest-bound) and E''s CUTLASS builder with EVT (H30).'
- 'Cross-problem code: a design session may read another problem''s archive kernel as a structure reference (the #35 tc_ws tcgen05 ring, tricks s5); within this problem lineages still share conclusions only. r4: get_kernel refused #35 tc_ws from lineage D, so the gate ran on a self-written probe. Flagged to the operator.'
- 'Per-SM operand ingest (r4 D gate, H31): a tcgen05 tile fed from L2 through a TMA ring ingests ~55-72 B/clk/SM (~135 GB/s/SM at 1.9 GHz), the same as a 1-D bulk ring (lineage C/D stubs), independent of stage depth. At the portal lock that is ~96-108 GB/s/SM. Any wide-C design must size tiles so that (A tile + B tile) bytes per k-step stay under ~64 B/clk of MMA time, i.e. 2SM pairs, multicast, or weight tiles shared across the N dimension; CTA counts below ~100 leave DRAM/L2 idle and make the row a per-SM ingest chain.'
- 'r4 dataflow lesson (H10/H27): a cheap release-to-observe (0.64 us for one flag polled by 295 CTAs; flat atomic barrier 0.86-1.07 us) does not translate into kernel time; the barrier version wins because each phase''s per-CTA work is uniform and the weight conversion is spread over all warps once per call. Any future dataflow must keep a one-pass weight conversion (or a weights-ready counter) and claim tiles dynamically.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: 3.3/5.4 us at 784x192x768, 7.2/11.8 at 98x3072x768, 129/270 at the largest, so FP16 is 1.4-2.1x faster everywhere. TF32 only where the weight conversion cannot be overlapped (candidate: 2,768,7,7)'
  - 'margin caveat (audit r3/r4/r5): the "5 sigma" is an rms statement; the max error over 1-3 rounds already sits at 2.37-2.39e-3 against atol 2.4e-3 on three rows and the 10-seeded-round log is still owed (precision note, H22)'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at every size because the implementations are staging-bound, not FMA-bound (K1 315 us in A at the largest against a 134 us FMA bound; 81 us at 32,128 against 29; 18.6-33.5 us of K1 on the W=7/14 rows against 1.5-10 us for a bulk-plane conv); a conv at ~1.5x its FMA bound is worth ~+0.01 at the top two rows, and the bulk-plane conv in A's K1 is worth 8-20 us on seven W=7/14/449x7 rows.
  status: supported
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); ~8 GB/s per SM through LDG/LDGSTS staging'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 us at 1,192, 23 at 8,320, 81 at 32,128, 315 at the largest (3x the FMA bound: LDS about equal to FMA, per-stage __syncthreads, ~40% FMA busy)'
  - 'r3 D: bulk-plane conv (1-D cp.async.bulk of 32-plane groups, thread per 7-output row segment) 1.5 us at 2,768, 6 at 8,768, 10 at 32,768, 8-9 at 8/16,384, correct 16/16'
  - 'r4 A (E1) SUPPORTED for the W=7/14/449x7 part: K1 interleaved x3: 2,768 18.1 -> 9.7; 8,768 23.2 -> 12.2; 32,768 33.4 -> 14.2; 16,384 28.9 -> 16.7; 1,160 17.9 -> 5.2 us; full rows gained 12.6-24.5 us on 5 of 7 rows (8,768 44.4, 1,160 34.9, 16,768 53.2, 16,384 71.5, 32,768 70.1 rented). Conv alone is 3.9-13.2 us; the weight pass (8.9-11 us at C=768) now bounds K1 at B<=8. The top-two-row part (conv at 1.5x FMA, 315 -> ~150 us) is still open (pool I24/I28)'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and 0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32'
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal (57 vs 38 us) at 134 MB; at the largest row K2 alone took 447 us (616 MB write + read)'
  - 'audit r4: the top two rows are NOT low priority (S is absolute per row); but the cheap lever there is K2 at stream rate (H29, E2 r5), not recompute. Parked until H29 lands'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192,28,28 and 46.7 vs 50.2 at 2,768,7,7; stubbed floors 4.6 vs 9.9 us'
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
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row, a 3-round chain); r3 D folds fp16(s_img*W2) per image; r4 D uses two accumulators (h and g = gw*gf*h, scale inv_b in the epilogue) at the price of doubled GEMM2 B ingest. E1 (r5) uses a one-round in-place scale kernel between the two library GEMMs; none has been isolated as a win yet'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager (157 us .. 11.5 ms), so even the fp32 anchor scores > 0.5 on the S/M band.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload; the anchor scores 0.10-0.51. Reaching S=0.5 needs t <= 68-222 us on 14 rows'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images' GEMM1 work at B >= 8 and removes two barrier waits.
  status: open
  evidence:
  - 'r4 B (E2) microbench: one red.release flag polled by 295 CTAs is observed in 0.64 us median; a flat atomic-counter barrier releases in 0.86-1.07 us; a two-level tree is worse (3.9 us). The per-tile dataflow kernel (H27) was 1.4-2x SLOWER on every S row for skeleton reasons (per-tile weight re-conversion, static lists). The B >= 8 image-overlap claim is untested at kernel level; any retry must keep a one-pass weight conversion and dynamic tile claiming. Low priority while the launch-pipeline lineages (A, E) hold the wide-C rows'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches; what sets the S rows is the SM work and dependent rounds inside each phase (8-10 us per phase at 1,192,28,28 in B; 6-17 us per custom launch in A). Cutting the per-phase chain to ~3-4 us gives S rows of 15-20 us rented (S ~0.78-0.82).
  status: open
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7 (P1 13.2 / P2 +7 / P3 +9.6)'
  - 'r1 A: cold dependent load 0.44 us per step, ~1 us per round once loaded; K1/K2/K3 are 2-3 round chains each'
  - 'r2 B audit at 1,192 (globaltimer): ~12 dependent steps of 1.5-2.5 us between P1 end (1.8 us median, 8.6 max on 48 conv CTAs) and the end at 28.2 us'
  - 'r3 B microbench + audit r4: barrier release 1.6-1.9 us per 296-CTA flip barrier, cold rounds 1.2-3.5 us, just-written 32 KB bulk round 0.83 us. r4 B microbench: a flat atomic-counter barrier releases in 0.86-1.07 us, so the two-barrier coop floor is 10-13 us and 15-18 us of SM work and tail waits remain at 1,192. Portal ratio on these rows is 1.16-1.26, not 1.28 (H21). Lineage B paused in r5: three rounds of load-path and sync changes did not move the S rows'
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
  - 'r1 C: DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM; v1 at 2,768,7,7 129 us. Floor ~50-60 us per cluster wave at C=768'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map; FLOAT32 maps leave bits unchanged. CUTLASS/cute maps the tfloat32_t element type to CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 in its TMA descriptors (to be verified bit-wise in E1 step 1), so a CUTLASS tf32 GEMM on fp32 memory gets the same rounding for free. Used by E1 (r5) on the M <= 400 wide-C rows, where the weight pass (8-11 us at C=768) exceeds the TF32 MMA penalty (ideal 1-4 us); kind::tf32 needs K-major A and B, so d and the hidden are written token-major'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; 8 heuristic algos per shape, top-3 within 0.3% (checked at 1,192 only)'
  - 'r1 A: NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd; channels-last GEMM2 + 32x32 transpose-add (4-8 us) instead'
  - 'audit r3: supported at 1,192 (and C<=192); at C=768 the two GEMMs cost 15.6-24 us against a 2.7 us weight stream, so the library IS a first-order term on four rows. The cold cuBLASLt algo/split-K sweep was never run; E1 (r5) tests the CUTLASS builder as the alternative library on those rows'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little (<= 4 us over the launch floor at 20-42 MB; ~1.4x ideal at 134 MB), so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r2 C (E3): an on-chip hidden with one barrier was no faster than B''s materialised hidden at 1,192 / 2,96'
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
  - 'r3 D: bulk-plane copies SUPPORTED for W=7/14 (conv phase 1.5-10 us on the six wide-C rows). r4 A: 449x7 SUPPORTED with per-lane 1-D bulk copies (K1 5.2 us at 1,160; conv-only 5.0); bulk issue rate never bound; plane slots must be 16-B aligned in smem (bank skew by channel mod 4 for odd HW)'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be overlapped across the grid syncs; together they cut 1,192,28,28 from 29.7 to <= 22 us rented.
  status: refuted
  evidence:
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring); the remaining cost is barrier release plus cold rounds plus the conv tail (H11, H24)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile in shared memory across the grid sync and reduces GEMM2 split-K partials with red.global.add runs the seven C <= 192 rows in <= 15-20 us rented.
  status: refuted
  evidence:
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3 (kill line 25 us); conv redundancy 4C/BN (2-6x) and serial dependent rounds dominate. Lineage C retired r3'
- id: H21
  statement: The portal/rented ratio is a per-path, per-row quantity, not a constant 1.28. Lineage A's small multi-launch rows run at 1.16-1.22 (less clock-bound), its other rows at 1.23-1.32; lineage B paths at 1.24-1.26; lineage D paths at about 1.32. Project every path with its own ratio.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented on 8 rows: 1.27-1.28 each; fp32 anchor 1.30-1.32'
  - 'portal #65878 (A r2, 16 rows): 1.16 at 1,96, 1.20 at 1,192, 1.22 at 2,96 and 2,768, 1.23-1.32 elsewhere; bands 49.7/40.7 = 1.22 S, 83.9/66.9 = 1.25 M, 199.1/157.9 = 1.26 L. #65879 (merge): B paths 1.24-1.26 (1,192 37.3/29.7), D paths ~1.32 (2,768 45.1/34.1; 8,384 70.7/53.5). The notebook agrees: S bench -8.5% became portal -4.0% (audit r4). r4 merge #65972: bands 45.7/33.0 = 1.38 S, 67.7/49.8 = 1.36 M, 174.6/130.6 = 1.34 L against the same-container rented numbers (the emulator projected 0.6524, portal 0.6553)'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol (2.4e-3) on the 77M-element rows; a hi/lo fp16 split of one operand (2 MMAs) restores fp32-class accuracy at no cost where MMA work is <= 2 us (all S/M rows), so per-row error logging decides where to spend it.
  status: open
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <= 1e-7 fraction beyond 2.4e-3 with randomized LN affine; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r2 A: fp64 reference, 1-3 rounds per row: 0 beyond 2.4e-3 + 1e-5|ref| on all 16 rows, max 2.39e-3 at 32,128. r2 B: 1 element beyond at 4,96 and 8,192 (randomized LN, seed 1). r3 A E1 max 2.35e-3 (1 round); r3 D max 1.52-1.84e-3 (1 seed)'
  - 'audit r4/r5: the 10-seeded-round per-row log against each row''s own atol has been skipped in six sessions; it is step 0 of E2 (r5) for A''s shipped paths and a gate of E1 for the TF32 paths'
- id: H23
  statement: tcgen05 kind::f16 (TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the four GEMM-heavy rows and is the lever for the L band once the S/M rows are near the fused floor.
  status: parked
  evidence:
  - 'arithmetic: mma.sync ideal 29/33/76/260 us vs tcgen05 7/8/19/65 us at those rows'
  - 'audit r3: with the score formula the ideal savings give +0.010 of mean, ~+0.005 against cuBLASLt. Audit r4: tcgen05 is the enabler for fused epilogues (H26 refuted on engine grounds), so it is built inside lineage D as H28 and, from r5, through the CUTLASS builder in lineage E (H30); the top-row epilogue fusion (K2 removal) waits for H29 and a working fused GEMM1'
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
  - 'audit r4 (overstated claim corrected): at 1,192 the MMA share of GEMM1f is 3-5 us, so a tcgen05 engine would save 2-4 us per S row; the rest is 2-3 dependent rounds per launch (H11). The engine matters on M-heavy rows (8,320; 16/32,768), where it is 4x'
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
  - 'r4 D (E3) first build, run_tests 16/16: 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us rented; targets met 0 of 6; beats A r4 only at 2,768 (32.4 vs 48.5; D v3 had 34.1). Gate GEMM: per 64-k stage 524-683 clk vs 224-448 ideal, independent of stage depth, i.e. bound by L2->SM operand ingest at ~55-72 B/clk/SM (H31), not by an ELECT waterfall. 32,768 and the 14x14 rows are ~2x above a bytes-and-ingest estimate (40-50 us) for undiagnosed reasons (no stamps). Validated descriptors (MN-major SW128 B: LBO 8192, SBO 1024) are reusable. Next step (r6, if lineage E does not land first): globaltimer stamps, then cta_group::2 pairs or B multicast, and the GEMM1 epilogue issue rate'
- id: H29
  statement: Rewriting lineage A's K2 (GELU + GRN stats over the fp16 hidden) as a stream-rate kernel (long token runs per CTA, 8 uint4 loads in flight per thread, register-resident sumsq, one atomic per 8 channels per CTA, exact erf, evict_last hidden stores) cuts 447 -> ~200 us at 64,96, 113 -> ~50 at 32,128 and 38 -> ~15 at 8,320 (about +0.01 of mean), and a tcgen05 GEMM1 with the same epilogue (H28/H30 engine) would remove K2 entirely.
  status: open
  evidence:
  - 'r1 A per-launch: K2 447 us at 64,96 (616 MB write + read, ideal ~175), 113 at 32,128 (ideal ~40), 37.8 at 8,320; 196-thread blocks with 4 loads in flight and smem reductions. Audit r4: S is absolute per row, 64,96 0.748 -> 0.834 and 32,128 0.720 -> 0.811 if K2 vanishes (+0.011); K2 at stream rate alone is about +0.01. Built in E2 (r5) with the critic''s probe (K2 vs a read+write copy of the same bytes) as step 1'
- id: H30
  statement: 'A CUTLASS 4.4 sm100 builder GEMM1 (batched per image, L = images, M = HW tokens padded to the tile, N = 4C, K = C) whose EVT epilogue carries the LN fold z = rstd[m]*(acc - mu[m]*u[n]) + v[n], the exact erf GELU, the hidden store and a per-batch column reduction of z*z into gf2[b][k], followed by a one-round in-place GRN scale kernel and a batched split-K GEMM2 (A = W2, B = scaled hidden, D [c][hw] per image = NCHW, residual x as the source, bias b2'' = W2*gb + b2) removes K2, K3 and K4 from the wide-C rows; with TF32 operands read straight from fp32 memory through TMA (round-to-nearest, H14) the weight pass disappears on the M <= 400 rows. Targets rented: 2,768 <= 32, 8,768 <= 38, 8,384 <= 42, 16,384 <= 58, 16,768 <= 50, 32,768 <= 68 (A r4: 48.5 / 44.4 / 50.4 / 71.5 / 53.2 / 70.1), about +0.015-0.02 of mean at A''s ratio, and the library engine then carries the K2 removal at the top rows in r6.'
  status: open
  evidence:
  - 'arithmetic: at 8,384 A''s K2 + K3 + K4 chain is ~30 us of 50.4; the leader-implied rented times are 27/34/36/52/46/67 at 2,768/8,768/8,384/16,384/16,768/32,768. TF32 ideal MMA 1/4/4/8 us at 2,768/8,768/8,384/16,384 vs a weight pass of 8-11 us at C=768. Risks: Sm90ColReduction may not instantiate under the sm100 epilogue collective (fallback: AuxStore + a cluster-of-8 per-image sumsq in the scale kernel); sm100 stream-K/split-K availability (fallback: batched over (image, k-split) with fp32 partials and a one-round reduce+residual kernel); portal compile time (300 s; keep <= 3 instantiations); per-SM ingest (H31) at low CTA counts. Built in E1 (r5) as lineage E'
- id: H31
  statement: 'On B200 a CTA fed from L2 through a TMA or 1-D bulk ring ingests at most ~55-72 B/clk/SM (~135 GB/s/SM at 1.9 GHz, ~100 GB/s at the portal lock) whatever the stage depth, so a tcgen05 tile whose (A + B) stage bytes exceed ~64 B per clk of MMA time is ingest-bound, and a GEMM with fewer than ~100 CTAs is a per-SM ingest chain. Designs must halve per-SM operand bytes (2SM pairs, multicast of the shared operand across a cluster, weight tiles reused across all tokens in smem) or raise the CTA count with split-K.'
  status: open
  evidence:
  - 'r4 D gate: per 64-k stage 683 clk at N=224 (ideal 448), 587 at N=112 (224), 524 at N=128 (256); stage depth 2-3 vs 8 unchanged; 874-1003 clk at 48 CTAs. r3 C/D stubs: 1-D bulk rings ingest ~120-135 GB/s/SM. The 148-SM aggregate (~20 TB/s at 1.9 GHz) matches the L2 bandwidth figure, so it is unknown whether the cap is the SM port or the L2 slice; a 2SM-pair or multicast probe (D r6) separates the two'
```

```json tasks
[{"id": "E1", "lineage": "E", "hypothesis": "H30", "title": "New lineage E: CUTLASS sm100 builder GEMMs with EVT fusion and TF32 operands straight from fp32 memory for the six wide-C small-image rows", "operation": "new_design",
  "parents": ["r0-convnext-coop-fp32-anchor"], "band": "M|L",
  "instructions": "Build from scratch (you may read only the r0 anchor and the lineage conclusions; do not read lineage A/B/D code). Target rows: 2,768,7,7; 8,768,7,7; 16,768,7,7; 32,768,7,7; 8,384,14,14; 16,384,14,14. All other rows run the r0 anchor path unchanged (the operator merges per workload). Language 'cutlass' (C++; two files: kernel.cu with no torch headers plus binding.cpp; CUTLASS 4.4.1 headers at $CUTLASS_DIR/include; target_hardware [B200] only, since tcgen05 code does not build for plain sm_100). Pipeline per call, 4-5 launches: (K1) your own bulk-plane depthwise conv for 7x7/14x14 images: 1-D cp.async.bulk of channel-plane groups into zero-padded smem, one thread per 7-output row segment, plus the dwconv bias; write d as fp32 TOKEN-MAJOR [b][hw][C] (tcgen05 kind::tf32 needs K-major A and B), per-token LN partial sums (sum, sumsq) via atomics or per-group partials finalised as mu[token], rstd[token]; spare CTAs compute the fold vectors u[n] = sum_c W1[n][c]*lnw[c], v[n] = sum_c W1[n][c]*lnb[c] + b1[n] and b2'[c] = sum_k W2[c][k]*gb[k] + b2[c] (coalesced 16-B loads, all loads issued before use, compile-time trip counts). No weight conversion in the first build: both GEMMs use TF32. (G1) CUTLASS CollectiveBuilder, arch Sm100, KernelTmaWarpSpecialized1SmSm100 (or 2Sm where the tile allows), batched L = B images, M = HW (49 or 196, tile M = 64 or 128), N = 4C, K = C; ElementA = ElementB = cutlass::tfloat32_t on the fp32 d and fp32 W1 (verify in step 1 that the resulting TMA descriptor uses the TFLOAT32 data type: compare one tile's accumulator against an fp64 reference of round-to-nearest TF32 operands and against truncation; the bit pattern must match RNE, H14). EVT epilogue: Sm90RowBroadcast(rstd, mu with batch stride), Sm90ColBroadcast(u, v), compute z = rstd*(acc - mu*u) + v, cutlass::epilogue::thread::GELU (the erf form, NOT GELU_taylor), store the hidden (fp32 for the TF32 rows, [b][hw][4C] K-major for GEMM2) and Sm90ColReduction of z*z over M per batch into gf2[b][k] (atomic across M tiles where M > tile). If the sm100 epilogue cannot instantiate the ColReduction, fall back to an AuxStore of the hidden and compute the per-image sumsq in K3 with a cluster of 8 CTAs per image exchanging the 4C partials through DSMEM. (K3) one-round GRN scale kernel: grid (image, k-slice), each CTA reads gf2[b][:] (12 KB), computes gf = sqrt, mean_k, s_b[k] = 1 + gw[k]*gf/(mean+eps), then rescales its hidden slice in place with all loads issued before any store; nothing else is chained. (G2) CUTLASS batched: A = W2 [C][4C] (tf32, batch stride 0), B = scaled hidden [hw][4C] per image (K-major, N = HW), D[b][c][hw] with batch stride C*HW = NCHW directly, source C = x (beta = 1) and per-row bias b2'. The CTA count is tiny at 7x7 (6*B tiles), so use split-K: first try the sm100 stream-K/split-K scheduler; if it is not available in 4.4.1, run batched over (image, k-split) with fp32 partials [B*KS][C][HW] and a one-round reduce+residual kernel (coalesced, all loads first). Steps: (0) time the r0 anchor interleaved x3 as the in-session control; (1) compile gate with no GPU needed first: G1 and G2 instantiations for 49x3072x768 (L=8) and 196x1536x384 (L=8), record nvcc wall time (abort the design if two instantiations exceed 200 s: the portal limit is 300 s), registers, smem, spills; then the H14 bit check; (2) time G1 cold (after a 2xL2 zero-fill) at 8,768 and 8,384 and compare with the leader-implied budget: kill the EVT route if G1 alone exceeds 14 us at 8,768 (that is cuBLASLt GEMM1 + K2 on lineage A); (3) full pipeline, run_tests 16/16 x 10 rounds with the portal-order check; (4) precision log on the six rows: 10 seeded rounds (seed 200) vs an fp64 reference, per row max |err| and count beyond atol + 1e-5|ref| (TF32 RNE is the same 10-bit mantissa as fp16, so expect max about 2.4e-3 at 32,768; any beyond-atol element switches that row's activation operand to an fp16 hi/lo split or fp32 SIMT); (5) per-launch cold breakdown (harness_time) of K1/G1/K3/G2 on 8,768 and 8,384, and if GEMM2 is ingest-bound at low CTA counts (H31), raise KS. Report every finding with its source, including compile times and which EVT nodes instantiated.",
  "success": "run_tests 16/16; the six target rows at or under 2,768 32 / 8,768 38 / 8,384 42 / 16,384 58 / 16,768 50 / 32,768 68 us rented on at least 4 of 6 rows, with the parent-equivalent merge rows timed in the same container (operator-provided numbers: A r4 48.5 / 44.4 / 50.4 / 71.5 / 53.2 / 70.1, D r4 32.4 at 2,768); CUTLASS build time under 200 s; precision log delivered with 0 beyond-atol elements.",
  "refuted_if": "G1 with the EVT epilogue runs slower cold than 14 us at 8,768 or 12 us at 8,384 (then the library epilogue is no cheaper than cuBLASLt + K2), or the per-batch column reduction and the split-K GEMM2 both fail to instantiate on sm100 and the fallbacks push the six rows above A's times, or the build exceeds 200 s.",
  "model": "opus"},
 {"id": "E2", "lineage": "A", "hypothesis": "H29", "title": "Lineage A lean passes: K2 at stream rate, W2 conversion split over warps, and the owed 10-round precision log", "operation": "structural_mutation",
  "parents": ["r4-a-e1-bulkplane-wprep"], "band": "all",
  "instructions": "Parent: r4-a-e1-bulkplane-wprep (lineage A's best). Step 0 (hard deliverable, do first): precision log for the parent as shipped, all 16 rows, the harness's 10 seeded rounds (seed 200) against an fp64 reference: per row the max |err| and the count of elements beyond that row's own atol (workload.jsonl) + 1e-5|ref|. If any row has a beyond-atol element, switch its hidden or W2 operand to a hi/lo fp16 split (2 cuBLASLt GEMMs accumulated, or hi/lo columns) and re-log. Step 1 (critic's probe, H29): time K2 alone after the 2xL2 zero-fill at 64,96 / 32,128 / 8,320 against a plain read+write copy kernel of the same fp16 bytes (616 / 134 / 21 MB), interleaved x3; record the copy rate in TB/s. Step 2: rewrite K2 (GELU + GRN sum of squares over the fp16 hidden) as a stream-rate kernel: 256-thread CTAs, each CTA owns a long token run of ONE image (thousands of tokens at 64,96; the whole image at small rows), each thread fixed on 8 hidden channels (one uint4), 8 uint4 loads in flight before the first erff, exact erff kept (a custom erf only if it is at least 1.3x faster AND |err| <= 2e-7 vs erff over [-10,10]), sumsq accumulated in registers across the run, one float4 atomic pair per 8 channels per CTA into the 4-copy accumulators, bias b1 folded as now. Try evict_last on the hidden stores for rows whose hidden is <= 42 MB and evict_first above; keep whichever wins per row class. Keep the kernel sequence identical on every call. Step 3: the W2 conversion in K1 (rows with the hidden > 16 MB: B >= 8 at C=768, B = 16 at C=384) is one warp per W2 row with 4 serial 768-column rounds: split each row over 4 warps with a CTA-level reduction and 16-B loads, all loads issued before use; target weights-only K1 <= 6 us at C=768 (now 8.9-11), measured with the conv-only / weights-only / both ablation at 2,768, 8,768, 16,768. Step 4: run_tests 16/16 x 10 rounds with the parent interleaved x3 in the same container as the control, and the r0 anchor once for the ratio; report per-row rented deltas, not band means. Step 5 (optional if time remains): the nwmin 4 vs 8 interleaved comparison left open in r4. Report findings with sources; a session that ends without the step 0 log returns no candidate.",
  "success": "K2 at 64,96 <= 260 us rented (from 447), 32,128 <= 65 (from 113), 8,320 <= 22 (from 38), so the full rows reach 64,96 <= 960, 32,128 <= 245, 8,320 <= 75 rented against the parent in the same container; weights-only K1 <= 6 us at C=768 with 8,768 and 16,768 gaining >= 3 us; precision log delivered with 0 beyond-atol elements on all 16 rows (or the hi/lo fix applied where needed).",
  "refuted_if": "The copy probe shows the parent's K2 already at >= 80% of the copy kernel's rate at 64,96 (then the kernel is SIMT-bound and only a cheaper erf helps), or the rewritten K2 is within 15% of the old one at 64,96 and 32,128 in the same container.",
  "model": "sonnet"}]
```

**Audit answers.** The uniform 1.28 ratio and the low-priority top rows were both corrected in the r4 ledger (H21, H3/H29); the new r4 merge bands read 1.34-1.38 against same-container rented numbers, added to H21. The engine-only reading of the r3 fused-GEMM loss was corrected in H26. The 20 µs cooperative floor is now 10-13 µs after the r4 flat-barrier microbench (H11), and lineage B is paused rather than retired. The precision claim stays unsupported: the 10-round log is now step 0 of E2 and a gate of E1, and a session without it returns no candidate. The 8,768 path choice was resolved in r4 (A r4 at 44.4 µs rented, about 62 µs portal).

**Pool handling.** Built or refuted in r4: the bulk-plane conv port (I5/I22), tile-level dataflow (I1), the hierarchical barrier (I10, worse), and the hand tcgen05 swap-AB (I7/I18/I25). E1 seeds the unbuilt library-fusion family (I6/I8/I16) merged with zero-conversion TF32 (I2/I20). E2 is the pipeline-repair family (I9) plus the W2 conversion lever. Still in the pool for later rounds: register and Toeplitz convs for the top rows (I24/I28, I13/I15/I29), fused persistent top-row kernels (I11/I12/I19/I30), image-group dataflow (I23), weight prefetch hints (I26), on-chip hidden revisits (I17/I21), and die-local placement (I14).