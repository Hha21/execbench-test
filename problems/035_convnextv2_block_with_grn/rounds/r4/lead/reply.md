### Assessment

The merge sits at 0.6355 against a leader at 0.690. Per row, the gap is 0.042 of mean on the six wide-C small-image rows (8,768 is the worst at S 0.533), 0.017 on the seven C<=192 rows (1,160,449,7 alone is 0.007: both A and B take 54 us there), and we lead by 0.03-0.06 on the top two rows, where the critic is right that an absolute +0.011 is still available from K2 (447 us for 616 MB). The score model says a 25% cut on the L band is worth +0.025 and 20% on S+M +0.032, so the wide-C engine and the small-row chain are both worth a lineage.

What r3 taught: lineage A's fused mma.sync GEMMs lost to cuBLASLt by 4x on MMA rate (H26 refuted), so fusion needs a tcgen05 or library engine. Lineage B showed the post-sync "2-4 us rounds" are barrier release (1.6-1.9 us) plus cold rounds, not copy issue rate (H24 refuted); the critic's floor correction (11-14 us, not 20) keeps B alive only on a barrier-free route. Lineage D passed its stub gate (17-32 us with real bytes) and then lost 40-140 us to its own mma.sync skeleton; its weight ingest was never the problem.

Audit answers: H21 becomes a per-path ratio table (no run needed); H3/H23 are rescoped (K2 fix queued as H29); the E1 claim is softened (S rows are round-bound, engine matters at M-heavy rows only); B's floor note corrected; the 10-seeded-round precision log is mandatory in every experiment this round (it covers the merge's paths lineage by lineage); the 8,768 path is checked interleaved in E1. I could not read the merge's card (the archive entry exceeds the tool limit and no file tool is available), so the dispatch check is delegated.

This round: A ports the bulk-plane conv and a coalesced weight pass into K1 for the W=7/14/449x7 rows (7 rows, low risk). B tests counter-released dataflow against the barriers after a microbench. D swaps its engine to tcgen05 in the swap-AB orientation, gated by a GEMM probe.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (TMA/bulk-staged dwconv+LN, tensor-core GEMMs with elementwise steps fused into epilogues, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: active
    started: r1
    best: r2-lib-cublaslt-fp16-tmaconv
    conclusions:
      - Each custom launch costs 6-17 us at S (2-3 dependent rounds plus ramp), so a 5-6 launch pipeline floors at 35-40 us; custom mma.sync fused GEMMs (r3 E1) lost 4x in MMA rate to cuBLASLt (tcgen05), so fusion must ride a tcgen05 or library epilogue, and on S rows the residual cost is per-CTA rounds, not the engine.
      - TMA 3-D box staging lifted conv ingest 11x (K1 5.6 us at 1,192, 315 us at the largest, still 3x the FMA bound, LDS/sync-bound); W=7/14/449x7 rows still use the 8 GB/s/SM staging (K1 18.6-33.5 us at C=768) while D's bulk-plane conv does the same conv in 1.5-10 us.
      - cuBLASLt FP16 is 1.4-2.1x faster than TF32 everywhere; at C=768 GEMM1+GEMM2 cost 15.6-24 us cold against a 2.7 us weight stream, so the library is a first-order term on the four C=768 rows, not at 1,192.
      - K2 (GELU+stats) is the largest custom launch at the top rows (447 us at 64,96 for 616 MB, 113 at 32,128): a stream-rate rewrite is worth about +0.01 of mean (H29, queued).
      - Portal/rented ratio for this lineage is 1.16-1.22 on the small multi-launch rows and 1.23-1.32 elsewhere (#65878); precision max |err| 2.37-2.39e-3 against atol 2.4e-3 on three rows, 10-seeded-round log still owed.
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), now aiming at S rows <= 24 us rented by replacing the grid barriers with counter-released dataflow so GEMM tiles start when their inputs land instead of after the slowest CTA.
    status: active
    started: r1
    best: r2-coop-flat-s
    conclusions:
      - Coop beats three plain launches by 3-7 us at S (F 4.6 vs 9.9 us); mma.sync f16 at 0.455 PF/s is enough on every S row.
      - H24 refuted at kernel level: a 32 KB just-written bulk round costs 0.83 us, so copy issue rate is worth ~0.5 us per round; the 2-4 us post-sync steps are barrier release (1.6-1.9 us per 296-CTA flip barrier) plus cold-DRAM rounds (1.2-3.5 us whatever the load method).
      - At 1,192 the conv runs on 48 CTAs and P1 ends at 1.8 us median but 8.6 us max, so sync 1 releases at 9.9 us: most of the pre-GEMM1 time is tail wait on one slow conv CTA, not conv work.
      - Floor with two barriers is about 11-14 us (F 4.6 + 2 releases + one cold round + two just-written rounds), leaving 15-18 us of SM work and waits at 1,192 (29.5 us) to attack.
      - Pitfalls: runtime guards or partial unrolls around mma.sync cost ~470 clk per k-block; register-direct 48-erff epilogues take 16 us per tile (smem-staged rolled loop with 16-B stores fixes it); a release arrive with loads in flight stalls; ctypes probe harnesses mis-time by 4-5 us, decide dispatch from run_tests with a same-build control.
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - Retired r3: H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) both refuted; no edge over a materialised fp16 hidden and no plan for 8,192 / C>=320.
      - Keep: one-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM; converting inside the ring is 2-2.5x slower, so keep slices fp32-resident and cvt.rn.f16x2 at fragment load.
      - Keep: TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs.
      - Keep: cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry.
      - Conv redundancy 4C/BN (2-6x) and per-CTA serial rounds dominated, not bytes.
  - id: D
    thesis: Cooperative wide-C specialist for the six C>=384, HW<=196 rows (bulk-plane conv, CTA-local GRN statistics, split-K GEMM2 with coalesced red.add) whose GEMM engine is tcgen05 kind::f16 in the swap-AB orientation (weights as M, tokens as N), so the output tile is NCHW and the stats are thread-local.
    status: active
    started: r3
    best: r3-ws-coop-widec
    conclusions:
      - The stub with real bytes (2 syncs, plane conv, 1.4 MB per CTA of W through a 5x32 KB bulk ring, coalesced red) runs 16-29 us on all six rows, so weight ingest and post-sync rounds are not the binding term; H25 refuted as stated because the mma.sync skeleton costs 40-140 us (0.64 us per 32-k chunk, 0.26 us per chunk of bare wait+sync, 16 warps/SM with no ILP).
      - Bulk-plane conv (1-D cp.async.bulk of 32-plane groups into zero-padded smem, thread per 7-output row segment) costs 1.5 us at 2,768, 6 at 8,768, 10 at 32,768, 8-9 at 8/16,384.
      - Scattered scalar red.add is 8-30 us for 2.4-9.6 MB; coalesced red.global.add.v4.f32 from 128 CTAs does the same in 2.4-4.2 us (stage partials as [img][c][hw]).
      - Fully resident fp32 W slices fit only where M <= ~400 at C=768; the design is weight-streaming per slice, not weight-stationary.
      - Latency-bound smem loops (runtime trip counts, rolled warp-per-row GELU, serial __ldcg stats) cost 5-17 us each; compile-time trip counts with all loads issued first fixed v3 (2,768 34.1 us, 8,384 53.5 us).
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r4 2026-10-12.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window. No programmatic dependent launch at all, even between our own kernels: the lint rejects griddepcontrol
  and the PDL launch attributes, so an idea that relies on PDL must be built without it.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids
  lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot.'
- 'Precision: FP16 and TF32 operands with fp32 accumulation pass every workload with ~5 sigma margin (simulated with the
  harness generators); BF16 fails. Any reduced-precision candidate must be re-simulated or run_tests 16/16 before a slot; the
  erf GELU must stay exact (no tanh approximation).'
- 'Weights are converted/transposed inside the call (no cross-call cache of fp16 weights or descriptors; pointers shift every call).'
- 'Workspaces come from torch.empty inside run() (no kernel launched); a cooperative launch counts as one kernel. Grid barriers
  via cg::grid_group or a self-resetting counter only.'
- 'Host state shared across shapes (function attributes such as max dynamic shared memory, cached plans, handles) must
  be right for every shape in any order: the portal sets up every workload before it times them (#65570: per-shape
  cudaFuncSetAttribute shrank the limit for earlier shapes, code 101 on 8 workloads). Set such attributes to the maximum
  any shape needs, or only ever raise them. run_tests re-runs every workload after all have run once and reports a
  PORTAL-ORDER CHECK failure.'
notes:
- 'Report effects per workload (or at least C=768 rows vs others), not per MB band: the MB bands mix weight-bound (C=768 7x7),
  latency-bound and stream-bound rows. Card fix pending: the MB cell for 64,96,112,112 is 616.9 MB, not "118.0 G".'
- 'Where the score is (r4, per-row S of the merge #65879 = 0.6355 vs leader 0.690 uniform): 1,192 0.646; 1,96 0.707; 1,160 0.572;
  2,96 0.654; 4,192 0.651; 8,384 0.593; 4,96 0.682; 8,192 0.661; 16,384 0.550; 2,768 0.632; 8,768 0.533; 16,768 0.560;
  8,320 0.655; 32,768 0.604; 32,128 0.720; 64,96 0.748. Gap to the leader: six wide-C small-image rows 0.042 of mean
  (8,768 alone 0.010), seven C<=192 rows 0.017 (1,160,449,7 alone 0.007), 8,320 0.002; we lead by 0.03/0.06 on 32,128 / 64,96,
  but the score is absolute per row: removing K2''s 447 us at 64,96 and 113 us at 32,128 is worth about +0.011 (audit r4).
  Leader-implied portal times: S rows 30.7-45.8 us; 2/8/16/32,768 35/44/58/86; 8/16,384 47/67; 1,160 41.8.'
- 'Portal/rented ratio (audit r4, replaces the uniform 1.28): per path and per row. Lineage A r2 (#65878, 16 rows): 1.16 at 1,96,
  1.20 at 1,192, 1.22 at 2,96 and 2,768, 1.23-1.32 elsewhere (bands 1.22 S / 1.25 M / 1.26 L). Lineage B paths in the merge
  (#65879): 1.24-1.26 (1,192 37.3/29.7). Lineage D paths: about 1.32 (2,768 45.1/34.1; 8,384 70.7/53.5). fp32 anchor 1.30-1.32.
  Project each path with its own ratio; rented gains of cooperative kernels over A''s small-row launches are overstated by
  5-10% when a uniform ratio is used. Normalise every rented run by its in-session anchor.'
- 'Precision logging (audit r2, tightened r3, owed r4): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the
  rms. Measured single-pass FP16 max |err| 2.37-2.39e-3 at 32,128 / 8,768 / 32,768 (A r2, 1-3 rounds) and 1.9-2.8e-3 (B, randomized
  LN) against the smallest atol 2.4e-3; atol varies 2.4e-3..3.8e-3 per row (workload.jsonl). Four sessions skipped the log,
  including D''s fp16(s*W2) path that ships in the merge. From r4 it is a hard deliverable: every run_tests records per
  workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded rounds
  (seed 200) against an fp64 reference, for every path the lineage ships (A r2 paths in E1, B paths in E2, D paths in E3; together
  they cover the merge). A row with any beyond-atol element switches the cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M).
  Passing through the 98% ratio is not acceptable.'
- 'Cross-session rented timings drift by ~6 us per call (v3fix 50.4/72.6/192.3 vs v3 44.4/66.4/182.8 from a host-only
  change; the notebook''s v3fix run_tests entry is a verbatim copy of v3''s numbers and must not be cited as a v3fix measurement).
  Every experiment times its parent and the r0 anchor interleaved x3 in its own container. Rented within-noise picks lose on the
  portal: the merge took 8,768 from a 62.4-vs-63.1 rented difference and reads 83.8 portal vs 80.1 for r2-lib (audit r4); E1
  re-times the two 8,768 paths interleaved and the operator re-dispatches.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65;
  8,192 0.63; 32,128 0.62; 64,96 0.65. predict_score r4 vs the merge: -20% on S and M -> 0.667; -25% on L -> 0.660;
  25 us + bytes/1 TB/s on every row -> 0.771; a 10% cut is worth +0.0075 (S) / +0.0078 (M) / +0.0093 (L).'
- 'Audit r3/r4 on tcgen05 (H23): the ideal GEMM savings on the four GEMM-heavy rows are worth about +0.010 of mean, +0.005 against
  cuBLASLt; but tcgen05 is also the enabler for fused epilogues (K2 removal at the top rows, K2/K3/K4 removal on wide-C rows),
  because the custom mma.sync engine lost 4x to cuBLASLt (H26). It is pursued as D''s engine (H28), not as a lineage thesis.'
- 'Cross-problem code: a design session may read another problem''s archive kernel as a structure reference (the #35 tc_ws
  tcgen05 ring, tricks s5); within this problem lineages still share conclusions only. Flagged to the operator (r4).'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload
    memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still
    under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: 3.3/5.4 us at 784x192x768, 7.2/11.8 at 98x3072x768, 129/270 at the largest, so FP16 is 1.4-2.1x faster everywhere. TF32 only where the weight conversion cannot be overlapped (candidate: 2,768,7,7)'
  - 'margin caveat (audit r3/r4): the "5 sigma" is an rms statement; the max error over 1-3 rounds already sits at 2.37-2.39e-3 against atol 2.4e-3 on three rows and the 10-seeded-round log is still owed (precision note, H22)'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at every size because the implementations are staging-bound,
    not FMA-bound (K1 315 us in A at the largest against a 134 us FMA bound; 81 us at 32,128 against 29; 18.6-33.5 us of K1
    on the W=7/14 rows against 1.5-10 us for a bulk-plane conv); a conv at ~1.5x its FMA bound is worth ~+0.01 at the top
    two rows, and the bulk-plane conv in A's K1 is worth 8-20 us on seven W=7/14/449x7 rows.
  status: open
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); ~8 GB/s per SM through LDG/LDGSTS staging'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 us at 1,192, 23 at 8,320, 81 at 32,128, 315 at the largest (3x the FMA bound: LDS about equal to FMA, per-stage __syncthreads, ~40% FMA busy)'
  - 'r3 D: bulk-plane conv (1-D cp.async.bulk of 32-plane groups, thread per 7-output row segment) 1.5 us at 2,768, 6 at 8,768, 10 at 32,768, 8-9 at 8/16,384, correct 16/16; A''s K1 on the same rows 18.6/23.7/33.5/28.4 us with the old staging plus the weight pass. E1 (r4) ports it into A''s K1'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and
    0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32'
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal (57 vs 38 us) at 134 MB; at the largest row K2 alone took 447 us (616 MB write + read)'
  - 'audit r4: the top two rows are NOT low priority (S is absolute per row); but the cheap lever there is K2 at stream rate (H29), not recompute. Parked until H29 lands'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card
    and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192,28,28 and 46.7 vs 50.2 at 2,768,7,7; stubbed floors 4.6 vs 9.9 us'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7) split-K is required to use the SMs; at 1,192,28,28 it is
    not needed.
  status: supported
  evidence:
  - 'r1 B ablation, split-K auto vs forced 1: 2,768,7,7 46.5 vs 64.6 us, 8,768,7,7 86.1 vs 120.6 (28-29%); 1,192,28,28 29.9 vs 30.6 (none)'
- id: H6
  statement: The fp32 SIMT GEMM core runs at 46-50 TF/s on the rented card at 1965 MHz, i.e. 35-38 TF/s at the 1500 MHz
    portal lock; the anchor loses to 17-25 TF/s through the epilogue and affine. The fp32 SIMT ceiling mean (0.680) is below
    the leader (0.690), so SIMT GEMMs cannot reach the top.
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s, 57-60 TF/s without global loads; ablations no-epilogue 34-35 TF/s, no-affine 33-35'
  - 'portal #65377 vs rented: 127.2/97.5, 270.0/204.7, 1317.4/1010.5 = 1.30-1.32'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes (416 tiles = 1.4 waves at 32,768,7,7; N=96/160
    padded to 128), so per-size tile tables or stream-K matter as much as the core rate.
  status: parked
  evidence:
  - 'r0 cost model picked 128x96 tiles for C=768; r1 B: 1,160,449,7 P3 ~26 us with N=160 padded to 2x128 tiles is the one measured padding loss. Both A and B take 54 us at 1,160 (leader-implied ~33 rented); E1 measures A''s per-launch split there'
- id: H8
  statement: Applying the GRN scale per image to W2 (or scaling the hidden tile in smem while converting to fp16) removes the
    per-element affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'probe r0: noaff phase 3: 495 vs 678 us at 32,128,64,64, 1816 vs 2684 us at the largest'
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row, a 3-round chain); r3 D folds fp16(s_img*W2) per image. In the swap-AB tcgen05 design (H28) the scale is a per-K-row rewrite of the B stage or a per-column scale of the W2 A tile; E3 picks one'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager (157 us .. 11.5 ms), so even the fp32 anchor scores > 0.5 on
    the S/M band.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload; the anchor scores 0.10-0.51. Reaching S=0.5 needs t <= 68-222 us on 14 rows'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images'
    GEMM1 work at B >= 8 and removes two barrier waits.
  status: open
  evidence:
  - 'none measured yet; a self-resetting counter workspace is allowed. Worth ~+0.01-0.03 at B >= 8 rows on top of any GEMM core (pool I1/I4/I23). r4: tested in E2 together with H27 (tile-level counters) at S rows; the B >= 8 image-overlap part follows if the counter release is cheap'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches;
    what sets the S rows is the SM work and dependent rounds inside each phase (8-10 us per phase at 1,192,28,28 in B;
    6-17 us per custom launch in A). Cutting the per-phase chain to ~3-4 us gives S rows of 15-20 us rented (S ~0.78-0.82).
  status: open
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7 (P1 13.2 / P2 +7 / P3 +9.6)'
  - 'r1 A: cold dependent load 0.44 us per step, ~1 us per round once loaded; K1/K2/K3 are 2-3 round chains each'
  - 'r2 B audit at 1,192 (globaltimer): ~12 dependent steps of 1.5-2.5 us between P1 end (1.8 us median, 8.6 max on 48 conv CTAs) and the end at 28.2 us'
  - 'r3 B microbench + audit r4: barrier release 1.6-1.9 us per 296-CTA flip barrier (3000-3600 clk), cold rounds 1.2-3.5 us, just-written 32 KB bulk round 0.83 us; so the two-barrier coop floor is 11-14 us (not 20), and 15-18 us of SM work and tail waits remain at 1,192. Portal ratio on these rows is 1.16-1.26, not 1.28 (H21)'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz.
  status: supported
  evidence:
  - 'r1 probes (B and C): 2018-2048 FLOP/clk/SM = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16'
  - 'r3 A (E1) and D: real tiles reach 160 TF/s (64x128, 4-stage cp.async) and 0.64 us per 32-k chunk vs 0.40 ideal (D); cuBLASLt does the same GEMMs ~4x faster, so mma.sync loses on every MMA-heavy row (8,320; 16/32,768; 32,128; 64,96)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, hidden slices in smem, statistics and GEMM2 partials reduced through
    DSMEM) runs the 7x7/14x14 workloads in <= 30 us.
  status: refuted
  evidence:
  - 'r1 C: DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM; v1 at 2,768,7,7 129 us. Floor ~50-60 us per cluster wave at C=768'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight
    from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map; FLOAT32 maps leave bits unchanged. Candidate use (pool I2/I20): M <= 400 wide-C rows where the weight conversion (5-10 us) exceeds the TF32 MMA penalty; E3 may use it for W1 at 2,768'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the
    library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; 8 heuristic algos per shape, top-3 within 0.3% (checked at 1,192 only)'
  - 'r1 A: NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd; channels-last GEMM2 + 32x32 transpose-add (4-8 us) instead'
  - 'audit r3: supported at 1,192 (and C<=192); at C=768 the two GEMMs cost 15.6-24 us against a 2.7 us weight stream, so the library IS a first-order term on four rows. The cold cuBLASLt algo/split-K sweep (E1 r3 step 0b) was never run; still owed, low priority while D builds tcgen05'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little (<= 4 us over the
    launch floor at 20-42 MB; ~1.4x ideal at 134 MB), so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r2 C (E3): an on-chip hidden with one barrier was no faster than B''s materialised hidden at 1,192 / 2,96'
- id: H17
  statement: The anchor's phase 1 at C=768 is block-level weight transposes plus the 4-tokens-per-lane conv; warp-level
    transposes and zero-padded smem planes cut it to ~45 us.
  status: supported
  evidence:
  - 'r0 probes: 42 vs 11.5 us transposes; 306 vs 59 us conv; smem planes 35.5 us at 32,768,7,7'
- id: H18
  statement: Staging the dwconv input through TMA boxes (3-D map over x for W%4==0 rows; 1-D cp.async.bulk plane copies for
    7x7, 14x14 and 449x7 rows) in a multi-stage mbarrier ring lifts conv ingest from ~8 GB/s per SM to >= 40 GB/s per SM,
    so the conv phase lands within 1.5x of its FMA bound.
  status: open
  evidence:
  - 'r2 A (E1): ingest part SUPPORTED for W%4==0 rows: 15-34 GB/s/SM, 11x the old staging; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96. The 1.5x-of-FMA-bound part is NOT met (3x at the largest). Box inner start must be a multiple of 16 B (start at w=-4, box W+8)'
  - 'r3 D: bulk-plane copies SUPPORTED for W=7/14 (conv phase 1.5-10 us on the six wide-C rows). 449x7 untested; E1 (r4) covers it with 4-channel 50 KB groups'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be
    overlapped across the grid syncs; together they cut 1,192,28,28 from 29.7 to <= 22 us rented.
  status: refuted
  evidence:
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring); the remaining cost is barrier release plus cold rounds plus the conv tail (H11, H24)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile in shared memory across the grid sync and
    reduces GEMM2 split-K partials with red.global.add runs the seven C <= 192 rows in <= 15-20 us rented.
  status: refuted
  evidence:
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3 (kill line 25 us); conv redundancy 4C/BN (2-6x) and serial dependent rounds dominate. Lineage C retired r3'
- id: H21
  statement: The portal/rented ratio is a per-path, per-row quantity, not a constant 1.28. Lineage A's small multi-launch rows
    run at 1.16-1.22 (less clock-bound), its other rows at 1.23-1.32; lineage B paths at 1.24-1.26; lineage D paths at about 1.32.
    Project every path with its own ratio.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented on 8 rows: 1.27-1.28 each; fp32 anchor 1.30-1.32'
  - 'portal #65878 (A r2, 16 rows): 1.16 at 1,96, 1.20 at 1,192, 1.22 at 2,96 and 2,768, 1.23-1.32 elsewhere; bands 49.7/40.7 = 1.22 S, 83.9/66.9 = 1.25 M, 199.1/157.9 = 1.26 L. #65879 (merge): B paths 1.24-1.26 (1,192 37.3/29.7), D paths ~1.32 (2,768 45.1/34.1; 8,384 70.7/53.5). The notebook agrees: S bench -8.5% became portal -4.0% (audit r4)'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol (2.4e-3) on the 77M-element rows;
    a hi/lo fp16 split of one operand (2 MMAs) restores fp32-class accuracy at no cost where MMA work is <= 2 us (all S/M rows),
    so per-row error logging decides where to spend it.
  status: open
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <= 1e-7 fraction beyond 2.4e-3 with randomized LN affine; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r2 A: fp64 reference, 1-3 rounds per row: 0 beyond 2.4e-3 + 1e-5|ref| on all 16 rows, max 2.39e-3 at 32,128. r2 B: 1 element beyond at 4,96 and 8,192 (randomized LN, seed 1). r3 A E1 max 2.35e-3 (1 round); r3 D max 1.52-1.84e-3 (1 seed)'
  - 'audit r4: the 10-seeded-round per-row log against each row''s own atol has been skipped in four sessions; it is a hard deliverable of E1, E2 and E3 for the paths each lineage ships, which together cover the merge'
- id: H23
  statement: tcgen05 kind::f16 (TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the four
    GEMM-heavy rows and is the lever for the L band once the S/M rows are near the fused floor.
  status: parked
  evidence:
  - 'arithmetic: mma.sync ideal 29/33/76/260 us vs tcgen05 7/8/19/65 us at those rows'
  - 'audit r3: with the score formula the ideal savings give +0.010 of mean, ~+0.005 against cuBLASLt. Audit r4: tcgen05 is the enabler for fused epilogues (H26 refuted on engine grounds), so it is built inside lineage D as H28; the top-row epilogue fusion (K2 removal) waits for H29 and a working tcgen05 GEMM1'
- id: H24
  statement: The 2-4 us cost of a 12-32 KB per-CTA load round right after a grid sync is an issue-rate artefact of 4-byte copies,
    integer div/mod and serialised waits, not post-barrier L2 latency; one cp.async.bulk per operand brings the round to <= 1.2 us,
    which cuts 1,192 to <= 22 us rented.
  status: refuted
  evidence:
  - 'r3 B microbench (296 CTAs, after the flush): just-written 32 KB round cp4+div 2.4 us / cp16 1.36 / ldg128 0.95 / one bulk 0.83 us median; cold DRAM source 2.7-3.5 us whatever the method; barrier release 1.5-2.0 us after the last arrival. Issue rate is worth ~0.5 us per round'
  - 'r3 B run_tests with a same-build control: bulk rows 1,192 31.4 vs ring 29.5 (+1.9); 1,160 / 4,96 within noise. The 2-4 us post-sync steps are barrier release plus cold or dependent rounds'
- id: H25
  statement: A weight-stationary cooperative kernel for the six wide-C small-image rows (CTA owns a hidden slice and whole-image
    token tiles, fp32 W slices ingested once through bulk rings, CTA-local GRN statistics, bulk-plane conv, split-K red.add GEMM2)
    runs 2,768 <= 35 us, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 rented.
  status: refuted
  evidence:
  - 'r3 D stub with real bytes (2 syncs, plane conv, 1.4 MB/CTA W ring at ~135 GB/s/SM, coalesced red): 16-29 us on the six rows, so ingest and post-sync rounds are NOT the binding term'
  - 'r3 D v3 run_tests: 2,768 34.1 (met); 8,768 63.9; 16,768 93.6; 32,768 169.6; 8,384 53.5; 16,384 93.7 (all missed). Binding term: mma.sync 0.64 us per 32-k chunk (ideal 0.40) plus 0.26 us per chunk of bare wait+sync at 16 warps/SM, 96 chunks at 32,768. Fully resident fp32 W fits only at M <= ~400. The skeleton, not the thesis''s ingest, failed: engine swap in H28'
- id: H26
  statement: Replacing cuBLASLt GEMM1 + K2 and K3 + cuBLASLt GEMM2 + K4 in lineage A with two custom mma.sync FP16 GEMMs whose
    epilogues carry the LN fold, GELU, sumsq (GEMM1) and the s_b reduction, A-scale, residual and NCHW store (GEMM2) cuts the
    pipeline to 3 launches and the S/M rows by 10-14 us rented.
  status: refuted
  evidence:
  - 'r3 A (E1): 1,192 36.3 us (target <= 26; library 34.3 in-container), 2,768 60.1 (<= 48), 8,384 97.5 (<= 60); GEMM1f alone 12-14 us cold at 1,192 vs cuBLASLt GEMM1 + K2 5.1 + 9.8; at 8,320 the custom GEMMs run at 160 TF/s vs cuBLASLt ~4x faster'
  - 'audit r4 (overstated claim corrected): at 1,192 the MMA share of GEMM1f is 3-5 us, so a tcgen05 engine would save 2-4 us per S row; the rest is 2-3 dependent rounds per launch (H11). The engine matters on M-heavy rows (8,320; 16/32,768), where it is 4x'
- id: H27
  statement: In the cooperative kernel, replacing grid sync 1 with per-token-tile ready counters (conv CTAs red.release on the
    tiles they complete; GEMM1 tiles poll with ld.acquire for their own channel groups) and sync 2 with per-image counters, plus
    spreading the conv over all CTAs, removes the two barrier releases (1.6-1.9 us each) and the conv tail wait (1,192 P1 median
    1.8 us vs max 8.6), cutting 1,192 to <= 24 us rented and 2,96 / 4,192 / 1,160 / 4,96 / 8,192 by 4-8 us each (about +0.01-0.015
    of mean); for B >= 2 the GEMM2 tiles of early images overlap GEMM1 of later ones (H10).
  status: open
  evidence:
  - 'for: r2 B stamps at 1,192: sync1 at 9.9 us with P1 median 1.8 us (conv on 48 CTAs); r3 microbench: barrier release 1.5-2.0 us, just-written 32 KB round 0.83 us; a monotone counter with a generation number is a legal self-resetting workspace'
  - 'against: the counter poll after red.release may cost as much as a barrier release (the critic''s test in E2 measures release-to-observe latency); at B=1 the GEMM2 phase still waits for the last GEMM1 tile. Built in E2 (r4)'
- id: H28
  statement: A tcgen05 kind::f16 engine (TMA 128B-swizzled ring, warp-uniform TMA/MMA/epilogue roles, TMEM accumulators) in the
    swap-AB orientation (A = weight tile of 128 hidden or output channels, B = activations [channel][token] MN-major with tokens
    of one image or image group as N) inside D's cooperative skeleton makes the six wide-C rows MMA- and byte-bound: thread-local
    LN fold and GRN sumsq per (image, k) with no atomics, hidden stored [k][token] fp16, GRN scale applied per K-row to the GEMM2
    B stage (or per column to the W2 A tile) by a transform warp, output tile [c][tokens] = NCHW for HW=49/196 with the residual
    in the epilogue. Targets rented: 2,768 <= 32, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60
    (about +0.03-0.04 of mean at the per-path ratio).
  status: open
  evidence:
  - 'arithmetic: kind::f16 ideal 0.5/2/4/8 us at 2/8/16/32,768 and 2/4 us at 8/16,384; D stub with real bytes 16-29 us; the per-stage floor (~565 clk) requires 128-k stages at N <= 256 (2N clk per 64-k); #35 tc_ws ring ran at 770 clk per stage vs 768 ideal with uniform roles'
  - 'leader stories: wide-C times 35-86 us portal grow linearly in B at ~270 TF/s marginal, consistent with a fused tensor-core pipeline without K2/K3/K4. Gate and build in E3 (r4)'
- id: H29
  statement: Rewriting lineage A's K2 (GELU + GRN stats over the fp16 hidden) as a stream-rate kernel (long token runs per CTA,
    8 uint4 loads in flight per thread, register-resident sumsq, one atomic per 8 channels per CTA, exact erf, evict_last hidden
    stores) cuts 447 -> ~200 us at 64,96, 113 -> ~50 at 32,128 and 38 -> ~15 at 8,320 (about +0.01 of mean), and a tcgen05 GEMM1
    with the same epilogue (H28 engine) would remove K2 entirely.
  status: open
  evidence:
  - 'r1 A per-launch: K2 447 us at 64,96 (616 MB write + read, ideal ~175), 113 at 32,128 (ideal ~40), 37.8 at 8,320; 196-thread blocks with 4 loads in flight and smem reductions. Audit r4: S is absolute per row, 64,96 0.748 -> 0.834 and 32,128 0.720 -> 0.811 if K2 vanishes (+0.011). Queued for A in r5 (A has one slot in r4: H2 conv port); the critic''s probe (K2 vs a read+write copy of the same bytes) runs first'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "A",
    "hypothesis": "H2",
    "title": "Bulk-plane conv and coalesced weight pass in K1 for the W=7/14/449x7 rows; interleaved 8,768 path check; 10-round precision log",
    "operation": "structural_mutation",
    "parents": ["r2-lib-cublaslt-fp16-tmaconv"],
    "band": "all",
    "instructions": "Step 0: build the parent and the r0 anchor in your container and time them interleaved x3 (run_tests, not a ctypes harness); also time r3-a-e1-mma-fused-3launch's 8,768 path against the parent's library path at 8,768 interleaved x3 and report which is faster (the merge chose the custom path from a 62.4-vs-63.1 difference and reads 83.8 vs 80.1 on the portal). Step 1: replace k1_dw's 4-byte cp.async staging for the rows with W=7, W=14 and 449x7 (2/8/16/32,768,7,7; 8/16,384,14,14; 1,160,449,7) by a bulk-plane conv, re-implemented from the lineage D conclusion: one 1-D cp.async.bulk per group of channel planes (32 planes of 196 B = 6272 B for 7x7; whole 784 B planes for 14x14; 4-channel 50 KB groups for 449x7; every group is 16-B aligned and a multiple of 16 B) into zero-padded smem, one thread per 7-output row segment with the 49 taps in registers, LN partials (sum, sumsq) per token reduced in smem, d written channels-last fp16 as the parent's GEMM1 expects. Issue all plane copies for a CTA's work up front (mbarrier expect_tx), issue every tap load before the first FMA. Step 2: replace the role-CTA weight prep by a coalesced fp32->fp16 conversion of W1'=W1*lnw (and u, c1, b2') and W2 spread over ALL K1 CTAs as a first round (16-B loads and stores, 2+ CTAs/SM), so the 19 MB read + 9.5 MB write at C=768 (about 4-6 us) overlaps the conv CTAs; keep the TT layout cuBLASLt already accepts. Step 3: per-launch harness_time of K1 at 2,768, 8,768, 32,768, 16,384 and 1,160 old vs new in one container x3; record the conv-only and weight-only ablations. Step 4: run_tests all 16 rows with the parent as same-build control; for every row log over the harness's 10 seeded rounds (seed 200) against an fp64 reference the max |err| and the count of elements beyond that row's OWN atol + 1e-5|ref|; any row with a beyond-atol element gets the hi/lo fp16 split of the cheap operand before shipping. Step 5 (only if time remains, separate kernel, same hypothesis family): run the critic's probe for H29: K2 alone at 64,96 and 32,128 vs a plain read+write copy of the same bytes after the flush. Report per-row rented times with the per-path portal ratio (1.16-1.22 small rows, 1.23-1.32 others) and the projected score.",
    "success": "K1 drops by >= 8 us rented at 8,768 and >= 12 us at 32,768 in the interleaved comparison, with run_tests 16/16 and the 10-round precision log showing 0 elements beyond each row's own atol on every shipped path; total per-row gains of 8-20 us on at least five of the seven W=7/14/449x7 rows (projected >= +0.010 of mean at the per-path ratio).",
    "refuted_if": "K1 at 32,768 loses less than 6 us rented, or the coalesced weight pass alone exceeds 8 us at C=768 and cannot be overlapped with the conv CTAs (then the weight conversion, not the conv staging, bounds K1 and the TF32-map route H14 is the follow-up).",
    "model": "sonnet"
  },
  {
    "id": "E2",
    "lineage": "B",
    "hypothesis": "H27",
    "title": "Counter-released dataflow replaces the two grid barriers in the cooperative kernel (S rows): microbench first, then build",
    "operation": "structural_mutation",
    "parents": ["r2-coop-flat-s"],
    "band": "S",
    "instructions": "Step 0: parent and r0 anchor interleaved x3 in your container (run_tests). Step 1 microbench (the critic's test plus I1's, 296 CTAs x 256 threads, 2xL2 flush before each launch, globaltimer): (a) release-to-observe latency when one CTA does red.release.gpu on a flag and 295 CTAs poll it with ld.acquire (one poller thread per CTA, no nanosleep; also a variant with ld.relaxed polling plus fence.acq_rel after the flip); (b) the same with 8 flags spread over separate 128-B lines, each polled by 37 CTAs; (c) the existing flip barrier (3000-3600 clk) and a two-level barrier (per-die or 8-line counter tree, final arriver writes per-group release flags) for reference. Report median and max clk per variant. Kill criterion for the dataflow build: if the best flag release-to-observe exceeds 1.2 us (about 2300 clk at 1.9 GHz) there is nothing to gain and you stop after shipping the best barrier variant into the parent instead. Step 2 (if the microbench passes): in the parent's ring path, (i) spread the conv units over all CTAs (at 1,192 it runs on 48 of 296, P1 median 1.8 us vs max 8.6); (ii) replace grid sync 1 by per-token-tile ready counters in a torch.empty int32 workspace: each conv CTA, after its d16 and LN-partial stores and a __threadfence/fence.proxy.async as needed, does one red.release.gpu.add per token tile it completed; each GEMM1 tile polls (one thread, ld.acquire, then __syncthreads) until its tile's count reaches the number of channel groups for this shape, then proceeds; (iii) replace grid sync 2 by per-image counters that GEMM1 tiles increment after their hidden and sumsq stores; GEMM2 tiles of image b poll for image b's GEMM1 tile count, so for B >= 2 images overlap; (iv) counters are monotone within a call and reset by the last GEMM2 tile of each image (or carry a per-call generation computed on the host from a module-level call counter is NOT allowed: use in-kernel reset by the last arriver, verified correct for every call and every order). Keep the cooperative launch (grid <= resident capacity, checked with cudaOccupancyMaxActiveBlocksPerMultiprocessor) so every producer is resident; order each CTA's static work list conv tiles -> GEMM1 tiles (image-major) -> GEMM2 tiles (image-major) so no CTA waits on work scheduled later in another CTA's list. Step 3: globaltimer stamps at 1,192 and 2,96 (P1 end median/max, first GEMM1 tile start, first GEMM2 tile start, end) old vs new. Step 4: run_tests all 16 rows with the parent as same-build control, dispatch per row only where the new path wins by > 1.5 us; log the 10-seeded-round per-row max |err| and beyond-own-atol counts for every B path (seed 200, fp64 reference). Report per-row times with the B ratio 1.24-1.26 and the projected score.",
    "success": "1,192 <= 24 us rented and at least four of {1,96; 1,160; 2,96; 4,192; 4,96; 8,192} improve by >= 4 us against the same-build parent, 16/16 pass, 0 beyond-own-atol elements over the 10 seeded rounds on every shipped B path (projected >= +0.008 of mean).",
    "refuted_if": "The best flag release-to-observe latency in the microbench exceeds 1.2 us, or the built dataflow kernel beats the parent by less than 2 us at 1,192 and 2,96 with the conv spread over all CTAs; then the cooperative S-row thesis is retired in r5 and its rows go to A or an E-lineage fused design.",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "D",
    "hypothesis": "H28",
    "title": "tcgen05 kind::f16 swap-AB engine inside D's cooperative wide-C kernel, gated by a GEMM probe",
    "operation": "structural_mutation",
    "parents": ["r3-ws-coop-widec"],
    "band": "L",
    "instructions": "Step 0: parent (v3) and r0 anchor interleaved x3 in your container. Gate probe (<= 1 hour): take the #35 tc_ws tcgen05 kind::f16 ring kernel from the archive (get_kernel; another problem's kernel, allowed as a structure reference and flagged to the operator) and run it unmodified on pre-converted fp16 operands after the 2xL2 flush for 1568x3072x768, 1568x768x3072 and 392x3072x768; also time one 128x208x768 single-CTA tile with an MN-major (token-contiguous) fp16 B operand from a [k][token] buffer against a K-major reference and check the accumulators are bit-identical. GO if each full GEMM is <= 6 us rented at the 1568 shapes and <= 3 us at 392 and the MN-major descriptor validates; KILL if any exceeds 10 us (then report and spend the session on the fallback: KC=64 chunks and a double-buffered fp16 B in the mma.sync ring to halve the per-chunk __syncthreads). Build (GO): keep D's skeleton (phase A bulk-plane conv writing d as fp16 [c][token] per image plus per-token LN partials, W conversion to fp16 L2 scratch by a first round over all CTAs or a TFLOAT32 map for W1 at 2,768 where MMA is 0.5 us, sync 1, GEMM1, sync 2, GEMM2 with coalesced red.add.v4 into out prefilled with x + W2*gb + b2 where KS > 1). Swap engine: cta_group::1 first (M = 128 weight rows), A = W tile K-major from fp16 scratch through a 128B-swizzled TMA ring, B = activations [channel][token] MN-major with N = tokens of one image (49 -> 64 padded by TMA OOB fill) or an image group (4 x 49 = 196 -> 208/256; one 14x14 image = 196), K = C then 4C, 128-k per stage (2 k-blocks) so each stage carries >= 600 clk of MMA; whole-warp TMA producer and MMA warp with elect.sync, full/empty mbarrier pairs, 4-6 stages, TMEM allocated once per CTA. GEMM1 epilogue: tcgen05.ld, thread owns hidden channel k and 32 token columns per load; LN fold z = rstd[t]*(acc - mu[t]*u[k]) + v[k] with mu/rstd from smem, exact erff GELU, fp16 hidden to L2 as [k][token] via 16-B stores, per-(image,k) sumsq as in-thread sums (no atomics within a tile; one red.add per (image,k) only if a tile spans a partial image). After sync 2: s_b[k] = 1 + gw[k]*gf/mean computed once per image by the CTA that needs it with all gf loads issued first; GEMM2 A = W2 tile [128 c][K=4C], B = hidden [4C][tokens] MN-major, the GRN scale applied per K-row of the B stage by a 4-warp transform group after the TMA lands (fence.proxy.async before the MMA; the transform warps must have no global loads in flight when they arrive), or alternatively per column of the W2 A stage; pick by a per-stage clk measurement. Epilogue: acc [c][tokens] + x (16-B coalesced along hw) stored directly in NCHW, or red.add.v4 for split-K. Tile counts per row as multiples of 148 (or 74 pairs if you move to cta_group::2 later). Measure clk per stage in the MMA warp against the 2N-clk-per-64-k ideal and fix any ELECT/BRA.U.ANY waterfall in SASS. run_tests all 16 rows (other rows keep the parent's dispatch), 10-seeded-round per-row max |err| and beyond-own-atol log (fp64 reference) for every D path. Report per-row rented times with the D ratio ~1.32 and the projected score.",
    "success": "Gate passed and the built kernel reaches at least four of: 2,768 <= 32, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 us rented (same-build control), 16/16 pass, 0 beyond-own-atol elements over 10 seeded rounds (projected >= +0.02 of mean).",
    "refuted_if": "The gate GEMMs exceed 10 us at the 1568 shapes, or the built kernel beats D v3 by less than 25% at 16,768 and 32,768, or its per-stage clk stays above 1.5x the MMA ideal after the waterfall fix; then wide-C rows move to a library-with-fused-epilogue route (CUTLASS EVT, pool I6/I8/I16) in r5.",
    "model": "opus"
  }
]
```