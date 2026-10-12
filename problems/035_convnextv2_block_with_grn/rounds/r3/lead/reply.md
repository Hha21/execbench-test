### Assessment

Where the score is (leader 0.690 vs A r2 at 0.6135 projected, per-row S from the Tb/Ts table at rented x1.28): the six wide-C small-image rows (8/16,384,14,14 and 2/8/16/32,768,7,7, per-row S 0.48-0.60 vs 0.69) hold 0.056 of mean; the seven C<=192 S/M rows hold 0.023; 8,320 and 32,768 hold 0.008; on the two largest rows we lead by 0.03-0.05. The round therefore goes to the wide-C rows and the S-row fixed cost, not to tcgen05.

Audit answers. (1) The 1.28 ratio is measured only on the 8 rows #65570 ran, none with W=7/14: H21 is rescoped to open and the operator should submit r2-lib-cublaslt-fp16-tmaconv (16 rows) this round. (2) The A>B>C ranking is cross-session: every experiment now times its parent and the r0 anchor interleaved in-session, the operator should run A r2 and B r2 interleaved once, and the notebook's v3fix entry is flagged as a copy of v3's. (3) The 4-8 B/clk/SM post-sync round is unsupported: E2 runs the critic's microbench first (new H24) and B's rebuild branches on it. (4) cuBLASLt at C=768 is a first-order term (15.6-24 us vs a 2.7 us weight stream): H15 rescoped; E1 sweeps algos and split-K, lineage D removes the library on those rows. (5) The FP16 margin is a max-error question over 1e7-1e8 elements, not rms: the notes now require per-row max error against the row's own atol over the harness's 10 seeded rounds; H22 decides the hi/lo split per row. (6) tcgen05's ideal savings at the four GEMM-heavy rows are worth +0.010 of mean at most (+0.005 against cuBLASLt), so H23 stays parked with the corrected number.

Portfolio. C is retired: H13 and H20 refuted, 0.4175, its rows tie B; its probe results live on as conclusions. D starts in its place: a weight-stationary cooperative kernel for the wide-C rows, fp32 weights ingested once per call through TMA/bulk rings (no conversion pass), whole-image token tiles so GRN statistics stay in the CTA, bulk-plane conv. A (one experiment, the leader's cap) fuses K2/K3/K4 into custom mma.sync GEMM epilogues: a 3-launch pipeline. B tests whether post-sync rounds are an issue-rate artefact and, if so, rebuilds them as bulk copies with everything prefetched at entry.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (TMA-staged dwconv+LN, tensor-core GEMMs with every elementwise step fused into their epilogues, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and the carrier of conv/epilogue techniques.
    status: active
    started: r1
    best: r2-lib-cublaslt-fp16-tmaconv
    conclusions:
      - "Each custom launch costs 6-17 us at S (2-3 dependent rounds plus ramp), so the 5-6 launch pipeline floors at 35-40 us; the lever is fusing K2/K3/K4 into GEMM epilogues (3 launches), not faster library calls at C<=192."
      - "TMA 3-D box staging (box starts at w=-4, inner coordinate a multiple of 16 B or the copy faults) lifted conv ingest 11x: K1 5.6 us at 1,192, 315 us at the largest (still 3x the FMA bound, LDS/sync-bound at ~40% FMA busy). W=7/14 rows still use the 8 GB/s/SM staging."
      - "cuBLASLt FP16 is 1.4-2.1x faster than TF32 at every shape, but at C=768 GEMM1+GEMM2 cost 15.6-24 us cold against a 2.7 us weight stream: the library is a first-order term on the four C=768 rows (audit r3), not at 1,192."
      - "H16: a materialised fp16 hidden is cheap under the dirty-L2 flush below 42 MB (3.6 us over the floor at 20 MB) and ~1.4x ideal at 134 MB."
      - "Precision: single-pass FP16 max |err| 2.37-2.39e-3 at 32,128 / 8,768 / 32,768 against atol 2.4e-3 (0 beyond on 1-3 rounds); the hi/lo split of the cheap operand is the fallback per row."
  - id: B
    thesis: One cooperative launch with every phase fused, FP16 mma.sync, fp16 hidden materialised in L2 between grid syncs; the fixed-cost specialist aiming at S rows <= 20 us rented by making every post-sync operand one bulk copy and prefetching everything phase-independent at entry.
    status: active
    started: r1
    best: r2-coop-flat-s
    conclusions:
      - "Coop beats three plain launches by 3-7 us at S (F 4.6 vs 9.9 us); mma.sync f16 = 0.455 PF/s, enough everywhere except 8,320 / 32,768 / 32,128 / 64,96."
      - "Flattening every phase to one load round did not move 1,192 (30.5 vs 30.4): ~12 dependent steps of 1.5-2.5 us remain, and a 12-32 KB cp.async round right after a grid sync costs 2-4 us. Whether that is issue rate (4-B copies, div/mod) or post-barrier L2 latency is NOT isolated (H24, E2)."
      - "Coalesced conv CTA unit (32 ch x 8 four-column positions, lanes along w) is 2x faster than lane-per-channel patches; 8-channel tiles matter at S (tile count, not FMA rate)."
      - "Pitfalls: runtime guards or '#pragma unroll 2' around mma.sync cost ~470 clk per k-block (per-shape templates, all fragment LDS first, shfl-uniform warp index); a register-direct 48-erff epilogue took 16 us per tile, an smem-staged 8-column rolled loop with 16-B stores fixed it; a release arrive with loads in flight stalls."
      - "Weight pass costs ~10 us of P1 at C=768 and 2-3 us at C<=192; split-K is worth 28% at C=768 and nothing at 1,192."
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output); clusters/DSMEM in r1, bulk rings in r2.
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - "Retired r3: H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) both refuted; 0.4175 projected, ties B on its best rows and has no plan for 8,192 / C>=320 (smem)."
      - "Keep: one-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM (590 KB in 5.0 us, ~16 TB/s aggregate from L2); converting inside the ring is 2-2.5x slower, so keep slices fp32-resident and cvt.rn.f16x2 at fragment load."
      - "Keep: TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs (1.0 TB/s from 32)."
      - "Keep: cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry (37.2 -> 30.8 us at 1,192). Zero-padded plane fills need all loads issued before the stores."
      - "Conv redundancy 4C/BN (2-6x) and per-CTA serial rounds dominated, not bytes; the on-chip hidden gives no edge over a materialised fp16 hidden."
  - id: D
    thesis: Weight-stationary cooperative kernel for the six wide-C small-image rows (C>=384, HW<=196) where each CTA owns a hidden slice and whole-image token tiles, ingests its fp32 W1/W2 slices once per call through TMA/bulk rings with no conversion pass, keeps GRN statistics CTA-local, and reduces GEMM2 through red.add or a materialised scaled hidden chosen per row.
    status: active
    started: r3
    best: null
    conclusions:
      - "Seeded from pool families I22/I25 (+I26 bulk-plane conv, I15 entry prefetch): the six target rows hold 0.056 of mean against the leader; the leader's implied rented times are 27/34/46/67 us at 2/8/16/32,768 and 37/52 at 8/16,384."
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r3 2026-10-12.
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
- 'Where the score is (r3, per-row S from the Tb/Ts table at rented x1.28, A r2 = 0.6135 vs leader 0.690): the six wide-C
  small-image rows (8/16,384,14,14; 2/8/16/32,768,7,7; S 0.48-0.60) hold 0.056 of mean; the seven C<=192 S/M rows hold 0.023;
  8,320 + 32,768 hold 0.008; we lead by 0.03-0.05 on 32,128 and 64,96. Leader-implied rented times: S rows 24-37 us;
  2/8/16/32,768 27/34/46/67 us; 8/16,384 37/52 us.'
- 'Portal/rented ratio (audit r3, rescoped): 1.27-1.28 on the 8 rows of #65570 (lineage A v3: 43.8/34.5, 50.0/39.1, 64.1/50.0,
  68.1/53.6, 72.5/56.5, 83.7/65.5, 569/445, 2286/1790), 1.30-1.32 for the fp32 anchor. No W=7/14/449x7 row has a portal
  measurement: their 19 MB weight stream is DRAM work and ~55% of small-row K1 is ns-bound latency, so use 1.15-1.28 as a
  range there until r2-lib-cublaslt-fp16-tmaconv runs all 16 rows on the portal (operator action, r3). Normalise every
  rented run by its in-session anchor.'
- 'Precision logging (audit r2, tightened r3): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the rms.
  Measured single-pass FP16 max |err| 2.37-2.39e-3 at 32,128 / 8,768 / 32,768 (A r2, 1-3 rounds) and 1.9-2.8e-3 (B, randomized
  LN) against the smallest atol 2.4e-3; atol varies 2.4e-3..3.8e-3 per row (workload.jsonl). Every run_tests from r3 on records
  per workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded
  rounds (seed 200), against an fp64 reference; a row with any beyond-atol element switches the cheap operand to a hi/lo fp16
  split (2 MMAs, free at S/M where MMA work is <= 2 us). Passing through the 98% ratio is not acceptable.'
- 'Cross-session rented timings drift by ~6 us per call (v3fix 50.4/72.6/192.3 vs v3 44.4/66.4/182.8 from a host-only
  change; the notebook''s v3fix run_tests entry is a verbatim copy of v3''s numbers and must not be cited as a v3fix measurement).
  The A 0.611 > B 0.564 > C 0.418 ranking is cross-session (audit r3): every experiment times its parent and the r0 anchor
  interleaved x3 in its own container, and the operator runs anchor + A r2 + B r2 interleaved once before dispatch decisions.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65;
  8,192 0.63; 32,128 0.62; 64,96 0.65. predict_score at the 1.28 ratio: A r2 0.6135, B r2 0.566; fused targets S 23 / M 46 /
  L 105 us portal -> 0.76; 20 us + bytes/1 TB/s on every row -> 0.80; 30 us + bytes/1 TB/s -> 0.75.'
- 'Audit r3 on tcgen05 (H23): with the score formula, the ideal GEMM savings (22/25/57/195 us portal at 8,320 / 32,768 / 32,128 /
  64,96) are worth +0.047/+0.057/+0.034/+0.027 per row = +0.010 of mean at most, and about +0.005 against cuBLASLt, which
  already runs those rows on tensor cores. tcgen05 is an engine swap for later, not a lineage thesis.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload
    memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still
    under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: 3.3/5.4 us at 784x192x768, 7.2/11.8 at 98x3072x768, 129/270 at the largest, so FP16 is 1.4-2.1x faster everywhere (audit r2 retracts "the 2x rate buys no score"; the leader''s 174 TF/s is a whole-call average, not a GEMM rate). TF32 only where the weight conversion cannot be overlapped (candidate: 2,768,7,7)'
  - 'margin caveat (audit r3): the "5 sigma" is an rms statement; the max error over 1-3 rounds already sits at 2.37-2.39e-3 against atol 2.4e-3 on three rows. See the precision-logging note and H22'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at every size because the implementations are staging-bound,
    not FMA-bound (K1 954 us in A and P1 446 us in B at the largest against a 134 us FMA bound; 234 us at 32,128 against 29;
    8-13 us of the S rows); a conv at ~1.5x its FMA bound is worth ~+0.015 at the top two rows and ~-5 us per S row.
  status: open
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); a 64-channel 7-row stage-only kernel took 9.5 us vs 5.9 for a contiguous float4 read vs 1.9 floor (~8 GB/s per SM); plain LDG+STS equals cp.async'
  - 'r1 B: cp.async-staged bands were slower than direct __ldg patches (41 vs 29 us at 1,192); 8-channel tiles cut P1 25 -> 9 us at 1,192 (tile count matters at S)'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 us at 1,192, 23 at 8,320, 81 at 32,128, 315 at the largest (3x the FMA bound: LDS about equal to FMA, per-stage __syncthreads, ~40% FMA busy). W=7/14/449x7 rows still on the 8 GB/s/SM staging (K1 18.6-33.5 us at C=768): bulk-plane copies (I26) tested in lineage D'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and
    0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32'
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal (57 vs 38 us) at 134 MB, so materialising is cheap below 42 MB; at the largest row K2 alone took 447 us (616 MB write + read), so recompute still wins on paper there'
  - 'parked r3: the top two rows already lead the leader by 0.03-0.05; revisit when the wide-C and S rows are near the fused floor'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card
    and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192,28,28 and 46.7 vs 50.2 at 2,768,7,7; stubbed floors 4.6 vs 9.9 us; only the largest row is 2.5% the other way (1793 vs 1749). The coop kernel wins 3-7 us per S/M row; the r0 6717-vs-6263 comparison was a register/occupancy artefact'
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
  - 'r0 cost model picked 128x96 tiles for C=768; nothing isolates quantisation. r1 B: 1,160,449,7 P3 ~26 us with N=160 padded to 2x128 tiles is the one measured padding loss. Revisit with a tensor-core tile table'
- id: H8
  statement: Applying the GRN scale per image to W2 (or scaling the hidden tile in smem while converting to fp16) removes the
    per-element affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'probe r0: noaff phase 3: 495 vs 678 us at 32,128,64,64, 1816 vs 2684 us at the largest'
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row, a 3-round chain: sumsq, sqrt-reduce, W2 rows); neither isolates the affine cost. r3: E1 moves the s_b reduction into the GEMM2 prologue and the scale to fragment load; E3 keeps gf CTA-local'
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
  - 'none yet; a self-resetting counter workspace is allowed. Worth ~+0.015-0.03 at B >= 8 rows on top of any GEMM core (pool I9/I14/I16). r3: the fallback branch of E2 if H24 shows the post-sync round is barrier latency rather than issue rate'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches;
    what sets the S rows is the SM work and dependent rounds inside each phase (8-10 us per phase at 1,192,28,28 in B;
    6-17 us per custom launch in A), all scaling 1.28x with the clock lock. Cutting the per-phase chain to ~3-4 us gives
    S rows of 15-20 us rented (S ~0.78-0.82); 30 us rented gives ~0.65 (lineage A today).
  status: open
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7 (P1 13.2 / P2 +7 / P3 +9.6)'
  - 'r1 A: cold dependent load 0.44 us per step, ~1 us per round once loaded; K1/K2/K3 are 2-3 round chains each'
  - 'r2 B audit at 1,192 (globaltimer): ~12 dependent steps of 1.5-2.5 us between P1 end (1.8 us median, 8.6 max) and the end at 28.2 us; r2 A: small-row K1 is ~55% first-stage latency (ns) and ~40% cycles'
  - 'predict_score r2: 20 us + bytes/1 TB/s on every row -> 0.80; fused targets S 23 / M 46 / L 105 us portal -> 0.76'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz.
  status: supported
  evidence:
  - 'r1 probes (B and C): 2018-2048 FLOP/clk/SM with 4-16 warps and 8 independent chains = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16; per-CTA clock64 with 2 CTAs/SM is bogus (not co-resident)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, hidden slices in smem, statistics and GEMM2 partials reduced through
    DSMEM) runs the 7x7/14x14 workloads in <= 30 us.
  status: refuted
  evidence:
  - 'r1 C: 16-CTA clusters schedule 7 concurrent at 200 KB smem; DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM (8-12k clk per 128 KB); register-streamed 590 KB W1 slice at 37 GB/s per SM = 15-20 us per CTA; v1 at 2,768,7,7 129 us (conv 20, GEMM1 52, GEMM2 + reduce 42). Floor ~50-60 us per cluster wave at C=768'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight
    from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map (ties to even checked both ways); FLOAT32 maps leave bits unchanged'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the
    library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; 8 heuristic algos per shape, top-3 within 0.3% (checked at 1,192 only)'
  - 'r1 A: NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd; channels-last GEMM2 + 32x32 transpose-add (4-8 us) instead'
  - 'r1 A: the 5-launch pipeline lands at 34.5-58.9 us at S/M rows, not 25-35, because K1/K2/K3 cost 6-17 us each'
  - 'audit r3 rescopes: supported at 1,192 (and C<=192); at C=768 the two GEMMs cost 15.6 us (2,768) to 24 us (32,768) against a 2.7 us weight stream plus 0.5-8 us of MMA, i.e. the library IS a first-order term on four rows (0.03-0.05 of mean). E1 sweeps all heuristic algos incl. split-K cold at 98/392/784/1568 x 3072 x 768; lineage D replaces the library there'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little (<= 4 us over the
    launch floor at 20-42 MB; ~1.4x ideal at 134 MB), so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r2 C (E3): an on-chip hidden with one barrier was no faster than B''s materialised hidden at 1,192 / 2,96 (30.8 / 46.3 vs 30.9 / 42.2)'
- id: H17
  statement: The anchor's phase 1 at C=768 is block-level weight transposes plus the 4-tokens-per-lane conv; warp-level
    transposes and zero-padded smem planes cut it to ~45 us.
  status: supported
  evidence:
  - 'r0 probes: 42 vs 11.5 us transposes; 306 vs 59 us conv; smem planes 35.5 us at 32,768,7,7. r1 B uses planes with 32-ch tiles; P1 at 32,768,7,7 = 48 us including F and the 10 us weight pass'
- id: H18
  statement: Staging the dwconv input through TMA boxes (3-D map over x as [B*C][H][W] with out-of-bounds zero fill for
    W%4==0 rows; 1-D cp.async.bulk plane copies for 7x7, 14x14 and 449x7 rows) in a 4-stage mbarrier ring at 2 CTAs/SM
    lifts conv ingest from ~8 GB/s per SM to >= 40 GB/s per SM, so the conv phase lands within 1.5x of its FMA bound
    (<= 250 us at the largest, <= 55 us at 32,128, <= 4 us at S rows).
  status: open
  evidence:
  - 'r1 A: staging removal cut K1 3x; stage-only 9.5 us vs 1.9 us floor for a 64-channel 7-row band. TMA row strides must be 16 B multiples, so W=7 and W=14 rows cannot use a row-boxed map (plane bytes 196/784 B: 4-channel groups of 784 B and single 14x14 planes are legal 1-D bulk copies)'
  - 'r2 A (E1): ingest part SUPPORTED for W%4==0 rows: stage-only ring 15 GB/s/SM at 8,320, 20 at 32,128, 34 at the largest (2.67 TB/s on x), 11x the old staging; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96. The 1.5x-of-FMA-bound part is NOT met (3x at the largest: LDS and per-stage sync bound). Box inner start must be a multiple of 16 B (start at w=-4, box W+8). Bulk-plane copies for W=7/14 untested (D, E3)'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be
    overlapped across the grid syncs (phase-2 operands issued before sync 1, W2 K-slices and the x residual before sync 2,
    s_b published by a per-image last arriver) and the C<=384 weight pass can move into the consuming CTAs; together they
    cut 1,192,28,28 from 29.7 to <= 22 us rented and the other S rows by 8-12 us.
  status: refuted
  evidence:
  - 'r1 B per-phase: P1 13.2 / P2 +7 / P3 +9.6 us at 1,192 for 0.46 GFLOP (MMA ideal ~0.01 us per SM); weight pass 10 us of P1 at C=768'
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring, interleaved); a 12-32 KB per-CTA cp.async round right after a grid sync still takes 2-4 us; removing the sumsq read saves ~1 us, so a last-arriver publisher removes no round. Flat vs ring per row: 1,160 -7%, 4,96 -7%, 4,192 +8%, 8,192 +20% (dispatch set). The mechanism behind the 2-4 us round is open (H24)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile (T tokens x BN hidden channels, fp16) in
    shared memory across the grid sync, ingests its W1/W2 slices from L2 through cp.async.bulk rings at >= 100 GB/s per SM,
    and reduces GEMM2 split-K partials with red.global.add into an output pre-filled with x + b2, runs the seven C <= 192
    rows in <= 15-20 us rented (S ~0.8 per row, ~+0.07 of mean).
  status: refuted
  evidence:
  - 'r2 C (E3) probe: one-shot bulk ingest 120 GB/s/SM (590 KB in 5.0 us at 148 CTAs); in-ring fp16 conversion 2-2.5x slower; smem binds (BN=128 at 1,192; no plan for 8,192 / C>=320)'
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3, 4,192 90.6, 4,96 88.0 (kill line 25 us at 1,192); conv redundancy 4C/BN (2-6x) and serial dependent rounds dominate; no edge over B''s materialised hidden. Lineage C retired r3; the bulk-ingest, convert-at-fragment-load, entry-bulk-vectors and red.add findings carry into D'
- id: H21
  statement: The portal/rented ratio is 1.27-1.28 on every row for FP16 mma.sync/cuBLASLt kernels (clock-bound, S and L
    alike), so rented gains carry over in proportion and portal rows can be projected as rented x 1.28.
  status: open
  evidence:
  - 'portal #65570 (A v3) vs rented on the 8 rows that ran: 1.27-1.28 each (43.8/34.5 ... 2286/1790); fp32 anchor 1.30-1.32'
  - 'audit r3: supported on those 8 rows (C<=192 large-HW and the top two); no W=7/14/449x7 row measured, and v3fix ran 6 us slower in another session (would read 1.12), so the ratio on the weight-bound rows is 1.15-1.28 until a 16-row portal result exists (operator: submit r2-lib-cublaslt-fp16-tmaconv in r3)'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol (2.4e-3) on the 77M-element rows;
    a hi/lo fp16 split of one operand (2 MMAs) restores fp32-class accuracy at no cost where MMA work is <= 2 us (all S/M rows),
    so per-row error logging decides where to spend it.
  status: open
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <= 1e-7 fraction beyond 2.4e-3 with randomized LN affine; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r2 A: fp64 reference, 1-3 rounds per row: 0 beyond 2.4e-3 + 1e-5|ref| on all 16 rows, max 2.39e-3 at 32,128, 2.37e-3 at 8,768 and 32,768. r2 B (flat, randomized LN, 2 seeds): 1 element beyond 2.4e-3 at 4,96 and 8,192. r2 C: not logged'
  - 'r3: every experiment logs the harness''s 10 seeded rounds per row against the row''s own atol (see the precision note); rows with any beyond-atol element get the split before a portal slot'
- id: H23
  statement: tcgen05 kind::f16 (2SM pairs, TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the four
    GEMM-heavy rows (8,320; 32,768; 32,128; 64,96) and is the lever for the L band once the S/M rows are near the fused floor.
  status: parked
  evidence:
  - 'arithmetic: mma.sync ideal 29/33/76/260 us vs tcgen05 7/8/19/65 us at those rows'
  - 'audit r3 corrects the value: with the score formula the ideal savings give +0.047/+0.057/+0.034/+0.027 per row = +0.010 of mean, ~+0.005 against cuBLASLt (already tensor cores). Not a lineage thesis; an engine swap inside A or D once a row is MMA-bound (32,768 at mma.sync: 33 us ideal)'
- id: H24
  statement: The 2-4 us cost of a 12-32 KB per-CTA load round right after a grid sync (and the ~8 GB/s/SM LDG/cp.async
    staging in A's old K1) is an issue-rate artefact of 4-byte copies, integer div/mod and serialised waits, not post-barrier
    L2 latency; one cp.async.bulk per operand (measured 120 GB/s/SM one-shot) brings the round to <= 1.2 us, which cuts
    1,192 to <= 22 us rented and the other S rows by 8-12 us in the cooperative kernel.
  status: open
  evidence:
  - 'for: probes show 0.44 us per cold dependent step and ~120 GB/s/SM one-shot bulk copies (predicting ~0.5 us for a 32 KB round); A''s k1_dw staged with 4-B cp.async plus div/mod'
  - 'against: the S rows scale 1.28x with the clock (cycles, not ns), and the flat kernel''s rounds after membar.gl read lines just written by other SMs. Microbench in E2 (critic''s test): 296 CTAs, grid.sync, then 32 KB/CTA as 4-B cp.async / 16-B all-issued cp.async or LDG.128 / one cp.async.bulk, source just-written vs cold, globaltimer per CTA'
- id: H25
  statement: A weight-stationary cooperative kernel for the six wide-C small-image rows (each CTA owns a hidden slice BN and
    whole-image token tiles T, ingests its fp32 W1/W2 slices once per call through TMA/bulk rings with conversion at fragment
    load, keeps the per-(image,k) GRN statistics CTA-local, bulk-plane conv, GEMM2 through red.add or a materialised scaled
    hidden per row) runs 2,768 in <= 35 us, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 rented
    (about +0.035-0.05 of mean).
  status: open
  evidence:
  - 'arithmetic (pool I22/I25): weights 19 MB read once = 2.7 us at 7 TB/s; per-CTA ingest 130 KB-1.6 MB at 120 GB/s/SM = 1-13 us; mma.sync f16 ideal 1/4/8/16 us at 2/8/16/32,768; red.add volume M*C*4*(4C/BN) at 4.2 TB/s (e.g. 14 MB = 3.5 us at 2,768 with BN=24). Risk: three post-sync rounds at 2-4 us each (H24) and L2-aggregate limits when slices are shared. Stub probe then kernel in E3'
- id: H26
  statement: Replacing cuBLASLt GEMM1 + K2 and K3 + cuBLASLt GEMM2 + K4 in lineage A with two custom mma.sync FP16 GEMMs whose
    epilogues carry the LN fold, exact GELU, per-image sumsq and fp16 hidden store (GEMM1) and the s_b reduction, A-scale at
    fragment load, residual and NCHW store (GEMM2) cuts the pipeline to 3 launches and the S/M rows by 10-14 us rented
    (1,192 <= 26 us, 2,768 <= 48, 8,384 <= 60; about +0.04 of mean).
  status: open
  evidence:
  - 'r1 A breakdown: K2 8-19 us, K3 6-17 us, K4 4-8 us per S/M row; cuBLASLt GEMM1 5.1 us at 1,192. Leader story: a 3-4 launch pipeline with fused epilogues lands at the leader''s implied 24-37 us rented on S rows. Pitfalls from B apply (per-shape templates, all fragment loads first, smem-staged rolled GELU epilogue). Built in E1'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "A",
    "hypothesis": "H26",
    "title": "Three-launch pipeline: custom mma.sync FP16 GEMM1 and GEMM2 with fused epilogues replace cuBLASLt GEMM1+K2 and K3+GEMM2+K4, dispatched per row against the library path",
    "operation": "structural_mutation",
    "parents": ["r2-lib-cublaslt-fp16-tmaconv"],
    "band": "all",
    "instructions": "Step 0 (10 min): time the parent and the r0 anchor interleaved x3 under harness_time on all 16 rows in this container and report the clock; these are the controls for every comparison below. Step 0b (10 min, audit H15): cold after the 2xL2 flush, time every cuBLASLt heuristic algo (requestedAlgoCount 32, including split-K configs via the algo attribute) for GEMM1 98/392/784/1568 x 3072 x 768 and GEMM2 98/392/784/1568 x 768 x 3072 in FP16; report the best per shape against bytes/7 TB/s + 2 us. Step 1: build GEMM1f as a per-shape template (M tile 64 or 128, N tile from 64/96/128, K = C): A = d fp16 channel-major as K1 writes it, B = W1 fp16 from K1's conversion, 4-stage cp.async ring with all fragment LDS issued before any HMMA and no runtime guards around HMMA; epilogue z = rstd*(acc - mu*u_k) + v_k + b1 using the parent's LN fold, exact erff, per-(image,k) sum of squares reduced per column in smem then one red.add per tile and column into the 4 accumulator copies, fp16 hidden store with 16-byte stores through an smem-staged rolled loop (B's 48-erff register-direct epilogue took 16 us per tile; the staged loop fixed it). Compare GEMM1f alone against cuBLASLt GEMM1 + K2 at 1,192 (5.1 + 9.8 us) and 8,320 (11 + 37.8 us), interleaved. Step 2: build GEMM2f: prologue sums the 4 sumsq copies for the CTA's image(s), reduces mean_k in-CTA and puts s_k = 1 + gw_k*gf_k/(mean+eps) in smem; A = hidden fp16 scaled by s_k at fragment load; swapped operands (M = C output channels, N = token tile) so each accumulator row is contiguous along hw, giving a coalesced residual x read and NCHW output store and no K4; add b2' = b2 + gb @ W2^T computed by K1's spare CTAs. Compare against K3 + cuBLASLt GEMM2 + K4 at 1,192, 2,768 and 8,384. Step 3: per-row dispatch by interleaved measurement (custom vs library) with the library path kept where it wins (expected 32,128 and 64,96, possibly 8,320 and 32,768 where mma.sync ideal is 29-33 us); the kernel sequence per shape must be fixed. Step 4 (only if steps 1-3 are done): K1 issues cp.async.bulk.prefetch.L2 for W1 and W2 at entry (I15) and the hidden stores use evict_last (I11); keep each only if it wins >= 2% interleaved x3 at 2,768 and 8,320. Step 5: run_tests 16/16 with the portal-order check; log per row the max |err| and the count of elements beyond that row's own atol over the harness's 10 seeded rounds against an fp64 reference; if any row has a beyond-atol element, switch its cheap operand to the hi/lo fp16 split and re-run. Report rented per-row times, the step 0 controls, and the register/spill counts of both GEMM kernels.",
    "success": "Interleaved against the parent: 1,192 <= 26 us, 2,768 <= 48, 8,384 <= 60, 8,192 <= 50 rented, no row slower than the parent by more than 2%; run_tests 16/16 with zero beyond-own-atol elements; projected portal >= 0.645.",
    "refuted_if": "GEMM1f alone at 1,192 is >= 8 us cold (no better than cuBLASLt GEMM1 + K2's chain would allow) or the 3-launch total at 1,192 is >= 30 us rented interleaved; then each custom launch still carries a 6-10 us dependent chain and the lineage's remaining lever is the conv and the C=768 GEMM path only.",
    "model": "opus"
  },
  {
    "id": "E2",
    "lineage": "B",
    "hypothesis": "H24",
    "title": "Post-sync round microbench, then rebuild the cooperative kernel's post-sync loads as single bulk copies with every phase-independent operand prefetched at entry",
    "operation": "structural_mutation",
    "parents": ["r2-coop-flat-s"],
    "band": "S",
    "instructions": "Step 0 (controls): parent and r0 anchor interleaved x3 under harness_time on all 16 rows; report the clock. Step 1 (microbench, the critic's test, 30 min): a cooperative stub with 296 CTAs that does grid.sync then one per-CTA load round of 6 KB, 12 KB and 32 KB as (a) 4-byte cp.async per element with the parent's index maths, (b) 16-byte cp.async or LDG.128 with every load issued before the first wait, (c) one cp.async.bulk plus one mbarrier wait; the source is (i) lines just written by other CTAs before the sync and (ii) cold DRAM after the 2xL2 flush; stamp globaltimer and clock64 per CTA and report median and max round time and B/clk/SM for each of the 18 cells, plus the grid.sync cost alone with and without pending stores before it. Step 2 (branch on step 1): if (c) is <= 1.2 us for 32 KB just-written, rebuild the flat kernel: one producer warp per CTA with mbarrier full/empty pairs issues at kernel entry the CTA's W1 and W2 fp32 slices (converted with cvt.rn.f16x2 at fragment load, so no weight pass at C<=192), all small vectors in one bulk copy, the residual x tile and the conv halo (3-D TMA box starting at w=-4 with box width W+8, taps offset +1 in smem, as A found; 1-D bulk plane copies for W=7/14); after each grid.sync exactly one bulk or tensor copy per operand (the a/hidden tile, gf); epilogues stage in smem and leave through 16-byte or bulk stores; keep the per-shape templates and the shfl-uniform warp index. If (c) is still >= 2 us but (b) is no better, the cost is post-barrier latency: do not rebuild the loads; instead replace sync2 with per-image counters (self-resetting, zeroed in phase 1 of the same call, ld.acquire.gpu spin with nanosleep) and an image-major tile order (H10 / I16) at 8,192,28,28 and 32,768,7,7, and measure that interleaved x3. Step 3: interleaved A/B against the parent x3 on 1,192; 1,96; 1,160; 2,96; 4,192; 4,96; 8,192; 2,768; set the dispatch per row from that. Step 4: run_tests 16/16 with the portal-order check and the per-row max |err| and beyond-own-atol counts over the harness's 10 seeded rounds; apply the hi/lo split where needed. Report the microbench table in the findings whatever the branch taken.",
    "success": "Bulk branch: 1,192 <= 22 us and 2,96 <= 32 us rented interleaved (parent 29.8 / 43.4), the other S rows -6 us or more; projected portal >= 0.60. Dataflow branch: >= 6 us faster per row at 8,192 and 32,768.",
    "refuted_if": "The 32 KB just-written bulk round after a grid.sync costs >= 2 us in the microbench and the rebuilt kernel (either branch) is within 2 us of the parent at 1,192 and 2,96; then the cooperative kernel's floor is barrier-side latency, H24 is refuted, and B's thesis of S rows <= 20 us needs a barrier-free dataflow or should be retired in r4.",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "D",
    "hypothesis": "H25",
    "title": "Weight-stationary cooperative kernel for the six wide-C small-image rows: fp32 weights ingested once per call through TMA/bulk rings, whole-image token tiles with CTA-local GRN statistics, bulk-plane conv",
    "operation": "new_design",
    "parents": ["r0-convnext-coop-fp32-anchor"],
    "band": "M",
    "instructions": "Target rows: 2,768,7,7; 8,768,7,7; 16,768,7,7; 32,768,7,7; 8,384,14,14; 16,384,14,14. The other 10 rows route to the r0 baseline paths through a shape dispatch (correct for all 16). Step 0 (controls): r0 anchor x3 interleaved under harness_time on the six rows; report the clock. Step 1 (stub probe, 30 min, kill gate): a cooperative kernel of <= 148 CTAs that, after the 2xL2 flush, issues at entry a bulk/TMA copy of each CTA's real fp32 W1 and W2 slice bytes (distinct slices, 19 MB total at C=768; 4.7 MB at C=384; TFLOAT32 or FLOAT32 maps, or 1-D cp.async.bulk of contiguous rows), copies 4-channel 7x7 plane groups (784 B) or single 14x14 planes (784 B) with 1-D cp.async.bulk into zero-padded smem planes with all copies issued before any store, does two grid syncs each followed by one bulk load of the real a-tile or hidden-tile size, and red.adds the real GEMM2 partial volume into a prefilled output; harness_time at 2,768 and 32,768. Kill the kernel build if the stub exceeds 20 us at 2,768 or 35 us at 32,768 and spend the rest of the session finding which term is responsible (weight ingest vs post-sync rounds vs red.add) with globaltimer stamps. Step 2: tile table per row with T a multiple of HW (tiles hold whole images) and BN a multiple of 16, e.g. 2,768: T=98, BN=24 (128 CTAs); 8,768: T=196, BN=48; 16,768: T=392, BN=64; 32,768: T=392, BN=96 (4 token tiles x 32 slices = 128 CTAs); 8,384: T=196, BN=64; 16,384: T=392, BN=96; adjust to fit smem (fp32 W1 slice BN*C*4 plus W2 slice C*BN*4, through a ring when above ~100 KB; hidden slice T*BN*2 fp16). Phase A per CTA: bulk-plane conv for its channel share, d written fp16 channel-major to L2 plus per-token LN partial sums, output prefilled with x + b2' (b2' = b2 + gb @ W2^T by spare warps), while a producer warp streams the W1 slice (fp32, resident or ring; convert with cvt.rn.f16x2 at fragment load, never inside the ring); all small vectors bulk-loaded at entry. Sync 1. Phase B: GEMM1 over T tokens x BN hidden with K = C using mma.sync m16n8k16 FP16 (per-shape templates, all fragment LDS first, no runtime guards around HMMA); A = d streamed from L2 in k-chunks; LN applied in the epilogue as r*(acc - mu*s1) + c1 (B's fold); exact erff; hidden slice fp16 kept in smem; per-(image,k) sum of squares is CTA-local because tiles hold whole images, so write gf[b][k] directly and one per-image partial sum over the CTA's k (no atomics). Sync 2. Phase C: read the B x (4C/BN) partial sums (one bulk round), form s_k for the CTA's own channels, scale the smem hidden slice, GEMM2 partial with K = BN against the W2 slice streamed during phase B, red.global.add into the prefilled NCHW output (rows contiguous along hw). Per row decide by arithmetic between red.add split-K (volume M*C*4*(4C/BN) at 4.2 TB/s) and materialising the scaled fp16 hidden in L2 followed by a c-sliced GEMM2 (B's path), and state the choice in the findings. Step 3: interleaved x3 against the r0 anchor on the six rows; globaltimer stamps per phase at 2,768 and 32,768. Step 4: run_tests 16/16 with the portal-order check and the per-row max |err| and beyond-own-atol counts over the harness's 10 seeded rounds; apply the hi/lo split where needed. Report the stub table, the tile table, phase stamps and per-row rented times.",
    "success": "2,768 <= 35 us, 8,768 <= 45, 32,768 <= 75 rented (A r2: 64.3 / 63.1 / 97.0; leader-implied 27 / 34 / 67), with 16,768 <= 55 and 8,384 <= 45; run_tests 16/16; the six rows together project >= +0.03 of mean over A r2.",
    "refuted_if": "The stub in step 1 exceeds 20 us at 2,768 or 35 us at 32,768, or the full kernel is not at least 10 us faster than A r2 at both 2,768 and 8,768 interleaved; then the wide-C rows are bound by post-sync rounds or weight ingest rather than by A's launch chain, and D's thesis moves to a 3-launch TF32 swap-AB specialist (I18) or is retired.",
    "model": "opus"
  }
]
```