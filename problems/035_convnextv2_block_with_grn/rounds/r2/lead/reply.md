### Assessment

**Where the score is.** Our best projected lineage (A, 0.594 at the measured 1.28 portal ratio) scores 0.59–0.65 on every row the portal timed, and the leader averages 0.69. The score model says the pot is in the 13 small rows: a fused kernel at 20 µs + bytes/1 TB/s scores 0.80, and the fused targets (S 23 / M 46 / L 105 µs portal) give 0.76. The two largest rows together are worth about 0.03 (conv and tensor rate), the GEMM-heavy mid rows another 0.02–0.03 with tcgen05. So r2 spends everything on the per-row floor and the conv, and defers tcgen05 to a later slot.

**What the lineages taught us.** A: the library GEMMs are cheap (12 µs at 1,192); the time is in K1 staging (8 GB/s per SM) and the K2/K3 chains, each launch costing 6–17 µs of SM work, not 2 µs. B: the cooperative floor is 4.6 µs and coop beats three launches by 3–7 µs at S; what remains is 8–10 µs per phase of dependent rounds and the 10 µs weight pass at C=768. C: 16-CTA clusters are schedulable, but DSMEM runs at 12 B/clk/SM and register-streamed weight ingest at 37 GB/s per SM, so cluster-per-image cannot reach 30 µs. C is in incubation and pivots to the on-chip-hidden design (one barrier, smem hidden, bulk weight rings, red.add split-K), which keeps its "no global intermediates" thesis.

**This round.** E1 (A) replaces the conv staging with TMA boxes and hardware zero padding, the one change every lineage needs. E2 (B) flattens the latency chain (cross-barrier prefetch, in-CTA weight conversion) with a globaltimer audit that also settles whether the per-launch cost is round trips or cycles. E3 (C) builds the one-barrier on-chip kernel for the seven C ≤ 192 rows after a 15-minute ingest probe. Every session logs per-row max error, because B's single-pass FP16 max error already touches the smallest atol.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (custom dwconv+LN, cuBLASLt FP16 GEMMs, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage, and the carrier of techniques (conv staging, epilogue fusion) that other lineages adopt through conclusions.
    status: active
    started: r1
    best: r1-lib-cublaslt-fp16-v3fix
    conclusions:
      - "Library GEMMs are not the limit: cuBLASLt FP16 GEMM1+GEMM2 = 5.1+6.8 us at 1,192,28,28; FP16 is 1.4-2.1x faster than TF32 at every shape, so the weight conversion must be hidden, not avoided."
      - "Each custom launch costs 6-17 us (K1 12.8-18.6, K2 8-14, K3 6-17 us at S), not 2 us; the 5-launch floor is 35-40 us at S, so A's ceiling is ~0.70-0.75 only with K2/K3/K4 fused into GEMM epilogues and a conv that is not staging-bound."
      - "K1 staging through LDG/LDGSTS runs at ~8 GB/s per SM (954 us at the largest, 234 at 32,128, 12.8-18.6 us at S); removing staging cuts K1 3x, so the conv needs TMA/bulk staging with hardware zero fill (E1)."
      - "H16: the fp16 hidden round trip is cheap under the dirty-L2 flush (3.6 us over the floor at 20 MB, 1.4x ideal at 134 MB); materialising the hidden is fine except at the top two rows."
      - "Portal #65570 timed 8 rows of v3 at 1.27-1.28x rented on S and L alike (S 0.59-0.65 per row); per-shape cudaFuncSetAttribute broke the other 8 (fixed in v3fix: attributes only ever raised)."
  - id: B
    thesis: One cooperative launch with every phase fused, FP16 tensor cores through legacy mma.sync (registers and smem, no TMEM/TMA), fp16 hidden materialised in L2 between grid syncs; the fixed-cost specialist aiming at S rows <= 20 us rented by flattening dependent round trips.
    status: active
    started: r1
    best: r1-convnext-coop-hmma-ring
    conclusions:
      - "H4 refuted at S: coop beats three plain launches by 3-7 us (29.7 vs 36.3 at 1,192; 46.7 vs 50.2 at 2,768); stubbed floor F = 4.6 us vs 9.9 for three launches; 2.5% slower only at the largest row."
      - "mma.sync m16n8k16 f16->f32 = 0.455 PF/s at 1.5 GHz (1/4 of tcgen05): enough for every row except 8,320 / 32,768 / 32,128 / 64,96, where the GEMM alone costs 29 / 33 / 76 / 260 us ideal."
      - "Per phase at 1,192,28,28: P1 13.2 (F 4.6 + conv + weight pass), P2 +7, P3 +9.6 us for 0.46 GFLOP; each phase is 2-3 dependent rounds plus ramps, i.e. the lever is overlap across barriers and fewer rounds, not the barrier count (E2)."
      - "Pitfalls: a runtime guard around mma.sync costs ~470 clk per k-block (use per-shape templates); a release arrive from a thread with loads in flight stalls; split-K is worth 28% at C=768 and nothing at 1,192; the C=768 weight pass costs ~10 us of P1."
      - "Largest row: P1 446 us (conv, FMA bound 134) and P2 ~840 us (short-K serial tile chain + erf epilogue) vs ~150 roofline; the conv staging lesson from A applies here too."
  - id: C
    thesis: On-chip hidden with no global intermediates and one grid barrier: each CTA owns a token tile x hidden slice, keeps the hidden in shared memory across the barrier, ingests its weight slices from L2 through cp.async.bulk rings, reduces GEMM2 split-K partials with red.global.add into an output pre-filled with x + b2; clusters dropped after the r1 probes.
    status: active
    started: r1
    best: r1-convnext-cluster16-dsmem
    conclusions:
      - "H13 refuted: 16-CTA clusters schedule (7 concurrent at 200 KB smem) and cluster launch is no dearer than plain, but DSMEM moves ~12 B/clk/SM (reduce-scatter 128 KB = 8 us) and register-streamed weight slices ingest at 37 GB/s per SM (590 KB = 16 us), so cluster-per-image floors at 50-60 us at C=768."
      - "H14 supported: TFLOAT32 TMA maps round to nearest even (bit-exact), so kind::tf32 could read fp32 weights straight from the inputs; FLOAT32 maps leave bits unchanged."
      - "red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs (1.0 TB/s from 32), so a split-K reduce of a few MB costs ~1 us; the v4 form is no faster."
      - "Zero-padded plane fills serialise on DRAM latency unless 16 loads are issued before the stores (15-20 us -> ~3 us for 48 planes)."
      - "Pivot r2: keep the in-SM thesis, replace DSMEM with one grid.sync + red.add and register prefetch with bulk rings; target the seven C <= 192 rows at <= 15-20 us rented (E3)."
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r2 2026-10-12.
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
- 'Portal/rented ratio (audit r2, corrected): 1.27-1.28 on all 8 rows of #65570 (lineage A v3: 43.8/34.5, 50.0/39.1, 64.1/50.0,
  68.1/53.6, 72.5/56.5, 83.7/65.5, 569/445, 2286/1790), S and L alike, and 1.30-1.32 for the fp32 anchor. There is no
  DRAM-bound 1.05 regime here; project portal = rented x 1.28, and rented gains carry over in proportion. The leader''s
  S-row times (30.7-47 us portal) are 24-37 us rented.'
- 'Precision logging (audit r2): lineage B measured single-pass FP16 max error 1.9-2.8e-3 (rms 3.9-4.2e-4) against atol
  2.4e-3 with randomized LN affine; lineage A never logged max error. Every run_tests from r2 on records per workload the
  max |err| and the count of elements beyond atol over 10 rounds; a row with any beyond-atol element must switch the
  cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M where MMA work is <= 2 us). Passing through the 98% ratio
  is not acceptable.'
- 'Cross-session rented timings drift by ~6 us per call (v3fix 50.4/72.6/192.3 vs v3 44.4/66.4/182.8 from a host-only
  change); rank lineages only from interleaved runs in one container, 3 rounds.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65;
  8,192 0.63; 32,128 0.62; 64,96 0.65. predict_score at the 1.28 ratio: A 0.591, B 0.563; fused targets S 23 / M 46 /
  L 105 us portal -> 0.76; 20 us + bytes/1 TB/s on every row -> 0.80.'
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
  - 'margin caveat: see the precision-logging note; B max error 1.9-2.8e-3 at rms 4e-4 means the tail touches the smallest atol on 77M-element rows (H22)'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at every size because the implementations are staging-bound,
    not FMA-bound (K1 954 us in A and P1 446 us in B at the largest against a 134 us FMA bound; 234 us at 32,128 against 29;
    8-13 us of the S rows); a conv at ~1.5x its FMA bound is worth ~+0.015 at the top two rows and ~-5 us per S row.
  status: open
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); a 64-channel 7-row stage-only kernel took 9.5 us vs 5.9 for a contiguous float4 read vs 1.9 floor (~8 GB/s per SM); plain LDG+STS equals cp.async'
  - 'r1 B: cp.async-staged bands were slower than direct __ldg patches (41 vs 29 us at 1,192); 8-channel tiles cut P1 25 -> 9 us at 1,192 (tile count matters at S)'
  - 'audit r2: unparked (was priced at the FMA bound). Measured conv is 25-53% of total time at the top two rows; TMA 3-D boxes with OOB zero fill tested in E1 (H18)'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and
    0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: open
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32'
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal (57 vs 38 us) at 134 MB, so materialising is cheap below 42 MB; at the largest row K2 alone took 447 us (616 MB write + read), so recompute still wins on paper there'
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
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row, a 3-round chain: sumsq, sqrt-reduce, W2 rows); neither isolates the affine cost'
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
  - 'none yet; a self-resetting counter workspace is allowed. Worth ~+0.015 at B >= 8 rows on top of any GEMM core (pool I6/I16); deferred behind the per-phase cycle cuts'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches;
    what sets the S rows is the SM work and dependent rounds inside each phase (8-10 us per phase at 1,192,28,28 in B;
    6-17 us per custom launch in A), all scaling 1.28x with the clock lock. Cutting the per-phase chain to ~3-4 us gives
    S rows of 15-20 us rented (S ~0.78-0.82); 30 us rented gives ~0.65 (lineage A today).
  status: open
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7 (P1 13.2 / P2 +7 / P3 +9.6)'
  - 'r1 A: cold dependent load 0.44 us per step, ~1 us per round once loaded; K1/K2/K3 are 2-3 round chains each'
  - 'audit r2: the S rows scale by exactly the clock ratio (1.28), so part of the per-phase cost is cycles, not latency; the split is measured in E2 (clock64 vs globaltimer)'
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
  status: supported
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; 8 heuristic algos per shape, top-3 within 0.3%'
  - 'r1 A: NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd; channels-last GEMM2 + 32x32 transpose-add (4-8 us) instead'
  - 'r1 A: the 5-launch pipeline lands at 34.5-58.9 us at S/M rows, not 25-35, because K1/K2/K3 cost 6-17 us each'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little (<= 4 us over the
    launch floor at 20-42 MB; ~1.4x ideal at 134 MB), so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
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
  - 'r1 A: staging removal cut K1 3x; stage-only 9.5 us vs 1.9 us floor for a 64-channel 7-row band. TMA row strides must be 16 B multiples, so W=7 and W=14 rows cannot use a row-boxed map (plane bytes 196/784 B: 4-channel groups of 784 B and single 14x14 planes are legal 1-D bulk copies). Tested in E1'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be
    overlapped across the grid syncs (phase-2 operands issued before sync 1, W2 K-slices and the x residual before sync 2,
    s_b published by a per-image last arriver) and the C<=384 weight pass can move into the consuming CTAs; together they
    cut 1,192,28,28 from 29.7 to <= 22 us rented and the other S rows by 8-12 us.
  status: open
  evidence:
  - 'r1 B per-phase: P1 13.2 / P2 +7 / P3 +9.6 us at 1,192 for 0.46 GFLOP (MMA ideal ~0.01 us per SM); weight pass 10 us of P1 at C=768. Audit r2 doubts the round-trip model because S rows scale 1.28x with the clock; E2 splits cycles from ns with clock64 + globaltimer'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile (T tokens x BN hidden channels, fp16) in
    shared memory across the grid sync, ingests its W1/W2 slices from L2 through cp.async.bulk rings at >= 100 GB/s per SM,
    and reduces GEMM2 split-K partials with red.global.add into an output pre-filled with x + b2, runs the seven C <= 192
    rows in <= 15-20 us rented (S ~0.8 per row, ~+0.07 of mean).
  status: open
  evidence:
  - 'pool I15/I21 arithmetic: per-SM weight ingest 295-590 KB at C=192, conv redundancy 4C/BN, red.add volume M*C*4*(4C/BN) at 4.1-4.4 TB/s (~1 us); hidden per SM <= 65 KB. Unknown: bulk-ring ingest rate from L2 (C measured 37 GB/s per SM through registers). Probe and kernel in E3'
- id: H21
  statement: The portal/rented ratio is 1.27-1.28 on every row for FP16 mma.sync/cuBLASLt kernels (clock-bound, S and L
    alike), so rented gains carry over in proportion and portal rows can be projected as rented x 1.28.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented on the 8 rows that ran: 1.27-1.28 each (43.8/34.5 ... 2286/1790); fp32 anchor 1.30-1.32'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol (2.4e-3) on the 77M-element rows;
    a hi/lo fp16 split of one operand (2 MMAs) restores fp32-class accuracy at no cost where MMA work is <= 2 us (all S/M rows),
    so per-row error logging decides where to spend it.
  status: open
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3, <= 1e-7 fraction beyond 2.4e-3 with randomized LN affine; A never logged. r0 simulation: FP16 hi/lo split err 1e-6. Logged per row in E1/E2/E3'
- id: H23
  statement: tcgen05 kind::f16 (2SM pairs, TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the four
    GEMM-heavy rows (8,320; 32,768; 32,128; 64,96) and is the lever for the L band once the S/M rows are near the fused floor.
  status: parked
  evidence:
  - 'arithmetic: mma.sync ideal 29/33/76/260 us vs tcgen05 7/8/19/65 us at those rows; per-row S difference ~0.06-0.1 each. Deferred: no free slot in r2; seed a lineage from pool I8/I11/I26 when a slot opens or C fails its kill criterion'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "A",
    "hypothesis": "H18",
    "title": "TMA/bulk-staged depthwise conv (K1) with hardware zero padding, replacing LDG/LDGSTS staging",
    "operation": "structural_mutation",
    "parents": ["r1-lib-cublaslt-fp16-v3fix"],
    "band": "all",
    "instructions": "Step 0: try `nvidia-smi -lgc 1500` on the rented card once and record whether it works (if it does, use it for every timing in this session and say so in findings). Interleave r1-lib-cublaslt-fp16-v3fix as the control in the same container for every comparison (3 rounds). Step 1 (stage-only probe, 15 min): a kernel that only stages x into smem through a 4-stage mbarrier ring and touches nothing else. For W%4==0 rows (28,32,56,64,112) build a 3-D CUtensorMap over x as dims {W, H, B*C} fp32, box {WT+6 rounded up to a 16 B multiple, RT+6, 8 channels}, coordinates (w0-3, h0-3, c0); the TMA unit zero-fills out-of-bounds rows and columns. For 7x7, 14x14 and 449x7 rows the row stride (28/56/28 B) is not a 16 B multiple, so stage whole planes with 1-D cp.async.bulk: 4-channel groups (784 B for 7x7, 50288 B for 449x7) or one 14x14 plane (784 B), and pad by index math in smem (issue all copies before any wait). 2 CTAs/SM, 64-100 KB smem per CTA. Measure GB/s per SM cold (after the 2xL2 zero-fill) at 64,96,112,112, 8,320,32,32 and 1,96,56,56 against the current staging (148 us at 8,320; 9.5 us for the 64-channel band). Kill if under 40 GB/s per SM or not 2x faster than the current staging. Step 2: put the winning staging into K1 (conv math, weight prep, LN partial stats and fp16 d output unchanged from v3fix); keep the 1-warp/tall-tile dispatch for small shapes only if it still wins. Keep attributes only ever raised (portal-order constraint) and rebuild the tensor map on the host every call (pointers shift). Step 3: add clock64 + globaltimer stamps inside K1 at 1,192,28,28 to report how much of K1 is cycles vs ns (audit item). Step 4: run_tests 16/16 x 10 rounds and log per workload the max |err| and the count of elements beyond atol; report them in findings.",
    "success": "K1 cold at 1,96,56,56 <= 9 us (from 16.4), at 8,320,32,32 <= 30 us (from 61.4), at 64,96,112,112 <= 500 us (from 954); run_tests 16/16 with zero beyond-atol elements; projected score >= 0.62 (from 0.594) with v3fix interleaved as control.",
    "refuted_if": "The stage-only TMA/bulk ring is less than 2x faster than the LDG staging at 8,320,32,32 or below 40 GB/s per SM at the largest (then the cost is halo redundancy / request pattern, not the instruction path, and conv work must move to fewer, taller tiles or tensor cores).",
    "model": "opus"
  },
  {
    "id": "E2",
    "lineage": "B",
    "hypothesis": "H19",
    "title": "Flatten the cooperative kernel's latency chain: cross-barrier prefetch, in-CTA weight conversion at C<=384, per-image s_b publisher",
    "operation": "structural_mutation",
    "parents": ["r1-convnext-coop-hmma-ring"],
    "band": "S",
    "instructions": "Step 1 (audit, 20 min): add %globaltimer and clock64 stamps at each phase boundary and after each dependent wait (weight pass, conv loads, sumsq read, s_b read, W2 first stage) at 1,192,28,28, 2,768,7,7 and 8,384,14,14 (CTA medians). Build a loads-only stub (same loads and stores, MMA and conv FMAs removed) and record how much of the ~25 us above F is pure latency vs SM cycles. Step 2: issue every phase-2 operand (first W1 cp.async stages, the d tile, b1, u/c1 vectors) before grid.sync 1 so the barrier wait overlaps the loads; before grid.sync 2 prefetch the W2 K-slices and the x residual tile; after sync 2 only the s_b read remains. Keep every arrive in warps that have no loads in flight. Step 3: for C <= 384 drop the global weight pass: load fp32 W1/W2 tiles with cp.async into an fp32 staging stage, cvt.rn.f16x2 and STS into the fp16 ring inside the consuming CTA (row sums for the LN fold from the converted values); keep the global pass only at C=768. Step 4: replace the sumsq-atomics-then-reread round with a per-image last-arriver (counter in the torch.empty workspace, self-resetting) that computes and publishes s_b[4C]. Step 5: keep registers at 128 (check for spills; a PF-deep prefetch that spills is a dead end). Step 6: run_tests 16/16 x 10 rounds, interleaved with the parent, and log per workload the max |err| and the count beyond atol.",
    "success": "Rented, interleaved: 1,192,28,28 <= 22 us (from 29.7), 2,96,56,56 <= 30 (42.2), 4,192,28,28 <= 34 (47.2), 2,768,7,7 <= 40 (47.9), no row slower than the parent by more than 2%; projected score >= 0.60 (from 0.561).",
    "refuted_if": "The loads-only stub shows less than 8 us of latency in the ~25 us of phases at 1,192,28,28 (then the lever is instruction count per phase, not round trips), or the cross-barrier prefetch spills and the S rows do not move.",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "C",
    "hypothesis": "H20",
    "title": "One-barrier on-chip-hidden kernel for the C<=192 rows: smem hidden tile, bulk weight rings, red.add split-K GEMM2",
    "operation": "new_design",
    "parents": ["r1-convnext-cluster16-dsmem"],
    "band": "S",
    "instructions": "Pivot: no clusters, no DSMEM. Step 1 (probe, 15 min): 148 CTAs (1/SM, cooperative launch) each cp.async.bulk-load 300 KB and 590 KB of fp32 from an L2-resident buffer through a 4-stage ring and convert to fp16 in smem; time under harness_time after the 2xL2 zero-fill and report GB/s per SM. Go if 300 KB takes <= 3 us; if above 6 us, use BN = C slices (more conv redundancy, half the ingest). Step 2 (kernel, dispatch:size, other rows routed to the lineage's existing paths so 16/16 passes): one cooperative launch, one grid.sync. Per-shape planner picks T tokens x BN hidden channels per CTA (tiles never straddle images; e.g. 1,192: 37 token tiles x 4 hidden blocks of 192; 1,96: BN=384 with ~21 tokens per CTA) so that conv x (4C/BN) / 148 + weight bytes(BN) / ingest rate + red.add M*C*4*(4C/BN) / 4.3 TB/s is minimised. Phase A, all loads issued at entry: the x halo patch for the CTA's tokens over all C channels, the W1 slice [BN x C] and W2 slice [C x BN] fp32 through the rings, converted to fp16 on the way into smem (no weight pass, no second barrier). Conv (register patches, plane fill with 16 loads issued before stores), LN per token (mean/rstd in fp32), mma.sync m16n8k16 GEMM1 (per-shape template, no runtime guard around HMMA), erf GELU, hidden tile [T x BN] fp16 kept in smem; per-(image, k) sumsq partials written to per-CTA slots in the torch.empty workspace (no atomics, no zeroing). CTAs of hidden block 0 write out = x + b2 with plain stores. grid.sync. Phase B: read the slots of the CTA's image (<= 60 KB from L2), mu_b and s_b, scale the smem hidden in place (h*s_b[k] + gb[k]), GEMM2 partial [T x C] over K = BN with the W2 slice, red.global.add.f32 into out (plain stores when BN = 4C). Step 3: run_tests 16/16 x 10 rounds; log per workload the max |err| and the count beyond atol; report the phase timeline (globaltimer) at 1,192 and 4,192.",
    "success": "Rented: 1,192,28,28 <= 18 us, 1,96,56,56 <= 16, 2,96,56,56 <= 20, 4,192,28,28 <= 24, 4,96,56,56 <= 26, 8,192,28,28 <= 32 (lineage B today: 30.9 / 28.3 / 42.2 / 47.2 / 62.5 / 66.1); 16/16 pass with zero beyond-atol elements.",
    "refuted_if": "The bulk-ring ingest probe exceeds 6 us for 300 KB per SM, or the full kernel exceeds 25 us at 1,192,28,28 and 2,96,56,56 (then on-chip hidden has no edge over B's phases at S and lineage C is re-seeded from the tcgen05 family, H23, in r3).",
    "model": "opus"
  }
]
```