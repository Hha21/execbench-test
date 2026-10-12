### Assessment

The anchor sits at 0.3038 against a leader at 0.69 and ceilings of 0.977 (TF32) / 0.988 (FP16). The audit is right that this is a fixed-cost problem, not a GEMM-rate problem: Tb is 68–222 µs on 13 of 16 workloads while the tensor-core work there is under 10 µs. The score model confirms it: every size at 12 µs + bytes/2 TB/s gives 0.88, at 30 µs + bytes/1 TB/s (a leader-like launch chain) 0.75, at 10 µs + bytes/4 TB/s 0.92. A 10% cut in one band is worth only 0.007–0.008. So the question each lineage must answer is its fixed floor F (launch + barriers + phase ramps and tails), with tensor cores as a precondition rather than the lever.

Audit answers: H9 refuted (Tb/eager 0.34–0.59, anchor 0.10–0.51). H6 restated at the 1500 MHz lock (35–38 TF/s; portal/rented 1.30–1.32 in every band). H2 reworded and parked: a second dwconv pass costs ≤0.003 of mean score until t < 500 µs at the largest. H5 and H4 downgraded to open; their isolating ablations (split-K forced to 1; cooperative vs three launches interleaved at S) go into E2. H7 parked until a tensor-core kernel exists. The FP16-only verdict is softened: TF32 straight from fp32 avoids the 19 MB conversion at C=768; the TFLOAT32 rounding probe goes into E3 and cuBLASLt TF32-fast vs FP16 into E1. The dirty-L2 hidden round-trip probe goes into E1. Bands by MB are misleading; the ledger now asks for per-workload reporting and records the card cell fix (616.9 MB, not 118.0 G).

Three lineages, one experiment each, maximally different in launch structure: A (plain launches, library GEMMs first, fp16 hidden materialised: fast control, should match the leader), B (one cooperative launch, legacy mma.sync FP16, grid syncs: the fixed-cost specialist, mutating the anchor), C (cluster-per-image with DSMEM: no grid barrier and no global intermediates, 14/16 workloads fit, starting on the 6 worst 7x7/14x14 rows). Each round-1 session also runs the probes that settle the open premises.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (custom dwconv+LN, library then custom tensor-core GEMMs, fp16 materialised hidden) that wins by per-phase efficiency and accepts about 2 us per launch; the control lineage expected to match the leader within one round.
    status: active
    started: r1
    best: null
    conclusions:
      - "Seed: cuBLASLt FP16/TF32 GEMMs need no tcgen05 code; 5 launches put the fixed floor near 10-12 us plus GEMM ramps, so S rows land at ~25-35 us (S ~0.7), L rows at tensor rates (~0.9)."
      - "Ceiling argument: 3 launches at ~2 us each plus ramps gives F ~8 us, i.e. ~0.82-0.85 mean if every phase is near its floor."
      - "Owns the premise tests for the materialised hidden under the dirty-L2 flush (H16) and cuBLASLt small-M heuristics (H15)."
  - id: B
    thesis: One cooperative launch with every phase fused, FP16 tensor cores through legacy mma.sync (registers and smem, no TMEM/TMA), grid syncs between phases; the fixed-cost specialist aiming at F <= 10 us on the S/M rows.
    status: active
    started: r1
    best: null
    conclusions:
      - "Seed: mutates the r0 anchor (same phases, split-K, dwconv geometry); only the GEMM core, the weight transposes (emit fp16) and the W<=14 dwconv path change."
      - "Ceiling argument: launch 1.8 + two 148-296-CTA barriers ~3 us + three short phases -> 8-12 us at S (S ~0.87); L bound by the unknown HMMA rate on sm_100 (H12)."
      - "Owns the premise tests on cooperative vs three launches (H4), split-K at small M (H5) and the fixed floor F (H11)."
  - id: C
    thesis: Cluster-per-image in-SM fusion: an 8-16 CTA cluster owns one image (two at 7x7), each CTA holds a hidden slice in smem, LN/GRN statistics and GEMM2 partials reduce through DSMEM; no grid barrier, no global intermediates; 14 of 16 workloads fit, the top two are dispatched to the r0 anchor path for now.
    status: active
    started: r1
    best: null
    conclusions:
      - "Seed: targets the six 7x7/14x14 rows first (anchor S 0.10-0.23, Tb 76-181 us); if they land at 15-30 us the kernel alone is worth ~+0.2 of mean score."
      - "Kill criterion from the probe: fewer than 6 concurrent 16-CTA clusters at 1 CTA/SM, or weight-slice ingest plus one swap-AB GEMM per CTA above 12 us under harness timing -> fall back to 8-CTA clusters or retire."
      - "Cluster use here is a GEMM split (hidden slices = split-N of GEMM1 / split-K of GEMM2 reduced via DSMEM); flagged to the operator against the clusters constraint before any portal slot."
      - "Owns the TFLOAT32 TMA rounding probe (H14)."
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r1 2026-10-12.
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
notes:
- 'Report effects per workload (or at least C=768 rows vs others), not per MB band: the MB bands mix weight-bound (C=768 7x7),
  latency-bound and stream-bound rows. Card fix pending: the MB cell for 64,96,112,112 is 616.9 MB, not "118.0 G".'
- 'Portal/rented ratio of the fp32 SIMT anchor: 1.30-1.32 in every band (clock-bound). Expect ~1.05 for DRAM-bound parts.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload
    memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still
    under 10 us except at the top three sizes).
  status: open
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'audit r1: the highest rate the leader implies is 174 TF/s (19% of TF32 peak), so the FP16 2x rate buys no score; TF32 avoids the 19 MB weight conversion at C=768 (11.5 us rented, ~15 us portal, against Tb 76 us at 2,768,7,7)'
- id: H2
  statement: The depthwise 7x7 conv becomes the binding compute term only once t < ~500 us at 64,96,112,112 (FMA bound
    134 us per pass at 1500 MHz, 29 us at 32,128,64,64); until then a second pass costs <= 0.003 of mean score, so recompute
    designs are not penalised and the dwconv is chosen for simplicity.
  status: parked
  evidence:
  - 'probe: SIMT 4x4 register patches measured 516-549 us at the largest at 1965 MHz (bound there is 102 us, so 5x off), latency-bound at 2 CTAs/SM'
  - 'probe: zero-padded smem planes do 7x7/14x14 images in 13-36 us; warp-level weight transposes 11.5 us vs 42 us block-level'
  - 'audit r1 arithmetic with portal Tb 4126 / Tsol 70: t=400 -> S 0.925, t=534 -> 0.897, i.e. 0.0017 of mean score per extra pass'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and
    0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA); elsewhere a materialised fp16 hidden (<= 42 MB) costs a dirty-line
    write-back per allocated line under the harness flush, so it is not a free tie.
  status: open
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32 (its P2/P3 phases are compute-bound at 17-25 TF/s so the traffic is hidden today)'
  - 'audit r1: #38 lost ~4 us at M/L to dirty write-backs; the cost here is untested (see H16, scheduled in E1)'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card
    (1.5 us per 296-CTA barrier vs 2.0 us per extra launch) and the portal difference is within noise.
  status: open
  evidence:
  - 'probe: empty kernel 1.8 us, 2 launches 3.8, 3 launches 5.8; 296 CTAs with 1/3 atomic barriers 3.6/6.7 us; 1184 CTAs 18 us'
  - 'r0: cooperative anchor 6717 us vs 6263 us for the same three phases as separate launches at the largest size (one comparison; a 454 us gap cannot be two barriers, so it is register budget/occupancy under launch_bounds(256,2) or grid size). Downgraded by the r1 audit; interleaved repeats at S sizes scheduled in E2'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7; 1,192,28,28) split-K (cross-CTA atomics into an x+b2-initialised
    output, DSMEM reduce, or warp-level K groups) is required to use the SMs; without it GEMM2 runs on < 50 CTAs.
  status: open
  evidence:
  - 'r0 v1 -> v2: 2,768,7,7 2091 -> 124 us, 1,192,28,28 546 -> 89 us, but the dwconv tiling, layout and split-K changed together; split-K share not isolated (audit r1). Ablation split3=1 vs auto scheduled in E2'
- id: H6
  statement: The fp32 SIMT GEMM core runs at 46-50 TF/s on the rented card at 1965 MHz, i.e. 35-38 TF/s at the 1500 MHz
    portal lock (the anchor is clock-bound, portal/rented 1.30-1.32 in every band); the anchor loses to 17-25 TF/s through
    the GELU/store/warp-sum epilogue on short-K tiles and the per-element GRN affine. The fp32 SIMT ceiling mean (0.665)
    is below the leader (0.690), so SIMT GEMMs cannot reach the top.
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s (unroll 1 best), 57-60 TF/s without global loads, layout variants within 5%'
  - 'probe ablations: no-epilogue P2 34-35 TF/s vs 20-23; no-affine P3 33-35 vs 22-25; launch_bounds(256,1) slower everywhere'
  - 'portal #65377 vs rented: 127.2/97.5, 270.0/204.7, 1317.4/1010.5 = 1.30-1.32'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes (416 tiles = 1.4 waves at 32,768,7,7; N=96/160
    padded to 128), so per-size tile tables or stream-K matter as much as the core rate.
  status: parked
  evidence:
  - 'r0 cost model (waves x tile work / efficiency) picked 128x96 tiles for C=768 and 32,768,7,7 still took 1217 us vs 478 us reference; nothing isolates quantisation (audit r1). Revisit once a tensor-core kernel exists'
- id: H8
  statement: Applying the GRN scale per image to W2 (or converting A to fp16 with the scale folded in) removes the per-element
    affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'probe: noaff variant of phase 3: 495 vs 678 us at 32,128,64,64, 1816 vs 2684 us at the largest'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager (157 us .. 11.5 ms), so even the fp32 anchor scores > 0.5 on
    the S/M band and the first FP16 tensor-core design moves most of the score.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload (0.43 at 1,192,28,28; 0.34 at 2,768,7,7; 0.36 at the largest); the anchor scores 0.10-0.51 and only 1,96,56,56 is above 0.5. Reaching S=0.5 needs t <= 68-222 us on 14 rows'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images'
    GEMM1 work at B >= 8 and removes two barrier waits at S.
  status: open
  evidence:
  - 'none yet; needs a self-resetting counter workspace (allowed under the constraints)'
- id: H11
  statement: The fixed floor F (launch + barriers/launch gaps + phase ramps and tails) sets the score on 13 of 16 workloads
    (Tb 68-222 us vs <= 10 us of tensor-core work); F <= 10 us gives S ~0.85-0.9 per row, F ~30 us gives ~0.6-0.7 (the
    leader's profile).
  status: open
  evidence:
  - 'leader gap table: implied leader times 30.6-67 us on the S/M rows need only 16-132 TF/s'
  - 'predict_score: every size at 12 us + bytes/2 TB/s -> 0.88; 30 us + bytes/1 TB/s -> 0.75; 10 us + bytes/4 TB/s -> 0.92; a 10% cut in one band is worth 0.007-0.008'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz, enough to make every
    workload except the top two fixed-cost-bound without TMEM/TMA code.
  status: open
  evidence:
  - 'none yet; microbench in E2 (clock64 + globaltimer, 148 CTAs, independent accumulator chains)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, each owning a hidden slice in smem; LN/GRN statistics and GEMM2
    partials reduced through DSMEM) runs the 7x7/14x14 workloads in <= 30 us with no grid barrier and no global intermediates;
    the bound is weight-slice ingest (~1.2 MB per SM at C=768 from L2).
  status: open
  evidence:
  - 'arithmetic (pool I2/I20/I27): per-image hidden fp16 <= 2.6 MB fits 16 x 160 KB for 14/16 workloads; 16-CTA clusters need a GPC (~18 SMs) so ~8 concurrent clusters. Probe and first kernel in E3'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest (not truncate), so kind::tf32 GEMMs can read fp32 weights and
    activations straight from the inputs with no conversion pass (saves 19 MB read + 9.5 MB write and a barrier at C=768).
  status: open
  evidence:
  - 'tricks.md: truncation gives ~2.5x the error (~1.1e-3 rms vs atol 2.4e-3, likely failing the 98% ratio); bit-exact probe in E3'
- id: H15
  statement: cuBLASLt FP16 (and TF32-fast from fp32) GEMM1+GEMM2 cost <= 10 us at the S sizes and pick sane kernels at
    M=98..784 with K=3072, so a 5-launch library pipeline lands at 25-35 us on the S rows (S ~0.7, roughly the leader).
  status: open
  evidence:
  - 'r0: cuBLAS fp32 sgemm both GEMMs 36-52 us at S (8-28 TF/s) is the only library number so far. Timed per shape in E1'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs a write-back per allocated
    line (~3-6 us at 8,320,32,32 and the C=768 rows), even when it fits L2.
  status: open
  evidence:
  - '#38/#85 lesson (dirty lines evicted inside our window); untested here. Probe in E1: write+read 20/42/134 MB scratch after the 2xL2 zero-fill vs no scratch'
- id: H17
  statement: The anchor's phase 1 at C=768 (333 us at 32,768,7,7) is block-level weight transposes (42 us) plus the
    4-tokens-per-lane conv (306 us); the measured replacements (warp-level transposes 11.5 us, zero-padded smem planes 35.5 us)
    cut it to ~45 us with no other change.
  status: supported
  evidence:
  - 'r0 probes: block-level transposes 42 us vs warp-level 11.5; 4 tokens/lane 306 us vs 1 token/lane 59 us; smem planes 35.5 us at 32,768,7,7, correct to 1e-5. Not in the submitted kernel; applied in E2'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "A",
    "hypothesis": "H15",
    "title": "Library pipeline: custom dwconv+LN, cuBLASLt FP16 GEMMs with fused bias/residual, fp16 hidden, strided-batched NCHW output",
    "operation": "new_design",
    "parents": [],
    "band": "all",
    "instructions": "Build a C++ solution (languages cuda_cpp + cublas; add -lcublasLt -lcublas to ld_flags; binding.cpp + kernel.cu layout; the cuBLASLt handle, matmul descriptors and the per-shape algorithm table are module-level caches keyed on (B,C,H,W) only, created on first call with cublasLtMatmulAlgoGetHeuristic and never re-tuned). Launch sequence per call, identical for a given shape: K1 custom: depthwise 7x7 conv + bias + LayerNorm -> A1 fp16 [M][C] row-major (use zero-padded smem planes per warp for W<=14, register patches or TMA-free float4 rows otherwise; LN needs all C per token so tile tokens x all channels); extra blocks of K1 convert W1 -> fp16 [4C][C] and W2 -> fp16 [C][4C] with cvt.rn.f16x2 and compute b2' = b2 + W2*gb (GEMV). GEMM1: cublasLtMatmul fp16 x fp16, fp32 compute, BIAS epilogue (b1), output fp16 or fp32 [M][4C] (fp16 only if run_tests passes 16/16; report max err both ways). K2 custom: exact erf GELU (erff, not tanh), write h fp16 [M][4C], per-(image,k) sum of squares via warp reduce + atomics into a [B][4C] fp32 workspace zeroed by K1. K3 custom: s_b[k] = 1 + gw[k]*sqrt(gf2)/(mean_k sqrt(gf2) + eps) and apply it: when B*C*4C*2 bytes <= 16 MB write W2_b = W2*diag(s_b) fp16 per image, else scale h rows in place (half2). GEMM2: strided-batched cublasLtMatmul computing D_b^T [C][HW] = W2_b [C][4C] x h_b^T so D lands directly in NCHW, beta=1 with C operand = x_b (stride C*HW) and D = output_b, bias b2' per output channel (either in the epilogue or pre-added to the C operand by K1 writing output = x + b2' and using beta=1 with C == D). All workspaces from torch.empty in run(). Then measure, under harness timing on the rented B200: (1) per-launch times at 1,192,28,28; 2,768,7,7; 32,768,7,7; 32,128,64,64; 64,96,112,112; (2) for GEMM1/GEMM2 at every one of the 16 shapes, the top-3 heuristic algorithms, FP16 operands vs CUBLAS_COMPUTE_32F_FAST_TF32 on the fp32 A1/W (no conversion kernel): pick the faster per shape into the table and record both; (3) the H16 probe: a kernel that writes then reads back a 20, 42 and 134 MB fp16 scratch after the harness's 2xL2 zero-fill, versus the same kernel with no scratch, to price the hidden round trip per size. run_tests on all 16 workloads x 10 rounds; report per-workload rented times, not band geomeans.",
    "success": "16/16 pass; rented time <= 40 us on every S/M row (1,192 .. 2,768), <= 80 us at 8/16/32,768,7,7, <= 700 us at the largest; run_tests projected portal score >= 0.60; the per-launch breakdown and the H15/H16 numbers are recorded as findings.",
    "refuted_if": "cuBLASLt GEMM1+GEMM2 together exceed 20 us at the S shapes for every heuristic algorithm (then the library floor caps the lineage near 0.6 and r2 replaces the GEMMs with custom tensor-core kernels), or the H16 probe shows the fp16 hidden round trip costs > 10 us at 8,320,32,32.",
    "model": "sonnet"
  },
  {
    "id": "E2",
    "lineage": "B",
    "hypothesis": "H12",
    "title": "Anchor with mma.sync FP16 GEMM cores, fp16 transposed weights and the measured phase-1 fixes; probes for the HMMA rate, the fixed floor, coop vs three launches and split-K",
    "operation": "structural_mutation",
    "parents": ["r0-convnext-coop-fp32-anchor"],
    "band": "all",
    "instructions": "Step 0 (<= 10 min): microbench mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 throughput on the rented B200: 148 CTAs x 4 and 8 warps, each warp running 8+ independent accumulator chains from registers, timed with clock64 and %globaltimer; report FLOP/clk/SM and PF/s at 1.5 GHz, and record it as a finding whatever the result (H12). Step 1, phase 1 of the anchor: switch transpose_mat to the measured warp-level transposes and emit fp16 W1t [C][4C] and W2t [4C][C] (cvt.rn); for W <= 14 use the zero-padded smem-plane dwconv (measured 35.5 us at 32,768,7,7); keep xn fp32 K-major [C][Mp] and the LN partial sums. Step 2, GEMM phases: keep the CTA tiles (128x128, 128x96, 64x64, 32x32), 8 warps, BK=32 and the register double-buffering, but store A as fp16 [k][m] (affine applied per element in fp32 before cvt.rn.f16x2, as now) and B from the fp16 Wt into padded smem, load fragments with ldmatrix (.trans where the layout needs it) and accumulate with mma.sync m16n8k16 in fp32; keep the exact-erf GELU epilogue, write the hidden as fp16 [4C][Mp] (halves P2 store and P3 load traffic; it is the GEMM2 operand rounding already simulated), keep per-image sumsq atomics, split-K and the x+b2 output init. Keep registers <= 128 so 2 CTAs/SM stay resident. Step 3 probes under harness timing, 3 interleaved repeats each: (a) the fixed floor F: this kernel with all three phases stubbed to a trivial write (grid syncs intact) at 1,192,28,28 and 2,768,7,7; (b) H4: the same phase code as one cooperative launch vs three plain launches at 1,192,28,28, 2,768,7,7 and 64,96,112,112, recording regs per phase kernel; (c) H5: split3 forced to 1 vs auto at 2,768,7,7, 8,768,7,7 and 1,192,28,28; (d) a per-phase breakdown at 1,192,28,28 and 1,160,449,7. run_tests on all 16 workloads x 10 rounds; report per-workload rented times.",
    "success": "16/16 pass; rented <= 45 us on every S row, <= 60 us at 2,768,7,7, <= 150 us at 32,768,7,7, <= 2500 us at the largest; projected portal score >= 0.55; F, the HMMA rate, the coop-vs-3-launch and split-K numbers recorded as findings.",
    "refuted_if": "mma.sync FP16 throughput < 0.15 PF/s at 1.5 GHz with the GEMM phases still > 100 us at 32,768,7,7 (then B moves to a tcgen05 engine in r2), or the stubbed-phase floor F > 15 us (then one cooperative launch with grid syncs cannot reach S ~0.85 and B's thesis changes to counters/dataflow).",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "C",
    "hypothesis": "H13",
    "title": "Cluster-per-image kernel for the 7x7 and 14x14 rows (hidden slices in smem, DSMEM reductions, no grid barrier), with cluster-occupancy, DSMEM and TFLOAT32-rounding probes; other shapes dispatched to the anchor path",
    "operation": "new_design",
    "parents": ["r0-convnext-coop-fp32-anchor"],
    "band": "all",
    "instructions": "Step 0 probes (<= 20 min, record each as a finding): (a) cudaOccupancyMaxActiveClusters for cluster sizes 8 and 16 (cudaFuncAttributeNonPortableClusterSizeAllowed) with 200 KB dynamic smem and 256 threads per CTA: how many clusters run concurrently and the launch-to-first-instruction latency under harness timing; (b) DSMEM microbench in clocks: 16 CTAs all-gather a 75 KB fp16 tile (each writes its 1/16 into every CTA via st.shared::cluster) and reduce-scatter 16 x [768 x 49] fp32 partials (each CTA reads 15 slices via ld.shared::cluster); (c) H14: TMA-load fp32 values with chosen low mantissa bits through a TFLOAT32 tensor map and check bit-exactly whether the result is round-to-nearest or truncation. If (a) gives fewer than 6 concurrent 16-CTA clusters, use 8-CTA clusters (two clusters per image pair is not allowed; use 8 slices of 4C/8). Step 1 kernel, for 2/8/16/32,768,7,7 and 8/16,384,14,14 (dispatch every other shape to the anchor's cooperative kernel copied from the r0 baseline): one cluster per image (two 7x7 images per cluster when B >= 8), CTA j owns C/G channels for the dwconv (zero-padded smem planes) and hidden slice j of 4C/G. LN: per-token partial sums exchanged through DSMEM (one cluster barrier), each CTA normalises its channels and writes the fp16 a tile [tokens][C] into every CTA's smem via st.shared::cluster (second barrier). GEMM1 swap-AB: A = W1 slice [4C/G x C] (fp32 from global/L2 converted to fp16 in smem, or TF32 if (c) confirms rounding), B = a^T, N = tokens; use whichever engine you can get correct in the session (mma.sync FP16 is acceptable; tcgen05 is not required for the thesis). GELU (exact erf) in registers; gf_k = sqrt(sum over tokens of h^2) is thread-local per hidden channel; one DSMEM reduce of the per-CTA partial sums of gf gives mean_k gf (third barrier); s_k = 1 + gw_k*gf_k/(mean+eps), h' = s_k*h + gb_k in registers, written fp16 to the CTA's own smem. GEMM2 partial: W2[:, slice] [C x 4C/G] x h'^T -> [C x tokens] fp32; reduce across the cluster by DSMEM reduce-scatter (CTA j finalises output channels j*C/G .. (j+1)*C/G: + b2 + x, store NCHW with lanes along hw). Weights: each CTA reads only its two slices (1.2 MB at C=768) and ideally once per cluster via TMA multicast or plain L2 reads; measure which. Time each phase in clocks for one cluster at 32,768,7,7 and compare with the 12 us kill criterion from the probe. run_tests on all 16 workloads x 10 rounds; report per-workload rented times.",
    "success": "16/16 pass; rented <= 30 us at 2,768,7,7 and 8,768,7,7, <= 45 us at 16,768,7,7 and 32,768,7,7, <= 40 us at 8/16,384,14,14 (anchor: 124-1217 us); projected portal score >= 0.50 with the other 10 rows still on the anchor path; probe numbers (concurrent clusters, DSMEM gather/reduce clocks, TFLOAT32 rounding) recorded.",
    "refuted_if": "fewer than 4 concurrent 8-CTA clusters, or weight-slice ingest plus one GEMM1 per CTA > 12 us at 32,768,7,7 under harness timing, or DSMEM all-gather of the 75 KB a tile > 3 us; any of these means the cluster path cannot beat ~30 us per row and the lineage is retired after incubation with the probe numbers as its conclusions.",
    "model": "opus"
  }
]
```