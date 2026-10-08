### Assessment

Where the score is: r1 confirmed that the compute band (6 workloads, Tb 6-9% below anything cuBLAS or CUTLASS gives us on the portal) is parked, and that the small band (M<=16, S 0.54-0.55) is within 0.04 of its ceiling. The remaining pool is the 12 mid workloads M=17-952, now at S 0.46-0.52. Reaching the ~7.8 us portal read floor at M=17-172 is worth about +0.03; cutting M=289-952 by 20-30% is worth another +0.01. The score model prices 15% off the M band plus 3% off L at 0.517 vs 0.487.

What we learned: up-front loading serialises all MMA work after the B stream (r1 E2), so the mid band needs a pipelined ring. The r1 evidence also points to two mechanisms behind cuBLAS's mid-band times that the ledger now states as hypotheses: wave quantisation of its tile picks (M=289: 150 CTAs, M=952: 304 CTAs on 148 SMs) and a per-SM L2-to-SMEM ingress bound that caps small tcgen05 tiles at ~40% of peak. Both are measurable in a session.

This round: E1 builds the pipelined cp.async/mma.sync streaming kernel for M=17-172 (H16, the largest pool). E2 is the cheap cublasLt sweep over all 12 mid sizes, enumerating beyond the heuristic (split-K, tiles, cluster shapes) with a 4% transfer bar (H14). E3 is exploratory: measure the per-SM ingress bound (H15), then test whole-wave tcgen05 tiles at M=289-952 through CUTLASS (H17), reporting residuals so r3 knows whether a custom tcgen05 kernel is justified. No large-M work.

```yaml ledger
# Hypothesis ledger for FlashInfer-Bench/009_gemm_n5120_k2048, seeded 2026-10-08 by the research phase (r0); updated r1 and r2 (2026-10-08).
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
- 'fp16 inputs with fp32 accumulation only: no FP8/FP4/TF32-style truncation of A or B, no fp16 accumulation.'
- 'Portal Tb at 9-20 us is +-5% noisy per size (r0 page): judge mid-M candidates on rented ours/reference ratios over
  interleaved repeats, never on one portal size.'
- 'Compute-band candidates cannot be ranked on the rented box: it power-caps to ~1117 MHz under sustained GEMM load and
  nvidia-smi -lgc is refused (r1). CUTLASS 256x256 read -1..-3% rented and +1..+2.5% portal. No large-M (M >= 8828) work
  until a portal-representative measurement method exists; the base keeps at::matmul_out there.'
- 'Transfer rule for the mid band (r1 portal 62881): only rented gains of >= 4% over interleaved repeats count (tile 183 was
  -7% rented and -1% portal; tile 314 -7/-10% rented and -7/-10% portal).'
- 'Every candidate must launch the same kernel sequence on every call; a second kernel (e.g. a cublasLt split-K reduction)
  is legal but its gap and tail are timed, so it must win by >= 4% net in harness timing.'
hypotheses:
- id: H1
  statement: For M <= 172 the time is the cost of streaming B (21 MB) under the harness, about 6.4 us (read floor incl. ~2 us dirty-L2 write-back), plus a kernel-specific tail; cuBLAS sits 1.4-3.9 us above that.
  status: supported
  evidence:
  - 'probe r0: 256-bit read-only kernel 6.4-6.7 us for every grid shape; empty kernel 1.5 us; read-only flush is 2.1 us faster than the zero flush (event timing)'
  - 'run_tests r0: cuBLAS 7.6-10.3 us at M <= 172'
  - 'portal r0: our 7.7-8.2 us at M <= 16 vs rented 6.9-7.2 (ratio 1.07-1.13), so the portal floor is ~7.0-7.5 us'
- id: H2
  statement: At M <= 16 a one-shot 640-CTA mma.sync kernel with 256-bit loads straight into fragments beats cuBLAS by ~10%; the residual 0.6 us is the post-load tail plus 640/148 CTA imbalance.
  status: supported
  evidence:
  - 'run_tests r0: 6.9-7.2 us vs cuBLAS 7.6-7.9'
  - 'portal r0: 7.7-8.2 us vs Tb 8.7-9.3, S 0.54-0.55 (-10% on both machines)'
  - 'probe r0: 8 warps x 4 loads/lane best; 4 warps 8.9 us, 16 warps 7.7-8.4 us, BN=16 7.5-7.7 us'
  - 'r1 E2: the 0.6 us tail is the mma.sync work serialised after the up-front stream (same mechanism as H3); 2/4 independent accumulator chains made it worse (8.0-9.2 us, 60-72 regs)'
- id: H3
  statement: For 17 <= M <= 172 the winning structure is BN = 32-64 B-rows per CTA with A staged once per k-chunk in shared memory and split-K across a 2-4 CTA cluster (DSMEM reduction) or a last-CTA-reduces counter, reaching ~7-7.5 us rented (~7.8-8.3 portal, S ~0.56); re-reading A per warp from L2 does not work.
  status: refuted
  evidence:
  - 'probe r0: per-warp A re-read (640 CTAs) 10.8 us at M=17, 19.5 us at M=64, 33.7 us at M=128 (A L2 traffic = 640 x M x 4 KB)'
  - 'r1 E2 (r1-splitk4-cluster-m17-64): up-front B loads + A slice via cp.async + 4-CTA DSMEM reduction: 10.4-21.1 us vs cuBLAS 7.8-8.6. All MMA runs after the stream (+1.4/+2.0/+5 us at M=17/34/64), A staging costs 0.4-2 us, the cluster reduction ~1.5 us, smem atomics are CAS loops. The up-front variant is dead; the pipelined variant is H16.'
- id: H4
  statement: mma.sync (~550 TFLOP/s on B200) is enough up to M ~ 64 when overlapped with the B stream; from M ~ 93 upward tcgen05 is required to stay memory-bound.
  status: open
  evidence:
  - 'probe r0: mma.sync m16n8k16 fp16 peaks at 553 TFLOP/s (148-1184 CTAs); 2.5 us of MMA at M=64, 6.5 us at M=172'
  - 'r1 E2: dependent-issue latency is 21 cycles, so chains are not the limit; the 550 TF/s figure is aggregate over 148 SMs, so a balanced pipelined grid could stretch mma.sync to M ~ 128-172 (6.5 us of MMA against a 6.4 us stream). E1 (r2) probes the upper limit.'
- id: H5
  statement: Triton tl.dot GEMMs cannot compete at any M on B200; CUDA C++ (mma.sync / tcgen05 / CUTLASS) is the only route.
  status: supported
  evidence:
  - 'probe r0: Triton 8.9-13 us at M <= 64 vs cuBLAS 7.8-8.0; 309 us vs 199 us at M=16294 (best of 5 configs)'
- id: H6
  statement: 'At M >= 8828 cuBLAS nvjet 128x256 2-CTA kernels are at the same ceiling as CUTLASS; only wave quantisation, tile choice for N=5120 and raster/epilogue overlap can gain 1-3%, and the portal cannot currently show it.'
  status: parked
  evidence:
  - 'probe r0: CuTe DSL dense_gemm_persistent 195.5 us (256x256, 2x1) vs cuBLAS 198.7 us harness'
  - 'r1: CUTLASS 256x256x64 2x1 (6 stages, CLC) = cuBLAS in interleaved repeats (111.6 vs 111.0 at 8828, 199.5 vs 200.3 at 16294); every other tile, raster, cluster and StreamK variant +2-17%'
  - 'portal 62881: the same CUTLASS path +1.1-2.5% vs cuBLAS at M >= 8828 while rented said -1..-3%: the power-capped rented box cannot rank large-M kernels. Tb is 6-9% below our portal cuBLAS time and is a stored constant. Parked by the operator (r2).'
- id: H7
  statement: On the portal's 1500 MHz lock the compute-band workloads slow by ~1.2x (not the full 1.31x clock ratio) for cuBLAS; Tb for large M is 1.16-1.21x the rented reference time, and our cuBLAS call slows by 1.22-1.26x.
  status: supported
  evidence:
  - 'rented box runs 1965 MHz under short GEMM load, 1117 MHz power-capped under sustained load (r1 nvidia-smi)'
  - 'portal r0 page: Tb 130.6-223.1 us at M >= 8828 vs rented reference 108.6-194.9 (1.16-1.21x); ours 140.6-242.6 vs rented 111.2-196.0 (1.22-1.26x)'
- id: H8
  statement: 'Portal Tsol = max(bytes / 6.75 TB/s, FLOPs / 1.81 PFLOP/s): 3.1-3.5 us for M <= 172, 5.7 us at 492, 11.1 us at 952, 102-189 us for M >= 8828. Tb is a stored per-workload constant. With Tb = cuBLAS the small-M ceiling at a 7.0 us portal time is S ~0.59; the compute band has Tsol = 85% of Tb.'
  status: supported
  evidence:
  - 'r0 portal page: the model reproduces all 25 scores to rms 0.0014 (sol.yaml, planner.sol_model)'
  - 'r1: identical Tb on every #38 page and on both #218 pages; the mid-band Tb is uneven per size (0.95-1.11x our cuBLAS time)'
- id: H9
  statement: evict_first on the streamed B loads does not reduce the dirty-L2 write-back cost for a read-dominated 21 MB kernel.
  status: refuted
  evidence:
  - 'probe r0: rd4e 6.46 us vs rd4 6.43 us; skinny k8w8m1e 7.07/7.28 vs 7.07/7.33'
- id: H10
  statement: cuBLAS's small-M nvjet kernels are latency-limited by their small grids (64x8 tiles over N=5120 = 80 CTAs), which is why a 640-CTA stream wins and why 1184-1280 CTAs with cluster split-K may win more.
  status: open
  evidence:
  - 'kernel names from torch.profiler (probe r0); grid sizes inferred, not measured'
  - 'low priority: M <= 16 already scores 0.54-0.55 against a ~0.59 ceiling; r1 E2 showed cluster reductions cost ~1.5 us, so the second half is unlikely'
- id: H11
  statement: For M = 289-952 a persistent tcgen05 kernel with 128-row tiles that keeps B streaming lands near max(memory, compute floor) + ~3 us, i.e. 1.5-2x faster than cuBLAS (11-18 us rented, 13.5-22 us portal).
  status: open
  evidence:
  - 'portal r0: 13.5/14.5/22.0 us vs Tb 13.0/14.1/21.3 at M=289/492/952 (S 0.48-0.49)'
  - 'r1: stock CUTLASS sm100 configs tried (128x128, 128x256, 256x128 2SM, split-K, stream-K) are 1.2-3x slower than cuBLAS at M=93-952; the specific whole-wave tiles of H17 were not tried. Refined into H15 (ingress bound) and H17 (wave quantisation).'
- id: H12
  statement: 'at::matmul_out(C, A, B.t()) runs a slower cuBLAS path than the reference torch.matmul(A, B.T); a direct cublasLt call or the matching torch call closes the 6-9% portal gap at M >= 8828.'
  status: refuted
  evidence:
  - 'r1 (E1): every entry point launches the same nvjet kernel; no faster cublasLt algorithm among the 8 heuristic results; alignment irrelevant; residual ~1% rented = harness zero-fill of the DPS output. The portal gap is the stored Tb.'
- id: H13
  statement: Locking the rented B200 SM clock to 1500 MHz (nvidia-smi -lgc) makes rented compute-band times match the portal within ~3%.
  status: refuted
  evidence:
  - 'Modal B200 refuses nvidia-smi -lgc/-pl/-ac (insufficient permissions). Rented timings stay at ~1965 MHz (or power-capped lower); transfer goes through ours/reference ratios and the emulator.'
- id: H14
  statement: 'cuBLAS''s mid-band picks (M=17-952) are not the best nvjet configurations in the library: the heuristic list (8 algos, workspace 0) is a subset, and a full cublasLt enumeration (every tile id, stage id, split-K with workspace, cluster shape and CTA swizzle that passes cublasLtMatmulAlgoCheck) contains >= 4% rented gains at >= 3 sizes beyond M=25/34/172.'
  status: open
  evidence:
  - 'r1: within the 8 heuristic results, tile 314 -7% (M=25), -8% (M=34), tile 183 -5% (M=172); nothing at the other 9 mid sizes. r1 set MAX_WORKSPACE_BYTES=0 when building plans, so split-K configurations were never offered.'
  - 'card: cuBLAS tile picks give 150 CTAs at M=289 (128x104), 304 CTAs at M=952 (256x136) on 148 SMs, i.e. 2-3 partly filled waves (see H17)'
- id: H15
  statement: 'Per-SM operand ingress (L2 -> shared memory) on B200 is bounded at roughly 64-72 B/clk per SM (~140 GB/s at 1965 MHz; the 21 TB/s aggregate L2 figure / 148). A tcgen05 tile needs >= ~87 FLOP per ingested byte (BM*BN/(BM+BN)) to be compute-bound, so 128x128 (64 FLOP/B) and smaller tiles are ingress-bound at ~70% or less of peak, which is why cuBLAS reaches only 40-50% of peak at M=289-952 even with a single-wave tile (144x128 2cta at M=492: 144 CTAs, 11.8 us vs 4.7 us of MMA). Multicast reduces L2 reads but not SM ingress.'
  status: open
  evidence:
  - 'INFERRED from chipsandcheese L2 bandwidth (21 TB/s local partition) and r1''s CUTLASS 128x128 result at M=289 (13.6 us for 120 CTAs x 1 MB of operands each); no direct measurement yet. E3 step 1 measures it.'
  - 'r1 E2: cp.async A staging of 22-41 MB through L2 cost 0.4-2 us on top of the stream at M=17-64, consistent with a per-SM ingress cost'
- id: H16
  statement: 'A pipelined ring (cp.async or TMA, 3-6 stages of KC=128, B rows plus the A k-chunk per stage, padded 16 B row stride for conflict-free ldmatrix) feeding mma.sync with B as the m16 operand, BN=32 (160 CTAs full-K, or 320 CTAs split-K 2 with a last-CTA-reduces counter and no cluster), keeps the tensor work overlapped with the B stream and reaches <= 7.5 us rented at M=17-64 and <= 8.5 us at M=93-172 (cuBLAS/Lt 7.8-10.2), worth ~+0.03 portal.'
  status: open
  evidence:
  - 'r1 E2 mechanism: compute serialises after an up-front stream; a bounded ring spreads each CTA''s loads over the kernel so chunk j''s MMA overlaps chunk j+1..j+S loads [INFERRED]'
  - 'budget at M=172, BN=32: per SM ~130 KB of B + ~690 KB of A ingress (5.7 us at 140 GB/s) and 6.5 us of mma.sync spread over 148 SMs, both close to the 6.4 us stream: the upper end is marginal (H4)'
- id: H17
  statement: 'At M=289-952 cuBLAS loses mainly to wave quantisation (150, 160 and 304 CTAs on 148 SMs); a tcgen05 tile whose grid is a whole number of waves (M=289: 128x112 1SM -> 138 CTAs; M=492: 256x144 2SM cluster 2x1 -> 144 CTAs; M=952: 256x144 2SM -> 288 CTAs = 2 full waves) built with CUTLASS or the CuTe DSL beats cuBLAS by >= 15% rented at >= 2 of the 3 sizes, subject to the H15 ingress bound.'
  status: open
  evidence:
  - 'card tile names (NxM naming inferred from the M <= 64 picks): 128x104 at 289 -> 3 x 50 = 150 CTAs (2 waves); 144x128 2cta at 492 -> 144 CTAs (1 wave, yet 11.8 us: H15); 256x136 at 952 -> 7 x 20 x 2 = 280-304 CTAs (3 waves, 17.9 us ~ 3 x 5.8 us per CTA)'
  - 'r1 refutation applies only to the tiles tried (128x128, 128x256, 256x128 2SM, split-K, stream-K): none of them is a whole-wave tile at these M'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H16",
    "title": "Pipelined cp.async ring + mma.sync streaming kernel for M = 17-172",
    "operation": "new_design",
    "parents": ["r1b-lt-tiles", "r1-splitk4-cluster-m17-64"],
    "band": "M",
    "instructions": "Build on r1b-lt-tiles (keep its M<=16 skinny path and cublasLt tile table; the new kernel replaces the cuBLAS/Lt path for 17<=M<=MAXM, with MAXM chosen by measurement). Kernel: 128-thread CTAs, BN=32 B-rows per CTA, full K (160 CTAs) and a split-K=2 variant (320 CTAs, no cluster: each CTA writes fp32 partials [M,32] to a module-level scratch buffer (allocated once with at::empty at first use), an atomic counter per n-block elects the last CTA to sum and write fp16 and reset the counter to 0; the counter buffer is zeroed once at module load, never inside run). Ring: STAGES (try 3,4,6) x k-chunk KC=128 (also 64/256); each stage holds the B tile [32 x KC] and the A chunk [M x KC] in smem with a 16-B padded row stride (KC*2+16 B) so ldmatrix is conflict-free; loads via cp.async.cg 16 B per thread (LDGSTS, no registers) with commit_group/wait_group per stage (or a 2-D TMA box with 128B swizzle and host-encoded CUtensorMap if cp.async issue cost shows up). Consume with ldmatrix.x4 (B as the m16 operand: 2 m16 tiles) and ldmatrix.x2 (A as the n8 operand, M padded to 8) into mma.sync m16n8k16 fp32; for M<=32 split K within the chunk across the 4 warps and reduce through smem at the end, for M>32 split the ceil(M/8) n8 tiles across warps (acc <= 2 x 6 x 4 regs at M=172) so registers stay <= 96 and 2 CTAs/SM fit (smem <= 110 KB). Steps: (1) B-only ring (no A, no MMA) must reach ~6.5-7.0 us at every M like r0's read-only probe; if it does not, fix bytes in flight (STAGES x stage bytes x CTAs/SM >= 64 KB per SM) before adding compute. (2) Add the A chunk and record the delta (this is the ingress cost of A: (5120/BN) x M x 4 KB through L2). (3) Add MMA and the epilogue; verify correctness at M=17,25,32,34,63,64,93,128,172 and all other sizes via run_tests (25/25). (4) Tune STAGES/KC/split per M with interleaved repeats (>=3) against the r1b path; record a per-M table. (5) Set MAXM to the largest M where the new kernel wins by >= 4% rented, and submit the resulting dispatcher (r2-ring-mma) to run_tests.",
    "success": "Rented harness time <= 7.5 us at M=17-64 and <= 8.5 us at M=93-172 (vs 7.8-10.2 for the r1b path), 25/25 correct, run_tests predicted portal >= 0.50.",
    "refuted_if": "After tuning STAGES/KC/BN/split the pipelined kernel is not >= 4% faster than the r1b path at any M in 17-172, or the B-only ring itself cannot get under 7.5 us (then the ring, not the overlap, is the limit and H16 is refuted as stated).",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H14",
    "title": "Full cublasLt configuration sweep over all 12 mid sizes (M = 17-952)",
    "operation": "knob_mutation",
    "parents": ["r1b-lt-tiles"],
    "band": "all",
    "instructions": "Write a probe (C++ or ctypes) that, for each M in {17,25,32,34,63,64,93,128,172,289,492,952}, enumerates cublasLt configurations beyond the heuristic: cublasLtMatmulAlgoGetIds for CUDA_R_16F/CUBLAS_COMPUTE_32F, then for each algo id read the caps (TILE_IDS, STAGES_IDS, SPLITK_SUPPORT, REDUCTION_SCHEME_MASK, CTA_SWIZZLING_SUPPORT, CUSTOM_OPTION_MAX, and cluster-shape ids if the 13.1 headers expose them), build every combination with cublasLtMatmulAlgoInit + AlgoConfigSetAttribute (tile, stages, splitK in {1,2,3,4,8}, reduction scheme, swizzle, custom option), keep those that pass cublasLtMatmulAlgoCheck with a 32 MB workspace allowed (allocate the workspace once at module level), and time each with the harness methodology (cold-L2 flush, shifted pointers, CUPTI span over ALL kernels the config launches, since split-K may add a reduction kernel). Prune: first pass 20 calls each, keep the top 10 per M, then >= 3 interleaved repeats against the current r1b pick (default or tile 314/183) and torch.matmul. Accept only configs that are >= 4% faster than the r1b pick in every repeat; prefer single-kernel configs and require the 4% net of any second kernel. Extend lt_tile_for into a per-M config table (tile, stages, splitK, reduction, swizzle, workspace bytes) in binding.cpp/kernel.cu, keeping the fallback to heuristic index 0 when a config is unavailable; run run_tests (25/25) and record the per-M table, including negative results per size, as findings.",
    "success": "At least 3 mid sizes beyond M=25/34/172 gain >= 4% rented over interleaved repeats, and run_tests predicts >= +0.004 over r1b-lt-tiles with 25/25 correct.",
    "refuted_if": "The full enumeration (not just the heuristic list) yields no config >= 4% faster than the current pick at any of the 9 remaining mid sizes; then H14 is refuted and the cuBLAS path is final for the sizes E1/E3 do not cover.",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H17",
    "title": "Exploratory: per-SM ingress bound (H15) and whole-wave tcgen05 tiles for M = 289-952",
    "operation": "new_design",
    "parents": ["r1-cutlass256-lt-tiles"],
    "band": "L",
    "instructions": "Part 1 (probe, H15): measure how fast one SM can ingest L2-resident data. Kernel with grid=148 (1 CTA/SM, then 2 and 4 CTAs/SM), each CTA streaming a 4-8 MB region that is already in L2 (touch it once before timing, event timing not the cold flush) through a 4-8 stage ring of 16-32 KB TMA/cp.async.bulk copies, no compute; report GB/s per SM and aggregate; repeat with cp.async 16 B (LDGSTS) and with a cluster-of-2/4 multicast TMA copy to see whether the limit is L2 read bandwidth (multicast helps) or SM ingress (it does not). Also report the figure for 148 CTAs reading DRAM-cold data for comparison. Part 2 (H17): for M=289/492/952 build CUTLASS sm100 GEMMs with whole-wave tiles: M=289: 128x112x64 1SM (3 x 46 = 138 CTAs) and 128x128 (120 CTAs) with the deepest stage count that fits; M=492: 256x144x64 2SM cluster 2x1 (72 pairs = 144 CTAs) and 128x144 1SM (144 CTAs); M=952: 256x144 2SM (288 CTAs, 2 full waves), 128x160 1SM (256 CTAs) and 256x256 2SM (160 CTAs) for comparison; raster AlongN, no CLC persistence unless it helps, epilogue without a C load. If CUTLASS rejects a tile N (must be a multiple of 16 for 2SM, 8 for 1SM), use the nearest valid one and state the CTA count. Time each with the harness methodology in interleaved repeats (>= 3) against cuBLAS (torch.matmul). Part 3: for every config compute the expected time max(FLOPs/(148 x per-SM peak at 1965 MHz), operand bytes per CTA / measured ingress) x waves + ~1.5 us ramp, and tabulate measured minus expected; this residual tells r3 whether a custom tcgen05 streaming kernel (A-stationary or B-stationary in smem/TMEM with fewer ingested bytes) can beat CUTLASS. If a config beats cuBLAS by >= 15% at a size, add it to r1b-lt-tiles as the path for that M range (289 band: 200 < M <= 400; 492: <= 700; 952: <= 1500), run run_tests (25/25) and record the per-size table and the ingress numbers as findings.",
    "success": "Part 1 yields a per-SM ingress number with its limiting mechanism; Part 2 beats cuBLAS by >= 15% rented at >= 2 of M=289/492/952 (~+0.008 portal) and the dispatcher passes 25/25.",
    "refuted_if": "Whole-wave tiles are within 5% of cuBLAS at all three sizes: then tile/wave choice is not the lever and the measured ingress bound (H15) or something else is; report the residual table so r3 can decide on a custom tcgen05 kernel.",
    "model": "opus"
  }
]
```