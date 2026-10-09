### Assessment

The first portal result (0.402) says the hidden baseline Tb is a fused, cuBLAS-class GEMM: at M = 8192 it runs 86 µs, which is 1.25 PFLOP/s at the lock, about 85% of cuBLAS's practical peak scaled to 1500 MHz. Back-solving Tsol from the 16 scores gives FLOPs / 1.8 PFLOP/s at every compute-bound size (59.8 µs at 8192, 29.9 at 4096, 15.3 at 2048), so SOLAR uses the theoretical 1.5 GHz tensor peak, not 2.25 PFLOP/s. Matching cuBLAS's practical peak at L (about 73 µs at the lock) would score about 0.66 there, so the L band is capped but far from exhausted.

The main loss is not the portal clock. The portal/rented ratio is 1.22 at S, 1.32 at M and 1.37 at L against a clock ratio of 1.31, which is what compute-bound scaling predicts. The loss is that our fused cublasLt call is already 7–15% slower than torch.matmul's plain nvjet kernel on the rented box (77.5 vs 65–72 µs at 8192), so at the portal it lands at eager speed. Recovering plain-nvjet speed with the residual fused is worth roughly +0.06 to +0.08 and would reach the top 5 (0.491).

This round: E1 identifies the baseline (which torch path and kernel names, flat per-size ratio against Tb, and a clock-lock check on the rented box). E2 attacks the cublasLt gap directly: enumerate heuristics per size, attribute the loss to the C≠D path or the algorithm, and submit a per-size forced-algorithm variant. E3 is the exploratory CUTLASS track: a 2SM tile or stream-K that removes the 4.32-wave quantisation, which is the only route above cuBLAS at L.

```yaml ledger
# Hypothesis ledger for #30 (030_attention_output_projection_with_residual). Seeded 2026-10-09; updated r1 2026-10-10.
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
hypotheses:
- id: H1
  statement: Fusing the residual as the GEMM's C operand (C != D) removes the reference's separate add kernel at no cost.
  status: refuted
  evidence:
  - 'run_tests: cublasLt fused 8.7..77.5 us vs reference 12.3..90.2 us (matmul alone 8.2..72 us), 16/16 pass'
  - 'probe: torch.addmm(out=) launches a DtoD memcpy of the residual plus the badd nvjet kernel; same time as the reference'
  - 'r0 rented: fused cublasLt 77.5 us at 8192 vs plain nvjet matmul 65-72 us: the fused call costs 7-15% over the GEMM alone, i.e.
    the add is removed but not for free (see H10)'
  - 'r0 portal: 106 us at 8192 vs Tb 86 and eager ~107 (leaderboard: eager ~25% slower than Tb); on the portal the fused call
    is no faster than matmul+add'
- id: H2
  statement: The L band is tensor-core bound and loses about 13% to wave quantisation (320 pair tiles of 256x256 over 74 SM pairs
    = 4.32 waves); stream-K or a tile count that divides evenly recovers most of it.
  status: open
  evidence:
  - 'probe: cuBLAS 1.64 PFLOP/s at M=8192 vs 1.88 at M=9472 (5 full waves) and 1.93 at M=18944 (rented B200, 1965 MHz)'
  - 'probe: K=2560 vs 5120 vs 10240 at M=8192 gives the same TFLOP/s, so per-tile prologue/epilogue is not the limit'
  - 'r0 portal: Tb at 8192 = 86 us = 1.25 PFLOP/s at the lock = 85% of the practical lock peak (1.47), consistent with a
    cuBLAS-class GEMM that still loses ~14% to quantisation; beating Tb at L needs the quantisation fix (E3)'
- id: H3
  statement: The S band (M <= 1571) is a latency chain of 40 serial K-steps on few tiles; splitting K over all 148 SMs with an
    in-kernel reduction approaches the 6-8 us harness copy floor.
  status: open
  evidence:
  - 'probe: cuBLAS at M=256 takes 4.9/6.2/8.1 us for K=640/1280/2560; a 17 MB copy takes 6.1-8.2 us under harness timing'
  - 'run_tests: CUTLASS 128x256 (20 CTAs) 25 us and 2SM 256x256 (20 clusters) 18.9 us at M=256 vs cublasLt 8.7 us'
  - 'r0 portal: S band t/Tb = 1.07-1.12 (10.6 vs 9.9 us at 256 tokens); a 10% S cut is worth +0.013, so S is the smallest
    lever this round; cublasLt split-K algorithms (E2) are the cheap first test'
- id: H4
  statement: 2-CTA tcgen05 MMA (cta_group::2, 256x256x64 tiles) beats 1SM 128x256x64 at every size.
  status: supported
  evidence:
  - 'run_tests: CUTLASS 2SM 18.9/32.9/48.0/73.7 us vs 1SM 25.1/43.0/61.0/95.0 us at M=256/2048/4096/8192'
- id: H5
  statement: A per-size dispatcher (cublasLt or split-K specialist at S/M, CUTLASS 2SM stream-K at L) beats cublasLt alone.
  status: open
  evidence:
  - 'run_tests: CUTLASS 2SM 256x256 is already 5% faster than cublasLt at M=7976-8192 (73.1-73.7 vs 76.9-77.5 us)'
  - 'r0 portal: L band is worth +0.023 per 10% cut, M +0.018, S +0.013 (predict_score); the dispatcher''s value is at M/L'
- id: H6
  statement: SOLAR's compute term uses the theoretical 1.5 GHz tensor peak of about 1.8 PFLOP/s (not 2.25); the practical lock
    peak is about 1.47 PFLOP/s, so the L-band score is capped near 0.66 even at cuBLAS's best rate, and S/M hold the rest.
  status: supported
  evidence:
  - 'probe: best cuBLAS rate 1.93 PFLOP/s at 1965 MHz with no power throttling (300 W of 1000 W); scales to 1.47 at 1500 MHz'
  - 'r0 portal, Tsol back-solved from (t, Tb, S): 59.8 us at 8192, 29.9 at 4096, 15.3 at 2048 = FLOPs / 1.75-1.80 PFLOP/s;
    at 256 tokens Tsol ~ 1.9-2.1 us (ill-conditioned, matches both mem8 2.1 and 3.36 GFLOP/1.8 PF)'
  - 'inferred: t = 73 us at 8192 (practical lock peak, full waves) would score (86-59.8)/((73-59.8)+(86-59.8)) = 0.66'
- id: H7
  statement: The hidden baseline Tb is close to the reference time (PyTorch-only code cannot fuse the residual into cuBLAS).
  status: refuted
  evidence:
  - 'probe: matmul+add and addmm(out=) are within 3% of each other at every size'
  - 'r0 portal: Tb is 7-23% faster than r0 at every size (9.9 vs 10.6 at 256 tokens; 86.0 vs 106.2 at 8192) and ~25% faster
    than eager on the leaderboard; Tb is a fused GEMM running at cuBLAS speed (see H11)'
- id: H8
  statement: evict_last on the output stores (TMA store cache hint) recovers part of the dirty-L2 flush cost at S/M, as in #38.
  status: open
  evidence:
  - '#38 portal: r5 0.609 vs r3 0.588 (M -8%); untested here, the first kernel is a library call'
- id: H9
  statement: cuBLAS heuristics pick the same algorithms at the portal's locked clock; the portal/rented ratio follows the
    compute/memory split (compute-bound parts slow by 1965/1500 = 1.31x, memory-bound parts hardly at all).
  status: open
  evidence:
  - 'r0 portal vs rented: ratio 1.22 at S (10.6/8.7), 1.32 at M (30.8/23.3), 1.37 at L (106.2/77.5); matches 1.31x compute
    scaling plus ~5% at L; the operator saw ~1.25 for plain cuBLAS GEMMs on another problem'
  - 'open: the extra ~5% at L could be a different algorithm at the lock or the residual read no longer hidden; E1 tries to
    lock the rented clocks to 1500 MHz to measure the ratio directly'
- id: H10
  statement: The fused cublasLt call (beta=1, C != D) picks a worse kernel or pays the residual read unhidden, so it runs 7-15%
    slower than torch.matmul's plain nvjet kernel at M/L; forcing the matmul's algorithm (or another heuristic candidate) with
    the residual as C recovers plain-GEMM speed with the add for free. This is the largest lever (about +0.06 to +0.08).
  status: open
  evidence:
  - 'r0 rented: fused 77.5 us vs matmul alone 65-72 us at 8192, 39.3 vs ~36 at 4096 (card section 5)'
  - 'r0 portal: fused lands at eager speed at L (106 vs ~107) while Tb (a fused GEMM, H11) is 86'
- id: H11
  statement: Tb is torch.compile of the reference (max-autotune or default), which emits one GEMM with the residual fused in the
    epilogue at cuBLAS-class speed; its per-size rented times scale to Tb by a flat ratio of about 1.2-1.4.
  status: open
  evidence:
  - 'r0 portal: Tb/t_r0 is 0.93 at S falling to 0.81 at L, and Tb at L implies 1.25 PFLOP/s at the lock (85% of practical
    peak): only a cuBLAS-level kernel with no separate add gets there; E1 identifies it'
- id: H12
  statement: cublasLt split-K heuristic candidates (SPLITK_NUM > 1, in-kernel or reduce-kernel reduction) beat the default
    algorithm by >= 4% at M <= 1571, a cheap stand-in for a custom split-K kernel (H3).
  status: open
  evidence:
  - 'none yet; E2 enumerates them'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H11",
    "title": "Identify the portal's scoring baseline: time the PyTorch-only candidates under harness timing and match their per-size ratio to Tb",
    "operation": "new_design",
    "parents": ["r0-oproj-cublaslt-fused-v0"],
    "band": "all",
    "instructions": "On the rented B200, under harness-style CUPTI cold-L2 timing (same method as run_tests: L2 flush, shifted pointers, 10 warm-up, 50 timed, median), time these PyTorch-only implementations at all 10 sizes (M = 256, 512, 586, 1024, 1571, 2048, 4096, 4106, 7976, 8192): (a) eager matmul + add (the reference); (b) torch.addmm(residual, x, W.t()); (c) torch.compile(reference) default mode; (d) mode='max-autotune-no-cudagraphs'; (e) mode='max-autotune' (cudagraphs; we cannot use it, but Tb may); (f) torch.compile with torch._inductor.config.max_autotune_gemm_backends='CUTLASS,TRITON,ATEN' if CUTLASS is available in the venv. For each, record with torch.profiler the ordered GPU activity list per size (kernel names such as nvjet_*, triton_*, cutlass_*, memcpy, memset) and the count. Then compute ratio_i = Tb_portal(size) / t_candidate(size) using the Tb column from the r0 portal table (9.9, 12.0, 14.0, 15.8, 22.0, 26.0, 44.2, 44.1, 85.7, 86.0 us). The baseline is the candidate whose ratio is flat across sizes (within +-5%) and sits in 1.2-1.4. Also try to lock the rented clocks (nvidia-smi -lgc 1500,1500 and -lmc 3996 if the box allows it; reset with -rgc / -rmc afterwards and say whether it worked). If locking works, re-time r0 and the winning candidate at the lock and report t_portal / t_rented_locked per size for r0 (this settles H9). Record every table as findings; no portal submission needed.",
    "success": "One candidate has a flat per-size ratio against Tb (spread <= 10% across the 10 sizes) and its kernel names are recorded; H11 is marked supported or refuted with the candidate named. If the clock lock works, r0's locked/portal ratio per size is reported.",
    "refuted_if": "No PyTorch-only candidate shows a flat ratio (spread > 15%), or the flattest candidate is the eager reference: then Tb is something else (a cudagraph-captured or hand-picked library path) and the baseline must be inferred from kernel-level timings instead.",
    "model": "sonnet"
  },
  {
    "id": "E2",
    "hypothesis": "H10",
    "title": "Close the cublasLt-fused vs plain-nvjet gap: attribute the loss, enumerate heuristics per size, submit a per-size forced-algorithm variant",
    "operation": "knob_mutation",
    "parents": ["r0-oproj-cublaslt-fused-v0"],
    "band": "all",
    "instructions": "Step 1, attribution (rented B200, harness-style timing, interleaved repeats, 3 rounds per point): at M = 2048, 4096, 8192 time (i) torch.matmul alone and record its nvjet kernel name, (ii) r0's cublasLt fused call and record its kernel name, (iii) a diagnostic-only cublasLt call with C = D = output (wrong numerics, timing only) to see whether the C != D path or the residual read costs the 5-12 us, (iv) r0 with beta = 0 (no C). Also print cublasLtGetVersion() from the extension and the version of torch's bundled libcublasLt (python -c 'import nvidia.cublas, os; ...' or ldd on torch's libtorch_cuda) and note whether the extension resolves to the toolkit's or torch's library (ldd / /proc/self/maps at import). Step 2, enumeration: for each of the 10 sizes request up to 32 heuristic results (cublasLtMatmulAlgoGetHeuristic with the fused layouts, 32 MB workspace), read per algo: CUBLASLT_ALGO_CONFIG_ID, TILE_ID, STAGES_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CLUSTER_SHAPE_ID, and CUBLASLT_ALGO_CAP_SPLITK_SUPPORT; time every candidate with interleaved repeats (at least 3 rounds of the full list, median per candidate) and count its GPU activities with the profiler (workspace memsets or reduce kernels count). For algos that support split-K, additionally try SPLITK_NUM in {2, 4, 8} with CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE (fp32) at M <= 1571 (H12). Step 3, build: a shape-keyed table (keyed by M only) that forces an algorithm where it beats heuristic-0 by >= 4% in the interleaved repeats, else the heuristic; the forced algo config must be validated with cublasLtMatmulAlgoCheck at run time and must give the same activity count every call. run_tests 16/16, then submit to the portal (one slot). Record per-size tables as findings, including the chosen kernel names.",
    "success": "Rented B200: <= 71 us at M = 8192 and <= 36 us at M = 4096 (>= 8% below r0's 77.5 / 39.3) with 16/16 correct, and the portal score >= 0.43. Secondary: the attribution step names which of (algorithm choice, C != D path, residual read) explains the gap.",
    "refuted_if": "No heuristic candidate beats heuristic-0 by >= 4% at any M >= 2048 and the C = D diagnostic runs at the same speed as C != D: the loss is intrinsic to cublasLt's fused path and plain-nvjet speed with a fused residual needs a CUTLASS kernel (E3 becomes the only L/M route).",
    "model": "opus"
  },
  {
    "id": "E3",
    "hypothesis": "H2",
    "title": "Exploratory: CUTLASS 2SM GEMM with a tiling or stream-K schedule that removes the 4.32-wave quantisation at L, behind a size dispatcher",
    "operation": "structural_mutation",
    "parents": ["r0-oproj-cublaslt-fused-v0"],
    "band": "L",
    "instructions": "Start from the r0 session's CUTLASS 4.4.1 sm100 2SM 256x256x64 cluster 2x1 kernel with the fused TMA-loaded C (residual) epilogue (73.7 us at 8192 rented, 16/16). Build and time, interleaved, at M = 4096, 4106, 7976, 8192 (plus 2048 as a check): (a) 2SM tile 256x128x64 (640 tiles over 74 pairs = 8.65 waves, 96% use); (b) 2SM 256x160x64 if the builder accepts N = 160 (512 tiles = 6.9 waves); (c) the same 256x256 tile with the StreamKScheduler (KernelTmaWarpSpecialized2SmSm100 + StreamKScheduler, workspace from at::empty); record with the profiler whether it adds a memset or reduce kernel inside the window and how much it costs; (d) raster order / swizzle variants (AlongM vs AlongN, max_swizzle_size) on the best of (a)-(c). Diagnostics to add: a cycle count of the mainloop vs epilogue per tile if easy (or the K = 5120 trick: if doubling K less than doubles the time, the epilogue is not the limit). Then try the epilogue TMA-store cache hint evict_last on D (H8) as one A/B on the best variant. Build a dispatcher: cublasLt (r0 path, or E2's table if it lands first) for M < threshold and the best CUTLASS kernel above it, threshold chosen from the interleaved measurements; run_tests 16/16; submit to the portal (one slot). Use run_tests, not the compile tool, for the bf16 lint issue, and flag the whitelist need to the operator in the card. Record all per-variant times and activity counts as findings, including any variant that fails to compile and why.",
    "success": "Rented B200: <= 66 us at M = 8192 (>= 10% below the 2SM 256x256 kernel's 73.7, >= 15% below cublasLt's 77.5) with 16/16 correct and the same activity sequence every call; the dispatcher scores >= 0.43 on the portal with L-band workloads >= 0.42 each.",
    "refuted_if": "No tiling or stream-K variant gets below 70 us at 8192: wave quantisation is not the L-band loss and the mainloop (stage count, 2SM MMA issue rate, TMA multicast) must be examined instead; park H2 and record the per-variant rates in PFLOP/s.",
    "model": "opus"
  }
]
```