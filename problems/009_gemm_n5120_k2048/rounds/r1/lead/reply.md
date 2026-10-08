### Assessment

Where the score is: r0 sits at 0.487. The six compute-band workloads score 0.39–0.42 and carry the most leverage (a 10% cut in the loop's L band is worth +0.04; the portal shows our cuBLAS call 7–9% slower than Tb there, so matching Tb alone is worth about +0.025). The 12 mid workloads (M 17–952) score 0.46–0.52; taking M 17–172 to the ~7 µs rented read floor (about 7.8 µs portal) gives S ≈ 0.56 each (+0.03 total). M ≤ 16 is within 10% of its practical floor and stays low priority.

What we now know from the portal page: Tb is the reference timed at 1500 MHz (1.16–1.21× rented at large M), Tsol follows `max(bytes/6.75 TB/s, FLOPs/1.81 PF)` (3.1–3.5 µs at small M, not 2.6), and the compute band's Tb is 84% of the 1.81 PF peak, so cuBLAS runs more efficiently at the locked clock than on the 1965 MHz box (74%). Beating Tb by 3% means ≥ 87% of peak at 1500 MHz. Mid-size Tb is ±5% noisy, so sessions must judge on rented ours/reference ratios.

This round: E1 finds why `at::matmul_out` is slower than the reference and fixes the cheap cuBLAS path, with the rented box clock-locked to 1500 MHz so the bench is representative of the compute band. E2 builds the skinny-band kernel (BN 32–64, A staged in shared memory, split-K across a cluster or last-CTA reduction) for M 17–172, with freedom to explore the reduction mechanism. E3 attempts a CUTLASS/CuTe 2-CTA tcgen05 GEMM with tiles chosen for N = 5120 and a persistent scheduler, covering M ≥ 289 (9 workloads), aiming to beat cuBLAS by 3% at 1500 MHz and 1.5× at M 289–952.

```yaml ledger
# Hypothesis ledger for FlashInfer-Bench/009_gemm_n5120_k2048, seeded 2026-10-08 by the research phase (r0); updated r1 (2026-10-08).
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
- 'Compute-band candidates are judged on the rented box with the SM clock locked to 1500 MHz where possible (E1 checks
  whether nvidia-smi -lgc works there); unlocked 1965 MHz numbers overstate cuBLAS''s gap to peak.'
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
  - 'probe r0: 8 warps x 4 loads/lane best; 4 warps 8.9 us, 16 warps 7.7-8.4 us, BN=16 7.5-7.7 us; tail/imbalance split not yet measured'
- id: H3
  statement: For 17 <= M <= 172 the winning structure is BN = 32-64 B-rows per CTA with A staged once per k-chunk in shared memory and split-K across a 2-4 CTA cluster (DSMEM reduction) or a last-CTA-reduces counter, reaching ~7-7.5 us rented (~7.8-8.3 portal, S ~0.56); re-reading A per warp from L2 does not work.
  status: open
  evidence:
  - 'probe r0: per-warp A re-read (640 CTAs) 10.8 us at M=17, 19.5 us at M=64, 33.7 us at M=128 (A L2 traffic = 640 x M x 4 KB)'
  - 'portal r0: cuBLAS path 9.1-12.2 us at M=17-172 vs Tb 8.9-11.5 (S 0.46-0.52); the 9 workloads are worth ~+0.03 at 7.8 us portal'
- id: H4
  statement: mma.sync (~550 TFLOP/s on B200) is enough up to M ~ 64 when overlapped with the B stream; from M ~ 93 upward tcgen05 is required to stay memory-bound.
  status: open
  evidence:
  - 'probe r0: mma.sync m16n8k16 fp16 peaks at 553 TFLOP/s (148-1184 CTAs); 2.5 us of MMA at M=64, 6.5 us at M=172'
- id: H5
  statement: Triton tl.dot GEMMs cannot compete at any M on B200; CUDA C++ (mma.sync / tcgen05 / CUTLASS) is the only route.
  status: supported
  evidence:
  - 'probe r0: Triton 8.9-13 us at M <= 64 vs cuBLAS 7.8-8.0; 309 us vs 199 us at M=16294 (best of 5 configs)'
- id: H6
  statement: 'At M >= 8828 cuBLAS nvjet 128x256 2-CTA kernels are at the same ceiling as CUTLASS (CuTe DSL persistent 256x256 2cta: 195 vs 199 us at M=16294); only wave-quantization, tile choice for N=5120 (256x160 or 256x128 tiles give 98-99% wave efficiency vs 96% for 256x256) and raster/epilogue overlap can gain 1-3%.'
  status: open
  evidence:
  - 'probe r0: CuTe DSL dense_gemm_persistent 195.5 us (256x256, 2x1), 216.7 us (128x256), 252.7 us (128x128 no 2cta); cuBLAS 198.7 us harness'
  - 'portal r0: Tb at M=16294 is 223.1 us = 84% of the 1.81 PF peak at 1500 MHz, so a 3% win needs >= 87% of peak; untested whether CUTLASS reaches that at the locked clock'
- id: H7
  statement: On the portal's 1500 MHz lock the compute-band workloads slow by ~1.2x (not the full 1.31x clock ratio) for cuBLAS; Tb for large M is 1.16-1.21x the rented reference time, and our cuBLAS call slows by 1.22-1.26x.
  status: supported
  evidence:
  - 'rented box runs 1965 MHz under GEMM load; cuBLAS 1.75 PFLOP/s = 73% of 2.38 PF at that clock (probe r0)'
  - 'portal r0 page: Tb 130.6-223.1 us at M >= 8828 vs rented reference 108.6-194.9 (1.16-1.21x); ours 140.6-242.6 vs rented 111.2-196.0 (1.22-1.26x)'
- id: H8
  statement: 'Portal Tsol = max(bytes / 6.75 TB/s, FLOPs / 1.81 PFLOP/s): 3.1-3.5 us for M <= 172, 5.7 us at 492, 11.1 us at 952, 102-189 us for M >= 8828. With Tb = cuBLAS the small-M ceiling at a 7.0 us portal time is S ~0.59; the compute band has Tsol = 85% of Tb, so each 1% of time there is worth 0.02-0.03 of S.'
  status: supported
  evidence:
  - 'r0 portal page: the model reproduces all 25 scores to rms 0.0014 (sol.yaml, planner.sol_model)'
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
  - 'low priority after r0: M <= 16 already scores 0.54-0.55 against a ~0.59 ceiling'
- id: H11
  statement: For M = 289-952 a persistent tcgen05 kernel with 128-row tiles that keeps B streaming lands near max(memory, compute floor) + ~3 us, i.e. 1.5-2x faster than cuBLAS (11-18 us rented, 13.5-22 us portal).
  status: open
  evidence:
  - 'floors from the card; cuBLAS tiles (128x104, 144x128, 256x136) give one under-filled wave'
  - 'portal r0: 13.5/14.5/22.0 us vs Tb 13.0/14.1/21.3 at M=289/492/952 (S 0.48-0.49)'
- id: H12
  statement: 'at::matmul_out(C, A, B.t()) runs a slower cuBLAS path than the reference torch.matmul(A, B.T) (3% rented, 6-9% portal at M >= 8828) because of the caller-supplied output (different nvjet heuristic pick, 256-B-shifted C alignment, workspace or Lt-vs-legacy path); a direct cublasLt call with a chosen algorithm and workspace, or the matching torch call, closes the gap. Alternative: Tb is not plain torch.matmul but a faster PyTorch-only solution, in which case only our own kernel can match it.'
  status: open
  evidence:
  - 'card section 7: same library, 3% slower rented (0.6-2.4% per size in the table), 6-9% slower on the portal; kernel names of the two calls not yet compared'
- id: H13
  statement: Locking the rented B200 SM clock to 1500 MHz (nvidia-smi -lgc) makes rented compute-band times match the portal within ~3% and makes cuBLAS's efficiency rise to ~84% of peak, so tile/raster gains measured at 1500 MHz transfer to the portal.
  status: open
  evidence:
  - 'portal/rented ratio 1.22-1.26 for our cuBLAS call vs clock ratio 1.31 (r0 page); whether the rented box allows clock locking is unknown'
```

```json tasks
[{"id": "E1", "hypothesis": "H12", "title": "Why is at::matmul_out slower than the reference torch.matmul at large M, and fix the cuBLAS path", "operation": "repair", "parents": ["r0-skinny16-cublas"],
  "band": "L", "instructions": "1) Try to lock the rented B200 clocks like the portal (nvidia-smi -lgc 1500,1500 and -lmc 3996 or the applications-clock equivalent); record whether it works (H13). All later timings in this session at the locked clock if possible, else at 1965 MHz with both numbers reported. 2) Under torch.profiler (CUPTI) at M=8828, 12251 and 16294, and at M=172, 492, 952, record kernel names, grid/cluster sizes and times for: (a) the reference torch.matmul(A, B.T) returning a new tensor; (b) torch.matmul(A, B.T, out=C) and at::matmul_out / at::mm_out from C++ into a harness-style C (shifted by 256 B, 512 B, 1 KB, 2 KB offsets); (c) torch.nn.functional.linear(A, B); (d) the same with CUBLAS_WORKSPACE_CONFIG unset vs :32768:8 (only as a diagnostic, never set from the solution); (e) a direct cublasLtMatmul from binding.cpp with cublasLtMatmulAlgoGetHeuristic (top 8 algos, 32 MB workspace allocated with torch::empty inside run()) and a direct cublasGemmEx. 3) Run the harness (run_tests) on the reference itself at the locked clock and compare with the portal Tb column (130.6-223.1 us): if the reference at 1500 MHz matches Tb within 3%, the gap is in our call; if it is >5% slower than Tb, Tb is a different solution and note that. 4) Build the fastest legitimate cuBLAS/cublasLt path found into r0's binding (one kernel per call, no cross-call state beyond the cached handle/preference; the algo choice keyed only on M) and run_tests all 25 workloads x 10 rounds. Record every finding with numbers.", "success": "The large-M path reaches the reference's rented time within 1% (ratio ours/reference <= 1.01 at M >= 8828, interleaved repeats), 25/25 pass, and the cause is named with profiler evidence.", "refuted_if": "Every cuBLAS/cublasLt entry point launches the same nvjet kernel with the same time as at::matmul_out, and the reference at 1500 MHz is also 5%+ slower than Tb: then Tb is not plain cuBLAS and H12 is refuted (the gap must be closed by our own kernel, E3).", "model": "opus"},
 {"id": "E2", "hypothesis": "H3", "title": "Skinny-band kernel for M = 17-172: BN 32-64, A staged in shared memory, split-K reduction", "operation": "new_design", "parents": ["r0-skinny16-cublas"],
  "band": "M", "instructions": "Build a CUDA C++ kernel for 17 <= M <= 172 that keeps r0's structure (8 warps x 4 x 32 B 256-bit loads of B per lane into mma.sync fragments, all B loads issued before any compute) but with each CTA owning BN=32 (then try 64) rows of B over a K slice of 2048/SPLIT (SPLIT = 2, 4), so the grid is 160*SPLIT to 640 CTAs. Stage the A slice [M, K/SPLIT] once per CTA in shared memory via cp.async or TMA 1-D bulk (M x 1 KB at SPLIT=4: up to 172 KB, so for M > 64 use a k-chunked ring or SPLIT=8), and feed mma.sync A fragments from smem (ldmatrix). Reduce the fp32 [M, BN] partials across the SPLIT CTAs with (i) a cluster of SPLIT CTAs and DSMEM (cluster <= 4), and (ii) as the exploratory alternative, a global fp32 scratch (torch::empty in run, no fill kernel) plus a per-tile counter where the last-arriving CTA sums and writes fp16 and resets the counter so no state leaks between calls. Probe first: for M=17, 64, 128, 172 measure (a) the B-stream-only version (no MMA) to see the floor for each grid shape, (b) with MMA, (c) with the reduction. Dispatch by M in binding.cpp: r0 skinny for M <= 16, the new kernel for 17-172 (or wherever it beats cuBLAS), cuBLAS above. run_tests on all 25 workloads x 10 rounds with interleaved repeats against the reference.", "success": "Rented times <= 7.5 us at M = 17-64 and <= 8.5 us at M = 93-172 (ours/reference <= 0.85 at every one of the 9 sizes), 25/25 pass; run_tests predicted score >= 0.51.", "refuted_if": "With A in smem and split-K the best variant is still >= 0.95x cuBLAS at M = 63-172 or the reduction costs >= 1 us (then H3 is refuted for mma.sync and the band needs the tcgen05 kernel from E3 instead).", "model": "opus"},
 {"id": "E3", "hypothesis": "H6", "title": "CUTLASS/CuTe DSL 2-CTA tcgen05 GEMM tuned for N = 5120, covering M >= 289 (and H11 at 289-952)", "operation": "new_design", "parents": ["r0-skinny16-cublas"],
  "band": "L", "instructions": "Lock the clock to 1500 MHz if E1's method works (otherwise report both clocks). Starting from the CuTe DSL dense_gemm_persistent example (or CUTLASS 4.4 C++ sm100 KernelTmaWarpSpecialized2SmSm100 collective with a persistent tile scheduler), build a fp16-in, fp32-accumulate, fp16-out TN GEMM compiled once with dynamic M. Sweep for M = 8828, 12251, 16294: tile 256x256, 256x160, 256x128 (N = 5120 = 20x256 = 32x160 = 40x128), cluster 2x1 and 2x2, K-block 64 vs 128, pipeline stages to fill 227 KB, raster order (along M vs along N) and swizzle size, and epilogue: TMA store vs direct. Report wave efficiency (tiles / (148 x waves)) and achieved PFLOP/s against the 1.82 PF peak at 1500 MHz and against the reference measured in the same session. Then try 128x128 and 128x256 1-CTA/2-CTA tiles for M = 289, 492, 952 and compare to cuBLAS (11.2/11.8/17.9 us rented at 1965 MHz) and to the floors max(memory, compute). Keep compile time under 60 s per instantiation (<= 3 instantiations); descriptors built on the host per call; dispatch by M in run(); cuBLAS stays for any M range where it is faster. run_tests on all 25 workloads.", "success": "At M >= 8828 ours/reference <= 0.97 at the locked clock (or <= 0.98 at 1965 MHz) on interleaved repeats, 25/25 pass; or at M = 289-952 ours/reference <= 0.7 at all three sizes. Either outcome is a win; both together are the target (predicted score >= 0.53).", "refuted_if": "No tile/cluster/raster combination beats the reference by more than 1% at any of the three large sizes at the locked clock, and the 128-row tiles at 289-952 are not >= 1.3x faster than cuBLAS: then H6's ceiling holds, and only E1's path fix is available in the compute band.", "model": "opus"}]
```