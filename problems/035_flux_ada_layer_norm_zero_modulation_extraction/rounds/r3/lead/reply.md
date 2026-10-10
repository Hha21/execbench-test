### Assessment

The remaining score sits in B = 192 to 512. Six workloads score 0.55 to 0.64 while B ≤ 131 already scores about 0.70 and the L band is pinned at cuBLAS's 65 to 68% TF32 efficiency. The r2 portal result settled two things: evict_first streaming works and scales 1.03 to 1.08 portal/rented when DRAM-bound, and everything from B = 192 up is SM-clock-bound on the portal (ratio 1.21 to 1.28), so rented microseconds overstate gains there. The e1 one-SM kernel at B = 192 runs at about 40% of the TF32 peak, and the CUTLASS 2SM tiles at 373 to 512 run at 52 to 60% with padding and a near-4-wave quantisation.

The score model says a 28% cut in M plus 14% in L is worth about +0.054 (0.665). A kernel at 75% efficiency everywhere lands near 0.68, the top-5 bar is 0.691.

This round builds the one design that attacks all three suspected causes at once, and two diagnostics that separate them:

- E1 builds a hand-written 2SM tcgen05 kernel where one CTA pair owns 256 W rows and the whole batch (N = Bpad ≤ 512 in up to two MMAs). W streams once from DRAM with evict_first, there is no padding beyond 16 and no wave quantisation, and emb shared memory per CTA halves so B = 192/211 get deeper rings.
- E2 microbenchmarks the UMMA rate itself: 1SM vs 2SM from smem-resident operands, then with the TMA feed, then the ring-depth slope, all in cycles.
- E3 tests the emb L2 re-read cause alone by multicasting emb across a 2-CTA cluster in the otherwise unchanged e1 kernel.

Every session must report cycles and the measured rented clock, since the card drifts between 1350 and 1965 MHz.

```yaml ledger
# Hypothesis ledger for #35 (035_flux_ada_layer_norm_zero_modulation_extraction). Seeded 2026-10-10 (research round r0); updated r1 2026-10-10; updated r2 2026-10-10; updated r3 2026-10-10.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'bf16 is this problem''s own dtype: write cutlass::bfloat16_t / __nv_bfloat16 directly. Never build type names with the ## token-paste operator (the lint rejects it: it reads as lint evasion to a reviewer).'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells, TMA multicast of a GEMM operand); inherited from #38: clustered row-wise grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot (r13, H21).'
- 'Any cublasLt algorithm forced per size must be chosen from shape only, verified 16/16 on 10 random rounds, and must launch the same activity sequence every call (split-K with a reduce kernel is allowed only if the count is constant per shape).'
- 'The loop''s lint rejects the literal bf16 type name in CUTLASS sources; CUTLASS candidates go through run_tests and the operator whitelists the type name for this problem (dtype is bf16 by definition).'
- 'Portal A/B packaging (r2): a candidate may dispatch by (batch_size, seq_len) among same-M shapes (4096 group: 4,1024 / 16,256 / 8,512 plus 2,2053; 8192 group: 16,512 / 8,1024 / 64,128 / 32,256). Shape-only dispatch is legitimate, but every variant must be a real candidate (16/16 correct, one launch); no diagnostic or deliberately wrong variant. The card must list which shape runs which variant and all 10 rented per-size times per variant, measured interleaved in one probe.'
- 'Rented differences under about 4% between fused variants are unproven at the portal (fused kernels scale 1.36-1.40 portal/rented, plain GEMMs 1.22); interleave variants in the same probe, 3 rounds, and only promote gains of 5% or more.'
- '#35 precision (operator ruling, 2026-10-10): plain TF32 tensor-core compute with fp32 accumulation is allowed (CUBLAS_COMPUTE_32F_FAST_TF32, tcgen05 kind::tf32, allow_tf32 inside our own GEMM call); inputs and outputs stay fp32 and no intermediate is stored below fp32. bf16/fp16 operands and BF16x9 emulation remain forbidden. TF32 max error 1.6e-3 vs atol 2e-3: verify every candidate on all 16 workloads x 10 rounds and never round outputs.'
- '#35 dtype is fp32 throughout; the bf16 constraint above is inherited text from #30 and does not apply here.'
- '#35 TF32 margin (operator, r1): accumulate in fp32 over the full K = 3072 inside one tensor-core accumulator; no split-K whose partials are stored or re-fed below fp32; no rounding of outputs or intermediates. Every candidate records max |err| vs fp64 per B (must stay <= 1.7e-3 against atol 2.0e-3) in its findings. TMA tensor maps must use CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 (round on load); FLOAT32 truncates and fails (r2).'
- '#35 cache hints (r2): L2 evict_first is used only on the TMA/bulk loads of this call''s own weight buffer and evict_last/evict_first only on this call''s outputs; no applypriority, no persisting-L2 windows. Record the hint in the card''s paths (w: ef) so the emulator can learn the portal ratio of hinted paths.'
- '#35 portal ratios (r2, H15): emulator predictions for tensor-core-bound paths must use 1.21-1.31 portal/rented; DRAM-bound paths 1.03-1.08. The r1 emulator overpredicted L (0.603 predicted vs 0.5884 portal) by ignoring this.'
- '#35 measurement (r3, operator): judge B >= 192 designs by SM cycles (clock64 in-kernel, with %globaltimer to recover the rented SM clock, which drifts 1350-1965 MHz between runs) or by tensor-pipe efficiency per clock, not by rented microseconds alone. Predict the portal time as max(1.05 x rented DRAM-bound time, rented compute cycles / ~1.55 GHz). Every probe records the clock it ran at.'
- '#35 dispatch (r3, operator): the r2 B <= 131 path (tc_small, evict_first, 144 one-SM CTAs) stays unchanged; new work targets 131 < B <= 512 first, then L.'
hypotheses:
- id: H1
  statement: The reference's cuBLAS SIMT sgemm is within 25-30% of the FFMA peak at M/L, so FFMA-only designs cannot gain more than about 1.3x there; packed fma.f32x2 adds no throughput.
  status: supported
  evidence:
  - 'probe: cutlass3x_sm100_simt_sgemm 55-57 TF/s at B=384/919 vs 74.4 TF/s peak at 1965 MHz [probe_b200]'
  - 'probe: FFMA 67.6 TF/s vs FFMA2 (fma.rn.f32x2) 73.5 TF/s on 148x8 CTAs: same 128 FMA/clk/SM datapath [probe_b200]'
  - 'r1: moot since the TF32 ruling and Tb = TF32 speed; FFMA paths cannot score anywhere (B=5 skinny 76.1 us portal vs Tb 82.8)'
- id: H2
  statement: 3xTF32 (hi/lo split, three TF32 tensor-core GEMMs, fp32 accumulate) is fp32-level accurate and 3x faster than the reference at L even with a separate split pass and scatter.
  status: parked
  evidence:
  - 'probe: max error vs fp64 4.3e-5 (fp32 cuBLAS 1.6e-5, tolerance atol 2e-3) at B=919; gemms 415 us + split 112 + scatter [probe_b200]'
  - 'run_tests r0: 575 us at 919 vs 1882 reference, 345 at 384 vs 762, 16/16 pass on 10 random rounds [r0-adamod-dispatch-skinny-fp32-3xtf32]'
  - 'portal r0: 688.9 us at 919 scores 0.151 against Tb 169.8 (TF32 speed); retired by the 2026-10-10 TF32 ruling'
- id: H3
  statement: A fused tcgen05 3xTF32 kernel (TMA load, hi/lo split in shared memory, 3 MMAs per k-block, bias + six-way split in the epilogue, one launch) removes the 107-112 us split pass and the 7-47 us scatter and reaches max(3 x FLOPs / 588 TF/s, memory floor).
  status: parked
  evidence:
  - 'estimate: 531 us at 919 at the lock vs r0 575 rented (~690 at the lock); 76-222 us at M vs 272-345; ~55-75 us at S vs 256'
  - 'r1: superseded by plain TF32 (ruling); 3 MMAs per k-block can never reach Tb (169.8 us at 919 = 613 TF/s). The fused-epilogue part lives on as H14'
- id: H4
  statement: For B <= 128 one M-tile covers all rows, so a tensor-core TF32 kernel streams W exactly once and the S band is memory-bound; the S floor is the evict_first reads-only stream (39-40 us rented, H16), not the 50 us plain stream cuBLAS sits at.
  status: supported
  evidence:
  - 'probe: plain cuBLAS TF32 runs 50-51 us at B=5..128 rented, i.e. at the dirty-L2 plain stream floor [probe_b200]'
  - 'portal r0: Tb 82.8-90.7 us at B<=128, Tsol about 15 us; t = 50 us scores 0.66, t = 40 us 0.73'
  - 'portal r1: CUTLASS 128x64/256x64 at 57.0/50.4/50.5/59.5/59.9 us (B=5..128) score 0.62-0.66; r1-ef mma.sync stream kernel 41.6/43.6 rented at B=5/16 shows the hinted floor is reachable with a GEMM attached'
  - 'portal r2: tc_small 44.1/44.1/44.5/45.2/47.1/47.8 us at B=5..131 (scores 0.70-0.72), ratio 1.03-1.08 vs rented: DRAM-bound as predicted [r2-e1-tc05-ef-small]'
- id: H5
  statement: The skinny SIMT kernel (lane = W row, cp.async-staged 128x64 tiles, emb broadcast from smem) is at the copy floor at B=5 but issue/latency bound at B=16 and 32; 2 CTAs/SM and 2-row-per-lane register tiling recover the 32-64 us FFMA floor.
  status: parked
  evidence:
  - 'probe: 54.6 / 88.8 / 145.4 us at B=5/16/32 (1 CTA/SM, 144 CTAs, 108-129 KB smem); run_tests 60.6 / 88.3 [probe_b200, r0]'
  - 'dead end: skinny<32> loses to cuBLASLt fp32 (122 us) so B=32 stays on the library path in r0'
  - 'portal r0: 76.1 us at B=5 (x1.26 vs rented: it scales with the SM clock, so it is issue-bound, not DRAM-bound); cuBLAS TF32 at 50 us beats it. Parked'
- id: H6
  statement: cuBLAS BF16x9 emulation is a legitimate one-call fp32-level path (error 3.4e-6) but is slower than 3xTF32 at L and falls back to SIMT below B~64.
  status: refuted
  evidence:
  - 'probe: 117 / 131 / 177 / 358 / 754 us at B=5/32/128/384/919, maxerr 3.4e-6; r0 3xTF32 path 575 at 919 [probe_b200]'
  - 'operator ruling 2026-10-10: BF16x9 emulation stays forbidden; and it is 4x slower than Tb at 919 anyway'
- id: H7
  statement: SOLAR's Tsol for this problem uses the FP32 CUDA-core peak, so 3xTF32 designs can go below Tsol at M/L (scores above 0.5 come cheaply there and the S band decides the ranking).
  status: refuted
  evidence:
  - 'portal r0: Tsol = (Tb - S t - S Tb)/(1 - 2S) gives 15.0 us at B=5, 23.7 at 373, 57.9 at 919 = FLOPs / 1.8 PF/s (bf16-like dense rate, 2x the TF32 peak), floored at about 15 us for B <= 211'
  - 'consequence: no TF32 kernel can reach Tsol; the lock-time TF32 peak (909 TF/s, 115 us at 919) scores at most 0.66 there'
  - 'r2: the Tsol model reproduces the r2 portal score 0.610 from the per-workload times; 75% of TF32 peak at every B would score about 0.68, 85% about 0.71 [operator brief r3]'
- id: H8
  statement: The hidden Tb is close to the fp32 reference (~2.4 ms at 919 at the lock), not a TF32-enabled PyTorch solution (~180 us).
  status: refuted
  evidence:
  - 'portal r0: Tb = 82.8 / 83.1 / 84.8 / 90.1 / 90.7 (B=5..128), 93.8-110.5 (M), 119.1-169.8 (L) us; 169.8 at 919 = 613 TF/s: a TF32 PyTorch matmul at the lock, slightly faster than our rented cuBLAS TF32 scaled by 1.31 (177 us)'
- id: H9
  statement: The separate scatter and any tmp[B,18432] round trip cost 5 us at B=5 and 22 us at 919 (3-13% of Tb, not the 28% first estimated); a fused epilogue that adds the bias and picks the output tensor by column chunk gains that much and no more before the GEMM itself improves.
  status: supported
  evidence:
  - 'probe r1-e1: GEMM alone 49.5 vs 55.1 with scatter at B=5; 135 vs 157 at 919. r0''s 47 us figure was inflated by the 453 MB scratch traffic [probe_b200]'
  - 'portal r1: fused CUTLASS 167.9 us at 919 vs Tb 169.8 (cuBLAS TF32 + bias pass): the fused epilogue is worth about the scatter and the GEMM rate is identical to Tb''s'
- id: H10
  statement: 'L2 evict_first on the W stream removes most of the dirty-L2 write-back penalty from the timed window (reads-only 45.9 -> 39.9 us plain LDG, 45.0 -> 38.7 bulk ring); store hints on the outputs are not a lever (<= 0.5 us).'
  status: supported
  evidence:
  - 'probe r1-ef: with the harness zero-flush, plain LDG.128 45.9 us, evict_first 39.9; bulk ring 45.0, bulk + evict_first 38.7; ld.global.v8 256-bit 41.5 (no gain) [probe_b200]'
  - 'probe r1-ef: read-based flush gives 38.9 us for every variant; so the dirty write-back costs 6-12 us for plain reads and 4-5 us with evict_first [probe_b200]'
  - 'probe r1-ef: reads + six outputs at B=128: st default 40.3, evict_last 39.8, evict_first 40.3 us [probe_b200]'
  - 'probe r2-e1: evict_first on the W TMA loads saves 6-9 us at every B <= 256 inside one container (45.5 vs 51.7 at B=5, 48.1 vs 57.0 at 211); the saving varied 1.5-9 us between containers [r2-e1-tc05-ef-small]'
  - 'portal r2: the hinted path landed at 44.1-47.8 us for B <= 131 vs 50.4-59.9 unhinted CUTLASS in r1 (-12..-24%) [r2-e1-tc05-ef-small]'
- id: H11
  statement: cuBLASLt CUBLAS_COMPUTE_32F_FAST_TF32 with a bias epilogue plus a scatter is a 16/16-correct path at 55-177 us rented (score about 0.54); cuBLAS 13's batched pointer-array mode over the six chunks removes the scatter at no GEMM cost.
  status: refuted
  evidence:
  - 'run_tests r1-e1: 55.5/54.1/54.8/59.5/60.4 (S), 62.7-94.5 (M), 99.3-177.0 (L) us, 16/16, emulator 0.537 [r1-e1-lt-fasttf32-bias-scatter]'
  - 'refuted part: pointer-array batched GEMM (bias via C broadcast, pinned pointer arrays) is 12-43% slower than big GEMM + scatter at every B; bias epilogue + pointer array is NOT_SUPPORTED; the pointer arrays must change per call and reach the GPU over PCIe [probe_b200]'
  - 'superseded: the CUTLASS fused epilogue (H14) is faster at every B and needs no library heuristics'
- id: H12
  statement: 'tcgen05 TF32 stops at about 68% of the 909 TF/s lock-time peak (cuBLAS 613 TF/s = Tb; our 2SM 256x192/256x256 the same) because of operand traffic through shared memory: a 2SM 256x256x8 MMA needs 8 KB of TMA writes and 12 KB of MMA reads per CTA per 128 clk (160 B/clk against 128 B/clk), a hardware ceiling no tile beats. The alternative is that the loss is shape-specific (72 W tiles, 1-5 batch tiles, K loop of 96 stages, W from DRAM) and a square hot GEMM shows >= 85%.'
  status: refuted
  evidence:
  - 'probe r0: cuBLAS TF32 770 TF/s unlocked at 919 = 65-67% of the 1.19 PF/s TF32 peak at 1965 MHz [probe_b200]'
  - 'probe r1: swapped 2SM 256x256 142 us, 256x192 135.5, 1SM 128x256 161 at 919: 2SM is 1.13-1.19x 1SM but the best core is 780 TF/s rented, the same as cuBLAS which already uses 2SM 256x256 [probe_b200]'
  - 'portal r1: L ratio portal/rented 1.25-1.27 (clock ratio 1.31): SM-clock-bound, not DRAM-bound'
  - 'probe r2-e3: cuBLAS TF32 on hot 4096^3 / 8192x8192x2048 / 8192^3 reaches 69-72 / 75 / 71% of the clock peak, BF16 71 / 79 / 69%: TF32 is not uniquely capped. Power-capped runs at 1190-1330 MHz reach 89-96% per clock, so the ceiling does not scale with the SM clock: not smem-port or issue-rate bound; likely L2/fabric operand bandwidth (~13 TB/s in both dtypes) [r2-e3-tf32-ceiling-r1tiles]'
  - 'probe r2-e3: cuBLAS on our shape at B=1024 (no padding) gets 75% per clock; B=919 gets 68.5% only because of N padding to 960. Both halves of the statement are refuted; what remains is a ~75% practical ceiling at 1965 MHz plus padding and quantisation losses [r2-e3-tf32-ceiling-r1tiles]'
- id: H13
  statement: Re-reading W from L2 once per 128- or 256-row M-tile is the second limiter at L; a cluster of 2-4 CTAs along M with TMA multicast of the W tile cuts it to 1-2 passes.
  status: refuted
  evidence:
  - 'probe r1: cluster 4x1 (W multicast across two pairs) slower everywhere: 167.5 vs 146 us at 919, 98.5 vs 63 at B=5 [probe_b200]'
  - 'probe r1: raster AlongN (batch tiles of one W tile together) already reads W from DRAM once; AlongM re-reads it and costs 306 vs 148 us [probe_b200]'
  - 'probe r2-e3: cluster 2x2 (W multicast to two pairs along batch) 20-30% slower at 384-919 [r2-e3-tf32-ceiling-r1tiles]'
- id: H14
  statement: 'A one-launch CUTLASS sm100 TF32 kernel with the bias and the six-way split in the epilogue (output pointer picked by N-tile) removes the scatter and the tmp round trip and matches cuBLAS GEMM-only time at every B.'
  status: supported
  evidence:
  - 'run_tests r1: 51.0/48.8/49.0/55.5/56.0 (S), 57.0-68.5 (M), 79.6-133.6 (L) us rented, 16/16, one launch, builds in 31-39 s per variant [r1-adamod-cutlass-tf32-2sm-chunkepi]'
  - 'portal r1: 0.5884; S/M/L 55.3/69.6/136.9 us; predicted 0.603 (L overpredicted: the GEMM rate did not beat Tb''s, see H12/H15)'
  - 'dead end: 2SM 128xN tiles with N >= 128 trip a device assertion in the no-smem epilogue TMEM load; 128x64 is fine [probe_b200]'
  - 'probe r2: among tiles that build, the r1 dispatch (256x192 for 131-384 and 691+, 256x256 for 449/512) is already the per-B minimum; K tile 16/64, swizzle 2/4 and clusters 2x2/4x1 all lose [r2-adamod-stream16-cutlass-tf32, r2-e3-tf32-ceiling-r1tiles]'
- id: H15
  statement: Portal/rented time ratios on #35 are 1.21-1.31 for SM-clock-bound paths (tensor core, issue-bound SIMT) and 1.03-1.08 for DRAM-bound paths; the emulator must use the path's bound, not the language, to predict the portal time. The portal SM clock is about 1.55 GHz against the rented card's ~1.97 GHz in short runs.
  status: supported
  evidence:
  - 'portal vs rented r0: skinny B=5 76.1/60.6 = 1.26; cuBLASLt fp32 B=32 142.3/123.7 = 1.15; 3xTF32 B=919 688.9/574.6 = 1.20'
  - 'portal vs rented r1 (CUTLASS): B=16/32 1.03, B=96/128 1.07, B=5 1.12 (128x64 tile, 2 waves), M 1.07-1.19 rising with B, L 1.25-1.27'
  - 'portal vs rented r2 (tc_small): B <= 131 1.03-1.08 (DRAM-bound); B=192/211 1.26/1.26 (62.2/64.0 vs 49.3/50.6 rented): the kernel is compute-bound at the lock from B=192, and its rented -14/-16% gain vanished (r1 was 61.3/65.6 there). B >= 373 1.21-1.28 as in r1 [r2-e1-tc05-ef-small]'
  - 'probe r2-e3: the rented clock drifts 1349-1965 MHz with sustained load, so rented differences of ~7% between sessions with identical code are clock, not kernel [r2-e3-tf32-ceiling-r1tiles]'
  - 'dead end: nvidia-smi -lgc is not permitted on the rented card; the ratio comes from portal pairs only'
- id: H16
  statement: Any kernel that reads W once pays the dirty-L2 write-back (about 126 MB) on top of 227 MB, so the S/M floor under harness conditions is 45-53 us.
  status: refuted
  evidence:
  - 'probe r1-ef: reads-only floor with the harness flush is 39-40 us with evict_first (38.9 with a read-based flush): the hinted reads recycle their own lines and the write-back leaves the timed window [probe_b200]'
  - 'consequence: S scores can reach about 0.73 (t = 40 us vs Tb 83); portal r2 reached 44.1-47.8 us (0.70-0.72) [r2-e1-tc05-ef-small]'
- id: H17
  statement: 'A fused tcgen05 TF32 kernel whose W loads carry L2 evict_first (one N tile for B <= 128: 144 one-SM CTAs of 128 W rows, N = B padded to 16, >= 96 KB of W in flight; N = 256 tiles for 131-256) reaches 42-45 us rented at every B <= 128 and about 45-50 us at 131-211, against 49-61 us for the unhinted CUTLASS tiles; portal ratio about 1.05 (DRAM-bound). Worth about +0.03.'
  status: supported
  evidence:
  - 'probe r1-ef: reads-only evict_first floor 39-40 us; mma.sync version 41.6/43.6 at B=5/16 but 63 at B=32 (mma.sync is compute/issue-bound above B ~ 16) [r1-ef-bulk-tf32-stream-b16]'
  - 'run_tests r2: 42.7/42.0/42.4/43.5/45.7/44.1 us at B=5..131 rented, 49.3/50.6 at 192/211; 16/16 [r2-e1-tc05-ef-small]'
  - 'portal r2: 0.6108 (+0.022 vs r1); B <= 131 44.1-47.8 us (-12..-24% vs r1, ratio 1.03-1.08). Supported for B <= 131. The 192/211 part is refuted at the portal: 62.2/64.0 us (ratio 1.26), the one-SM kernel at kbs=1 runs about 440 TF/s rented = 40% of the TF32 peak and is clock-bound at the lock [r2-e1-tc05-ef-small, operator brief r3]'
- id: H18
  statement: 'Batch-side tile quantization costs 6-24% at L: the r1 dispatch pads B=449/691/773/853 to 512/768/960/960 columns; per-B N tiles of 240/240/208/224 (multiples of 16, 4 or fewer batch tiles) pad to 480/720/832/896 and gain 6-13% there, about +0.01, provided CUTLASS 2SM 256xN builds and passes for those N.'
  status: refuted
  evidence:
  - 'portal r1: 449 101.4 us = 512 101.9 (same padded work); 691 138.0; 773 164.9 = 853 166.7 = 919 167.9 (all padded to 960): times follow padded N, not B'
  - 'probe r2: CUTLASS 2SM 256x144/160/176/208/224/240 fail static divisibility asserts in the B-operand smem layout; only batch tiles that are multiples of 64 build [r2-adamod-stream16-cutlass-tf32]'
  - 'probe r2: 128-wide tiles cost about 5% more per padded column than 192/256, so the reachable padding cut (960 -> 896 at 773/853) is worth 0.7-1.2 us, noise. The padding lever needs a hand-written kernel with N = Bpad (H19) [r2-adamod-stream16-cutlass-tf32]'
- id: H19
  statement: 'A hand-written 2SM tcgen05 TF32 kernel for 131 < B <= 512, where one CTA pair (cluster 2, cta_group::2, M = 256 W rows) computes the whole batch in one pass (N = Bpad <= 512 as up to two UMMAs of N <= 256, each CTA supplying N/2 emb rows and 128 W rows, accumulator 128 x Bpad in TMEM, full K = 3072 in one accumulator), with evict_first on the W TMA stream, removes N padding beyond 16, wave quantisation (72 pairs on 74, one wave) and the unhinted write-back, and halves emb shared memory per CTA so B = 192/211 get >= 6 stages. Expected: B = 192/211 at the DRAM floor (42-46 us rented, portal ~1.05x); B = 373/384 at >= 70% per-clock efficiency (<= 52 us rented, ~60-65 us portal vs 83.1); B = 449/512 at <= 62/68 us rented (~75-85 portal vs 103). Worth about +0.035 to +0.04.'
  status: open
  evidence:
  - 'estimate: compute per SM at 100% is 2 x 128 x Bpad x 3072 FLOP / 4096 FLOP/clk = 73.7k clk at Bpad=384 (37.5 us at 1.965 GHz, 49 at 1.5 GHz) and 98k clk at 512 (50 / 65.6 us); at 75-85% this is 44-50 us rented at 384 vs 68.8 today, 59-67 at 512 vs 81.2 [analysis r3]'
  - 'r2 open note: a cta_group::2 pair halves emb smem (192/211 run kbs=1 at 5 stages today; 2x2 stages cost +13 us at 211, so ring depth matters) and a 2-accumulator N <= 512 version could take 373-512 off CUTLASS [r2-e1-tc05-ef-small]'
  - 'smem operand traffic per CTA per k-block of 32 for N=512: 16 KB W + 32 KB emb read in 1024 clk (47 B/clk) plus the same in TMA writes: well under the 128 B/clk port, unlike the 1SM 128x192 shape (213 B/clk if writes and reads share the port) [analysis r3]'
  - 'E1 (r3) builds it'
- id: H20
  statement: 'The one-SM kernel''s 40% efficiency at B = 192-256 comes mainly from emb: every one of the 144 CTAs re-reads the same B x 12 KB of emb from L2 (340 MB at B=192, more than W) and all CTAs hit the same 128 B lines at the same time. TMA multicast of each emb k-block across a 2-CTA cluster (each CTA loads half the rows to both) halves that traffic and gains >= 10% in cycles at 192/211/256; the alternative explanation is ring depth (5 stages of 40 KB at kbs=1 cover only ~1 us of DRAM latency) or the 1SM UMMA shape itself (H21).'
  status: open
  evidence:
  - 'r2: at B=192/211 the coupled ring runs kbs=1 with 5 stages because emb fills shared memory; 2x2 stages at 211 cost 63.3 vs 50.6 us: strongly stage-count sensitive [r2-e1-tc05-ef-small]'
  - 'r2: a separate emb ring with its own producer warp was 13-16 us slower than the coupled ring at every B; cause not isolated [r2-e1-tc05-ef-small]'
  - 'portal r2: 192/211 at 62.2/64.0 us, ratio 1.26: SM-clock-bound, so whatever limits them scales with the SM clock (TMA issue, mbarrier round trips, L2-to-SM transfer), not DRAM [r2-e1-tc05-ef-small]'
  - 'E3 (r3) tests multicast directly; E2 (r3) separates ring depth and UMMA shape'
- id: H21
  statement: 'From shared-memory-resident operands (no TMA), 1SM tcgen05 128xN kind::tf32 reaches >= 90% of 4096 FLOP/clk/SM for N = 128-256, and 2SM 256xN the same per SM; adding the TMA feed costs the 1SM shape more than the 2SM shape because it moves N emb rows per CTA instead of N/2 (operand bytes per FLOP 1.5x at N=192). If instead the 1SM shape is already below 70% with resident operands, the UMMA shape itself is the M-band limiter and only 2SM designs can reach 75%.'
  status: open
  evidence:
  - 'r1: CUTLASS 2SM 256x256 is 1.13-1.19x faster than 1SM 128x256 at B=919 with TMA feed; no resident-operand measurement exists [r1-adamod-cutlass-tf32-2sm-chunkepi]'
  - 'r2-e3: the cuBLAS 2SM ceiling (~75% at 1965 MHz, 90%+ when power-capped to 1.2-1.3 GHz) does not scale with the SM clock, so the portal at ~1.55 GHz may sit nearer 80% for the same kernel: the rented card understates the portal efficiency of 2SM tiles [r2-e3-tf32-ceiling-r1tiles]'
  - 'E2 (r3) measures the rates in cycles'
- id: H22
  statement: 'If H19 holds, the same pair kernel extended to two batch halves per pair (persistent over 2 tiles of N = ceil(Bpad/2) <= 512, W tile re-read from L2 on the second pass, 72 pairs x 2 tiles) covers 691-919 at >= 80% per-clock efficiency: about 130-140 us portal at 919 vs 170 (score 0.60 vs 0.50), worth about +0.025 over the four L workloads.'
  status: open
  evidence:
  - 'estimate: 919 -> 928 columns as 2 x 464 (vs 960 padded today), one accumulator pass each, W from L2 on the second pass (3 MB per pair) [analysis r3]'
  - 'E1 (r3) stretch step; not a success criterion this round'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H19",
    "title": "Hand-written 2SM tcgen05 TF32 pair kernel: one pass over the whole batch (N = Bpad <= 512) for 131 < B <= 512, evict_first W stream",
    "operation": "new_design",
    "parents": ["r2-e1-tc05-ef-small"],
    "band": "M",
    "instructions": "Start from tc_small.cu in r2-e1-tc05-ef-small and keep its B <= 131 path, CUTLASS 256x192 for B > 512, binding and dispatch unchanged. Add tc_pair.cu: launch 144 CTAs as 72 clusters of 2 (cudaLaunchKernelEx, cluster dims 2x1x1), 1 CTA per SM, 128 threads. CTA rank r owns W rows m0 = 256*cluster + 128*r (so the pair is the UMMA M = 256) and, for each UMMA i of N_i (N_0 = min(Bpad, 256), N_1 = Bpad - 256 if Bpad > 256; Bpad = B rounded up to 16; N_i multiple of 16, >= 32), emb rows [sum_{j<i} N_j + r*N_i/2, +N_i/2). Per stage per CTA: kbs x (16 KB W + Bpad/2 x 128 B emb), both via TMA with SWIZZLE_128B and CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 (FLOAT32 truncates and fails: see r2 findings); W with createpolicy evict_first cache hint, emb unhinted. Choose kbs as the largest of 4/3/2/1 that leaves >= 3 stages in 224 KB (B=192: kbs 2 with 3-4 stages or kbs 1 with 7). Use cta_group::2 PTX: tcgen05.alloc.cta_group::2 of 512 columns (power of 2 >= Bpad) issued in both CTAs as CUTLASS does; the leader (rank 0) issues tcgen05.mma.cta_group::2.kind::tf32 with M=256, N=N_i, four per 32-wide k-block per UMMA, TMEM column offset sum_{j<i} N_j; the full barrier of each stage lives in the leader, both CTAs' TMA loads complete_tx on it through its shared::cluster address (mapa), expect_tx = both CTAs' stage bytes; the leader's tcgen05.commit.cta_group::2 ... .multicast::cluster with ctaMask 0b11 arrives at the empty barrier of both CTAs, each producer waits on its own. Read the exact PTX forms (fences, cluster barrier at start and before exit, instruction descriptor bits for M=256 / cta_group::2, UMMA smem descriptors for the SW128 K-major atoms) from CUTLASS 4.4.1 headers: cute/arch/mma_sm100_umma.hpp, cute/arch/mma_sm100_desc.hpp, cutlass/gemm/collective/sm100_mma_warpspecialized.hpp, cutlass/pipeline/sm100_pipeline.hpp. There is no public tcgen05 example in read_example. Epilogue as tc_small: each warp reads its 32 TMEM lanes (W rows) in 16-column chunks, adds bias, stores out_c[b*3072 + m - c*3072] for b < B; c = m0/3072 is uniform per CTA (3072 % 128 == 0). Steps: (1) correctness at B = 132, 192, 211, 256, 373, 384, 449, 512 vs torch fp64, report max |err| per B (must be <= 1.7e-3). (2) In-kernel clock64 and %globaltimer deltas (first TMA issue to last store, thread 0 of each CTA, max over CTAs written to a diagnostic buffer in probes only) to report cycles, the rented SM clock and per-clock efficiency = 2*128*Bpad*3072 / (cycles * 4096) per CTA. (3) Interleaved probe, 3 rounds, harness flush + CUPTI: the new kernel vs the r2 dispatch (tc_small at 192/211, CUTLASS 256x192/256x256 at 373-512) at B = 192, 211, 256, 373, 384, 449, 512; also kbs/stage variants at 192 and 512. (4) Predict the portal time per B as max(1.05 x rented time if DRAM-bound, cycles / 1.55 GHz) and state it in the card. (5) Dispatch 131 < B <= 512 to tc_pair only where it wins by >= 5% in cycles; run_tests 16/16; build must stay under 300 s (the LOCAL target needs the __CUDA_ARCH_FEAT_SM100_ALL guard). Stretch (H22, only if steps 1-5 are done): a persistent variant doing two batch halves per pair for 691-919, measured the same way; do not let it delay the submission candidate.",
    "success": "16/16 correct with max |err| <= 1.7e-3; in the interleaved probe, B = 192/211 <= 46 us rented (DRAM-bound), B = 373/384 <= 52 us rented with >= 70% per-clock efficiency, B = 449/512 <= 62/68 us rented; predicted portal M-band geomean <= 52 us (from 66.6).",
    "refuted_if": "The pair kernel's cycles at 373-512 are within 10% of the CUTLASS 256x192/256x256 tiles (efficiency <= 60% per clock), or B = 192/211 stay above 48 us rented with >= 6 stages, meaning one-pass wide tiles do not beat persistent padded tiles and the limiter is elsewhere (H20/H21).",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H21",
    "title": "Microbenchmarks in cycles: 1SM vs 2SM UMMA TF32 rate from resident operands, cost of the TMA feed, and ring-depth slope of tc_small",
    "operation": "new_design",
    "parents": ["r2-e1-tc05-ef-small"],
    "band": "M",
    "instructions": "Probe-only session; no submission candidate. All timings in SM cycles (clock64 around the mainloop in the MMA-issuing thread, plus %globaltimer to report the rented SM clock for every run; run each case 3 times interleaved and report the clock). (a) Resident-operand UMMA rate: strip tc_small's TMA producer; fill one stage of W and emb smem once (any values), then issue the full 96 k-blocks x 4 UMMAs of 1SM 128xN kind::tf32 against that stage with commits every k-block as in the real kernel, for N = 128, 192, 256. Report FLOP/clk/SM vs the 4096 peak. Repeat with cta_group::2 256xN (cluster of 2, leader issues, both CTAs hold operands; mirror CUTLASS's sm100 PTX forms from cute/arch/mma_sm100_umma.hpp) for N = 128, 192, 256; report per-SM efficiency. (b) TMA feed from L2: same loops but with the real coupled TMA ring, W replaced by a 148 x 128 x 3072 fp32 buffer (227 MB -> 232 MB is too large to be L2-resident, so instead give every CTA the same 128 W rows: 1.5 MB, L2-resident) and emb as today; this isolates the smem-port and TMA-issue cost from DRAM. Compare 1SM 128x192 kbs=1 5 stages (the shipped B=192 config) against kbs=2 (if the smem fits with the small W) and against a 2SM 256x192 ring where each CTA holds 96 emb rows. (c) Ring-depth slope on the real tc_small at B = 192 and 256 with real W from DRAM (harness flush): force stages = 3, 4, 5 at kbs=1 and report cycles per k-block; a steep slope means latency-bound, a flat one means issue/operand-bound. (d) emb L2 hot-spotting, diagnostic only: at B = 192 and 256, run tc_small unchanged vs a variant whose tensor map points each CTA at its own copy of emb (a scratch buffer holding emb replicated 144 times, filled outside the timed region); if private copies are >= 10% fewer cycles, the shared-line L2 contention is real and E3's multicast is the fix. Write every number into findings with the clock it ran at and the per-clock efficiency; name the limiter for the B = 192 case (ring depth, UMMA shape, smem port, or emb L2).",
    "success": "Each of (a)-(d) reported in cycles with the measured clock, and a one-line verdict on what limits tc_small at B = 192-256: e.g. resident 1SM 128x192 >= 90% and TMA-fed 1SM <= 60% while 2SM-fed >= 75% supports H21; a steep stage slope supports the ring-depth explanation; private emb copies >= 10% faster supports H20.",
    "refuted_if": "Resident 1SM 128x192 is already <= 70% of 4096 FLOP/clk (then the UMMA shape, not the feed, is the limiter and H21's first clause is false), or 2SM-fed is no better per SM than 1SM-fed (then the 2SM route of H19 gains nothing from operand traffic).",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H20",
    "title": "tc_small with TMA multicast of the emb k-block across a 2-CTA cluster for 131 < B <= 256",
    "operation": "structural_mutation",
    "parents": ["r2-e1-tc05-ef-small"],
    "band": "M",
    "instructions": "Minimal change to tc_small.cu, keep everything else (1SM UMMA, kbs rule, evict_first W, epilogue) identical. Launch the 144 CTAs as 72 clusters of 2 (cudaLaunchKernelEx, cluster dims 2x1x1); CTAs 2p and 2p+1 keep their own W tiles (rows 128*(2p) and 128*(2p+1)). For each stage, CTA rank r loads emb rows [r*Bpad/2, (r+1)*Bpad/2) of each k-block with cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster, ctaMask 0b11, into the same smem offset in both CTAs, so each CTA receives the full Bpad rows while L2 traffic for emb halves. Each CTA's full barrier expects its own W bytes plus the full emb bytes of the stage (both halves arrive from the two producers). Empty barrier: a producer may overwrite a stage only when both CTAs' MMAs have consumed it, so init each empty barrier with count 2 and have each CTA's MMA thread commit with tcgen05.commit.cta_group::1 ... .multicast::cluster ctaMask 0b11 (one arrival in each CTA). Add a cluster barrier (barrier.cluster.arrive/wait) after mbarrier init and before exit so no CTA leaves while its peer can still write its smem. Bpad/2 must be a multiple of 8 rows for the SW128 atoms: use Bpad rounded to 32 for this variant (B = 192 -> 192, 211 -> 224, 256 -> 256; emb rows beyond B zero-filled by TMA OOB). Measure with in-kernel clock64 + %globaltimer (report the rented clock) in one interleaved probe, 3 rounds, harness flush + CUPTI: multicast vs unchanged tc_small at B = 132, 192, 211, 256. Also try kbs=2 at 192 if the stage still fits (it will not, smem is unchanged; note it). Report cycles, per-clock efficiency = 2*128*Bpad*3072/(cycles*4096), max |err| vs fp64 per B (<= 1.7e-3), and the predicted portal time = max(1.05 x rented DRAM-bound time, cycles / 1.55 GHz). If multicast wins by >= 5% in cycles at 192 and 211, package it as a candidate: dispatch 131 < B <= 256 to the multicast variant, B <= 131 unchanged, B > 256 unchanged, run_tests 16/16, card paths with cluster: 2 and w: ef. Otherwise record the negative result and no candidate.",
    "success": ">= 10% fewer cycles than tc_small at B = 192 and 211 (and 256), 16/16 correct, error <= 1.7e-3: emb L2 re-read is a confirmed M-band limiter and the pair kernel of E1 should also multicast emb across pairs.",
    "refuted_if": "Multicast changes cycles by <= 3% at 192/211/256 in the interleaved probe: emb L2 traffic is not the limiter (ring depth or UMMA shape are, see E2) and E1 need not add multicast.",
    "model": "opus"
  }
]
```