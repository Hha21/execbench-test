### Assessment

Where the score is. The best kernel (r1 CUTLASS 2SM TF32, 0.5884) scores 0.62-0.66 at S, 0.60-0.65 at M and 0.47-0.57 at L, so L is at Tb's own GEMM rate and only the removed scatter separates us. Three levers remain:

- **S and small M (8 workloads, worth about +0.03).** The reads-only floor under the harness flush is 39-40 us with evict_first W loads, and r1-ef already runs 41.6/43.6 us at B=5/16. Every CUTLASS tile sits at 49-61 us for B=32-211 because its TMA loads carry no eviction hint and the padded-N compute at 68% of peak collides with the stream. A fused tcgen05 kernel with evict_first W loads should reach 42-45 us for all B ≤ 128 and about 47 us for 131-211.
- **L tile quantization (about +0.01).** The dispatch pads B=449/691/773/853 to 512/768/960/960 batch columns; N tiles of 240/240/208/224 cut that to 480/720/832/896 (6-13% less work).
- **L TF32 ceiling (0 to +0.03).** Both cuBLAS and our 2SM kernel stop at 68% of the TF32 peak. If a square hot GEMM reaches 85%, our shape is losing 15%; if it stops at 70% (and bf16 does not), TF32 is hardware-capped and L is closed.

What we learned in r1: H16 refuted (floor 39-40 not 45-53), H13 refuted (multicast slower), H11's batched pointer-array refuted (12-43% slower), H15 supported (portal/rented 1.03-1.07 DRAM-bound, 1.25 tensor-bound). This round: E1 builds the evict_first tcgen05 S/M kernel (CUTLASS patch first as a one-hour test, then hand-written), E2 assembles a portal-ready dispatch with r1-ef at B ≤ 16 and per-B N tiles, E3 measures the TF32 ceiling and diagnoses L.

```yaml ledger
# Hypothesis ledger for #35 (035_flux_ada_layer_norm_zero_modulation_extraction). Seeded 2026-10-10 (research round r0); updated r1 2026-10-10; updated r2 2026-10-10.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'bf16 is this problem''s own dtype: write cutlass::bfloat16_t / __nv_bfloat16 directly. Never build type names with the ## token-paste operator (the lint rejects it: it reads as lint evasion to a reviewer).'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot (r13, H21).'
- 'Any cublasLt algorithm forced per size must be chosen from shape only, verified 16/16 on 10 random rounds, and must launch the same activity sequence every call (split-K with a reduce kernel is allowed only if the count is constant per shape).'
- 'The loop''s lint rejects the literal bf16 type name in CUTLASS sources; CUTLASS candidates go through run_tests and the operator whitelists the type name for this problem (dtype is bf16 by definition).'
- 'Portal A/B packaging (r2): a candidate may dispatch by (batch_size, seq_len) among same-M shapes (4096 group: 4,1024 / 16,256 / 8,512 plus 2,2053; 8192 group: 16,512 / 8,1024 / 64,128 / 32,256). Shape-only dispatch is legitimate, but every variant must be a real candidate (16/16 correct, one launch); no diagnostic or deliberately wrong variant. The card must list which shape runs which variant and all 10 rented per-size times per variant, measured interleaved in one probe.'
- 'Rented differences under about 4% between fused variants are unproven at the portal (fused kernels scale 1.36-1.40 portal/rented, plain GEMMs 1.22); interleave variants in the same probe, 3 rounds, and only promote gains of 5% or more.'
- '#35 precision (operator ruling, 2026-10-10): plain TF32 tensor-core compute with fp32 accumulation is allowed (CUBLAS_COMPUTE_32F_FAST_TF32, tcgen05 kind::tf32, allow_tf32 inside our own GEMM call); inputs and outputs stay fp32 and no intermediate is stored below fp32. bf16/fp16 operands and BF16x9 emulation remain forbidden. TF32 max error 1.6e-3 vs atol 2e-3: verify every candidate on all 16 workloads x 10 rounds and never round outputs.'
- '#35 dtype is fp32 throughout; the bf16 constraint above is inherited text from #30 and does not apply here.'
- '#35 TF32 margin (operator, r1): accumulate in fp32 over the full K = 3072 inside one tensor-core accumulator; no split-K whose partials are stored or re-fed below fp32; no rounding of outputs or intermediates. Every candidate records max |err| vs fp64 per B (must stay <= 1.7e-3 against atol 2.0e-3) in its findings.'
- '#35 cache hints (r2): L2 evict_first is used only on the TMA/bulk loads of this call''s own weight buffer and evict_last/evict_first only on this call''s outputs; no applypriority, no persisting-L2 windows. Record the hint in the card''s paths (w: ef) so the emulator can learn the portal ratio of hinted paths.'
- '#35 portal ratios (r2, H15): emulator predictions for tensor-core-bound paths must use 1.25-1.31 portal/rented; DRAM-bound paths 1.03-1.07. The r1 emulator overpredicted L (0.603 predicted vs 0.5884 portal) by ignoring this.'
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
- id: H11
  statement: cuBLASLt CUBLAS_COMPUTE_32F_FAST_TF32 with a bias epilogue plus a scatter is a 16/16-correct path at 55-177 us rented (score about 0.54); cuBLAS 13's batched pointer-array mode over the six chunks removes the scatter at no GEMM cost.
  status: refuted
  evidence:
  - 'run_tests r1-e1: 55.5/54.1/54.8/59.5/60.4 (S), 62.7-94.5 (M), 99.3-177.0 (L) us, 16/16, emulator 0.537 [r1-e1-lt-fasttf32-bias-scatter]'
  - 'refuted part: pointer-array batched GEMM (bias via C broadcast, pinned pointer arrays) is 12-43% slower than big GEMM + scatter at every B; bias epilogue + pointer array is NOT_SUPPORTED; the pointer arrays must change per call and reach the GPU over PCIe [probe_b200]'
  - 'superseded: the CUTLASS fused epilogue (H14) is faster at every B and needs no library heuristics'
- id: H12
  statement: 'tcgen05 TF32 stops at about 68% of the 909 TF/s lock-time peak (cuBLAS 613 TF/s = Tb; our 2SM 256x192/256x256 the same) because of operand traffic through shared memory: a 2SM 256x256x8 MMA needs 8 KB of TMA writes and 12 KB of MMA reads per CTA per 128 clk (160 B/clk against 128 B/clk), a hardware ceiling no tile beats. The alternative is that the loss is shape-specific (72 W tiles, 1-5 batch tiles, K loop of 96 stages, W from DRAM) and a square hot GEMM shows >= 85%.'
  status: open
  evidence:
  - 'probe r0: cuBLAS TF32 770 TF/s unlocked at 919 = 65-67% of the 1.19 PF/s TF32 peak at 1965 MHz [probe_b200]'
  - 'probe r1: swapped 2SM 256x256 142 us, 256x192 135.5, 1SM 128x256 161 at 919: 2SM is 1.13-1.19x 1SM but the best core is 780 TF/s rented, the same as cuBLAS which already uses 2SM 256x256. The 2SM half of the original claim is refuted [probe_b200]'
  - 'portal r1: L ratio portal/rented 1.25-1.27 (clock ratio 1.31): SM-clock-bound, not DRAM-bound'
  - 'E3 (r2) measures CUTLASS 2SM 256x256 TF32 and BF16 on 8192^3 hot: TF32 >= 85% means shape loss (fixable), TF32 ~70% with BF16 >= 80% means TF32-specific hardware cap'
- id: H13
  statement: Re-reading W from L2 once per 128- or 256-row M-tile is the second limiter at L; a cluster of 2-4 CTAs along M with TMA multicast of the W tile cuts it to 1-2 passes.
  status: refuted
  evidence:
  - 'probe r1: cluster 4x1 (W multicast across two pairs) slower everywhere: 167.5 vs 146 us at 919, 98.5 vs 63 at B=5 [probe_b200]'
  - 'probe r1: raster AlongN (batch tiles of one W tile together) already reads W from DRAM once; AlongM re-reads it and costs 306 vs 148 us [probe_b200]'
- id: H14
  statement: 'A one-launch CUTLASS sm100 TF32 kernel with the bias and the six-way split in the epilogue (output pointer picked by N-tile) removes the scatter and the tmp round trip and matches cuBLAS GEMM-only time at every B.'
  status: supported
  evidence:
  - 'run_tests r1: 51.0/48.8/49.0/55.5/56.0 (S), 57.0-68.5 (M), 79.6-133.6 (L) us rented, 16/16, one launch, builds in 31-39 s per variant [r1-adamod-cutlass-tf32-2sm-chunkepi]'
  - 'portal r1: 0.5884; S/M/L 55.3/69.6/136.9 us; predicted 0.603 (L overpredicted: the GEMM rate did not beat Tb''s, see H12/H15)'
  - 'dead end: 2SM 128xN tiles with N >= 128 trip a device assertion in the no-smem epilogue TMEM load; 128x64 is fine [probe_b200]'
- id: H15
  statement: Portal/rented time ratios on #35 are 1.25-1.31 for SM-clock-bound paths (tensor core, issue-bound SIMT) and 1.03-1.07 for DRAM-bound paths; the emulator must use the path's bound, not the language, to predict the portal time.
  status: supported
  evidence:
  - 'portal vs rented r0: skinny B=5 76.1/60.6 = 1.26; cuBLASLt fp32 B=32 142.3/123.7 = 1.15; 3xTF32 B=919 688.9/574.6 = 1.20'
  - 'portal vs rented r1 (CUTLASS): B=16/32 1.03, B=96/128 1.07, B=5 1.12 (128x64 tile, 2 waves), M 1.07-1.19 rising with B, L 1.25-1.27'
  - 'dead end: nvidia-smi -lgc is not permitted on the rented card; the ratio comes from portal pairs only'
- id: H16
  statement: Any kernel that reads W once pays the dirty-L2 write-back (about 126 MB) on top of 227 MB, so the S/M floor under harness conditions is 45-53 us.
  status: refuted
  evidence:
  - 'probe r1-ef: reads-only floor with the harness flush is 39-40 us with evict_first (38.9 with a read-based flush): the hinted reads recycle their own lines and the write-back leaves the timed window [probe_b200]'
  - 'consequence: S scores can reach about 0.73 (t = 40 us vs Tb 83); the CUTLASS tiles at 49-56 us rented leave 20% on the table'
- id: H17
  statement: 'A fused tcgen05 TF32 kernel whose W loads carry L2 evict_first (one N tile for B <= 128: 144 one-SM CTAs of 128 W rows, N = B padded to 16, >= 96 KB of W in flight; N = 256 tiles for 131-256) reaches 42-45 us rented at every B <= 128 and about 45-50 us at 131-211, against 49-61 us for the unhinted CUTLASS tiles; portal ratio about 1.05 (DRAM-bound). Worth about +0.03.'
  status: open
  evidence:
  - 'probe r1-ef: reads-only evict_first floor 39-40 us; mma.sync version 41.6/43.6 at B=5/16 but 63 at B=32 (mma.sync is compute/issue-bound above B ~ 16) [r1-ef-bulk-tf32-stream-b16]'
  - 'r1 open: CUTLASS 256x64/256x128/256x192 all 55-61 us at B=96-211; 128x64 16-stage 49-52 at B <= 32 [r1-adamod-cutlass-tf32-2sm-chunkepi]'
  - 'E1 (r2) tests it twice: CUTLASS mainloop with EVICT_FIRST on the A (W) TMA loads, then a hand-written tcgen05 kernel for B <= 256'
- id: H18
  statement: 'Batch-side tile quantization costs 6-24% at L: the r1 dispatch pads B=449/691/773/853 to 512/768/960/960 columns; per-B N tiles of 240/240/208/224 (multiples of 16, 4 or fewer batch tiles) pad to 480/720/832/896 and gain 6-13% there, about +0.01, provided CUTLASS 2SM 256xN builds and passes for those N.'
  status: open
  evidence:
  - 'portal r1: 449 101.4 us = 512 101.9 (same padded work); 691 138.0; 773 164.9 = 853 166.7 = 919 167.9 (all padded to 960): times follow padded N, not B'
  - 'E2 (r2) builds 256x208/224/240 and measures per B'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H17",
    "title": "Fused tcgen05 TF32 GEMM with evict_first W loads for B <= 256 (S band and small M)",
    "operation": "new_design",
    "parents": ["r1-adamod-cutlass-tf32-2sm-chunkepi", "r1-ef-bulk-tf32-stream-b16"],
    "band": "S",
    "instructions": "Goal: bring every B <= 128 to 42-45 us rented and B=131-211 to <= 50 us, one launch, fp32 accumulate over the full K. Step 1 (time-box 1.5 h, the cheap test of the mechanism): in the r1 CUTLASS kernel, copy cutlass/gemm/collective/sm100_mma_warpspecialized.hpp (and its dispatch-policy tag) into the solution under a new class name, and pass TMA::CacheHintSm90::EVICT_FIRST to the A-operand (W) TMA loads via copy_atom.with(barrier, mcast_mask, cache_hint); rebind the builder's CollectiveOp template arguments to the new mainloop. Probe interleaved, 3 rounds, harness flush: 128x64x16-stage at B=5/16/32 and 256x64 / 256x192 / 256x256 at B=96/128/131/192/211, hinted vs unhinted. Also probe the unhinted 128x64 at B=5 with a read-based flush instead of zero_ to measure how much of its 51 us is the dirty-L2 penalty. Step 2: hand-written tcgen05 kernel for B <= 128 (then <= 256 if time): 144 CTAs x 1 SM, 128 W rows per CTA (chunk c = blockIdx*128/3072 uniform), producer warp issues cp.async.bulk.tensor.2d with .L2::cache_hint (createpolicy evict_first) for W boxes [128 rows x 32 k] fp32 with 128B swizzle into the canonical K-major SW128 UMMA layout, emb boxes [Bpad x 32 k] without hint; 6-7 stages (16 KB W + <= 16 KB emb each); tcgen05.alloc Bpad columns (N = B rounded up to 16, <= 128, or 256 for 129-256 with 1SM M=128); kind::tf32 UMMA M=128 N=Bpad K=8, four per stage, tcgen05.commit to the per-stage empty mbarrier; epilogue tcgen05.ld 32x32b by 4 warps (one TMEM lane quadrant each), add bias[m], store out_c[b*3072 + m - c*3072] coalesced over m. Use the CUTLASS 4.4 headers on the box (cute/arch/mma_sm100_umma.hpp, cute/arch/copy_sm100_tma.hpp, cutlass/arch/barrier.h, examples/70_blackwell_gemm) for the instruction and smem descriptor encodings; build tensor maps on the host each call with cuTensorMapEncodeTiled (-lcuda is linked by default). Validate bit-level against the r1 CUTLASS output and record max |err| vs fp64 per B (<= 1.7e-3). Ship a dispatch: new kernel for the B it wins, r1 CUTLASS tiles elsewhere; run_tests 16/16; also record the r1-ef stream kernel's max |err| per B if it stays in the dispatch for B <= 16. Record hinted and unhinted times per tile per B as findings; set w: ef in paths.",
    "success": "<= 45 us rented at every B <= 128 (currently 49-56) and <= 50 us at B=131-211, 16/16 pass, max |err| <= 1.7e-3, one launch; emulator predicts >= 0.61",
    "refuted_if": "Both the hinted CUTLASS tile and the hand-written tcgen05 kernel stay >= 48 us at B=96/128 with evict_first on W (then the S gap is compute/epilogue/pipeline-fill, not the dirty-L2 write-back), or the hint saves < 3 us on the 128x64 tile at B=5",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H18",
    "title": "Per-B batch-tile N to remove padding waste at L, plus r1-ef streaming kernel for B <= 16: portal-ready dispatch",
    "operation": "knob_mutation",
    "parents": ["r1-adamod-cutlass-tf32-2sm-chunkepi", "r1-ef-bulk-tf32-stream-b16"],
    "band": "L",
    "instructions": "Step 1: add CUTLASS 2SM swapped tiles Tf32Gemm<256,N,0> for N = 208, 224, 240 (and 128, 160, 176 for the M band) as separate translation units; confirm each builds and passes a 10-round correctness check at the B it serves (watch for the no-smem-epilogue TMEM assertion seen with 128xN). Step 2: probe interleaved, 3 rounds, harness flush: for each L workload compare the r1 tile with the tile that minimises padded columns using <= ceil(B/256) batch tiles: 449 -> 240 (2 tiles, 480), 512 -> 256, 691 -> 240 (3, 720), 773 -> 208 (4, 832), 853 -> 224 (4, 896), 919 -> 240 (4, 960) vs 192 (5, 960); for M: 131 -> 144 or 160, 192 -> 192, 211 -> 224, 373 -> 192 (2, 384) vs 128 (3, 384), 384 -> 192. Pick per B by the rule 'fastest, ties to fewer waves' and record the padded-work model vs measured times. Step 3: build the dispatch: r1-ef stream_tc_kernel<1,6> for B <= 8 and <2,5> for 9-16 (one launch, evict_first bulk ring; record its max |err| vs fp64 per B, missing from r1), CUTLASS 128x64x16 for 17-64, best tile per B above. run_tests 16/16, record all per-B times and max |err| <= 1.7e-3. This candidate goes to the portal as the safe gain even if E1 fails; if E1 delivers, E1's kernel replaces the B <= 128 rows of this dispatch.",
    "success": ">= 6% faster rented at B=449, 691, 773, 853 (773: <= 123 us vs 131.4), no regression > 2% at any B, 16/16; combined dispatch emulator >= 0.60",
    "refuted_if": "The N=208/224/240 tiles do not build or run no faster than 192/256 at equal padded work (then the time is not set by padded columns and the model in H18 is wrong), or the 2SM epilogue asserts for N not a multiple of 64",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H12",
    "title": "Is 68% of the TF32 peak a hardware ceiling? Square hot GEMM probe, then L diagnosis",
    "operation": "new_design",
    "parents": ["r1-adamod-cutlass-tf32-2sm-chunkepi"],
    "band": "L",
    "instructions": "Step 1 (probe, 30 min): with the r1 Tf32Gemm template (plain LinearCombination epilogue, no chunk wrapper) time CUTLASS 2SM 256x256 and 1SM 128x256 TF32 on M=N=K=8192 with hot inputs (no flush, 3 rounds, CUPTI), plus cuBLAS TF32 on the same shape, plus the same CUTLASS 2SM 256x256 template in BF16 (256x256x16 tile) as the control (expected 1.6-2.0 PF/s rented). Report TF/s and the fraction of 1.19 PF/s (TF32) and 2.37 PF/s (BF16) at 1965 MHz. Step 2: our shape with W fitting L2: M=6144 (two chunks, 75 MB), N=919, K=3072, hot vs harness-flushed, and M=18432 flushed: this separates DRAM/dirty-L2 effects from the compute rate. Step 3, if TF32 square >= 85%: the loss is shape-specific; sweep on B=919 and 691 interleaved: K-tile 16/32/64 at auto stages, stage count, CLC vs static persistent scheduler, raster swizzle size (max_swizzle 1/2/4/8), AlongN vs AlongM with the swapped layout fixed, and the epilogue (NoSmem vs TMA-store epilogue). Build the best as a candidate with per-B tiles from E2 and run_tests 16/16 (max |err| <= 1.7e-3). Step 3, if TF32 square <= 72% while BF16 >= 80%: the TF32 instruction rate is the cap; record the per-clock MAC rate and stop L work; write the finding so later rounds do not revisit it. Step 3, if both TF32 and BF16 stop near 70%: the smem-traffic model holds; test whether K-tile 64 (fewer, larger TMA writes) or 1SM 128x256 with 2 CTAs per SM changes the rate by > 5%.",
    "success": "Either a candidate >= 8% faster rented at B=691-919 (919: <= 123 us vs 133.6) with 16/16 pass, or a decisive ceiling measurement (TF32 and BF16 square rates with < 3% spread over 3 rounds) that settles H12",
    "refuted_if": "CUTLASS 2SM 256x256 TF32 on 8192^3 hot reaches <= 72% of 1.19 PF/s while BF16 reaches >= 80%: TF32 is hardware-capped at the rate cuBLAS and we already achieve, and H12's headroom is gone",
    "model": "opus"
  }
]
```