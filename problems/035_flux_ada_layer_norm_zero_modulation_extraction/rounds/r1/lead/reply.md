### Assessment

The r0 portal page settles the two big unknowns. Tb is a TF32 PyTorch solution (82.8 µs at B=5, 169.8 at 919 = 613 TF/s), and Tsol works out to about 15 µs for B ≤ 211 and max(15 µs, FLOPs / 1.8 PF/s) above that (23.7 µs at 373, 57.9 at 919). Exact fp32 cannot score: H7 and H8 are refuted, H2/H3/H6 retired. With the TF32 ruling the plan is now the standard one.

Where the score is: the emulator says a cuBLAS-level TF32 path (≈55/100/170 µs S/M/L portal) gives ≈0.53, and a fused tcgen05 kernel near the TF32 peak (≈47/57/102 µs) gives ≈0.67. The gap between those two is the GEMM core and the epilogue: cuBLAS runs at ~65% of the 909 TF/s lock-time TF32 peak, which is exactly what a shared-memory-bandwidth bound for fp32 operands in 1SM mode predicts (H12), and the un-fused scatter costs 47 µs of the 170 at B=919 (H9). The S and M bands are bounded instead by the harness's dirty-L2 writeback (reads 227 MB + ~126 MB write-back ≈ 50 µs), which caps S scores at ~0.65–0.70 unless a reads-only probe shows otherwise (H16).

This round: E1 ships the cuBLASLt TF32 anchor (one launch if cuBLAS 13's pointer-array batch mode works) to bank ≈0.5; E2 builds the CUTLASS sm100 2SM TF32 GEMM core with cluster multicast and a fused six-way epilogue, testing H12/H13/H14; E3 measures the memory floor under harness conditions, the dirty-L2 penalty and the clock-lock ratio, which decide whether an S-band custom kernel is worth a later round.

```yaml ledger
# Hypothesis ledger for #35 (035_flux_ada_layer_norm_zero_modulation_extraction). Seeded 2026-10-10 (research round r0); updated r1 2026-10-10.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'bf16 is this problem''s own dtype: write cutlass::bfloat16_t / __nv_bfloat16 directly. Never build type names with the ## token-paste operator (the lint rejects it: it reads as lint evasion to a reviewer).'
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
- 'Portal A/B packaging (r2): a candidate may dispatch by (batch_size, seq_len) among same-M shapes (4096 group: 4,1024 / 16,256 /
  8,512 plus 2,2053; 8192 group: 16,512 / 8,1024 / 64,128 / 32,256). Shape-only dispatch is legitimate, but every variant must be a
  real candidate (16/16 correct, one launch); no diagnostic or deliberately wrong variant. The card must list which shape runs
  which variant and all 10 rented per-size times per variant, measured interleaved in one probe.'
- 'Rented differences under about 4% between fused variants are unproven at the portal (fused kernels scale 1.36-1.40 portal/rented,
  plain GEMMs 1.22); interleave variants in the same probe, 3 rounds, and only promote gains of 5% or more.'
- '#35 precision (operator ruling, 2026-10-10): plain TF32 tensor-core compute with fp32 accumulation is allowed
  (CUBLAS_COMPUTE_32F_FAST_TF32, tcgen05 kind::tf32, allow_tf32 inside our own GEMM call); inputs and outputs stay fp32 and
  no intermediate is stored below fp32. bf16/fp16 operands and BF16x9 emulation remain forbidden. TF32 max error 1.6e-3
  vs atol 2e-3: verify every candidate on all 16 workloads x 10 rounds and never round outputs.'
- '#35 dtype is fp32 throughout; the bf16 constraint above is inherited text from #30 and does not apply here.'
- '#35 TF32 margin (operator, r1): accumulate in fp32 over the full K = 3072 inside one tensor-core accumulator; no split-K whose
  partials are stored or re-fed below fp32; no rounding of outputs or intermediates. Every candidate records max |err| vs fp64 per
  B (must stay <= 1.7e-3 against atol 2.0e-3) in its findings.'
hypotheses:
- id: H1
  statement: The reference's cuBLAS SIMT sgemm is within 25-30% of the FFMA peak at M/L, so FFMA-only designs cannot gain more than
    about 1.3x there; packed fma.f32x2 adds no throughput.
  status: supported
  evidence:
  - 'probe: cutlass3x_sm100_simt_sgemm 55-57 TF/s at B=384/919 vs 74.4 TF/s peak at 1965 MHz [probe_b200]'
  - 'probe: FFMA 67.6 TF/s vs FFMA2 (fma.rn.f32x2) 73.5 TF/s on 148x8 CTAs: same 128 FMA/clk/SM datapath [probe_b200]'
  - 'r1: moot since the TF32 ruling and Tb = TF32 speed; FFMA paths cannot score anywhere (B=5 skinny 76.1 us portal vs Tb 82.8)'
- id: H2
  statement: 3xTF32 (hi/lo split, three TF32 tensor-core GEMMs, fp32 accumulate) is fp32-level accurate and 3x faster than the
    reference at L even with a separate split pass and scatter.
  status: parked
  evidence:
  - 'probe: max error vs fp64 4.3e-5 (fp32 cuBLAS 1.6e-5, tolerance atol 2e-3) at B=919; gemms 415 us + split 112 + scatter [probe_b200]'
  - 'run_tests r0: 575 us at 919 vs 1882 reference, 345 at 384 vs 762, 16/16 pass on 10 random rounds [r0-adamod-dispatch-skinny-fp32-3xtf32]'
  - 'portal r0: 688.9 us at 919 scores 0.151 against Tb 169.8 (TF32 speed); retired by the 2026-10-10 TF32 ruling'
- id: H3
  statement: A fused tcgen05 3xTF32 kernel (TMA load, hi/lo split in shared memory, 3 MMAs per k-block, bias + six-way split in the
    epilogue, one launch) removes the 107-112 us split pass and the 7-47 us scatter and reaches max(3 x FLOPs / 588 TF/s, memory floor).
  status: parked
  evidence:
  - 'estimate: 531 us at 919 at the lock vs r0 575 rented (~690 at the lock); 76-222 us at M vs 272-345; ~55-75 us at S vs 256'
  - 'r1: superseded by plain TF32 (ruling); 3 MMAs per k-block can never reach Tb (169.8 us at 919 = 613 TF/s). The fused-epilogue part lives on as H14'
- id: H4
  statement: For B <= 128 one M-tile covers all rows, so a tensor-core TF32 kernel streams W exactly once and the S band is
    memory-bound at the harness floor (about 50 us); cuBLAS TF32 is already there and only the floor itself (H16) can move S further.
  status: supported
  evidence:
  - 'probe: plain cuBLAS TF32 runs 50-51 us at B=5..128 rented, i.e. at the dirty-L2 stream floor (53 us for a plain 227 MB read) [probe_b200]'
  - 'portal r0: Tb 82.8-90.7 us at B<=128, Tsol about 15 us; t = 50 us scores 0.66, t = 40 us 0.73'
- id: H5
  statement: The skinny SIMT kernel (lane = W row, cp.async-staged 128x64 tiles, emb broadcast from smem) is at the copy floor at B=5
    but issue/latency bound at B=16 and 32; 2 CTAs/SM and 2-row-per-lane register tiling recover the 32-64 us FFMA floor.
  status: parked
  evidence:
  - 'probe: 54.6 / 88.8 / 145.4 us at B=5/16/32 (1 CTA/SM, 144 CTAs, 108-129 KB smem); run_tests 60.6 / 88.3 [probe_b200, r0]'
  - 'dead end: skinny<32> loses to cuBLASLt fp32 (122 us) so B=32 stays on the library path in r0'
  - 'portal r0: 76.1 us at B=5 (x1.26 vs rented: it scales with the SM clock, so it is issue-bound, not DRAM-bound); cuBLAS TF32 at 50 us beats it. Parked'
- id: H6
  statement: cuBLAS BF16x9 emulation is a legitimate one-call fp32-level path (error 3.4e-6) but is slower than 3xTF32 at L and falls
    back to SIMT below B~64.
  status: refuted
  evidence:
  - 'probe: 117 / 131 / 177 / 358 / 754 us at B=5/32/128/384/919, maxerr 3.4e-6; r0 3xTF32 path 575 at 919 [probe_b200]'
  - 'operator ruling 2026-10-10: BF16x9 emulation stays forbidden; and it is 4x slower than Tb at 919 anyway'
- id: H7
  statement: SOLAR's Tsol for this problem uses the FP32 CUDA-core peak, so 3xTF32 designs can go below Tsol at M/L (scores above
    0.5 come cheaply there and the S band decides the ranking).
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
  statement: The separate scatter (7-47 us) and any tmp[B,18432] round trip are a large fraction of Tb at M/L; a fused epilogue that
    adds the bias and picks the output tensor by column chunk gains that much before touching the GEMM.
  status: supported
  evidence:
  - 'probe: scatter 47 us at 919 rented = 28% of Tb (169.8); 7 us at 131 = 7% of Tb [probe_b200, r0]'
  - 'constraint: outputs are six separately allocated tensors, so a plain cuBLAS GEMM cannot write them without a scatter or a batched pointer-array mode (E1 probes it)'
- id: H10
  statement: The dirty-L2 flush makes the harness stream floor 4.2-4.8 TB/s (53-64 us) rather than 8 TB/s; evict_first on the W stream
    and evict_last on outputs recover part of it for the S band.
  status: open
  evidence:
  - 'brief: plain stream floor 53.5 us for 227 MB; #38 gained 8-9% from evict_last at M sizes'
  - 'r1 model: the flush leaves ~126 MB of dirty zero lines; every W line allocated evicts one, so DRAM traffic is ~227 + 126 MB = 353 MB, i.e. 44 us at 8 TB/s and 50 us at 7 TB/s. Hints change which lines go, not how many; E3 measures a clean-L2 control'
- id: H11
  statement: cuBLASLt CUBLAS_COMPUTE_32F_FAST_TF32 with a bias epilogue is a 16/16-correct path at 50-135 us rented (about 55-177 us
    at the lock) and scores about 0.5-0.55; cuBLAS 13's batched pointer-array mode over the six 3072-wide chunks removes the scatter
    at no GEMM cost because the tile count is unchanged.
  status: open
  evidence:
  - 'probe r0: cuBLAS TF32 50/51/74/135 us at B=5/128/384/919 rented, maxerr 1.6e-3 vs atol 2e-3 [probe_b200]'
  - 'emulator: S/M/L = 55/100/171 us portal gives 0.5255 vs 0.2349 now [predict_score]'
  - 'dead end (r0) was six separate fp32 SIMT launches (298 us at B=5); a batched TF32 launch with six problems has the same 576 tiles at 919 as the big GEMM'
- id: H12
  statement: TF32 tcgen05 with fp32 operands is shared-memory-bandwidth bound in 1SM mode (TMA writes 12 KB and the MMA reads 12 KB
    per 128-clk 128x256x8 MMA = 192 B/clk against 128 B/clk), which caps it at about 67% of the 909 TF/s lock-time peak, exactly
    cuBLAS's 613 TF/s; cta_group::2 (256x256 tiles) halves the per-SM operand traffic and reaches 85-95% (about 125 us at 919 at the lock).
  status: open
  evidence:
  - 'probe r0: cuBLAS TF32 770 TF/s unlocked at 919 = 65-67% of the 1.15 PF/s TF32 peak at 1965 MHz [probe_b200]'
  - 'portal r0: Tb 613 TF/s at the lock = 67% of 909 TF/s'
  - 'E2 tests it: CUTLASS sm100 1SM 128x256 vs 2SM 256x256 at B=919, and the kernel names cuBLAS picks per B (2sm or not)'
- id: H13
  statement: Re-reading W from L2 once per 128- or 256-row M-tile (4-8 x 227 MB at 919, 0.9-1.8 GB at 16-21 TB/s = 45-110 us) is the
    second limiter at L; a cluster of 2-4 CTAs along M with TMA multicast of the W tile cuts it to 1-2 passes.
  status: open
  evidence:
  - 'estimate only; E2 compares cluster (2,1,1) vs (4,1,1) multicast with 2SM tiles at B=919'
- id: H14
  statement: A one-launch CUTLASS sm100 TF32 kernel with the bias and the six-way split in the epilogue (grouped/ptr-array GEMM over the
    six chunks, or an epilogue that picks the output pointer by N-tile since 3072 = 12 x 256) removes the scatter (H9) and the tmp
    round trip and reaches max(FLOPs / 800 TF/s, dirty-L2 floor): about 50 / 55 / 70 / 125 us at B=128/384/449/919 at the lock,
    score about 0.65.
  status: open
  evidence:
  - 'emulator: S/M/L = 47/57/102 us portal gives 0.667 [predict_score]'
  - 'compile-time risk: CUTLASS sm100 kernels take 1-3 min to build; the 300 s per-solution limit applies'
- id: H15
  statement: Portal/rented time ratios on #35 are 1.26-1.31 for SM-clock-bound paths (tensor core, issue-bound SIMT) and 1.0-1.15 for
    DRAM-bound paths; the emulator must use the path's bound, not the language, to predict the portal time.
  status: open
  evidence:
  - 'portal vs rented r0: skinny B=5 76.1/60.6 = 1.26; cuBLASLt fp32 B=32 142.3/123.7 = 1.15; 3xTF32 B=919 688.9/574.6 = 1.20 (= 1.31 on the 415 us of GEMM + ~1.0 on the 160 us of split/scatter)'
  - 'E3 tries nvidia-smi -lgc 1500 on the rented card to measure the ratio directly for a pure stream and for cuBLAS TF32'
- id: H16
  statement: Any kernel that reads W once pays the dirty-L2 write-back (about 126 MB) on top of 227 MB, so the S/M floor under harness
    conditions is 45-53 us and S scores are capped near 0.65-0.70; only reads with no L2 allocation (none exist) or discard (forbidden)
    could lower it.
  status: open
  evidence:
  - 'brief: 53.5 us measured stream floor rented for 227 MB (4.2 TB/s); cuBLAS TF32 50 us at B=5'
  - 'E3 measures reads-only variants (LDG, evict_first, bulk ring) with the harness flush vs a read-based clean flush; a reads-only time <= 45 us refutes the cap and justifies an S-band custom kernel'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H11",
    "title": "cuBLASLt FAST_TF32 anchor with bias epilogue; one launch via batched pointer-array over the six chunks",
    "operation": "structural_mutation",
    "parents": ["r0-adamod-dispatch-skinny-fp32-3xtf32"],
    "band": "all",
    "instructions": "1) Strip r0 to one path for every B: cuBLASLt CUBLAS_COMPUTE_32F_FAST_TF32 (fp32 in/out, fp32 accumulate) with CUBLASLT_EPILOGUE_BIAS into tmp[B,18432] plus the existing scatter6 kernel (2 launches). Keep the plan cache keyed on B only; keep the column-major TN view from r0. 2) Probe the one-launch fusion: a batched cublasLtMatmul with batch count 6 where problem c is M=B, N=3072, K=3072, A = W + c*3072*3072 (strided batch, stride 3072*3072 floats), bias = bias + c*3072 (CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 3072), and D given as a pointer array (CUBLASLT_MATRIX_LAYOUT_BATCH_MODE = CUBLASLT_BATCH_MODE_POINTER_ARRAY on the D/C layouts, cuBLAS 13.x; device array of the six output pointers rebuilt each call via a small pinned/ device buffer written with cudaMemcpyAsync is NOT allowed since it adds an activity; pass the pointer array in a torch::empty int64 tensor filled on the host before launch only if that fill launches no kernel, otherwise use cublasLt's host-side pointer array if the API accepts one). If pointer-array mode is rejected or mixes badly with strided A, try all-pointer-array (A, B, C, D, bias arrays). Record what the API accepts. 3) For each variant, request 8 heuristics per B, time them interleaved (3 rounds) on the 16 shapes, choose the fastest per shape (shape-only, deterministic), and log the kernel names cuBLAS selects (note whether they contain 2sm / cta_group / cluster hints: evidence for H12). 4) Run run_tests on all 16 workloads x 10 rounds; record max |err| vs an fp64 reference per B. 5) Submit the best 16/16 variant as the r1 portal anchor; record rented times per size and the number of launches in the card paths.",
    "success": "16/16 pass with max error <= 1.7e-3 at every B; rented <= 55 us at B <= 128 and <= 140 us at B = 919; the batched one-launch variant is within 3% of the single big GEMM's time and removes the 7-47 us scatter; predicted portal score >= 0.50.",
    "refuted_if": "any workload exceeds atol under TF32 (margin too thin: report to the operator), or the batched six-problem launch is more than 10% slower than the big GEMM at B = 919 or at B = 5 (then the scatter stays and H9's fusion moves entirely to E2).",
    "model": "sonnet"
  },
  {
    "id": "E2",
    "hypothesis": "H12",
    "title": "CUTLASS 4.4 sm100 TF32 tcgen05 GEMM core: 2SM 256x256 tiles and W multicast, then the fused six-way bias epilogue (H13, H14)",
    "operation": "new_design",
    "parents": ["r0-adamod-dispatch-skinny-fp32-3xtf32"],
    "band": "all",
    "instructions": "Two-file CUDA C++ layout (kernel.cu with CUTLASS, binding.cpp with torch glue); record build time, it must stay well under 300 s. Part A, GEMM core (tests H12/H13): with the Sm100 CollectiveBuilder (OpClassTensorOp, ElementA = ElementB = cutlass::tfloat32_t read straight from the fp32 pointers, ElementAccumulator float, A = emb [B,3072] K-major, B = W [18432,3072] K-major, D row-major [B,18432] fp32 into a tmp tensor, persistent tile scheduler, auto stage count), build and time at B = 5, 128, 384, 919 interleaved with cuBLAS TF32 (rented reference 50/51/74/135 us), 3 rounds: (i) 1SM KernelTmaWarpSpecialized1SmSm100, tile 128x256x32, cluster 1x1x1; (ii) 2SM KernelTmaWarpSpecialized2SmSm100, tile 256x256x32, cluster 2x1x1; (iii) 2SM with cluster 4x1x1 (W tile multicast across two pairs along M); (iv) 2SM tile 256x128x32 if 256x256 spills TMEM or smem stages drop below 3. Report TF/s and the tile/wave counts (M-tiles x 72 N-tiles over 148 SMs). Measure max |err| vs fp64 per B: tcgen05 truncates fp32 to tf32 while cuBLAS may round; if truncation exceeds 1.7e-3 at any B, add cvt.rna.tf32 rounding of the W/emb tiles before the MMA (in-smem transform) and re-measure. Part B, fused epilogue (H14): in order of preference, (a) CUTLASS grouped / ptr-array GEMM (pattern of examples/75_blackwell_grouped_gemm) with six problems M=B, N=3072, K=3072, per-group A offset c*3072*3072, per-group D pointer and per-group bias through the EVT per-column-bias fusion (problem sizes and pointer arrays passed in a device tensor built on the host via torch::empty + cudaMemcpyAsync is an extra activity, so prefer passing them as kernel arguments or in constant memory, or fill the device array inside the kernel's prologue from kernel arguments); (b) a custom epilogue functor/EVT node that maps global column n to (outs[n / 3072], n % 3072) with plain fp32 stores and adds bias[n] (3072 = 12 x 256 so the output pointer is uniform per 256-wide N-tile); (c) fallback: GEMM into tmp + scatter6 (2 launches) using the best Part A core. Correctness on all 16 x 10 rounds; exactly one launch for (a)/(b); include an S-band check that B = 5..128 run at <= 55 us (single M-tile, W streamed once).",
    "success": "Part A: 2SM 256x256 is >= 1.2x faster than 1SM 128x256 at B = 919 (H12 supported) and the best core is <= 115 us at 919 rented; cluster-4 multicast adds >= 5% at 919 (H13). Part B: one launch, 16/16 pass with max error <= 1.7e-3, <= 120 us at 919, <= 80 us at 449, <= 55 us at B <= 128 rented; predicted portal score >= 0.60.",
    "refuted_if": "2SM tiles are within 5% of 1SM at B = 919 (H12 refuted: not smem-bound; look at TMEM/stage limits or L2 instead) and cluster multicast changes 919 by < 3% (H13 refuted); or no CUTLASS configuration beats cuBLAS TF32 by more than 5% at any B (then the fused epilogue alone, worth 7-47 us, is the only gain and E1's batched path is the better vehicle).",
    "model": "opus"
  },
  {
    "id": "E3",
    "hypothesis": "H16",
    "title": "S/M-band memory floor under harness conditions: dirty-L2 write-back cost, cache hints, and the clock-lock ratio",
    "operation": "new_design",
    "parents": [],
    "band": "S",
    "instructions": "Write a standalone probe that reproduces the harness timing protocol on the rented B200 (before every timed call: zero_ a 252 MB buffer, copy inputs to a shifted address, zero the outputs; time with CUDA events around the kernel only; 10 warm-up, 50 timed, median; 3 interleaved rounds). Measure, for W = 226.5 MB: (a) reads-only kernel, plain LDG.128 from 148 x 8 CTAs with >= 64 KB in flight per SM, result reduced into one float per CTA so it cannot be elided; (b) same with ld.global.L2::cache_hint evict_first via createpolicy; (c) a 1-D cp.async.bulk ring, 148 persistent CTAs, 4 x 32 KB stages, mbarrier; (d) variant (c) with LDG.E.256 (256-bit loads, sm_100 only) in place of bulk copies; (e) each of (a)-(c) after a read-based flush (read 252 MB instead of zero_) to isolate the dirty write-back penalty; (f) reads + the six outputs for B = 128 (9.4 MB) with st default vs evict_last vs evict_first. Also time cuBLAS TF32 at B = 5 and 128 under the same protocol as the reference point (expect about 50 us). Then, if permitted on the rented card, run nvidia-smi -lgc 1500,1500 (and -rgc afterwards) and repeat (a), (c) and cuBLAS TF32 at B = 5 and 919 to measure the portal/rented ratio for DRAM-bound vs tensor-core paths (H15); if the lock is not permitted, say so and estimate the ratio from r0's portal/rented pairs. Finally measure the two-launch gap: cuBLAS TF32 GEMM + scatter vs GEMM alone at B = 32 and 919. Report every number in a table with the implied DRAM bandwidth.",
    "success": "A clean quantitative floor: the reads-only time with the harness flush, the clean-L2 time, and their difference. If any reads-only variant reaches <= 45 us with the harness flush, H16's cap is refuted and an S-band custom streaming kernel (TF32 tcgen05 with a single M-tile and a deep TMA ring) becomes a round-2 experiment worth about +0.03.",
    "refuted_if": "all reads-only variants land at 50-55 us with the harness flush while the clean-L2 variants are >= 8 us faster: the dirty write-back is real and unavoidable, H16 is supported, and the S band is closed at cuBLAS/CUTLASS speed (about 0.65 per workload).",
    "model": "sonnet"
  }
]
```