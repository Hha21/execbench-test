# Compute playbook: GEMM-like problems on B200 (sm_100a, tcgen05)

This is for problems with a FLOP formula in sol.yaml (GEMMs, projections, fused GEMM + epilogue). It sits alongside
`tricks.md` (verified cross-problem lessons) and `b200_arch.md` §9.

Most numbers here come from our own #35 / #30 / #009 rounds. A few come from public write-ups, cited inline:
- [modular-p3/p4]: Modular, "Matrix Multiplication on Blackwell" parts 3-4, Sept 2025.
- [kernelarc]: arXiv 2608.17071.

## 0. Decide the precision and the ceiling first

1. Read the LEADER GAP section. It gives the best mean score reachable at each precision's tensor rate. If the
   leader's score needs a faster rate than the precision you are using, change the precision before tuning anything.
2. Tensor rates on sm_100, per SM per clock, dense:
   - kind::tf32 is 1x.
   - kind::f16 (FP16/BF16 operands, fp32 accumulate) is 2x.
   - kind::f8f6f4 is 4x.
   - Peaks at the portal lock (1500 MHz × 148 SMs): kind::tf32 4,096 FLOP/clk/SM = 0.91 PF/s, kind::f16 8,192 =
     1.82 PF/s, kind::f8f6f4 16,384 = 3.64 PF/s. Per 64-k block of a 128 × N tile, the kind::f16 MMA ideal is 2·N clk
     (N = 464: 928 clk). Achieved, for scale: cuBLAS bf16 1.47 PF/s at 1500 MHz (#30); #35 TF32 tc_ws ~0.62 PF/s.
     Do not use an achieved rate as a ceiling: 1.3 PF/s for FP16 (our TF32 rate × 2) misled #35 r10-r11.
   - SOLAR's Tsol uses ~1.8 PF/s.
3. Check the tolerance arithmetic before writing code (tricks.md §1):
   - FP16 operands equal TF32 accuracy when the inputs fit FP16's range.
   - BF16 is ~8x worse.
   - Convert fp32 to fp16 with round-to-nearest (`cvt.rn.f16x2.f32`).
4. fp32 inputs still cost 4 bytes each in DRAM, whatever precision the maths uses. So a memory-bound size (a skinny
   GEMM with small M) gains nothing from a lower precision. Only the compute-bound sizes do.

## 1. Structure ladder (each step was measured to matter)

The ladder runs from naive to cuBLAS-class. [modular-p3/p4] gives this progression at 4096³ BF16:

| Step | TFLOP/s | % of cuBLAS |
|---|---|---|
| Naive tcgen05 + swizzle | ~300 | |
| 2SM MMA + TMA multicast | 360 | |
| Pipelined ring + warp specialisation | 1430 | 81% |
| Double-buffered output | 1493 | 85% |
| Persistent with cluster launch control | 1773 | 100.6% |

1. **Swizzled smem operands with TMA** (128-byte swizzle; a 128-byte row is 32 fp32/tf32 or 64 fp16 elements) and
   UMMA smem descriptors matching the swizzle. Wrong descriptors give silently wrong results: validate on all sizes.
2. **2SM pairs** (`cta_group::2`, cluster 2x1):
   - The UMMA M = 256 is split 128 per SM, and each CTA loads half of the B operand.
   - Only the even CTA issues the MMA.
   - Completion is signalled with the multicast commit (`tcgen05.commit...multicast::cluster` to both CTAs' barriers).
   - This beat 1SM at every size on #30 and #35. The per-SM operand bytes halve, which matters when B is wide.
   - Public caveat: a GPU MODE NVFP4 write-up found 2-CTA mode gave only marginal gains when the tiles were small.
3. **Warp specialisation with a multi-stage ring** (the biggest step):
   - One TMA warp, one MMA warp and four epilogue warps.
   - Full/empty mbarrier pairs per stage, 4-6 stages, as many as smem allows.
   - Make every role **warp-uniform** (tricks.md §5): a single-lane issuer inside divergent code costs ~300-400 clk
     per stage in nvcc's ELECT waterfall. With uniform roles our ring ran at 770 clk per stage against a 768-clk
     ideal.
4. **Stage size**: each stage should carry at least ~600 clk of MMA work, because a fixed per-stage cost of ~565 clk
   remains. Use 2 k-blocks per stage, or 48-k stages built from a 32-k plus a 16-k TMA box, when N is small.
5. **TMEM accumulators**:
   - 512 columns × 128 lanes × 32 bit per SM.
   - Allocate once, relinquish the permit, and keep 1 CTA/SM for large allocations.
   - Two accumulators let the MMA warp start the next tile while the epilogue drains the previous one
     [modular-p4], or cover N up to 512 in one pass (#35 tc_ws).
6. **Epilogue**:
   - TMEM → registers (`tcgen05.ld`) → fused bias, activation, residual or scatter → store.
   - Direct global stores were fine for #35. For big tiles, stage through smem and use TMA stores with ping-pong
     buffers (`commit_group` / `wait_group 1`), which gave +4% [modular-p3].
   - Fuse every elementwise op of the reference into the epilogue, plus any output split: #35's six-way chunk saved
     5-22 µs.
7. **Persistent scheduling**:
   - Use a static tile loop, or cluster launch control (CLC) for hardware-driven tile stealing [modular-p4].
   - This removes per-wave smem and barrier re-initialisation and overlaps the epilogue with the next mainloop.
   - Use it when the tile count is several waves.
8. **Tile shape per size** (tile shape is a per-size dispatch decision):
   - Make the tile count divide 148 SMs or 74 pairs. For example, a 256x224 instruction shape gives exactly 148
     CTAs for M=512, N=8192 [modular-p4].
   - Wave quantisation cost #30 ~13% at L.
   - Pad the batch side to multiples of 16 only, not to 256.
9. **Rasterisation**: order tiles so a wave shares A or B rows in L2, e.g. a zig-zag over N with a small swizzle
   [modular-p4]. For a weight read once (skinny GEMM), give each W tile to exactly one CTA pair and mark its loads
   evict_first.

## 2. Libraries (often enough at S/M; a strong first candidate)

- **cuBLASLt** (`cublasLtMatmul`):
  - Pick the algorithm per size from `cublasLtMatmulAlgoGetHeuristic`, and keep a static per-M table of validated
    algorithms.
  - Fused epilogues: BIAS, GELU, RELU.
  - beta=1 with a C operand fuses a residual add. Reported trick [kernelarc, L1-030]: alias the output onto the
    residual buffer so the residual add happens in the GEMM epilogue (C == D), with a per-M table of algorithms. It
    reached 0.546 on #30, where our C != D attempt picked slower kernels (#30 H10). Not yet tested by us. Check that
    it never writes an input tensor the harness reuses.
  - TF32: `CUBLAS_COMPUTE_32F_FAST_TF32`. FP16 inputs with fp32 compute: convert first, or use the A/B type
    CUDA_R_16F.
- **CUTLASS 4.4 C++ sm100 collectives**: `CollectiveBuilder` with `KernelTmaWarpSpecialized2SmSm100`, cluster 2x1,
  `NoSmemWarpSpecialized2Sm` epilogue. A custom epilogue wrapper can re-point `ptr_D` per tile (#35 r1).
- CuTe DSL (Python, `nvidia-cutlass-dsl` 4.4.2): Blackwell `dense_gemm` / `dense_gemm_persistent` examples. Good for
  quick variants.
- Triton `tl.dot` could not compete on B200 (#009).

## 3. Diagnosis

- Measure in SM cycles (`clock64` around the mainloop in the MMA warp, plus `%globaltimer` for the clock). The rented
  card power-caps to 1.1-1.35 GHz under GEMM load.
- Compute the MMA ideal (FLOPs per SM ÷ rate per clock) and compare clk per k-block with it. If you are above ~1.1x
  the ideal, look in this order:
  1. issue waterfall (SASS: ELECT, BRA.U.ANY)
  2. per-stage floor
  3. operand ingest (bytes landing per SM per clk; 2SM halves B)
  4. DRAM
- SASS checks: count UTCHMMA/UTCBAR and look for ELECT loops, local memory (STL/LDL) and spills.

## 4. Search lenses (diversify when progress stalls)

These lenses come from [kernelarc]: library, memory, compute, fusion, **precision**, reduction, scheduling and
custom-sm100.

When two rounds have each moved the portal score by < 0.005, the next round should include a fundamentally
different option rather than another refinement of the current best:
- another lens
- another precision
- a library path versus a custom kernel
- another data layout or algorithm

## 5. References (public; read through WebFetch when needed)

- Modular, "Matrix Multiplication on Blackwell", parts 1-4 (2025): modular.com/blog. The kernel-by-kernel progression
  includes 2SM, multicast, warp specialisation, output double-buffering and CLC persistence. The code is on GitHub.
- Colfax Research, "CUTLASS tutorial: writing GEMM kernels using tensor memory for NVIDIA Blackwell GPUs" (3 parts):
  tcgen05 and TMEM, then clusters with CTA pairs and multicast, then low precision and block scaling.
- mlc.ai, "Modern GPU programming for MLSys", chapter "Scaling GEMM with warp specialization and clusters".
- CUTLASS `examples/` (70-72 blackwell_gemm*, narrow precision), and `include/cutlass/pipeline/sm100_pipeline.hpp`
  (`PipelineTmaUmmaAsync`).
- DeepSeek DeepGEMM (sm100 FP8 GEMMs). ThunderKittens (Blackwell kernels).
- PTX ISA, tcgen05 section: instruction descriptors, smem descriptors, `kind::` operand types and K per instruction
  (tf32 K=8, f16 K=16, f8 K=32).
- Never read other teams' SOL-ExecBench leaderboard submissions.
