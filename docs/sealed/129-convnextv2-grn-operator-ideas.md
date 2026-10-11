# Sealed: the operator's own design ideas for #129 (L2/035_convnextv2_block_with_grn)

Written by Claude (operator) on 2026-10-11, before the loop saw the problem and before any loop changes were built.
Purpose: a fair test of the new lead-driven, ideas-first loop. **No loop model may see this file**: sessions and the
lead run with `--tools ""` and only the solx MCP tools, which cannot read docs/. The operator must not copy these
ideas into any prompt, card, ledger or protocol. Reveal and compare after the experiment (about 4 build rounds).

What I knew when writing: the reference code, the 16 workload shapes and tolerances (atol 2.4-3.8e-3, rtol 1e-5,
98% of elements), the harness input heuristics (W1/W2 ~ randn/sqrt(fan_in), LayerNorm weight ones / bias zeros,
dwconv and GRN params randn), and the public board (s1 0.690, s5 0.619, 16 ranked; SOL ~3 us geomean, scoring
baseline ~160 us geomean). Nothing else: no portal run, no timing, no simulation.

## Observations

- 14 of 16 workloads are small (P = B*H*W from 98 to ~13k pixels); two are big (P = 131k at C=128, 803k at C=96).
  Every workload counts equally, so the small ones decide the score: launches and latency, not FLOPs.
- The 4C-wide intermediate h = GELU(LN(dwconv(x)) W1^T + b1) must never touch DRAM at large P (1.2 GB at P=803k).
- GRN needs a reduction over the whole image (per batch, per channel) between the two GEMMs. That is the crux.

## My ideas, in my order of expected value

1. **Fold GRN into GEMM2.** y = h*(1 + gamma*N_b) + beta, so y W2^T = h (W2 * s_b)^T + (beta W2^T + b2), with
   s_b = 1 + gamma*N_b per batch. GRN becomes a per-batch column scaling of W2 plus a constant bias vector. Also
   fold the LayerNorm affine into W1 / b1 (generic, not assuming ones/zeros).
2. **Two phases, recompute instead of store.** Phase A: dwconv -> LN -> GEMM1 -> GELU -> per-(b, j) sum of h^2, no
   stores. Phase B: recompute dwconv -> LN -> GEMM1 -> GELU, then GEMM2 with the folded W2, + bias + residual x,
   write NCHW. Phase B reads x once for both the conv halo and the residual.
3. **One launch, grid-wide barrier between the phases** (own counter in global memory, persistent grid sized to
   be co-resident). Saves a ~2 us launch on the 14 small workloads.
4. **Small P: keep h on chip.** For P up to ~1.5k, each SM keeps its GEMM1 accumulator tile in TMEM across the
   barrier, so phase B does no recompute: TMEM -> GELU -> scale -> fp16 SMEM -> GEMM2.
5. **FP16 operands with fp32 accumulate** for both GEMMs (inputs are O(1) after LN, inside fp16 range). Simulate
   the error first; if GEMM2 is marginal, keep GEMM1 in FP16 and give GEMM2 a TF32 or split path.
6. **dwconv + LN as the GEMM1 A-operand producer warps**: 7x7 depthwise in NCHW planes, transpose through SMEM to
   pixel-major, LayerNorm per pixel, write the fp16 UMMA tile directly. Weights converted fp32 -> fp16 in-kernel
   (no caching across calls).
7. **Size dispatch**: tiny-P on-chip single kernel / medium two-phase persistent / large two-phase with 2SM tiles.
   At mid sizes (x up to ~70 MB) phase B re-reads x from L2.

## Long shots

8. **Phase A in FP8.** N_b = G_b / mean_c(G_b) is a ratio of sums over many pixels, so per-element FP8 rounding
   averages out and any uniform relative bias cancels in the ratio. Phase A could run at 2x the FP16 rate.
   Legitimate only if the final output still meets the tolerance.
9. **Stream-K / split over pixels for phase A** with atomics into the (b, j) sums, so phase A always fills all
   148 SMs even at P = 98.

## My prediction

- A clean version of 1-5 + 7 beats the current leader (0.690); I would guess 0.72-0.78.
- The loop, after an ideas round and ~4 build rounds: 0.64-0.70 (top 5 likely, rank 1 unlikely).
