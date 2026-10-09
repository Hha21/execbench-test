# Problem card: L1 #30 `030_attention_output_projection_with_residual`

Definition from `definition.json` (Jamba-Reasoning-3B o_proj). Measurements from the rented B200 (unlocked, SM 1965 MHz;
the portal locks 1500 MHz) on 2026-10-09. No portal result yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim):

```python
@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    projected = torch.matmul(attn_output, o_proj_weight.t())
    return projected + residual
```

- DPS signature: `run(attn_output, residual, o_proj_weight, output)`.
- `attn_output`, `residual`, `output`: `[B, S, 2560]` bf16 contiguous. `o_proj_weight`: `[2560, 2560]` bf16.
- View the activations as `[M, 2560]` with `M = B·S` tokens. `D = A · Wᵀ + R`, a "TN" GEMM with M×N×K = M×2560×2560
  plus an elementwise residual: exactly `cublasLt`'s `D = α·op(A)·op(B) + β·C` with `C = residual`, `D = output`,
  α = β = 1, or CUTLASS's linear-combination epilogue with a TMA-loaded C.
- Only `M` varies; the three `M = 4096` and four `M = 8192` shapes time identically. 16 workloads = 10 sizes.
- N = 2560 = 10 × 256 = 20 × 128; K = 2560 = 40 × 64. M is a multiple of 128 except 586 (2×293), 1571, 4106 (2×2053)
  and 7976 (8×997), so M-tiles need a residue mask (TMA zero-fills out-of-bounds rows for free).

## 2. Numerics and tolerance

- Tolerances per workload: `max_rtol = 0.05`, `max_atol = 0.02` (0.0093 for 4×128), matched ratio 0.99 [workload.jsonl].
- Inputs are N(0,1) bf16, so the dot product over K = 2560 has std ≈ 50.6 and |D| reaches ≈ 250. The bf16 output
  rounding is 2⁻⁸ relative (≈ 0.2 at |D| = 50), inside rtol 0.05 by a factor of 25. fp32 accumulation in any order,
  then one bf16 rounding, is the correct precision: bf16 tensor-core MMA with fp32 accumulate is what the reference does.
- Fused residual: adding `R` in fp32 before the single bf16 rounding is slightly more accurate than the reference's
  two roundings; both pass.
- What would fail: fp16 accumulate, or bf16 accumulation across K (error ≈ 2⁻⁸·√K·|D|); reduced-precision reductions
  (`allow_bf16_reduced_precision_reduction` style split-K accumulating in bf16) are to be avoided.
- The loop's lint regex flags the literal bf16 type name in CUTLASS sources; the dtype is bf16 by problem definition
  (the cuBLASLt source avoids the literal by using `CUDA_R_16BF`).

## 3. The 16 workloads (sorted by size)

Bytes = W (13.11 MB, read once) + 3 × M × 5120 B (A, R read; D written). FLOPs = 2·M·2560². `cmp2.25` = FLOPs at the
datasheet dense bf16 peak of 2.25 PFLOP/s; `cmp1.47` = at the practical peak measured here scaled to the 1500 MHz lock
(1.93 PFLOP/s at 1965 MHz × 1500/1965) [INFERRED]. `mem8` = bytes at 8 TB/s. Floor/ref columns are the rented-B200
measurements from the task brief; `lt` is our first kernel (cuBLASLt fused, run_tests).

| M = B·S | shapes | MB | GFLOP | cmp2.25 µs | cmp1.47 µs | mem8 µs | stream floor µs | reference µs | lt µs | band |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1×256 | 17.0 | 3.36 | 1.5 | 2.3 | 2.1 | 6.05 | 12.3 | 8.7 | S |
| 512 | 4×128 | 21.0 | 6.71 | 3.0 | 4.6 | 2.6 | 6.91 | 14.8 | 10.4 | S |
| 586 | 2×293 | 22.1 | 7.68 | 3.4 | 5.2 | 2.8 | 6.98 | 17.1 | 12.2 | S |
| 1024 | 1×1024 | 28.8 | 13.4 | 6.0 | 9.1 | 3.6 | 8.22 | 17.8 | 13.4 | S |
| 1571 | 1×1571 | 37.2 | 20.6 | 9.2 | 14.0 | 4.7 | 9.68 | 25.4 | 19.6 | S |
| 2048 | 16×128, 1×2048 | 44.6 | 26.8 | 11.9 | 18.2 | 5.6 | 11.0 | 28.2 | 23.3 | M |
| 4096 | 4×1024, 16×256, 8×512 | 76.0 | 53.7 | 23.9 | 36.5 | 9.5 | 17.7 | 48.1 | 39.3 | M |
| 4106 | 2×2053 | 76.2 | 53.8 | 23.9 | 36.6 | 9.5 | 17.9 | 48.4 | 39.1 | L |
| 7976 | 8×997 | 135.6 | 104.5 | 46.4 | 71.1 | 17.0 | 29.8 | 89.8 | 76.9 | L |
| 8192 | 16×512, 8×1024, 64×128, 32×256 | 138.9 | 107.4 | 47.7 | 73.0 | 17.4 | 30.3 | 90.2 | 77.5 | L |

## 4. Bounds and what dominates per band

- Arithmetic intensity grows with M: FLOPs/bytes = 197 at M = 256, 773 at 8192. The B200 ridge at the datasheet peak is
  281 FLOP/B, so on paper only M = 256 is memory-bound. In practice the tensor pipe reaches ≈ 1.93 PFLOP/s at
  1965 MHz (cuBLAS, 5 full waves) and the SM clock lock costs another 24%, so **every size from M ≈ 400 up is
  compute-bound** at the portal, and the L band is pure tensor-core throughput plus wave quantisation.
- S band (M ≤ 1571): the harness floor is 6–10 µs even for a plain copy (17 MB copy = 8.2 µs under cold-L2 CUPTI
  timing here, versus 2.1 µs at 8 TB/s), and cuBLAS sits at 1.4–2.0× that. The GEMM here has few output tiles
  (2–13 M-tiles × 10–20 N-tiles) and 40 serial K-steps per tile, so it is a latency chain: cuBLAS at M = 256 takes
  4.9/6.2/8.1 µs for K = 640/1280/2560. Lever: split the K loop across all 148 SMs.
- M band (2048–4096): 2048 → 80 pair-tiles of 256×256 (1.1 waves of 74 pairs), 4096 → 160 (2.2 waves). Wave
  quantisation and the fixed cost both matter.
- L band (7976–8192): 320 pair-tiles / 74 pairs = 4.32 waves, so a plain 256×256 tiling runs 5 waves at 86% use.
  Measured: cuBLAS 1.64 PFLOP/s at M = 8192 versus 1.88 at M = 9472 (exactly 5 waves) and 1.93 at 18944. **About 13%
  is on the table from stream-K or a tile that divides the work evenly.** Memory traffic (139 MB, 17 µs) is fully
  hidden by compute.
- SOLAR guess: Tsol = max(FLOPs/2.25 PF, bytes/8 TB/s) → 2.1 µs at M = 256, 47.7 µs at 8192. If so, L-band scores are
  capped by the unreachable datasheet peak, as #38's were by the 0.53× floor. Confirmed only by the first portal result.

## 5. Where the reference loses and what a fast kernel must do

- The reference is two kernels: a cuBLAS `nvjet` GEMM (65–72 µs at L, 8.2 at S) and a separate add that reads
  `projected` and `residual` and writes `output` (3 × M × 5 KB): 4 µs at S, 12–18 µs at L. `torch.addmm(..., out=)`
  is no better: a DtoD memcpy of the residual and then the `badd` GEMM variant (same time as the reference).
- A fast kernel: one launch; the residual read in the epilogue (TMA-loaded C tile) and the fp32 accumulator + C rounded
  once to bf16; per-size tiling (small tiles or split-K at S, 2SM 256×256 at L); a scheduler without a partial last
  wave at L; 2-CTA `tcgen05.mma` (cta_group::2) so each SM pair shares B-tile loads.
- Measured with CUTLASS 4.4.1 collective builders (fused residual, persistent scheduler, correct 16/16):
  1SM 128×256×64 → 25 µs at S, 95 at L; 2SM 256×256×64 cluster 2×1 → 18.9 at S, 73.7 at L. The 2SM kernel beats
  cuBLASLt by 5% at L and loses 2.2× at S (20 clusters of 40 serial K-steps). cuBLASLt wins S and M.

## 6. Design priorities

1. **S band: split/stream-K across all SMs.** cuBLAS already uses 80×64 tiles with 2-CTA at M = 256 yet takes 8.7 µs
   against a 6 µs copy floor. A custom kernel with K split 4–8 ways and an in-cluster (DSMEM) or fp32-atomic reduction
   into a single epilogue could reach ≈ 6.5–7 µs. Worth ≈ 20% on 5 workloads.
2. **L band: fix wave quantisation.** Stream-K (CUTLASS `StreamKScheduler`, needs a workspace memset in the window) or
   a 2SM tile whose count divides 74 pairs (e.g. 256×128 → 640 tiles = 8.65 waves, 96% use; 256×160 → 512 tiles,
   6.9 waves) on top of the 2SM 256×256 kernel: target 65 µs here (≈ 85 at the lock) versus 77.5 for cuBLASLt.
3. **M band:** dispatch by M between the S and L specialists; 2048 tokens (1.08 waves) is the worst quantisation case
   and wants stream-K most.
4. Cache hints: `evict_last` on D stores gained 8–9% at M sizes in #38 because of the dirty-L2 flush; try the same on
   the TMA store (`CUTLASS` epilogue has a cache-hint hook) once a custom kernel exists.
5. Keep cuBLASLt as the fallback path in any dispatcher; it is the strongest S/M kernel we have.

## 7. Results so far (rented B200, run_tests, µs)

| kernel | 256 | 512 | 586 | 1024 | 1571 | 2048 | 4096 | 4106 | 7976 | 8192 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (matmul + add) | 12.3 | 14.8 | 17.1 | 17.8 | 25.4 | 28.2 | 48.1 | 48.4 | 89.8 | 90.2 | – |
| **oproj-cublaslt-fused (this submission)** | 8.7 | 10.4 | 12.2 | 13.4 | 19.6 | 23.3 | 39.3 | 39.1 | 76.9 | 77.5 | 16/16 |
| CUTLASS 1SM 128×256×64 persistent | 25.1 | 25.2 | 25.9 | 26.0 | 26.2 | 43.0 | 61.0 | 61.6 | 94.6 | 95.0 | 16/16 |
| CUTLASS 2SM 256×256×64, cluster 2×1 | 18.9 | 19.0 | 19.2 | 19.9 | 21.3 | 32.9 | 48.0 | 48.0 | 73.1 | 73.7 | 16/16 |

## 8. Open questions

- Hidden Tb per workload: is it the reference (no PyTorch-only way to fuse the residual), or something smarter (e.g.
  torch.compile)? The first portal result answers this.
- Does SOLAR use 2.25 PFLOP/s dense bf16 for the compute term? If so the L band is capped near 0.6 even at the
  practical peak.
- How do cuBLAS's heuristics behave at the 1500 MHz lock (same algorithms, compute terms 1.31× slower)?
- Can a split-K kernel at S reduce inside one launch without a second kernel (cluster DSMEM reduction versus fp32
  atomics into a workspace plus an ordered final epilogue)?
- Does the dirty-L2 flush (126 MB) cost the same ≈ 6 µs fixed here as the 17 MB copy suggests, and does `evict_last`
  on D stores recover part of it?
- Is there a CUTLASS stream-K configuration for sm100 2SM kernels that does not add a memset or reduce kernel to the
  timed window?
