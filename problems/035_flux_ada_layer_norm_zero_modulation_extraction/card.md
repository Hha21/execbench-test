# Problem card: L1 #35 `035_flux_ada_layer_norm_zero_modulation_extraction`

Definition from `definition.json` (FLUX.1-Kontext AdaLayerNormZero modulation). Measurements on the rented B200
(unlocked, ~1965 MHz SM; the portal locks 1500 MHz) on 2026-10-10. No portal result yet: Tb and Tsol unknown.

## 1. Semantics

Reference (verbatim):

    emb_out = torch.matmul(emb, weight.t()) + bias          # [B, 18432]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=1)

- DPS signature: `run(emb, weight, bias, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
- `emb` `[B, 3072]` fp32; `weight` `[18432, 3072]` fp32 (226.5 MB, read once); `bias` `[18432]`; six outputs
  `[B, 3072]` fp32, **separately allocated** (chunk c = columns c·3072 … (c+1)·3072 of the GEMM result).
- It is a "TN" GEMM with M = B, N = 18432, K = 3072, fp32 in/out, plus a bias epilogue and a 6-way column split.
  Only B varies. N = 6 × 3072 = 144 × 128; K = 48 × 64; B is arbitrary (5 … 919, several odd values).
- Because the outputs are six separate tensors, a single library GEMM cannot write them directly: either six
  N = 3072 GEMMs (3x slower at small B: too few tiles) or one GEMM into a `[B, 18432]` scratch plus a scatter, or a
  custom kernel whose epilogue picks the output pointer by column chunk (c = n / 3072 is uniform per 128-row tile).

## 2. Numerics and tolerance

- Tolerances: `max_atol` 0.002–0.0023 (grows with B), `max_rtol` 1e-5, matched ratio 0.99 [workload.jsonl].
- Inputs: emb ~ N(0,1), weight ~ N(0,1) (the harness draws randn for all three; our probes used weight/√3072),
  bias ~ N(0,1). With randn weights the dot products have std √3072 ≈ 55, so rtol 1e-5 is ~5e-4 absolute: the atol
  is the binding term only near zero.
- Operator rule (task note): exact fp32 for now: FFMA on CUDA cores, or a **3xTF32** split with fp32-level error.
  Plain TF32 is not allowed until ruled on. Measured max error vs fp64 (weight/√3072 scale, B = 919):
  fp32 cuBLAS 1.6e-5; 3xTF32 (hi·hi + hi·lo + lo·hi, fp32 accumulate) 4.3e-5; cuBLAS BF16x9 emulation 3.4e-6;
  plain TF32 1.6e-3 (would scrape the atol). 3xTF32 is fp32-level by the operator's definition.
- 3xTF32 construction: hi = `cvt.rna.tf32.f32`(x) (11 significant bits), lo = x − hi exact in fp32; the dropped
  lo·lo term is 2^-22 relative. The two cross terms are rounded to TF32 inside the MMA (2^-11 of 2^-11).
- cuBLAS 13.2 also exposes `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` (BF16x9 emulation, one call, fp32-exceeding
  accuracy). Same class as 3xTF32 but needs the operator's ruling (bf16 operands internally).

## 3. The 16 workloads (sorted by size)

Bytes = W 226.5 MB + emb B·12 KB + bias 72 KB + outputs 6·B·12 KB. FLOPs = 2·B·18432·3072 = B × 113.25 MFLOP.
`ffma` = FLOPs at the 1500 MHz FFMA peak (148 × 128 × 2 × 1.5 GHz = 56.8 TF/s); `3xtf32` = 3 × FLOPs at the
measured tcgen05 TF32 rate scaled to 1500 MHz (770 TF/s × 1500/1965 = 588 TF/s → 196 TF/s effective);
`mem8` = bytes at 8 TB/s. Floor / reference are the rented measurements from the brief; `r0` is this candidate.

| B | band | MB | GFLOP | ffma µs | 3xtf32 µs | mem8 µs | stream floor µs | reference µs | r0 µs | r0 path |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | S | 227.0 | 0.57 | 10 | 2.9 | 28.4 | 53.5 | 114.4 | 60.6 | skinny<8> |
| 16 | S | 227.9 | 1.81 | 32 | 9.2 | 28.5 | 53.7 | 117.5 | 88.3 | skinny<16> |
| 32 | S | 229.3 | 3.62 | 64 | 18.5 | 28.7 | 53.8 | 118.3 | 123.7 | cuBLASLt fp32 + scatter |
| 96 | S | 234.8 | 10.9 | 191 | 55 | 29.4 | 54.7 | 250.2 | 255.8 | 3xTF32 (6 launches) |
| 128 | S | 237.6 | 14.5 | 255 | 74 | 29.7 | 55.0 | 251.5 | 255.5 | 3xTF32 |
| 131 | M | 237.8 | 14.8 | 261 | 76 | 29.7 | 53.1 | 407.4 | 271.6 | 3xTF32 |
| 192 | M | 243.1 | 21.7 | 383 | 111 | 30.4 | 55.5 | 437.1 | 276.3 | 3xTF32 |
| 211 | M | 244.7 | 23.9 | 421 | 122 | 30.6 | 55.7 | 479.9 | 278.1 | 3xTF32 |
| 373 | M | 258.7 | 42.2 | 744 | 215 | 32.3 | 57.5 | 762.2 | 342.7 | 3xTF32 |
| 384 | M | 259.6 | 43.5 | 766 | 222 | 32.5 | 57.5 | 762.4 | 344.7 | 3xTF32 |
| 449 | L | 265.2 | 50.8 | 895 | 259 | 33.1 | 57.8 | 946.0 | 370.4 | 3xTF32 |
| 512 | L | 270.6 | 58.0 | 1021 | 296 | 33.8 | 59.0 | 949.6 | 377.8 | 3xTF32 |
| 691 | L | 286.0 | 78.3 | 1378 | 399 | 35.8 | 60.9 | 1386.4 | 469.6 | 3xTF32 |
| 773 | L | 293.1 | 87.5 | 1541 | 447 | 36.6 | 61.8 | 1631.7 | 551.7 | 3xTF32 |
| 853 | L | 299.9 | 96.6 | 1701 | 493 | 37.5 | 62.8 | 1879.5 | 565.7 | 3xTF32 |
| 919 | L | 305.6 | 104.1 | 1832 | 531 | 38.2 | 63.6 | 1882.4 | 574.6 | 3xTF32 |

## 4. Bounds and what dominates per band

- **FFMA SIMT is the reference's regime and it is nearly saturated.** cuBLAS's `cutlass3x_sm100_simt_sgemm` runs
  at 55–57 TF/s unlocked (74% of the 74.4 TF/s peak at 1965 MHz); at the lock it will be ~2.4 ms at B = 919.
  Packed `fma.rn.f32x2` does **not** double FP32 throughput (73.5 vs 67.6 TF/s measured): FFMA peak is
  128 FMA/clk/SM. So any FFMA-only design is capped at ~1.3x over the reference for B ≥ 64.
- **Tensor-core fp32 emulation changes the regime.** tcgen05 TF32 runs 770 TF/s unlocked (cuBLAS, B = 919);
  legacy `mma.sync` TF32 only 276 TF/s (bf16 554), so sm80-style CUTLASS 3xTF32 kernels would reach ~92 TF/s
  (1.6x) while tcgen05 3xTF32 reaches ~257 TF/s unlocked (4.5x). Crossover between memory and 3xTF32 compute at
  the lock is B ≈ 150: **S band is memory-bound, M and L are tensor-core-bound** under 3xTF32.
- **Memory floor.** The harness stream floor is 53–64 µs for 227–306 MB (4.2–4.8 TB/s, dirty-L2 flush before
  each call), against 28–38 µs at 8 TB/s. Every workload must read the 226.5 MB weight once; the outputs are
  0.4–68 MB.
- **S band (B ≤ 128).** FFMA floor 10–255 µs; the skinny kernel hits the stream floor at B = 5 (60.6 µs) but is
  issue/latency-bound at B = 16 (88 µs vs a 32 µs FFMA floor at the lock). From B ≈ 30 FFMA alone exceeds the
  memory floor; only tensor-core emulation can reach ~55 µs at B = 96/128 (currently 255 µs: 4.6x the floor).
- **M band (131–384).** The r0 path is the fixed split pass (107 µs: read 227 MB, write 453 MB) plus three
  memory- to compute-bound TF32 GEMMs (50–74 µs each) plus scatter: 272–345 µs, of which ~110 µs is pure
  split overhead. A fused kernel's bound is max(3xtf32, floor) ≈ 76–222 µs.
- **L band (449–919).** 3 × TF32 GEMM = 405 µs at B = 919 unlocked (3.0 GEMM-equivalents at 770 TF/s), plus
  112 µs split and 47 µs scatter = 575 µs. Fused bound ≈ 531 µs at the lock (405 × 1.31).
- **SOLAR guess.** If Tsol = FLOPs / FP32-CUDA-core peak (80 TF/s datasheet) then Tsol(919) ≈ 1.3 ms and a
  3xTF32 design sits *below* SOL at L; if it uses the TF32 tensor peak (1.1–2.2 PF/s) then Tsol(919) ≈ 50–95 µs
  and L scores stay low. The first portal result decides (H7).

## 5. Where the reference loses and what a fast kernel must do

- Reference = cuBLAS SIMT sgemm (1981 µs at 919) + an elementwise bias add over `[B, 18432]` (47 µs at 919,
  3.5 µs at 32). The chunk is a view, free. At small B the sgemm is tile-starved (104 µs at B = 5 for 0.57 GFLOP:
  5 TF/s) and at large B it is FFMA-bound.
- A fast kernel: tensor-core 3xTF32 (tcgen05, TF32 kind, fp32 accumulate) with the hi/lo split done **in shared
  memory after the TMA load** (no 453 MB scratch, no split pass); bias and the six-way output split in the epilogue
  (c = n / 3072 is uniform per N-tile of 128 or 256); one launch; for B ≤ 128 a single M-tile covers all rows so W
  streams exactly once at the memory floor; for B > 128 W is re-read per M-tile from L2 (≤ 8 × 227 MB, fine).
- Things measured not to work: six N = 3072 cuBLAS GEMMs (298 µs at B = 5, 2050 at 919); skinny SIMT at B = 32
  (145 µs vs 122 cuBLAS); FFMA2 as a throughput lever.

## 6. Design priorities

1. **Fused tcgen05 3xTF32 GEMM with chunked epilogue (all bands).** Removes 110–160 µs of split + scatter per
   call and reaches the memory floor at S: predicted ~55–75 µs at B ≤ 128 (3.4–4.6x on 5 workloads), ~80–230 µs
   at M (1.5–3.4x), ~410–530 µs at L (1.1–1.4x). Build from CUTLASS 4.4 sm100 collectives (TF32 `tcgen05.mma`,
   TMA, TMEM accumulator) with a custom smem transform stage, or a hand-written tcgen05 kernel.
2. **S band skinny kernel tuning** (cheap, while 1 is built): 2 CTAs/SM (64 rows, 2 stages), lane-level 2-row ×
   B tiling, 256-bit loads; target the 54 µs floor at B = 16 and ~70 µs at B = 32.
3. **Ask the operator about BF16x9** (`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`): one call, error 3.4e-6, 754 µs at
   919 (slower than r0's 575 but no 453 MB scratch); only useful if 1 slips.
4. Cache hints: `evict_first` on the weight stream and `evict_last` on outputs (dirty-L2 flush, as in #38).
5. Keep the cuBLASLt fp32 path as the fallback for any B the specialists do not cover.

## 7. Results so far (rented B200, run_tests, µs; 16/16 correct)

| kernel | 5 | 16 | 32 | 96 | 128 | 131 | 192 | 211 | 373 | 384 | 449 | 512 | 691 | 773 | 853 | 919 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (brief) | 114 | 117 | 118 | 250 | 251 | 407 | 437 | 480 | 762 | 762 | 946 | 950 | 1386 | 1632 | 1879 | 1882 |
| cuBLASLt fp32 + bias + scatter | 120 | 121 | 122 | 261 | 265 | 418 | 451 | 497 | 792 | 794 | 975 | 980 | 1421 | 1669 | 1922 | 1924 |
| **r0 dispatch (this submission)** | 61 | 88 | 124 | 256 | 256 | 272 | 276 | 278 | 343 | 345 | 370 | 378 | 470 | 552 | 566 | 575 |

Probe-level numbers (harness_time, same GPU): skinny<32> 145 µs at B = 32; cuBLAS TF32 (plain, not allowed)
50/51/74/135 µs at B = 5/128/384/919; BF16x9 117/177/358/754 µs; 3xTF32 gemms only 137/142/213/415 µs,
split pass 107–112 µs.

## 8. Open questions

- Tsol: FP32 CUDA-core peak or TF32 tensor peak? Tb: the fp32 reference (~2.4 ms at 919 at the lock) or a
  TF32-enabled PyTorch solution (~180 µs)? Both from the first portal page.
- Will the operator allow BF16x9 (cuBLAS emulation) and, later, plain TF32? Plain TF32's max error 1.6e-3 is
  within 30% of the atol at this input scale: risky even if allowed.
- CUTLASS 4.4 sm100: is there a mainloop hook to transform A/B tiles in smem before `tcgen05.mma` (needed for
  in-kernel 3xTF32), or must the kernel be hand-written?
- How does the dirty-L2 flush interact with a 453 MB scratch write (r0 M/L paths)? Fused kernel removes it.
- The scoring clock lock: tensor-core paths scale 1.31x, memory paths ~1.0–1.1x; the S-band gain of a fused kernel
  will look larger on the portal than here.
