I'm done measuring; writing up the problem card, ledger and the final candidate now.

### Rationale
The op is an fp32 GEMM with M = batch (5–919), N = 18432, K = 3072 plus bias, chunked six ways. The reference is a cuBLAS **SIMT** sgemm already at ~75% of the FFMA peak, so FFMA-only designs cap at ~1.3x at M/L. The only large legal lever is tensor-core fp32 emulation: 3xTF32 (operator-approved) runs 3.3x faster at L with error 4e-5 (fp32 cuBLAS: 1.6e-5). Small B is memory-bound (227 MB weight stream), where a custom skinny SIMT kernel reaches the copy floor at B=5. The first candidate is a shape dispatcher over these three paths; it passed 16/16 twice and is the anchor submission.

```markdown card
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
```

```yaml ledger
# Hypothesis ledger for #35 (035_flux_ada_layer_norm_zero_modulation_extraction). Seeded 2026-10-10 (research round r0).
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
- '#35 precision (operator note, 2026-10-10): exact fp32 only: FFMA on CUDA cores or a 3xTF32 split with fp32-level error.
  Plain TF32 (CUBLAS_COMPUTE_32F_FAST_TF32 alone, allow_tf32) and bf16 operands are not allowed until the operator rules.
  cuBLAS BF16x9 emulation (CUBLAS_COMPUTE_32F_EMULATED_16BFX9) needs an explicit ruling before a portal slot.'
- '#35 dtype is fp32 throughout; the bf16 constraint above is inherited text from #30 and does not apply here.'
hypotheses:
- id: H1
  statement: The reference's cuBLAS SIMT sgemm is within 25-30% of the FFMA peak at M/L, so FFMA-only designs cannot gain more than
    about 1.3x there; packed fma.f32x2 adds no throughput.
  status: supported
  evidence:
  - 'probe: cutlass3x_sm100_simt_sgemm 55-57 TF/s at B=384/919 vs 74.4 TF/s peak at 1965 MHz [probe_b200]'
  - 'probe: FFMA 67.6 TF/s vs FFMA2 (fma.rn.f32x2) 73.5 TF/s on 148x8 CTAs: same 128 FMA/clk/SM datapath [probe_b200]'
- id: H2
  statement: 3xTF32 (hi/lo split, three TF32 tensor-core GEMMs, fp32 accumulate) is fp32-level accurate and 3x faster than the
    reference at L even with a separate split pass and scatter.
  status: supported
  evidence:
  - 'probe: max error vs fp64 4.3e-5 (fp32 cuBLAS 1.6e-5, tolerance atol 2e-3) at B=919; gemms 415 us + split 112 + scatter [probe_b200]'
  - 'run_tests r0: 575 us at 919 vs 1882 reference, 345 at 384 vs 762, 16/16 pass on 10 random rounds [r0-adamod-dispatch-skinny-fp32-3xtf32]'
- id: H3
  statement: A fused tcgen05 3xTF32 kernel (TMA load, hi/lo split in shared memory, 3 MMAs per k-block, bias + six-way split in the
    epilogue, one launch) removes the 107-112 us split pass and the 7-47 us scatter and reaches max(3 x FLOPs / 588 TF/s, memory floor).
  status: open
  evidence:
  - 'estimate: 531 us at 919 at the lock vs r0 575 rented (~690 at the lock); 76-222 us at M vs 272-345; ~55-75 us at S vs 256'
  - 'probe: tcgen05 TF32 770 TF/s (cuBLAS) vs mma.sync TF32 276 TF/s: the kernel must use tcgen05, not sm80-style mma.sync [probe_b200]'
- id: H4
  statement: For B <= 128 one M-tile covers all rows, so a fused tensor-core kernel streams W exactly once and the S band becomes
    memory-bound at ~55 us (4.6x on B=96/128).
  status: open
  evidence:
  - 'probe: plain cuBLAS TF32 (not allowed) already runs 50-51 us at B=5..128, i.e. at the stream floor [probe_b200]'
- id: H5
  statement: The skinny SIMT kernel (lane = W row, cp.async-staged 128x64 tiles, emb broadcast from smem) is at the copy floor at B=5
    but issue/latency bound at B=16 and 32; 2 CTAs/SM and 2-row-per-lane register tiling recover the 32-64 us FFMA floor.
  status: open
  evidence:
  - 'probe: 54.6 / 88.8 / 145.4 us at B=5/16/32 (1 CTA/SM, 144 CTAs, 108-129 KB smem); run_tests 60.6 / 88.3 [probe_b200, r0]'
  - 'dead end: skinny<32> loses to cuBLASLt fp32 (122 us) so B=32 stays on the library path in r0'
- id: H6
  statement: cuBLAS BF16x9 emulation is a legitimate one-call fp32-level path (error 3.4e-6) but is slower than 3xTF32 at L and falls
    back to SIMT below B~64.
  status: parked
  evidence:
  - 'probe: 117 / 131 / 177 / 358 / 754 us at B=5/32/128/384/919, maxerr 3.4e-6; r0 3xTF32 path 575 at 919 [probe_b200]'
  - 'parked until the operator rules on bf16-operand emulation'
- id: H7
  statement: SOLAR's Tsol for this problem uses the FP32 CUDA-core peak, so 3xTF32 designs can go below Tsol at M/L (scores above
    0.5 come cheaply there and the S band decides the ranking).
  status: open
  evidence:
  - 'to be read off the first portal page: Tsol = (Tb - S t - S Tb)/(1 - 2S)'
- id: H8
  statement: The hidden Tb is close to the fp32 reference (~2.4 ms at 919 at the lock), not a TF32-enabled PyTorch solution (~180 us).
  status: open
  evidence:
  - 'probe: torch.matmul with allow_tf32=True takes 135 us at 919 vs 1836 fp32 (rented); if NVIDIA''s agent used it, L scores stay <0.3'
- id: H9
  statement: The split pass (107-112 us, fixed per call) and scatter (7-47 us) are 40% of the M-band time in r0; any kernel with a
    fused epilogue and in-smem split gains that much before touching the GEMM.
  status: supported
  evidence:
  - 'probe: split only 107-112 us; gemms only 137-415 us; run_tests r0 272-345 us at M [probe_b200, r0]'
- id: H10
  statement: The dirty-L2 flush makes the harness stream floor 4.2-4.8 TB/s (53-64 us) rather than 8 TB/s; evict_first on the W stream
    and evict_last on outputs recover part of it for the S band.
  status: open
  evidence:
  - 'brief: plain stream floor 53.5 us for 227 MB; #38 gained 8-9% from evict_last at M sizes'
```

```json solution-spec
{"name": "r0-adamod-dispatch-skinny-fp32-3xtf32",
 "definition": "035_flux_ada_layer_norm_zero_modulation_extraction",
 "author": "solx-loop",
 "description": "Shape dispatch on batch size: B<=16 custom cp.async-staged SIMT skinny GEMM (memory-bound band); 16<B<=64 cuBLASLt fp32 FFMA GEMM with bias epilogue into scratch plus a scatter kernel; B>64 3xTF32 (hi/lo tf32 split kernel, three cuBLASLt TF32 tensor-core GEMMs accumulating in fp32, scatter). All maths fp32-level exact; no plain TF32, no bf16.",
 "spec": {"languages": ["cublas"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcublasLt", "-lcublas"]}}}
```

```cuda file=kernel.cu
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdint>
#include <map>
#include <mutex>

// AdaLayerNormZero modulation extraction: out_c[b, j] = sum_k emb[b,k] * W[c*3072+j, k] + bias[c*3072+j], c = 0..5.
// Shape-only dispatch on B:
//   B <= 8   : skinny<8>   custom SIMT kernel, W streamed once through shared memory (memory-bound band)
//   B <= 16  : skinny<16>
//   B <= 64  : cuBLASLt CUBLAS_COMPUTE_32F (FFMA) GEMM with bias epilogue into tmp[B,18432], then scatter to 6 outputs
//   B >  64  : 3xTF32: W = Whi + Wlo, emb = ehi + elo (hi = round-to-tf32, lo = exact remainder), then
//              tmp = Whi*ehi + bias; tmp += Whi*elo; tmp += Wlo*ehi  (cuBLASLt TF32 tensor-core GEMMs, fp32 accumulate)
//              The dropped Wlo*elo term is ~2^-22 relative: fp32-level error (measured 4e-5 max vs 1.6e-5 for FFMA).

namespace {
constexpr int kD = 3072;        // inner_dim
constexpr int kN = 6 * kD;      // output_dim = 18432
constexpr int kTf32MinB = 65;   // first B served by the 3xTF32 path

// ------------------------------------------------------------------ cuBLASLt plans (shape-keyed)
struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo;
  bool have_algo = false;
};

cublasLtHandle_t g_handle = nullptr;
std::mutex g_mu;
std::map<long, Plan> g_plans;   // key = B * 4 + kind

// kind 0: fp32 FFMA + bias epilogue; kind 1: TF32 + bias epilogue; kind 2: TF32, plain (used with beta = 1)
Plan& get_plan(int B, int kind, size_t ws_bytes) {
  if (!g_handle) cublasLtCreate(&g_handle);
  const long key = (long)B * 4 + kind;
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  Plan p;
  cublasLtMatmulDescCreate(&p.desc, kind == 0 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtEpilogue_t epi = (kind == 2) ? CUBLASLT_EPILOGUE_DEFAULT : CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  // Column-major view: A = W as K x N (ld K), op T; B = emb as K x B (ld K); D = N x B (ld N) == row-major [B, N].
  cublasLtMatrixLayoutCreate(&p.la, CUDA_R_32F, kD, kN, kD);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_32F, kD, B, kD);
  cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, kN, B, kN);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
  cublasLtMatmulHeuristicResult_t res[1];
  int n = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, res, &n) == CUBLAS_STATUS_SUCCESS && n > 0) {
    p.algo = res[0].algo;
    p.have_algo = true;
  }
  cublasLtMatmulPreferenceDestroy(pref);
  return g_plans.emplace(key, p).first->second;
}

cublasStatus_t lt_gemm(Plan& p, const float* A, const float* Bm, const float* bias, float beta, float* D,
                       void* ws, size_t ws_bytes, cudaStream_t stream) {
  const float alpha = 1.f;
  if (bias) cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
  return cublasLtMatmul(g_handle, p.desc, &alpha, A, p.la, Bm, p.lb, &beta, D, p.lc, D, p.lc,
                        p.have_algo ? &p.algo : nullptr, ws, ws_bytes, stream);
}

// ------------------------------------------------------------------ kernels
struct OutPtrs { float* p[6]; };

// tmp[B, 18432] -> six [B, 3072] outputs, 16 B per thread.
__global__ void __launch_bounds__(256) scatter6_kernel(const float4* __restrict__ src, OutPtrs outs, int B) {
  constexpr int kN4 = kN / 4;    // 4608 float4 per row
  constexpr int kD4 = kD / 4;    // 768 float4 per chunk row
  const size_t total = (size_t)B * kN4;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int b = (int)(i / kN4);
  const int n4 = (int)(i - (size_t)b * kN4);
  const int c = n4 / kD4;
  const int j = n4 - c * kD4;
  const float4 v = __ldcs(src + i);
  reinterpret_cast<float4*>(outs.p[c])[(size_t)b * kD4 + j] = v;
}

// x = hi + lo with hi = tf32(x) (round to nearest, 10 explicit mantissa bits), lo = x - hi (exact in fp32).
__device__ __forceinline__ void tf32_split(float x, float& hi, float& lo) {
  unsigned u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  hi = __uint_as_float(u);
  lo = x - hi;
}

__global__ void __launch_bounds__(256) split_tf32_kernel(const float4* __restrict__ src, float4* __restrict__ hi,
                                                         float4* __restrict__ lo, size_t n4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 v = __ldcs(src + i);
  float4 h, l;
  tf32_split(v.x, h.x, l.x);
  tf32_split(v.y, h.y, l.y);
  tf32_split(v.z, h.z, l.z);
  tf32_split(v.w, h.w, l.w);
  hi[i] = h;
  lo[i] = l;
}

// Skinny GEMM for B <= NB. CTA = 128 W rows x all of K; 8 warps = 4 row groups (32 rows, lane = row) x 2 K halves.
// W tile [128 rows x 64 k] (padded rows, 272 B) and emb chunk [NB x 64 k] staged by cp.async, 3 stages.
constexpr int kRows = 128, kKC = 64, kStages = 3, kRowBytes = kKC * 4 + 16;
constexpr size_t kWStage = (size_t)kRows * kRowBytes;

__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g));
}

template <int NB>
__global__ void __launch_bounds__(256, 1) skinny_kernel(const float* __restrict__ W, const float* __restrict__ emb,
                                                        const float* __restrict__ bias, OutPtrs outs, int B) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char* wbuf = smem;
  float* ebuf = reinterpret_cast<float*>(smem + kStages * kWStage);   // [stage][NB][kKC]
  float* red = ebuf;                                                  // reused after the main loop: [NB][kRows]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rg = warp & 3, kh = warp >> 2;
  const int n0 = blockIdx.x * kRows;
  const float* Wb = W + (size_t)n0 * kD;
  const int bmax = B - 1;
  float acc[NB];
#pragma unroll
  for (int b = 0; b < NB; b++) acc[b] = 0.f;
  constexpr int NKC = kD / kKC;   // 48

  auto load_stage = [&](int s, int kc) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int c = tid + i * 256;
      const int r = c >> 4, q = c & 15;
      cp_async16(wbuf + s * kWStage + (size_t)r * kRowBytes + q * 16, Wb + (size_t)r * kD + kc + q * 4);
    }
    for (int c = tid; c < NB * 16; c += 256) {
      const int b = c >> 4, q = c & 15;
      const int bsrc = b < bmax ? b : bmax;   // rows >= B read a valid row; their results are discarded
      cp_async16(ebuf + (s * NB + b) * kKC + q * 4, emb + (size_t)bsrc * kD + kc + q * 4);
    }
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; s++) {
    load_stage(s, s * kKC);
    asm volatile("cp.async.commit_group;");
  }
  for (int it = 0; it < NKC; it++) {
    asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 2));
    __syncthreads();
    const int nxt = it + kStages - 1;
    if (nxt < NKC) load_stage(nxt % kStages, nxt * kKC);
    asm volatile("cp.async.commit_group;");
    const int s = it % kStages;
    const float4* wrow = reinterpret_cast<const float4*>(wbuf + s * kWStage + (size_t)(rg * 32 + lane) * kRowBytes) + kh * 8;
    const float4* eb = reinterpret_cast<const float4*>(ebuf + s * NB * kKC) + kh * 8;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const float4 w = wrow[q];
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const float4 e = eb[b * 16 + q];
        acc[b] = fmaf(w.x, e.x, acc[b]);
        acc[b] = fmaf(w.y, e.y, acc[b]);
        acc[b] = fmaf(w.z, e.z, acc[b]);
        acc[b] = fmaf(w.w, e.w, acc[b]);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  const int r = rg * 32 + lane;
  if (kh == 1) {
#pragma unroll
    for (int b = 0; b < NB; b++) red[b * kRows + r] = acc[b];
  }
  __syncthreads();
  if (kh == 0) {
    const int n = n0 + r;
    const int c = n / kD, j = n - c * kD;   // kD % kRows == 0, so c is uniform per CTA
    float* o = outs.p[c];
    const float bv = bias[n];
#pragma unroll
    for (int b = 0; b < NB; b++)
      if (b < B) o[(size_t)b * kD + j] = acc[b] + red[b * kRows + r] + bv;
  }
}

template <int NB>
int launch_skinny(const float* W, const float* emb, const float* bias, const OutPtrs& outs, int B, cudaStream_t stream) {
  const int smem = (int)(kStages * kWStage + (size_t)kStages * NB * kKC * sizeof(float));
  static bool attr_set = false;   // host-side function attribute, set once per instantiation
  if (!attr_set) {
    cudaFuncSetAttribute(skinny_kernel<NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  skinny_kernel<NB><<<kN / kRows, 256, smem, stream>>>(W, emb, bias, outs, B);
  return (int)cudaGetLastError();
}
}  // namespace

extern "C" int adamod_uses_tf32(int B) { return B >= kTf32MinB; }

// Scratch buffers (allocated by the caller through the PyTorch allocator; no kernels):
//   tmp [B, 18432] and ws when B > 16; whi/wlo [18432, 3072] and ehi/elo [B, 3072] only when adamod_uses_tf32(B).
extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             float* tmp, float* whi, float* wlo, float* ehi, float* elo,
                             void* ws, size_t ws_bytes, cudaStream_t stream) {
  OutPtrs op;
  for (int i = 0; i < 6; ++i) op.p[i] = outs[i];
  if (B <= 8) return launch_skinny<8>(weight, emb, bias, op, B, stream);
  if (B <= 16) return launch_skinny<16>(weight, emb, bias, op, B, stream);

  std::lock_guard<std::mutex> lk(g_mu);
  cublasStatus_t st;
  if (B < kTf32MinB) {
    Plan& p = get_plan(B, 0, ws_bytes);
    st = lt_gemm(p, weight, emb, bias, 0.f, tmp, ws, ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return 1000 + (int)st;
  } else {
    const size_t w4 = (size_t)kN * kD / 4, e4 = (size_t)B * kD / 4;
    split_tf32_kernel<<<(unsigned)((w4 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const float4*>(weight), reinterpret_cast<float4*>(whi), reinterpret_cast<float4*>(wlo), w4);
    split_tf32_kernel<<<(unsigned)((e4 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const float4*>(emb), reinterpret_cast<float4*>(ehi), reinterpret_cast<float4*>(elo), e4);
    Plan& p1 = get_plan(B, 1, ws_bytes);
    Plan& p2 = get_plan(B, 2, ws_bytes);
    st = lt_gemm(p1, whi, ehi, bias, 0.f, tmp, ws, ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return 2000 + (int)st;
    st = lt_gemm(p2, whi, elo, nullptr, 1.f, tmp, ws, ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return 3000 + (int)st;
    st = lt_gemm(p2, wlo, ehi, nullptr, 1.f, tmp, ws, ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) return 4000 + (int)st;
  }
  const size_t total = (size_t)B * (kN / 4);
  scatter6_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const float4*>(tmp), op, B);
  return (int)cudaGetLastError();
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" int adamod_uses_tf32(int B);
extern "C" int adamod_launch(const float* emb, const float* weight, const float* bias, float* const* outs, int B,
                             float* tmp, float* whi, float* wlo, float* ehi, float* elo,
                             void* ws, size_t ws_bytes, cudaStream_t stream);

void run(torch::Tensor emb, torch::Tensor weight, torch::Tensor bias,
         torch::Tensor shift_msa, torch::Tensor scale_msa, torch::Tensor gate_msa,
         torch::Tensor shift_mlp, torch::Tensor scale_mlp, torch::Tensor gate_mlp) {
  TORCH_CHECK(emb.is_cuda() && emb.dtype() == torch::kFloat32 && emb.is_contiguous(), "emb must be contiguous fp32 cuda");
  TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat32, "weight must be contiguous fp32");
  TORCH_CHECK(bias.is_contiguous() && bias.dtype() == torch::kFloat32, "bias must be contiguous fp32");
  TORCH_CHECK(emb.dim() == 2 && emb.size(1) == 3072 && emb.size(0) >= 1, "emb must be [B, 3072]");
  TORCH_CHECK(weight.size(0) == 6 * 3072 && weight.size(1) == 3072, "weight must be [18432, 3072]");
  TORCH_CHECK(bias.numel() == 6 * 3072, "bias must be [18432]");
  const int B = (int)emb.size(0);
  torch::Tensor outs_t[6] = {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  float* outs[6];
  for (int i = 0; i < 6; ++i) {
    TORCH_CHECK(outs_t[i].is_contiguous() && outs_t[i].dtype() == torch::kFloat32 && outs_t[i].size(0) == B &&
                outs_t[i].size(1) == 3072, "bad output tensor");
    outs[i] = outs_t[i].data_ptr<float>();
  }
  const c10::cuda::OptionalCUDAGuard guard(emb.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // Scratch through the caching allocator (torch::empty launches no kernel). Which buffers exist depends on B only.
  torch::Tensor tmp, ws, whi, wlo, ehi, elo;
  float *ptmp = nullptr, *pwhi = nullptr, *pwlo = nullptr, *pehi = nullptr, *pelo = nullptr;
  void* pws = nullptr;
  const size_t ws_bytes = size_t(32) << 20;
  if (B > 16) {
    tmp = torch::empty({(long)B, 6 * 3072}, emb.options());
    ws = torch::empty({(long)ws_bytes}, emb.options().dtype(torch::kUInt8));
    ptmp = tmp.data_ptr<float>();
    pws = ws.data_ptr();
  }
  if (adamod_uses_tf32(B)) {
    whi = torch::empty_like(weight);
    wlo = torch::empty_like(weight);
    ehi = torch::empty_like(emb);
    elo = torch::empty_like(emb);
    pwhi = whi.data_ptr<float>();
    pwlo = wlo.data_ptr<float>();
    pehi = ehi.data_ptr<float>();
    pelo = elo.data_ptr<float>();
  }
  int st = adamod_launch(emb.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), outs, B,
                         ptmp, pwhi, pwlo, pehi, pelo, pws, ws_bytes, stream);
  TORCH_CHECK(st == 0, "adamod_launch failed with status ", st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "AdaLayerNormZero modulation extraction (skinny SIMT / cuBLASLt fp32 / 3xTF32 by batch size)");
}
```

```yaml design-card
id: r0-adamod-dispatch-skinny-fp32-3xtf32
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: cpasync
  st: direct
  grid: oneshot
  launch: split
  tile: rows128
  red: cta
  cache: stream
  spec: dispatch:size
hypothesis: >-
  The op is an fp32 GEMM (M = batch, N = 18432, K = 3072) whose reference is FFMA-bound at 75% of the SIMT peak;
  tensor-core 3xTF32 (hi/lo split, three TF32 GEMMs, fp32 accumulate) is fp32-level accurate and 3x faster at
  M/L, while the small-B band is a 227 MB weight stream that a cp.async-staged skinny SIMT kernel serves at the
  copy floor. First, correctness-first anchor submission for Tb/Tsol.
expected_effect:
  S: {pct: -20, confidence: medium}
  M: {pct: -45, confidence: medium}
  L: {pct: -68, confidence: medium}
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 116736
  threads_per_cta: 256
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 69632
  launches_per_call: 1
knobs: {ROWS: 128, KC: 64, STAGES: 3, NB: [8, 16], TF32_MIN_B: 65, WS_MB: 32}
dispatch:
  - {max_tokens: 8, kernel: skinny_kernel<8>, meta: {launches: 1}}
  - {max_tokens: 16, kernel: skinny_kernel<16>, meta: {launches: 1}}
  - {max_tokens: 64, kernel: cublasLt_fp32_bias + scatter6_kernel, meta: {launches: 2}}
  - {max_tokens: null, kernel: split_tf32 x2 + cublasLt_tf32 x3 + scatter6_kernel, meta: {launches: 6}}
tests: [H1, H2, H5, H9]
findings:
  - "probe: reference = cutlass3x_sm100_simt_sgemm (55-57 TF/s, 74% of the 74.4 TF/s FFMA peak at 1965 MHz) + elementwise bias add (47 us at B=919) [probe_b200]"
  - "probe: packed fma.rn.f32x2 gives 73.5 vs 67.6 TF/s for scalar FFMA: no 2x FP32 lever on sm_100 [probe_b200]"
  - "probe: tcgen05 TF32 770 TF/s (cuBLAS, B=919) vs mma.sync m16n8k8 TF32 276 TF/s and mma.sync bf16 554 TF/s: 3xTF32 must use tcgen05 [probe_b200]"
  - "dead end: six N=3072 cuBLAS GEMMs into the six outputs: 298 us at B=5 and 2050 at 919 vs 104/1836 for one N=18432 GEMM [probe_b200]"
  - "probe: 3xTF32 via pre-split + 3 cuBLASLt TF32 GEMMs: 244/245/324/530 us at B=5/128/384/919 (split pass 107-112 us, gemms 137-415), maxerr 4.3e-5 vs 1.6e-5 fp32 [probe_b200]"
  - "probe: cuBLAS BF16x9 emulation (CUBLAS_COMPUTE_32F_EMULATED_16BFX9): 117/177/358/754 us at B=5/128/384/919, maxerr 3.4e-6; SIMT fallback below B~64 [probe_b200]"
  - "probe: plain TF32 (not allowed): 50-135 us, maxerr 1.6e-3, within 30% of the atol [probe_b200]"
  - "probe: skinny cp.async kernel 54.6/88.8/145.4 us at B=5/16/32 (1 CTA/SM, 144 CTAs); B=32 loses to cuBLASLt fp32 (122 us) [probe_b200]"
  - "run_tests: cuBLASLt fp32 + bias epilogue + scatter alone is reference speed (120-1924 us), 16/16 [run_tests]"
  - "run_tests: this dispatch 60.6/88.3/123.7/255.8/255.5 (S) 271.6-344.7 (M) 370.4-574.6 (L) us, 16/16 on two runs; threshold B>64 vs B>128 gains 2-4% at 96/128 [run_tests]"
  - "static sm_100a: skinny<16> 128 regs, skinny<8> 108 regs, split 17, scatter 20; no spills; skinny smem 108-117 KB (1 CTA/SM) [compile_b200]"
paths:
  - {max_tokens: 8, lang: cuda, width: 128, threads: 256, rows: 128, grid: oneshot, mem: cpasync, launches: 1, x: none, w: none, st: none}
  - {max_tokens: 16, lang: cuda, width: 128, threads: 256, rows: 128, grid: oneshot, mem: cpasync, launches: 1, x: none, w: none, st: none}
  - {max_tokens: 64, lang: cuda, width: 128, threads: 256, rows: 128, grid: oneshot, mem: ldg, launches: 2, x: none, w: none, st: none}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: oneshot, mem: tma, launches: 6, x: ef, w: none, st: none}
runs_on:
  H200: {runs: true, representative: false, note: "cuBLASLt TF32 uses wgmma kernels on sm_90 (different rate); cp.async path representative"}
  A100: {runs: true, representative: false, note: "TF32 mma.sync rate and 2 TB/s HBM; correctness check only"}
  L40S: {runs: true, representative: false}
risks:
  - "3xTF32 path allocates 453 MB of hi/lo scratch per call through the caching allocator and launches 6 kernels; the W split pass (107 us) is 40% of the M-band time"
  - "cuBLASLt heuristic choice per B is deterministic but unverified at the 1500 MHz lock; a split-K pick would add a reduce kernel (constant per shape)"
  - "skinny kernels run 1 CTA/SM on 144 of 148 SMs and are issue-bound at B=16 (88 us vs 54 floor)"
  - "precision rule may change: if plain TF32 or BF16x9 is later allowed, the M/L paths are superseded"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```