# Problem card: L2 #035 `035_convnextv2_block_with_grn`

Definition from `definition.json` (ConvNeXtV2-base block: depthwise 7×7 conv → LayerNorm → 1×1 expand (4×) → GELU → GRN →
1×1 project → residual). Measurements from the rented B200 (unlocked, SM 1965 MHz; the portal locks 1500 MHz) on 2026-10-11.
No portal result yet, so Tb and Tsol per workload are unknown.

## 1. Semantics

- DPS signature: `run(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias,
  grn_weight, grn_bias, pwconv2_weight, pwconv2_bias, eps, layer_norm_eps, output)`. `x`, `output`: `[B, C, H, W]` fp32
  NCHW contiguous. Weights fp32: dw `[C,1,7,7]`, W1 `[4C, C]`, W2 `[C, 4C]`, GRN `[1,1,1,4C]`, biases `[C]`/`[4C]`.
  `eps = layer_norm_eps = 1e-6` (Python floats).
- Maths per token t = (b, h, w), M = B·H·W tokens, K-expansion 4C:
  1. `d[t][c] = Σ_{i,j} x[b][c][h+i-3][w+j-3]·dw[c][i][j] + dwb[c]` (zero padding, 49 taps, 98 FLOP/output).
  2. `a = LayerNorm_C(d)·lnw + lnb` (biased variance, eps 1e-6).
  3. `h = gelu_erf(a @ W1ᵀ + b1)` → `[M, 4C]` (exact erf GELU, not tanh).
  4. GRN: `gf[b][k] = ‖h[b,:,:,k]‖₂` over the image's H·W tokens; `nf = gf / (mean_k gf + eps)`;
     `h' = h·(1 + gw[k]·nf[b][k]) + gb[k]` (an affine per (image, channel): the scale needs the **whole image** first).
  5. `y = h' @ W2ᵀ + b2`, `output = x + y` written back in NCHW.
- The GRN is the only cross-token dependency: nothing of GEMM2 for image b can start until every token of b has gone through
  GEMM1+GELU. Everything else is per token (steps 2, 3, 5) or a 7×7 stencil per channel (step 1).
- Useful algebra: because GRN is affine per channel, `pwconv2(h') = (h ⊙ s_b) @ W2ᵀ + (W2·gb + b2)` with `s_b[k] = 1 + gw[k]·nf[b][k]`;
  the scale can be applied to the A operand (per element), to W2 per image, or `gb` can simply be added to A before GEMM2.

## 2. Numerics and tolerance

- Tolerance per workload: `atol 0.0024–0.0038`, `rtol 1e-5`, matched ratio 0.98 [workload.jsonl]. Since outputs are O(1–10),
  the budget is **≈ atol ≈ 2.4e-3 absolute**, i.e. ~1e-3 relative, far looser than #38.
- Harness generators (io.py, read on the B200): `x` N(0,1); names ending `_weight` with ≥2 dims are **"weight matrices" =
  randn/√fan_in** (dwconv_weight: fan_in = 7, so std 0.38; W1: /√C; W2 and grn_weight: /√4C, so grn_weight ≈ 0.02–0.06);
  `layernorm_weight` = ones, `layernorm_bias` = zeros (norm heuristics); all other biases randn. With these: d ~ N(bias, 7),
  a ~ N(0,1), h rms 1.35 (max ≈ 9), y rms ≈ 1.9 (max ≈ 10). (With all-N(0,1) weights the outputs would be in the thousands
  and only fp32-equivalent maths would pass; this is not what the harness generates.)
- Precision simulation against the fp32 torch reference on 6 workloads × 2 seeds, exact products in fp64 (probe):

| GEMM1/GEMM2 operands | err rms | fraction > tol | verdict |
|---|---|---|---|
| fp32 (ours) | ~5e-7 vs fp64 | 0 | pass |
| TF32 / TF32 (round-to-nearest) | 4.5e-4 | 0.000 % (0.3 % > tol/2) | pass, ~5σ margin |
| FP16 / FP16 | 4.5e-4 | 0.000 % | pass, ~5σ margin, 2× TF32 rate |
| FP16 / fp32 or fp32 / FP16 | 2–4e-4 | 0 | pass |
| BF16 / BF16 | 3.6e-3 | 28–50 % | **fail** |
| FP16 hi/lo split (3 MMAs), BF16 split (3) | 1e-6 / 7e-6 | 0 | fp32-equivalent |

  So **single-pass FP16 tensor cores (kind::f16, fp32 accumulate) are the format for both GEMMs**; the error budget is 5σ
  per element and the 98 % ratio adds a huge second margin. BF16 is out. Converting the dwconv input/output to fp16 (for a
  tensor-core dwconv) would add ≈5e-4 rms (LN removes the scale), still ≈3.5σ: feasible but less margin; check per design.
- fp32 SIMT path: var = E[d²]−mean² in fp32 is fine (mean² ≈ 1 ≪ var ≈ 7). `erff` under fast-math is exact enough; the tanh
  GELU approximation (≈1e-3 abs error) would eat the whole budget: do not use it. GRN sums accumulated with fp32 atomics in any
  order are fine (relative 1e-7).

## 3. The 16 workloads (sorted by size)

Bytes = x read + output written (compulsory; weights ≤ 19 MB for C=768). GEMM FLOP = 16·M·C² (dwconv adds 98·M·C).
`fp32` = GEMM FLOP at the 56.8 TF/s SIMT peak at 1500 MHz (128 FMA/clk/SM; measured FFMA 67.7 TF/s and FFMA2 74.0 TF/s at
1965 MHz, i.e. packed f32x2 does **not** raise the rate). `f16` = at 1.82 PF/s kind::f16. `dw` = dwconv FMA-bound time at 1500 MHz.
Floor/ref from the brief (rented); `anchor` = this submission (run_tests).

| workload | M | C | MB | GEMM GFLOP | fp32 µs | f16 µs | dw µs | floor µs | ref µs | anchor µs | band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,192,28,28 | 784 | 192 | 2.4 | 0.46 | 8 | 0.3 | 0.3 | 2.8 | 157 | 89 | S |
| 1,96,56,56 | 3136 | 96 | 2.7 | 0.46 | 8 | 0.3 | 0.5 | 2.8 | 173 | 65 | S |
| 1,160,449,7 | 3143 | 160 | 4.9 | 1.29 | 23 | 0.7 | 0.9 | 3.4 | 164 | 111 | S |
| 2,96,56,56 | 6272 | 96 | 5.1 | 0.92 | 16 | 0.5 | 1.0 | 3.4 | 192 | 100 | S |
| 4,192,28,28 | 3136 | 192 | 6.1 | 1.85 | 33 | 1.0 | 1.0 | 3.5 | 200 | 140 | S |
| 8,384,14,14 | 1568 | 384 | 9.6 | 3.70 | 65 | 2.0 | 1.0 | 4.5 | 210 | 273 | M |
| 4,96,56,56 | 12544 | 96 | 10.0 | 1.85 | 33 | 1.0 | 2.1 | 4.4 | 226 | 142 | M |
| 8,192,28,28 | 6272 | 192 | 10.9 | 3.70 | 65 | 2.0 | 2.1 | 4.4 | 248 | 218 | M |
| 16,384,14,14 | 3136 | 384 | 14.5 | 7.40 | 130 | 4.1 | 2.1 | 5.3 | 307 | 353 | M |
| 2,768,7,7 | 98 | 768 | 19.7 | 0.92 | 16 | 0.5 | 0.1 | 7.0 | 226 | 124 | M |
| 8,768,7,7 | 392 | 768 | 21.5 | 3.70 | 65 | 2.0 | 0.5 | 7.3 | 223 | 265 | L |
| 16,768,7,7 | 784 | 768 | 23.9 | 7.40 | 130 | 4.1 | 1.0 | 7.7 | 335 | 585 | L |
| 8,320,32,32 | 8192 | 320 | 24.3 | 13.4 | 236 | 7.4 | 4.5 | 6.8 | 596 | 626 | L |
| 32,768,7,7 | 1568 | 768 | 28.7 | 14.8 | 260 | 8.1 | 2.1 | 8.5 | 478 | 1217 | L |
| 32,128,64,64 | 131072 | 128 | 134.8 | 34.4 | 605 | 18.9 | 29 | 24.5 | 2635 | 1650 | L |
| 64,96,112,112 | 802816 | 96 | 118.0 G | 118 | 2080 | 65 | 134 | 99.2 | 11525 | 5596 | L |

## 4. Bounds and what dominates per band

- **fp32 SIMT is 10–30× too slow everywhere**: even at the 57 TF/s peak the GEMMs take 8 µs (smallest) to 2.1 ms (largest),
  against stream floors of 2.8–99 µs. With FP16 tensor cores (allowed, §2) the GEMMs cost ≤ 19 µs except the largest (65 µs),
  i.e. **every workload becomes memory-/latency-bound**, like #38 but with a grid-wide dependency in the middle.
- The depthwise conv then becomes the residual compute term: 49 FMA per output, FMA-bound 134 µs per pass at 1500 MHz for
  64,96,112,112 (> the 99 µs stream floor), 29 µs for 32,128,64,64, ≤ 5 µs elsewhere. Any design that recomputes it twice
  pays this twice at the top two sizes.
- Weights: W1+W2 = 8·C² ·4 B = 19 MB at C=768 (two thirds of those workloads' traffic); each m-tile re-reads them from L2.
- S band (≤ 6 MB): floors 2.8–3.5 µs; launch/barrier overhead measured on the rented card: empty kernel 1.8 µs, +2.0 µs per
  extra launch, atomic grid barrier with 296 CTAs ≈ 1.5 µs (≈ 5 µs with 1184 CTAs). Two barriers ≈ two extra launches.
- SOLAR guess: Tsol = max(FLOPs/peak, bytes/8 TB/s); whether "peak" is the fp32 SIMT rate (then Tsol ≈ 2 ms at the top and
  tensor-core kernels beat it) or a tensor rate is unknown until the first portal result.

## 5. Where the reference loses (torch.profiler on the B200)

- 17–18 GPU activities per call: cuDNN `conv_depthwise2d_forward_kernel_generic` (492 µs of 2635 at 32,128,64,64; 11 µs at
  8,768,7,7), a permute copy, `vectorized_layer_norm` (176 µs), two `cutlass3x_sm100_simt_sgemm` fp32 GEMMs (400+346 µs at
  32,128,64,64, 60+81 µs at 8,768,7,7; 64×32 tiles at small M run at 8–9 TF/s, 128×128 at 44–51 TF/s), bias add, GELU,
  `NormTwoOps` reduce, mean reduce, six elementwise kernels for the GRN affine (≈190 µs each at 32,128,64,64 = 1 ms total!),
  residual add, output permute. Roughly: 40 % elementwise/permute traffic, 30 % GEMM, 20 % dwconv, 10 % LN/reductions.
- cuBLAS fp32 SIMT GEMM alone (warm, both GEMMs) is 36–52 µs at S sizes and 2758 µs at the largest, so a fused fp32 kernel can
  never beat ≈1/3 of the reference at L; the gains at L come only from tensor cores.

## 6. Decisions any design must make (options, with arithmetic; nothing chosen)

1. **Where the GRN barrier goes.** (a) One cooperative kernel with grid-wide syncs (296 CTAs: ≈1.5 µs per sync, requires
   `cudaLaunchCooperativeKernel`; cg `grid.sync()` links without `-rdc` on nvcc 13). (b) Separate launches (+2 µs each).
   (c) Per-image dataflow: persistent CTAs process GEMM2 tiles of image b once an image counter reaches its GEMM1 tile count
   (no global barrier, needs a self-resetting counter workspace; images are processed in order so the pipeline fills).
   The two-phase structure (GEMM1 for all tokens → norms → GEMM2) is inherent; (c) only hides the wait.
2. **Hidden activations h [M,4C]: store or recompute.** Store fp32: write+read 8·M·4C B = 2.5 GB at the largest (350 µs),
   268 MB·2 at 32,128,64,64 (75 µs), fits L2 (≤ 42 MB) for the other 14. Recompute GEMM1 in phase 2: +50 % GEMM FLOPs (32 µs
   FP16 at the largest) plus a second dwconv+LN pass (or store the LN'd activations once: M·C·2 B fp16 = 154 MB write+read at
   the largest, ≈ 45 µs). With tensor cores, recompute wins at the top two sizes and is roughly a tie elsewhere.
3. **Depthwise conv implementation.** SIMT register patches (4×4 outputs per lane, 30 LDG.128 + 784 FMA per channel) are
   FMA-bound at 134 µs/pass at the largest but measured 516–549 µs (latency-bound at 2 CTAs/SM); zero-padded smem planes per
   warp for 7×7/14×14 images; a Toeplitz GEMM on tensor cores (16-wide column blocks: K = 7×22 = 154 → ~3.3× FLOP inflation
   at 32× the rate, but the A operand must be gathered into smem and fp16 input rounding costs ≈5e-4 rms); fusing the dwconv
   into the GEMM1 producer (one CTA per token tile covering all N so the stencil is computed once).
4. **GEMM engine and precision.** fp32 SIMT (this anchor: 128×128 tiles reach 46–50 TF/s bare, 57–60 without global loads,
   17–25 TF/s with the real prologue/epilogue/affine), FP16 `tcgen05` (1.82 PF/s; 2SM pairs; the per-token LN affine and the
   per-(image,k) GRN scale must be applied while converting fp32→fp16 into smem, or folded into W2 per image), or
   legacy `mma.sync` fp16 as an intermediate step. Weights must be converted to fp16 inside the call (no cross-call cache):
   W1T/W2T conversion is a 19 MB pass at C=768 (11.5 µs with warp-level tiles, 42 µs with block-level tiles).
5. **Tile shapes and parallelism per size.** M ranges 98…802816, N ∈ {4C, C}, K ∈ {C, 4C}. Small-M/large-K shapes (2,768,7,7:
   M=98, K=3072) need split-K (cross-CTA atomics into an output pre-initialised with x+b2, or warp-level K groups) or the
   GEMM runs on < 50 CTAs; 32,768,7,7 with 128×96 tiles gives 416 tiles = 1.4 waves (30 % quantisation loss). C=96/160 need
   96- or 32-wide N tiles to avoid 20–25 % padding in GEMM2.
6. **Layouts.** K-major activations (`[C][Mp]`, `[4C][Mp]`) make GEMM A tiles straight float4 copies into smem with the LN or
   GRN affine applied per element (constant per thread's token quad); the dwconv writes them with coalesced float4 stores along
   w; output goes back to NCHW with lanes along hw. Channels-last (`[M][C]`) needs transposed smem stores (8-way conflicts).
7. **Cache hints.** Not explored. Outputs with evict_last and weight streams with evict_first are the obvious #38/#85 carry-overs.

## 7. Results so far (rented B200, run_tests, µs; 16/16 pass)

| kernel | 1,192 | 1,96 | 1,160 | 2,96 | 4,192 | 8,384 | 4,96 | 8,192 | 16,384 | 2,768 | 8,768 | 16,768 | 8,320 | 32,768 | 32,128 | 64,96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 157 | 173 | 164 | 192 | 200 | 210 | 226 | 248 | 307 | 226 | 223 | 335 | 596 | 478 | 2635 | 11525 |
| v1 (all-channel dwconv tiles, [M][C]) | 546 | 100 | 548 | 144 | 656 | 899 | 185 | 745 | 1465 | 2091 | 2411 | 2595 | 831 | 2219 | 1697 | 6717 |
| **v2 anchor (this submission)** | 89 | 65 | 111 | 100 | 140 | 273 | 142 | 218 | 353 | 124 | 265 | 585 | 626 | 1217 | 1650 | 5596 |

Per-phase breakdown of v2 (phases as separate launches, harness timing): 32,768,7,7: P1 333 / P2 425 / P3 359 µs;
16,384,14,14: 38 / 194 / 159; 8,320,32,32: 41 / 270 / 359; 32,128,64,64: 127 / 750 / 828; 64,96,112,112: 544 / 2973 / 2709.
GEMM phases run at 17–25 TF/s. Ablations (same shapes): no epilogue → P2 +35–70 % faster (GELU/stores/warp sums cost more than
the short K=96–128 main loops); no A-affine → P3 +35 % (4 sumsq loads + sqrt per element); no prologue → ≈ +5 %;
`__launch_bounds__(256,1)` (238–255 regs, 148 CTAs) is slower everywhere. Phase-1 ablations: block-level transposes 42 µs vs
warp-level 11.5 µs at C=768; per-token conv with 4 tokens per lane 306 µs vs 59 µs with 1; zero-padded smem planes: 35.5 µs
(32,768,7,7), 12.7 (2,768,7,7), 32.4 (16,384,14,14), correct to 1e-5; prefetching patch conv 516 vs 549 µs at the largest
(still 4× the FMA bound). None of these later variants is in the submitted kernel.

## 8. Open questions

- Hidden Tb per workload (plain eager ≈ the reference, or an agentic torch.compile/channels_last baseline several times faster?)
  and which peak SOLAR uses for Tsol on this fp32 problem. First portal result.
- Does the harness's dirty-L2 flush make the hidden-activation round trip cost DRAM traffic even when it fits L2 (42 MB)?
- Best tensor-core formulation of the GRN scale: per-element on A during fp16 conversion vs per-image scaling of W2.
- Can the dwconv reach its FMA bound in SIMT (more warps per SM, smem staging, fewer live registers), or is a Toeplitz tensor-core
  conv needed at the top two sizes?
- Cooperative launch vs 3 launches on the portal (rented: the single cooperative kernel was 7 % slower than three launches at
  the largest size in one comparison; barrier cost ≈ launch cost).
- Portal/rented ratio for this mixed kernel (clock-bound SIMT parts ×1.3, DRAM parts ×1.05).
