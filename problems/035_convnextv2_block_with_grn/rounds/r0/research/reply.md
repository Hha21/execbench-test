```markdown card
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
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0).
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids
  lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot.'
- 'Precision: FP16 and TF32 operands with fp32 accumulation pass every workload with ~5 sigma margin (simulated with the
  harness generators); BF16 fails. Any reduced-precision candidate must be re-simulated or run_tests 16/16 before a slot; the
  erf GELU must stay exact (no tanh approximation).'
- 'Weights are converted/transposed inside the call (no cross-call cache of fp16 weights or descriptors; pointers shift every call).'
- 'Workspaces come from torch.empty inside run() (no kernel launched); a cooperative launch counts as one kernel. Grid barriers
  via cg::grid_group or a self-resetting counter only.'
hypotheses:
- id: H1
  statement: FP16 tensor cores (kind::f16, fp32 accumulate) for both GEMMs meet the tolerance and turn every workload into a
    memory/latency-bound problem (GEMMs <= 19 us except 65 us at the largest).
  status: open
  evidence:
  - 'probe: fp16/fp16 err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% of elements beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
- id: H2
  statement: The depthwise 7x7 conv is the binding compute term once GEMMs are on tensor cores (FMA-bound 134 us per pass at
    64,96,112,112, 29 us at 32,128,64,64); a design must do it once and near the FMA bound.
  status: open
  evidence:
  - 'probe: SIMT 4x4 register patches measured 516-549 us at the largest (4x the FMA bound), latency-bound at 2 CTAs/SM'
  - 'probe: zero-padded smem planes do 7x7/14x14 images in 13-36 us; warp-level weight transposes 11.5 us vs 42 us block-level'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes (2.5 GB and
    0.54 GB of extra traffic vs 32 and 9 us of FP16 MMA), and ties elsewhere where it fits L2.
  status: open
  evidence:
  - 'arithmetic in card section 6.2; anchor v2 materialises hidden in fp32 (its P2/P3 phases are compute-bound at 17-25 TF/s so the traffic is hidden today)'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card
    (1.5 us per 296-CTA barrier vs 2.0 us per extra launch) and the portal difference is within noise.
  status: open
  evidence:
  - 'probe: empty kernel 1.8 us, 2 launches 3.8, 3 launches 5.8; 296 CTAs with 1/3 atomic barriers 3.6/6.7 us; 1184 CTAs 18 us'
  - 'r0: the cooperative anchor ran 6717 us vs 6263 us for the same three phases as separate launches at the largest size (one comparison)'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7; 1,192,28,28) split-K (cross-CTA atomics into an x+b2-initialised
    output, or warp-level K groups) is required to use the SMs; without it GEMM2 runs on < 50 CTAs.
  status: supported
  evidence:
  - 'r0 v1 -> v2: 2,768,7,7 2091 -> 124 us, 1,192,28,28 546 -> 89 us with channel-chunked dwconv tiles and split-K configs (cost-model picked S=8..16)'
- id: H6
  statement: The fp32 SIMT GEMM core can run at ~50 TF/s (65% of peak) on B200; the anchor loses to 17-25 TF/s through the
    GELU/store/warp-sum epilogue on short-K tiles and the per-element GRN affine (4 sumsq loads + sqrt per element).
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s (unroll 1 best), 57-60 TF/s without global loads, layout variants within 5%'
  - 'probe ablations: no-epilogue P2 34-35 TF/s vs 20-23; no-affine P3 33-35 vs 22-25; launch_bounds(256,1) slower everywhere'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes (416 tiles = 1.4 waves at 32,768,7,7; N=96/160
    padded to 128), so per-size tile tables or stream-K matter as much as the core rate.
  status: open
  evidence:
  - 'r0 cost model (waves x tile work / efficiency) picked 128x96 tiles for C=768 and 32,768,7,7 still took 1217 us vs 478 us reference'
- id: H8
  statement: Applying the GRN scale per image to W2 (or converting A to fp16 with the scale folded in) removes the per-element
    affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'probe: noaff variant of phase 3: 495 vs 678 us at 32,128,64,64, 1816 vs 2684 us at the largest'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager (157 us .. 11.5 ms), so even the fp32 anchor scores > 0.5 on
    the S/M band and the first FP16 tensor-core design moves most of the score.
  status: open
  evidence:
  - 'profiler: 17-18 kernels per call, 40% elementwise/permute traffic, cuDNN depthwise conv 492 us at 32,128,64,64; unknown whether NVIDIA''s baseline uses torch.compile'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images'
    GEMM1 work at B >= 8 and removes two barrier waits at S.
  status: open
  evidence:
  - 'none yet; needs a self-resetting counter workspace (allowed under the constraints)'
```

### Rationale

Research-phase anchor: one cooperative fp32 SIMT launch, three phases (channel-chunked depthwise conv tiles writing K-major activations plus LayerNorm partial sums; GEMM1 + exact-erf GELU + per-image GRN sums; GEMM2 with the GRN affine applied to A, split-K atomics at small sizes, residual add into NCHW). Correct 16/16 on the rented B200; it exists to obtain Tb and Tsol per workload. The card and ledger record that FP16 tensor cores pass the tolerance with 5σ margin and that the depthwise conv becomes the binding compute term, which later rounds should attack first.

```json solution-spec
{"name": "r0-convnext-coop-fp32-anchor", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop",
 "description": "ConvNeXtV2 block with GRN: one cooperative fp32 SIMT launch, three phases (patch depthwise conv + LN partial sums; GEMM1+GELU+GRN sums; GEMM2 with GRN affine on A, split-K atomics at small sizes, residual into NCHW).",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, fp32 SIMT maths.
//  P1: depthwise 7x7 conv (+bias) -> xn [C][Mp] (K-major), per-(token, 32-channel chunk) LN partial sums;
//      zero GRN sums; transpose weights to [K][N]
//  P2: hidden[4C][Mp] = gelu(LN(xn) @ W1^T + b1); sumsq[b][k] += hidden^2 (per image)
//  P3: out = x + (hidden * (1 + gw*nf) + gb) @ W2^T + b2, written NCHW (split-K with atomics at small sizes)
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define NT 256
#define MAXIMG 5
#define GEMM_SMEM_FLOATS (128 * 132)
#define SMEM_BYTES (GEMM_SMEM_FLOATS * 4 + MAXIMG * 128 * 4 + 64)
#define P1_SLOTS 512

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};

__device__ __forceinline__ float gelu_f(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---------------------------------------------------------------- phase 1
__device__ void transpose_mat(const float* __restrict__ src, int R, int Cc, float* __restrict__ dst, float* tile) {
  const int rt = (R + 31) / 32, ct = (Cc + 31) / 32;
  const int tid = threadIdx.x, j = tid & 31, i0 = tid >> 5;
  for (int t = blockIdx.x; t < rt * ct; t += gridDim.x) {
    const int tr = t / ct, tc = t % ct;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, r = tr * 32 + i, c = tc * 32 + j;
      tile[i * 33 + j] = (r < R && c < Cc) ? src[(size_t)r * Cc + c] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, c = tc * 32 + i, r = tr * 32 + j;
      if (r < R && c < Cc) dst[(size_t)c * R + r] = tile[j * 33 + i];
    }
    __syncthreads();
  }
}

// 4 rows x TW cols output patch of one channel; wt = 49 weights (registers)
template <int TW>
__device__ __forceinline__ void conv_patch(const float* __restrict__ xc, const float* wt, int H, int W, int h0, int w0,
                                           float (&acc)[4][TW]) {
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int cc = 0; cc < TW; cc++) acc[r][cc] = 0.f;
#pragma unroll
  for (int i = 0; i < 10; i++) {
    const int hh = h0 - 3 + i;
    const bool rowok = (hh >= 0) && (hh < H);
    float v[TW + 6];
    if (TW == 4) {
      const float* rp = xc + hh * W;
      float4 L = make_float4(0.f, 0.f, 0.f, 0.f), Cq = L, R = L;
      if (rowok) {
        if (w0 >= 4) L = __ldg(reinterpret_cast<const float4*>(rp + w0 - 4));
        Cq = __ldg(reinterpret_cast<const float4*>(rp + w0));
        if (w0 + 4 < W) R = __ldg(reinterpret_cast<const float4*>(rp + w0 + 4));
      }
      v[0] = L.y; v[1] = L.z; v[2] = L.w; v[3] = Cq.x; v[4] = Cq.y; v[5] = Cq.z; v[6] = Cq.w;
      v[7] = R.x; v[8] = R.y; v[9] = R.z;
    } else {
#pragma unroll
      for (int jj = 0; jj < TW + 6; jj++) {
        const int ww = w0 - 3 + jj;
        v[jj] = (rowok && ww >= 0 && ww < W) ? __ldg(xc + hh * W + ww) : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int ki = i - r;
      if (ki >= 0 && ki < 7) {
#pragma unroll
        for (int cc = 0; cc < TW; cc++)
#pragma unroll
          for (int j = 0; j < 7; j++) acc[r][cc] = fmaf(v[cc + j], wt[ki * 7 + j], acc[r][cc]);
      }
    }
  }
}

// MODE 1: patch tiles (TWR = TW in {4,1}); MODE 0: flattened token tiles (TWR = tokens per lane in {1,2,4})
template <int MODE, int TWR>
__device__ void phase1_conv(const Params& p, float* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nch = p.nch;
  float* st_s = smem;                     // [8][P1_SLOTS][2]
  float* wsm = smem + 8 * P1_SLOTS * 2;   // [32][49]
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg;
  const int rpt = lpr * TWR;  // slots per row (mode 1)

  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    int b = 0, cch = 0, hbi = 0, wbi = 0, tok0 = 0;
    if (MODE == 1) {
      const int per = nch * hb * wb;
      b = tile / per; int rem = tile - b * per;
      cch = rem / (hb * wb); rem -= cch * (hb * wb);
      hbi = rem / wb; wbi = rem - hbi * wb;
    } else {
      cch = tile % nch; tok0 = (tile / nch) * (32 * TWR);
    }
    for (int i = tid; i < 8 * P1_SLOTS * 2; i += NT) st_s[i] = 0.f;
    for (int i = tid; i < 32 * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * 32 * 49 + i);
    __syncthreads();

    int rgi = 0, cgi = 0, h0 = 0, w0 = 0; bool lane_ok = true;
    if (MODE == 1) {
      rgi = lane / lpr; cgi = lane - rgi * lpr;
      lane_ok = rgi < rg;
      h0 = hbi * rows_per_tile + rgi * 4;
      w0 = (wbi * 32 + cgi) * TWR;
      lane_ok = lane_ok && (w0 < W);
    }
    int tk[MODE == 0 ? TWR : 1], tb[MODE == 0 ? TWR : 1], th[MODE == 0 ? TWR : 1], tw_[MODE == 0 ? TWR : 1];
    bool tv[MODE == 0 ? TWR : 1];
    if (MODE == 0) {
#pragma unroll
      for (int r = 0; r < TWR; r++) {
        const int t = tok0 + r * 32 + lane;
        tv[r] = t < M;
        const int tt = tv[r] ? t : 0;
        const int bb = tt / HW; const int rem = tt - bb * HW; const int hh = rem / W;
        tk[r] = t; tb[r] = bb; th[r] = hh; tw_[r] = rem - hh * W;
      }
    }

#pragma unroll 1
    for (int cl = 0; cl < 4; cl++) {
      const int clc = warp * 4 + cl;
      const int c = cch * 32 + clc;
      float wt[49];
#pragma unroll
      for (int i = 0; i < 49; i++) wt[i] = wsm[clc * 49 + i];
      const float bias = __ldg(p.dwb + c);
      float* xrow = p.xn + (size_t)c * Mp;
      float* sts = st_s + warp * P1_SLOTS * 2;
      if (MODE == 1) {
        if (lane_ok) {
          float acc[4][TWR];
          conv_patch<TWR>(p.x + ((size_t)b * C + c) * HW, wt, H, W, h0, w0, acc);
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int h = h0 + r;
            if (h < H) {
              const int t = b * HW + h * W + w0;
              const int slot = (rgi * 4 + r) * rpt + cgi * TWR;
              if (TWR == 4) {
                float4 o = make_float4(acc[r][0] + bias, acc[r][1] + bias, acc[r][2] + bias, acc[r][3] + bias);
                *reinterpret_cast<float4*>(xrow + t) = o;
                sts[(slot + 0) * 2] += o.x; sts[(slot + 0) * 2 + 1] += o.x * o.x;
                sts[(slot + 1) * 2] += o.y; sts[(slot + 1) * 2 + 1] += o.y * o.y;
                sts[(slot + 2) * 2] += o.z; sts[(slot + 2) * 2 + 1] += o.z * o.z;
                sts[(slot + 3) * 2] += o.w; sts[(slot + 3) * 2 + 1] += o.w * o.w;
              } else {
#pragma unroll
                for (int cc = 0; cc < TWR; cc++) {
                  const float o = acc[r][cc] + bias;
                  xrow[t + cc] = o;
                  sts[(slot + cc) * 2] += o; sts[(slot + cc) * 2 + 1] += o * o;
                }
              }
            }
          }
        }
      } else {
        float acc[TWR];
#pragma unroll
        for (int r = 0; r < TWR; r++) acc[r] = 0.f;
#pragma unroll
        for (int i = 0; i < 7; i++)
#pragma unroll
          for (int j = 0; j < 7; j++) {
            const float wv = wt[i * 7 + j];
#pragma unroll
            for (int r = 0; r < TWR; r++) {
              const int hh = th[r] - 3 + i, ww = tw_[r] - 3 + j;
              if (tv[r] && hh >= 0 && hh < H && ww >= 0 && ww < W)
                acc[r] = fmaf(__ldg(p.x + ((size_t)tb[r] * C + c) * HW + hh * W + ww), wv, acc[r]);
            }
          }
#pragma unroll
        for (int r = 0; r < TWR; r++) {
          if (tv[r]) {
            const float o = acc[r] + bias;
            xrow[tk[r]] = o;
            const int slot = r * 32 + lane;
            sts[slot * 2] += o; sts[slot * 2 + 1] += o * o;
          }
        }
      }
    }
    __syncthreads();
    const int nslots = (MODE == 1) ? rows_per_tile * rpt : 32 * TWR;
    for (int slot = tid; slot < nslots; slot += NT) {
      int t; bool valid;
      if (MODE == 1) {
        const int row = slot / rpt, col = slot - row * rpt;
        const int h = hbi * rows_per_tile + row, w = wbi * 32 * TWR + col;
        valid = (h < H) && (w < W);
        t = b * HW + h * W + w;
      } else {
        t = tok0 + slot; valid = t < M;
      }
      if (valid) {
        float S1 = 0.f, S2 = 0.f;
#pragma unroll
        for (int ws = 0; ws < 8; ws++) { S1 += st_s[(ws * P1_SLOTS + slot) * 2]; S2 += st_s[(ws * P1_SLOTS + slot) * 2 + 1]; }
        float* pp = p.part + ((size_t)cch * Mp + t) * 2;
        pp[0] = S1; pp[1] = S2;
      }
    }
    __syncthreads();
  }
}

__device__ void phase1(const Params& p, float* smem) {
  const int tid = threadIdx.x;
  for (int i = blockIdx.x * NT + tid; i < p.B * p.C4; i += gridDim.x * NT) p.sumsq[i] = 0.f;
  transpose_mat(p.w1, p.C4, p.C, p.w1t, smem);
  transpose_mat(p.w2, p.C, p.C4, p.w2t, smem);
  if (p.p1_mode == 1) {
    if (p.p1_twr == 4) phase1_conv<1, 4>(p, smem); else phase1_conv<1, 1>(p, smem);
  } else {
    if (p.p1_twr == 4) phase1_conv<0, 4>(p, smem);
    else if (p.p1_twr == 2) phase1_conv<0, 2>(p, smem);
    else phase1_conv<0, 1>(p, smem);
  }
}

// ---------------------------------------------------------------- GEMM phases
// A = act [K][Mp] (K-major), B = Wt [K][N]. 8 warps = WM x WN spatial warps x KS K-groups. Lane grid 8 x 4.
template <int BM, int BN, int WM, int WN, int KS, int BKc, int PHASE>
__device__ void gemm_phase(const Params& p, float* smem, float* sq_s, float* inv_s, float* red_s, int split) {
  constexpr int WS = WM * WN;
  constexpr int GT = WS * 32;
  constexpr int TM = BM / (WM * 8), TN = BN / (WN * 4);
  constexpr int AM4 = BM / 4, BN4 = BN / 4;
  constexpr int AQ = (BKc * AM4) / GT, BQ = (BKc * BN4) / GT;
  constexpr int STAGE = BKc * (BM + BN);
  constexpr int CS_LD = BM + 4;
  constexpr int NSTEP = NT / BM, NPT = BN / NSTEP;
  static_assert(WS * KS == 8, "8 warps");
  static_assert((BKc * AM4) % GT == 0 && (BKc * BN4) % GT == 0, "quad mapping");
  static_assert(GT % AM4 == 0, "token quad constant per thread");
  static_assert(KS * 2 * STAGE <= GEMM_SMEM_FLOATS && KS * BN * CS_LD <= GEMM_SMEM_FLOATS, "smem");
  static_assert(TM == 8 || TM == 4, "TM");
  static_assert(TN == 8 || TN == 6, "TN");

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kg = warp / WS, ws = warp - kg * WS, wm = ws / WN, wn = ws - wm * WN;
  const int lm = lane >> 2, ln = lane & 3;
  const int gt = ws * 32 + lane;
  const int M = p.M, Mp = p.Mp, HW = p.HW;
  const int N = (PHASE == 2) ? p.C4 : p.C;
  const int K = (PHASE == 2) ? p.C : p.C4;
  const float* __restrict__ Ag = (PHASE == 2) ? p.xn : p.hidden;
  const float* __restrict__ Bg = (PHASE == 2) ? p.w1t : p.w2t;
  const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
  const int ntiles = mt * nt * split, nk = K / BKc;
  float* As = smem + kg * 2 * STAGE;
  float* Bs = As + 2 * BKc * BM;
  const int row_m = wm * (BM / WM) + lm * TM;
  const int col_n = wn * (BN / WN) + ln * TN;
  const int a_m4 = (gt % AM4) * 4;
  const int a_kq = gt / AM4;
  constexpr int KSTEP_A = GT / AM4;

  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    int rem = tile;
    const int ni = rem % nt; rem /= nt;
    const int mi = rem % mt; const int s = rem / mt;
    const int m0 = mi * BM, n0 = ni * BN;
    const int c0 = (int)(((long)nk * s) / split), c1 = (int)(((long)nk * (s + 1)) / split);
    const int b_first = m0 / HW;
    const int nimg = min(MAXIMG, (min(m0 + BM, M) - 1) / HW - b_first + 1);

    // per-thread token quad constants
    float tc0[4], tc1[4]; int timg[4]; bool tval[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int t = m0 + a_m4 + e;
      tval[e] = t < M;
      const int tt = tval[e] ? t : 0;
      if (PHASE == 2) {
        float S1 = 0.f, S2 = 0.f;
        for (int ch = 0; ch < p.nch; ch++) {
          const float2 pp = __ldg(reinterpret_cast<const float2*>(p.part + ((size_t)ch * Mp + tt) * 2));
          S1 += pp.x; S2 += pp.y;
        }
        const float mean = S1 / (float)p.C;
        float var = S2 / (float)p.C - mean * mean; var = fmaxf(var, 0.f);
        tc0[e] = mean; tc1[e] = rsqrtf(var + p.ln_eps); timg[e] = 0;
      } else {
        timg[e] = tt / HW - b_first; tc0[e] = 0.f; tc1[e] = 0.f;
      }
    }
    if (PHASE == 3) {
      for (int img = 0; img < nimg; img++) {
        const float* sq = p.sumsq + (size_t)(b_first + img) * K;
        float sacc = 0.f;
        for (int k = tid; k < K; k += NT) sacc += sqrtf(sq[k]);
        sacc = warp_sum(sacc);
        if (lane == 0) red_s[warp] = sacc;
        __syncthreads();
        if (tid == 0) {
          float tot = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < 8; w2++) tot += red_s[w2];
          inv_s[img] = 1.f / (tot / (float)K + p.eps);
        }
        __syncthreads();
      }
#pragma unroll
      for (int e = 0; e < 4; e++) tc1[e] = inv_s[timg[e] < MAXIMG ? timg[e] : 0];
    }

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    float4 ra[AQ], rb[BQ];
    auto load_chunk = [&](int kc) {
      const int k0 = kc * BKc;
#pragma unroll
      for (int q = 0; q < AQ; q++) {
        const int ka = k0 + q * KSTEP_A + a_kq;
        float4 v = __ldg(reinterpret_cast<const float4*>(Ag + (size_t)ka * Mp + m0 + a_m4));
        float vv[4] = {v.x, v.y, v.z, v.w};
        if (PHASE == 2) {
          const float lw = __ldg(p.lnw + ka), lb = __ldg(p.lnb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) vv[e] = tval[e] ? fmaf((vv[e] - tc0[e]) * tc1[e], lw, lb) : 0.f;
        } else {
          const float gwk = __ldg(p.gw + ka), gbk = __ldg(p.gb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float sq = __ldg(p.sumsq + (size_t)(b_first + timg[e]) * K + ka);
            vv[e] = tval[e] ? fmaf(vv[e], fmaf(gwk * tc1[e], sqrtf(sq), 1.f), gbk) : 0.f;
          }
        }
        ra[q] = make_float4(vv[0], vv[1], vv[2], vv[3]);
      }
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        const int col = n0 + n4;
        rb[q] = (col < N) ? __ldg(reinterpret_cast<const float4*>(Bg + (size_t)(k0 + kb) * N + col))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    };
    auto store_regs = [&](int buf) {
#pragma unroll
      for (int q = 0; q < AQ; q++)
        *reinterpret_cast<float4*>(As + (buf * BKc + q * KSTEP_A + a_kq) * BM + a_m4) = ra[q];
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        *reinterpret_cast<float4*>(Bs + (buf * BKc + kb) * BN + n4) = rb[q];
      }
    };
    auto gsync = [&]() {
      if (KS == 1) __syncthreads();
      else if (GT == 32) __syncwarp();
      else asm volatile("bar.sync %0, %1;" ::"r"(kg + 1), "r"(GT));
    };

    int kc = c0 + kg;
    if (kc < c1) { load_chunk(kc); store_regs(0); }
    gsync();
    for (int it = 0; kc < c1; it++, kc += KS) {
      const int buf = it & 1;
      const bool more = (kc + KS) < c1;
      if (more) load_chunk(kc + KS);
      const float* Ab = As + buf * BKc * BM + row_m;
      const float* Bb = Bs + buf * BKc * BN + col_n;
#pragma unroll 4
      for (int k = 0; k < BKc; k++) {
        float a[TM], b[TN];
        {
          const float4 a0 = *reinterpret_cast<const float4*>(Ab + k * BM);
          a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
          if (TM == 8) {
            const float4 a1 = *reinterpret_cast<const float4*>(Ab + k * BM + 4);
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
          }
        }
        if (TN == 8) {
          const float4 b0 = *reinterpret_cast<const float4*>(Bb + k * BN);
          const float4 b1 = *reinterpret_cast<const float4*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
        } else {
          const float2 b0 = *reinterpret_cast<const float2*>(Bb + k * BN);
          const float2 b1 = *reinterpret_cast<const float2*>(Bb + k * BN + 2);
          const float2 b2 = *reinterpret_cast<const float2*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b1.x; b[3] = b1.y; b[4] = b2.x; b[5] = b2.y;
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
      if (more) store_regs(buf ^ 1);
      gsync();
    }

    // ---- epilogue: partial tiles -> smem [KS][BN][CS_LD], then one token per thread
    __syncthreads();
    float* Cs = smem;
#pragma unroll
    for (int j = 0; j < TN; j++) {
      float* dst = Cs + (kg * BN + col_n + j) * CS_LD + row_m;
      *reinterpret_cast<float4*>(dst) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
      if (TM == 8) *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
    }
    __syncthreads();
    const int m_l = tid % BM, n_l0 = tid / BM;
    const int t = m0 + m_l;
    const bool tv = t < M;
    const int img = tv ? t / HW : b_first;
    const int hw = t - img * HW;
    if (PHASE == 2) {
      const int img0 = __shfl_sync(0xffffffffu, img, 0);
      const bool uniform = __all_sync(0xffffffffu, (img == img0) && tv);
      const int il0 = img0 - b_first;
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        float v = 0.f;
#pragma unroll
        for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
        const float gl = gelu_f(v + __ldg(p.b1 + col));
        if (tv) p.hidden[(size_t)col * Mp + t] = gl;
        float g2 = tv ? gl * gl : 0.f;
        if (uniform) {
          g2 = warp_sum(g2);
          if (lane == 0) {
            if (il0 < MAXIMG) atomicAdd(&sq_s[il0 * BN + n_l], g2);
            else atomicAdd(p.sumsq + (size_t)img0 * N + col, g2);
          }
        } else if (tv) {
          const int il = img - b_first;
          if (il < MAXIMG) atomicAdd(&sq_s[il * BN + n_l], g2);
          else atomicAdd(p.sumsq + (size_t)img * N + col, g2);
        }
      }
      __syncthreads();
      for (int idx = tid; idx < nimg * BN; idx += NT) {
        const int im = idx / BN, n = idx - im * BN;
        const float v = sq_s[idx]; sq_s[idx] = 0.f;
        atomicAdd(p.sumsq + (size_t)(b_first + im) * N + n0 + n, v);
      }
      __syncthreads();
    } else {
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        if (col < N && tv) {
          float v = 0.f;
#pragma unroll
          for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
          const size_t idx = ((size_t)img * N + col) * HW + hw;
          if (split == 1) p.out[idx] = v + __ldg(p.b2 + col) + __ldg(p.x + idx);
          else atomicAdd(p.out + idx, v);
        }
      }
      __syncthreads();
    }
  }
}

template <int PHASE>
__device__ __forceinline__ void run_gemm(const Params& p, int cfg, int split, float* smem, float* sq_s, float* inv_s, float* red_s) {
  switch (cfg) {
    case 0: gemm_phase<128, 128, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 1: gemm_phase<128, 96, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 2: gemm_phase<64, 64, 2, 2, 2, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    default: gemm_phase<32, 32, 1, 1, 8, 16, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
  }
}

__global__ void __launch_bounds__(NT, 2) convnext_fused(Params p) {
  extern __shared__ float4 smem4[];
  float* smem = reinterpret_cast<float*>(smem4);
  float* sq_s = smem + GEMM_SMEM_FLOATS;
  float* inv_s = sq_s + MAXIMG * 128;
  float* red_s = inv_s + 8;
  for (int i = threadIdx.x; i < MAXIMG * 128; i += NT) sq_s[i] = 0.f;
  __syncthreads();
  cg::grid_group grid = cg::this_grid();

  phase1(p, smem);
  grid.sync();
  if (p.split3 > 1) {  // initialise the output with residual + bias for the split-K atomics
    const long total = (long)p.M * p.C;
    for (long i = (long)blockIdx.x * NT + threadIdx.x; i < total; i += (long)gridDim.x * NT) {
      const int c = (int)((i / p.HW) % p.C);
      p.out[i] = p.x[i] + p.b2[c];
    }
  }
  run_gemm<2>(p, p.cfg2, 1, smem, sq_s, inv_s, red_s);
  grid.sync();
  run_gemm<3>(p, p.cfg3, p.split3, smem, sq_s, inv_s, red_s);
}

// ---------------------------------------------------------------- host
struct Pick { int cfg; int split; };
static Pick pick_cfg(long M, long N, long K, int grid, bool allow_split) {
  struct CfgD { int id, BM, BN, BK; double eff; };
  const CfgD cfgs[4] = {{0, 128, 128, 32, 0.80}, {1, 128, 96, 32, 0.78}, {2, 64, 64, 32, 0.60}, {3, 32, 32, 16, 0.40}};
  double best = 1e300; Pick r{0, 1};
  for (int ci = 0; ci < 4; ci++) {
    const CfgD& c = cfgs[ci];
    if (c.id == 1 && (N % 96) != 0) continue;
    const long tiles = ((M + c.BM - 1) / c.BM) * ((N + c.BN - 1) / c.BN);
    const long nk = K / c.BK;
    for (int S = 1; S <= (allow_split ? 16 : 1); S *= 2) {
      if (S > 1 && nk / S < 2) break;
      const long total = tiles * S;
      const long waves = (total + grid - 1) / grid;
      double cost = (double)c.BM * c.BN * ((double)K / S) / c.eff * (double)waves;
      if (S > 1) cost *= 1.08 + 0.03 * S;
      if (cost < best) { best = cost; r.cfg = c.id; r.split = S; }
    }
  }
  return r;
}

extern "C" const char* convnext_launch(Params p, void* stream_v) {
  static int grid_size = 0;
  cudaError_t e;
  if (grid_size == 0) {
    e = cudaFuncSetAttribute(convnext_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int bps = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, convnext_fused, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    if (bps < 1) bps = 1;
    grid_size = sms * bps;
  }
  const Pick k2 = pick_cfg(p.M, p.C4, p.C, grid_size, false);
  const Pick k3 = pick_cfg(p.M, p.C, p.C4, grid_size, true);
  p.cfg2 = k2.cfg; p.cfg3 = k3.cfg; p.split3 = k3.split;
  // phase-1 geometry
  {
    const int tw = (p.W % 4 == 0) ? 4 : 1;
    const int cgt = p.W / tw;
    int lpr, rg, wb;
    if (cgt >= 32) { lpr = 32; rg = 1; wb = (cgt + 31) / 32; }
    else { lpr = cgt; rg = 32 / cgt; wb = 1; }
    if (rg > 8) rg = 8;
    const int rows = 4 * rg;
    if (rows <= p.H) {
      p.p1_mode = 1; p.p1_twr = tw; p.p1_lpr = lpr; p.p1_rg = rg; p.p1_wb = wb;
      p.p1_hb = (p.H + rows - 1) / rows;
      p.p1_tiles = p.B * p.nch * p.p1_hb * p.p1_wb;
    } else {
      p.p1_mode = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_wb = 0; p.p1_hb = 0;
      int r0 = 4;
      while (r0 > 1 && ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch < grid_size) r0 >>= 1;
      p.p1_twr = r0;
      p.p1_tiles = ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch;
    }
  }
  void* args[] = {&p};
  e = cudaLaunchCooperativeKernel((const void*)convnext_fused, dim3(grid_size), dim3(NT), args, (size_t)SMEM_BYTES,
                                  (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <string>

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};
extern "C" const char* convnext_launch(Params p, void* stream);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  TORCH_CHECK(C % 32 == 0, "dim must be a multiple of 32");
  TORCH_CHECK(H * W >= 32, "H*W must be >= 32");
  const int HW = H * W, M = B * HW, C4 = 4 * C, nch = C / 32;
  const int Mp = ((M + 127) / 128) * 128;
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");

  // workspace (no state kept across calls)
  const size_t off_xn = 0;
  const size_t off_part = align256(off_xn + (size_t)C * Mp * 4);
  const size_t off_hidden = align256(off_part + (size_t)nch * Mp * 2 * 4);
  const size_t off_sumsq = align256(off_hidden + (size_t)C4 * Mp * 4);
  const size_t off_w1t = align256(off_sumsq + (size_t)B * C4 * 4);
  const size_t off_w2t = align256(off_w1t + (size_t)C * C4 * 4);
  const size_t total = align256(off_w2t + (size_t)C4 * C * 4);
  torch::Tensor ws = torch::empty({(long)total}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();

  Params p;
  p.x = x.data_ptr<float>(); p.dww = dwconv_weight.data_ptr<float>(); p.dwb = dwconv_bias.data_ptr<float>();
  p.lnw = layernorm_weight.data_ptr<float>(); p.lnb = layernorm_bias.data_ptr<float>();
  p.w1 = pwconv1_weight.data_ptr<float>(); p.b1 = pwconv1_bias.data_ptr<float>();
  p.gw = grn_weight.data_ptr<float>(); p.gb = grn_bias.data_ptr<float>();
  p.w2 = pwconv2_weight.data_ptr<float>(); p.b2 = pwconv2_bias.data_ptr<float>();
  p.out = output.data_ptr<float>();
  p.xn = (float*)(wp + off_xn); p.part = (float*)(wp + off_part); p.hidden = (float*)(wp + off_hidden);
  p.sumsq = (float*)(wp + off_sumsq); p.w1t = (float*)(wp + off_w1t); p.w2t = (float*)(wp + off_w2t);
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = HW; p.M = M; p.Mp = Mp; p.C4 = C4; p.nch = nch;
  p.eps = (float)eps; p.ln_eps = (float)layer_norm_eps;
  p.p1_mode = 0; p.p1_twr = 1; p.p1_tiles = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_hb = 0; p.p1_wb = 0;
  p.cfg2 = 0; p.cfg3 = 0; p.split3 = 1;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = convnext_launch(p, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (fused, fp32)"); }
```

```yaml design-card
id: r0-convnext-coop-fp32-anchor
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: persistent
  launch: fused
  tile: rows128
  red: cta
  cache: default
  spec: all
hypothesis: >-
  Research anchor: one cooperative fp32 SIMT launch (296 persistent CTAs, two grid.sync()s) with three phases: channel-chunked
  depthwise 7x7 patch tiles writing K-major activations plus LayerNorm partial sums; 128x128/128x96/64x64/32x32 SIMT GEMM1
  with exact-erf GELU and per-image GRN sums; GEMM2 with the GRN affine applied to A, cross-CTA split-K atomics at small
  sizes and the residual added into NCHW. Correct on all 16 workloads; it exists to obtain Tb and Tsol.
expected_effect:
  S: {pct: -45, confidence: medium}
  M: {pct: -20, confidence: medium}
  L: {pct: -10, confidence: low}
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 70208
  threads_per_cta: 256
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs: {BM: 128, BN: 128, BK: 32, GEMM_CFGS: 4, SPLITK_MAX: 16, P1_PATCH: 4x4, NT: 256, GRID: 296}
tests: [H4, H5, H6]
findings:
  - "precision: with the harness generators (weights /sqrt(fan_in), LN weight ones) fp16/fp16 and tf32/tf32 GEMMs give err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol over 6 workloads x 2 seeds; bf16 fails 28-50% [probe_b200]"
  - "fp32 SIMT peak: FFMA 67.7 TF/s and packed fma.f32x2 74.0 TF/s at 1965 MHz = 128 FMA/clk/SM either way; 56.8 TF/s at the 1500 MHz lock [probe_b200]"
  - "launch floor: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic grid barrier ~1.5 us, 1184 CTAs ~5 us per barrier [probe_b200]"
  - "reference profile: 17-18 kernels per call; cuDNN depthwise conv 492 us, two simt sgemm 400+346 us, six GRN elementwise kernels ~190 us each at 32,128,64,64 [probe_b200]"
  - "cuBLAS fp32 sgemm both GEMMs (warm): 36-52 us at S sizes (8-28 TF/s), 2758 us at the largest (41-45 TF/s) [probe_b200]"
  - "v1 (all-channel dwconv tiles, [M][C] layout, transposed smem stores): 16/16 pass but 546 us at 1,192,28,28 and 2091 us at 2,768,7,7 because phase 1 ran on 1-28 CTAs [run_tests]"
  - "v2 (this kernel): 16/16 pass, 65-140 us at S, 124-353 us at M, 265-5596 us at L; still slower than the reference at 8,384,14,14, 16,384,14,14, 8,768..32,768,7,7 and 8,320,32,32 [run_tests]"
  - "v2 phase breakdown (separate launches): 32,768,7,7 P1 333 / P2 425 / P3 359 us; 64,96,112,112 544 / 2973 / 2709 us; GEMM phases 17-25 TF/s [probe_b200]"
  - "bare 128x128x32 SIMT core on the same shapes: 46-50 TF/s (unroll 1 best), 57-60 TF/s without global loads; lane layouts within 5% [probe_b200]"
  - "ablations: removing the GELU/store/warp-sum epilogue lifts P2 to 32-35 TF/s; removing the per-element GRN affine lifts P3 to 33-35 TF/s; prologue ~5%; launch_bounds(256,1) (238-255 regs) is slower everywhere [probe_b200]"
  - "phase 1: block-level weight transposes 42 us at C=768 vs 11.5 us warp-level; per-token conv 4 tokens/lane 306 us vs 1 token/lane 59 us at 32,768,7,7; zero-padded smem planes 35.5/12.7/32.4 us at 32,768,7,7 / 2,768,7,7 / 16,384,14,14 (correct to 1e-5); prefetching 4x4 patch conv 516 vs 549 us at the largest (FMA bound 134 us at 1500 MHz) [probe_b200]"
  - "dead end: cooperative single kernel measured 6717 us vs 6263 us for the same three phases as separate launches at the largest size (one comparison) [run_tests, probe_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1}
runs_on:
  H200: {runs: true, representative: true, note: "cooperative launch and cg grid sync work on sm_90; fp32 SIMT rate differs"}
  A100: {runs: true, representative: false, note: "needs cudaLaunchCooperativeKernel and 70 KB dynamic smem (ok on sm_80); ranking signal only"}
  L40S: {runs: true, representative: false, note: "100 KB smem per SM allows only 1 CTA/SM; correctness smoke test"}
risks:
  - "fp32 SIMT GEMMs are 10-30x off the FP16 tensor-core design that the tolerance permits; L-band times are 2-12x the stream floor"
  - "cooperative launch fails if the device cannot co-schedule sms x 2 CTAs at 70 KB smem each; grid is sized from the occupancy API at first call"
  - "split-K phase 3 relies on fp32 atomics into an x+b2-initialised output (non-deterministic summation order, within tolerance)"
  - "workspace up to 1.6 GB (hidden fp32 [4C][Mp]) allocated with torch.empty each call via the caching allocator"
measure_first: [H200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```