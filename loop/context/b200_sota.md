# B200 memory-bound state of the art, and what it means for #38

Public kernels and measurements relevant to row-wise, memory-bound fp32 kernels on B200 (sm_100a), as of
2026-10-07. Source keys are in `sources.md`. Labels as in `core_brief.md`; `[portal]` = our own B200
submissions (`poc/results/b200_portal_workloads.csv`). Excerpts with commentary are in `examples/`.

## 1. What SOTA kernels do

| Kernel (key) | Memory path | Bytes/thread, CTA, grid | Hints | Small sizes | B200 evidence |
|---|---|---|---|---|---|
| FlashInfer CuTe `qk_rmsnorm` [fi-qknorm] (`examples/flashinfer_qk_rmsnorm_sm100_excerpt.py`) | 128-bit `cp.async` → smem → regs; weights by a sync copy issued between commit and wait; plain 128-bit stores | On sm_100/103/107 only: **4 threads per row** for head_dim 128 (16 elsewhere), so 32 elements per thread; 128-thread CTA, 32 rows; one-shot grid | none; optional PDL | nothing special | PR #5305 (bf16, head_dim 128): **1.14× at M=8192, 1.20× at M=32768**, 0.98–1.00× at M ≤ 512; "adding threads per CTA instead was measured to be monotonically worse" [LITERATURE: fi-pr5305] |
| quack RMSNorm fwd [quack] (`examples/quack_rmsnorm_fwd_excerpt.py`) | same cp.async → smem → regs pattern | N=128 fp32: 128 threads, 16 threads/row, 2×16 B per thread, 8 rows per CTA; one-shot | none | none; one config ladder for Hopper and Blackwell | none published; H100: about 3 TB/s (≈ 90%) for N ≥ 4 K. quack uses `torch.clone` (90.4% of H100 peak) as the empirical ceiling [LITERATURE: quack-blog, quack-notes] |
| CAKE `cake_rmsnorm_train` fwd [cake-pr5741] (`examples/cake_rmsnorm_fwd_l2hint_excerpt.cu`) | direct 128-bit LDG into a register cache | per-size program table, chosen by paired CUPTI cold-L2 tests on B200: tiny T = one row per 64-thread CTA; mid T = persistent, software-pipelined, capped at 4–6 CTAs/SM; large T = one-shot (H=512: 2 rows per 64-thread CTA) | **x loads `L2::cache_hint` = EVICT_FIRST constant; weights `ld.global.nc`; stores plain** | separate programs below about 1024 rows | full table in §2. "256-bit single loads … not faster" (H=512, SM100a/103a); an evict-first looped variant "no gain" (GB300); pipelined persistent grids 0.96–1.00× at large sizes on SM100/103 |
| FlashInfer CUDA `QKRMSNorm` [fi-norm-cuh] | 128-bit `vec_t`; x read twice (second read hits L1/L2) | a warp per (token, head), 4 warps per CTA, grid = min(occupancy × SMs, jobs/4), grid-stride | none; PDL `griddepcontrol` | – | – |
| SGLang FLUX.2 QK-norm epilogue [sglang-flux2] (`examples/sglang_flux2_qknorm_excerpt.cuh`) | bf16 vector loads | a warp per (token, head), 8 B per lane, 256-thread CTAs, persistent at occupancy | none | – | – (written by a "Kernel Design Agent") |
| TensorRT-LLM `fusedQKNormRope` [trtllm-qknorm] | bf16 packed vectors | a warp per (token, head) | none | – | – |
| ThunderKittens [tk] | layernorm/rotary: 2 warps, per-warp double-buffered async loads | Hopper-era | – | – | no Blackwell memory-bound kernels (its Blackwell work is GEMM/attention) |
| Cursor's public #38 [cursor-results] (no licence: read only, not excerpted) | float4 LDG | a warp per (token, head), 1024-thread CTAs | none | – | board score 0.551; v1.0-harness times 5.5 µs (12.6 MB) … 128.0 µs (805 MB, 6.29 TB/s) |

Common to all of them: the row stays in registers, the reduction stays inside a warp (or within 4–16 lanes), there is one
pass over DRAM, and **no one uses TMA for short rows**. Clusters/DSMEM appear only for rows ≥ 16–64 K elements
[LITERATURE: quack, fi-pr2777]. The only B200-specific tuning found in public code is **more bytes per thread**
(FlashInfer) and **per-size program tables plus evict_first on streamed loads only** (CAKE).

## 2. Published B200 numbers (CUPTI or cold-L2 timing)

CAKE RMSNorm forward, bf16, B200, CUPTI kernel time with cold L2 (rotating buffers), median of counterbalanced pairs
[LITERATURE: cake-pr5741]. Bytes = read x + write y. Ours = r3 CuTe on the portal, same byte count for #38 [portal].

| Bytes | CAKE shape | CAKE µs | CAKE TB/s | Ours (#38 size) | Ours µs | Ours TB/s |
|---|---|---|---|---|---|---|
| 2.1 MB | H2048 T256 | 2.53 | 0.83 | – | – | – |
| 6.3 MB | H6144 T256 | 3.58 | 1.76 | – | – | – |
| 8.4 MB | H2048 T1024 | 3.49 | 2.40 | 12.6 MB | 4.8 | 2.62 |
| 25.2 MB | H6144 T1024 | **6.37** | 3.95 | 25.2 MB | **6.95** | 3.62 |
| 33.6 MB | H2048 T4096 | 8.06 | 4.16 | 57.6 MB | 11.5 | 5.01 |
| 100.7 MB | H6144 T4096 | 18.98 | 5.30 | 100.7 MB | **17.9** | 5.62 |
| 134 MB | H512 T65536 (512-wide rows) | 23.36 | 5.75 | 201.3 MB | 32.3 | 6.23 |
| 397 MB | H6144 T16172 | 61.70 | 6.44 | 402.7 MB | 61.0 | 6.60 |
| 805 MB | H6144 T32768 | 119.2 | 6.76 | 805.3 MB | **117.8** | 6.84 |
| 1611 MB | H6144 T65536 | 231.9 | 6.94 | – | – | – |

Fits `t = fixed + bytes/BW` [INFERRED by least squares]: ours (≥ 25 MB) **3.5 µs + 7.03 TB/s**; CAKE H=6144
**4.3–5.3 µs + 7.06–7.10 TB/s**. Conclusions:

- **Marginal bandwidth ≈ 7.05–7.1 TB/s (86–87% of 8.18 TB/s) is the public ceiling for a 1:1 read+write stream** under
  cold-L2 CUPTI timing. Our L band already sits there, and we beat CAKE at ≥ 100 MB. [INFERRED]
- The **tiny-kernel floor on B200 is ≈ 2.2–2.5 µs** (2.1 MB in 2.53 µs). Our fixed cost of about 3.1–3.5 µs is about
  1 µs above it. CAKE's mid-size persistent pipelined program beats us by 0.6 µs at 25 MB. [INFERRED]
- Other figures: arXiv 2605.04178 says "sustained HBM is 6.8–7.1 TB/s vs. 8.0" (consistent, but the same paper lists 176
  SMs and a 64 MB L2, so give it low weight). arXiv 2512.02189 v1: STREAM triad (2 reads : 1 write) 7.48 TB/s; v3:
  4.14 TB/s at 4–16 GB, which it attributes to "larger arrays would be needed", an implausible explanation. Treat v3 as a methodology
  artefact. [LITERATURE]
- Noise: CAKE sees 5–9% launch-order dependence for kernels under about 10 µs (B200/GB300), even between bit-identical
  builds. Treat S-band differences under 5% from one submission as noise. [LITERATURE: cake-pr5741]

## 3. Cache hints on B200: evidence

1. **Our own portal data** (µs; S = 12.6/25.2 MB, L = 805 MB) [portal]:

   | Variant | Width | Hints | 12.6 MB | 25.2 MB | 100.7 MB | 402.7 MB | 805 MB |
   |---|---|---|---|---|---|---|---|
   | g2-os-r8w4 (Triton) | 128 | x EF, w EL, stores EF | 4.8 | 6.8 | 18.7 | 63.9 | 123.9 |
   | r2-ldg256-os-r16 (CUDA) | 256 | x EF + L1::no_allocate, w EL, stores EF | 4.9 | 6.8–7.0 | 18.7 | 64.0 | 123.3 |
   | r3-cute-ldg256-os-r16 | 256 | **none** | 4.8 | 6.9–7.0 | **17.9** | **61.0** | **117.8** |

   At the same hints, 256-bit = 128-bit (r2 ≈ g2 within 0.5%). Without hints, M/L are 2–5% faster. S is identical
   in all three. The most likely reading: **the hints, not the width, explain r3's gain** [INFERRED; the CuTe and CUDA
   code generation differ in other small ways].
2. quack (sm100 GEMM, July 2026): "we wired D stores as evict_first … and B loads as evict_last … and measured a net
   REGRESSION (-0.9% plain / -3.5% AG)" [LITERATURE: quack].
3. CAKE's paired B200 search kept evict_first **only on streamed loads**, with plain stores and `.nc` weights. A further evict-first
   variant gained nothing on GB300 (32.29 vs 32.16 µs) [LITERATURE: cake-pr5741].
4. evict_first helps when it protects **reused** data: FlashInfer FP8 MoE GEMM, a once-read weight stream marked
   evict_first, 1.07–1.11× on B200, because activations are re-read 8 times [LITERATURE: fi-pr5692]. #38 reuses only
   48 KB of weights, which stay in L2 anyway.
5. PTX: ptxas ≥ 12.9 accepts `ld.global.nc.L2::evict_first` only on 256-bit accesses (`.v8.b32`/`.v4.b64`; 12.8 rejects it on `ld`). For any width,
   `.L2::cache_hint` with a 64-bit policy works: `createpolicy`, or the CUTLASS constants EVICT_NORMAL
   `0x1000000000000000`, EVICT_FIRST `0x12F0000000000000`, EVICT_LAST `0x14F0000000000000` [LITERATURE: te-3601;
   SPEC: cutlass-cachehint].

Probable mechanism [INFERRED]: `timing.py` zero-fills a 2×L2 buffer on the same stream immediately before every
timed call [PAPER], so the L2 starts full of **dirty** lines. evict_first on stores makes our own dirty outputs the
first victims, which adds DRAM writes (and read/write turnarounds) inside the timed window. The Ascend NPU benchmark
documented the same effect: a zero-fill flush leaves dirty lines whose write-back "is included in the subsequent
kernel" [LITERATURE: ascend-l2]. NVIDIA exposes no way to flush dirty L2 lines [LITERATURE: nv-forum-l2]. Because
S-band times do not move with hints, the dirty L2 is not visibly on the critical path at ≤ 25 MB.

**Update (portal, 7 October): `evict_last` on output stores scored 0.609 vs 0.588 without hints (the problem card (`problems/<name>/card.md`) §7); it supersedes the store rule below.** **Rule for #38: default policy on stores, no evict_last on weights. evict_first on x loads only is optional (CAKE's
choice; expect ±1%).** Never touch memory we do not own (e.g. `discard.global.L2` on harness buffers): that is
manipulating the environment.

## 4. Small-input latency (the fixed cost)

- Ours: one wave of 768 CTAs (5.2 per SM) at 12.6 MB takes 4.8 µs, about 3.0 µs above bytes/7.03 TB/s. The fixed cost of
  about 3.5 µs is present at **every** size: it is 60% of the time at 12.6 MB and 3% at 805 MB. [portal; INFERRED]
- Floor reference: about 2.2–2.5 µs for a tiny streaming kernel on B200 (CAKE), which covers launch, first CTA, one DRAM
  round trip, stores and exit. CUPTI excludes CPU launch cost. The megakernel blog measured about 2.1 µs per launch with streams on
  H100 (1.3 µs with graphs); that cost appears in our window only as gaps between kernels [LITERATURE: hazy-megakernel].
- What SOTA does at small sizes: per-size dispatch to small 64–128-thread CTAs (CAKE) or a persistent pipelined
  program capped at 4 CTAs/SM (CAKE, 25 MB: 6.37 µs vs our 6.95). FlashInfer's bytes-per-thread change is neutral
  at small sizes. Nobody publishes a sub-2 µs B200 streaming kernel.
- Chips and Cheese: "Perhaps Nvidia's scheduler tries to fill one partition's SMs before going to the other". A
  small grid could then load one die more than the other. Unverified; keep S grids ≥ 2 × 148 CTAs. [LITERATURE:
  chipsandcheese-b200]
- PDL cannot help a single-kernel solution. Overlapping the harness's own memset would hide work, so it is forbidden.

## 5. Corrections to our context docs

- `b200_arch.md` §2 and `sources.md`: the arXiv 2512.02189 v1 "58% reduction in memory access latency in cache-misses" is
  **TMEM at 420 cycles vs Hopper's 1000-cycle global memory**, not B200 DRAM latency. v3 drops the claim. It is not
  evidence about DRAM latency.
- arXiv 2507.10789 ("Dissecting the NVIDIA Blackwell Architecture") studies the **RTX 5080 (GB203)**, not B200.
- the problem card (`problems/<name>/card.md`) §7 credits r3's M/L gain to 256-bit width, and `playbook_membound.md` §6 recommends `cache:stream`
  (evict_first stores). On B200 the evidence in §3 points the other way: evict_first stores cost 2–5% at M/L, and width
  alone gains about 0.
- `b200_arch.md` §7: add the 256-bit-only rule for `.L2::evict_first` on loads (§3.5).
- `b200_arch.md` §2 "85–92%" planning value: the measured marginal is 86–87% (7.05–7.1 TB/s). The size-averaged efficiency is
  lower because of the 3–3.5 µs fixed cost.

## 6. Applying this to #38

**What r3 already does right:** one launch; one-shot grid; a half-warp per row, so each instruction covers one
contiguous 512 B row, which is quack's fastest coalescing class (four separate 128 B pieces cost about 8% on H100 loads, and
fragmented stores cost more) [LITERATURE: quack-notes]; x and w loads issued before first use; default cache policy; marginal
bandwidth already at the public ceiling (7.03 vs 7.06–7.10 TB/s). **The remaining lever is mostly the fixed cost:**
1 µs off at every size ≈ +0.02–0.03 in score (the problem card (`problems/<name>/card.md`) §6 sensitivities) [INFERRED].

Ideas ranked by expected score gain:

1. **Attack the ≈ 3.5 µs fixed cost** (+0.005 to +0.02; confidence low). First measure where it goes, on H200 with
   the harness: the same grid with an empty body; reads only; zero-fill vs `sum()` flush; and nsys to confirm that `run()` issues
   exactly one GPU activity (CuTe DSL launch, no extra memcpy). Then try a CAKE-like S/M specialist: a grid of 148 × 4 CTAs,
   each thread holding 2–4 rows' 256-bit loads in flight, with the next row's loads issued before the current row's reduction. Compare
   it with 64/128-thread one-shot CTAs. Evidence: CAKE beats us by 0.6 µs at 25 MB with a persistent, software-pipelined
   program capped at 4 CTAs/SM (6144-wide rows, so not the same layout); the floor is ~1 µs below us. [LITERATURE; INFERRED]
2. **More bytes in flight per thread at M/L** (+0.003 to +0.008; medium-low). Give each half-warp 2 rows, r and
   r+48, which share one weight row and so need one weight load. Alternatively, use 128-thread CTAs with the same 16-row tile.
   Issue every load first. FlashInfer measured +14–20% on B200 at ≥ 128 MB going from 16 to 64 B per thread. We are at 32 B and already at the
   bandwidth ceiling, so expect M −2…−5% and L −0…−2%. Keep each instruction on one contiguous row: do **not** copy FlashInfer's 4-lanes-per-row
   interleaving. [LITERATURE: fi-pr5305; INFERRED]
3. **Cache policy discipline** (protects about 2–5% at M/L, ≈ +0.01, in every descendant; medium). No evict_first stores,
   no evict_last weights. Optional single A/B: r3 plus EF on x loads only, via inline PTX with the policy constant
   (`examples/quack_cutedsl_inline_ptx_excerpt.py`), expected ±1%. [portal; LITERATURE]
4. **Diagnostic slot: g2-os-r8w4 (Triton, 128-bit) with all eviction hints removed.** The prediction is ≈ r3 (0.588). If it
   holds, width does not matter, and Triton/Gluon designs (faster to iterate on) are back in play for #38. No score gain by itself, but it
   decides where effort goes. [INFERRED]
5. **Shorten the M-band tail** (+0.002 to +0.006; low). At 100–200 MB the fixed part is about 3.7 µs. Size one-shot grids as
   whole multiples of 148 × resident CTAs, or use a balanced persistent split with in-register prefetch for M only. CAKE
   found pipelined persistent grids no better than one-shot at large sizes, so do not apply it to L. [LITERATURE; INFERRED]

**Not worth slots:** TMA/bulk rings (ours: +1–1.5 µs fixed), clusters/DSMEM (rows of 128), 256-bit width for its own
sake, evict_last on weights, PDL, persistent loops without in-flight prefetch (v039, SGLang-style), and warp-per-row
float4 (Cursor, 0.551).
