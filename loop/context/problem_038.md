# Problem card: L1 #38 `038_flux_multi_head_rmsnorm_qk`

Leaderboard data is from the public API (evaluation stack v1.1), fetched 2026-10-03. Definition and workloads are from
`$SOLX/SOL-ExecBench/data/benchmark/L1/038_flux_multi_head_rmsnorm_qk/` on CSF3.

## 1. Semantics

Reference (verbatim, `definition.json`), from FLUX.2-dev:

```python
@torch.no_grad()
def run(query, key, weight_q, weight_k, eps):
    input_dtype = query.dtype
    q_variance = query.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    query_norm = query * torch.rsqrt(q_variance + eps)
    k_variance = key.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    key_norm = key * torch.rsqrt(k_variance + eps)
    query_norm = query_norm * weight_q.unsqueeze(0).unsqueeze(0)
    key_norm = key_norm * weight_k.unsqueeze(0).unsqueeze(0)
    return query_norm.to(input_dtype), key_norm.to(input_dtype)
```

- DPS signature: `run(query, key, weight_q, weight_k, eps, query_norm, key_norm)`.
- `query`, `key`, `query_norm`, `key_norm`: `[B, S, 48, 128]` fp32, contiguous. `weight_q`, `weight_k`: `[48, 128]`
  fp32. `eps`: Python float, 1e-6 in every workload.
- View each tensor as `[R, 128]` with `R = B·S·48` rows. Row r uses weight row `h = r % 48`, and
  `y[r,:] = x[r,:] · rsqrt(mean(x[r,:]²) + eps) · w[h,:]`. Rows are independent and every byte is touched once.
  Q and K are two independent streams of the same shape.
- `R` is always a multiple of 48, so tiles of 1, 2, 4, 8 or 16 rows (any divisor of 48) need no row mask and keep a
  fixed head pattern. 32 or 64 rows need masking when `B·S` is odd (the 1×131 workload). A tile of 48 rows is exactly
  one token, and its weight tile is the whole `[48,128]` matrix.
- Traffic per call: read Q and K, write two outputs = `B·S·48·128·4·4 = B·S·98,304` bytes. The weights add 48 KB,
  fetched once from DRAM and then served from L2.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|` for at least 99% of elements, with no NaN/Inf and not all zeros
  [PAPER: correctness.py, workload.jsonl]. The fp32 ulp is 2^-23 ≈ 1.19e-7 relative, so the budget is **about 84 ulp
  relative**, or 1e-5 absolute for tiny outputs.
- Inputs are `randn`; the weights are also `randn` (the names `weight_q`/`weight_k` miss the "norm weight → ones"
  heuristic), so |y| reaches about 20 and some weights are close to 0 [PAPER: io.py].
- Error sources, all far inside the budget [INFERRED]: sum of 128 squares in any order (tree reduction ≤ about 7 ulp
  worst case, typically under 1); `rsqrt.approx.f32`, the PTX that Triton's `tl.math.rsqrt` and CUDA's `rsqrtf`
  produce, has maximum relative error 2^-22.9 [SPEC: PTX ISA]; Triton's fp32 `/` lowers to `div.full.f32`
  (approximate, ≤ 2 ulp) [VERIFIED-CSF3], so write `* (1.0 / 128)`, which is exact; `(x·r)·w` versus `x·(r·w)` differs
  by ≤ 1 ulp. Total ≤ about 10 ulp.
- What would fail: bf16/fp16/TF32 storage or compute (bf16 rounding alone is 2^-9 relative, about 16,000 fp32 ulp). It is also forbidden as a precision
  downgrade. Leaving rows unwritten (outputs arrive as zeros) fails too.
- Keep everything fp32. No compensated summation is needed.

## 3. The 16 workloads (sorted by size)

`floor8` = bytes / 8 TB/s. `T_sol model` assumes SOLAR's per-workload time is proportional to bytes, scaled so the
geometric mean is 9.195 µs [INFERRED]. A100/L40S columns are the best of 72 PoC variants per workload (PoC, CSF3,
`poc/results/rehearsal_sim/table.csv`); `eff` = bytes / 2.039 TB/s / time.

| wl# | uuid | B | S | B·S | rows per tensor | MB | floor8 µs | T_sol model µs | Q+K rows per SM | A100 best µs (eff) | L40S best µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | dcdec775 | 1 | 128 | 128 | 6,144 | 12.6 | 1.57 | 0.91 | 83 | 9.8 (0.63) | 18.4 |
| 12 | e4d4aebe | 1 | 131 | 131 | 6,288 | 12.9 | 1.61 | 0.93 | 85 | 9.9 (0.64) | 21.4 |
| 0 | c47946fe | 2 | 128 | 256 | 12,288 | 25.2 | 3.15 | 1.81 | 166 | 17.6 (0.70) | 36.6 |
| 15 | 3565f3a6 | 1 | 256 | 256 | 12,288 | 25.2 | 3.15 | 1.81 | 166 | 16.9 (0.73) | 37.1 |
| 5 | 388488d1 | 2 | 293 | 586 | 28,128 | 57.6 | 7.20 | 4.14 | 380 | 37.2 (0.76) | 88.7 |
| 4 | fbd78167 | 8 | 128 | 1024 | 49,152 | 100.7 | 12.58 | 7.24 | 664 | 61.0 (0.81) | 156.8 |
| 7 | 3d08c934 | 1 | 1024 | 1024 | 49,152 | 100.7 | 12.58 | 7.24 | 664 | 61.6 (0.80) | 161.4 |
| 8 | 13ac723d | 4 | 256 | 1024 | 49,152 | 100.7 | 12.58 | 7.24 | 664 | 61.8 (0.80) | 158.8 |
| 11 | 93b2fe42 | 4 | 512 | 2048 | 98,304 | 201.3 | 25.17 | 14.48 | 1,328 | 121.4 (0.81) | 314.1 |
| 13 | 277f8cdf | 8 | 256 | 2048 | 98,304 | 201.3 | 25.17 | 14.48 | 1,328 | 121.9 (0.81) | 315.8 |
| 2 | 0648abad | 4 | 1024 | 4096 | 196,608 | 402.7 | 50.33 | 28.96 | 2,657 | 248.3 (0.80) | 622.0 |
| 6 | d6f8d507 | 16 | 256 | 4096 | 196,608 | 402.7 | 50.33 | 28.96 | 2,657 | 247.9 (0.80) | 620.8 |
| 10 | ce51aff5 | 32 | 128 | 4096 | 196,608 | 402.7 | 50.33 | 28.96 | 2,657 | 247.7 (0.80) | 622.3 |
| 3 | 0f5618d6 | 8 | 773 | 6184 | 296,832 | 607.9 | 75.99 | 43.72 | 4,011 | 366.7 (0.81) | 937.2 |
| 1 | 62bb15e8 | 4 | 1657 | 6628 | 318,144 | 651.6 | 81.44 | 46.86 | 4,299 | 391.0 (0.82) | 997.3 |
| 14 | 473ba98e | 1 | 8192 | 8192 | 393,216 | 805.3 | 100.66 | 57.92 | 5,314 | 480.0 (0.82) | 1,227.1 |

Floors over the 16 workloads: geometric mean **15.98 µs**, arithmetic mean **32.1 µs**, median 25.2 µs [INFERRED by
arithmetic]. Only shapes matter, so dispatch keys on `B·S`, the number of tokens (the 3 workloads of 1024 tokens
behave the same).

Size bands for specialists and dispatch (our convention; the "Character" column is INFERRED from the §6 model):

| Band | Workloads | MB | Character |
|---|---|---|---|
| S | 9, 12, 0, 15, 5 | 12.6–57.6 | latency-bound: fixed costs are about 30–50% of t |
| M | 4, 7, 8, 11, 13 | 101–201 | mixed |
| L | 2, 6, 10, 3, 1, 14 | 403–805 | bandwidth-bound: sustained DRAM bandwidth decides |

## 4. Leaderboard anchors (B200, v1.1, 2026-10-03)

| Entry | Aggregate latency | Score |
|---|---|---|
| SOL bound (SOLAR) | 9.195 µs | 1.0 |
| #1 Infinigence AI (2026-10-02) | 22.158 µs | 0.6275 |
| #2 Databricks | 22.909 µs | 0.6160 |
| #3 RIAC as well | 22.960 µs | 0.6151 |
| #9 doubleAI | 23.938 µs | 0.5983 |
| #20 ac4k | 25.548 µs | 0.5733 |
| #21 Cursor | 26.947 µs | 0.5531 |
| Scoring Baseline (hidden, PyTorch-only agentic) | 31.762 µs | 0.5 |
| Reference implementation | 239.569 µs | 0.0876 |

Places 1–20 lie within 15% of each other. Every top-20 entry has `fast_1_count` = 16/16 (read as: all workloads
faster than the baseline); #21 has 12/16. Other teams' code
is not downloadable. The field is active: entries are dated as recently as 2026-10-02.

## 5. Floors, SOLAR and the aggregate (updated with portal data, 6 October)

- The read+write floor at 8 TB/s is 16.0 µs (geometric mean), so the leader runs at **72% of the floor** (70.5% of the
  15.6 µs floor at the locked-clock peak of 8.18 TB/s). [INFERRED]
- The aggregate latency is the geometric mean of workload latencies and the score the arithmetic mean of workload
  scores [VERIFIED: portal pages; see harness_scoring.md §5].
- Per-workload Tb and Tsol, recovered from our five submission pages [VERIFIED for Tb, derived for Tsol]:

| Workloads (B·S) | Tb µs | Tsol µs | Tsol / 8 TB/s floor | Score if exactly at 8 TB/s |
|---|---|---|---|---|
| 128, 131 | 9.3–9.4 | 1.24–1.29 | 0.77–0.82 | 0.96 |
| 256 (×2) | 10.4–10.5 | 1.94–1.99 | 0.62 | 0.88 |
| 586 | 17.5 | 4.24 | 0.59 | 0.82 |
| 1024 (×3) | 22.4–22.5 | 6.8–7.0 | 0.55 | 0.73 |
| 2048 (×2) | 37.4–37.6 | 13.4–13.5 | 0.53 | 0.67 |
| 4096 (×3) | 66.5–66.7 | 26.7–26.8 | 0.53 | 0.63 |
| 6184, 6628 | 102.7, 107.2 | 40.0, 42.8 | 0.53 | 0.63 |
| 8192 | 123.4 | 53.0 | 0.53 | 0.60 |

- SOLAR is about 0.53× the read+write floor on medium and large inputs (it behaves as if bandwidth were about
  15 TB/s or half the bytes moved) and closer to the floor on the smallest. **Score 1.0 is unreachable.** Exactly
  8 TB/s everywhere scores **0.736**; 8 TB/s plus 1.5 µs fixed per call scores about **0.68**. Latency depends only on
  B·S (the three B·S = 4096 shapes time identically), so 16 workloads are really 10 sizes.
- Score headroom per workload is far larger on small inputs (ceiling 0.82–0.96) than large ones (0.60–0.64).

## 6. Score targets

Model [INFERRED]: Tsol_i ∝ bytes (geometric mean 9.195 µs); Tb_i = a + bytes/BW_b fitted to a geometric mean of
31.762 µs for a ∈ {3, 4, 5} µs; the leader is fitted as c + bytes/BW_l to (22.158 µs, 0.6275). The leader then comes
out as about 0–1.2 µs fixed plus 5.8–6.5 TB/s. Results barely depend on a:

| Target score | Uniform speed-up over the leader | Geomean latency | Simple aggregate formula |
|---|---|---|---|
| 0.6275 (tie #1) | 1.00× | 22.2 µs | 22.6 µs |
| **0.65** | **5.5% faster** | **about 20.95 µs** | 21.35 µs |
| **0.70** | **16% faster** | **about 18.5 µs** | 18.87 µs |
| 0.75 | ≈ the 8 TB/s floor everywhere | about 16 µs | 16.7 µs |

Reference point: a kernel at "1.5 µs + bytes / 7.2 TB/s" scores about 0.655 (geomean 20.96 µs).

Sensitivity per workload, at leader-like times (same model):

| Band | Leader-like t (µs) | Δ mean score for −1 µs on one workload | Δ mean score for −10% on one workload |
|---|---|---|---|
| S (12.6–57.6 MB) | 2.7–10 | +0.0065 … +0.0025 | +0.0017 … +0.0024 |
| M (101–201 MB) | 17–33 | +0.0015 … +0.0008 | +0.0026 … +0.0027 |
| L (403–805 MB) | 66–131 | +0.0004 … +0.0002 | +0.0027 |

Read this as: **relative** speed-ups are worth about the same in every band, and 1 µs off a small workload is worth
about 20× a µs off a large one. Shaving fixed costs and raising sustained bandwidth are both needed.

## 7. What the proof of concept learned (PoC, CSF3)

The PoC kernel (`poc/kernels/rmsnorm_qk.py`) is one Triton design with knobs ROWS, NUM_WARPS, NUM_STAGES, FUSE (one
launch for Q+K), PERSIST, PROGS_PER_SM and EVICT. 72 variants were run through NVIDIA's harness on A100 and L40S; all
passed, with 0.15% run-to-run noise.

- Best A100 variant v028 (ROWS=32, 2 warps, persistent 8 per SM, fused): geomean 82.8 µs, **76% of the A100 floor**.
  It reaches about 80–82% of peak bandwidth on large workloads and **about 60%** on the smallest. Best L40S variant
  v050 (ROWS=8, 1 warp, unfused, one-shot): 73% of the L40S floor.
- Picking the best variant per workload gains only 1.4% (A100) and 2.3% (L40S). **The gap is in the design, not the
  knobs.**
- A100 and L40S agree on the broad ranking (Spearman 0.77) but not on the winner. The emulator helped a lot when the
  cheap source resembled the target (HBM) and little on L40S (GDDR6).
- **First B200 result (6 October, portal #61435):** v028 scored **0.451 at 35.9 µs**, faster than the scoring
  baseline on only 5 of 16 workloads, and no faster than on H200 (36.4 µs). Likely cause: on sm_100a v028 needs 149
  registers, so only 6 of its 8 persistent programs per SM fit, giving a second wave (see playbook, anti-patterns).
  Lesson: H200 timings cannot see sm_100a register growth; check the sm_100a compile before trusting them.
  [portal; VERIFIED-CSF3 for the register counts; INFERRED for the cause]
- **Five B200 submissions with per-workload results (6 October, `poc/results/b200_portal_workloads.csv`):**
  - v039 (fused, persistent, ROWS=16, 8 warps, 8 programs per SM, 32 registers) is fastest on **every** workload,
    so dispatching between the five gains nothing. Score 0.531, geomean 28.56 µs, faster than Tb on 9/16.
  - v039 bandwidth by band: small 2.67 TB/s, medium 4.90, large 5.77. Baseline: small 1.81, medium 4.52,
    **large 6.11**. We beat Tb on small/medium (1.3–1.7× on the smallest) and lose by 3–11% on large.
  - B200 time / H200 time, geomean by band (bandwidth alone predicts 0.60): v039 small 1.02, medium 0.74, large 0.65;
    v040 (one program per 2 rows) 1.21 / 1.13 / 1.11, i.e. **slower on B200 at every size**: hundreds of thousands
    of tiny programs cannot keep B200's memory busy. **Small inputs get no faster from H200 to B200**; fixed cost
    dominates. Per-workload ranking agreement with B200 over 4 variants: H200 0.60, A100 −0.10.
  - What-if scores from v039 using the recovered Tb/Tsol: medium+large at 7.0 TB/s (+2 µs fixed) → 0.592;
    small at 6.4 TB/s + 2 µs fixed → 0.555; both → 0.616; both with 1 µs fixed on small → **0.633 (above the
    leader's 0.628)**. Raising sustained bandwidth on medium/large is the larger lever; small-input fixed cost
    is needed too.
- **H200 (added 4 October):** best is again v028, geomean 36.4 µs, 73% of the 4.8 TB/s floor. H200 and A100 rank the
  72 variants almost identically (Spearman 0.93; H200's top 5 are all in A100's top 6); L40S is the outlier (0.77–0.85).
  v028 reaches 83–84% of peak on large workloads but only **44–45% on the two smallest**, against about 60% on A100:
  the faster the memory, the larger the share of fixed cost on small inputs. Expect B200 (8 TB/s) to be worse still,
  which makes a low-latency small-input path the biggest lever on the mean score. [VERIFIED-CSF3; INFERRED for B200]
- If A100 efficiency carried over unchanged, B200 would land near 21 µs (score about 0.65). It probably will not,
  because B200 needs about 4× the bytes in flight per SM and fixed costs weigh more at 1.6 µs floors.
- Triton 3.7 cross-compiles TMA kernels for sm_90a/sm_100a with no GPU present (UTMALDG/UTMASTG, no LDG/STG). Neither
  device-side nor host-side descriptors have been tested on a GPU yet. Device-side descriptors need `triton.set_allocator`
  and 128 B of global scratch per descriptor per program [VERIFIED-CSF3: `global_scratch` = 512 for 4 descriptors].
- FlashGPU-Sim cannot parse `createpolicy` (cache hints), and its simulated favourites (small inputs only) misranked
  the large ones.
- First portal submissions are queued with the user (`poc/b200_submit/`: v028, v039, v023, v037, v046).

## 8. Design priorities for #38 [INFERRED]

1. Single fused launch for Q and K.
2. L band: sustain ≥ 85% of 8 TB/s. That needs ≥ 48–64 KB of loads in flight per SM, few instructions per byte (TMA or
   1-D bulk rings, or 256-bit accesses in CUDA C++), streaming stores, and no tail effect (persistent grid of 148×k
   CTAs, or one-shot with many small tiles).
3. S band: minimise fixed cost. One even wave over all SMs, all loads issued up front, no device-side descriptor
   setup, tiny prologue.
4. Size dispatch: one specialist per band (S, M, L), chosen on the host by `B·S`.
5. Weights: keep them out of the DRAM stream. They are L2/L1 hits; a weight-stationary layout removes even those loads
   (`grid:persistent-wstat` in the playbook).

## 9. Open questions (and how to answer them)

| Question | How to find out |
|---|---|
| Does Triton run on the portal (the `launch.h` gap)? | first private submission (v028) |
| How much of A100's 76% efficiency carries to B200, per band? | portal results for the 5 queued variants against the H200/A100 predictions |
| B200 fixed cost: empty-kernel span, CTA ramp, tail? | H200 harness microbenchmarks; one paired portal test (2 slots) |
| Does the L2 flush leave dirty lines, so small kernels pay extra write-backs? | H200: time a reads-only kernel against read+write at S-band sizes |
| TMA ring vs pointer loads on HBM3e at large sizes? | H200 timing of `mem:tma-*` against `mem:ldg128` niches, then 1–2 B200 submissions |
| Do 256-bit accesses (CUDA C++) beat 128-bit on B200? | B200 only (H200 has no 256-bit); one paired submission |
| Exact per-workload Tb and Tsol? | unknown (hidden). If the portal's result page shows per-workload latency and score (unknown until the first result), back out Tb_i assuming Tsol_i ∝ bytes |
