# Problem card: L1 #85 `085_geglu_activation`

Definition and workloads from the task prompt (SOL-ExecBench v1.1). Measurements are from the rented B200 (unlocked
clocks, SM up to 1965 MHz; the portal locks 1500 MHz) through `probe_b200` and `run_tests`, 10 October 2026. No portal
submission yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim, `definition.json`), from stable-diffusion-xl-refiner-1.0:

```python
@torch.no_grad()
def run(x):
    x_gate, x_linear = x.chunk(2, dim=-1)
    return F.gelu(x_gate, approximate='tanh') * x_linear
```

- DPS signature: `run(x, output)`. `x`: `[B, S, 10240]` fp32 contiguous; `output`: `[B, S, 5120]` fp32.
- View `x` as `[R, 10240]` with `R = B·S` rows (128 to 8192). Row r: gate half `x[r, 0:5120]`, linear half
  `x[r, 5120:10240]`, 20 KB apart. `out[r, j] = gelu_tanh(x[r, j]) · x[r, 5120 + j]`.
- Pure elementwise: no reduction, no reuse, every byte touched once. 12 B of traffic per output element (8 read, 4
  written). Traffic per call = `B·S·61,440` bytes.
- Identity used by the kernel: `0.5·a·(1 + tanh(u)) = a · sigmoid(2u)`, so gelu·b = `a·b / (1 + exp(−2u))` with
  `u = sqrt(2/π)·(a + 0.044715·a³)`. One exp and one reciprocal per element, no cancellation near a = 0.
- 5120 = 2^10·5, so every row is 640 chunks of 8 floats, or 5 CTAs of 128 threads × 8 floats. No masks, no tails,
  no row with a size that needs a tail check for any B·S.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|`, ≥ 99% of elements, no NaN/Inf [PAPER: workload.jsonl]. Inputs are
  `randn`, so |out| reaches about 25 and the relative budget is about 84 ulp.
- The reference computes tanhf (≤ 2 ulp) in fp32 and materialises gelu, then multiplies. Our sigmoid form with
  `__expf` (ex2.approx) and `__fdividef` under `--use_fast_math` measured: max abs error 1.9e-6, 0 elements out of
  tolerance at 128 and 2048 rows [probe]. The `tanhf` form measured 9.5e-7 max abs error and was 3 to 10% slower
  (34 regs, more MUFU work) [probe]. Both have ≥ 5× margin on the absolute term.
- For u < −44, exp(−2u) overflows to inf and the result is exactly 0; the reference also gives 0 there
  (tanh = −1 exactly). No NaN path for finite inputs.
- `--use_fast_math` does not change the accuracy of either form (measured with and without) [probe].
- What would fail: bf16/fp16/TF32 anywhere; `tanh.approx.f32` (2^-11 relative, about 2000 ulp); unwritten rows.

## 3. The 16 workloads (sorted by size)

Bytes = read 8·B·S·5120 + write 4·B·S·5120. `floor8` = bytes / 8 TB/s. `copy` = the loop's plain copy of the same
bytes on the rented B200 (from `run_tests`). `ours` = r0-geglu-ldg256-2d-stel (`run_tests`, rented B200).
`ref` = PyTorch reference timed like the harness (task prompt).

| B,S | B·S rows | MB | band | floor8 µs | copy µs | ours µs | ours TB/s | ref µs | ref/ours |
|---|---|---|---|---|---|---|---|---|---|
| 1,128 | 128 | 7.86 | S | 0.98 | 3.9 | 3.6 | 2.2 | 11.98 | 3.3 |
| 1,131 | 131 | 8.05 | S | 1.01 | 3.9 | 3.7 | 2.2 | 11.93 | 3.2 |
| 2,211 | 422 | 25.93 | S | 3.24 | 7.4 | 6.2 | 4.2 | 20.49 | 3.3 |
| 1,512 | 512 | 31.46 | S | 3.93 | 8.3 | 7.0 | 4.5 | 21.89 | 3.1 |
| 4,128 | 512 | 31.46 | S | 3.93 | 8.2 | 6.9 | 4.6 | 22.06 | 3.2 |
| 1,1024 | 1024 | 62.91 | M | 7.86 | 14.4 | 10.8 | 5.8 | 37.77 | 3.5 |
| 2,512 | 1024 | 62.91 | M | 7.86 | 14.4 | 10.9 | 5.8 | 37.76 | 3.5 |
| 1,2048 | 2048 | 125.83 | M | 15.73 | 26.8 | 18.6 | 6.8 | 70.93 | 3.8 |
| 8,256 | 2048 | 125.83 | M | 15.73 | 27.0 | 18.7 | 6.7 | 70.86 | 3.8 |
| 4,541 | 2164 | 132.96 | M | 16.62 | 28.2 | 19.6 | 6.8 | 75.11 | 3.8 |
| 16,256 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.2 | 7.1 | 134.88 | 3.8 |
| 4,1024 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.1 | 7.2 | 135.20 | 3.9 |
| 32,128 | 4096 | 251.66 | L | 31.46 | 50.8 | 35.1 | 7.2 | 135.26 | 3.9 |
| 8,613 | 4904 | 301.30 | L | 37.66 | 59.8 | 42.5 | 7.1 | 160.01 | 3.8 |
| 16,449 | 7184 | 441.38 | L | 55.17 | 83.6 | 62.8 | 7.0 | 229.75 | 3.7 |
| 8,1024 | 8192 | 503.32 | L | 62.92 | 93.8 | 71.7 | 7.0 | 261.78 | 3.7 |

Only B·S matters (the three 4096-row shapes time identically), so 16 workloads are 11 sizes. Geometric mean of
floor8: 11.9 µs. Geometric mean of ours: 18.8 µs.

Bands (loop convention): S ≤ 31.5 MB (5 workloads, 128–512 rows), M ≤ 133 MB (5 workloads, 1024–2164 rows),
L above (6 workloads, 4096–8192 rows).

## 4. Bounds and what dominates per band

- **Memory-bound everywhere.** FLOPs per element about 12 (cubic, exp, reciprocal, 3 muls) → 0.67e12 elements/s at
  8 TB/s needs about 8 TFLOP/s against 57 TFLOP/s FP32 SIMT at 1.5 GHz. MUFU: 2 ops per element → 1.3e12/s against
  3.55e12/s at 1.5 GHz (37% of MUFU capacity; 28% at the rented clock). A copy-only variant (no GELU maths) timed
  identically to the full kernel at every size on the rented B200 [probe], so compute is hidden at 1965 MHz. At the
  portal's 1500 MHz the SM-side budget shrinks by 24%; MUFU at 37% should still hide, but it is the one compute risk
  (H5).
- **S band (7.9–31 MB):** fixed cost dominates. Empty kernel CUPTI span 1.73 µs; read-only kernel 3.5 µs and full
  kernel 3.6–3.8 µs at 7.9 MB, against a 0.98 µs byte floor [probe]. About 2.7 µs of fixed cost per call.
- **M band (63–133 MB):** mixed. 5.8–6.8 TB/s achieved; the dirty L2 left by the harness's flush and the store
  policy decide (evict_last stores gained 8–13% here).
- **L band (252–503 MB):** sustained bandwidth. 7.0–7.2 TB/s with evict_last stores, matching the 7.05–7.1 TB/s
  public ceiling for 1:1 read+write streams on B200 (`b200_sota.md` §2). Reads alone reach only 5.3 TB/s at 503 MB
  after the dirty flush [probe], the same effect #38 recorded as H7.
- **SOL (Tsol):** unknown until the first portal result. If SOLAR uses bytes/8 TB/s, our geomean is 1.58× it. On #38
  Tsol sat at about 0.53× the byte floor at M/L, so expect a similar unreachable anchor here.

## 5. Where the reference loses and what a fast kernel must do

- The reference runs two kernels: `gelu` on the strided gate view (read 4 B, write 4 B per element to a temporary),
  then `mul` (read 8 B, write 4 B). That is 20 B per element instead of 12, plus a temporary allocation, a second
  launch and a gap. Measured 3.1–3.9× our time.
- A fast kernel: one launch; each thread loads 8 gate floats and 8 linear floats with 256-bit loads, computes, stores
  8 floats with one 256-bit store under `L2::evict_last`; a one-shot grid with every load issued before any use.

## 6. Design priorities

1. **Keep evict_last on stores.** Measured gain at M/L (unlocked B200): 63 MB −8%, 126 MB −13%, 252 MB −12%,
   503 MB −6%; 0 at S [probe]. Same mechanism as #38 H1 (fewer in-window write-backs of the flush's dirty lines).
2. **One chunk of 8 floats per thread, small CTAs.** 2 chunks/thread +1–5%, 4 chunks/thread +3–10% (82 regs) [probe].
   CTA size 128/256/512 within 1% [probe]. 2-D grid (5 CTAs per row, no division) −2% at S versus a 1-D grid with
   div-by-640, equal at L [probe].
3. **S-band fixed cost (about 2.7 µs above the byte floor at 7.9 MB).** Largest score headroom; the lever set from #38
   applies (no prologue, all loads up front). The empty-kernel span of 1.73 µs is 45% of the S time.
4. **Compute at 1.5 GHz.** If H5 shows the maths costs time at the locked clock, cut MUFU work: a single-MUFU
   sigmoid (rcp of 1+exp via FMA Newton from a cheap seed), or `ex2` on a pre-scaled argument to drop the extra mul.
5. **Load hints:** `.nc` vs plain loads made no difference [probe]. evict_first on loads is untested here (#38: hurt L
   by 2%).

## 7. Results so far

r0-geglu-ldg256-2d-stel (CUDA C++, 32 regs, 0 smem, one launch): 16/16 pass; times in §3. Faster than the loop's
plain copy at every size (0.69–0.95× copy). Sustained 7.0–7.2 TB/s at L, 5.8–6.8 TB/s at M, 2.2–4.6 TB/s at S.
Fit on M/L (`t = fixed + bytes/BW`): about 2.5 µs + 7.25 TB/s. Awaiting the first portal submission for Tb and Tsol.

## 8. Open questions

- Tb and Tsol per workload (portal). Whether Tsol is bytes/8 TB/s or lower, as on #38.
- Does the GELU maths show at 1500 MHz (H5)? Needs a portal pair (copy-only is wrong, so compare a 1-MUFU variant).
- Can the S-band fixed cost drop below 2.5 µs? The empty-kernel span alone is 1.73 µs on the rented B200.
- Does the harness's dirty-flush write-back cost land the same way here (read-only 5.3 TB/s at L)? Any legitimate
  kernel-side mitigation remains unknown (#38 H7/H8).
- Does the portal's CUPTI window differ for a 2-D grid of 5×8192 CTAs versus a 1-D grid (CTA dispatch rate at L)?
