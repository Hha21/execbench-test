# Problem card: L1 #084 `084_silu_activation_backward`

Measurements are from the rented B200 (unlocked clocks; the portal locks SM at 1500 MHz) via probe_b200 and run_tests,
9 October 2026. No portal result yet, so Tb and Tsol are unknown.

## 1. Semantics

Reference (verbatim maths, `definition.json`, from stable-diffusion-xl):

```python
@torch.no_grad()
def run(grad_output, x, sigmoid_x):
    one_minus_sigmoid = 1.0 - sigmoid_x
    x_times_one_minus_sigmoid = x * one_minus_sigmoid
    bracket_term = 1.0 + x_times_one_minus_sigmoid
    local_grad = sigmoid_x * bracket_term
    return grad_output * local_grad
```

- DPS signature: `run(grad_output, x, sigmoid_x, grad_input)`. All four tensors are `[num_elements]` fp32, contiguous.
- Pure elementwise: `o[i] = g[i] * (s[i] * (1 + x[i] * (1 - s[i])))`. No reduction, no reuse, no broadcast operand.
  Every byte is touched exactly once; 16 B and 5 FLOPs per element.
- `num_elements` ranges from 131 to 16,777,216. Three sizes (131, 2053, 3089) are not multiples of 4, so a vector
  kernel needs a scalar tail (at most 3 elements). The harness keeps pointers 256-B aligned, so 16/32-B vector
  accesses are always legal on the main body.
- Traffic per call: `16 · num_elements` bytes (3 reads + 1 write), 2 KB at the smallest and 268 MB at the largest.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|` for ≥ 99 % of elements, no NaN/Inf, not all zeros [PAPER].
- All three inputs are `randn` (so `sigmoid_x` is NOT in [0,1]; it is N(0,1) too). Outputs are products of up to
  four N(0,1) values and reach magnitude > 100, so rtol dominates; near cancellation in `1 + x(1−s)` the output is
  small and atol covers it.
- Our kernel uses `__fmul_rn/__fadd_rn/__fsub_rn` in the reference's exact order, so `--use_fast_math` cannot
  contract to FMA and the result is **bit-identical to PyTorch eager** (verified: 0 mismatching elements at every
  size tested, max |err| = 0). Any fp32 reordering or FMA would also pass (≤ 2–3 ulp), but exactness is free here.
- What would fail: bf16/fp16/TF32 anywhere (forbidden anyway); leaving the tail elements unwritten (outputs arrive
  zero-filled, 3 of 131 elements = 2.3 % > the 1 % allowance).

## 3. The 16 workloads (sorted by size)

`floor8` = 16n / 8 TB/s. "stream" = plain read+write stream of the same bytes measured on the rented B200 (briefing).
"ref" = PyTorch reference timed like the harness (5 eager kernels). "ours" = run_tests of the first kernel.
"compile" = `torch.compile` of the reference (one Triton kernel), our estimate of what the hidden PyTorch-only
baseline could be.

| n | MB | floor8 µs | stream µs | ref µs | ours µs | ours TB/s | compile µs | band |
|---|---|---|---|---|---|---|---|---|
| 131 | 0.002 | 0.0003 | 2.02 | 17.9 | 2.1 | – | 2.11 | S |
| 2053 | 0.033 | 0.004 | 2.05 | 17.8 | 2.4 | – | – | S |
| 3089 | 0.049 | 0.006 | 2.05 | 17.9 | 2.5 | – | – | S |
| 4096 | 0.066 | 0.008 | 2.05 | 17.8 | 2.4 | – | – | S |
| 16384 | 0.26 | 0.03 | 2.13 | 18.0 | 2.3 | 0.11 | 2.16 | S |
| 40960 | 0.66 | 0.08 | 2.17 | 18.1 | 2.4 | 0.27 | – | M |
| 163840 | 2.62 | 0.33 | 2.59 | 18.2 | 2.8 | 0.94 | 2.72 | M |
| 262144 | 4.19 | 0.52 | 3.14 | 18.5 | 3.1 | 1.35 | – | M |
| 655360 | 10.49 | 1.31 | 4.35 | 21.0 | 4.2 | 2.50 | – | M |
| 786432 | 12.58 | 1.57 | 4.77 | 21.3 | 4.6 | 2.73 | 4.67 | M |
| 2097152 | 33.55 | 4.19 | 8.80 | 27.7 | 7.3 | 4.60 | 8.13 | L |
| 4194304 | 67.11 | 8.39 | 15.83 | 41.4 | 11.8 | 5.69 | – | L |
| 5242880 | 83.89 | 10.49 | 19.55 | 47.0 | 13.8 | 6.08 | – | L |
| 8388608 | 134.22 | 16.78 | 30.00 | 65.4 | 20.0 | 6.71 | 25.8 | L |
| 10485760 | 167.77 | 20.97 | 37.01 | 81.8 | 25.1 | 6.68 | – | L |
| 16777216 | 268.44 | 33.55 | 55.50 | 131.8 | 38.5 | 6.97 | 46.6 | L |

Bands (loop convention): S ≤ 0.3 MB (5 workloads), M 0.66–12.6 MB (5), L 33.5–268 MB (6). Only `num_elements`
matters, so dispatch (if ever needed) keys on it.

## 4. Bounds and what dominates per band

- Arithmetic intensity 5/16 FLOP/B. At 57 TFLOP/s fp32 SIMT (1.5 GHz) the largest workload needs 1.5 µs of compute
  against 33.6 µs of DRAM time: **memory-bound at every size, launch-bound below about 1 MB**.
- Floor at 8 TB/s: 0.0003–33.6 µs; geometric mean of floor8 over the 16 workloads ≈ 0.86 µs, which is meaningless
  for S where the real floor is the launch span. SOLAR likely sits even lower (for #38 it was 0.53× the byte floor);
  **score 1.0 is unreachable, and S-band scores depend almost entirely on the hidden Tb**.
- Measured fixed cost: empty kernel 1.57–1.66 µs CUPTI span; our kernel at 131 elements 2.1–2.3 µs. So the S band
  is one DRAM round trip above the empty-kernel floor and has essentially no headroom.
- Our kernel fits `t ≈ 2.1 µs + bytes / 7.3 TB/s` at L (marginal 134→268 MB: 7.25 TB/s); at 33.5 MB the effective
  rate is only 4.6 TB/s because the fixed 2.1 µs and the dirty-L2 write-back cost (briefing H7) are a large share.
- run_tests compares against a plain copy of the same bytes: ours is 1.05–1.2× the copy at S/M (noise and the tail
  thread) and **0.67–0.83× the copy at L** (the cache hints beat a hint-less copy).

## 5. Where the reference loses and what a fast kernel must do

- Eager PyTorch launches 5 kernels (rsub, mul, add, mul, mul) and materialises 4 intermediates: 17.9 µs at S
  (5 launches × ~3.5 µs with gaps) and 3.4× the bytes at L (131.8 µs at 268 MB ≈ 2.4× our time).
- `torch.compile` fuses it into one Triton kernel: 2.11 µs at S (equal to ours) and 46.6 µs at 268 MB (21 % slower
  than ours, because it uses default cache policies and hits the dirty-L2 write-back cost). If NVIDIA's agentic
  baseline found this, Tb ≈ our time at S and ≈ 1.2× ours at L.
- A fast kernel must: launch once; issue all loads before any use; use L2 evict_first on the three input streams
  and evict_last on the output (measured −10 % at ≥ 34 MB, neutral at S); keep registers ≤ 32 (ours: 24) so 2048
  threads per SM are resident; write the ≤ 3 tail elements.

## 6. Design priorities

1. **M/L bandwidth under the dirty-L2 regime** is the only lever with headroom: 6.7–7.0 TB/s now vs the 8 TB/s
   floor. Candidates: compare EF+EL against `.cs` loads (tied in probes), 256-bit loads with EF (`ld.global.nc.L2::evict_first`
   qualifier form), and whether partial/fractional policies shift the balance. Expect ≤ 3 %.
2. **S band: nothing to gain** (2.1–2.4 µs vs 1.6 µs empty). Keep the S path minimal; do not add prologues,
   persistent loops or TMA. Check on the portal whether the tail-thread path costs anything at 2053/3089 (run_tests
   shows 2.4–2.5 vs 2.1 at 131, probably noise).
3. No per-size dispatch is needed: one config (256 threads, 1 float4 per thread) ties the best at every size.
4. Avoid: TMA/bulk rings (+1 µs fixed, refuted on #38), persistent grids, clusters, 256-bit accesses at tiny sizes
   (+0.8 µs at 131 elements, cause unknown).

## 7. Results so far

First kernel `silu-bw-cuda-ldg128-os-efel` (CUDA C++, two files): 16/16 PASSED on the rented B200 (run_tests,
20.8 s). Times in §3 "ours". sm_100a: 24 regs, 0 smem, 0 spills, 3 × LDG.128 + 1 × STG.128 per thread.
Probe ladder at 268 MB: plain 43.0 µs → EL stores 40.2 → EF loads + EL stores 38.8 (= `.cs` loads 38.9;
`nc`+EF 38.9; `L2::256B` prefetch hint 40.2, no gain; `nc` alone 40.2). Block size 128/256/512/1024 and unroll
1/2/4: all within ±1 % (3 repeats), so noise. Portal submission pending for Tb/Tsol anchors.

## 8. Open questions

- What is the hidden Tb per workload? If it is torch.compile-like (2.1 µs at S), S scores are capped near 0.5 and
  the 6 L workloads decide the problem. If it is eager-like (18 µs), S scores approach 0.9.
- Does the portal's 1500 MHz SM lock change the S floor (empty span 1.6 µs here at unlocked clocks)?
- Why do 256-bit loads cost +0.8 µs at 131 elements (one CTA, 16 active lanes)? Irrelevant for the design but
  worth understanding before any 256-bit variant.
- Can anything reduce the dirty-L2 write-back cost further than EF loads + EL stores (e.g. ordering the output
  stores before loads of later tiles, or fractional evict_last)? The remaining L gap to 8 TB/s is about 5 µs at
  268 MB.
