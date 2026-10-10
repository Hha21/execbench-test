# Problem card: L1 #88 `088_rotary_position_embedding_application`

Research round r0, 10 October 2026. No portal result yet; Tb and Tsol per workload are unknown until the first
submission. All timings below are from the rented B200 (unlocked clocks, harness CUPTI methodology, cold L2).

## 1. Semantics

Reference (from HunyuanImage-2.1): `query_rotated = query * cos + rotate_half(query) * sin`, same for key, with
`rotate_half(x) = cat(-x[..., 64:], x[..., :64])`. Everything fp32.

- DPS signature: `run(query, key, cos, sin, query_rotated, key_rotated)`.
- `query`: `[B, H, S, 128]`, `key`: `[B, Hkv, S, 128]`, `cos`/`sin`: `[S, 128]`, all fp32 contiguous. Outputs have
  the input shapes. No scalars.
- Per element, for column i < 64 of a row at sequence position s:
  `out[i] = x[i]·cos[s,i] − x[i+64]·sin[s,i]` and `out[i+64] = x[i+64]·cos[s,i+64] + x[i]·sin[s,i+64]`.
  Each 512 B row is independent; the only cross-element dependence is between columns i and i+64 of the same row.
- Row view: `R = B·(H+Hkv)·S` rows of 128 floats, Q rows first then K rows. Row `rl` of a stream sits at sequence
  position `s = rl % S`, so cos/sin row `s` is shared by `B·H` (or `B·Hkv`) rows that are `S·512` B apart.
- `cos` and `sin` are generated independently as `cos(randn.clamp(-2,2))` and `sin(randn.clamp(-2,2))`
  [PAPER: io.py `_is_rope_cos_sin`]: values in [−1, 1], **no** cos²+sin²=1 and no symmetry between the halves. The
  kernel must read all 128 columns of both tables.
- Traffic per call: read Q and K once, write both outputs once, plus the two tables:
  `bytes = 2·R·512 + 2·S·512`. The tables are at most 8 MB (S = 8192) and are re-read from L2: 2 B of table per
  1 B of x read.

## 2. Numerics and the 1e-5 tolerance

- Tolerance: `|out − ref| ≤ 1e-5 + 1e-5·|ref|` on ≥ 99% of elements, no NaN/Inf, not all zero
  [PAPER: workload.jsonl, correctness.py]. One fp32 ulp is 1.19e-7 relative, so the budget is about 84 ulp.
- The reference does two rounded products and one rounded add per element. Our kernel contracts into
  `fma(x, cos, −x2·sin)` under `--use_fast_math`: the difference is ≤ 2 ulp of the larger product. Measured max
  abs difference against the reference: 2.4e-7 [probe_b200].
- Cancellation is harmless: when the two products nearly cancel, the absolute error stays at the ulp of the products
  (≤ about 5e-7 for |x| ≤ 5), well under the 1e-5 absolute term.
- Nothing may be stored or computed below fp32. There are no reductions, rsqrt or divisions.

## 3. The 16 workloads (sorted by bytes)

`floor8` = bytes / 8 TB/s. "copy" = the plain read+write stream of the same bytes on the rented B200 (from the
briefing). "ref" = the PyTorch reference timed like the harness. "r0" = our first kernel (run_tests, same B200).
CTAs: 64-thread path has 8 rows per CTA, 256-thread path 32 rows.

| B,H,Hkv,S | R rows | Q rows | MB | FLOP M | floor8 µs | copy µs | ref µs | ref/copy | r0 µs | r0/copy | band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,8,8,128 | 2,048 | 1,024 | 2.23 | 0.8 | 0.28 | 2.47 | 44.1 | 17.9 | 3.0 | 1.23 | S |
| 1,40,8,131 | 6,288 | 5,240 | 6.57 | 2.4 | 0.82 | 3.46 | 49.0 | 14.2 | 3.6 | 1.04 | S |
| 2,32,4,211 | 15,192 | 13,504 | 15.77 | 5.8 | 1.97 | 5.00 | 65.0 | 13.0 | 4.9 | 0.98 | S |
| 4,24,6,293 | 35,160 | 28,128 | 36.30 | 13.5 | 4.54 | 8.13 | 90.4 | 11.1 | 7.5 | 0.93 | S |
| 1,56,8,853 | 54,592 | 47,768 | 56.78 | 21.0 | 7.10 | 11.46 | 120.9 | 10.6 | 10.4 | 0.91 | S |
| 8,16,4,373 | 59,680 | 47,744 | 61.49 | 22.9 | 7.69 | 12.31 | 131.7 | 10.7 | 11.3 | 0.92 | M |
| 2,48,6,919 | 99,252 | 88,224 | 102.58 | 38.1 | 12.82 | 18.83 | 204.0 | 10.8 | 16.6 | 0.88 | M |
| 64,8,8,128 | 131,072 | 65,536 | 134.35 | 50.3 | 16.79 | 23.96 | 248.7 | 10.4 | 20.0 | 0.83 | M |
| 8,24,8,1024 | 262,144 | 196,608 | 269.48 | 100.7 | 33.69 | 44.94 | 489.4 | 10.9 | 38.8 | 0.86 | M |
| 8,28,4,1024 | 262,144 | 229,376 | 269.48 | 100.7 | 33.69 | 44.89 | 489.5 | 10.9 | 39.6 | 0.88 | M |
| 8,32,4,1087 | 313,056 | 278,272 | 321.68 | 120.2 | 40.21 | 52.93 | 579.9 | 11.0 | 51.3 | 0.97 | L |
| 4,48,12,1321 | 317,040 | 253,632 | 326.00 | 121.7 | 40.75 | 53.72 | 582.3 | 10.8 | 51.6 | 0.96 | L |
| 4,36,6,2048 | 344,064 | 294,912 | 354.42 | 132.1 | 44.30 | 57.93 | 630.8 | 10.9 | 56.3 | 0.97 | L |
| 2,40,8,4096 | 393,216 | 327,680 | 406.85 | 151.0 | 50.86 | 65.54 | 717.0 | 10.9 | 63.2 | 0.96 | L |
| 1,64,8,8192 | 589,824 | 524,288 | 612.37 | 226.5 | 76.55 | 97.95 | 1055.7 | 10.8 | 95.8 | 0.98 | L |
| 32,16,4,4096 | 2,621,440 | 2,097,152 | 2688.55 | 1006.6 | 336.07 | 413.22 | 4564.8 | 11.0 | 400.0 | 0.97 | L |

Geometric means over the 16 workloads: floor8 14.2 µs, plain copy about 21 µs, reference 284 µs, r0 23.5 µs.
Unlike #38, the shapes do not collapse into a few sizes: `R` differs for all 16, and the Q/K split varies
(K is 11–50% of the rows). Dispatch keys on `R` (and `S` for the table size).

## 4. Bounds and what dominates per band

- Arithmetic intensity is 3 FLOP per 8 B of DRAM traffic (0.375 FLOP/B). At 57 TFLOP/s fp32 SIMT the biggest
  workload needs 18 µs of maths against 336 µs of bytes. Every workload is memory-bound; the SOL time is almost
  surely `bytes / peak BW`, so Tsol ≈ floor8 (maybe without the cos/sin bytes, which are < 1.5% of traffic except at
  2.2 MB where they are 6%).
- Extra on-chip traffic: cos/sin rows come from L2 at 2 B per 1 B of x read, about 4 TB/s of L2 reads when DRAM runs
  at 7 TB/s. L2 serves it (21 TB/s), and the probe where two heads share one table load was not faster, so it is not
  on the critical path yet [probe_b200].
- S band (2–57 MB): fixed cost. An empty grid of the same CTA count spans 1.76 µs at 2.2 MB; our kernel takes 2.6–3.0
  µs there against a 0.28 µs byte floor. The smallest workload has only 2,048 rows: 256 CTAs of 64 threads, under two
  per SM, so the first DRAM round trip and the tail are the whole time.
- M band (61–269 MB): mixed. r0 is already 12–17% faster than the plain copy because evict_last stores keep part of the
  output write-back outside the window (the #38 H1 effect carries over). This is where the copy baseline loses most.
- L band (322–2689 MB): sustained bandwidth, 6.3–6.7 TB/s in r0 (unlocked clocks). CTA dispatch is a real cost here:
  an empty grid of 64-thread CTAs takes 0.55 µs per 1,000 CTAs, 26 µs for the 49k CTAs of the 407 MB workload
  [probe_b200], so the L path uses 256-thread CTAs (32 rows each). 512- and 1024-thread CTAs are 3–8% slower again.

## 5. Where the reference loses and what a fast kernel must do

- The reference is 10–18× the plain copy. It materialises `-x2`, the `cat` (a full extra tensor), `x*cos`,
  `rotate_half(x)*sin` and the sum, for Q and for K: about 5 full passes of read+write per stream, plus about ten
  kernel launches whose gaps dominate at small sizes (44 µs for 2.2 MB).
- A fast kernel: one fused launch over Q and K; each thread holds both halves of its column pair (lane j owns
  columns 8j..8j+7 and 64+8j..64+8j+7), so no shuffles and no second pass; 256-bit loads and stores; per-row
  cos/sin fetched from L2; evict_last on the output stores; CTA size chosen per row count.
- Measured on the rented B200 (within one probe session; sessions differ by up to 10%, compare only within a line):

| design knob | 36 MB | 61 MB | 269 MB | 407 MB |
|---|---|---|---|---|
| 256 thr, 1 row/thread, plain stores | 8.02 | 11.48 | 42.10 | 61.93 |
| 256 thr, 1 row/thread, evict_last stores | 7.11 | 10.49 | 39.05 | 56.73 |
| 64 thr, 1 row/thread, evict_last | 6.94 | 10.18 | 38.77 | 59.75 |
| 128 thr, 1 row/thread, evict_last | 7.07 | 10.13 | 38.82 | 59.66 |
| 256 thr, 2 rows/thread (108 regs) | 7.75 | 11.49 | 43.43 | 65.09 |
| float4 layout, 16 lanes/row (32 regs), 128 thr | 6.85 | – | 38.85 | 59.83 |
| 2 heads per thread sharing cos/sin (80 regs), 256 thr | 7.04 | – | 38.39 | 60.96 |

## 6. Design priorities

1. Keep: one launch, evict_last output stores (−7 to −11%), 256-bit accesses, dispatch on `R`
   (64-thread CTAs up to 262,144 rows, 256-thread above).
2. S band fixed cost: 2.6–3.0 µs against a 1.76 µs empty-grid span and a 0.28 µs byte floor. Try fewer, fatter
   CTAs only if they issue all loads up front; test x-load hints (`.nc`/`L1::no_allocate` vs plain) and whether
   the `rl % S` modulo and the 60-register body cost anything at the locked 1.5 GHz.
3. L band bandwidth: 6.3–6.7 TB/s. Levers: register count (60 → 4 CTAs/SM at 256 threads; the float4 layout at 32
   regs ties, so occupancy is not the limit yet), the DRAM page pattern of the Q/K split, and a grid that avoids the
   dispatch-rate ceiling without going to 512-thread CTAs.
4. M band: already 12–17% below the plain copy. Check whether evict_last on only the first output or a fraction is
   better (#38 H15 said no; verify here since bytes differ).
5. Do not spend slots on: TMA/bulk rings, clusters, multi-row-per-thread register tiles (108 regs lost everywhere),
   or sharing cos/sin across heads (neutral).

## 7. Results so far

- `r0-rope-ldg256-os-stel` (CUDA C++, 60 regs, 0 smem, no spills): 16/16 workloads pass on the B200 test bench.
  Times in §3: geomean 23.5 µs, 0.83–0.98× the plain copy except 1.23× at 2.2 MB and 1.04× at 6.6 MB.
- First submission to the portal will give Tb and Tsol per workload and unlock the score model.

## 8. Open questions

- Does SOLAR count the cos/sin bytes and the real 8.18 TB/s, or something lower like #38's 0.53× floor?
- Why do probe sessions differ by 10% at M/L (39.05 vs 42.8 µs at 269 MB for the same binary)? Clock state on
  the rented card; the portal locks clocks, so only within-session comparisons are trusted.
- Is the smallest workload (2.2 MB, 2,048 rows) better served by 4-row or 1-row CTAs, or by fewer CTAs with
  several rows in flight per thread?
- Does the 32-byte alignment of the shifted pointers always hold (the harness shifts by multiples of 256 B; the
  binding checks it and raises otherwise)?
