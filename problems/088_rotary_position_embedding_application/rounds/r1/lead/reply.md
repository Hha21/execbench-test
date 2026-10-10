The portal page settles two things the r0 session could not: Tsol per workload and how the rented card maps onto the locked-clock portal. Working through the numbers before writing the ledger.

Per-workload Tsol back-solved from the portal scores (t, Tb, S) comes out at about 0.53 × bytes/8 TB/s at every size from 134 MB up, exactly the #38 pattern. That fixes the ceiling: hitting 7 TB/s at L would score about 0.68 per workload, 8 TB/s about 0.74.

The more important observation is the portal-versus-bench gap. It is a uniform +15 to +19% on every band, including the 2.7 GB workload, even though the DRAM clock is the same on both cards. r0 sustains only 5.6 to 6.0 TB/s on the portal at L, while the #38 kernel with the same structure reached 6.84 TB/s on the same portal. The one structural difference is that RoPE reads 2 B of cos/sin from L2 per byte of x, versus 1 B of weights for #38. L2 bandwidth scales with the SM clock, so the unlocked bench hides it. That is the round's main hypothesis, and it also explains why the baseline is unusually fast on the S=128 shapes where the table fits in L1.

### Assessment

Score 0.6351. The per-workload table shows three distinct losses. The smallest workload (2.2 MB) scores 0.437 because our 3.6 µs is slower than the 2.9 µs baseline, so a sub-3 µs kernel exists on this harness. M/L workloads sit at 0.59–0.69 with the portal running r0 at only 5.6–6.0 TB/s at L, against 6.84 TB/s for #38's structurally identical stream on the same portal. The portal/bench ratio is a uniform +15–19% on all bands, including 2.7 GB, despite identical DRAM clocks, so the L band is bound by something that scales with the SM clock. The prime suspect is the cos/sin table traffic from L2 (2 B per byte of x, about 12 TB/s of total L2 traffic at 6 TB/s DRAM, near the die-crossing limit at 1.5 GHz). This re-opens H3, which the unlocked bench refuted for a reason the bench cannot see.

Tsol is about 0.53 × bytes/8 TB/s at ≥134 MB (H10 refuted, as in #38), so the per-workload ceiling is 0.68 at 7 TB/s and 0.74 at 8 TB/s. A 10% cut at L is worth +0.015, at M +0.012, at S +0.010.

This round: E1 cuts table L2 traffic 4–8× with (s-tile × head-group) one-shot CTAs (L1-shared remap, smem-staged, and register-held variants) and tries locking the bench clocks to reproduce the portal regime. E3 builds the persistent s-stationary pipelined program (CAKE's mid-size design) for the same hypothesis in a different grid niche. E2 is a disciplined S-band A/B scan of the load/store path against the 2.9 µs baseline target.

```yaml ledger
# Hypothesis ledger for #88 (RoPE application), seeded 10 October 2026 by the r0 research session.
# r1 update 10 October 2026 after the first portal result (r0-rope-ldg256-os-stel, 0.6351).
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot (r13, H21).'
- 'Clock locking on the rented card (nvidia-smi -lgc/-lmc) is a probe tool only; never change clocks or any device state from a solution.'
hypotheses:
- id: H1
  statement: evict_last on the output stores cuts in-window dirty write-backs, as in #38, so M sizes gain most.
  status: supported
  evidence:
  - 'probe r0: EL vs plain stores, same kernel: 36 MB 7.11 vs 8.02 us, 61 MB 10.49 vs 11.48, 269 MB 39.05 vs 42.10, 407 MB 56.73 vs 61.93 [probe_b200]'
  - 'run_tests r0: 0.83-0.88x the plain copy at 61-269 MB'
- id: H2
  statement: CTA dispatch rate (about 0.55-0.65 us per 1000 CTAs regardless of 64 or 256 threads) caps 64-thread one-shot grids at L, so 256-thread CTAs win there while 64-thread CTAs win at S/M.
  status: supported
  evidence:
  - 'probe r0: empty grids 64x32768 18.0 us, 64x49152 26.4 us, 256x8192 5.3 us, 256x12288 7.4 us, 256x81920 43.5 us'
  - 'probe r0: 64-thr vs 256-thr: 36 MB 6.94 vs 7.11, 269 MB 38.77 vs 39.05, 407 MB 59.75 vs 56.73'
  - 'probe r0: 512 and 1024 threads per CTA are 3-8% slower than 256 at 269-2689 MB'
- id: H3
  statement: Sharing one cos/sin load across several heads (rows S*512 B apart) cuts L2 traffic and helps L.
  status: open
  evidence:
  - 'probe r0 (unlocked clocks): 2-head variant (80 regs): 269 MB 38.39 vs 39.05 (noise), 407 MB 60.96 vs 56.73'
  - 'r1 lead: re-opened. The bench runs the SM/L2 clock above 1.5 GHz, so it cannot show an L2-bandwidth limit that the locked portal has (H11). Re-test with the portal, or with locked clocks on the bench if nvidia-smi allows it.'
- id: H4
  statement: Occupancy (60 regs -> 4 CTAs/SM at 256 threads) limits L-band bandwidth.
  status: refuted
  evidence:
  - 'probe r0: float4 layout with 32 regs (full occupancy) ties the 60-reg v8 layout at 36 and 269 MB and is 5% slower at 407 MB'
- id: H5
  statement: More rows per thread (register tiles) raise bytes in flight and help L.
  status: refuted
  evidence:
  - 'probe r0: 2 rows/thread at 256 threads (108 regs): slower at every size (+9% at 36 MB, +15% at 407 MB)'
- id: H6
  statement: The S-band fixed cost (2.6-3.0 us at 2.2 MB vs 1.76 us for an empty grid) can be cut by issuing all of a CTA's loads before any use with fewer CTAs, or by 1-4 row CTAs.
  status: open
  evidence:
  - 'probe r0: 64/128/256-thread one-row-per-thread variants are within 0.15 us of each other at 2.2 MB (2.62-2.77 us)'
  - 'portal r0: (1,8,8,128) 3.6 us vs Tb 2.9 us, score 0.437, the worst workload; a 2.2 MB kernel at <= 2.9 us exists on this harness (the PyTorch-side baseline). Portal/bench ratio at S is 1.19 (6.2 vs 5.2 us geomean).'
- id: H7
  statement: Load hints on x (.nc, L1::no_allocate) and on cos/sin (.nc) change nothing measurable; plain loads would do.
  status: open
  evidence:
  - 'r0 uses .nc + L1::no_allocate on x and .nc on tables without an A/B'
  - 'r1 lead: the 0.7 us gap to the 2.9 us baseline at 2.2 MB makes the .nc/texture path and the 256-bit loads suspects for extra latency; E2 A/Bs them.'
- id: H8
  statement: The per-row integer modulo (rl % S) and the 60-register body are cheap at 1.5 GHz; a (stream, bh, s-tile) grid that avoids the modulo gains nothing.
  status: open
  evidence:
  - 'probe r0: the 2-head variant uses a modulo-free grid and was not faster (unlocked clocks)'
- id: H9
  statement: Partial or first-output-only evict_last is no better than evict_last on all stores (as #38 H15), since the gain saturates at about 25-30 MB of parked write-back.
  status: open
  evidence: []
- id: H10
  statement: SOLAR for this problem is bytes/8 TB/s including cos/sin, so scores at the floor would be about 0.75-0.8 as in #38.
  status: refuted
  evidence:
  - 'portal r0: Tsol back-solved from (t, Tb, S) is about 0.53 x bytes/8 TB/s at >= 134 MB (2689 MB: ~177 us vs floor8 336; 269 MB: ~18 vs 33.7; 134 MB: ~9.2 vs 16.8), the same 0.53x as #38. At S it is 0.6-0.9x floor8 but ill-conditioned (0.1 us rounding).'
  - 'consequence: per-workload ceiling at L is about 0.68 at 7 TB/s and 0.74 at 8 TB/s; r0 sits at 0.60-0.62 there.'
- id: H11
  statement: At the locked 1.5 GHz SM/L2 clock, the cos/sin table reads from L2 (2 B per byte of x, about 12 TB/s of total L2 traffic at 6 TB/s DRAM, half of it crossing dies) are the M/L limiter on the portal; cutting table traffic 4-8x (s-tile x head-group CTAs sharing the table through L1 or shared memory, or table-stationary persistent CTAs) recovers 8-15% at M/L.
  status: open
  evidence:
  - 'portal r0: L band 5.6-6.0 TB/s (407 MB 72.8 us, 612 MB 107.0 us, 2689 MB 447.6 us) vs #38 r3-cute-ldg256-os-r16 at 6.84 TB/s on the same portal (805 MB, 117.8 us) with 1 B of L2 weight reads per byte of x instead of 2 B.'
  - 'portal vs bench: +15% at M and L (25.9 vs 22.5, 96.1 vs 83.7 us) and +19% at S, with the same DRAM clock (3996 MHz is the HBM3e maximum); a DRAM-bound kernel would not scale with the SM clock.'
  - 'portal baseline: Tb is 4.7 TB/s at (64,8,8,128), where the 64 KB tables fit in L1, but 4.1-4.3 TB/s at S >= 1087: the baseline kernel also pays for table traffic at large S.'
  - 'probe r0 (unlocked): sharing the table across 2 heads was neutral at 269 MB; not evidence against H11 because the bench L2 clock is higher.'
- id: H12
  statement: The 0.7 us gap at 2.2 MB between r0 (3.6 us) and the baseline (2.9 us) comes from the load/store path (ld.global.nc texture path, 256-bit accesses, the cache_hint store) rather than from CTA shape, since 64/128/256-thread shapes tie on the bench.
  status: open
  evidence:
  - 'portal r0: 3.6 us vs Tb 2.9 us at (1,8,8,128); bench shapes within 0.15 us of each other [probe_r0]'
- id: H13
  statement: A persistent, table-stationary grid (each CTA owns an s-tile whose cos/sin slice sits in shared memory, loops over heads with the next row's x loads issued before the current row's FMAs) beats the one-shot grid at M (61-269 MB) by removing table traffic and the per-CTA ramp, as CAKE's mid-size program does on B200.
  status: open
  evidence:
  - 'literature: CAKE pipelined persistent program capped at 4-6 CTAs/SM beat our #38 one-shot by 0.6 us at 25 MB, tied at >= 400 MB [cake-pr5741]'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H11",
    "title": "Cut cos/sin L2 traffic 4-8x with (s-tile x head-group) one-shot CTAs; try locking the bench clocks to reproduce the portal regime",
    "operation": "structural_mutation",
    "parents": ["r0-rope-ldg256-os-stel"],
    "band": "all",
    "instructions": "Context: on the portal r0 sustains only 5.6-6.0 TB/s at L (72.8 us at 407 MB, 447.6 us at 2689 MB) while #38's structurally identical stream reached 6.84 TB/s on the same portal; the portal/bench ratio is a uniform +15% at M/L with the same DRAM clock. Suspect: table reads from L2 (2 B per byte of x) at the locked 1.5 GHz L2 clock. Step 1 (probes, seconds each, 3 interleaved repeats, same session): (a) try `nvidia-smi -lgc 1500,1500` (and `-lmc 3996`) on the rented card; if it is permitted, measure r0 at 269, 407 and 2689 MB locked vs unlocked and record the ratio (the portal shows +15%); reset clocks afterwards. (b) Build a probe-only 'no-table' copy of r0 (cos/sin replaced by constants, wrong output, never submitted) and a plain copy kernel; measure r0 vs no-table vs copy at 134, 269, 407, 2689 MB. The r0 minus no-table gap is the table cost at the bench clock; at locked clocks (if (a) works) it should grow. Step 2: build variant A 'remap': r0's body unchanged, but each CTA covers a tile of TS sequence positions x HG heads of one stream (256-thread CTA: TS x HG = 32 as (8,4), (4,8), (16,2); 64-thread CTA: 8 as (4,2), (2,4)). Map threads so the HG warps that share an s-tile are in the same CTA, keep x loads L1::no_allocate and table loads default-cached .nc so the HG-fold reuse is served by L1. Tile list: Q tiles = ceil(BH/HG) x ceil(S/TS), then K tiles with Hkv; mask s >= S and bh >= BH per thread (S = 131, 211, 293, 853, 919, 1087, 1321 are not multiples of 8). Step 3: variant B 'smem': same tiling, but the CTA first loads its TS-row cos and sin slice (TS KB) into static shared memory with one 256-bit load per thread, __syncthreads, then each thread reads its 4 x 32 B from shared memory; x path and evict_last stores as r0. Step 4: variant C 'reg': table held in registers, a non-unrolled loop over HG heads per thread (stride S*512 B), x loads for head h+1 issued before the FMAs of head h if registers stay <= 64. Step 5: measure A, B, C for each (TS,HG) against r0, interleaved, at 61, 134, 269, 407, 612 and 2689 MB; run_tests 16/16 for the two best. Keep r0's S path (<= 36 MB) unless a variant ties or wins there too. Submit the best dispatcher to the portal (one slot; a second only if the first is within noise and a clearly different variant remains). Record in findings: the clock-lock outcome, the no-table gap per size, the best (TS,HG) and the A/B/C ranking.",
    "success": "Portal L-band geomean <= 91 us (>= 5% below r0's 96.1) and M-band <= 25.1 us (>= 3%), with no size slower than r0 on the bench by more than 2%. Supporting signal before the slot: the no-table probe is >= 4% faster than r0 at L on the bench, or the locked-clock run shows r0 slowing >= 10% while the no-table probe slows less.",
    "refuted_if": "The best table-sharing variant is within 2% of r0 at L on the portal AND the no-table probe is within 2% of r0 on the bench: the table traffic is not the limiter, and the +15% portal/bench gap must be SM-side issue rate or dispatch at 1.5 GHz (then open a new hypothesis on instructions per byte).",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H12",
    "title": "S-band load/store path A/B scan against the 2.9 us portal baseline at 2.2 MB",
    "operation": "knob_mutation",
    "parents": ["r0-rope-ldg256-os-stel"],
    "band": "S",
    "instructions": "Target: (1,8,8,128) scores 0.437 because r0 takes 3.6 us on the portal against a 2.9 us baseline; on the bench r0 is 3.0 us. Build single-change variants of r0's 64-thread S path and measure each at 2.2, 6.6, 15.8 and 36 MB, interleaved with r0, at least 5 repeats per variant, report medians and spread (kernels under 10 us show 5-9% launch-order noise): (a) x loads plain `ld.global.v8.f32` (no .nc, no L1::no_allocate); (b) table loads plain `ld.global.v8.f32`; (c) a 128-bit layout: 16 lanes per row, 128 threads per 8-row CTA, plain `ld.global.v4.f32` and `st.global.v4` (Triton-like, tests whether 256-bit accesses add latency); (d) plain `st.global.v8` stores without createpolicy/cache_hint, S path only; (e) 32-thread CTAs with 4 rows (512 CTAs at 2.2 MB) and 32-thread CTAs with 2 rows at 16 lanes per row (1024 CTAs); (f) probe-only references: the same grid with no table loads (constants, wrong output) and a plain copy kernel, to bound what is reachable at each size. Then combine the winning changes, run_tests 16/16, and if the combination beats r0 by >= 0.25 us at 2.2 MB and is not slower at 6.6-36 MB, submit it to the portal with the M/L path unchanged (dispatch threshold stays at 262,144 rows unless the scan says otherwise). Record every variant's median per size in findings, including the losers.",
    "success": "Bench: >= 0.25 us (8%) faster than r0 at 2.2 MB and >= 3% at 6.6-15.8 MB across the interleaved repeats. Portal: (1,8,8,128) <= 3.2 us (from 3.6) and S-band geomean <= 5.9 us (from 6.2).",
    "refuted_if": "No single change or combination beats r0 by more than 0.1 us at 2.2 MB after 5 interleaved repeats, and the no-table probe is also within 0.1 us: the fixed cost lives in launch, first DRAM touch and tail, not in the load/store path; H12 refuted, H6 narrows to CTA count and row mapping.",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H13",
    "title": "Persistent table-stationary pipelined grid for M/L (CAKE mid-size program adapted to RoPE)",
    "operation": "new_design",
    "parents": ["r0-rope-ldg256-os-stel"],
    "band": "M",
    "instructions": "Fill the ldg256/persistent-wstat niche. Design: order the work s-tile-major: tile (stile, stream, bh) covers rows bh*S + stile*TS .. +TS-1 of that stream (TS = 32 rows for 256 threads, 8 lanes per row as in r0; also try TS = 16 with 128 threads). Grid = min(n_tiles, 148 * k) with k from the sm_100a occupancy of the compiled kernel (expect 2-4 at 256 threads). Each CTA takes a contiguous range of the tile list (balanced split, not strided), so its s-tile changes rarely; when it changes, reload the TS-row cos/sin slice into shared memory (TS KB; one 256-bit load per thread, __syncthreads). Software pipeline: hold two register sets for x (2 x 16 regs); issue the next tile's two 256-bit x loads before computing and storing the current tile; evict_last stores as r0. Check regs <= 80 and no spills on the sm_100a compile; if the pipeline pushes regs past 96, drop to 128-thread CTAs. Correctness masks for s >= S. Measure on the bench against r0 and against E1's one-shot variant if available, interleaved, at 36, 61, 102, 134, 269, 407, 2689 MB; try k in {2,3,4} and TS in {16,32}; run_tests 16/16. Dispatch: use this program only where it wins on the bench (expect M, 61-269 MB); keep r0's paths elsewhere. If it wins at M by >= 3% on the bench, submit to the portal. Record in findings the per-size times for each (k, TS), the register count, and whether the s-tile-major ordering changed the DRAM pattern cost (compare with a bh-major ordering probe).",
    "success": "Bench: M-band geomean >= 5% below r0 (22.5 -> <= 21.4 us) with no L size slower than r0 by more than 2%. Portal: M-band geomean <= 24.6 us (from 25.9).",
    "refuted_if": "Every (k, TS) configuration is slower than r0 at 61-269 MB on the bench by >= 2%, or ties while E1's one-shot table-sharing variant wins: the persistent structure adds serial latency that the table saving does not repay (as CAKE found at large sizes); park H13 and keep one-shot grids.",
    "model": "opus"
  }
]
```