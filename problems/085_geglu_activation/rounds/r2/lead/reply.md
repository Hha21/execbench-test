### Assessment

Score 0.6755. Per-workload scores: S 0.79–0.82, the two 63 MB M workloads 0.86, the 126–133 MB M workloads 0.57–0.68, L 0.52–0.55 (Tb only 5–10% above us). Worth of 10%: S +0.007, M +0.011, L +0.020.

What r1 settled: the whole dirty-flush penalty is write-back bytes (ncu), eviction-driven (about 46% of our read allocations evict a dirty line, matching a half-dirty L2). In-window writes fit `outputs + 64 MB flush − ~80 MB retention`, so M/L bytes are fixed by hardware; EF loads trim M only. Bytes in flight do not cap reads (fat threads, 192 KB rings: no gain), yet pure reads reach only 6.2 TB/s while the 2:1 mix reaches 7.0 CLEAN. S variants are all within noise; the S cost is launch/exit, one DRAM round trip and dirty-eviction latency.

Remaining levers are speculative, so r2 is diagnostic and exploratory: (E1) find what caps reads and whether any legitimate access pattern or CTA ordering lifts the 2:1 stream above 7.0 TB/s at L; (E2) probe whether the two-die L2/HBM topology gives an exploitable locality (far-partition latency, scheduler filling one die first at S); (E3) measure the SM-clock dependence directly with ncu base-clock runs, because the portal's extra time is +1.0 µs at 63 MB but +0.5 µs at S and L, which a constant launch cost does not explain; if part is per-byte, instruction-lean variants gain on the portal. The constraint mentioning "H16" is inherited from the #84 ledger; new hypotheses here are H14–H18.

```yaml ledger
# Hypothesis ledger for #85 geglu_activation. Seeded 10 October 2026 by the research phase; r1 update 10 October 2026; r2 update 10 October 2026.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.

constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot (r13, H21).'
- 'L2 prefetch (cp.async.bulk.prefetch.L2) only on the current call''s own inputs, inside our kernel.'
- 'Bench methodology: every timing must reproduce timing.py (input copy to a shifted address, output zero, cudaCtxResetPersistingL2Cache, 252 MB zero_ flush) with interleaved repeats; without the reset, evict_last lines survive across calls and invert the H1 result (r1-geglu-efld-m).'
hypotheses:
- id: H1
  statement: evict_last on output stores cuts in-window write-backs of the flush's dirty L2 lines, so M and L sizes gain 6-13%.
  status: supported
  evidence:
  - 'probe r0 (rented B200): EL vs plain stores: 63 MB 10.98 vs 11.97 us, 126 MB 18.74 vs 21.64, 252 MB 35.35 vs 40.00, 503 MB 71.55 vs 76.07; S unchanged at 128 rows'
  - 'r1-geglu-s-lean32: plain stores +0.3 us (~5%) at 422 and 512 rows, so EL also pays at the upper S sizes'
  - 'r1 ncu: mechanism is bytes: EL retains ~80 MB of our outputs dirty at kernel end; plain/EF stores are +5..+15% at every M/L size under the harness sequence'
  - 'portal r0 (0.6742) and r1 (0.6755): M/L times equal rented times + 0.5-1.0 us, so the EL gain carries to the portal; no portal pair without EL'
- id: H2
  statement: One 8-float chunk per thread in small one-shot CTAs is faster than fatter threads; the one-shot grid needs no persistence.
  status: supported
  evidence:
  - 'probe r0: 2 chunks/thread +1..5%, 4 chunks/thread +3..10% (82 regs); block 128/256/512 within 1%'
  - 'r1 (bulkring session): fat one-shot threads with all loads first and __launch_bounds__(128,8) tie r0 within 1% at 252/503 MB; persistent register-prefetch 148x8 is 4-8% slower; bulk rings 7-10% slower at every depth'
  - 'r1-geglu-s-lean32: 64-thread CTAs, 512-thread 1-D, 148x4 fat CTAs: all within noise or worse at S'
- id: H3
  statement: The sigmoid form (a*b/(1+exp(-2u)), __expf + __fdividef) is both within tolerance and faster than tanhf.
  status: supported
  evidence:
  - 'probe r0: max abs err 1.9e-6, zero elements out of tolerance at 128/2048 rows; tanhf variant 3-10% slower'
  - 'portal r0, r1: 16/16 pass'
- id: H4
  statement: The 2-D grid (5 CTAs x rows, no integer division, no bounds checks) beats the 1-D grid at S by about 2% and ties at L.
  status: supported
  evidence:
  - 'probe r0: 7.9 MB 3.73 vs 3.81 us, 31 MB 6.79 vs 6.94, 126 MB 18.66 vs 18.69, 503 MB 71.56 vs 71.60'
- id: H5
  statement: At the portal's locked 1500 MHz the GELU maths (2 MUFU per element, ~37% of MUFU capacity) becomes visible; a 1-MUFU formulation would gain 1-3% at M/L.
  status: refuted
  evidence:
  - 'probe r0 (1965 MHz): copy-only kernel (no GELU) times equal to the full kernel at every size'
  - 'portal r0 vs rented: 503 MB 72.2 vs 71.7 us (+0.7%), 252 MB 36.0 vs 35.1, 126 MB 19.5 vs 18.6, 7.9 MB 4.1 vs 3.6: the portal adds 0.5-1.0 us, not a per-byte cost, so compute is hidden at 1500 MHz'
  - 'caveat: the inference rests on two portal points per size with 5% portal noise; r2 E3 measures the clock dependence directly at base clock (H15)'
- id: H6
  statement: The S-band fixed cost (~2.7 us above the byte floor at 7.9 MB on the rented B200, ~3.1 us on the portal) can be reduced by a dedicated small-size path.
  status: parked
  evidence:
  - 'probe r0: empty kernel 1.73 us; read-only 3.52 us; full 3.6-3.8 us at 7.9 MB'
  - 'portal r0/r1: 4.1-4.2 us at 7.9 MB, 6.8-6.9 at 26 MB, 7.7 at 31 MB; Tsol ~0.9 us; S scores 0.79-0.82'
  - 'r1-geglu-s-lean32 ladder at 7.9 MB: launch/exit ~1.5 (empty kernel 1.46 regardless of CTA count), first DRAM round trip ~0.5, CTA ramp ~0.3, streaming 5.2 MB ~1.1-1.4 (2x the byte floor: dirty evictions), compute+store drain ~0.35'
  - 'r1-geglu-s-lean32: 64-thread CTAs, 512-thread 1-D, 148x4 fat CTAs, L2 prefetch in 16/20/64 KB pieces, 32-bit math: all within noise (medians of 9 interleaved repeats); run_tests S +0.1%'
  - 'r1-geglu-efld-m: DIRTY-CLEAN gap 0.8 us at 7.9 MB from only 2-5 MB of flush write-back; EF loads do not shrink it; no legitimate in-kernel policy avoids the flush lines'
  - 'parked: every in-kernel S lever is exhausted; reopen only if E2 finds a die-placement effect at S or E3 finds a clock-scaled component we can cut'
- id: H7
  statement: 'The harness''s dirty L2 (zero-fill flush) costs reads: a read-only kernel reaches only 5.3 TB/s at 503 MB while read+write reaches 7.0 TB/s.'
  status: supported
  evidence:
  - 'probe r0: read-only 62.95 us vs full 71.6 us at 503 MB (335 MB read)'
  - 'r1 ncu: in-window DRAM reads always equal the input bytes; the whole dirty penalty is writes. Read-only DIRTY write-back 3.3/7.1/14.4/30.3/58.4/63.7 MB at 7.9/31/63/126/252/503 MB, i.e. ~46% of read allocations evict a dirty line (L2 is ~64/126 dirty after the flush) until the 64 MB of flush lines are gone'
  - 'r1: DIRTY vs CLEAN for r0: 3.87/3.04, 6.62/5.82, 10.91/8.99, 18.72/15.87, 35.17/31.26, 71.52/62.37 us (13-27% at every size)'
  - 'mechanism settled: H11 (write-backs), not H12 (bytes in flight)'
- id: H8
  statement: Load hints (.nc, evict_first, L1::no_allocate) do not matter for this once-read stream.
  status: refuted
  evidence:
  - 'probe r0: ld.global.nc vs plain ld.global: identical at every size'
  - 'r1-geglu-efld-m: L2::evict_first on loads is -3.6..-6.9% at 31-133 MB (H11), +0.9..-1.8% at S, +1.3..-0.4% at L; .nc and L1::no_allocate make no difference'
- id: H9
  statement: Tsol for this problem sits below the bytes/8 TB/s floor, so score 1.0 is unreachable and only relative speed-ups matter.
  status: supported
  evidence:
  - 'portal r0 derived Tsol (from S, t, Tb): 0.9 us at 7.9 MB (floor8 0.98), 2.1 at 26 MB (3.24), 2.5 at 31 MB (3.93), 4.5 at 63 MB (7.86), 29 at 441 MB (55.2), 33 at 503 MB (62.9): 0.52-0.92x the byte floor'
  - 'score derivative: ~0.05 per us at 7.9 MB, ~0.0064 per us at 503 MB (per workload); 10% at S/M/L = +0.007/+0.011/+0.020'
- id: H10
  statement: The portal (1500 MHz) adds a constant ~0.5-0.9 us to every workload relative to the rented B200 (1965 MHz); this is SM-clock-bound fixed cost (launch, dispatch, address chain, exit), not bandwidth.
  status: supported
  evidence:
  - 'portal vs rented r0: +0.5 us at 7.9 MB, +0.7 at 31 MB, +1.0 at 63 MB, +0.9 at 126 MB, +0.9 at 252 MB, +0.5 at 503 MB; r1: +0.7 at 7.9 MB, +0.5 at 63 MB, +0.6 at 126 MB, +0.9 at 252 MB, +0.7 at 503 MB'
  - 'the empty-kernel span 1.46 us x 1965/1500 = 1.9 us predicts +0.45 us, matching S and L; the M offsets of +0.9-1.0 us are larger than a pure launch/exit scaling explains (portal noise 5% = 0.5-1 us at M, so unresolved); E3 tests H15'
- id: H11
  statement: evict_first on the input loads (createpolicy + ld.global.L2::cache_hint) makes later reads evict our own clean read lines instead of the flush's dirty lines, removing in-window write-backs and lifting the 5.4 TB/s read cap at M/L.
  status: supported
  evidence:
  - 'r1-geglu-efld-m ncu: in-window writes r0 vs EF loads (DIRTY) 1.9/4.6, 9.5/7.2, 20.2/10.8, 34.9/24.4, 69.1/65.4, 155.5/149.1 MB at 7.9/31/63/126/252/503 MB; read-only kernel: EF loads cut flush write-back 63.7 -> 12.1 MB at 503 MB (-11% time)'
  - 'full kernel: M -5.6% on the bench (1,1024 -7.5%, 2,512 -7.2%, 8,256 -6.8%, 1,2048 -3.5%, 4,541 -3.1%), portal -2.2% (0.6755 vs 0.6742); L +0.3..1.3% so L keeps default loads (dispatch at 3072 rows)'
  - 'traffic model: in-window writes ~= max(0, outputs + ~64 MB flush-dirty - ~80 MB dirty retention); CLEAN retention plateaus at 77-83 MB; at L the bytes are fixed by this cap, so no policy headroom remains'
- id: H12
  statement: The read cap is a bytes-in-flight limit at the read side (one-shot CTAs ramping/draining), and a persistent 1-D bulk ring with >=128 KB in flight per SM and few instructions lifts L by >=3%.
  status: refuted
  evidence:
  - 'r1-geglu-bulkring-s3k8-l: rings with 64-192 KB/SM in flight are 7-10% slower at 126/252/503 MB (full kernel) and 2-3 us slower read-only; fat threads (2-4 chunks, all loads first) tie; persistent register prefetch 4-8% slower'
  - 'EF policy on bulk copies: no gain. Blocked tile assignment worse than strided'
- id: H13
  statement: 'Tb is erratic per workload (same 126 MB: 31.2 vs 22.6 us; 63 MB Tb 45.3 > 126 MB Tb 22.6) and only 5-10% above us at L, so L scores stay near 0.52-0.55 unless L gains >=10%; S workloads already score ~0.80.'
  status: supported
  evidence:
  - 'portal r0 page: Tb 13.4/13.4/22.5/25.9/26.0 (S), 45.3/45.3/31.2/22.6/23.6 (M), 40.6/40.8/40.8/47.5/67.5/75.8 (L); r1 page identical (stored constant)'
  - 'r1 per-workload scores: S 0.79-0.82, 63 MB 0.855, 126-133 MB 0.57-0.68, L 0.52-0.55'
- id: H14
  statement: 'Pure reads cap at ~6.2 TB/s (335 MB in 56 us with EF loads, 12 MB write-back) while the 2:1 read+write mix reaches 7.0 TB/s CLEAN; the cap sits on the L2/fabric/DRAM side (request rate or page policy), not in SM bytes in flight, and a different access order (CTA-to-address swizzle, half interleaving, TMA prefetch-to-L2 feeding plain loads) can lift the L stream by >=3%.'
  status: open
  evidence:
  - 'r1: read-only 503 MB: r0 63.5 us DIRTY, 56.0 us with EF loads; rings and fat threads do not help (H12 refuted)'
  - 'r1: r0 CLEAN full kernel 436 MB in 62.4 us = 7.0 TB/s, equal to the public 1:1 ceiling; STREAM triad (2:1) 7.48 TB/s is claimed in arXiv 2512.02189 v1, so 2:1 may allow more'
  - 'r2 E1 tests it'
- id: H15
  statement: 'Part of the portal''s extra time is per-byte, not fixed: the L2/crossbar clock domain follows the SM clock lock (1500 vs 1965 MHz), so L2 allocation, eviction and write-back run 24% slower on the portal; variants with fewer L2 transactions or instructions per byte then gain on the portal but show nothing on the rented box.'
  status: open
  evidence:
  - 'portal minus rented is +0.9-1.0 us at 63-252 MB but +0.5 us at 7.9 and 503 MB (r0); r1 +0.5..+0.9 us; a fixed launch/exit cost predicts a uniform +0.45 us; the excess at M is 0.4-0.5 us, within portal noise'
  - 'the portal gain of EF loads at M (-2.2%) was less than half the bench gain (-5.6%)'
  - 'ncu --clock-control base runs on the rented B200 (permitted, r1) can measure this directly; r2 E3'
- id: H16
  statement: 'The two L2 partitions (one per die) make latency and possibly bandwidth depend on whether a line''s home partition is on the SM''s die; if the address-to-partition mapping is coarse (>=1 KB granularity) and the SM''s die is readable (%smid), a kernel can assign tiles so each SM streams mostly from its own die, cutting latency at S and lifting bandwidth at L. Separately, the scheduler may fill one die''s SMs first for small grids, leaving S workloads on one die.'
  status: open
  evidence:
  - 'Chips and Cheese B200: L2 latency ~150 ns local, "dramatically" higher in the other die''s partition; 21 TB/s local vs 16.8 TB/s crossing; "perhaps the scheduler fills one partition''s SMs first"'
  - 'no measurement of the mapping granularity or the %smid-to-die layout; r2 E2 probes it'
- id: H17
  statement: 'The per-partition dirty retention (~80 MB of 126 MB) and the eviction choice are hardware policy; no load/store policy changes the in-window write bytes at L (model: outputs + 64 - 80 MB).'
  status: supported
  evidence:
  - 'r1 ncu: CLEAN retention 77-83 MB with EF loads at 252/503 MB; DIRTY writes 149-155 MB at 503 MB for every policy tried (EL, EL fractional 0.5, EF loads); plain/EF stores write more and are slower'
- id: H18
  statement: 'A row-ordered mixed store policy (evict_first for the first rows, evict_last for the last ~80 MB of outputs) smooths write-back timing and gains at 126-503 MB.'
  status: parked
  evidence:
  - 'by the H17 model the in-window write bytes are identical to all-EL (the flush lines are written back either way) and all-EL already streams the write-backs through LRU once the cap is hit; the fractional 0.5 variant lost 0-3.6% (r1). Not worth a session unless E1 shows write timing matters'
```

```json tasks
[{"id": "E1", "hypothesis": "H14", "title": "What caps reads at 6.2 TB/s, and does any legitimate access order lift the L stream?", "operation": "structural_mutation", "parents": ["r1-geglu-efld-m"], "band": "L", "instructions": "Start from r1-geglu-efld-m and its bench that reproduces timing.py (input copy to a shifted address, output zero, cudaCtxResetPersistingL2Cache, 252 MB zero_, interleaved repeats, CUPTI span). First check whether `nvidia-smi -lgc 1500,1500 -lmc 3996` is permitted on the rented box and report it. Step 1, read-only ceiling at 503 MB and 252 MB in CLEAN state (flush, then a read-only sum over another 252 MB) with ncu dram__bytes_read.sum.per_second, lts__t_sectors_op_read, l1tex__m_xbar2l1tex_read_bytes: (a) r0 read-only LDG.256 one-shot; (b) LDG.128 one-shot; (c) a kernel that only issues cp.async.bulk.prefetch.L2 over the whole input (no SM data return) to measure the DRAM-to-L2 fill rate; (d) one-shot CTAs that issue a cp.async.bulk 1-D copy of their 4 KB gate and 4 KB linear chunk into smem and exit after the mbarrier (no L1 path, no ring); (e) an L2-resident re-read of a 32 MB buffer to get the L2-to-SM ceiling. Report TB/s for each; if (c) exceeds 6.8 TB/s the cap is on the SM return path and (d)-style loads are the lead. Step 2, access order on the full kernel at 252/441/503 MB, 21 interleaved reps each, EL stores, default loads: (i) current grid (5, rows); (ii) 1-D grid with a tile swizzle so concurrently dispatched CTAs touch regions 1-4 MB apart (e.g. tile = (i % G) * (rows/G) + i / G for G in 8, 16, 64; 5120 = 2^10*5 so no division by 640 is needed if the swizzle is on the row index); (iii) reversed row order; (iv) CTAs that handle the gate chunk of row r and the linear chunk of row r+1 so each CTA's two loads are 60 KB apart instead of 20 KB; (v) each CTA loads gate+linear of two rows but stores nothing until both are in (already tied, include as a control). Step 3: take the best of step 2 and, if a bulk-to-smem load path won step 1, build a one-shot bulk1d variant (no ring, one 8 KB copy per CTA, consume from smem, EL 256-bit stores). Run run_tests on the best variant (all 16 workloads) and record every number in findings, including negative ones.", "success": "A variant beats r1-geglu-efld-m by >= 3% at 252, 441 and 503 MB (21 interleaved reps, medians) with 16/16 pass, or step 1 identifies the read cap's location with a TB/s figure that changes the design.", "refuted_if": "No read-only variant exceeds 6.5 TB/s CLEAN, prefetch-to-L2 also caps at ~6.2 TB/s, and no access order beats r1 by more than 1.5% at any L size: then H14 is refuted and L is at the hardware ceiling for this traffic.", "model": "opus"},
 {"id": "E2", "hypothesis": "H16", "title": "Two-die locality: is the L2 partition / HBM home of an address exploitable from %smid?", "operation": "new_design", "parents": ["r1-geglu-efld-m"], "band": "all", "instructions": "Probe first, build second. Probe 1 (dispatch): launch the r1 kernel's exact grids for 128, 422, 512 rows and 2048 rows with each CTA recording %smid into a scratch buffer; report the histogram of SMs used and the order in which SM ids fill (do small grids land on one die?). Probe 2 (latency map): a single-CTA pointer-chase kernel pinned by launching 148 CTAs that read %smid and only the CTA on a chosen SM (sweep SM 0, 37, 74, 111, 147) chases; chase within L2-resident lines (first touch the buffer with a read, then chase) over a 64 MB buffer with stride candidates 128 B, 1 KB, 4 KB, 64 KB, 1 MB, and record per-line latency for 4096 lines; look for a bimodal latency (~150 ns vs far) and the address-bit pattern that separates the modes. Probe 3 (bandwidth): a read-only stream over 256 MB executed only by CTAs on SMs 0-73 vs only SMs 74-147 vs all (the others exit at once), each at the same bytes; compare TB/s; then, if probe 2 found a coarse mapping, the same with the address set restricted to lines homed locally vs remotely. If and only if probe 2 shows a >= 20% latency split with a granularity >= 1 KB that a kernel can compute from the address, build a variant of r1-geglu-efld-m: each CTA reads %smid, derives its die, and takes its row (or 4 KB chunk) from a per-die work counter in a scratch buffer that the last CTA resets (one launch, deterministic kernel sequence, correct for every call, no data-dependent output), assigning local-homed chunks first. Measure under the harness-exact bench at 7.9/31/126/503 MB, 21 interleaved reps, and run run_tests. Record the mapping result in findings whatever the outcome; also report the probe-1 histogram as it bears on H6.", "success": "Probe 2 finds a computable local/remote split and the locality-aware variant is >= 2% faster at M or >= 3% at L, or >= 0.2 us at S, with 16/16 pass.", "refuted_if": "The latency split is < 10% or the home partition changes at < 1 KB granularity with no computable pattern, or small grids are already spread across both dies and the locality variant gains < 1%.", "model": "opus"},
 {"id": "E3", "hypothesis": "H15", "title": "Clock dependence at base clock: is the portal's extra time fixed or per-byte, and does an instruction-lean variant recover it?", "operation": "knob_mutation", "parents": ["r1-geglu-efld-m"], "band": "all", "instructions": "Use ncu on the rented B200 (permitted in r1 with --cache-control none). For r1-geglu-efld-m at 7.9/31/63/126/252/503 MB, run the harness-exact sequence (reset, 252 MB zero_, then the kernel) under ncu with --cache-control none and (a) --clock-control none, (b) --clock-control base. Collect gpu__time_duration.sum, sm__cycles_elapsed.avg, gpc__cycles_elapsed.avg.per_second, dram__cycles_elapsed.avg.per_second, dram__bytes_read.sum, dram__bytes_write.sum, lts__t_sectors.sum; 5 samples per cell, medians. Report the SM and DRAM clocks in each mode. Compute delta = t(base) - t(none) per size and fit delta = fixed + per_MB * traffic. If the DRAM clock also dropped under base, normalise by the achieved DRAM-clock ratio and state the assumption. Second, build two instruction-lean variants of the same structure and run them under the same two clock modes: (1) one-MUFU GEGLU: e = ex2.approx(k*(a + 0.044715 a^3)) with k = -2*sqrt(2/pi)*log2(e) folded into one FMA chain, r = 1/(1+e) by an integer-seed reciprocal plus two Newton FMA steps (no MUFU.RCP), out = a*b*r; verify max abs error < 3e-6 against the fp32 reference at 128 and 2048 rows; (2) same maths as r1 but the createpolicy hoisted so the kernel has the minimum instruction count before the first load and after the last store (check SASS: loads must be the first two memory instructions). Third, if nvidia-smi -lgc 1500,1500 is permitted, repeat the r1 vs variant (1) comparison at the lock with the harness-exact bench, 21 interleaved reps. Report per-size numbers, the fit, and run run_tests on variant (1).", "success": "At base clock the delta has a per-byte component >= 3% of t at 126-503 MB, and variant (1) or (2) recovers >= 2% of t at base clock while being within noise at unlocked clock: then it is a portal candidate (expected +0.005 to +0.01).", "refuted_if": "The base-clock delta is a constant offset within 0.15 us across sizes (per_MB consistent with zero) and the lean variants change base-clock time by < 1%: H15 refuted, H5 stays refuted, and clock-lean work stops.", "model": "sonnet"}]
```