### Assessment

r1-rope-v4nc-s128 scores 0.6772. The portal fit is about 4.8 µs fixed plus 6.82 TB/s marginal, against 5.3 µs plus 7.0 TB/s on the bench. The portal penalty is 0.5 µs at ≤ 16 MB, 1.5–2.5 µs at 36–320 MB and 2.5% at 2689 MB. The (64,8,8,128) workload, whose tables are L1-resident, carries the same +10% penalty as 61 and 269 MB, so table L2 traffic does not explain the M-band gap. The r0-to-r1 change gained 16.7% at L on the portal but 6.7% on the bench, so the portal is more sensitive to per-SM latency tolerance than the bench. The 2.2 MB workload is at the bench copy floor (2.59 vs 2.64 µs), so its 3.2 µs portal time is harness regime, not design.

Remaining score, by the model: 10% per band is worth S +0.010, M +0.012, L +0.015. A realistic round (S −4, M −5, L −3) gives about 0.691.

This round tests three mechanisms. E1 doubles x bytes per thread while halving table traffic (two heads per thread sharing cos/sin in the 16-lane layout, ≤ 56 regs), which also cuts CTA count 4× at 6.6–57 MB; its portal delta versus bench delta is the direct test of H11 and the new H15. E2 retests H13 cleanly: a register-pipelined persistent grid with no shared memory, the form CAKE used, since both r1 persistent attempts were confounded by shared-memory tables. E3 is a bounded L-band ceiling probe: if no memory path beats 7.0 TB/s with evict_last stores on the bench, L is closed and effort moves to M.

```yaml ledger
# Hypothesis ledger for #88 (RoPE application), seeded 10 October 2026 by the r0 research session.
# r1 update 10 October 2026 after the first portal result (r0-rope-ldg256-os-stel, 0.6351).
# r2 update 10 October 2026 after the second portal result (r1-rope-v4nc-s128, 0.6772).
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
- 'Clock locking on the rented card (nvidia-smi -lgc/-lmc) is a probe tool only; never change clocks or any device state from a solution. (r1: the rented card refuses -lgc anyway.)'
- 'Any kernel that pairs rows (heads h, h+1 or rows s, s+1) must dispatch to a one-row path when the pairing does not divide the shape; every candidate stays correct for every shape.'
hypotheses:
- id: H1
  statement: evict_last on the output stores cuts in-window dirty write-backs, as in #38, so M sizes gain most.
  status: supported
  evidence:
  - 'probe r0: EL vs plain stores, same kernel: 36 MB 7.11 vs 8.02 us, 61 MB 10.49 vs 11.48, 269 MB 39.05 vs 42.10, 407 MB 56.73 vs 61.93 [probe_b200]'
  - 'run_tests r0: 0.83-0.88x the plain copy at 61-269 MB'
  - 'r1 (v4 layout): plain stores +3% at 15.8 MB, +17% at 36 MB; at 2.2 MB no difference [probe_b200]'
- id: H2
  statement: CTA dispatch rate (about 0.55-0.65 us per 1000 CTAs regardless of 64 or 256 threads) caps 64-thread one-shot grids at L, so 256-thread CTAs win there while 64-thread CTAs win at S/M.
  status: refuted
  evidence:
  - 'probe r0: empty grids 64x32768 18.0 us, 64x49152 26.4 us, 256x8192 5.3 us, 256x12288 7.4 us, 256x81920 43.5 us'
  - 'probe r0: 64-thr vs 256-thr: 36 MB 6.94 vs 7.11, 269 MB 38.77 vs 39.05, 407 MB 59.75 vs 56.73 (one session)'
  - 'r1 (3 sessions): r0 with 64-thread CTAs at every size is -6..-8% at 322-407 MB and -2..-3% at 612-2689 vs the 256-thread path; 49k-327k 64-thread CTAs are not dispatch-bound; with the v4 layout 128 and 256 threads tie within 0.3% at L [probe_b200]'
  - 'dispatch rate 1800 CTAs/us x 8 KB = 15 TB/s of reads: it caps nothing above 4-row CTAs in steady state; it only sets the single-wave fill time at S (H6)'
- id: H3
  statement: Sharing one cos/sin load across several heads (rows S*512 B apart) cuts L2 traffic and helps L, most at S=8192 (612 MB) where the 16 MB of tables live in L2.
  status: open
  evidence:
  - 'probe r0 (8-lane layout, 80 regs): 2-head variant 269 MB 38.39 vs 39.05 (noise), 407 MB 60.96 vs 56.73'
  - 'r1: L1-hit table probe (s0) vs 64-thread r0: -0.5..-1.6% at 269-2689 MB but -5.5% at 612 MB (S=8192) [probe_b200]'
  - 'r1 variants A/C/W (8-lane layout, 64-80 regs): all +2..+4% vs 64-thread r0 at L on the bench; the register cost of the 8-lane layout confounds the test. E1 retests in the 16-lane layout at <= 56 regs.'
- id: H4
  statement: Occupancy (resident threads per SM) limits bandwidth; the 8-lane 256-bit layout at 50-60 regs loses to the 16-lane 128-bit layout at 32 regs partly through occupancy.
  status: supported
  evidence:
  - 'probe r0: float4 layout with 32 regs ties the 60-reg v8 layout at 36 and 269 MB and is 5% slower at 407 MB (one session; contradicted by r1)'
  - 'r1: v4 16-lane layout (32 regs) vs r0 (50 regs): -9/-6/-6% bench, -10/-10/-17% portal at S/M/L [run_tests, portal]'
  - 'r1: the no-table probe gains -10% at 36-61 MB but the L1-hit table probe (same 50 regs) only -2.5..-5%: the rest is occupancy (30 vs 50 regs) [probe_b200]'
  - 'r1: per-thread depth does not replace resident threads: persistent D=3/D=4 at 104-138 regs +5..+30% [probe_b200]'
- id: H5
  statement: Two rows per thread raise bytes in flight per SM and help M/L once the register cost is <= 56 regs (16-lane layout, both rows sharing one table load), i.e. the r0 result was a register-cost artefact of the 8-lane layout.
  status: open
  evidence:
  - 'probe r0: 2 rows/thread at 256 threads in the 8-lane layout (108 regs): +9% at 36 MB, +15% at 407 MB'
  - 'r2 lead: in the 16-lane layout 2 rows sharing a table need 16 x + 16 table + 8 output regs, about 48 regs: 5 CTAs/SM at 256 threads = 80 KB of x in flight vs 64 KB for r1. E1 tests it.'
- id: H6
  statement: At 6.6-57 MB (2k-7k CTAs, one to three waves) the single-wave fill time (0.55-0.7 us per 1000 CTAs) and the per-CTA replacement gap are the fixed cost; 4x fewer CTAs (32 rows per 256-thread CTA, all loads issued up front) cut about 0.5-1 us there. At 2.2 MB the kernel is already at the copy floor.
  status: open
  evidence:
  - 'probe r0: 64/128/256-thread one-row-per-thread variants are within 0.15 us of each other at 2.2 MB (2.62-2.77 us)'
  - 'r1: at 2.2 MB v4 2.59 us vs plain copy 2.64 and v4 copy 2.56; no-table probe 2.69: the kernel is at the copy floor [probe_b200]'
  - 'portal r1: (1,8,8,128) 3.2 us vs Tb 2.9 (0.470); portal/bench 1.23 at 2.2 MB but +13..+22% at 6.6-57 MB (3.9/5.2/8.0/11.0 vs 3.46/4.61/6.63/8.99): the 6.6-57 MB workloads are where S-band design can still pay.'
- id: H7
  statement: Load hints on x (.nc, L1::no_allocate) and on cos/sin (.nc) change nothing measurable; plain loads would do.
  status: refuted
  evidence:
  - 'r1: at 2.2 MB plain vs nc/no_allocate x loads, plain vs nc table loads, plain vs cache_hint stores: all within 0.03 us [probe_b200, 5 interleaved repeats]'
  - 'r1: nc on x is -7% at 57 MB and -1% at 102 MB (8.99 vs 9.69 us); nc on the tables alone changes nothing; at 134-269 MB plain is 1-3% faster than nc [probe_b200]'
  - 'r1: table loads with L2::evict_last: -2.7% at 612 MB, -0.9% at 2689, neutral or +0.7% elsewhere [probe_b200]'
- id: H8
  statement: The runtime modulo rl % S (about 20-30 instructions of reciprocal division in the prologue before any load issues) costs a measurable fraction of the per-CTA replacement gap at 1.5 GHz; a host-computed magic divisor (mulhi + shift) or a power-of-two fast path recovers it at S/M.
  status: open
  evidence:
  - 'probe r0: the 2-head variant uses a modulo-free grid and was not faster (unlocked clocks, 80 regs: confounded)'
  - 'r2 lead: 8 of 16 workloads have power-of-two S; the modulo runs once per thread per row. Effect expected <= 2%; test as a knob inside E1/E2, not as its own slot.'
- id: H9
  statement: Partial or first-output-only evict_last is no better than evict_last on all stores (as #38 H15), since the gain saturates at about 25-30 MB of parked write-back.
  status: parked
  evidence:
  - 'r2 lead: worth <= 1% by the #38 result; not a slot.'
- id: H10
  statement: SOLAR for this problem is bytes/8 TB/s including cos/sin, so scores at the floor would be about 0.75-0.8 as in #38.
  status: refuted
  evidence:
  - 'portal r0: Tsol back-solved from (t, Tb, S) is about 0.53 x bytes/8 TB/s at >= 134 MB (2689 MB: ~177 us vs floor8 336; 269 MB: ~18 vs 33.7; 134 MB: ~9.2 vs 16.8), the same 0.53x as #38. At S it is 0.6-0.9x floor8 but ill-conditioned (0.1 us rounding).'
  - 'consequence: per-workload ceiling at L is about 0.68 at 7 TB/s and 0.74 at 8 TB/s; r1 sits at 0.67-0.68 there; at 2689 MB 7.05 TB/s would give 0.687 vs 0.669.'
- id: H11
  statement: At the locked 1.5 GHz SM/L2 clock, the cos/sin table reads from L2 (2 B per byte of x) are an M/L limiter on the portal; cutting table traffic 2-4x recovers several % at L (most at S >= 4096) but not the M-band portal gap.
  status: open
  evidence:
  - 'portal r1: L band marginal 6.82 TB/s (fit on 612 and 2689 MB) vs #38 r3 at 7.03 TB/s on the same portal with 1 B of weight reads per byte of x; 612 MB (S=8192) runs at 6.47 TB/s vs 6.74 at 2689 MB (S=4096).'
  - 'portal r1 vs bench: +10.4% at (64,8,8,128), whose 128 KB of tables are L1-resident, the same as +15% at 61 MB and +6% at 269 MB: table L2 traffic does not explain the M-band portal penalty. The claimed 8-15% at M is withdrawn; L stays open.'
  - 'r1: the bench L1-hit table probe gains -5.5% at 612 MB, -0.5..-1.6% elsewhere at L; bench clocks cannot be locked, so only a portal slot can show the locked-L2 effect [probe_b200]'
  - 'r1: s-tile x head-group one-shot CTAs (variant A) tie at 2689 MB on the bench: a design that is bench-neutral and halves table traffic is the right portal probe (E1).'
- id: H12
  statement: The 0.7 us gap at 2.2 MB between r0 (3.6 us) and the baseline (2.9 us) comes from the load/store path (ld.global.nc texture path, 256-bit accesses, the cache_hint store) rather than from CTA shape, since 64/128/256-thread shapes tie on the bench.
  status: supported
  evidence:
  - 'r1: 8-lane 256-bit -> 16-lane 128-bit: 2.98 -> 2.59 us at 2.2 MB on the bench; hints are not the cause (all within 0.03 us) [probe_b200]; portal 3.6 -> 3.2 us. The remaining 0.3 us to Tb at 2.2 MB is within tiny-kernel portal noise (CAKE: 5-9% launch-order effects) and is not a design lever.'
- id: H13
  statement: A register-pipelined persistent grid (grid = 148 x resident CTAs, each thread issues the next row''s x and table loads before the current row''s FMAs, no shared memory, no barriers) beats the one-shot grid at 6.6-269 MB by removing the per-CTA replacement gap and the single-wave fill, as CAKE''s mid-size program does on B200.
  status: open
  evidence:
  - 'literature: CAKE pipelined persistent program capped at 4-6 CTAs/SM beat our #38 one-shot by 0.6 us at 25 MB, tied at >= 400 MB [cake-pr5741]'
  - 'r1-rope-pers-tstat-m: table-stationary persistent with the table in shared memory (80 regs, 3 CTAs/SM) +9..+20% at 57-269 MB; the first version with the table in registers (98 regs) likewise. Both are confounded by register cost, shared-memory carve-out (which strips L1 from the tables, see the r1 padding trap) and per-tile table stalls; neither tested the register-only pipeline. E2 does.'
- id: H14
  statement: The bench''s 7.0 TB/s marginal (r1, LDG.128 one-shot, evict_last stores) is the B200 read+write ceiling for this traffic pattern; no memory path (1-D bulk or TMA rings at 32-96 KB per CTA, 256-bit accesses, interleaved Q/K CTA order, STG.256) exceeds it by >= 3% at 612-2689 MB.
  status: open
  evidence:
  - 'literature: public B200 streams fit 7.05-7.10 TB/s marginal (CAKE, LDG.128); no published TMA stream number [b200_sota]'
  - 'r0: plain copy 413 us at 2689 MB (6.5 TB/s, plain stores) vs r0 kernel 400 us: evict_last stores beat the naive copy, so the ceiling must be probed with evict_last stores [probe_b200]'
- id: H15
  statement: The portal penalty versus the bench (about +0.5 us at <= 16 MB, +1.5-2.5 us at 36-320 MB, +2.5% at 2689 MB) is latency starvation at the locked clock (longer L2/XBAR latency in ns, slower CTA replacement), so designs with more x bytes in flight per SM and fewer CTA replacements lose less on the portal than on the bench; the portal delta of such a design exceeds its bench delta.
  status: open
  evidence:
  - 'portal vs bench, r0 -> r1 (50 regs/256-bit -> 32 regs/128-bit, 2x resident threads): S +10.0% bench / +10.5% portal; M +6.8 / +10.3; L +6.7 / +16.7 [lab notebook]'
  - 'portal r1 per workload vs bench: 2.2 MB +23%, 6.6 +13%, 15.8 +13%, 36 +21%, 57 +22%, 134 +10%, 269 +6%, 612 +2%, 2689 +2.5%'
  - 'test: E1 (2 rows per thread, 80-96 KB of x in flight per SM) and E2 (pipelined persistent) submitted after the bench; H15 holds if the portal gain exceeds the bench gain by >= 3 points at M.'
```

```json tasks
[{"id": "E1", "hypothesis": "H5", "title": "Two heads per thread sharing one cos/sin load, 16-lane layout, <= 56 regs, one-shot; bench sweep then portal slot (tests H3, H5, H6, H11, H15)", "operation": "structural_mutation", "parents": ["r1-rope-v4nc-s128"], "band": "all", "instructions": "1. Start from r1-rope-v4nc-s128 (16 lanes per row, lane j owns columns 4j..4j+3 and 64+4j..). Add a kernel rope_pair<THREADS, HG> where each thread handles HG rows at the same sequence position s from HG consecutive heads of the same batch element: pair index p over (b, hp, s) with hp in [0, H/HG); s = p % S; row0 = HG*(p - s) + s (for HG=2: row0 = 2p - s), rows row0 + i*S for i < HG. Load the four table vectors once (cl, ch, sl, sh), issue all 2*HG x loads (ld.global.nc.L1::no_allocate.v4.f32) and the 4 table loads before any FMA, then compute and store each row with the evict_last cache_hint stores. Q pairs first, then K pairs (separate Rq/HG, Rk/HG counts; the modulo uses the in-stream pair index). Dispatch to r1's one-row kernel whenever H % HG != 0 or Hkv % HG != 0, and keep that path for the smallest workload (< ~4k rows) where 64 CTAs would underfill 148 SMs. 2. Compile for sm_100a: target <= 56 regs for HG=2 (use __launch_bounds__(256,4) or (128,8)); report regs, spills and SASS counts; no shared memory. 3. Bench, interleaved with r1 as the control, 3 repeats per size, all 16 workloads: HG=2 at 128 and 256 threads; HG=4 at 128 threads if it fits in <= 72 regs without spills. Record the per-size winner and the no-table twin of the best (tables replaced by constants) to measure how much of any gain is table traffic vs bytes in flight. 4. Knob: replace rl % S by a host-computed magic divisor (mulhi + shift, passed as two kernel args) in both the pair kernel and r1's path; report its effect at 6.6-57 MB (H8). 5. Build the dispatcher (per-band thresholds from step 3), run_tests, and if the bench geomean is within +1% of r1 at every band submit it: the portal delta vs bench delta at M and at 612 MB is the H11/H15 reading.", "success": "Bench: >= 3% faster than r1 at 36-134 MB or >= 3% at 612 MB with no band more than 1% slower; portal: score > 0.682, or the portal gain at M/L exceeds the bench gain by >= 3 points (H15 supported).", "refuted_if": "HG=2 at <= 56 regs is within +-1% of r1 at every size on the bench and the portal delta matches the bench delta: then more bytes in flight per thread and halved table traffic buy nothing, H5 and H3 are refuted in the 16-lane layout, and H15 loses its main support.", "model": "opus"},
 {"id": "E2", "hypothesis": "H13", "title": "Register-pipelined persistent grid (CAKE-style) for 6.6-269 MB: no shared memory, next row's loads issued before the current row's FMAs", "operation": "structural_mutation", "parents": ["r1-rope-v4nc-s128"], "band": "M", "instructions": "1. Keep r1's 16-lane layout and per-row loads. Write rope_persist<THREADS>: grid = 148 * k (k = resident CTAs per SM read from cudaOccupancyMaxActiveBlocksPerMultiprocessor at module init, never from another chip), each 16-lane group walks rows r = g, g + G, g + 2G, ... (G = total 16-lane groups in the grid). Software pipeline depth 1 in registers: before computing row i, issue the x loads (2 x LDX4) and table loads (4 x LD4) of row i+1 into a second register set; compute and store row i with evict_last stores; swap. Two register sets = 48 live floats + addressing: target <= 64 regs, 0 spills, 0 shared memory (any shared memory strips L1 from the tables: r1 padding trap). No __syncthreads. 2. Row order knob: (a) linear (consecutive groups take consecutive rows, table re-read per row from L1/L2), (b) s-stationary (a group keeps s fixed and steps through heads: rows S apart, table loaded once per s, so only x is prefetched and the live set is 16 table + 2 x 8 x regs; needs the modulo only once per group). 3. Bench vs r1 interleaved, 3 repeats, sizes 6.6, 15.8, 36, 57, 61, 102, 134, 269 MB, plus 612 and 2689 to check L is not hurt; sweep k = occupancy, occupancy-1, and THREADS 128/256. Report the per-CTA replacement gap directly: r1's time minus the persistent time at the same bytes, and an empty persistent grid span. 4. If any configuration wins >= 3% at 36-134 MB, build the dispatcher (persistent for the winning sizes, r1 elsewhere), run_tests, and submit; the portal delta vs bench delta is a second reading of H15.", "success": "Bench >= 3% faster than r1 at 36-134 MB with L unchanged (within 1%); portal score > 0.682.", "refuted_if": "Every register-only pipelined configuration at <= 64 regs and k = occupancy is within +-1% of r1 or slower at 36-134 MB on the bench: H13 is refuted for this problem (the CAKE result does not transfer to 128-float rows with a table), and persistent designs are parked.", "model": "opus"},
 {"id": "E3", "hypothesis": "H14", "title": "L-band ceiling probe: fastest read+write stream on the bench at 612 and 2689 MB across memory paths, with evict_last stores", "operation": "new_design", "parents": ["r1-rope-v4nc-s128"], "band": "L", "instructions": "1. Write pure copy kernels (read Q/K rows, write the outputs, no tables, no maths) that share r1's harness binding, so each can be timed on the (1,64,8,8192) and (32,16,4,4096) workloads exactly like a solution. All use evict_last cache_hint stores unless stated. Variants: (a) r1's exact load/store code with the FMAs removed (the control, LDG.128 x 16 lanes, 128 and 256 threads); (b) LDG.256 / STG.256 with 8 lanes per row (ld.global.nc.L1::no_allocate.v8.f32); (c) 1-D cp.async.bulk ring: persistent grid, 32 KB, 64 KB and 96 KB per CTA in flight (dynamic shared memory, mbarrier expect_tx, 2-4 stages), stores either direct from registers or cp.async.bulk shared-to-global with the evict_last policy on the copy; (d) 2-D TMA (cuTensorMapEncodeTiled on the host each call) with the same ring sizes; (e) variant (a) with Q and K CTAs interleaved (even blockIdx -> Q, odd -> K) and with the grid reversed; (f) variant (a) with plain stores and with STG.256 stores only. 2. Time each at 612 and 2689 MB interleaved with (a) as control, 3 repeats; report TB/s = bytes / t and the marginal slope between the two sizes; also time (c)/(d) at 36 and 134 MB to record their fixed cost. 3. If any variant beats (a) by >= 3% at both L sizes, add the RoPE maths and the tables to it (table rows via LD4 from L2 as in r1, or bulk-copied once per s-tile if the ring is s-tile x head-group ordered), make it the > 262144-row path of r1's dispatcher, run_tests all 16, and submit if the bench L geomean improves >= 3% with S/M unchanged. If nothing beats (a), record the ceiling and the fixed cost of each path as findings and stop.", "success": "A path reaches >= 7.2 TB/s at 2689 MB on the bench (>= 3% over r1's copy control) and the RoPE kernel built on it is >= 3% faster at L in run_tests; portal score > 0.682.", "refuted_if": "No path exceeds the LDG.128 control by 3% at either L size: H14 is supported, the L band is at the bench ceiling, and all later rounds target S/M and the portal-vs-bench gap only.", "model": "sonnet"}]
```