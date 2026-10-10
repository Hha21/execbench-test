### Assessment

We are at 0.6827 and need +0.0057 (about +0.09 summed over 16 workloads). The score model says a 10% cut is worth +0.0096 at S, +0.0121 at M, +0.0154 at L, so uniform 1-3% gains on many workloads are the route, not a single workload. The 2.2 MB workload (0.487) is parked: its portal time equals a Triton-class baseline at the launch-plus-round-trip floor, and bench variants tie within 0.15 us (H12). The (64,8,8,128) workload (0.619) runs on the normal fixed-cost curve (fit: 2.9 us + bytes/7.4 TB/s up to 134 MB); its low score is a fast Tb, not a kernel defect.

What r2 taught us: more resident warps was amplified on the portal (r0 to r1, L +16.7% portal vs +6.7% bench) but more bytes per thread was not (r1 to r2, M -1.7% vs -2.2%), so H15 is now about warps or waves, not bytes in flight. The portal penalty jumps from +0.4..0.7 us at sizes under one wave (6.6-15.8 MB) to +1.5..2.0 us from 1.5 waves on (36 MB and up), then stays flat: that is the signature of an extra round trip per wave boundary at the locked clock, which a depth-2 pair kernel can test (E1, single wave at 36 MB, half the waves at 57-134 MB). At L the bench-verified free items are unused: table loads with evict_last (-2.7% at 612 MB), plain x loads at 134 MB and above (+1-3% faster than nc), and the pair path at S > 1536 whose two L workloads appear to have gained ~3% on the portal (E2). E3 covers the operator's small-shape request with cheap probes and the pair path's 124 KB L1 that cannot hold the 128-211 KB tables of the S = 128-211 workloads.

```yaml ledger
# Hypothesis ledger for #88 (RoPE application), seeded 10 October 2026 by the r0 research session.
# r1 update 10 October 2026 after the first portal result (r0-rope-ldg256-os-stel, 0.6351).
# r2 update 10 October 2026 after the second portal result (r1-rope-v4nc-s128, 0.6772).
# r3 update 10 October 2026 after the third portal result (r2-rope-pair2-magic-occ10, 0.6827; board s5 0.6884, s1 0.6978).
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only at <= 300 tokens (H16 of #38): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot (r13, H21).'
- 'Cache-policy qualifiers on our own loads and stores (evict_first/evict_last/no_allocate/.nc) are ordinary kernel properties (CAKE and quack use them) and are allowed; they are not "aimed at the flush". The line is: we may choose how our own lines are cached, never act on lines we do not own.'
- 'Clock locking on the rented card (nvidia-smi -lgc/-lmc) is a probe tool only; never change clocks or any device state from a solution. (r1: the rented card refuses -lgc anyway.)'
- 'Any kernel that pairs rows (heads h, h+1 or positions s, s+1) must dispatch to a one-row path when the pairing does not divide the shape; every candidate stays correct for every shape.'
- 'cudaFuncSetAttribute on our own kernel (carve-out preference, max dynamic smem) is ordinary occupancy control and allowed; it stays flagged in the design card for review.'
hypotheses:
- id: H1
  statement: evict_last on the output stores cuts in-window dirty write-backs, as in #38, so M sizes gain most.
  status: supported
  evidence:
  - 'probe r0: EL vs plain stores, same kernel: 36 MB 7.11 vs 8.02 us, 61 MB 10.49 vs 11.48, 269 MB 39.05 vs 42.10, 407 MB 56.73 vs 61.93 [probe_b200]'
  - 'r1 (v4 layout): plain stores +3% at 15.8 MB, +17% at 36 MB; at 2.2 MB no difference [probe_b200]'
  - 'r2 ceiling: plain stores +5% at 612 MB but -0.8% at 2689 MB: the parked write-back matters below about 1 GB and converges at 2689 MB [r2-rope-ceiling-v4]'
  - 'portal r2 fit: marginal 7.4 TB/s from 6.6 to 134 MB (parked outputs), 6.4-6.9 TB/s above 269 MB, 6.84 at the top: the portal shows the same parking effect.'
- id: H2
  statement: CTA dispatch rate caps 64-thread one-shot grids at L, so 256-thread CTAs win there while 64-thread CTAs win at S/M.
  status: refuted
  evidence:
  - 'r1 (3 sessions): 64-thread CTAs at every size are -6..-8% at 322-407 MB vs the 256-thread path; with the v4 layout 128 and 256 threads tie within 0.3% at L [probe_b200]'
  - 'r2: empty one-shot grids span 0.5 us per 1000 CTAs (1.6 us at 6.6 MB, 39 us at 612 MB) but dispatch overlaps memory work; r1 at 269 MB is not dispatch-bound [r2-rope-psst-m]'
- id: H3
  statement: Sharing one cos/sin load across several heads (rows S*512 B apart) cuts L2 table traffic and helps L, most at S >= 4096 where the 4-16 MB tables live only in L2.
  status: open
  evidence:
  - 'r1: L1-hit table probe vs 64-thread r0: -0.5..-1.6% at 269-2689 MB but -5.5% at 612 MB (S=8192) [probe_b200]'
  - 'r2: no-table twin of the pair kernel vs pair: -2.6% at 612 MB, -1.5% at 407, +-1% at 269-354, +1% at 2689: on the bench the whole table cost is <= 2.6% [r2-rope-pair2-magic-occ10]'
  - 'r2 portal: the pair path (half the table traffic) is used at 322/326 MB only; r2 vs r1 at L is -1.1% on the portal while -0.2% slower on the bench, i.e. about -3% on the two pair workloads if the four unchanged ones are noise. Suggestive, not proven; E2 extends the pair path to 354-2689 MB and submits.'
  - 'r1 variants A/C/W (8-lane layout, 64-80 regs) were +2..+4% at L: register cost confounded them, not table sharing.'
- id: H4
  statement: Resident warps per SM (not bytes in flight) set the latency tolerance; the portal at 1.5 GHz rewards resident warps more than the bench does.
  status: supported
  evidence:
  - 'r1: v4 16-lane layout (32 regs, 64 warps/SM) vs r0 (50 regs): -9/-6/-6% bench, -10/-10/-17% portal at S/M/L [run_tests, portal]'
  - 'r2: pair kernel (40 warps/SM, 2x bytes per thread) vs r1: bench S -2.6 / M -1.7 / L -0.1%; portal -2.5 / -1.7 / -1.1%: no amplification for bytes per thread [lab notebook]'
  - 'r2: 12 CTAs/SM (96 KB of x in flight) is +2..+8% slower than 10 CTAs/SM (80 KB) with identical SASS at 6.6-134 MB; row kernel 16 CTAs/SM best, 13 +2..+4.5%, 10 +5..+10% [r2-rope-pair2-magic-occ10]'
  - 'r1/r2: per-thread depth does not replace resident threads: persistent D=3/D=4 at 104-138 regs +5..+30%; HG=4 pairs (160 KB/SM) +7% at 57-61 MB, +2..+3% at L [probe_b200]'
- id: H5
  statement: Two rows per thread sharing one table load (16-lane layout, <= 48 regs) raise bytes in flight and help M/L.
  status: refuted
  evidence:
  - 'r2: pair HG=2 (48 regs NVRTC / 40 nvcc, 10 CTAs/SM) vs r1+magic within +-1.5% at every size on the bench; portal M -1.7% (same as bench). The gain that exists comes from fewer instructions per byte and the magic divisor, not from bytes in flight [r2-rope-pair2-magic-occ10, portal]'
  - 'superseded by H17 (waves) for the one remaining depth question.'
- id: H6
  statement: At 6.6-57 MB the single-wave fill time and the per-CTA replacement gap are the fixed cost; fewer, fatter CTAs with all loads issued up front cut 0.5-1 us there.
  status: open
  evidence:
  - 'r1: at 2.2 MB the kernel is at the copy floor (2.59 vs copy 2.56-2.64) [probe_b200]'
  - 'r2: empty one-shot grids span 1.6 us at 6.6 MB and 3.2 us at 36 MB vs 2.2 us for an empty 148x16 persistent grid; r1 minus persistent time is negative at every size, so dispatch is hidden behind memory work on the bench [r2-rope-psst-m]'
  - 'portal r2 vs bench: +0.4 us at 6.6 MB, +0.7 at 15.8 (both < 1 wave at 10 CTAs/SM), +1.5 at 36 MB (1.5 waves), +1.7 at 57-61 (2.3-2.5 waves), +1.9-2.0 at 102-269 MB, +1.3 at 612 MB: the penalty steps up at the first wave boundary and then stays flat. Reframed as H17; E1 tests it.'
- id: H7
  statement: Load hints on x (.nc, L1::no_allocate) and on cos/sin (.nc) change nothing measurable.
  status: refuted
  evidence:
  - 'r1: nc on x is -7% at 57 MB and -1% at 102 MB but plain x loads are 1-3% faster at 134-269 MB; nc on the tables alone changes nothing; table loads with L2::evict_last -2.7% at 612 MB, -0.9% at 2689 [probe_b200]'
  - 'r2 kernel uses nc/no_allocate on x and plain nc on the tables at every size, so the 134 MB+ plain-x and the 612 MB table-EL gains are unclaimed (H19, E2).'
- id: H8
  statement: The runtime modulo rl % S costs a measurable fraction of the per-CTA time; a host-computed magic divisor recovers it at S/M.
  status: supported
  evidence:
  - 'r2: magic divisor vs rl % S: -2.5% at 2.2 MB, -1.0..-2.1% at 6.6-16 MB, -0.5..-1.1% at 36-134 MB, 0 at L [r2-rope-pair2-magic-occ10]; in the portal kernel since r2.'
- id: H9
  statement: Partial or first-output-only evict_last is no better than evict_last on all stores.
  status: parked
  evidence:
  - 'r2 lead: worth <= 1% by the #38 result; not a slot.'
- id: H10
  statement: SOLAR for this problem is bytes/8 TB/s including cos/sin.
  status: refuted
  evidence:
  - 'portal r0: Tsol back-solved is about 0.53 x bytes/8 TB/s at >= 134 MB (2689 MB ~177 us vs floor8 336; 134 MB ~9.2-9.3 vs 16.8). Per-workload ceiling at L about 0.687 at 7 TB/s; r2 sits at 0.670-0.686 there.'
  - 'r2 portal: Tb is relatively fast at (64,8,8,128) (28.5 us, 4.7 TB/s) and at (1,8,8,128) (2.9 us), which is why those score 0.619 and 0.487 while running on the same fixed-cost curve as their neighbours.'
- id: H11
  statement: At the locked 1.5 GHz the cos/sin reads from L2 (2 B per byte of x) cost more on the portal than on the bench; halving table traffic gains 1-3% at L on the portal while bench-neutral.
  status: open
  evidence:
  - 'portal r1: L marginal 6.82 TB/s vs #38 at 7.03 TB/s with half the per-byte L2 reads.'
  - 'r2 bench: table cost <= 2.6% at 612 MB, about 1% elsewhere (no-table twin) [r2-rope-pair2-magic-occ10]'
  - 'r2 portal: L -1.1% vs r1 with only the two pair workloads (322/326 MB) changed; bench -0.2% the other way (see H3). E2 is the clean portal test: pair path at 354-2689 MB plus table evict_last.'
  - 'bench clocks cannot be locked; only portal slots can decide H11.'
- id: H12
  statement: The 2.2 MB workload is at the launch-plus-one-round-trip floor; the remaining gap to Tb (3.0 vs 2.9 us) is not a design lever.
  status: supported
  evidence:
  - 'r1: 16-lane 128-bit layout 2.59 us vs copy 2.56-2.64 on the bench; 64/128/256-thread shapes and all hint variants within 0.15 us [probe_b200]'
  - 'r2 portal 3.0 us vs Tb 2.9: Tb is a fused PyTorch (Triton-class) kernel at the same floor. Worth per 0.1 us is only +0.0006 on the mean. Parked; E3 does one cheap floor decomposition (empty 1-CTA kernel, loads-only, stores-only) for the record.'
- id: H13
  statement: A register-pipelined persistent grid beats the one-shot grid at 6.6-269 MB by removing CTA replacement and single-wave fill.
  status: refuted
  evidence:
  - 'r2: no register-only persistent configuration at <= 64 regs beats r1 by >= 3% at 36-134 MB; persistent with no prefetch at r1''s own occupancy is +3.6% at 6.6 MB rising to +13.8% at 612 MB (a rate deficit, not a fixed cost); dynamic work-stealing adds 15-20 us [r2-rope-psst-m]'
  - 'implication: hardware CTA dispatch is a free dynamic balancer here; persistent designs stay parked for #88.'
- id: H14
  statement: The bench 6.96 TB/s marginal (LDG.128 one-shot, evict_last stores) is the B200 read+write ceiling; no memory path exceeds it by >= 3% at 612-2689 MB.
  status: supported
  evidence:
  - 'r2 ceiling: control copy 6.95-6.97 TB/s; LDG/STG.256 0.5-1.3% below; Q/K interleave -0.4%; one-shot 1-D bulk ties within 1%; persistent bulk rings 11-12% slower regardless of ring size; r1 RoPE is 1.0-1.5% above the copy control at 2689 MB [r2-rope-ceiling-v4]'
  - 'consequence: L-band memory-path work is closed; the L levers left are table traffic (H3/H11) and hints (H19).'
- id: H15
  statement: The portal penalty versus the bench (+0.4-0.7 us below one wave, +1.5-2 us from 1.5 waves on, +2.5% at 2689 MB) is latency starvation at the locked clock; designs with more resident warps or fewer wave boundaries lose less on the portal than on the bench.
  status: open
  evidence:
  - 'r0 -> r1 (2x resident threads): S +10.0% bench / +10.5% portal; M +6.8 / +10.3; L +6.7 / +16.7 [lab notebook]'
  - 'r1 -> r2 (2x bytes per thread, 40 warps/SM): S +2.9 / +2.5; M +2.2 / +1.7; L -0.2 / +1.1: bytes per thread are not amplified, resident warps and (possibly) table traffic are.'
  - 'portal r2 per workload minus bench: 2.2 MB +0.5 us, 6.6 +0.4, 15.8 +0.7, 36 +1.5, 57 +1.7, 61 +1.7, 102 +2.0, 134 +1.9, 269 +1.9, 612 +1.3, 2689 +9.7 (2.5%).'
  - 'E1 (fewer waves) and E2 (pair at L) are the r3 tests; H15 holds if a bench-neutral change gains >= 2% on the portal in its band.'
- id: H16
  statement: x loads with an L2::evict_first cache_hint policy shrink the kernel''s L2 footprint so more evict_last output parks and fewer lines are displaced in-window; expected -1..-3% at 36-612 MB, neutral at 2689 MB.
  status: open
  evidence:
  - 'literature: CAKE kept EVICT_FIRST on streamed loads after paired B200 tests; quack saw a regression only with evict_first on stores [b200_sota]'
  - 'never tested on the B200 bench for #88 (H7 tested .nc and L1::no_allocate, which are L1-level); on A100 EF x + EL stores hurt L by 4% (#38 r5), A100 is not representative. Knob in E2.'
- id: H17
  statement: A one-shot grid with more than one wave pays an extra round trip at the wave boundary that is longer at the locked clock; a depth-2 pair kernel (2 pairs at positions s, s+1 per 16-lane group, 4 rows, 2 table rows, <= 64 regs, 8 CTAs/SM) makes <= 36 MB a single wave and halves the waves at 57-134 MB, gaining more on the portal than on the bench, most at 36 MB (portal +23% vs bench).
  status: open
  evidence:
  - 'portal r2 vs bench steps from +0.7 us at 15.8 MB (0.64 wave) to +1.5 us at 36 MB (1.5 waves) and stays about +1.5-2 us at all larger sizes (H15 evidence).'
  - 'counter-evidence: HG=4 pairs (160 KB/SM, 4 strided heads per CTA) +7% at 57-61 MB; 12 vs 10 CTAs/SM +2..+8% [r2]. Depth-2 at s, s+1 differs: 8 KB contiguous x per head per CTA (vs 4 KB) and 128 KB/SM. E1 tests it with the occupancy cap as a knob (6-8 CTAs/SM).'
- id: H18
  statement: The pair path''s 124 KB of L1 (132 KB carve-out) cannot hold the 128-211 KB tables of the S = 128-211 workloads (6.6, 15.8, 134 MB), so their table reads come from L2; an uncapped pair (12 CTAs/SM, full L1) or the row path (16 CTAs/SM, full L1) is better there on the portal (H11-coupled), neutral on the bench.
  status: open
  evidence:
  - 'r2 bench: the padded pair still beat the row path by -2.4..-4.5% at 6.6 MB and -0.5..-1% at 134 MB, so any L1 loss is < 1-4% on the bench [r2-rope-pair2-magic-occ10]'
  - 'r1: padding smem alone costs +0.4..+3.5% at 57-102 MB for the row kernel (smaller L1 for tables) [r2]. E3 tests S <= 256 routing.'
- id: H19
  statement: Dispatching load hints by size (plain x loads at >= 134 MB, nc/no_allocate below; table loads with L2::evict_last at >= 450k rows) recovers 1-3% at 134-2689 MB that r2 leaves on the table.
  status: open
  evidence:
  - 'r1: plain x 1-3% faster than nc at 134-269 MB (18.95 vs 19.17 at 134 MB; 38.77 vs 38.95 at 269); table EL -2.7% at 612 MB, -0.9% at 2689, neutral or +0.7% at 322-407 [r1-rope-v4nc-s128, r1-rope-ldg256-os64-tabel]'
  - 'r2 kernel: nc/no_allocate x and plain-nc tables everywhere. Knobs in E2 with 3 interleaved repeats; the earlier numbers are single-session.'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H17",
    "title": "Depth-2 pair kernel: one wave at <= 36 MB, half the waves at 57-134 MB",
    "operation": "structural_mutation",
    "parents": ["r2-rope-pair2-magic-occ10"],
    "band": "M",
    "instructions": "Build rope_pair2 in kernel.cu alongside the r2 kernels: each 16-lane group handles heads (2hp, 2hp+1) at positions (2*s2, 2*s2+1): rows row0 = 2*(bhp)*S + 2*s2, row0+1, row0+S, row0+S+1, with the second position predicated off when 2*s2+1 == S (odd S). Group index g -> (bhp, s2) via a magic divisor for S2 = ceil(S/2) computed on the host; Q groups first, then K; requires H and Hkv even, else fall back to r2's paths. Issue all 8 x loads (nc, L1::no_allocate) and 4 table loads (two table rows) before any FMA; evict_last cache_hint stores; 128-thread CTAs (8 groups = 32 rows, 16 KB of x per CTA), __launch_bounds__(128, 8). Check with compile_b200 (nvcc, not NVRTC) that regs <= 64 and spills = 0; if it spills, try computing outputs in place or 96-thread CTAs. Knobs: occupancy cap 8 vs 6 CTAs/SM (dynamic smem padding with the r2 carve-out trick) and 128 vs 256 threads. Measure vs r2 with 3 interleaved repeats at 6.6, 15.8, 36, 57, 61, 102, 134, 269 MB (6.6 and 15.8 MB are expected to lose from too few CTAs; find the row threshold where depth-2 starts winning, probably 25-35k rows). Record the first-wave size: at 36 MB (35,160 rows) the grid is 1,099 CTAs vs 1,184 resident, a single wave. Then wire the best threshold into the dispatcher (depth-2 for R in [threshold, 140k], r2 pair/row elsewhere), run_tests, and if no workload is worse than +1% submit as the portal probe of H15/H17 even when the bench gain is only 0..-2%: the mechanism predicts the gain appears on the portal (36 MB +23% vs bench) and r2 already proved pairs are not amplified, so a null portal result here is informative too.",
    "success": "Bench -3% or better at 36 MB and no loss above +1% at 57-134 MB; or bench within +-1.5% at 36-134 MB and portal -4% or better at 36 MB with M geomean -2% (H17 supported, H15 narrowed to wave boundaries).",
    "refuted_if": "Depth-2 is >= +2% slower than r2 at 36 and 57 MB on the bench at both occupancy caps, or bench-neutral and portal within +-1.5% at 36-61 MB (then depth and wave count are closed for #88; the portal penalty at M is something else).",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H11",
    "title": "L/M hint and pair-path dispatch: pair at S > 1536, table evict_last, plain x at >= 134 MB, evict_first x",
    "operation": "knob_mutation",
    "parents": ["r2-rope-pair2-magic-occ10"],
    "band": "L",
    "instructions": "Four knobs on the r2 code, each measured against unchanged r2 with 3 interleaved repeats per size (bench noise is 1%; report medians and the spread). (a) H3/H11: remove the S <= 1536 limit so the pair path runs at 354, 407, 612 and 2689 MB; also try rope_pair<256> (16 pairs, 32 rows per CTA) there, since 128 vs 256 threads tie for the row kernel at L. (b) H19: table loads via ld.global.nc.L2::cache_hint with a createpolicy evict_last policy (r1 saw -2.7% at 612 MB, -0.9% at 2689), tested at 322-2689 MB and at 134-269 MB. (c) H19: plain ld.global.v4 (no .nc, no no_allocate) on x for R >= 120k rows (134 MB and up), tested at 134, 269, 612, 2689 MB; keep nc below 134 MB where r1 measured -7% at 57 MB. (d) H16: x loads with ld.global.nc.L1::no_allocate.L2::cache_hint and a createpolicy fractional evict_first policy (verify the PTX form compiles with compile_b200; fall back to .L2::evict_first qualifier if cache_hint on .nc is rejected), tested at 36, 57, 134, 269, 612, 2689 MB. Then compose the best setting per size range into one dispatcher (shape-only rules), run compile_b200 for regs/spills on every kernel, run_tests, and submit if no workload is worse than +1% on the bench. In the design card list every hint per path (x, w, st) so the emulator can learn them. Report per-knob tables; a knob that is neutral on the bench but mechanism-plausible for the portal (a and b) may still be included if it never loses more than 0.5%.",
    "success": "Composite kernel at least -1.5% at 612 and 2689 MB and -1% at 134-269 MB on the bench with nothing worse than +1%; portal L geomean -2% or better (about +0.003), M -1%.",
    "refuted_if": "(a) pair at L is >= +1.5% slower at 354-2689 MB in all three repeats and (b)+(c) are within +-0.7% (then H11 is closed on the bench side and only the portal probe of (a) remains); (d) worse than +1% at any M size closes H16.",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H18",
    "title": "Small-shape specialisation: S <= 256 table-in-L1 routing, 64-thread pairs at 6.6-16 MB, and the 2.2 MB floor decomposition",
    "operation": "specialisation",
    "parents": ["r2-rope-pair2-magic-occ10"],
    "band": "S",
    "instructions": "Part 1 (H18): for shapes with S <= 256 (tables 128-256 KB: workloads 6.6, 15.8 and 134 MB) compare three routings with 3 interleaved repeats: r2 padded pair (10 CTAs/SM, 124 KB L1), unpadded pair with default carve-out (12 CTAs/SM, full L1), and rope_row<128> (16 CTAs/SM, full L1). Also check whether the carve-out switch itself costs anything: time an empty 256-CTA kernel launched with PAIR_PAD_SMEM + 50% carve-out vs the same kernel with 0 smem, each immediately after a harness-style memset, 5 interleaved repeats. Part 2 (H6 at S): 64-thread pair CTAs (4 pairs, 8 rows) and 128-thread pairs at 2.2-15.8 MB vs r2 (2.2 MB currently takes the row path because R <= 4096; test the pair there too with 64-thread CTAs = 256 CTAs). Part 3 (H12, record only): decompose the 2.2 MB and 6.6 MB times on the bench: empty 1-CTA kernel, empty grid of r2's shape, loads-only (x + tables, result reduced into one predicated store), stores-only, full kernel; report each span so the fixed-cost model (launch floor, round trip, store drain) gets numbers. Wire any routing that wins by >= 2% on its workloads into the dispatcher (shape-only rules), run_tests, and submit only if at least one of the 6.6/15.8/134 MB workloads improves by >= 2% with nothing worse than +0.5%; otherwise record findings only. Expected value is small (+0.001-0.002); the session's main product is the measurements for H6/H12/H18.",
    "success": "A routing that is >= 2% faster at 6.6, 15.8 or 134 MB on the bench with no regression elsewhere, or a carve-out switch cost >= 0.15 us that can be removed; plus the floor decomposition recorded.",
    "refuted_if": "All three routings within +-1% at 6.6, 15.8 and 134 MB, the carve-out switch within 0.05 us, and 64-thread pairs within +-1% at 2.2-15.8 MB: H18 closed, S-band CTA shape closed, 2.2 MB stays parked.",
    "model": "sonnet"
  }
]
```