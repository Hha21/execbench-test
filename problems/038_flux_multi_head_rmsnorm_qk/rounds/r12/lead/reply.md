### Assessment

Remaining score: we are at 0.6125, the leader at 0.6275. Eleven rounds have shown the one-shot LDG.256 kernel runs at plain-copy speed at M/L and at the hardware floor at S (about 1.05 µs outside the CTAs, 0.67 µs first DRAM latency, reads at the 6 TB/s read cap, stores +0.67 µs). Closed this round: H11 supported (half the SMs reach full read bandwidth, so the ceiling is DRAM/L2-side), H12 refuted (die affinity gains nothing and the home hash is invisible), H13 refuted (4-lane layout no gain).

What is still unexplained and worth money: (1) evict_first loads give a repeatable −8% at 586 tokens but only −2% at 1024 and nothing above, which looks like an L2 footprint threshold between about 37 and 58 MB of new allocations; if real, a footprint-capped store policy could recover several percent at 1024–2048 tokens (+0.005–0.01). (2) r7's one-wave-ahead L2 prefetch gave −2 to −3% on the portal at 256 and 1024 tokens and has never been combined with cluster-2 or R=3. (3) Cluster-2 is the only S lever found and its mechanism is unknown; if it is placement, it may be reproducible and extensible. (4) Block-to-tile order has never been measured at L, and the rented B200 has never been asked to lock its clocks, which would retire the emulator.

This round: E1 maps the EF/dirty-flush curve and tests hybrid store policies (H18); E2 dissects cluster-2 (H17); E3 builds the consolidation candidate c3 with prefetch (H19); E4 tries the clock lock and block ordering at L (H20).

```yaml ledger
# Hypothesis ledger for #38, maintained by the research lead (round.py lead) and read by every design session.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
# Seeded 8 October from rounds r1-r10 and the portal results; updated by the research lead for r11 and r12 (8 October).

constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
hypotheses:
- id: H1
  statement: Output stores marked L2 evict_last keep output in L2 past the end of the timed window, so medium sizes gain most.
  status: supported
  evidence:
  - 'portal: r5 0.609 vs r3 0.588 (M -8%)'
  - 'probe r10: evict_last stores beat plain stores by 8-19% at 128-1024 tokens, 2.5-5% at 4096-8192'
  - 'probe r10: the saving is capped at 3-4 us (~25-30 MB) from 1024 tokens up; partial/fractional evict_last is no better
    (see H15)'
- id: H2
  statement: evict_first on stores and evict_last on weights hurt on B200.
  status: supported
  evidence:
  - 'portal: r2 0.576 (hinted 256-bit) = g2 0.577 (hinted 128-bit); d1 0.582 (no hints)'
  - quack sm100 regression
  - 'portal r7 vs c1 at L (same code except x evict_first + nc weights): +1.9..+2.6%; evict_first on x costs at L too'
- id: H3
  statement: 256-bit loads/stores help medium and large sizes a little.
  status: supported
  evidence:
  - 'portal: r3 (256-bit, no hints) vs d1 (128-bit, no hints): M -2.4%, L -1.2%'
- id: H4
  statement: CTA launch/retire overhead is a bottleneck, so fewer, fatter CTAs or one resident wave help.
  status: refuted
  evidence:
  - 'probe r10: one-wave grids tie at 12.6-101 MB and lose 11-15% above 200 MB'
  - 'portal: r8 512-thread CTAs at L +3.2%'
  - 'probe r10: all 768 CTAs of the 12.6 MB grid start within 0.06 us; the empty-grid span overlaps fully with memory work
    at L'
  - 'probe r11 E3(1): empty-kernel CUPTI floor ~1.4 us at any block size; 768x256 1.63 us; launch params/API/smem/priority
    move it by < 0.03 us'
- id: H5
  statement: TMA / bulk-copy pipelines cut SM instructions per byte and win at the portal's 1500 MHz SM clock.
  status: refuted
  evidence:
  - 'probe r10: bulk stores 27% slower than STG.256 at 805 MB; bulk loads +3%; TMA + mbarrier setup adds ~0.8 us at 12.6 MB'
  - 'portal: g2-tma 0.510, g2-tmaws 0.537; bench r10-bulk1d-ring-stel predicted 0.5815'
- id: H6
  statement: Persistent grids with work stealing remove the tail and beat one-shot.
  status: refuted
  evidence:
  - 'probe r10: one failing atomic per CTA costs ~1 us at 12.6 MB; pipelined stealing +9-10% at S/M, +0.7% at L; static persistent
    3% worse than stealing at L'
  - 'portal: v039, g2-ws, g2-tma all slower; bench r10-ws-l-steal-disp predicted 0.605'
- id: H7
  statement: The harness's zero-fill flush leaves dirty lines in L2; their write-back during our kernel costs ~0.5 us at S
    and ~4 us (about 30-40 MB) at M/L, and the cost lands on reads, not writes.
  status: supported
  evidence:
  - 'probe r10: in-kernel span after zero flush vs read-only flush: 12.6 MB 3.07 vs 2.56 us, 25 MB 4.10 vs 3.58, 100 MB 16.6
    vs 12.3'
  - 'probe r10: read-only kernels +7..11 us at 100-400 MB after a zero flush, copy +5..6 us, write-only +-1 us'
  - 'no legitimate kernel-side mechanism found; load hints do not avoid it at >= 1024 tokens (H8); the 586-token exception is
    now H18'
- id: H8
  statement: evict_first on the streamed loads makes later allocations evict our own clean lines instead of dirty flush lines,
    cutting H7's write-backs.
  status: refuted
  evidence:
  - 'probe r10: read-only kernel -13 to -15% with evict_first loads, but the full kernel with evict_last stores gains only
    1-2% at 256-1024 tokens (8% once at 586), 0 to +1.5% above; .cs/.lu/no_allocate/nc+EF all within +-2%'
  - 'bench r10-ldef-real-stel-disp: S -0.8%, M -1.2%, L +0.3% vs r6; kept as a ~1% tweak for B*S <= 1024, not a lever'
  - 'r10: the CUTLASS EVICT_FIRST constant passed as a kernel argument is a no-op; the qualifier or createpolicy form works'
  - 'r11 probes: evict_first at 586 tokens -8% in three separate sessions (r10, r11-consol, r11-fi4lane); at 1024 -1.9..-2.5%;
    at 2048 +1.5%. The size dependence is the open question (H18)'
- id: H9
  statement: 'The portal (SM 1500 MHz) penalises SM-bound designs more than the bench (~1940 MHz): kernels that lag a plain
    copy on the bench lose more on the portal; the clock-bound share is ~2.5 us of 4.2 us at 12.6 MB and ~6 us of 14.5 us
    at 100 MB.'
  status: supported
  evidence:
  - 'emulator fit on 18 kernels: about +5% portal time per 20% copy lag at L'
  - r8/r9 bench -1% -> portal +3-4%
  - 'portal/bench ratios: c1 S +15%, M +9%, L 0% (INFERRED split of the clock-bound share)'
  - 'r11-consol: copy lag of the current paths: S +3-5% (the shuffle/rsqrt chain, 0.15-0.2 us), 1024 R=3 +0.1%, 2048-8192 0%'
  - 'r10-rtok R=3 (48 regs) at 1024 tokens: bench -2.7%, portal -2.4% vs c1; the 48-register path did not lose extra on the portal'
  - 'untested: whether the rented B200 can lock clocks (nvidia-smi -lgc/-lmc); if it can, the emulator is replaced by direct timing (E4)'
- id: H10
  statement: At 12.6 MB the one-shot kernel already equals a plain copy; the remaining S time is ~1 us outside the CTAs (clock-bound),
    one DRAM round trip (~0.7 us), streaming at the ~6 TB/s read cap, and H7's write-backs.
  status: supported
  evidence:
  - 'probe r10 timelines: all 768 CTAs start within 0.06 us, first load returns at 0.67 us, last at 2.9 us; in-kernel span
    3.1-3.3 us vs 4.16 us CUPTI'
  - 'probe: copy 4.4 us vs r5 4.2 us (bench); empty 768-CTA grid 1.57 us'
  - 'probe r11 E3(3), 128 tokens, medians of 105: CUPTI/in-kernel: EL stores 4.51/3.42, plain stores 4.35/3.30, no stores
    3.80/2.75, empty body 2.02/0.38. Outside-the-CTAs 1.05 us with or without stores (launch front-end + teardown); reads
    12.6 MB in ~2.1 us (~6 TB/s); stores add 0.67 us'
  - portal marginal 12.6->25 MB 7-8 TB/s; fixed ~3.0 us on the portal
- id: H11
  statement: 'The ~7 TB/s read+write ceiling at L is DRAM/L2-side: read-only bandwidth (5.0-6.3 TB/s clean) does not rise with
    more active SMs, more in-flight bytes per SM, TMA, or die-affine addressing, so no SM-side change can lift L.'
  status: supported
  evidence:
  - 'probe r10: read-only 402 MB 71.9 us (64 us clean, 5.6-6.3 TB/s); write-only 54.6 us (7.4 TB/s); copy 6.7-6.95 TB/s'
  - 'probe r10: 64-128 KB in flight per SM (R=2..4) and bulk loads do not beat r6 at L'
  - 'probe r11 E1: read-only 400 MB gated by %smid: 148 SMs 83.9 us, 74 SMs (any half) 81.5-84.1 us, 37 SMs 91.3 us. Half the
    SMs reach 100% of full read bandwidth'
  - 'probe r11 E1: affine/mixed/sorted reads 77.4/77.8/78.2 us; anti-affine 96.9 (+25%, cross-die cap ~3.8 TB/s)'
  - 'open corollary: the mixed read+write stream (7.0 TB/s) beats alternating pure phases (read 6.3, write 7.4 -> harmonic 6.8),
    so read/write phasing cannot help'
- id: H12
  statement: Each B200 die homes half the HBM/L2; the address-to-partition hash is visible in low address bits, and a CTA
    reading only lines homed on its own die gets higher bandwidth or lower latency than mixed traffic, so a die-affine tile
    assignment lifts M/L.
  status: refuted
  evidence:
  - 'probe r11 E1: cold-miss latency is bimodal (~790 near vs ~1160 cycles far); die map recovered (group A 72 SMs, group B 76);
    home granularity exactly 4 KB; no XOR mask of bits 7..20 correlates (> 0.08): the hash uses physical bits above 2 MB'
  - 'probe r11 E1: read 350 MB classified: affine 77.4 us, mixed 77.8, anti-affine 96.9; copy 265+265 MB affine/mixed within
    1%; per-die queue needs one atomic per CTA (+6-7% at L)'
  - 'r11-e1-h12-control: dead end, do not revisit'
- id: H13
  statement: FlashInfer's sm_100 layout (4 lanes per row, 32 floats per thread, 8 rows per warp instruction) beats the half-warp-per-row
    layout on B200, especially at 4-60 MB.
  status: refuted
  evidence:
  - 'fi-pr5305: 1.14x at M=8192, 1.20x at M=32768 rows (bf16) on B200'
  - 'probe r11-fi4lane: 4-lane and 8-lane layouts within +-3% of c1 at 128-8192 tokens with matched hints, 1-4% slower at 586;
    4/8-lane plain copies no faster than 16-lane copies. The bf16 gain was about bytes per thread, which fp32 already has'
- id: H14
  statement: Part of the ~1 us outside the CTAs at S depends on launch attributes we control; a cluster-of-2 launch cuts the
    smallest sizes.
  status: supported
  evidence:
  - 'probe r10: empty 768-CTA grid spans 1.57 us CUPTI; r6 CUPTI 4.16 vs in-kernel 3.1-3.3 us'
  - 'probe r11 E3(2): param count, launch API, dyn smem, memory-sync domain, priority, carveout (except 100) each move the real
    kernel by 0 +- 0.03 us; cluster 2 is the only attribute that moves it (-3..-4% at 128-256 tokens); cluster 4/8/16 worse'
  - 'portal r11-e3-cluster2-rtok-disp (#62670) vs r10-rtok: 128 tok -6.2%, 131 -4.2%, 256 -1.6/-3.1%'
  - 'portal c2-cluster2-s-only (#62679, clusters at <= 300 tokens only): 0.6125, new best; 128 tok 4.6 us (-4.2%), M/L unchanged'
  - 'mechanism unknown: the empty kernel does not change with cluster dims, so it is not a faster launch path (see H17)'
- id: H15
  statement: L2 protects at most ~25-30 MB of evict_last lines (and holds a similar amount of dirty flush lines), a hardware
    cap; the store-side end state cannot be improved further.
  status: parked
  evidence:
  - 'probe r10: evict_last saves 3-4 us (~25-30 MB) at every size from 1024 to 8192 tokens; last-48/80/112 MB-only and fractional
    0.5/0.75 variants are within +-1% or worse (tested at 2048-8192 tokens only; 1024 untested, see H18)'
  - 'probe r10: H7''s cost is also ~30-40 MB worth at M/L'
- id: H16
  statement: 'Cluster launches of large grids cost on the portal: directly at the clustered sizes, and they seem to slow later
    plain launches in the same harness process; small-grid clusters do not.'
  status: supported
  evidence:
  - 'portal r11-e3-cluster2-rtok-disp: +1.6..+5.0% at 2048-8192 tokens (clustered there; bench -0.7..+1.9%) and +2.5..+2.9%
    at 586-1024 tokens where code and launch equal r10-rtok (bench +0.3..+0.6%)'
  - 'bench probe: one cluster launch does not slow later plain launches in the same process (within 1%)'
  - 'portal c2 (clusters only <= 300 tokens): 586-8192 tokens back to r10-rtok within +-0.4%, so r11''s M/L loss came from
    clustering >= 2048 tokens'
  - 'harness workload order: r11''s clustered 4,1657 / 4,1024 / 8,773 run at positions 2-4, before the plain 586/1024-token
    workloads (positions 5-9) that slowed 2.5-2.9% (single observation)'
- id: H17
  statement: 'The cluster-of-2 gain at <= 256 tokens comes from CTA placement or dispatch order (pairs co-scheduled on one GPC/TPC,
    a different SM fill order, or paired L2 traffic), not from the launch path; the same effect can be reproduced or enlarged
    with a plain launch and a blockIdx-to-tile permutation, or with a different CTA size inside the cluster.'
  status: open
  evidence:
  - 'r11 E3: empty kernels unchanged by cluster dims; real kernel -3..-4% at 128-256 tokens; cluster 4/8/16 worse than 2;
    scheduling policy (spread/load-balance) changes nothing'
  - 'r11 E1: plain one-shot grids are dispatched round-robin over GPCs in TPC pairs (blockIdx 0.. -> smid 142,143,0,1,16,17,...),
    every SM gets exactly 8 CTAs at 1184; the per-SM histogram for cluster-2 launches was not run'
  - 'r8: a balanced 148xk interleaved blockIdx mapping under a plain launch was within +-1% (so a naive permutation is not it)'
- id: H18
  statement: 'The -8% from evict_first loads at 586 tokens is a footprint effect: dirty flush lines are not written back while
    the kernel''s new L2 allocations (evict_last stores plus the small evict_first recycling set) stay below a clean-capacity
    threshold somewhere between ~37 MB (586 tokens) and ~58 MB (1024 tokens). Capping the dirty-output footprint at 1024-2048
    tokens (evict_first stores for the early CTAs, evict_last for the last ~25-30 MB) recovers part of the -8% there.'
  status: open
  evidence:
  - 'EF loads: 586 tok -8% in three sessions; 1024 -1.9..-2.5%; 2048 +1.5% (r10, r11-consol, r11-fi4lane probes)'
  - 'r7 prefetch (normal-priority allocations) at 586: +1.7%, consistent with extra allocations evicting dirty lines'
  - 'counter-evidence: last-48/80/112 MB evict_last with earlier EN/EF stores was within +-1% at 2048-8192 tokens (r10); the
    1024 case and a fine size sweep were never measured'
  - 'H7: write-only kernels pay no dirty cost, read-only pay most; so the hybrid must keep READ allocations small (EF loads) too'
- id: H19
  statement: 'L2 bulk prefetch one resident wave ahead (r7) is additive to cluster-2 at the 256-token sizes and to R=3 at 1024
    tokens; with an evict_first cache policy on the prefetch it stops hurting at 586-2048 tokens and gains there too.'
  status: open
  evidence:
  - 'portal r7 vs c1 (same loads at <= 1536 tokens): 1x256 6.2 vs 6.5, 2x128 6.3 vs 6.4, 1024 tok 16.0/16.0/16.1 vs 16.4/16.4/16.2
    (-2..-3%); 586 10.6 vs 10.4; 128 tok 4.8 vs 4.8 (one wave: no-op)'
  - 'bench r7/r8: prefetch -2% at 256 tokens, +1.7% at 586, -2.5% at 1024, neutral at 2048, +4-5% at L; evict_last prefetch
    worse everywhere; evict_first prefetch untested'
  - 'never combined with cluster-2 (c2) or R=3; both r7''s prefetch and R=3 give -2.4% at 1024 on the portal (16.0 vs 16.4)'
- id: H20
  statement: 'The blockIdx-to-tile order (Q then K vs Q/K interleaved per CTA or per wave, reversed, channel-spreading permutations)
    changes DRAM efficiency by >= 2% at 2048-8192 tokens.'
  status: open
  evidence:
  - 'r10: block order within 1% at 128-256 tokens; 586-8192 never measured (a CUPTI error ended the run)'
  - 'r10 one-wave probes: per-CTA contiguous chunks were the worst layout at L (132-137 vs 115 us), so address order does matter
    at L in at least one direction'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H18",
    "title": "Map the evict_first / dirty-flush curve from 300 to 2048 tokens and test footprint-capped store policies at 1024 and 2048",
    "operation": "structural_mutation",
    "parents": ["c2-cluster2-s-only"],
    "band": "M",
    "instructions": "Use c2's kernel.cu as the base (plain launch, no clusters; R=1 paths only). Step 1, the curve: with harness-style timing (zero flush) and the probe's clean-flush option, time the R=1 kernel with (a) default x loads + EL stores and (b) EF x loads (qualifier or createpolicy form, not the constant-argument form) + EL stores, at token counts 300, 400, 500, 586, 700, 800, 900, 1024, 1200, 1536, 2048 (any B with B*S equal to these; use 1xN). 5 interleaved reps, medians. Report per size: time dirty, time clean, dirty minus clean, and the EF gain, for both variants. Also run (c) EF loads + EF stores and (d) default loads + plain stores at 586, 1024, 2048 dirty and clean, to see which allocations pay the dirty cost. Step 2, hybrids at 1024 and 2048 tokens: order CTAs so that the output written LAST is a contiguous tail; make stores evict_last only for the last T MB of output and evict_first (and separately evict_normal) for the earlier output, with T in {20, 30, 40}; keep EF loads. Also test the reverse (EL first, EF last) as a control. 7 interleaved reps against the c2 1024 path (R=3) and the c2 2048 path (R=1). Step 3: if any hybrid gains >= 3% at 1024 or 2048, build a dispatcher (c2 everywhere else, the hybrid at 600 < tok <= 2500 or the measured range), run_tests, and record findings. Record the curve in findings whatever the outcome.",
    "success": "A store/load policy that is >= 3% faster than c2's path at 1024 or 2048 tokens in 7 interleaved reps, confirmed by run_tests on the 1024- or 2048-token workloads; or the curve shows a sharp EF-gain threshold between 586 and 1024 tokens that explains the -8%.",
    "refuted_if": "The EF gain decays smoothly with output size (no threshold) and every hybrid is within +-1% of all-evict_last at 1024 and 2048 tokens. Then close H18 and record that the 586-token effect is not extensible.",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H17",
    "title": "Why does a cluster of 2 cut 128-256 tokens by 3-4 percent, and can a plain launch or a different cluster shape do more",
    "operation": "structural_mutation",
    "parents": ["c2-cluster2-s-only"],
    "band": "S",
    "instructions": "Step 1, placement: add a %smid + globaltimer log (one record per CTA: smid, start, first-load-return, end) to c2's S path and run plain vs cluster-2 launches at 128 and 256 tokens (5 reps each, cold L2 via harness-style flush). Report the per-SM CTA histogram, which SM pairs the two CTAs of a cluster land on, the blockIdx-to-SM order, the start-time spread, and the distribution of first-load-return times for both launches. Step 2, isolate: (i) under a PLAIN launch, apply the blockIdx-to-tile permutation that reproduces the cluster-2 SM order observed in step 1; (ii) under a cluster-2 launch, make the two CTAs of a cluster process tiles 1184 apart instead of adjacent (and the reverse: adjacent heads of the same token); (iii) cluster 2 of 128-thread CTAs (8 rows each) and cluster 2 of 512-thread CTAs; (iv) cluster 2 with the weight load issued by only one CTA of the pair and shared through DSMEM (only if (i)-(iii) point at L2 request count). 8-10 interleaved reps each at 128, 131, 256 tokens against c2's S path (cluster-2, EF, nc). Step 3: if a variant beats c2 by >= 2% at the 128- and 256-token sizes, confirm with run_tests (S band) and report the predicted score; if the placement data explains the gain, write the mechanism in findings.",
    "success": "Either a plain-launch permutation reproduces >= 2/3 of the cluster-2 gain (mechanism = placement/order), or a cluster variant beats c2's S path by >= 2% at 128 and 256 tokens in interleaved probes and run_tests predicts >= 0.614.",
    "refuted_if": "SM placement and dispatch order are identical for plain and cluster-2 launches, no permutation or cluster shape moves the time by more than 1%, and the gain stays unexplained. Then park H17 and stop S launch work.",
    "model": "opus"
  },
  {
    "id": "E3",
    "hypothesis": "H19",
    "title": "Consolidation candidate c3: c2 plus one-wave-ahead L2 prefetch at the 256-token and 1024-token sizes, with an evict_first prefetch policy tried at 586-2048",
    "operation": "knob_mutation",
    "parents": ["c2-cluster2-s-only", "r7-ldg256-l2pf-mdisp"],
    "band": "all",
    "instructions": "Start from c2-cluster2-s-only (kernel.cu + binding.cpp, two-file build). Add r7's prefetch: thread 0 of each CTA issues cp.async.bulk.prefetch.L2.global of the x tile for the CTA one resident wave ahead (distance = num_SMs x resident CTAs per SM: 8 for the R=1 paths, 5 for the R=3 path; read num_SMs once at module level). Add a second form with a cache policy: cp.async.bulk.prefetch.L2.global.L2::cache_hint [addr], bytes, pol with pol from createpolicy.fractional.L2::evict_first.b64. Probes, harness timing, 7 interleaved reps, medians, always including the unmodified c2 path for that size: (1) 256 tokens (both 2x128 and 1x256 shapes): c2 (cluster-2) vs c2 + plain prefetch vs c2 + EF prefetch; (2) 128 and 131 tokens: c2 vs c2 + prefetch (expect no-op; check it costs nothing); (3) 586 tokens: c2 vs + plain prefetch vs + EF prefetch; (4) 1024 tokens: c2 R=3 vs R=3 + prefetch (plain and EF) vs r7's R=1 + EF loads + prefetch; (5) 2048 tokens: c2 vs + EF prefetch. Then build c3 with every combination that gained >= 1.5% and nothing that lost, keep dispatch thresholds on B*S only, run compile_b200 (check regs stay <= 32 on the R=1 paths and 48 on R=3), run_tests against c2, and report the predicted portal score and per-size deltas. Also report in findings whether prefetch and cluster-2 are additive at 256 tokens.",
    "success": "c3 passes 16/16 and run_tests shows 256-token sizes >= 2% faster than c2 or 1024-token sizes >= 1.5% faster, with no size slower by more than 0.5%; predicted portal score >= c2 + 0.002.",
    "refuted_if": "Prefetch adds nothing on top of cluster-2 at 256 tokens and nothing on top of R=3 at 1024 tokens (within +-1%), and the evict_first prefetch does not help at 586-2048. Then record H19 refuted and submit nothing from this task.",
    "model": "sonnet"
  },
  {
    "id": "E4",
    "hypothesis": "H20",
    "title": "Clock lock on the rented B200, then blockIdx-to-tile order at 2048-8192 tokens",
    "operation": "knob_mutation",
    "parents": ["c2-cluster2-s-only"],
    "band": "L",
    "instructions": "Part A (tooling, 10 minutes max): try to lock the rented B200 to the portal clocks: nvidia-smi -lgc 1500,1500 and nvidia-smi -lmc 3996 (also try nvidia-smi -ac and the NVML equivalents via pynvml if nvidia-smi lacks permission); verify with nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv under load. If locking works: time c2 at all 10 distinct sizes locked and unlocked (3 reps each, harness timing), compare the locked numbers with c2's portal table (4.6/4.7/6.4/6.4/10.5/16.0-16.3/29.9-30.0/58.4/87.5/93.9/115.7 us), and record the per-size ratios; reset clocks (-rgc, -rmc) afterwards. If it does not work, record the exact error and move on. Part B (H20): in c2's R=1 M/L path, make the blockIdx-to-tile map a template parameter and compare at 2048, 4096 and 8192 tokens, 5 interleaved reps, harness timing: (1) natural (all Q tiles then all K tiles, the current order); (2) Q/K interleaved per CTA (even blockIdx = Q tile, odd = K tile, same tile index); (3) Q/K interleaved per 1184-CTA wave; (4) reversed; (5) a stride permutation that places consecutive CTAs 1184 tiles apart (tile = (b % 1184) * ceil(n/1184) + b / 1184); (6) the per-GPC-aware order: tile = pair-major so that the two CTAs of a TPC pair (blockIdx 2i, 2i+1) read tiles 8 KB apart vs 4 MB apart. Report the per-size medians and the plain copy with the same orders for reference. If any order is >= 2% faster at two L sizes, confirm with run_tests on the L workloads against c2 and build a dispatcher candidate.",
    "success": "Part A: clocks lock and the locked bench times match the portal per size within 3%, which retires the emulator. Part B: an order that is >= 2% faster than natural at two or more L sizes in interleaved probes and in run_tests.",
    "refuted_if": "Part B: every order is within +-1% of natural at 2048-8192 tokens (then close H20: address order does not matter for one-shot grids at L). Part A failing is a tooling result, not a refutation.",
    "model": "sonnet"
  }
]
```