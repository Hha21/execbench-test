### Assessment

The one-shot LDG.256/STG.256 kernel with evict_last stores sits at 0.6105 against the leader's 0.6275. Round r10 closed most structural axes on the bench: TMA and bulk stores, persistent and work-stealing grids, fat CTAs, more rows per thread, and load hints (except a ~1% evict_first gain below 1024 tokens). Marginal bandwidth at S is already 8 TB/s and at L 6.95 TB/s, so the remaining score is in (a) the ~3.3 µs portal fixed cost at S, of which ~1 µs lies outside the CTAs and scales with the SM clock, (b) the ~4 µs dirty-flush write-back at M/L that hints cannot remove, and (c) the 1:1 read+write ceiling at L, where reads alone reach only 5.6-6.3 TB/s while writes alone reach 7.4 TB/s. A 10% cut is worth +0.010 (S), +0.014 (M), +0.019 (L).

This round stops tuning hints and tests three untried structural ideas plus one consolidation. E1 asks whether the read ceiling is per-SM or DRAM-side and whether the two dies' L2 partitions are addressable through the address hash, the only mechanism that could lift L above the public ceiling. E2 ports FlashInfer's sm_100 layout (4 lanes per row, 128 B per thread), the one public B200-specific change we have never measured and which they credit with +14-20% at S/M-band sizes. E3 maps which launch attributes, if any, move the clock-bound ~1 µs outside the CTAs. E4 packages r10's two kept ~1-3% effects into one candidate and measures its copy lag, so the user can submit it paired with r10-ldef-real-stel-disp. Recommended portal pair now: r10-ldef-real-stel-disp and r10-rtok-m-disp (both predicted 0.611).

```yaml ledger
# Hypothesis ledger for #38, maintained by the research lead (round.py lead) and read by every design session.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
# Seeded 8 October from rounds r1-r10 and the portal results; updated by the research lead for r11 (8 October).

constraints:
  - "No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness's measurement, not kernel speed. Off-limits unless NVIDIA approves."
  - "Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call."
  - "No overlap with the harness's own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window."

hypotheses:
  - id: H1
    statement: "Output stores marked L2 evict_last keep output in L2 past the end of the timed window, so medium sizes gain most."
    status: supported
    evidence: ["portal: r5 0.609 vs r3 0.588 (M -8%)", "probe r10: evict_last stores beat plain stores by 8-19% at 128-1024 tokens, 2.5-5% at 4096-8192", "probe r10: the saving is capped at 3-4 us (~25-30 MB) from 1024 tokens up; partial/fractional evict_last is no better (see H15)"]
  - id: H2
    statement: "evict_first on stores and evict_last on weights hurt on B200."
    status: supported
    evidence: ["portal: r2 0.576 (hinted 256-bit) = g2 0.577 (hinted 128-bit); d1 0.582 (no hints)", "quack sm100 regression"]
  - id: H3
    statement: "256-bit loads/stores help medium and large sizes a little."
    status: supported
    evidence: ["portal: r3 (256-bit, no hints) vs d1 (128-bit, no hints): M -2.4%, L -1.2%"]
  - id: H4
    statement: "CTA launch/retire overhead is a bottleneck, so fewer, fatter CTAs or one resident wave help."
    status: refuted
    evidence: ["probe r10: one-wave grids tie at 12.6-101 MB and lose 11-15% above 200 MB", "portal: r8 512-thread CTAs at L +3.2%", "probe r10: all 768 CTAs of the 12.6 MB grid start within 0.06 us; the empty-grid span overlaps fully with memory work at L"]
  - id: H5
    statement: "TMA / bulk-copy pipelines cut SM instructions per byte and win at the portal's 1500 MHz SM clock."
    status: refuted
    evidence: ["probe r10: bulk stores 27% slower than STG.256 at 805 MB; bulk loads +3%; TMA + mbarrier setup adds ~0.8 us at 12.6 MB", "portal: g2-tma 0.510, g2-tmaws 0.537; bench r10-bulk1d-ring-stel predicted 0.5815"]
  - id: H6
    statement: "Persistent grids with work stealing remove the tail and beat one-shot."
    status: refuted
    evidence: ["probe r10: one failing atomic per CTA costs ~1 us at 12.6 MB; pipelined stealing +9-10% at S/M, +0.7% at L; static persistent 3% worse than stealing at L", "portal: v039, g2-ws, g2-tma all slower; bench r10-ws-l-steal-disp predicted 0.605"]
  - id: H7
    statement: "The harness's zero-fill flush leaves dirty lines in L2; their write-back during our kernel costs ~0.5 us at S and ~4 us (about 30-40 MB) at M/L, and the cost lands on reads, not writes."
    status: supported
    evidence: ["probe r10: in-kernel span after zero flush vs read-only flush: 12.6 MB 3.07 vs 2.56 us, 25 MB 4.10 vs 3.58, 100 MB 16.6 vs 12.3", "probe r10: read-only kernels +7..11 us at 100-400 MB after a zero flush, copy +5..6 us, write-only +-1 us", "no legitimate kernel-side mechanism found; load hints do not avoid it (H8)"]
  - id: H8
    statement: "evict_first on the streamed loads makes later allocations evict our own clean lines instead of dirty flush lines, cutting H7's write-backs."
    status: refuted
    evidence: ["probe r10: read-only kernel -13 to -15% with evict_first loads, but the full kernel with evict_last stores gains only 1-2% at 256-1024 tokens (8% once at 586), 0 to +1.5% above; .cs/.lu/no_allocate/nc+EF all within +-2%", "bench r10-ldef-real-stel-disp: S -0.8%, M -1.2%, L +0.3% vs r6; kept as a ~1% tweak for B*S <= 1024, not a lever", "r10: the CUTLASS EVICT_FIRST constant passed as a kernel argument is a no-op; the qualifier or createpolicy form works"]
  - id: H9
    statement: "The portal (SM 1500 MHz) penalises SM-bound designs more than the bench (~1940 MHz): kernels that lag a plain copy on the bench lose more on the portal; the clock-bound share is ~2.5 us of 4.2 us at 12.6 MB and ~6 us of 14.5 us at 100 MB."
    status: supported
    evidence: ["emulator fit on 18 kernels: about +5% portal time per 20% copy lag at L", "r8/r9 bench -1% -> portal +3-4%", "portal/bench ratios: c1 S +15%, M +9%, L 0% (INFERRED split of the clock-bound share)", "untested on the portal: the 48-register R=3 path of r10-rtok-m-disp"]
  - id: H10
    statement: "At 12.6 MB the one-shot kernel already equals a plain copy; the remaining S time is ~1 us outside the CTAs (clock-bound), one DRAM round trip (~0.7 us), streaming at ~8 TB/s marginal, and H7's write-backs."
    status: supported
    evidence: ["probe r10 timelines: all 768 CTAs start within 0.06 us, first load returns at 0.67 us, last at 2.9 us; in-kernel span 3.1-3.3 us vs 4.16 us CUPTI", "probe: copy 4.4 us vs r5 4.2 us (bench); empty 768-CTA grid 1.57 us", "portal marginal 12.6->25 MB 8.4 TB/s, 25->57.6 MB 8.1 TB/s; fixed ~3.3 us on the portal"]
  - id: H11
    statement: "The ~7 TB/s read+write ceiling at L is DRAM-side: read-only bandwidth (5.6-6.3 TB/s clean) does not rise with more active SMs, more in-flight bytes per SM, or TMA, so no SM-side change can lift L."
    status: open
    evidence: ["probe r10: read-only 402 MB 71.9 us (64 us clean, 5.6-6.3 TB/s); write-only 54.6 us (7.4 TB/s); copy 6.7-6.95 TB/s", "probe r10: 64-128 KB in flight per SM (R=2..4) and bulk loads do not beat r6 at L", "untested: bandwidth vs number of active SMs (smid gating)"]
  - id: H12
    statement: "Each B200 die homes half the HBM/L2; the address-to-partition hash is visible in low address bits, and a CTA reading only lines homed on its own die gets higher bandwidth or lower latency than mixed traffic, so a die-affine tile assignment lifts M/L."
    status: open
    evidence: ["chipsandcheese-b200: L2 21 TB/s local vs 16.8 TB/s cross-die, far-partition latency 'dramatically' higher", "no measurement yet; dispatch pattern of blockIdx across dies unknown (r10 smid trace shows CTAs spread over GPCs)"]
  - id: H13
    statement: "FlashInfer's sm_100 layout (4 lanes per row, 32 floats per thread, 8 rows per warp instruction) beats the half-warp-per-row layout on B200, especially at 4-60 MB."
    status: open
    evidence: ["fi-pr5305: 1.14x at M=8192, 1.20x at M=32768 rows (bf16, 4-16 MB traffic) on B200; 'adding threads per CTA instead was monotonically worse'", "counter-evidence: r10 R=2..4 contiguous rows per thread (64-128 KB in flight per SM) gave no gain, but kept 16 lanes per row; the strided layout itself is untested"]
  - id: H14
    statement: "Part of the ~1 us outside the CTAs at S depends on launch attributes we control (grid/block shape, parameter bytes, shared-memory carve-out, launch API, cluster dims)."
    status: open
    evidence: ["probe r10: empty 768-CTA grid spans 1.57 us CUPTI; r6 CUPTI 4.16 vs in-kernel 3.1-3.3 us", "CAKE: 2.53 us for a 2.1 MB kernel, so about 1 us of our fixed cost is above the public floor"]
  - id: H15
    statement: "L2 protects at most ~25-30 MB of evict_last lines (and holds a similar amount of dirty flush lines), a hardware cap; the store-side end state cannot be improved further."
    status: parked
    evidence: ["probe r10: evict_last saves 3-4 us (~25-30 MB) at every size from 1024 to 8192 tokens; last-48/80/112 MB-only and fractional 0.5/0.75 variants are within +-1% or worse", "probe r10: H7's cost is also ~30-40 MB worth at M/L"]
```

```json tasks
[
  {"id": "E1", "hypothesis": "H12", "title": "Die affinity and the read ceiling: is L DRAM-bound or addressable?", "operation": "new_design", "parents": ["c1-control-r6"],
   "band": "L",
   "instructions": "Probe-only unless step 4 succeeds. (1) Read-only grid-stride kernel (LDG.256, 8 CTAs/SM resident) over 400 MB after a zero_+read flush (clean L2). Record %smid per CTA. Repeat with CTAs on odd smids (and separately smid>=74) exiting at once, the rest covering all bytes: report bandwidth at 148 vs 74 active SMs. If 74 SMs reach within 10% of 148, the ceiling is DRAM/L2-side (H11 supported). (2) Latency probe: one warp pinned by exiting all CTAs except one on a chosen smid; dependent-load chase over 128 B lines inside one 2 MB-aligned region, varying address bits 7..20 one at a time and in XOR pairs; look for a bimodal near/far latency and derive the bit function that predicts the far partition. Repeat from an SM on the other die (smid>=74 or by the parity found in step 1). (3) If a hash is found: read-only kernel where each CTA reads only near-homed lines vs only far-homed vs mixed, same bytes; report bandwidth and latency. (4) If near-only is >=3% faster: inspect the smid trace of the one-shot grid for a deterministic blockIdx-to-die pattern; if one exists, build a variant of c1-control-r6 whose blockIdx-to-tile mapping gives each CTA tiles homed on its die (every blockIdx still maps to exactly one tile, so correctness never depends on dispatch). run_tests twice interleaved with c1-control-r6. Record all findings including negative ones.",
   "success": "Near-only reads >=5% faster than mixed at 400 MB, or a die-affine kernel that is >=3% faster at L in two interleaved run_tests.",
   "refuted_if": "No bimodal latency over address bits 7..20 from either die, or near-only vs mixed within 2%, or 74 active SMs reach >=90% of the 148-SM read bandwidth with no affinity effect (then close H12 and mark H11 supported).",
   "model": "opus"},

  {"id": "E2", "hypothesis": "H13", "title": "FlashInfer sm_100 layout: 4 lanes per row, 128 B per thread", "operation": "structural_mutation", "parents": ["c1-control-r6"],
   "band": "all",
   "instructions": "CUDA C++ port of the FlashInfer QKRMSNorm sm_100 tiling with our memory path: 4 lanes per row, 32 rows per 128-thread CTA (16 KB per CTA), thread t of a row owns columns [8t,8t+8)+32k for k=0..3 so each ld.global.v8.f32 warp instruction covers 128 B of each of 8 rows; 4 loads per thread issued up front, then the sum of squares over 32 floats, 2 shuffles (xor 1, 2), rsqrtf, weights via 4 x ld.global.nc.v8.f32, stores st.global.L2::cache_hint v8 with createpolicy evict_last (as r6). Reuse the x registers for y; target <=80 registers (check spills with the sm_100a compile); if above, try 8 lanes per row (2 loads per thread, 64 B contiguous per lane) as the second variant. Build both as one dispatcher so run_tests covers all 16 workloads. Probe interleaved against c1-control-r6 at 128, 256, 586, 1024, 2048 and 8192 tokens, 5 repeats each, under harness timing. Also record copy lag per variant (plain copy with the same layout and EL stores). If a variant wins at some sizes only, make a size dispatcher with c1 elsewhere and run run_tests twice.",
   "success": ">=3% faster than c1-control-r6 at any band in repeated interleaved probes, and predicted portal score >=0.615.",
   "refuted_if": "Both lane layouts are within +-1% of c1-control-r6 at every size, or slower; then record that the FlashInfer gain does not transfer to fp32 direct loads and close H13.",
   "model": "opus"},

  {"id": "E3", "hypothesis": "H14", "title": "Which launch attributes move the clock-bound ~1 us outside the CTAs?", "operation": "new_design", "parents": ["c1-control-r6"],
   "band": "S",
   "instructions": "Measurement only. Under harness (CUPTI) timing on the rented B200, 5 interleaved repeats each: (1) empty kernel: grid 1, 148, 768, 1536, 12288 CTAs x block 64, 128, 256, 512, 1024 threads. (2) empty kernel at 768x256 with: 7 scalar/pointer params vs a single pointer to a device-side struct (struct must be rebuilt each call without a torch kernel; if that needs a memcpy, measure it but note it would be a second GPU activity) vs 1 param; cudaFuncAttributePreferredSharedMemoryCarveout 0 and 100; dynamic smem 0 vs 1 KB; launch via <<<>>> vs cudaLaunchKernelEx; cluster dims 1 vs 2. (3) c1-control-r6 at 128 tokens: in-kernel globaltimer span (first CTA start, last CTA end) relative to the CUPTI start and end, to split the ~1 us into before-first-CTA and after-last-CTA parts; repeat with the store policy plain to see if the after-last-CTA part is store drain. (4) Apply any attribute that cut the empty span by >=0.1 us to c1-control-r6, run_tests twice interleaved. Record a table of all spans.",
   "success": "An attribute change cuts the 12.6 MB CUPTI time of c1-control-r6 by >=0.2 us (about 5%) in 5 interleaved repeats, with no change at M/L.",
   "refuted_if": "All attribute variants are within +-0.1 us of the baseline empty span and the after-last-CTA part is store drain or fixed; then mark H14 refuted and treat the remaining S fixed cost as hardware.",
   "model": "sonnet"},

  {"id": "E4", "hypothesis": "H9", "title": "Consolidate r10's kept effects into one portal candidate and measure its copy lag", "operation": "crossover", "parents": ["r10-ldef-real-stel-disp", "r10-rtok-m-disp"],
   "band": "all",
   "instructions": "Build r11-consol-disp from the two parents, two-file layout: x loads with the ld.global.L2::evict_first qualifier for B*S <= 1024 (from r10-ldef-real-stel-disp); the R=3 rows-per-thread path (one weight load per 3 rows, 48 registers) only for B*S == 1024 (the three 100.7 MB workloads, from r10-rtok-m-disp); c1-control-r6's path everywhere else. Keep evict_last stores at every size. Confirm sm_100a registers and no spills for each path. Run run_tests 3 times interleaved with c1-control-r6 and report per-workload deltas with their spread. Then measure copy lag for each path: compare each path at its sizes with a plain copy kernel of the same CTA shape, width and EL stores, since H9 says lag predicts the portal penalty. If the R=3 path lags the copy by more than 5%, drop it and keep only the evict_first change. Report the predicted portal score and recommend the submission pair (this candidate plus r10-ldef-real-stel-disp) to the user.",
   "success": "Predicted portal score >=0.613 +-0.009 with every path at most 2% slower than its plain copy, and the 1024-token workloads >=2% faster than c1-control-r6 in all three runs.",
   "refuted_if": "The 1024-token gain does not reproduce in 3 interleaved runs (within +-1%), or the R=3 path lags its copy by >5%; then ship only the evict_first change and note that R=3 was bench noise.",
   "model": "sonnet"}
]
```