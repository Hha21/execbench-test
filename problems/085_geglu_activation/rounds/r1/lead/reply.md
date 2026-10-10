### Assessment

Portal r0 scored 0.6742 (top 5 needs 0.680). Portal times are the rented times plus a constant 0.5–0.9 µs at every size (S 4.1 vs 3.6, L 72.2 vs 71.7 µs), so the 1500 MHz lock inflates only the SM-clock-bound fixed cost; the GELU maths stays hidden (H5 refuted). Derived Tsol sits at 0.52–0.65× the 8 TB/s byte floor (0.9 µs at 7.9 MB, 33 µs at 503 MB): unreachable, as on #38 (H9 supported). Tb is erratic per workload and only 5–10% above us at L, so L scores are 0.52–0.55 while S scores are 0.80.

Score model: 0.5 µs off the fixed cost at every size is worth about +0.011; 0.5 TB/s of marginal bandwidth about +0.019; 10% at L alone +0.020.

The biggest unknown is the read side. Marginal bandwidth between 63 and 126 MB is 42 MB of reads in 7.8 µs, which is 5.4 TB/s, the same cap the read-only probe hit at 503 MB. Read+write reaches 7.0 only because evict_last keeps outputs in L2. Whether that cap is the flush's dirty write-backs (fixable by evict_first loads that recycle their own lines) or a bytes-in-flight limit (fixable by a bulk ring) decides where M/L headroom is. This round tests both directly on the bench, plus an S-band cost ladder to see whether any of the ~3 µs fixed cost above the byte floor is removable at 1500 MHz.

```yaml ledger
# Hypothesis ledger for #85 geglu_activation. Seeded 10 October 2026 by the research phase; r1 update 10 October 2026.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.

constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only at <= 300 tokens (H16): clustered large grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot (r13, H21).'
- 'L2 prefetch (cp.async.bulk.prefetch.L2) only on the current call''s own inputs, inside our kernel.'
hypotheses:
- id: H1
  statement: evict_last on output stores cuts in-window write-backs of the flush's dirty L2 lines, so M and L sizes gain 6-13%.
  status: supported
  evidence:
  - 'probe r0 (rented B200): EL vs plain stores: 63 MB 10.98 vs 11.97 us, 126 MB 18.74 vs 21.64, 252 MB 35.35 vs 40.00, 503 MB 71.55 vs 76.07; S unchanged'
  - 'same mechanism as #38 H1'
  - 'portal r0 (0.6742): M/L times equal rented times + 0.5-0.9 us, so the EL gain carries to the portal; no portal pair yet'
- id: H2
  statement: One 8-float chunk per thread in small one-shot CTAs is faster than fatter threads; the one-shot grid needs no persistence.
  status: supported
  evidence:
  - 'probe r0: 2 chunks/thread +1..5%, 4 chunks/thread +3..10% (82 regs); block 128/256/512 within 1%'
  - 'caveat: the fat-thread variants were one-shot with high register counts; a pipelined persistent variant with all loads issued first is untested (H12)'
- id: H3
  statement: The sigmoid form (a*b/(1+exp(-2u)), __expf + __fdividef) is both within tolerance and faster than tanhf.
  status: supported
  evidence:
  - 'probe r0: max abs err 1.9e-6, zero elements out of tolerance at 128/2048 rows; tanhf variant 3-10% slower'
  - 'portal r0: 16/16 pass'
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
  - 'portal r0 vs rented: 503 MB 72.2 vs 71.7 us (+0.7%), 252 MB 36.0 vs 35.1, 126 MB 19.5 vs 18.6, 7.9 MB 4.1 vs 3.6: the portal adds a constant 0.5-0.9 us, not a per-byte cost, so compute is hidden at 1500 MHz. A 1-MUFU variant is not worth a slot.'
- id: H6
  statement: The S-band fixed cost (~2.7 us above the byte floor at 7.9 MB on the rented B200, ~3.1 us on the portal) can be reduced by a dedicated small-size path.
  status: open
  evidence:
  - 'probe r0: empty kernel 1.73 us; read-only 3.52 us; full 3.6-3.8 us at 7.9 MB'
  - 'portal r0: 4.1 us at 7.9 MB, 4.2 at 8.05 MB, 6.8 at 26 MB, 7.7 at 31 MB; Tsol ~0.9 us; S scores 0.79-0.82'
  - 'score model: 0.5 us off every size = +0.011; 10% at S alone = +0.007'
  - '#38 experience: launch params moved the empty span < 0.03 us; most of the remainder is DRAM latency + dirty flush cost'
  - 'r1 E3 builds the cost ladder (dispatch, first DRAM round trip, streaming, exit) and tries lean one-wave S paths'
- id: H7
  statement: 'The harness''s dirty L2 (zero-fill flush) costs reads: a read-only kernel reaches only 5.3 TB/s at 503 MB while read+write reaches 7.0 TB/s.'
  status: supported
  evidence:
  - 'probe r0: read-only 62.95 us vs full 71.6 us at 503 MB (335 MB read)'
  - 'rented r0 marginal bandwidth: 63->126 MB adds 42 MB of reads (outputs retained in L2 by EL) in 7.8 us = 5.4 TB/s, the same cap; 252->503 MB adds 168 MB reads + 84 MB write-back in 36.6 us = 6.9 TB/s'
  - 'mechanism unproven: could be flush write-backs triggered by our read allocations (H11) or a bytes-in-flight limit on reads (H12); r1 E1 separates them with clean-flush vs dirty-flush timing and DRAM byte counters'
- id: H8
  statement: Load hints (.nc, evict_first, L1::no_allocate) do not matter for this once-read stream.
  status: open
  evidence:
  - 'probe r0: ld.global.nc vs plain ld.global: identical at every size'
  - 'evict_first on loads untested on B200 for this problem; #38 H2 (A100): hurt L by ~2%; CAKE kept EF on streamed loads in paired B200 tests'
  - 'r1 E1 tests EF loads via L2::cache_hint together with EL stores'
- id: H9
  statement: Tsol for this problem sits below the bytes/8 TB/s floor, so score 1.0 is unreachable and only relative speed-ups matter.
  status: supported
  evidence:
  - 'portal r0 derived Tsol (from S, t, Tb): 0.9 us at 7.9 MB (floor8 0.98), 2.1 at 26 MB (3.24), 2.5 at 31 MB (3.93), 4.5 at 63 MB (7.86), 29 at 441 MB (55.2), 33 at 503 MB (62.9): 0.52-0.92x the byte floor'
  - 'score derivative: ~0.05 per us at 7.9 MB, ~0.0064 per us at 503 MB (per workload)'
- id: H10
  statement: The portal (1500 MHz) adds a constant ~0.5-0.9 us to every workload relative to the rented B200 (1965 MHz); this is SM-clock-bound fixed cost (launch, dispatch, address chain, exit), not bandwidth.
  status: supported
  evidence:
  - 'portal vs rented r0: +0.5 us at 7.9 MB, +0.7 at 31 MB, +1.0 at 63 MB, +0.9 at 126 MB, +0.9 at 252 MB, +0.5 at 503 MB'
  - 'implies ~1.6 us of the rented 1.73 us empty-kernel span scales with SM clock; a leaner S path (fewer instructions before the first load, after the last store) is the only kernel-side lever (E3)'
- id: H11
  statement: evict_first on the input loads (createpolicy + ld.global.L2::cache_hint) makes later reads evict our own clean read lines instead of the flush's dirty lines, removing in-window write-backs and lifting the 5.4 TB/s read cap at M/L.
  status: open
  evidence:
  - 'no measurement yet; derived from the H1 mechanism and the M-band marginal bandwidth'
  - 'r1 E1: dirty vs clean flush, load/store policy matrix, DRAM byte counters if ncu is permitted'
- id: H12
  statement: The read cap is a bytes-in-flight limit at the read side (one-shot CTAs ramping/draining), and a persistent 1-D bulk ring with >=128 KB in flight per SM and few instructions lifts L by >=3%.
  status: open
  evidence:
  - 'r0 holds ~131 KB/SM in flight at full occupancy in theory, but each CTA lives one round trip; public 1:1 streams cap at 7.05-7.1 TB/s'
  - 'CAKE: pipelined persistent grids 0.96-1.00x one-shot at large sizes (#38 context), so expect at most a few %'
  - 'r1 E2 tests it'
- id: H13
  statement: Tb is erratic per workload (same 126 MB: 31.2 vs 22.6 us; 63 MB Tb 45.3 > 126 MB Tb 22.6) and only 5-10% above us at L, so L scores stay near 0.52-0.55 unless L gains >=10%; S workloads already score ~0.80.
  status: supported
  evidence:
  - 'portal r0 page: Tb 13.4/13.4/22.5/25.9/26.0 (S), 45.3/45.3/31.2/22.6/23.6 (M), 40.6/40.8/40.8/47.5/67.5/75.8 (L)'
  - 'score model: 10% at L = +0.020, M = +0.011, S = +0.007'
```

```json tasks
[{"id": "E1", "hypothesis": "H11", "title": "Dirty-L2 traffic accounting and the load/store policy matrix", "operation": "knob_mutation", "parents": ["r0-geglu-ldg256-2d-stel"], "band": "all",
  "instructions": "1) Write a bench probe that reproduces timing.py's pre-call sequence (cudaCtxResetPersistingL2Cache, zero_() of a 252 MB buffer, shifted input copy, output zero) and times only our kernel. Add two alternative pre-conditions: CLEAN flush (the same zero_() followed by a read-only reduction over a different 252 MB buffer, so L2 holds clean lines) and NONE (no flush, warm L2 with unrelated data). 2) Build r0 variants as compile-time switches: loads in {ld.global.nc (current), EF via createpolicy.fractional.L2::evict_first + ld.global.L2::cache_hint.v8.f32, EF + L1::no_allocate}; stores in {EL (current), plain, EF}; plus read-only and write-only kernels. 3) Time every variant at 7.9, 31, 63, 126, 252 and 503 MB under DIRTY and CLEAN, interleaved, at least 5 repeats, report medians. 4) If ncu is installed and counters are permitted, collect dram__bytes_read.sum and dram__bytes_write.sum for r0 and for the EF-load variant under the DIRTY flush (--cache-control none --clock-control none --replay-mode application); report ERR_NVGPUCTRPERM if blocked. 5) Fit in-window DRAM bytes = reads + output write-back + flush write-back per size and state the flush write-back D in MB per size with its uncertainty. 6) For any policy that beats r0 by >=3% at M or L (and >=0 at S) run run_tests (16/16) and archive it as a candidate with its findings.", "success": "A load/store policy beating r0 by >=3% at 126-503 MB on the bench with 16/16 pass, or a traffic model giving D per size to within 10 MB together with the DIRTY-vs-CLEAN time delta.", "refuted_if": "DIRTY and CLEAN times differ by <2% at every size and every load policy ties r0 within 1%: then H11 is refuted, H8 is supported, and the read cap is not the flush's write-backs.", "model": "opus"},
 {"id": "E2", "hypothesis": "H12", "title": "Read-side throughput at L: persistent 1-D bulk ring vs plain 256-bit one-shot", "operation": "structural_mutation", "parents": ["r0-geglu-ldg256-2d-stel"], "band": "L",
  "instructions": "1) First measure read-only ladders under the harness flush at 252 and 503 MB: r0's loads (1 chunk per thread, one-shot), 2 and 4 chunks per thread with every load issued before the first use and __launch_bounds__(128, 8) to keep <=64 regs, and a persistent grid of 148 x k CTAs looping over (row, x) tiles with the next tile's loads issued before the current compute. 2) Build the bulk-ring kernel (mem:bulk1d, st:direct): persistent 148 x k CTAs (k 4-8 by smem), each stage one 4 KB gate segment plus the matching 4 KB linear segment copied with cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes (optionally .L2::cache_hint EF), STAGES 4-8 so >=128 KB per SM is in flight, consumers read smem, compute GEGLU, and store 8 floats per thread with the existing st.global.L2::cache_hint EL 256-bit store; dynamic smem above 48 KB. Keep the 2-D one-shot r0 kernel for rows <= 2164 via dispatch on rows only (the ring is for L). 3) Compare full kernels at 126, 252 and 503 MB, interleaved, >=5 repeats; also check that the dispatch keeps S/M at r0's times. 4) run_tests on the best variant; record regs, smem, CTAs/SM and bytes in flight per SM in the card.", "success": ">=3% faster than r0 at 252-503 MB on the bench with 16/16 pass and no S/M regression.", "refuted_if": "Every read-side variant (fat threads with loads issued first, pipelined persistent, bulk ring with >=128 KB/SM in flight) ties r0 within 1% at L: then in-flight bytes do not limit reads and H12 is refuted; L headroom rests on H11.", "model": "opus"},
 {"id": "E3", "hypothesis": "H6", "title": "S-band cost ladder and a lean one-wave small-size path", "operation": "specialisation", "parents": ["r0-geglu-ldg256-2d-stel"], "band": "S",
  "instructions": "1) Under the harness flush at 7.9 MB (and 31 MB), measure CUPTI spans for: an empty kernel with 1, 640 and 1280 CTAs; one CTA issuing a single 32 B load then a store (DRAM round trip as seen in the span); 148 CTAs each loading 4 KB; r0 read-only; r0 full. Write the ladder (launch/exit, CTA dispatch, first DRAM round trip, streaming, store drain) in the findings. 2) Build S-path variants selected by rows <= 512 only: (a) 64-thread CTAs (1280 CTAs); (b) 512-thread CTAs with 1 chunk per thread; (c) a one-wave grid of 148 x 4 CTAs (256 threads) where each thread issues ALL its loads (2 chunks, <=48 regs) before any compute; (d) the first-wave CTAs issue cp.async.bulk.prefetch.L2 for the whole input in 16-64 KB pieces (our own input only) before their normal loads, to test whether DRAM request issue is limited by CTA ramp; (e) a minimal-prologue variant: 32-bit address math, no createpolicy (plain stores) at S since EL gains 0 there. 3) Interleave >=7 repeats per variant, report medians; keep r0 for rows > 512; run_tests for 16/16. 4) State, for each variant, the predicted portal effect given H10 (the portal adds ~0.5 us of SM-clock-bound cost).", "success": "An S path >=5% (>=0.2 us) faster than r0 at 7.9-31 MB on the bench with 16/16 pass and no M/L change.", "refuted_if": "No variant beats r0 by >2% at any S size: then the S time is the empty-kernel span plus one DRAM round trip plus streaming, H6 is parked, and S-band effort stops.", "model": "sonnet"}]
```