Evidence is in hand. Two things stand out before writing the round: the portal per-workload scores imply the hidden SOL time is roughly one tensor's bytes at 8 TB/s (so L scores are compressed and 4% at L is worth about +0.013 per L workload), and r1's E3 decomposition shows M-band time is already DRAM-bound once the forced dirty write-backs are counted. That leaves the launch path and the S-band memory round trip as the only levers below 1.2M elements.

### Assessment

Score is 0.5166; top 5 needs +0.021. Per-workload S sits at 0.49-0.52 for all ten workloads up to 786K, where we now tie Tb (portal 2.3-4.8 µs vs Tb 2.2-5.0). With Tsol ≈ 0 there, S = Tb/(t+Tb), so 0.2 µs off those ten is worth about +0.015; at L (Tsol ≈ 4n/8 TB/s, inferred from the portal page) a 4% cut is worth about +0.005.

What we know: r1 E3 settled the memory physics. In-window DRAM traffic must include one flush-line write-back per output line, plus one per L2 set touched by reads (≈ read bytes at M, ≈ 8 MB at L). Counting that, M runs at ≈ 8 TB/s and L at ≈ 7.5 TB/s (92%). Fractional policies, persistent pipelines and CTA shaping are dead ends. So M/S can only move through the fixed cost: the 1.6 µs empty-kernel span, of which the cluster launch recovered 0.15-0.2 µs and clusters of 4/8 another 0.04 at S. The +0.4 µs post-memset queueing penalty is off limits to exploit, but we do not yet know whether it is a launch-path/timestamp effect or DRAM write drain; that decides whether any kernel-side S work can pay.

This round: E1 (opus, exploratory) maps the launch path with an empty kernel, then sweeps legitimate launch attributes, SM carveout, cluster 4/8 and CTA shape at ≤ 786K. E2 (opus) decomposes the S-band memory round trip under the dirty flush and tries store types and one-shot bulk copies, the operator's "bulk at tiny sizes" item. E3 (sonnet) replicates 256-bit with the exact EF/EL policy forms and two-stream ordering at L; a tie closes L for good.

```yaml ledger
# Hypothesis ledger for #084 silu_activation_backward, seeded 9 October 2026; updated r2 (10 October 2026) by the research lead.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset: a kernel that arrives ~60 us late, after the 73 us flush, measures ~0.4 us faster at small sizes on the same code (#84 r1 E1). It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Launch attributes (cluster dims, cooperative, scheduling-policy preference, carveout) are legitimate only if they change how our own kernel is dispatched; none may alter ordering or overlap with harness work (no PDL, no priority games, same stream).'
- 'Clusters only on one-wave grids (<= 1184 CTAs of 256 threads, i.e. n <= ~1.2M elements): clustered large grids lost 2-5% on the #38 portal and seemed to slow later plain launches (carried over from #38 H16).'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot.'
- 'torch.compile-based or inductor-derived Triton solutions are legitimate PyTorch/Triton code, but set dynamic=False and raise torch._dynamo cache_size_limit above 16, or extract the generated Triton kernel and launch it directly; never let a recompile or eager fallback happen inside the timed loop (CUPTI sequence must repeat).'
hypotheses:
- id: H1
  statement: L2 evict_last on the output stores cuts in-window dirty write-backs, gaining 5-7% at L and 3% at 12.6 MB.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB plain 43.0 us vs EL stores 40.2; 134 MB 23.65 vs 21.5; 33.5 MB 7.83 vs 7.41; 12.6 MB 4.68 vs 4.51; S ties'
  - 'r1 E3: store evict_unchanged 43.28 and evict_normal 43.19 at 268 MB (+11%): without EL the outputs are written back in-window on top of displacing flush lines'
- id: H2
  statement: L2 evict_first on the three input streams adds a further 3-4% at L on top of EL stores, because after each L2 set holds one of our EF lines our own clean lines become the victims instead of the flush's dirty lines.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB EL 40.2 -> EF+EL 38.8; 134 MB 21.5 -> 20.1; 83.9 MB 14.8 -> 13.8; 33.5 MB 7.41 -> 7.10; S ties'
  - 'probe_b200 r0: ld.global.cs, nc+EF and L1::no_allocate+EF all equal EF (within 0.1 us); L2::256B prefetch hint = no gain'
  - 'portal r0: our L marginal 7.3 TB/s vs Tb 6.8 TB/s: the hints are our L lead'
  - 'r1 E3: reads-only EF at 268 MB still pays ~0.85 us dirty cost (33.23 dirty vs 32.39 clean) = the first miss per set (~8 MB); EF cannot remove that'
  - 'r1 V3 (no hints) ties at S, loses 10% at 2.1M and 16.8M: keep hints at every size'
- id: H3
  statement: 256-bit loads/stores do not beat 128-bit for this stream, and cost +0.8 us at 131 elements.
  status: supported
  evidence:
  - 'probe_b200 r0: k8 vs k4 tie at 12.6-268 MB (within 1%); at 131 elements k8 3.06 us vs k4 2.29, repeatable across block sizes; cause unknown'
  - 'caveat: the r0 probe did not use the exact L2::cache_hint EF/EL policy form on the 256-bit path; r2 E3 replicates at L only (dispatch keeps 128-bit at <= 1.2M, so the 131-element penalty is moot)'
- id: H4
  statement: The S band (<= 0.3 MB) is at the launch floor; no kernel-side change can gain more than ~0.2 us there.
  status: open
  evidence:
  - 'probe_b200 r0 (unlocked): empty kernel 1.57-1.66 us; float4/scalar kernels 2.2-2.4 us at 131-16384 for every block size, unroll and hint'
  - 'r1: cluster-2 launch 1.89 us at 131 on the bench, i.e. ~0.3 us above the (non-cluster) empty kernel; portal r1 2.3-2.5 us = Tb 2.2-2.5, scores 0.485-0.508'
  - 'r1 V1: pre-load instruction count irrelevant (ties r0 within 0.04 us)'
  - 'to test: r2 E1 (launch path below cluster-2) and r2 E2 (memory round trip at S); if both find <= 0.03 us, mark supported and stop S work'
- id: H5
  statement: The hidden baseline Tb is one fused torch.compile kernel plus the dynamo launch delay (which lands it after the flush memset); at L it is an inductor kernel without cache hints at 6.8 TB/s marginal.
  status: supported
  evidence:
  - 'portal r0: Tb = 2.4/2.5/2.2/2.4/2.5/2.5/3.0/3.4/4.8/5.0/8.5/13.7/16.9/24.1/29.0/43.1 us; eager is ~6x slower'
  - 'r1 bench: inductor-equivalent Triton under harness timing 4.82/7.87/23.23/42.52 us at 786K/2.1M/8.4M/16.8M vs Tb 5.0/8.5/24.1/43.1 (+4-8% portal inflation, same as ours)'
  - 'r1 bench: torch.compile callable 1.58-1.95 us at 131-16384 (dynamo delay lets it arrive after the memset); the same kernel launched directly 2.03-2.08 = r0'
- id: H6
  statement: More bytes in flight per thread (unroll 2-4, 512-1024 threads) raises L-band bandwidth.
  status: refuted
  evidence:
  - 'probe_b200 r0: block 128/256/512/1024 x unroll 1/2/4 all within +-1% at 67-268 MB over 3 repeats'
  - 'r1 E3: U2/U4/128-thread/148xk grids tie at <= 12.6 MB and are 1-5% slower at 33.6 MB; persistent register double-buffer +7-10% at L'
- id: H7
  statement: In-window DRAM traffic cannot be less than reads + output bytes (each output line displaces a dirty flush line, or is written back itself) + ~8 MB (one dirty eviction per L2 set from reads). r0/r1 move that traffic at ~7.5 TB/s (92% of 8.18), so L headroom is <= 5-8% and lies in DRAM read/write mix efficiency, not in cache policy.
  status: supported
  evidence:
  - 'r1 E3 Part A (r0, dirty / clean sweep): 268 MB 38.41/31.71; 134 MB 20.32/16.30; 33.6 MB 7.52/5.83 -> in-window dirty cost 5.9/2.6/1.05 us ~= output bytes / 8 TB/s'
  - 'r1 E3: reads-only dirty/clean 33.23/32.39 at 268 MB (reads add <= 1 us); writes-only EL unaffected by flush type (10.19 vs 10.12)'
  - 'arithmetic: 201 MB reads + 67 MB write-backs + 8 MB in 38.4 - 1.6 us = 7.5 TB/s; floor at 8.18 TB/s + 1.6 us fixed = 35.4 us vs 38.4 bench / 39.8 portal'
  - 'r1 E3 dead ends: fractional EF/EL policies, evict_unchanged, persistent store-before-load pipelines all 0 to +11%'
- id: H8
  statement: Per-size dispatch is unnecessary; one config (256 threads, one float4 per thread, EF+EL) is within noise of the best at every size.
  status: refuted
  evidence:
  - 'r1 portal: cluster-2 dispatch at <= 1184 CTAs gained S -19% / M -7% (0.4941 -> 0.5166); the kernel body is unchanged, so dispatch on n is required for the launch path'
  - 'still true for the body: block size, unroll and hints tie at every size (r0, r1 E3)'
- id: H9
  statement: TMA/bulk rings, persistent grids, clusters or weight-style prefetch could help this problem.
  status: parked
  evidence:
  - '#38 evidence: TMA +1-1.5 us fixed, persistent slower at every size, clusters on large grids lost 2-5%; r1 E3 confirmed persistent pipelines +7-10% here'
  - 'exception: cluster launch on ONE-WAVE grids saves 0.12-0.20 us (H12, supported)'
  - 'last check: a one-shot 1-D bulk copy at n <= 16384 (no ring, one mbarrier) is tested in r2 E2; if it loses, bulk/TMA is closed for this problem'
- id: H10
  statement: The portal-vs-bench inflation (+0.4 us at S, +5% at L) is the 1500 MHz SM lock scaling the clock-bound part of the fixed cost (launch, CTA dispatch, store drain), so every launch-path saving on the bench is worth ~1.2x on the portal.
  status: open
  evidence:
  - 'portal r1 vs bench r1 band geomeans: S 2.4/2.0 (1.20x), M 3.5/3.1 (1.13x), L 17.6/16.8 (1.05x); r0 ratios the same'
  - 'r1: cluster-2 gain transferred 1:1 in percent (S bench -19.4% / portal -19.3%; M -7.0% / -6.9%), consistent with a scaled fixed cost'
  - 'untestable on the rented B200: nvidia-smi -lgc/-ac refused (no permission); only portal pairs can test it; keep for interpretation'
- id: H11
  statement: Tb's S-band advantage over r0 comes from kernel structure (inductor-style Triton kernel), so a Triton port would score ~2.4 us at S on the portal.
  status: refuted
  evidence:
  - 'r1 bench: the inductor kernel launched directly (CachingAutotuner.run or plain Triton) = r0 (2.03-2.08 at 131); its advantage exists only through the compiled path''s launch delay (constraint 1)'
  - 'r1 Triton b512w4 port ties r0 within noise at every band (run_tests)'
  - 'portal r1: we now tie Tb at S without any launch-delay effect (2.3-2.5 vs 2.2-2.5)'
- id: H12
  statement: A cluster-of-2 launch on one-wave grids (n <= ~1.2M elements) cuts 0.12-0.20 us of per-launch overhead outside the CTAs; clusters of 4/8 give the same gain plus ~0.04 us at S and lose ~0.04 us at 786K-2.1M.
  status: supported
  evidence:
  - 'r1 probe (ABAB, 7 reps): 131: 2.06 -> 1.89; 4096: 2.08 -> 1.92; 40960: 2.32 -> 2.18; 262144: 3.02 -> 2.88; 786432: 4.48 -> 4.30; 2.1M: 7.19 -> 6.98; 16.8M tie; same-padded grid without the attribute = r0'
  - 'r1 probe (11 reps): C4/C8 ~0.04 us better than C2 at S, ~0.04 us worse at 786K-2.1M; at 4.2M the gain is 0.1-0.2 us (1.7%)'
  - 'portal r1: 0.4941 -> 0.5166, S -19.3%, M -6.9%, L +0.5% (noise); no sign of cross-workload slowdown of the plain L launches'
- id: H13
  statement: At M sizes every read miss evicts a dirty flush line (reads touch <= ~1 line per L2 set), so the in-window dirty cost ~= read bytes / 8 TB/s (0.25 us at 2.6 MB, 0.9 us at 12.6 MB) and saturates near 8 MB; stores at M victimise our own EF read lines and are free. Counting this traffic, M already runs at ~8 TB/s, so only the fixed cost is reducible at M.
  status: supported
  evidence:
  - 'r1 E3 dirty/clean: 2.6 MB 2.59/2.37; 12.6 MB 4.51/3.58; reads-only 12.6 MB 4.26/3.33 (the whole cost is on the reads); writes-only unaffected'
  - 'arithmetic at 12.6 MB: 9.4 MB reads + ~9 MB write-backs + 3.1 MB stores = 22 MB in 4.3 - 1.6 us = 8.1 TB/s'
  - 'clean case 3.58 us = 1.6 fixed + ~0.8 DRAM latency + 9.4 MB / 8 TB/s: fully accounted for'
  - 'portal r0/r1: Tb shares the same M slope, as it must (its reads evict the same dirty lines)'
- id: H14
  statement: Fractional eviction policies or ordering output stores before the next tile's loads change which lines are victimised and recover part of the L-band write-back cost.
  status: refuted
  evidence:
  - 'r1 E3 (5 reps): ld EF0.25/0.5/0.75 and st EL0.5/0.75 and EL0.5+EF secondary all 0 to +5% vs r0 at 12.6-268 MB; persistent 2-tile pipelines +7-10%'
  - 'mechanism (H7): each output line must displace one dirty line whatever the policy'
- id: H15
  statement: Part of the 1.6 us empty-kernel span is front-end launch processing and SM setup that legitimate launch attributes or function attributes can shorten beyond the cluster-of-2 launch (cluster 4/8 at S, cooperative launch, cluster scheduling-policy preference, shared-memory carveout matching the memset kernel's configuration, grid shape, smaller CTAs). Expected 0.05-0.15 us at every size <= 786K (~+0.005-0.012).
  status: open
  evidence:
  - 'r1: cluster-2 saved 0.12-0.20 us although the CTAs do identical work; C4/C8 add 0.04 us at S: the launch path is tunable'
  - 'unknown: whether an SM L1/shared carveout reconfiguration happens between the torch fill kernel and ours; a mismatch costs an SM drain'
  - 'to test: r2 E1'
- id: H16
  statement: The ~0.3-0.5 us that the S-band kernel spends above an empty launch is one DRAM round trip plus the store acknowledgement under a dirty, post-memset L2, and a different access mechanism (one-shot 1-D bulk copy by one thread, write-through or no-allocate stores, bulk store, 32/64-thread CTAs) shortens it by >= 0.05 us.
  status: open
  evidence:
  - 'r1 bench: cluster-2 1.89 us at 131 vs empty kernel 1.57-1.66 (non-cluster); r0 probes show block size and hints do not move S'
  - 'untested: reads-only vs stores-only decomposition at 131-40960 (r1 E3 started at 2.6 MB), and whether the S-band memory leg depends on the flush type'
  - 'to test: r2 E2'
- id: H17
  statement: Portal Tsol is ~4n bytes / 8 TB/s (one tensor's worth), i.e. ~0 at S and 2.1-8.4 us at L. So S-band score = Tb/(t+Tb) (0.2 us off every size <= 786K ~ +0.015), and L scores are compressed (4% at L ~ +0.013 per L workload, ~+0.005 on the problem).
  status: supported
  evidence:
  - 'portal r1 page, solving S = (Tb-Tsol)/((t-Tsol)+(Tb-Tsol)): 16.8M -> Tsol 8.4 us (4n/8 TB/s = 8.4); 8.4M -> 4.3 (4.2); 10.5M -> 5.4 (5.2); 4.2M/5.2M -> 2.9/3.0 (2.1/2.6, ill-conditioned near S=0.5)'
  - 'predict_score: 10% off S/M/L alone is worth +0.0092/+0.0087/+0.0125'
- id: H18
  statement: Each L2 line has a home partition (die); a CTA that reads %smid and takes work whose lines are homed on its own die would cut cross-die traffic and latency. Requires discovering the address-to-partition hash with a latency probe; gain unknown (NV-HBI is not bandwidth-limited at 7.5 TB/s, so at most a latency/queueing effect).
  status: parked
  evidence:
  - 'chipsandcheese-b200: L2 latency is "dramatically" higher when data sits in the other die''s partition; 16.8 TB/s cross-partition vs 21 TB/s local'
  - 'not scheduled: revisit only if r2 E3 shows L is not DRAM-efficiency bound'
- id: H19
  statement: The +0.4 us penalty on a launch queued early behind the flush memset is a launch-path or timestamp effect (it appears for an empty kernel too), not DRAM write-drain contention. If so, launch-path changes (H15) can recover part of it legitimately; if the penalty needs memory traffic, nothing kernel-side at S can.
  status: open
  evidence:
  - 'r1 bench: 131 elements, CPU delay 0/20/60/150 us -> 2.11/1.89/1.73/1.73; CUPTI kernel duration 1.47 (compiled path) vs 1.79 (direct) with positive gaps after the memset, so it is not overlap'
  - 'to test: r2 E1 step 1 (empty kernel with and without delay, diagnostic only; delay never goes in a candidate)'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H15",
    "title": "Launch path below the cluster-of-2 launch: empty-kernel anatomy, launch/function attributes, cluster 4/8 and CTA shape at n <= 786432",
    "operation": "knob_mutation",
    "parents": ["r1-silu-bw-cluster2-small"],
    "band": "S",
    "instructions": "Rented B200, harness timing (dirty 2xL2 zero-fill flush, shifting pool), ABAB interleaving, >= 9 reps, report medians per size. Step 1 (diagnostic, H19): an empty kernel (1 CTA, 32 threads, no memory access) launched (a) plain, (b) __cluster_dims__(2), (c) cluster 8, (d) cudaLaunchKernelEx with cudaLaunchAttributeCooperative, and (a) again with a 150 us CPU busy-wait before launch. The delay variant is a diagnostic only and must never appear in a candidate. Report whether the +0.4 us penalty exists without memory traffic and whether the cluster gain exists on an empty kernel. Step 2: on the parent kernel body at n = 131, 2053, 4096, 16384, 40960, 163840, 262144, 786432, 2097152 compare against cluster-2: cluster 4 and 8 (grid padded to a multiple of the cluster size; pad CTAs exit on i < nvec); cooperative launch attribute (with and without cluster dims); cudaLaunchAttributeClusterSchedulingPolicyPreference spread vs load-balancing; cudaFuncAttributePreferredSharedMemoryCarveout at 0, 50 and 100 and a variant declaring 1 KB of static __shared__ (does the SM reconfigure after the torch fill kernel?); 2-D grid (gridDim.y = 2) vs 1-D; launch through cudaLaunchKernelExC with the same attributes. Step 3: combine the best launch path with CTA shape: 64/128/512-thread CTAs at <= 16384 (more, smaller CTAs across SMs with cluster 8), 128 vs 256 at 40960-786432. Step 4: if a configuration beats cluster-2 by >= 0.05 us at every size <= 786432 (or >= 0.08 us at every size <= 16384 for an S-only dispatch step), build the dispatch candidate (dispatch on n only, L path unchanged: plain launch above 1184 CTAs), run_tests all 16 workloads, write the design card with every probe result as findings, including negative ones. Check the SASS for LDG.EF/STG.EL suffixes while you are there (r1 did not).",
    "success": ">= 0.05 us faster than cluster-2 at every size <= 786432 in interleaved medians (predicted +0.005 or better), 16/16 run_tests, candidate ready for a portal slot",
    "refuted_if": "No launch or function attribute, cluster size or CTA shape beats cluster-2 by more than 0.03 us at any size; then H15 is refuted and (with E2) H4 becomes supported",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H16",
    "title": "S-band memory round trip under the dirty flush: decomposition, store types, and a one-shot 1-D bulk copy at n <= 16384",
    "operation": "structural_mutation",
    "parents": ["r1-silu-bw-cluster2-small"],
    "band": "S",
    "instructions": "Rented B200, harness timing, ABAB, >= 9 reps, sizes 131, 2053, 4096, 16384, 40960. Keep the cluster-2 launch in every variant; compare against the parent. Step 1 (decomposition): empty kernel; loads-only (three EF loads, results consumed by a store predicated on an impossible condition so they are not dead); parent (loads + EL stores); stores-only (EL). Repeat the four under a clean read-sweep flush (monkeypatch sol_execbench timing._clear_cache as r1 E3 did) and tabulate dirty vs clean per leg. Step 2 (store leg, if it costs >= 0.08 us over loads-only): st.global.wt, st.global.cs, default policy, st.global.L1::no_allocate with the EL policy, and a bulk store (registers -> static smem -> fence.proxy.async.shared::cta -> cp.async.bulk.global.shared::cta.bulk_group + commit_group + wait_group 0 by one thread) for n <= 16384. Step 3 (load leg, if it is >= 0.1 us above empty): ld.global.cv, ld.global.nc.L1::no_allocate with the EF policy, 32/64-thread CTAs, and a one-shot 1-D bulk copy: one elected thread does mbarrier.init(1), arrive.expect_tx(3 * 16 * (n >> 2)... bytes rounded to a multiple of 16 B), three cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint copies with the EF policy into static smem (<= 48 KB, so n <= 3072 per CTA; use several CTAs of 1024 floats each for 4096-16384), all threads wait on try_wait.parity, compute from smem, store EL; the <= 3 tail elements use plain loads. Inputs are 256-B aligned so the bulk source alignment holds. Step 4: adopt any variant >= 0.05 us faster at 131-40960; build the dispatch candidate (bulk/alternate path only below the size where it wins; parent path elsewhere), run_tests, design card with all findings (the decomposition table is the key deliverable even if nothing wins).",
    "success": "A variant >= 0.05 us faster than the parent at every size <= 16384 (and not slower at 40960), 16/16 run_tests; or, failing that, a complete dirty/clean decomposition table showing which leg holds the 0.3-0.5 us",
    "refuted_if": "Every store type, load type and the one-shot bulk copy are within 0.03 us of the parent, or slower; then H16 is refuted, H9 closes for this problem, and S work stops unless E1 found a launch-path gain",
    "model": "opus"
  },
  {
    "id": "E3",
    "hypothesis": "H3",
    "title": "L band: 256-bit accesses with the exact EF/EL policy forms and two-stream chunk ordering at n > 1.2M",
    "operation": "knob_mutation",
    "parents": ["r1-silu-bw-cluster2-small"],
    "band": "L",
    "instructions": "Rented B200, harness timing, ABAB, 7 reps, sizes 2097152, 4194304, 8388608, 16777216; the plain (non-cluster) launch path only. Variants against the parent (a: 128-bit, 4 floats per thread, 256 threads): (b) 8 floats per thread with ld.global.L2::cache_hint.v8.f32 and st.global.L2::cache_hint.v8.f32 (policy constants 0x12F0000000000000 for loads, 0x14F0000000000000 for stores; thread i covers floats [8i, 8i+8), 32-B aligned); (c) the same 8-float mapping with 2 x 128-bit per stream (isolates width from mapping); (d) 256-bit with the qualifier form ld.global.nc.L2::evict_first.v8.f32 and st.global.L2::evict_last.v8.f32; (e) two-stream ordering: thread handles float4 i and float4 i + nvec/2 with 128-bit (two sequential DRAM streams per wave); (f) variant b with 512 threads and __launch_bounds__(512). Confirm in the SASS that b/d/f emit 256-bit LDG/STG (look for .256 and the EF/EL or ENL2 suffixes) and report regs per variant. Tail handling for 8-float threads: the thread with i == n >> 3 covers the <= 7 remaining floats with scalar loads. Adopt only a variant that is >= 2% faster at >= 3 of the 4 sizes; dispatch it at n > 1.2M only (the <= 1.2M cluster-2 path is unchanged, so the 131-element 256-bit penalty does not apply); run_tests 16/16; design card with findings. If everything ties within +-1%, write that down as the closing result for width and ordering at L.",
    "success": ">= 2% faster than the parent at >= 3 of 4 L sizes in interleaved medians (~+0.003-0.004 predicted), 16/16 run_tests, candidate ready",
    "refuted_if": "All variants within +-1% of the parent at every L size; H3 is then closed for the policy form too, and L is parked at the 92% DRAM-efficiency ceiling (H7)",
    "model": "sonnet"
  }
]
```