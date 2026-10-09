### Assessment

Portal r0 = 0.494. The score is split three ways: the 5 S workloads sit at 0.42–0.48, the 5 M at 0.47–0.50, the 6 L at 0.52–0.55. Tb is a single fused kernel: 2.2–2.5 µs at ≤ 40960 elements, the same marginal rate as ours through M (both pay the dirty write-back), and 6.8 TB/s marginal at L against our 7.3. So Tb's kernel has a fixed cost about 0.4–0.5 µs below ours on the portal, while on the rented bench torch.compile and r0 tie at 2.1 µs. Our bench-to-portal inflation is +17% at S but +5% at L, which points at the 1500 MHz SM lock scaling the clock-bound part of the fixed cost, or at a launch-path or kernel-structure difference that only shows at the lock.

Worth: 10% off S is +0.009, off M +0.009, off L +0.013. Matching Tb at the 10 sizes ≤ 786432 is about +0.03; getting 0.2 µs under it is about +0.05, which is what top 5 (0.538) needs. L has at most +0.01 left unless the dirty write-back is larger than the fit suggests.

This round: E1 (opus) identifies the baseline and reproduces portal conditions (clock lock, mean vs median, hot vs cold inputs) with an inductor-equivalent Triton kernel and a Triton port of r0, giving a portal-ready A/B of the launch path. E2 (sonnet) applies the carried-over fixed-cost cuts (cluster-of-2 one-wave launch, fewer pre-load instructions, branch-free tail) in CUDA. E3 (opus, exploratory) measures the in-window dirty write-back cost with clean-flush probes and tests fractional policies and tile ordering at L, falling back to M-band fat-CTA designs if L is at the ceiling.

```yaml ledger
# Hypothesis ledger for #084 silu_activation_backward, seeded 9 October 2026; updated r1 (10 October 2026) by the research lead.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement,
  not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the
  timed window.'
- 'Clusters only on one-wave grids (<= 1184 CTAs of 256 threads, i.e. n <= ~1.2M elements): clustered large grids lost 2-5%
  on the #38 portal and seemed to slow later plain launches (carried over from #38 H16).'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output
  buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a
  portal slot.'
- 'torch.compile-based or inductor-derived Triton solutions are legitimate PyTorch/Triton code, but set dynamic=False and
  raise torch._dynamo cache_size_limit above 16, or extract the generated Triton kernel and launch it directly; never let
  a recompile or eager fallback happen inside the timed loop (CUPTI sequence must repeat).'
hypotheses:
- id: H1
  statement: L2 evict_last on the output stores cuts in-window dirty write-backs, gaining 5-7% at L and 3% at 12.6 MB.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB plain 43.0 us vs EL stores 40.2; 134 MB 23.65 vs 21.5; 33.5 MB 7.83 vs 7.41; 12.6 MB 4.68 vs 4.51; S ties'
- id: H2
  statement: L2 evict_first on the three input streams adds a further 3-4% at L on top of EL stores, because reads are 75% of
    the traffic here, so after each L2 set holds one of our EF lines our own clean lines become the victims instead of the
    flush's dirty lines.
  status: supported
  evidence:
  - 'probe_b200 r0: 268 MB EL 40.2 -> EF+EL 38.8; 134 MB 21.5 -> 20.1; 83.9 MB 14.8 -> 13.8; 33.5 MB 7.41 -> 7.10; S ties'
  - 'probe_b200 r0: ld.global.cs, nc+EF and L1::no_allocate+EF all equal EF (within 0.1 us); L2::256B prefetch hint = no gain'
  - 'portal r0: our L marginal 7.3 TB/s (7.9 -> 40.0 us over 235 MB) vs Tb 6.8 TB/s (8.5 -> 43.1): the hints are our L lead'
- id: H3
  statement: 256-bit loads/stores do not beat 128-bit for this stream, and cost +0.8 us at 131 elements.
  status: supported
  evidence:
  - 'probe_b200 r0: k8 vs k4 tie at 12.6-268 MB (within 1%); at 131 elements k8 3.06 us vs k4 2.29, repeatable across block sizes; cause unknown'
- id: H4
  statement: The S band (<= 0.3 MB) is at the launch floor; no kernel-side change can gain more than ~0.2 us there.
  status: open
  evidence:
  - 'probe_b200 r0 (unlocked clocks): empty kernel 1.57-1.66 us; float4/scalar kernels 2.2-2.4 us at 131-16384 elements for every block size, unroll and hint tried'
  - 'portal r0 CHALLENGES this: ours 2.7-2.9 us at <= 40960 elements vs Tb 2.2-2.5 (Tb is itself a kernel), so a kernel 0.4-0.5 us faster than ours exists under portal conditions'
- id: H5
  statement: The hidden baseline Tb is one fused PyTorch/Triton kernel (torch.compile-like) at ~2.4 us at S; at L it is better
    than default torch.compile (43.1 us at 268 MB on the portal vs 46.6 bench default compile; 6.8 TB/s marginal), so it is
    probably a max-autotune or hand-tuned Triton config without cache hints.
  status: open
  evidence:
  - 'probe_b200 r0: torch.compile(reference) 2.11 us at 131, 2.72 at 163840, 4.67 at 786432, 8.13 at 2.1M, 25.8 at 8.4M, 46.6 at 16.8M; eager 17.9-131.8 us'
  - 'portal r0: Tb = 2.4/2.5/2.2/2.4/2.5/2.5/3.0/3.4/4.8/5.0/8.5/13.7/16.9/24.1/29.0/43.1 us; eager is ~6x slower, so Tb is not the reference; operator survey: reference median 4.5x slower than Tb across 235 boards'
  - 'portal r0: Tb M-band slope 0.66 -> 12.6 MB equals ours (~4.8 TB/s effective), so Tb pays the same dirty write-back; its advantage at S/M is a ~0.4 us lower fixed cost'
- id: H6
  statement: More bytes in flight per thread (unroll 2-4, 512-1024 threads) raises L-band bandwidth.
  status: refuted
  evidence:
  - 'probe_b200 r0: with EF+EL hints, block 128/256/512/1024 x unroll 1/2/4 all within +-1% at 67-268 MB over 3 repeats; the 512x2 37.4 us reading did not reproduce (38.4-38.5)'
- id: H7
  statement: The marginal bandwidth ceiling for this 3:1 read:write stream under the dirty-L2 flush is ~7.3 TB/s; the remaining
    ~5 us at 268 MB (vs 8 TB/s) is dirty write-back traffic that no load/store hint removes.
  status: open
  evidence:
  - 'run_tests r0: 134 MB 20.0 us, 268 MB 38.5 us -> marginal 7.25 TB/s; fit t = 2.1 us + bytes/7.3 TB/s; portal 20.9/40.0'
  - 'untested: clean-flush (read-sweep) vs zero-fill flush comparison to size the in-window write-back cost (E3 r1)'
- id: H8
  statement: Per-size dispatch is unnecessary; one config (256 threads, one float4 per thread, EF+EL) is within noise of the
    best at every size.
  status: supported
  evidence:
  - 'probe_b200 r0: all configs tie at S/M on the unlocked bench; H6 ties at L; run_tests r0 16/16 with one config'
  - 'caveat: bench S ties were measured unlocked; H10/H11 may reopen dispatch (e.g. cluster-2 or hint-free path at <= 1.2M elements)'
- id: H9
  statement: TMA/bulk rings, persistent grids, clusters or weight-style prefetch could help this problem.
  status: parked
  evidence:
  - '#38 evidence: TMA +1-1.5 us fixed, persistent slower at every size, clusters on large grids lost 2-5%; nothing in this problem (no reduction, no reuse) changes that reasoning'
  - 'exception carried over from #38: cluster-of-2 launch on ONE-WAVE grids saved 0.13-0.16 us outside the CTAs (H12)'
- id: H10
  statement: The portal-vs-bench inflation (+0.4-0.5 us = +17% at S, +5% at L) is the 1500 MHz SM lock scaling the clock-bound
    part of the fixed cost (front-end launch, CTA dispatch, pre-load instructions, store drain), not the DRAM round trip. If so,
    a rented B200 with nvidia-smi -lgc 1500 reproduces the portal numbers, and every instruction before the first load matters more
    on the portal than on the unlocked bench.
  status: open
  evidence:
  - 'portal r0 vs rented r0: S 2.8-2.9 vs 2.1-2.5; M 2.8-5.0 vs 2.4-4.6; L 7.9-40.0 vs 7.3-38.5 (+4-8%)'
  - 'to test: E1 r1 (clock lock on the rented B200)'
- id: H11
  statement: Tb's 0.4-0.5 us S-band advantage over r0 on the portal comes from kernel structure or launch path (inductor-style
    Triton kernel: int32 offsets, no createpolicy/cache_hint, masked tail, 4-8 floats per thread, no tail branch), so a Triton
    port of r0 or an inductor-equivalent kernel that ties r0 on the unlocked bench would score ~2.4 us at S on the portal.
  status: open
  evidence:
  - 'portal r0: Tb 2.2-2.5 at <= 40960 vs ours 2.7-2.9; bench: torch.compile 2.11 vs ours 2.1 at 131 (unlocked)'
  - 'alternative explanation to keep in mind: Tb is the minimum over many agentic attempts and so sits at the bottom of the portal noise band (Tb spread 2.2-2.5 across identical-work sizes)'
  - 'to test: E1 r1 (bench, locked if possible) then one portal slot for the Triton candidate'
- id: H12
  statement: A cluster-of-2 launch (cudaLaunchAttributeClusterDimension = {2,1,1}) on one-wave grids (n <= ~1.2M elements) cuts
    0.13-0.16 us of per-launch overhead outside the CTAs, as measured on #38 with the same harness and GPU.
  status: open
  evidence:
  - '#38 (same harness, same rented B200): cluster-of-2 saved ~0.13-0.16 us for one-wave grids; large clustered grids lost 2-5% on the portal'
  - 'to test: E2 r1'
- id: H13
  statement: The M-band excess over fixed + bytes/7.3 TB/s is the write-back of the flush's dirty lines evicted by our first loads
    (about one line per L2 set, ~L2/ways ~= 8 MB, ~1 us at 12.6 MB, proportionally less below), and no load/store policy can
    remove it because every L2 miss must allocate a line.
  status: open
  evidence:
  - 'portal r0: Tb and ours share the same M slope (~4.8 TB/s effective from 0.66 to 12.6 MB) despite different hints'
  - 'bench r0: 12.6 MB 4.6 us vs 2.1 + 12.6 MB / 7.3 TB/s = 3.8 us predicted: ~0.8 us unexplained'
  - 'to test: E3 r1 clean-flush vs dirty-flush probe at 2.6-268 MB'
- id: H14
  statement: Fractional eviction policies (EF loads at 0.25-0.75, EL stores at 0.5, evict_unchanged) or ordering output stores
    before the next tile's loads in a persistent 2-tile software pipeline change which lines are victimised and recover part of
    the L-band write-back cost (<= 3%).
  status: open
  evidence:
  - 'only worth testing if E3 r1 finds an in-window write-back cost >= 2 us at 268 MB'
```

```json tasks
[
  {
    "id": "E1",
    "hypothesis": "H10",
    "title": "Identify the baseline and reproduce portal conditions: clock lock, inductor kernel, Triton port of r0",
    "operation": "port",
    "parents": ["r0-silu-bw-cuda-ldg128-os-efel"],
    "band": "all",
    "instructions": "1) Read the rented harness's timing.py and io.py: record (a) whether the per-trial statistic is mean or median of the 50 calls, and (b) whether the input copy into the shifting pool runs before or after the 2xL2 zero-fill (i.e. whether S-band inputs are L2-hot or cold). 2) Try to lock clocks on the rented B200: nvidia-smi -lgc 1500,1500 and -lmc 3996 (also try -ac). If it works, run every timing below under the lock and additionally re-run r0 at all 16 sizes; compare with the portal column (2.9/2.9/2.9/2.8/2.7/2.8/3.1/3.6/4.7/5.0/7.9/12.4/14.4/20.9/25.7/40.0 us). If the lock is refused, say so and continue unlocked. 3) Obtain the inductor kernel: torch.compile(reference, dynamic=False) with TORCH_LOGS=output_code at 131, 16384, 786432, 2097152 and 16777216 elements, default mode and mode='max-autotune-no-cudagraphs'; copy the generated Triton kernel(s) and their launch configs (XBLOCK, num_warps) into a standalone triton-language DPS solution that writes grad_input directly (no copy kernel, no dynamo in run(), one launch). 4) Write a Triton port of r0: one-shot grid, BLOCK 1024 elements, 4 warps, int32 offsets when n < 2^31, masked tail via tl.load/tl.store masks (no scalar loop), eviction_policy evict_first on the three loads and evict_last on the store; check the sm_100a SASS shows LDG.E.EF.128 and STG.E.EL.128; also a no-hint variant. 5) Time, interleaved ABAB with >= 5 repeats under harness timing: r0, r0 without hints, Triton port (hints / no hints), inductor default, inductor max-autotune, at all 16 sizes; report per size the median of medians AND the per-call mean of the 50 (outliers matter if the portal averages). 6) Deliver the best Triton candidate as a portal-ready solution (run_tests 16/16) and a one-line recommendation on whether to spend a portal slot on it as the launch-path A/B against r0. Also deliver the inductor-equivalent solution as an optional diagnostic submission (identifies Tb directly).",
    "success": "Either (a) under the lock r0 reproduces the portal S/M/L numbers within 0.1 us and the decomposition shows which component (hints, index math, tail, CTA count) scales with the clock, or (b) a Triton kernel beats r0 by >= 0.2 us at the 10 sizes <= 786432 on the bench while tying at L; in both cases run_tests 16/16 for the delivered candidate.",
    "refuted_if": "Every kernel ties r0 within 0.1 us at S both unlocked and locked (or the lock is unavailable and all tie): then H10/H11 can only be settled by a portal slot for the Triton port, and Tb's S advantage is probably selection of the minimum over noisy runs.",
    "model": "opus"
  },
  {
    "id": "E2",
    "hypothesis": "H12",
    "title": "Cut the per-launch fixed cost in CUDA at n <= 786432: cluster-of-2 one-wave launch, fewer pre-load instructions, branch-free tail",
    "operation": "knob_mutation",
    "parents": ["r0-silu-bw-cuda-ldg128-os-efel"],
    "band": "S",
    "instructions": "Start from r0 (kernel.cu + binding.cpp, two-file layout). Build these variants, each a minimal diff: V1 = r0 with (a) 32-bit index math when n < 2^31 (template on index type), (b) the two createpolicy instructions replaced by the constant policies EF 0x12F0000000000000 and EL 0x14F0000000000000 (verify the SASS still shows LDG.E.EF.128 / STG.E.EL.128), (c) the tail handled by thread nvec issuing up to 3 predicated scalar loads at once (#pragma unroll, no loop, no 64-bit trip count), so the first global load is reached in as few instructions as possible. V2 = V1 launched with cudaLaunchKernelEx and cudaLaunchAttributeClusterDimension {2,1,1} whenever the grid is <= 1184 CTAs (n <= 1,212,416 elements; pad the grid to an even CTA count, extra CTA exits on the i < nvec test); above that, the plain launch. V3 = V2 with hints removed for n <= 40960 (S path) to check whether cache_hint loads cost anything at 1.5 GHz-like conditions. V4 = V2 with 8 floats per thread (two float4 per stream issued before any compute) for n <= 1.2M, halving the CTA count. Probe each against r0 interleaved ABAB, 7 repeats, at the 10 sizes <= 786432 plus 2097152 and 16777216; report medians and the per-call mean. If the E1 session reports that the clock lock works, run under the lock. Run run_tests on the best variant and record regs/SASS from compile_b200. Dispatch only on n (shape), never on pointers or call count; the sequence of launches per call must be exactly one kernel.",
    "success": ">= 0.1 us faster than r0 at >= 7 of the 10 sizes <= 786432 on the bench (>= 0.13 us if the cluster launch is the cause), no loss (> 1%) at 2.1M and 16.8M, run_tests 16/16.",
    "refuted_if": "All four variants are within 0.05 us of r0 at every size <= 786432 over 7 interleaved repeats: then the cluster finding does not transfer to this problem and the pre-load instruction count is irrelevant; H12 refuted for #84.",
    "model": "sonnet"
  },
  {
    "id": "E3",
    "hypothesis": "H13",
    "title": "Exploratory: size the in-window dirty write-back cost, then attack L (fractional policies, store-before-load ordering) or M (one-wave fat CTAs)",
    "operation": "structural_mutation",
    "parents": ["r0-silu-bw-cuda-ldg128-os-efel"],
    "band": "L",
    "instructions": "Part A (decide where the headroom is): time r0 at 2.6, 12.6, 33.5, 134 and 268 MB after three flush types, interleaved, 5 repeats: (i) the harness's zero-fill of 2xL2 (dirty L2), (ii) a read-sweep of a 252 MB buffer (clean L2, inputs cold), (iii) zero-fill followed by a read-sweep of a different 252 MB buffer (dirty lines written back before the window, inputs cold). The (i)-(iii) gap is the in-window write-back cost per size; also run a reads-only variant (reduce the three inputs to one float per CTA) and a writes-only variant to split read and write costs. Record the resulting table as findings and update H7/H13 numbers. Part B, if the write-back cost at 268 MB is >= 2 us: probe (1) fractional policies: createpolicy.fractional.L2::evict_first on loads at 0.25/0.5/0.75, evict_last on stores at 0.5, evict_unchanged on stores, evict_normal stores + EF loads; (2) a persistent grid of 148 x 8 CTAs (256 threads) processing 2 tiles in flight per thread where tile t's store is issued before tile t+1's values are consumed (register double buffer, no smem), to test whether store-before-next-load ordering reduces dirty evictions; (3) same persistent grid but each CTA walking contiguous 64 KB chunks per stream to lengthen DRAM bursts. Part B', if the write-back cost at 268 MB is < 1 us (L at the ceiling): move to M: decompose 0.66-12.6 MB into fixed + streaming + write-back; probe one-wave fat-CTA designs (grid = 148 x k, each thread 2-4 float4 per stream issued before any compute, k from occupancy) against r0, interleaved, 7 repeats, at the 5 M sizes. In every variant keep fp32 exact order (__fmul_rn etc.), one launch, EF/EL policies only on our own buffers, no applypriority/discard. Deliver run_tests 16/16 for any variant that wins, plus all negative results as findings.",
    "success": "A quantified write-back cost per size (closing H7/H13 one way or the other) and a variant >= 3% faster than r0 at >= 3 of the 6 L sizes (or >= 0.15 us at >= 3 of the 5 M sizes), run_tests 16/16.",
    "refuted_if": "The clean-vs-dirty gap is < 1 us at 268 MB and < 0.3 us at 12.6 MB, and every policy/ordering/fat-CTA variant ties r0 within 1%: then H13 and H14 are refuted, L and M are at the practical ceiling, and all remaining effort goes to the fixed cost (E1/E2).",
    "model": "opus"
  }
]
```