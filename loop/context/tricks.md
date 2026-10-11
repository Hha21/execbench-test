# Verified tricks and lessons (cross-problem)

Distilled from the hypothesis ledgers of #38, #84, #85, #88 (memory-bound), #009/#218, #30 and #35 (GEMMs), up to
2026-10-11. Each line was measured on the rented B200 and, where marked, on the portal. A lesson tagged with one problem
may not transfer, so test it before relying on it. Problem-specific numbers live in each problem's ledger.

## 1. Before tuning: is the gap tuning at all?

- **Read the LEADER GAP section first** (`poc/leader_gap.py`). It backs out the per-workload times implied by the public
  top scores and the best mean score reachable at the hardware ceilings for each precision. On #35, TF32 tops out at
  about 0.70, while the leader scores 0.795 and FP16 can reach 0.84. Five rounds of TF32 tuning could never have
  closed that gap; the first FP16 round gained +0.042 on the portal (0.626 to 0.668).
- **Precision ladder** for fp32-in/fp32-out problems (operator policy 2026-10-11: any precision that meets the
  tolerance is fine; check the arithmetic first):
  - **TF32**: 10-bit mantissa. Use round-to-nearest (TMA tensor map `TFLOAT32`, or `cvt.rna.tf32.f32`). Truncation
    (plain `FLOAT32` map fed to kind::tf32) gives about 2.5x the error.
  - **FP16**: the same 10-bit mantissa as TF32, with 5 exponent bits (max 65504, normal down to 6.1e-5). When the
    inputs sit in that range, FP16 operands round exactly like TF32 and run at **2x the TF32 tensor rate**.
  - **BF16**: 7-bit mantissa, so about 8x TF32's error. It failed #35's atol 2e-3 on 39% of elements, against the
    1% the portal allows.
  - **Splits** (3xTF32, BF16x3/x9) are fp32-accurate but slower than plain TF32.
  - **Check the arithmetic**: per-element error std ≈ u·sqrt(2/3)·||e||·||w||, where u = 2^-11 for TF32/FP16 and
    2^-8 for BF16. Simulate in fp64 with numpy on the harness's input distribution, and count the elements outside
    atol + rtol·|ref|.
- **Tb** (hidden baseline) is a stored per-workload constant. Back it out from the portal page:
  `Ts = (Tb - S·t - S·Tb)/(1 - 2S)`. This is ill-conditioned near S = 0.5.
- **Tsol** for GEMMs uses about 1.8 PF/s, a 16-bit dense rate. A TF32 kernel cannot reach it.

## 2. Portal vs rented B200

- The portal runs the same Modal B200 with the SM clock locked (~1.45-1.5 GHz). The rented card boosts to 1.96 GHz,
  and under sustained GEMM load it is **power-capped to 1.1-1.35 GHz**: its clock drifts between and within runs.
  So rented differences of ~7% between sessions can be clock, not code (#35). Measure compute-bound kernels in **SM
  cycles** (`clock64` with `%globaltimer`), report the clock, and interleave candidate and control in one container.
- Typical portal/rented time ratios:
  - DRAM-bound: 1.03-1.10.
  - Clock-bound SIMT: about 1.2-1.3.
  - tcgen05 kernels at the MMA ideal: 1.35 (#35 r5).
  - tcgen05 kernels bound by load latency (FP16 with in-kernel conversion): 1.16-1.27 (#35 r6). Latency in ns
    does not stretch with the clock, so a rented tie can be a portal win: #35 r6 373/384 were flat rented and
    13% faster on the portal.
  - cuBLAS at large M: 1.16-1.22.
  - Rented gains on clock-bound code can grow (#35 r3: -7% rented became -18% portal) or shrink (#35 r5: -16% became
    -6%) on the portal. Predict from cycles, not rented µs.
- **Launch-gap and persistence savings shrink on the portal** (#35 r8). Fusing a pre-pass into the main kernel and making
  a 2-wave kernel persistent gave -5..-7% rented under harness_time but only -1..-2% on the portal. In the same
  rounds, gains in DRAM streaming rate carried over at 1.03-1.09 (r7). Prefer gains that move fewer bytes or use
  the bandwidth better over gains that remove launch or synchronisation overhead.
- The portal amplifies latency-tolerance gains: more resident warps lifted #88 by +16.7% portal against +6.7% bench.
- Large **clustered** grids cost 2-5% on the portal and seem to slow later plain launches. Use clusters only where a
  GEMM tile needs them (2SM pairs) or on small one-wave grids.
- Portal noise: memory-bound sizes repeat to ~0.1 µs, while compute-bound sizes vary 2-3% between submissions of
  the same code. Effects under ~1% are invisible.
  - #35 r8 vs r9: byte-identical FP16 tensor-core code ran 4-7% apart (one submission to the next), costing ~0.01
    of mean score. DRAM-bound sizes moved 1-2%.
  - Across three submissions (r8-r10), the identical sizes ran at speed factors 0.965-1.045. The factor moves
    every size together and hits tensor-heavy sizes hardest (6-9% spread) and DRAM-bound sizes least (1.5-2.5%).
    Score noise per submission is about +-0.008.
  - So one portal result cannot confirm a < 5% change on tensor-heavy sizes. Compare on the rented card in one
    container. The board keeps each team's best score, so a real gain also gets more chances to show.

## 3. Harness memory effects (every problem)

- The harness zero-fills 2x L2 before each timed call, so **L2 is full of dirty lines** when our kernel starts. Every
  line we allocate evicts one; its write-back lands inside our window and slows our reads:
  - A read-only stream reaches ~5.3 TB/s, against 7.0 for read+write (#85).
  - #38 lost ~0.5 µs at S and ~4 µs at M/L to it.
- **Output stores with L2 evict_last** (`st.global.L2::cache_hint` + `createpolicy...evict_last`) keep our outputs
  from displacing dirty lines inside the window. This is supported on #38, #84 (5-7% at L), #85 (6-13% at M/L) and #88.
- **Input loads with L2 evict_first** make later allocations evict our own clean read lines instead of dirty flush
  lines:
  - Supported on #84 (+3-4% at L), #85 (lifts the 5.4 TB/s read cap) and #35 (W stream: 45.9 to 39.9 µs rented).
  - Refuted on #38, where it hurt at some sizes.
  - Test it per problem. Evict_first on stores and evict_last on read-once weights hurt (#38).
- Floors: compulsory traffic ≈ reads + outputs, plus one dirty eviction per L2 set (~8 MB). Public read+write streams
  reach 7.05-7.1 TB/s marginal, so no SM-side change lifts L past that (#38 H11, #88 H14).
- Forbidden here, even though they work: discard/invalidate of lines we do not own, L2 persistence windows, any
  deliberate launch delay, PDL or overlap with harness kernels (see core_brief).

## 4. Memory-bound kernel design

- One-shot grids of small CTAs with one 8-float (2x128-bit) chunk per thread beat persistent grids, work stealing,
  TMA/bulk rings and fatter threads (#38, #85, #88). Persistent register-pipelined grids tied or lost (#88 H13).
- 128-bit accesses are enough. 256-bit loads gave a little on #38 and nothing or worse elsewhere (#84: +0.8 µs at tiny
  sizes).
- **Resident warps per SM**, not bytes in flight, set latency tolerance. Keep registers ≤ 32-40 so 2x the threads fit.
  The portal rewards this more than the bench (#88 H4).
- Remove integer division from address maths: use a 2-D grid (#85: -2% at S) or a host-computed magic divisor (#88).
- A cluster-of-2 launch on one-wave grids cuts ~0.1-0.2 µs of launch overhead at the smallest sizes (#38, #84).
- Launch floor: a tiny streaming kernel costs ~2.2-3 µs on the portal. Below ~3 MB the gap to Tb is not a design lever.
- Activation maths: `__expf` + `__fdividef` forms (e.g. GELU as a·b/(1+exp(-2u))) stay within 1e-5 and are faster
  than `tanhf` (#85).

## 5. GEMM / tensor cores (sm_100, tcgen05)

- **Warp-uniform roles** (#35 r5; the biggest single lesson). Issuing `tcgen05.mma` from one lane inside a divergent
  branch (`if (warp == 1 && lane == 0)`) makes nvcc wrap each UTCHMMA in an ELECT/BRA.U.ANY waterfall, costing
  ~300-400 clk per pipeline stage. The fix:
  - Get the role from a warp-uniform value (`__shfl_sync` of warp id).
  - Run a whole-warp TMA producer and a whole-warp MMA warp that loops uniformly and wraps only the issue in
    `elect.sync`.
  - Commit to the empty barriers with the multicast form.
  - With this, a 2SM ring ran at 770 clk per stage against a 768-clk MMA ideal (95-96% at B=373-919).
  - CUTLASS's `PipelineTmaUmmaAsync` and `cute::elect_one_sync` do exactly this. Copy its structure.
- **2SM pairs** (`cta_group::2`, cluster 2x1, M = 256 split 128 per SM) beat 1SM 128xN everywhere tested (#30 H4, #35):
  - Each SM loads half the B operand, which halves per-SM operand ingest.
  - Use two TMEM accumulators to cover N up to 512 in one pass.
- **A release arrive waits for the thread's own loads** (#35 r6). `mbarrier.arrive.release` (cta or cluster scope)
  from a thread that still has LDGs in flight (e.g. a register prefetch of the next k-block) stalls until those
  loads land, so the prefetch hides nothing. Converter warps that LDG, convert and STS into a UMMA stage lost
  13-27% per k-block to this. Hand off through a warp that has no loads in flight (a forwarder after a named
  barrier recovered part of it), or keep global loads out of the arriving threads (TMA or cp.async staging,
  or several warp groups taking k-blocks in turn with nothing prefetched at the arrive). A relaxed arrive is
  fast but leaves the shared-memory writes unordered: do not ship it. The same wait hits global stores (#35 r11):
  an epilogue warp's release arrive on the TMEM-free barrier waited ~1 us for its own output stores. That arrive
  hands over only the end of TMEM reads, so tcgen05.wait::ld + tcgen05.fence::before_thread_sync + a relaxed arrive
  is enough there.
- **Overlap the accumulator drain with the next tile's MMA** (#35 r11): give the epilogue its own warps and finish
  one accumulator early (its last k-blocks first), so it drains while the other accumulator completes. This cut an
  8 us MMA idle to ~5 us. What is left is store-bound: when every CTA drains at the same moment, the output burst
  runs at ~6 TB/s whatever the warp count. Stagger the CTAs' drains in time to hide more.
- **Per-stage floor**: a rotating ring's stage costs max(MMA clk, ~565 clk fixed with uniform issue). Make stages carry
  ≥ ~600 clk of MMA, e.g. 2 k-blocks per stage, or 48-k stages via a 32-k plus a 16-k TMA box.
- **Wave and tile quantisation**: 320 pair tiles over 74 SM pairs = 4.32 waves cost ~13% at L on #30. Pick tile counts
  that divide the SM pairs, or use stream-K. Batch-side padding (B=373 padded to 512) wastes MMA work, so choose N
  tiles in multiples of 16 per size.
- **Skinny GEMMs** (small batch × huge weight): the weight stream sets the time, so read it once with evict_first. With
  one M tile covering every batch row, each W tile goes to exactly one CTA (#35: 144 CTAs × 128 rows).
- **Fused epilogues**: fold the bias and any output split/scatter into the epilogue. #35's six-way split saved 5-22 µs.
  A cuBLASLt epilogue with C != D can pick a slower kernel than plain matmul (#30 H10).
- cuBLAS/cuBLASLt is the right anchor at S/M on fp16/bf16 GEMMs. Hand-written or CUTLASS 2SM kernels win where the
  shape is unusual: skinny, fused epilogue, odd sizes. Triton `tl.dot` could not compete on B200 (#009 H5).
- TMEM: allocate once per CTA, relinquish the permit, keep 1 CTA/SM when the allocation is large. A dynamically indexed
  pointer array in params costs a few STL/LDL, which is fine.

## 6. Working method

- One experiment per hypothesis with a refutation criterion. Interleave the control in the same container, 3 rounds.
  Report the clock for compute-bound work.
- run_tests on all 16 workloads × 10 rounds before shipping. Record max |err| per size for any reduced-precision path.
- When a rented gain disagrees with the portal, update the model (cycles vs µs, DRAM vs clock) before the next round.
