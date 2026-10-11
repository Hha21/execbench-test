# r11 design report: #35, large-B band (B = 373–919)

The best bet is to swap the GEMM's orientation (D3): put W rows in the MMA's M position and the batch in N, with the accumulator up to 512 columns wide. My estimate is about 95 µs at 919 and about 53 µs at 512, which meets the 1.1 PF/s target. The second probe should test W in tensor memory (D4), the design with the most upside. Two designs fail on paper: D7 (emb in tensor memory) and the TMA-staged form of the obvious design.

## 0. Common arithmetic

These assumptions apply to every design below.

- **MMA rate.** I use the settled 1.3 PF/s ceiling, which is 2,930 fp16 MAC per clock per SM at 1.5 GHz. A k-block is 64 K (one 128-B fp16 row), so a 128×N×64 MMA takes 2.8·N clocks. If the true issue rate is 4,096 MAC/clk (1.82 PF/s), every shared-memory budget below gets 1.4× tighter, so headroom matters.
- **Per-SM MMA time in the swap-AB layout.** With W rows in the MMA's M position, B pads only to a multiple of 16: 373 and 384 pad to 384, 449 to 464, 691 to 704, 773 to 784, 853 to 864, and 919 to 928. Per-SM MMA time is then 0.0895·N_pad µs. That gives 34.4 µs at 373/384, 45.8 µs at 512 and 83.0 µs at 919. Only 144 of the 148 SMs have a W tile.
- **Padding in the current layout.** With B in the M position, tiles of 256 batch rows pad 919 to 1024 (+11%), 773 to 1024 (+32%), 384 to 512 (+33%) and 373 to 512 (+37%).
- **The accumulator ceiling.** Tensor memory holds 128 × 512 fp32 per SM. 919 × 18432 / 148 ≈ 114K outputs per SM, so any B above about 526 needs at least two accumulator rounds per SM, whatever the tiling.
- **The per-stage floor is Little's law on the ring, not a constant.** One stage takes at least (ring round-trip) / (stages). The round trip of a converter-fed ring includes the W load latency, about 1,200–1,500 clk from DRAM. With 3–4 stages that gives about 1,000 clk, which matches the measured 950–1,200 clk floor. Designs that put more MMA work into each stage, or take the load latency out of the round trip, escape it.
- **Ingest per MAC.** The current layout puts B in M with 256 W columns per pair. Each SM then loads 32 KB of fp32 W for only 716 MMA clocks, which is under the floor. That alone explains the measured 0.7–0.9 PF/s.
- **emb for TMA-fed designs.** emb would be converted in-kernel. Each CTA converts 1/148 of emb into an fp16 scratch buffer (at most 5.6 MB), then increments a ticket counter (`target = (ticket/n + 1)·n`, which is correct on every call with no host state). Consumers wait on the counter before their first emb TMA. W loads start at once, so about 0.5–1 µs is exposed. The scratch buffer is fully rewritten each call, but the operator should confirm it is allowed.
- **Score.** If D3 lands at L −22% and 373/384 at −30%, `predict_score` gives about 0.745, up from 0.694. Its M band also includes the out-of-scope 131–211 sizes, so the true gain is somewhat less than +0.05.

## 1. Designs

### D1 — The obvious design with round-robin converters (variant 1 of the obvious design)

**a. Idea.** Keep B in M: a 2SM pair tile of 256 batch rows × 256 W rows, with two 256-column accumulators. Four converter warpgroups each take every 4th k-block. Each loads its block, converts it, stores it to shared memory and arrives, with **nothing else in flight** at the arrive. This removes the "release arrive waits for own loads" stall without needing a forwarder warp.

**b. Path.**
- W: DRAM → L2 → LSU (`LDG.256`, evict_first) → registers → `cvt.rn.f16x2` → `STS.128` (SW128) → MMA B operand.
- emb: fp16 scratch → TMA → shared memory → MMA A operand.

**c. Budget per SM per k-block** (the same at 919 and 512):

| Item | Value |
|---|---|
| MMA | 716 clk |
| L2→SM | W 32 KB + emb 16 KB = 67 B/clk |
| DRAM (W) | 8 KB at 919 (4 pairs share a tile), 16 KB at 512 |
| Shared memory | 64 KB = 89 B/clk |
| Issue | about 240 warp-instructions against 2,864 slots (8%) |
| Tensor memory | 512 columns |

Each warpgroup's cycle is about 1,700 clk, so four of them supply one block every 425 clk, which is under 716.

**d. Binding limit and time.** It is bound by MMA time plus padding. 919 needs 4 waves × 34.4K clk = 91.7 µs at 100% efficiency, so about **102 µs**. 512 is about **53 µs**.

**e. Risk and probe.** Four warpgroups of ~90 registers plus the other roles need about 62K of the 64K registers. Probe: compile with ptxas `-v` for 2, 3 and 4 warpgroups, then run the ring on synthetic data and record clk/stage for each.

Staging fp32 W through shared memory with TMA instead is **dead on paper**: it adds 64 KB per k-block and reaches 179 B/clk, above the 128 B/clk limit.

### D2 — L2-prefetch staging (variant 2 of the obvious design)

**a. Idea.** Same as D1, but one thread issues `cp.async.bulk.prefetch.tensor.2d.L2` for W 6–8 k-blocks ahead. The converters' LDGs then hit L2 at about 400–600 clk instead of DRAM latency. The ring round trip drops by about 2.5×, so two warpgroups are enough.

**b. Path.**
- W: DRAM → L2 (pulled by the TMA unit, no shared memory) → LDG hit → registers → cvt → STS → MMA.
- emb: as D1.

**c. Budget.** As D1, plus 32 KB of extra L2 lookups per k-block. Only about 38 MB of prefetched data is outstanding across the GPU.

**d. Time.** It is bound by MMA plus padding: about **100 µs** at 919 and about **52 µs** at 512.

**e. Risk and probe.** The prefetched lines may be evicted before use: L2 is full of the flush's dirty lines, so the prefetch should not carry an evict_first hint. Probe: a 148-CTA W stream that measures `clock64` latency per LDG at prefetch distances 0, 4 and 8, as a latency histogram.

### D3 — Swap-AB: W in M, emb in N up to 512 (2SM)

**a. Idea.** The pair's M = 256 W rows (128 per SM). The batch goes in N as two accumulators of up to 256 columns (at most 512). Each fp32 W byte now feeds **2×** the MMA work of D1. Each stage takes 1,297–1,432 clk, above any floor, and B pads only to a multiple of 16.

**b. Path.**
- W: DRAM → L2 → `LDG.256` (evict_first) → registers → cvt → STS → MMA **A**.
- emb: fp16 scratch → TMA (cluster of 2) → shared memory, with each SM holding N/2 rows → MMA **B**.

**c. Budget per SM per k-block:**

| Item | B = 919 (rounds of N = 464) | B = 512 (one round of N = 512) |
|---|---|---|
| MMA | 1,297 clk | 1,432 clk |
| L2→SM | 32 + 29.7 KB = 48 B/clk | 64 KB = 45 B/clk |
| DRAM | W 16 KB (two rounds share, in lockstep) | W 32 KB = 22 B/clk |
| Shared memory | 16 STS + 32 A-reads + 29.7 + 29.7 = 107 KB = 83 B/clk (71 with `collector::a` reuse) | 112 KB = 78 B/clk |
| Issue | about 5% | about 5% |
| Tensor memory | 464 columns | 512 columns |

Two round-robin converter warpgroups are enough, because each warpgroup's cycle (about 1,700 clk) is shorter than two stages.

At 691–919, the two rounds of a W tile run on **two pairs in the same wave**. Running them on one pair would re-read 226 MB of W from DRAM, for 532 MB in total.

**d. Binding limit and time.** It is MMA-bound at about 92% efficiency.

- 919: 90.2 µs + 1.5 prologue + 1.2 mid-drain + 1.3 final drain ≈ **95 µs**.
- 512: 49.8 + 1.5 + 1.3 ≈ **53 µs**.
- Other sizes: 449 ≈ 49 µs and 691 ≈ 74 µs. 373 ≈ 42 µs, where DRAM binds: W needs 30 B/clk against a 27 B/clk per-SM share.

The final drain is halved by finishing accumulator 0's last four k-blocks first, so its drain overlaps accumulator 1. Four stages of 48 KB fit in shared memory.

**e. Risk and probe.** The riskiest assumption is that LDG converters keep a 1,432-clk stage at ≥90% MMA busy across 144 CTAs. Probe: a synthetic ring (72 pairs, N = 512, 48 k-blocks, emb from an fp16 buffer) that reports clk/stage. It passes at ≤1,575 clk. In the same run, read `cudaOccupancyMaxActiveClusters` (needs ≥72).

### D4 — W in tensor memory (TS-MMA, A from TMEM)

**a. Idea.** Converters write fp16 W straight into tensor memory with `tcgen05.st`. The MMA reads A from tensor memory, the same TS form FA4 uses for P·V. W then never touches shared memory: no STS, no async-proxy fence, no A reads.

**b. Path.**
- W: DRAM → L2 → LDG (a `.16x256b` mapping, so 4 threads share a row and each warp-wide LDG touches 8 lines) → registers → cvt → `tcgen05.st` → TMEM A stage → `wait::st` + `fence::before_thread_sync` + mbarrier → MMA.
- emb: as D3 (shared memory, B operand).

**c. Budget per SM per k-block.** A k64 stage takes 32 tensor-memory columns, and the accumulator gets what is left.

| Item | B = 919 (3 rounds, N ≈ 310) | B = 512 (2 rounds of 256, k128 stages) |
|---|---|---|
| MMA | 867 clk | 716 clk per k64 |
| L2→SM | 32 + 19.8 KB = 60 B/clk | 32 + 16 KB = 67 B/clk |
| DRAM | 10.7 KB | 16 KB |
| Shared memory | emb only, 40 KB = 46 B/clk | 32 KB = 45 B/clk |
| Tensor-memory writes | 16 KB | 16 KB |
| Issue | about 200 against 3,468 slots | — |
| Tensor memory | 310 + 128 = 438 columns | 256 + 256 = 512 columns |

The 919 rounds are 216 pair-units spread over 3 lockstep waves.

**d. Binding limit and time.** It is MMA-bound: about **94 µs** at 919 and about **53 µs** at 512. It has the most shared-memory headroom of any design, which makes it robust if the true MMA rate is 1.82 PF/s. A 1SM fallback (N = 384) also fits, at 89 B/clk.

**e. Risk and probe.** Three unknowns: whether kind::f16 with `cta_group::2` and A in tensor memory runs at full rate, the A layout in tensor memory (row = lane, two fp16 per column), and `tcgen05.st` throughput. Probe: one pair writes a known fp16 pattern with `.32x32b`, then checks 4,096 MMAs of 128×256×16 against the host, timing A-from-TMEM against A-from-smem.

### D5 — L2-resident fp16 W ring (convert once, pure-TMA consumer)

**a. Idea.** Inside the kernel, converter warps write fp16 W into a 16–24 MB global ring that stays in L2, and publish a flag per chunk. The MMA side is a pure-TMA swap-AB ring, close to the CUTLASS floor of about 600 clk. Each W element is converted exactly once, even when two rounds use it.

**b. Path.**
- W: DRAM → L2 → LDG → registers → cvt → **STG fp16 → L2 ring** → flag (release) → consumer acquire + `fence.proxy.async.global` → TMA → shared memory → MMA A.
- emb: as D3.

**c. Budget per SM per k-block:**

| Item | B = 919 | B = 512 |
|---|---|---|
| MMA | 1,297 clk | 1,432 clk |
| L2→SM | 16 KB fp32 + 16 KB fp16 + 29.7 KB emb = 48 B/clk | 32 + 16 + 32 KB = 56 B/clk |
| SM→L2 | 8 KB | 16 KB |
| Shared memory | 83 B/clk | 78 B/clk |
| Converter issue | small | small |

Extra DRAM traffic comes from first touches of the ring evicting dirty flush lines, about the size of the ring.

**d. Time.** It is MMA-bound at about 95%: about **93 µs** at 919 and about **53 µs** at 512, where it gives no conversion saving.

**e. Risk and probe.** Two risks: cross-proxy visibility and flag-polling latency, and ring lines being evicted before they are consumed. The ring also needs operator approval as scratch, and its eviction hint would need a policy check. Probe: 74 producer CTAs and 74 consumer CTAs, 1,000 handoffs, measuring flag-to-TMA-complete latency and checking every byte.

### D6 — Cluster-4 conversion sharing over DSMEM (B in M)

**a. Idea.** Two pairs (four CTAs) that need the same W tile each convert half of each W k-block. They push it to their peer with `cp.async.bulk.shared::cluster`, completing on the peer's mbarrier. This halves both the conversion work and the W L2 traffic of D1.

**b. Path.**
- W: L2 → LDG → cvt → local STS → proxy fence → bulk DSMEM copy into the peer's shared memory → MMA B.
- emb: TMA → shared memory → MMA A.

**c. Budget per SM per k-block** (716 clk MMA):

| Item | Value |
|---|---|
| L2→SM | 16 + 16 KB = 45 B/clk |
| DSMEM | 8 KB out, 8 KB in |
| Shared memory | 8 + 8 + 8 + 16 + 16 + 16 = 72 KB = 100 B/clk (tight) |
| Tensor memory | 512 columns |

**d. Time.** Padding binds: about **100 µs** at 919 and about **52 µs** at 512, but only if 36 clusters of 4 fit. At about 18–20 SMs per GPC, only 32 may fit. That would make 512 take 3 waves (about 75 µs).

**e. Risk and probe.** GPC packing, plus the latency DSMEM copies add to the ring. Probe: `cudaOccupancyMaxActiveClusters` for clusters of 4, then time an 8 KB-per-stage bulk DSMEM ring round trip.

### D7 — emb in tensor memory (B in M), rejected on paper

**a. Idea.** emb becomes the A operand in tensor memory, converted on the fly from fp32, which needs no scratch buffer. W stays in shared memory.

**b. Path.**
- emb: L2 → LDG → cvt → `tcgen05.st`.
- W: as D1.

**c. Budget.** Tensor memory is split as 2×192 accumulator columns plus 128 A columns, so N is 192 per pair and a k-block is 537 clk. L2→SM is W 24 KB + emb fp32 32 KB = **105 B/clk**, which is above the measured 93–100 feed ceiling and about 23 TB/s in aggregate.

**d. Time.** About 110 µs at 919 (6 waves) and about 60 µs at 512. **Do not build it.** It shows that emb belongs on the side that is read once.

**e. Probe.** None needed.

### D8 — Stream-K add-on to D3 (148/148 SMs, fp32 DSMEM fix-up)

**a. Idea.** Spread D3's total MAC count over all 148 SMs, and reduce the split tiles' fp32 partials through DSMEM before the store. This keeps fp32 throughout, but the operator should confirm it under the "one accumulator" rule.

**b. Path.** As D3, plus partial tiles moving from tensor memory to registers, `st.async` to the owning SM, and an fp32 add there.

**c. Budget.** As D3, plus about 16 KB of DSMEM per split tile.

**d. Time.** It saves the 2.7% lost to idle SMs, minus about 1 µs of fix-up: about **93 µs** at 919 and about **52 µs** at 512.

**e. Probe.** Time a DSMEM fp32 reduction of a 128×64 partial tile.

## 2. Summary and ranking

| Design | 919 (µs) | 512 (µs) | Chance in a 2-hour build | Notes |
|---|---|---|---|---|
| **D3** swap-AB | 95 | 53 | 0.40 | Large stages; no B padding; needs emb flag |
| **D2** L2 prefetch | 100 | 52 | 0.60 | Smallest change, but the gain is uncertain |
| D1 round-robin warpgroups | 102 | 53 | 0.50 | Register-tight; 919 padding |
| **D4** W in tensor memory | 94 | 53 | 0.25 | Best headroom; TS layout is unknown |
| D8 stream-K (on D3) | 93 | 52 | 0.30 after D3 | Add-on |
| D6 cluster-4 DSMEM | 100 | 52 (or 75) | 0.20 | GPC packing risk |
| D5 L2 ring | 93 | 53 | 0.15 | Flags, proxy fences, scratch approval |
| D7 emb in tensor memory | 110 | 60 | — | Dead |

**Ranking by time × chance:** D3, D2, D1, D4, D8, D6, D5, D7.

**Probes to run first:**
1. **D3's synthetic swap-AB ring.** It is clock-portable and tests the central claim: that 1,432-clk stages escape the ring floor with two LDG converter warpgroups. It also confirms that 72 pairs of clusters are co-resident.
2. **D4's TS-MMA rate and layout check.** It is cheap, and if it passes, W leaves shared memory entirely. That gives the most margin, which matters if the true MMA rate is above the settled 1.3 PF/s.

D2's prefetch-latency probe comes third, because it can be bolted onto both the current kernel and D3.