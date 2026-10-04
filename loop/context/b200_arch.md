# B200 (sm_100a) for kernel writers

Facts that change design decisions, mainly for memory-bound kernels; §9 briefly covers compute-bound kernels. Source
keys refer to `sources.md`. Labels: VERIFIED-CSF3, SPEC, PAPER, LITERATURE, INFERRED.

## 1. The chip at a glance

| Item | B200 value | Source |
|---|---|---|
| Compute capability, target | 10.0, compile `sm_100a` (`a` = arch-specific features: tcgen05, TMEM) | [SPEC: CUDA guide], [PAPER: problem_packager.py] |
| SMs | 148 enabled (2 dies × 74; 80 per die physically), 8 GPCs | [LITERATURE: chipsandcheese-b200, arxiv-2512.02189] |
| Harness clocks | SM locked at **1500 MHz**, DRAM at 3996 MHz | [PAPER: device_config.py] |
| HBM3e | 8 stacks (4 per die), 8192-bit bus; 8 TB/s; 3996 MHz × 2 × 8192 bit ≈ **8.18 TB/s** peak at the locked clock | [SPEC: datasheet, blackwell-arch], [INFERRED from the clock preset] |
| Capacity | 180 GB (tuning guide, HGX), 192 GB (product pages), 186 GB (one third-party page); irrelevant here | [SPEC] (sources disagree) |
| L2 | **126 MB**, two partitions (one per die) | [SPEC: blackwell-tuning], [LITERATURE: chipsandcheese-b200] |
| L1 + shared memory | 256 KB per SM unified; shared-memory carve-out 0/8/16/32/64/100/132/164/196/**228 KB**; 227 KB per CTA (1 KB reserved per CTA); static `__shared__` ≤ 48 KB | [SPEC: blackwell-tuning, cuda-guide-cc] |
| Threads | 2048 per SM, 64 warps, **32 CTAs** per SM, 1024 per CTA | [SPEC] |
| Registers | 64K 32-bit per SM, ≤ 255 per thread | [SPEC] |
| TMEM | 256 KB per SM (128 lanes × 512 columns × 32 bit), tensor cores only | [SPEC: PTX ISA tcgen05], [LITERATURE] |
| Die-to-die link | NV-HBI 10 TB/s | [SPEC: Blackwell architecture page] |
| Clusters | portable ≤ 8 CTAs; B200 allows 16 with `cudaFuncAttributeNonPortableClusterSizeAllowed` | [SPEC: blackwell-tuning] |

## 2. Memory hierarchy numbers

| Level | Number | Source |
|---|---|---|
| L1D hit latency | 39 cycles (19.6 ns at about 2 GHz); at the 1.5 GHz lock, 39 cycles ≈ 26 ns | [LITERATURE: chipsandcheese-b200], [INFERRED] |
| L2 latency, local partition | about 150 ns; "dramatically" higher when the data sits in the other die's partition | [LITERATURE: chipsandcheese-b200] |
| L2 bandwidth | about 21 TB/s within a partition, 16.8 TB/s crossing partitions (Vulkan test) | [LITERATURE: chipsandcheese-b200] |
| DRAM latency | **unknown**. Chips and Cheese finds it higher than H100/A100 (no number in text); arXiv 2512.02189 v1 claims 58% lower than H200 on cache misses. **These disagree.** Planning value: 800 ns loaded (range 700–1000 ns) | [LITERATURE], [INFERRED] |
| Achieved HBM bandwidth | arXiv 2512.02189 **v1**: STREAM triad 7.48 TB/s (94%); **v3** of the same paper: 4.14 TB/s (51.8%). The versions disagree; v3 gives no thread/block configuration. Planning value: 85–92% of 8 TB/s for a well-fed streaming kernel | [LITERATURE], [INFERRED] |
| Global atomics | 90–100 ns between threads on the same partition, 190–220 ns across | [LITERATURE: chipsandcheese-b200] |

Dual die: each die has 4 of the 8 HBM3e stacks attached, and the dies are joined by the 10 TB/s NV-HBI
[SPEC: blackwell-arch]. [INFERRED:] CTAs are spread over both dies and physical addresses are interleaved across all 8
stacks, so about half of every SM's traffic crosses NV-HBI. There is no user-level control. Expect a little more latency than a
monolithic die, which means more bytes must be in flight.

## 3. Little's law: bytes in flight

Bytes in flight = bandwidth × latency. The DRAM rate a kernel sustains is capped at
`(outstanding bytes per SM × SMs) / latency`.

| GPU | BW (peak) | SMs | Latency assumed | In flight, whole GPU | **Per SM** | B/clk/SM at the clock used |
|---|---|---|---|---|---|---|
| B200 | 8.0 TB/s | 148 | 800 ns (700–1000) | 6.4 MB | **43 KB** (38–54) | 36 at 1.5 GHz (harness) |
| H200 | 4.8 TB/s | 132 | 650 ns | 3.1 MB | 24 KB | 18 at about 1.98 GHz (CSF3, unlocked) |
| A100 80GB | 2.04 TB/s | 108 | 600 ns | 1.2 MB | 11 KB | 13 at 1.41 GHz |
| L40S | 0.86 TB/s | 142 | 600 ns | 0.5 MB | 3.7 KB | 2.4 at 2.52 GHz |

Latencies are planning assumptions [INFERRED; the PoC analytic model used 600/600/650/700 ns]. Per SM, B200 needs
**about 4× A100 and about 1.8× H200**. Latency rises under load, so aim for about 1.5× the table value.

How designs supply it (sm_100a register and shared-memory counts are VERIFIED-CSF3 from `probes/`):

| Design (from probes) | Resources per CTA | CTAs/SM | Outstanding input bytes per SM |
|---|---|---|---|
| Triton pointer tile, ROWS=8, 4 warps (2 × LDG.128 per thread) | 32 regs, 0 smem | 16 (thread limit) | 16 × 4 KB = **64 KB** |
| Triton pointer tile, ROWS=16, 4 warps | 38 regs (40 after rounding) | 12 (register limit) | 12 × 8 KB = 96 KB |
| Triton pointer tile, ROWS=32, 8 warps | 38 regs | 6 | 6 × 16 KB = 96 KB |
| Gluon TMA ring, ROWS=16, 4 stages, weights in smem | 40 regs, 73.8 KB smem | 3 (smem limit) | 3 × 4 × 8 KB = 96 KB |
| CUDA C++ 1-D bulk ring, ROWS=16, 4 stages | 25 regs, 50 KB smem | 4 | 4 × 32 KB = 128 KB |

These pointer designs only reach their numbers if each thread issues all its loads before the first use (the
reduction), and if many CTAs are resident at once. Register pressure above 32 per thread cuts occupancy; above 64 it
halves it again.

## 4. Occupancy rules for CC 10.0

- Resident CTAs per SM = min(32, ⌊2048/threads⌋, ⌊65536/(regs_rounded × threads)⌋, ⌊228 KB/(smem + 1 KB)⌋).
  Registers are allocated in 256-register chunks per warp (8 per thread) [SPEC: occupancy calculator convention;
  INFERRED for sm_100]. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (CUDA) or Triton's `kernel.n_regs` and
  `kernel.metadata.shared` give the inputs.
- One wave = 148 × CTAs/SM. For the smallest #38 workload (12,288 rows of Q+K), ROWS=8 gives 1,536 CTAs. That is
  10.4 per SM, one wave at 16 CTAs/SM. [INFERRED]
- A Triton kernel whose shared memory exceeds about 113 KB per CTA allows only 1 CTA per SM; that is fine for a
  persistent design, bad for one-shot.

## 5. TMA, bulk copies and mbarrier pipelines

Hardware mechanisms (Hopper and Blackwell) [SPEC: PTX ISA, CUDA guide §4.11–4.12]:

- **Tensor TMA** (`cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes`) uses a 128-B tensor map
  (CUtensorMap). Rules: global address 16-B aligned; strides a multiple of 16 B; `boxDim[i] ≤ 256`; inner box bytes a
  multiple of 16; the shared-memory destination 128-B aligned; with swizzle 32/64/128 B the inner box must be ≤ the
  swizzle span. Out-of-bounds rows are zero-filled on load and clipped on store. Ranks 1–5 in the hardware;
  **Triton's `make_tensor_descriptor` supports ranks 2–5 only**.
- **1-D bulk copy** (`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dst],[src],size,[mbar]`) needs
  no tensor map. Source and destination must be 16-B aligned and the size a multiple of 16. A `[ROWS,128]` fp32 tile
  of a row-major tensor is one contiguous ROWS×512 B chunk, so this is the simplest TMA path for row-wise kernels.
  SASS: `UBLKCP.S.G` (load) and `UBLKCP.G.S` (store) [VERIFIED-CSF3].
- **mbarrier protocol**: `mbarrier.init(count)`; the producer calls `arrive.expect_tx(bytes)` and issues the copy; the
  hardware decrements tx-count as bytes land; consumers spin on `try_wait.parity(phase)`. The phase flips each time the
  stage is reused, so use `(i / STAGES) & 1`. Before refilling a stage, every consumer warp must be done reading it:
  use a CTA barrier, or a second "empty" mbarrier per stage.
- **Bulk store**: write registers to shared memory, `fence.proxy.async.shared::cta`, barrier, then one elected thread
  issues `cp.async.bulk.global.shared::cta.bulk_group` (or `.tensor`) and `cp.async.bulk.commit_group`. Before reusing
  that buffer, `cp.async.bulk.wait_group.read N`.
- **Hints**: both bulk forms accept `.L2::cache_hint` with a `createpolicy` operand. `cp.async.bulk.prefetch.L2`
  prefetches to L2 without using shared memory.
- **Ampere-style `cp.async`** (LDGSTS, 4/8/16 B per thread) still exists. Triton uses it when a `tl.range(...,
  num_stages>1)` loop pipelines pointer loads [VERIFIED-CSF3].
- Bytes in flight per CTA = STAGES × tile bytes; shared memory is the budget (228 KB per SM). A TMA design gets its
  in-flight bytes from **few instructions** (one per tile). At 36 B/clk/SM that matters more on B200 than on H200.
  [INFERRED]

## 6. Clusters and distributed shared memory

- Up to 8 CTAs per cluster (16 opt-in on B200). The CTAs of a cluster are co-scheduled on one GPC and can read and
  write each other's shared memory (DSMEM). TMA can multicast one global load to several CTAs' shared memory.
  [SPEC: blackwell-tuning, CUDA guide]
- For row-wise memory-bound kernels clusters help little. The only data reused across CTAs is small (the #38 weights,
  24 KB per tensor, already L2-resident). One use: split a very long row across a cluster and reduce through DSMEM.
  Size occupancy with `cudaOccupancyMaxActiveClusters`. [INFERRED]

## 7. Cache-eviction hints

- PTX: `ld/st.global.L1::{evict_first,evict_last,no_allocate}`, `.L2::{evict_first,evict_last}` (sm_100+ for the
  `level2::eviction_priority` form), `.L2::cache_hint` with `createpolicy.fractional.L2::evict_*`, cache operators
  `.cg/.cs/.lu`, and prefetch sizes `.L2::64B/128B/256B`. [SPEC: PTX ISA]
- What Triton 3.7 emits on sm_100a [VERIFIED-CSF3, probes/probe1]:

| Triton | PTX | SASS |
|---|---|---|
| `tl.load(..., eviction_policy="evict_first")` | `createpolicy … evict_first` + `ld.global.L2::cache_hint.v4` | `LDG.E.EF.128` |
| `eviction_policy="evict_last"` | `… evict_last …` | `LDG.E.EL.128` |
| `tl.store(..., eviction_policy="evict_first")` | `st.global.L2::cache_hint` | `STG.E.EF.128` |
| `tl.load(..., cache_modifier=".cg")` | `ld.global.cg.v4` | `LDG.E.128.STRONG.GPU` (a coherent load: avoid for streaming) |
| `tl.store(..., cache_modifier=".cs")` | `st.global.cs.v4` | `STG.E.EF.128` |
| tensor-descriptor `.load/.store` | no hint parameter in Triton 3.7 (`load(offsets, latency=None)`) | `UTMALDG/UTMASTG` |

- Expected effect for one-pass streams (each byte touched once) is small [INFERRED]. Mark the streamed inputs and
  outputs `evict_first` and the reused weights `evict_last`. The PoC's EVICT knob was not decisive on A100/L40S. The
  harness resets persisting-L2 state before each call, and changing access-policy windows counts as environment
  manipulation; don't do it.

## 8. Launch overhead and small-workload latency

- CPU launch cost is **not timed** for a single kernel. Measured t ≈ CTA dispatch ramp + first DRAM round trip +
  streaming time + tail (last stores drained, last CTA exits). [PAPER: timing.py; INFERRED]
- Back-of-envelope for the smallest #38 workload (12.6 MB; 42.5 KB of reads per SM): about 0.3 µs ramp, plus about
  0.8 µs latency, plus about 0.8 µs streaming at 54 GB/s per SM, plus about 0.5 µs tail ≈ **2.4 µs**, against the
  1.57 µs bandwidth floor [INFERRED]. These overheads are about half the time on small workloads and about 2% on large
  ones.
- Tactics: one launch; a single wave sized to fill every SM evenly; issue all of a CTA's loads before any dependent
  work; avoid long prologues (device-side tensor-map creation, mbarrier init chains, large shared-memory zeroing) in the
  small-size path; avoid persistent loops whose per-iteration latency serialises a CTA's few tiles.
- **Programmatic dependent launch** (PDL, CC ≥ 9.0) lets kernel 2 start its prologue while kernel 1 drains. Triton
  3.7: `launch_pdl=True` launch option plus `gdc_wait()` / `gdc_launch_dependents()` from
  `triton.language.extra.cuda` (import at module level, not inside `@triton.jit`) [VERIFIED-CSF3: PTX contains
  `griddepcontrol.*`]. It is only relevant if a design needs two kernels.
- **Unknown** for B200: the empty-kernel CUPTI span, the CTA launch rate, the kernel-to-kernel gap. How to find out:
  measure them on H200 with the harness (an empty kernel, and a 2-launch variant of a good kernel). On B200, a pair of
  portal submissions that differ only in one extra trivial launch (legitimate, correct, costs 2 slots).

## 9. Compute-bound kernels, in brief (for later problems)

- `wgmma` (sm_90a) **does not exist** on sm_100a. Blackwell tensor cores use `tcgen05.mma`: issued by a single
  thread, asynchronous, A/B from shared memory (A may come from TMEM), accumulator in **TMEM** (256 KB per SM),
  completion through `tcgen05.commit` → mbarrier, and `tcgen05.ld` to move TMEM → registers. `cta_group::2` lets a
  CTA pair share one MMA (M=256). [SPEC: PTX ISA]
- Types: FP64, TF32, BF16, FP16, FP8 (E4M3/E5M2), FP6, FP4, INT8; block-scaled MX formats and **NVFP4** (16-element
  blocks, E4M3 scale) [SPEC: CUDA guide CC table; PAPER: Quant problems use NVFP4 16-element block scaling].
- Rough dense peaks at the harness's 1.5 GHz [INFERRED from 1024 16-bit MAC/clk per SM sub-partition (chipsandcheese)
  × 4 × 148 × 1.5 GHz]: BF16 ≈ 1.8 PFLOP/s, FP8 ≈ 3.6, FP4 ≈ 7.3. The datasheet's 2.25 PFLOP/s dense BF16 implies
  about 1.85 GHz boost. FP32 SIMT: 128 lanes per SM ≈ 57 TFLOP/s at 1.5 GHz, and sm_100 adds packed `f32x2`
  mul/add/fma (Gluon emitted `mul.f32x2` in our probes [VERIFIED-CSF3]). FP16 SIMT is no longer 2× FP32
  [LITERATURE: chipsandcheese-b200].
- Tooling: CUTLASS 4.4 C++ (sm100 collectives), CuTe DSL 4.4.2 (compiles for sm_100a on the CSF3 login node with
  `CUTE_DSL_ARCH=sm_100a`, about 20 s for a trivial kernel [VERIFIED-CSF3]), Triton `tl.dot` lowers to tcgen05 on
  sm_100.

## 10. What each cheap GPU can tell us

| | B200 (target) | H200 (CSF3) | A100 80GB (CSF3) | L40S (CSF3) |
|---|---|---|---|---|
| Arch | sm_100a | sm_90a | sm_80 | sm_89 |
| SMs; threads/SM; CTAs/SM | 148; 2048; 32 | 132; 2048; 32 | 108; 2048; 32 | 142; 1536; 24 |
| Shared memory per SM | 228 KB | 228 KB | 164 KB | 100 KB |
| L2 | 126 MB (2 partitions) | 50 MB | 40 MB | 96 MB |
| DRAM | HBM3e 8 TB/s | HBM3e 4.8 TB/s | HBM2e 2.04 TB/s | GDDR6 0.86 TB/s |
| TMA / mbarrier / clusters | yes | yes | no (cp.async only) | no |
| 256-bit LDG/STG | yes | no | no | no |
| tcgen05 / TMEM | yes | no (wgmma) | no | no |
| Clocks in our timing | SM 1500 MHz locked | default boost | default boost | default boost |

Sources: B200 above; H200/A100/L40S from NVIDIA datasheets [SPEC] and the CUDA CC table. Check CSF3's exact
H200/A100 SKUs with `nvidia-smi -q` in a job (unknown: SXM or NVL/PCIe).

**What they can and cannot tell us** [INFERRED unless marked]:

- **H200** is the best proxy. It runs every TMA/mbarrier/cluster design (Triton descriptor kernels compile to the same
  UTMALDG/UTMASTG/SYNCS mix on sm_90a and sm_100a [VERIFIED-CSF3]), and it has HBM. It cannot show: 256-bit accesses,
  the about 1.8× larger per-SM bytes-in-flight requirement (a design that saturates H200 can starve B200), dual-die L2 effects, or
  SM-side costs at 1.5 GHz (H200 runs faster per SM relative to memory).
- **A100** is a good ranking signal for plain-load designs on HBM. In the PoC the emulator found the best variant in
  about 8 measurements, against about 20 for plain Bayesian optimisation (PoC, CSF3). It cannot run TMA designs: Triton
  silently falls back on sm_80 to different code (4-byte LDGSTS, 72 regs for the host-descriptor kernel
  [VERIFIED-CSF3]), so A100 timings of TMA niches are meaningless. Its bytes-in-flight need is 4× lower.
- **L40S** is a weak predictor. It has GDDR6, 1536 threads per SM and 100 KB of shared memory; in the PoC it picked a
  different winner and helped the emulator only early (PoC, CSF3). Use it as a correctness smoke test and to catch
  gross regressions.
- **Static sm_100a compile** (no GPU) gives the exact registers, shared memory, spills and instruction mix B200 will
  run, for Triton, Gluon, CUDA C++ (nvcc 12.8 + `ptxas-blackwell` 13.1) and CuTe DSL [VERIFIED-CSF3]. It cannot give
  timing.
