# Sources

Access date is 2026-10-04 unless stated. Short keys in brackets are the ones used in the other docs.

## Our own work (VERIFIED-CSF3, "PoC, CSF3")

| Key | Path / location | What |
|---|---|---|
| [plan] | `/home/lain/execbench-test/poc/PLAN.md` | three-loop plan, leaderboard anchors, PoC implications |
| [poc-readme] | `/home/lain/execbench-test/poc/README.md` | emulator rehearsal results, simulator caveats, `launch.h` fix |
| [poc-kernel] | `poc/kernels/rmsnorm_qk.py`, `poc/make_variants.py`, `poc/variants/{variants.csv,v000.json}` | PoC Triton design, knobs, solution format |
| [poc-data] | `poc/results/rehearsal_sim/table.csv` | per-variant × per-workload A100/L40S times (72 variants) |
| [poc-static] | `poc/static_features.py`, `poc/results/static_features.csv` | how to cross-compile Triton for sm_100a without a GPU |
| [poc-analytic] | `poc/analytic.py` | latency assumptions 600/600/650/700 ns (A100/L40S/H200/B200) |
| [probes] | `loop/context/probes/` (copies of `/scratch/t95317ha/solx/tmp/ctxdocs/` on CSF3), `probes/RESULTS.md` | compile-only probes run 2026-10-04 |
| [csf3-venv] | `/scratch/t95317ha/solx/venv` (CSF3) | fbtriton 3.7.1, torch 2.9.0+cu130, nvidia-cutlass-dsl 4.4.2, cuda-tile 1.2.0, cuda-python 13.1.1, cupti-python 13.0.1, ptxas 12.9.86, ptxas-blackwell 13.1.80 |
| [p038] | `/scratch/t95317ha/solx/SOL-ExecBench/data/benchmark/L1/038_flux_multi_head_rmsnorm_qk/{definition.json,workload.jsonl}` (CSF3) | problem definition and the 16 workloads |

## SOL-ExecBench harness and paper (PAPER)

Local clone `/home/lain/execbench-test/SOL-ExecBench`, commit `a9fa080` ("Add v1.1 timing methodology").

| Key | File | Used for |
|---|---|---|
| [dockerfile] | `docker/Dockerfile` | CUDA 13.1.1 base image, CUTLASS v4.4.1, `fbtriton==3.7.1`, cutlass-dsl-libs 4.4.2 |
| [pyproject] | `pyproject.toml` | torch 2.9.0 (cu130), nvidia-cutlass-dsl 4.4.2, cuda-tile 1.2.0, cudnn-frontend 1.18.0, etc. |
| [readme] | `README.md` | B200 at 1500 MHz, ≈5% (≤10%) noise, CLI and config defaults |
| [timing.py] | `src/sol_execbench/core/bench/timing.py` | CUPTI span, L2 flush (2×L2 zero-fill), warm-up/rep, median |
| [cupti_utils.py] | `src/sol_execbench/core/bench/cupti_utils.py` | activity sequence matching (kernels, memcpy, memset) |
| [io.py] | `src/sol_execbench/core/bench/io.py` | input generators (name heuristics), shifting pointer allocator |
| [correctness.py] | `src/sol_execbench/core/bench/correctness.py` | tolerance test, NaN/Inf, all-zero, matched ratio |
| [reward_hack.py] | `src/sol_execbench/core/bench/reward_hack.py` | monkey-patch, thread, lazy-output, integrity checks |
| [eval_driver.py] | `src/sol_execbench/driver/templates/eval_driver.py` | 10 correctness rounds, DPS call, check order, `cpp_extension` block |
| [device_config.py] | `src/sol_execbench/core/bench/config/device_config.py` | clock presets B200 1500/3996 MHz |
| [benchmark_config.py] | `src/sol_execbench/core/bench/config/benchmark_config.py` | warm-up 10, iterations 50, seed 200 |
| [packager] | `src/sol_execbench/driver/problem_packager.py`, `driver/templates/build_ext.py` | sm_100a gencode injection, C++ build |
| [solution.py] | `src/sol_execbench/core/data/solution.py`, `core/data/workload.py` | schema, languages, compile-option defaults, tolerance defaults |
| [docs] | `docs/solution.md`, `docs/workload.md` | DPS convention, CuTe DSL practice, dependencies |
| [examples] | `examples/{triton,cuda_cpp,cute_dsl}/…` | binding patterns |
| [sol_score.py] | `src/sol_execbench/sol_score.py` | score formula |
| [paper] | `/home/lain/execbench-test/SOL-ExecBench-LaTeX/04_dataset_evaluation.tex`, `04_2_solar.tex`; arXiv 2603.19173 (https://arxiv.org/abs/2603.19173) | scoring, evaluation protocol (CUDA events, 3 trials: v1.0), reward-hack taxonomy, scoring baselines, SOLAR |
| [portal-guide] | https://research.nvidia.com/benchmarks/sol-execbench/api/blog/submission-guide (post dated 2026-03-11) | formats, evaluation environment, prohibited behaviour, score = mean over workloads, rate limits |
| [portal-v1.1] | https://research.nvidia.com/benchmarks/sol-execbench/api/blog/evaluation-stack-v1-1 (post dated 2026-07-09) | v1.0 and v1.1 not comparable; 7 hacks reported by doubleAI fixed |
| [leaderboard] | https://research.nvidia.com/benchmarks/sol-execbench/api (public `/kernels` and per-kernel leaderboard GETs), **fetched 2026-10-03**, cached in the session scratchpad | #38 anchors, rankings, `fast_1_count`, other norm/RoPE problems |

## NVIDIA documentation (SPEC)

| Key | URL | Used for |
|---|---|---|
| [cuda-guide-cc] | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (v13.4.2) | CC 10.0 limits: 32 CTAs, 64 warps, 2048 threads, 64K regs, 255 regs/thread, 228/227 KB smem, carve-outs, data types |
| [cuda-guide-async] | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html | TMA alignment table, bulk copies, tensor-map usage |
| [cuda-guide-pdl] | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html | PDL (CC ≥ 9.0) |
| [cuda-guide-l2] | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html | persisting-access semantics |
| [cuda-guide-pm] | https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html | clusters, DSMEM |
| [cuda-math] | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html | `rsqrtf` 2 ulp, `__frsqrt_rn` 0 ulp, fast-math effects |
| [blackwell-tuning] | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html | CC 10.0 occupancy, 228 KB smem, 1 KB reserved, 48 KB static limit, clusters 8/16, L2 126 MB, HBM ≤ 180 GB |
| [ptx-isa] | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | `rsqrt.approx.f32` error 2^-22.9; `.v8.b32` / `L2::evict_*` need sm_100 (PTX 8.8); `cp.async.bulk(.tensor)`, `mbarrier`, `createpolicy`, `prefetch.L2`, `griddepcontrol`, tcgen05/TMEM |
| [driver-tensormap] | https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html | `cuTensorMapEncodeTiled`: boxDim ≤ 256, inner box multiple of 16 B, swizzle limits |
| [datasheet] | NVIDIA HGX/DGX B200 product pages and datasheet (e.g. https://www.nvidia.com/en-us/data-center/hgx/) via search 2026-10-04 | 8 TB/s, 192 GB (product) vs 180 GB (HGX); tensor throughputs |
| [blackwell-arch] | NVIDIA Blackwell architecture page (https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/); launch coverage in AnandTech (https://www.anandtech.com/show/21310) | NV-HBI 10 TB/s die-to-die; 2 dies × 4 HBM3e stacks = 8192-bit bus |
| [h200/a100/l40s] | NVIDIA H200, A100, L40S datasheets; summarised by https://www.gpuperhour.com/compare/h200-sxm-vs-l40s | SMs, memory bandwidth, L2 (H200 50 MB; L40S 96 MB; A100 40 MB) |

## Third-party (LITERATURE)

| Key | URL | Used for |
|---|---|---|
| [chipsandcheese-b200] | https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut | 148 SMs (74 per die of 80), 126 MB L2, L2 about 150 ns local, 21 / 16.8 TB/s L2 BW, L1D 39 cycles (19.6 ns), VRAM latency higher than H100/A100, 1024 16-bit MAC/clk per partition, atomics latencies |
| [arxiv-2512.02189] | https://arxiv.org/abs/2512.02189 (v1: https://arxiv.org/html/2512.02189v1, v3: https://arxiv.org/html/2512.02189v3) | 148 SMs in 8 GPCs, TMEM 256 KB/SM, STREAM triad **7.48 TB/s (v1)** vs **4.14 TB/s (v3)**, latency claim vs H200 |
| [klockwood] | https://glennklockwood.com/garden/processors/b200 | 186 GB, 8 stacks (capacity disagreement only) |

## Disagreements between sources

1. **B200 achieved bandwidth**: arXiv 2512.02189 v1 reports 7.48 TB/s (STREAM triad); v3 of the same paper reports
   4.14 TB/s (51.8%). We plan on 85–92% of 8 TB/s and will calibrate from portal results.
2. **B200 DRAM latency**: Chips and Cheese says it is higher than H100/A100; arXiv v1 says 58% lower than H200 on cache
   misses. Unknown; we plan on 800 ns (700–1000).
3. **HBM capacity**: 192 GB (product, arXiv) vs 180 GB (tuning guide, HGX) vs 186 GB (klockwood). It does not matter
   here.
4. **Timing protocol**: the paper says CUDA events and the mean of 3 trials; the portal guide says 3 trials × 50 timed
   iterations, mean across trials; the v1.1 code uses CUPTI and returns the median of 50. We assume portal = mean over
   3 trials of the CUPTI median.
5. **Scoring baseline Tb**: the paper describes an agentic PyTorch-only optimised baseline (hidden); the portal guide
   says "from a PyTorch reference implementation". The leaderboard shows the baseline (31.8 µs) and the reference
   (239.6 µs) as separate entries, which matches the paper.
6. **SOL bound**: SOLAR's 9.195 µs for #38 is below the 8 TB/s read+write floor (16.0 µs geometric mean). On some
   other problems entries beat SOL (L1 `062_kv_cache_update_with_rope_backward`). SOLAR is not a physical lower bound.

## Added 2026-10-07 for `b200_sota.md` and `examples/` (access date 2026-10-07)

Licences are given for code we excerpt or summarise. Clones are in the session scratchpad (`research_raw/`); none
were executed.

### Third-party code and write-ups (LITERATURE)

| Key | URL | Used for |
|---|---|---|
| [quack] | https://github.com/Dao-AILab/quack, commit `35266c3298f0` (2026-09-12): `quack/rmsnorm.py`, `quack/rmsnorm_config.py`, `quack/copy_utils.py`, `quack/reduction_base.py`, `quack/gemm_base.py` (L2-hint note), `quack/tile_scheduler.py`. Apache-2.0 | N=128 forward config (128 threads, 16 threads/row, 128-bit cp.async → smem); same ladder on Hopper and Blackwell; no cache hints; sm100 GEMM "evict_first D stores + evict_last B loads → net REGRESSION (−0.9% / −3.5%)"; excerpt `examples/quack_rmsnorm_fwd_excerpt.py` |
| [quack-blog] | https://sysml.cs.princeton.edu/blogs/memory-bound-kernels.html (Guo, Zadouri, Dao; copy in repo `media/2025-07-10-membound-sol.md`) | memory-bound SOL recipe (vectorised coalesced loads, hierarchical reduction, clusters for long rows); H100 only: ≈3 TB/s (≈90%) for N ≥ 4 K |
| [quack-notes] | quack `AI/global_memory_coalescing_notes.md`, `AI/hadamard_roofline_analysis.md`, `microbenchmarks/global_memory_coalescing.py` (same commit). Apache-2.0 | H100: contiguous 512 B per warp instruction is fastest; four far 128 B pieces ≈8% slower; sparse stores much worse; `torch.clone` = 90.4% of H100 peak as the empirical ceiling; CuTe DSL inline-PTX load/store pattern (excerpt `examples/quack_cutedsl_inline_ptx_excerpt.py`) |
| [fi-qknorm] | https://github.com/flashinfer-ai/flashinfer, commit `b6b4f1ec35c4` (2026-10-07), `flashinfer/norm/kernels/rmsnorm.py` (`QKRMSNormKernel`), `flashinfer/norm/utils.py`. Apache-2.0 | CuTe DSL QK RMSNorm: `_LATENCY_BOUND_SMS = (100, 103, 107)`, 4 threads/row for head_dim 128 on Blackwell, cp.async → smem, no hints; excerpt `examples/flashinfer_qk_rmsnorm_sm100_excerpt.py` |
| [fi-norm-cuh] | FlashInfer `include/flashinfer/norm.cuh` (same commit). Apache-2.0 | CUDA `QKRMSNormKernel`: warp per (token, head), 128-bit, 4 warps/CTA, occupancy-sized grid-stride grid, PDL |
| [fi-pr5305] | https://github.com/flashinfer-ai/flashinfer/pull/5305 (merged 2026-09-18) | B200 qk_rmsnorm, bf16: 1.14× (M=8192), 1.20× (M=32768), 0.98–1.00× (M ≤ 512) for head_dim 128 from fewer threads per row; "adding threads per CTA … monotonically worse"; "only 32 KB in flight per SM" diagnosis |
| [fi-pr2777] | https://github.com/flashinfer-ai/flashinfer/pull/2777 (merged 2026-03-17) | CuTe DSL norm rewrite: multi-row CTAs with cp.async, cluster reduction for large H; B200/H200 heatmaps are images only (no numbers extracted) |
| [cake-pr5741] | https://github.com/flashinfer-ai/flashinfer/pull/5741 (merged 2026-10-05); kernels `csrc/cake_rmsnorm_train/*_kernel.cu`, table `flashinfer/jit/cake_rmsnorm_train.py` (commit `b6b4f1ec35c4`). Apache-2.0 | B200/GB300/R200 CUPTI cold-L2 RMSNorm times (table in `b200_sota.md` §2); per-size program tables; x loads with EVICT_FIRST policy constant, `.nc` weights, plain stores; 256-bit "not faster"; evict-first looped variant "no gain" (GB300); pipelined persistent 0.96–1.00× at large sizes; 5–9% launch-order noise under 10 µs; excerpt `examples/cake_rmsnorm_fwd_l2hint_excerpt.cu` |
| [fi-pr5692] | https://github.com/flashinfer-ai/flashinfer/pull/5692 (2026-09-29) | evict_first on a once-read weight stream: 1.07–1.11× on B200 because it protects re-read activations in L2 (a reuse case #38 does not have) |
| [sglang-flux2] | https://github.com/sgl-project/sglang, commit `0b635266d4a0` (2026-10-07): `python/sglang/kernels/kda_kernels/csrc/diffusion/flux2_qkv_epilogue.cuh`, `python/sglang/kernels/kda_kernels/README.md`, `python/sglang/kernels/ops/diffusion/README.md`; docs https://docs.sglang.io/docs/sglang-diffusion/fused_kernels.md. Apache-2.0 | production FLUX.2 QK RMSNorm(+RoPE) by a "Kernel Design Agent": warp per (token, head), persistent occupancy grid, no hints; excerpt `examples/sglang_flux2_qknorm_excerpt.cuh` |
| [trtllm-qknorm] | https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu (main, fetched 2026-10-07). Apache-2.0 | fused QK-norm+RoPE: warp per (token, head), bf16 packed vectors |
| [tk] | https://github.com/HazyResearch/ThunderKittens `kernels/layernorm/layernorm.cu` (HEAD, pushed 2026-09-12). MIT | Hopper-era layernorm (2 warps, double-buffered async loads); no Blackwell memory-bound kernels |
| [te-3601] | https://github.com/NVIDIA/TransformerEngine/issues/3601 and PR #3602 (2026-10-01). Apache-2.0 | ptxas 12.9 accepts `ld.global.nc.L2::evict_first` only with `.v8.b32`/`.v4.b64` (256-bit); 12.8 rejects it on `ld`; `createpolicy` + `.L2::cache_hint` works at any width |
| [cursor-blog] | https://cursor.com/blog/multi-agent-kernels (2026-04-14) | multi-agent SOL-ExecBench work; no memory-bound technique details |
| [cursor-results] | https://github.com/anysphere/kernel-optimization-results (pushed 2026-04-14), `L1/038_flux_multi_head_rmsnorm_qk/{solution.json,traces.jsonl}`, `problem_level_metrics.csv`. **No licence file: read only, nothing excerpted** | public #38 solution (CUDA, warp per (token, head), float4, 1024-thread CTAs, SOL 0.551); v1.0-harness B200 times 5.5 µs (12.6 MB) … 128.0 µs (805 MB) |
| [sf-tensor] | https://sf-tensor.com/news/sol-execbench (2026-09-25) | #1 overall team write-up: no kernel-level details for norms; ≈5% environment noise; 32 kernels ≥ 0.95 |
| [kernelarc] | https://arxiv.org/abs/2608.17071 (2026-08-17) | multi-agent framework evaluated on SOL-ExecBench (H100/B200); abstract only, no memory-bound details |
| [hazy-megakernel] | https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles | H100 launch cost ≈2.1 µs with streams, ≈1.3 µs with CUDA graphs (CPU-side; context for gaps) |
| [ascend-l2] | https://github.com/triton-lang/triton-ascend/pull/2349 (2026-09-22) | a zero-fill L2 flush leaves dirty lines whose write-back is charged to the next kernel (Ascend NPU, not NVIDIA); read-only `sum()` flush as the fix |
| [nv-forum-l2] | https://forums.developer.nvidia.com/t/flushing-dirty-l2-cache-lines/258812 | NVIDIA (R. Crovella): no CUDA method to flush dirty L2 lines |
| [sglang-l2rule] | https://github.com/sgl-project/sglang/pull/41545 (2026-09-28) | SGLang kernel-benchmark rule: measure with cold L2; no NVIDIA dirty-line data |
| [arxiv-2605.04178] | https://arxiv.org/abs/2605.04178 (HTML https://arxiv.org/html/2605.04178) | "sustained HBM is 6.8–7.1 TB/s vs. 8.0" on B200; low weight (its table lists 176 SMs and a 64 MB L2) |
| [arxiv-2507.10789] | https://arxiv.org/abs/2507.10789 | "Dissecting the NVIDIA Blackwell Architecture": RTX 5080 (GB203) and H100 PCIe, **not B200**; not used for B200 facts |
| [arxiv-2512.02189-recheck] | https://arxiv.org/html/2512.02189v1, …v3 (re-read) | the "58% reduction in memory access latency in cache-misses" is TMEM (420 cycles) vs Hopper's 1000-cycle global memory, not B200 DRAM; v3 STREAM 4.14 TB/s is blamed on 4–16 GB arrays being too small |
| [chipsandcheese-b200-recheck] | https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut (re-read) | "Perhaps Nvidia's scheduler tries to fill one partition's SMs before going to the other"; no VRAM bandwidth/latency number in the text |

### NVIDIA code (SPEC)

| Key | URL | Used for |
|---|---|---|
| [cutlass-cachehint] | https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/include/cute/arch/copy_sm90_desc.hpp (2026-09-23). BSD-3-Clause | `CacheHintSm90`: EVICT_NORMAL `0x1000000000000000`, EVICT_FIRST `0x12F0000000000000`, EVICT_LAST `0x14F0000000000000` |
| [cutlass-dsl-ex] | https://github.com/NVIDIA/cutlass/tree/0b55a2f691d69981583568fd9eb69687b1f0de8a/examples/python/CuTeDSL (`cute/blackwell/kernel/rmsnorm/rmsnorm.py`, `cute/blackwell/tutorial/tutorial_tma/`, `dsl_tutorials/programmatic_dependent_launch.py`, `cute/notebooks/elementwise_add.ipynb`). BSD-3-Clause | Blackwell RMSNorm example is quack-derived (128-bit, cp.async, clusters for large N); TMA tutorial's peak "2048 B/clk × 4000 MHz = 8.192 TB/s" |

### Disagreements found 2026-10-07

7. **Why r3 beat r2/g2 on B200**: `problem_038.md` §7 credits 256-bit width. Our per-workload data show r2 (256-bit,
   hinted) ≈ g2 (128-bit, hinted), and r3 (256-bit, unhinted) is 2–5% faster at M/L, which points to the cache hints (see
   `b200_sota.md` §3). This is unconfirmed until one A/B isolates it.
8. **evict_first on stores**: `playbook_membound.md` §6 recommends it (`cache:stream`). On B200, quack (GEMM), CAKE
   (paired search) and our portal data all point against it.
9. **arXiv 2512.02189 latency claim**: our docs read it as B200 DRAM latency; the paper's number is TMEM latency.
