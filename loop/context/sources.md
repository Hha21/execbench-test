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
