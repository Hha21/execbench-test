# Harness, timing and scoring (SOL-ExecBench v1.1)

Source of truth: the harness clone at `/home/lain/execbench-test/SOL-ExecBench` (commit `a9fa080`, "Add v1.1 timing
methodology"), the paper LaTeX, and the portal's public submission guide (read 2026-10-04). File paths below are relative
to `SOL-ExecBench/src/sol_execbench/`. The portal may run a newer verifier than this clone; the v1.1 post says doubleAI
reported 7 reward hacks that were then closed. Where the portal text and the code differ, both are given.

## 1. Pinned evaluation environment

| Component | Version | Evidence |
|---|---|---|
| Base image | `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` (nvcc 13.1) | docker/Dockerfile [PAPER] |
| PyTorch | 2.9.0 from the cu130 index (`torch.version.cuda` = 13.0) | pyproject.toml; CSF3 venv [PAPER, VERIFIED-CSF3] |
| Triton | `fbtriton==3.7.1` (replaces `triton`; `triton.__version__` = `3.7.0+fb.beta`) | Dockerfile; CSF3 [PAPER, VERIFIED-CSF3] |
| ptxas used by Triton | `ptxas-blackwell` 13.1 for arch ≥ 100, `ptxas` 12.9 otherwise | `triton/backends/nvidia/compiler.py:get_ptxas` [VERIFIED-CSF3] |
| CUTLASS (C++ headers) | v4.4.1 at `/usr/local/cutlass` (`CUTLASS_DIR`) | Dockerfile [PAPER] |
| CuTe DSL | nvidia-cutlass-dsl 4.4.2 (+ libs-cu13 4.4.2) | pyproject, Dockerfile [PAPER] |
| Others | cuda-python/bindings 13.1.1, cuda-tile 1.2.0, cupy-cuda13x 14.0.1, cudnn-frontend 1.18.0, cupti-python ≥13.0.1, Python 3.12 | pyproject.toml [PAPER] |

Packaging gap: the fbtriton wheel ships `triton/runtime/launch.h`, but its NVIDIA driver looks in
`triton/backends/nvidia/`. Every Triton launch on CSF3 failed until we added a symlink (PoC, CSF3) [VERIFIED-CSF3].
Whether the portal image has the same gap is **unknown**; the first portal submission tests it. fbtriton also ships a
ctypes launcher that needs no `launch.h` (`TRITON_USE_NO_COMPILE_LAUNCHER=1`, `backends/nvidia/no_compile_launcher.md`)
[VERIFIED-CSF3]. Setting that variable from a solution changes the environment, so ask the user before relying on it.

## 2. Calling convention and solution format

- DPS (destination-passing style) is the default and what we use: `run(*inputs, *outputs)`, positional, inputs in
  `definition.inputs` order, then outputs in `definition.outputs` order. Write the outputs in place and return nothing.
  Scalars arrive as Python numbers. [PAPER: eval_driver.py `_call_and_collect_outputs`, docs/solution.md]
- #38: `run(query, key, weight_q, weight_k, eps, query_norm, key_norm)`. Shapes are `[B,S,48,128]` fp32 (contiguous),
  `[48,128]` fp32 for the weights, and `eps` is a Python float (1e-6).
- Solution JSON. Required: `spec`, `sources`. The local schema also requires `name`, `definition` and `author`; the
  portal fills these in. A working example is `poc/variants/v000.json`:

```json
{"name": "rmsnorm_qk_triton_v000", "definition": "flux_multi_head_rmsnorm_qk", "author": "solx-poc",
 "description": "…",
 "spec": {"languages": ["triton"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.py::run",
          "dependencies": ["torch", "triton"], "destination_passing_style": true},
 "sources": [{"path": "kernel.py", "content": "<full source>"}]}
```

- Languages: `pytorch`, `triton`, `cute_dsl`, `cutile`, `cudnn_frontend` (Python, `.py` entry), and `cuda_cpp`,
  `cutlass`, `cudnn`, `cublas` (C++, entry `.cu`/`.cpp`). C++ and Python languages cannot be mixed in one solution.
  [PAPER: core/data/solution.py]
- C++ solutions are built by `torch.utils.cpp_extension.load(name="benchmark_kernel")`. Bind with
  `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`; the portal guide writes `benchmark_kernel`, which is the same name.
  `compile_options` defaults are `cuda_cflags=["-O3","--use_fast_math"]` and `ld_flags=["-lcuda"]`; the portal says
  your flags are added to the defaults. If you set no `-arch`/`-gencode`, the packager adds
  `-gencode=arch=compute_100a,code=sm_100a` for B200. Include paths are `$CUTLASS_DIR/include` and
  `tools/util/include`. Launch on `at::cuda::getCurrentCUDAStream()`. [PAPER: driver/templates/build_ext.py,
  driver/problem_packager.py, examples/cuda_cpp]
- Static `__shared__` is limited to 48 KB per CTA. Above that, use dynamic shared memory with
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` (up to 227 KB). [SPEC: Blackwell
  tuning guide]
- CuTe DSL: compile once at module level (`cute.compile`) and use `mark_layout_dynamic()` so all workloads share one
  compile. [PAPER: docs/solution.md]
- Portal upload formats: `.py`, `.cpp`, archives (`.zip`/`.tar.gz`; the entry must be `submission.py`, or a `.cpp`
  containing `PYBIND11_MODULE`), or `.json` (preferred, because it states the language and DPS explicitly). [PAPER: portal guide]

## 3. Correctness checks (per workload)

From `driver/templates/eval_driver.py` and `core/bench/correctness.py` [PAPER]:

1. 10 rounds. Each round calls `gen_inputs` to draw fresh random inputs (the RNG is seeded once per process with
   `seed=200`), runs the reference (return-value style), then runs your `run` with freshly zeroed outputs.
2. Round 0 only: integrity check of the eval-driver functions; snapshot of the thread count; lazy-output check
   (`type(t) is torch.Tensor`); exact output shape and dtype.
3. Every round: any NaN/Inf in your output or the reference fails. An all-zero output where the reference is non-zero
   fails. An element passes if `|out − ref| ≤ atol + rtol·|ref|`. The workload passes if at least
   `required_matched_ratio` of elements pass (default 0.99) and, if set, `max_abs ≤ max_error_cap`.
4. #38 tolerances: `max_atol = max_rtol = 1e-5`, other fields at their defaults.
5. Input generators that matter for row-wise problems (`core/bench/io.py`): fp32 tensors are `torch.randn`. A name
   equal to `weight`/`norm_weight`, or ending `_norm_weight`, gets ones; `cos`/`sin` names get values in range; and so
   on. **#38's `weight_q`/`weight_k` match none of these patterns, so they are N(0,1)**, which includes negative and
   near-zero weights.

## 4. Timing (v1.1 code)

`core/bench/timing.py::time_runnable(methodology="cupti")` [PAPER]:

1. `ShiftingMemoryPoolAllocator(inputs, outputs, warmup+rep+1, seed)`. For each call it copies the inputs into a
   pre-allocated pool at an offset that advances by a seeded random 256–2048 B (a multiple of 256), and it zero-fills
   the outputs at the same offset. Every call therefore sees **new `data_ptr`s**. They stay 256-B aligned and
   contiguous (strides are kept).
2. Before each call: `cudaCtxResetPersistingL2Cache`, then `buffer.zero_()` over `2 × L2_cache_size` bytes, which is
   about 252 MB on B200 (the paper says 256 MB).
3. 10 warm-up calls with a synchronise after each. Then one **discovery** call under CUPTI that records the ordered
   list of your GPU activities (kernel names, memcpys and memsets, with sizes).
4. 50 timed calls (`iterations` = 50). The setup is queued **without** synchronising, then the CPU timestamp is taken,
   then `run(...)`, then `torch.cuda.synchronize()`. In each call's CPU window the harness finds the discovered sequence
   and records `max(end) − min(start)` over those activities.
5. The local code returns the **median** of the 50. The portal guide says "3 trials, mean across trials" (the paper
   says CUDA events, which is the v1.0 method). Assume the per-workload number is the mean over 3 trials of the median
   of 50. That combination is INFERRED.
6. Clock lock presets: B200 SM 1500 MHz and DRAM 3996 MHz (H100 1410/1593, A100 1065/1215). These are applied by the
   container entrypoint. [PAPER: core/bench/config/device_config.py]

Consequences [INFERRED from the code above]:

- **CPU launch overhead is not timed** for a single kernel. Host-side work (tensor-map encoding, Python dispatch) is
  free in score terms.
- **Gaps between your kernels are timed.** The setup queued just before your call (input copies, output zero, the 252
  MB zero-fill) gives the CPU tens of µs of head start, so back-to-back launches are usually queued before the first
  kernel ends. Even so, each extra kernel adds at least the GPU's kernel-to-kernel gap plus a second ramp-up and tail.
  PoC (CSF3), one matched pair only: two launches instead of one cost +19 µs on A100 and −1 µs on L40S at the smallest
  sizes. The evidence is thin, but avoid splitting.
- The kernel sequence must repeat exactly. Data- or call-count-dependent dispatch, or autotuning on a later call,
  breaks the CUPTI matching (`assert kernel_activity_counts(...)`) and gives "Timing failed".
- Don't launch torch kernels from `run()` (`fill_`, `zero_`, `copy_`, `contiguous()`, `empty_like().zero_()`). They
  are timed, and their names can collide with the harness's own setup kernels in CUPTI matching.
- **Cold L2:** inputs always come from DRAM. The flush is a write (`zero_`), so the L2 probably holds about 126 MB of
  dirty lines when your kernel starts. Every line you allocate may then force a write-back, so assume the full
  read+write DRAM traffic even for small workloads. Unknown; it can be tested on H200 by comparing a reads-only kernel
  against reads+writes.
- Outputs arrive zero-filled, but you must still write every element.
- The first touch of the weights (24 KB per tensor) misses to DRAM; later CTAs hit in L2.

## 5. Score and aggregate

- Per workload: `S(t) = (Tb − Tsol)/((t − Tsol) + (Tb − Tsol))` (`sol_score.py`). Tb and Tsol come from precomputed
  per-workload tables; the reference is not timed in your run. [PAPER: portal guide]
- Problem score = arithmetic mean of the per-workload S; failed workloads count as 0. [PAPER: portal guide]
- Leaderboard "latency" is the **geometric mean of the per-workload latencies**; "Avg Speedup" is the **geometric
  mean of baseline/latency** per workload; "Fast₁" is the **number of workloads faster than the baseline**
  [VERIFIED: our portal submission pages, 6 Oct 2026. #61441: geomean 28.564 µs vs 28.562 shown, mean of workload
  scores 0.531119 = page score; #61435: geomean speedup 0.885 vs 0.89 shown, 5/16 faster]. So the score is averaged
  per workload, not computed from the aggregate latency, and the two can rank entries differently.
- Your own submission page shows, per workload: latency, the hidden **baseline latency Tb**, speedup and score.
  Tsol per workload then follows from `Tsol = (Tb − S·t − S·Tb)/(1 − 2S)` (ill-conditioned when S ≈ 0.5).
  `poc/ingest_portal.py` parses saved pages into `poc/results/b200_portal_workloads.csv`.
- Tb is the "Scoring Baseline": the fastest PyTorch-only solution found by NVIDIA's agentic system, hidden and allowed
  to change over time. The portal guide calls it "from a PyTorch reference implementation". [PAPER]
- Tsol (SOLAR) is `max(FLOPs/peak, fused bytes/BW)` at 1.5 GHz. It can sit below what is physically reachable. For #38
  it is below the read+write bandwidth floor (see the problem card) [INFERRED], and on other problems some entries beat
  it (for example L1 `062_kv_cache_update_with_rope_backward`: the leader is at 0.0098 ms against SOL 0.0175 ms,
  leaderboard 2026-10-03).
- Scores are not comparable between evaluation stacks v1.0 and v1.1. [PAPER: portal post]

## 6. Legitimate vs forbidden

| Legitimate | Forbidden (and how it is caught) |
|---|---|
| One fused kernel per call; per-size dispatch by shape (`B·S` bands) to different configs or kernels | dispatch on data values, pointers or call count (caching; LLM judge) |
| Module-level caches of compiled kernels or config tables keyed by shape/dtype | caches of outputs or intermediates; anything keyed on `data_ptr` (shifting allocator; 10 random rounds; judge) |
| Host-side TMA descriptors built each call | reusing descriptors across calls (they point at the old addresses: wrong results) |
| Cache-eviction hints, L2 prefetch hints inside your kernel | changing L2 persistence windows or device limits (environment manipulation; the harness resets persisting lines anyway) |
| Programmatic dependent launch between **your** kernels on the current stream | other streams, events tricks, CUDA graphs that hide work (CUPTI window plus synchronise; streams disallowed) |
| fp32 maths with any rounding order; approximate rsqrt (error ≤ 2^-22.9); TF32 compute only where the problem summary's note records an operator ruling for it (#35) | storing or computing in bf16/fp16/TF32 otherwise (tolerance; judge) |
| Triton, Gluon/TLX (ship with fbtriton), CuTe DSL, CUDA C++ with inline PTX | embedded cubin/ELF, `cuModuleLoadData` from bytes, `cpp_extension.load_inline` at runtime (blocked; judge) |
| `@triton.autotune` keyed on shapes (runs during round 0) | autotuning or recompiling during the timed loop; threads; `torch.jit.fork` |

Note on autotune: it is allowed, but it adds compile and benchmark time inside the 300 s per-solution limit (paper)
and it picks a config from noisy local timings. Prefer an explicit dispatch table.

## 7. Portal operations

- One kernel and one GPU type per submission, public or private. The first 5 submissions per UTC day return results
  without delay. The 6th waits 10 min, the 7th 20 min, and so on (+10 min each, capped at 60). At most 5 in flight.
  Immediate system failures are free. [PAPER: portal guide]
- Collection submissions (`.py` only, per-kernel directories): 2 per day, 6 h delay.
- Noise: about 5% between runs on the portal, up to 10% for many small kernels (README). So differences under 5%
  between two single portal results are not significant. Repeat or pair submissions before concluding.
- Local runs on CSF3: `sol-execbench <problem_dir> --solution sol.json` in `/scratch/t95317ha/solx` (env.sh). There
  are no clock locks (no sudo), and other GPUs are used, so treat local timings only as relative signals.
