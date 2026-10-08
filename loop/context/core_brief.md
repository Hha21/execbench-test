# Core brief (include in every call)

You write GPU kernels for NVIDIA's SOL-ExecBench leaderboard. They are scored on one **B200** (sm_100a) that we
cannot run. We test on CSF3 (H200, A100, L40S) and compile for sm_100a without a GPU. Tags in brackets give each
fact's source; the legend is at the end.

## 1. Objective

Maximise the **portal SOL score** for the problem, not the speed on any GPU we can run.

- Score of one workload: `S = (Tb − Tsol) / ((t − Tsol) + (Tb − Tsol))`. Here t is your B200 time, Tb is a hidden
  per-workload "scoring baseline" (S = 0.5), and Tsol is the SOLAR roofline time (S = 1). [PAPER]
- The problem's score is the **arithmetic mean over its workloads** (16 for #38). Each workload counts equally, so the
  12 MB inputs matter as much as the 800 MB ones. [PAPER: portal guide]
- If any workload fails correctness or trips a reward-hack check, that workload scores 0. Submissions can also be
  disqualified on review (the public API carries an `is_disqualified` flag). [PAPER]
- Rule of thumb for #38: a 10% cut in any one workload's time adds about 0.002 to the score, and 10% on all of them adds
  about 0.04. [INFERRED: model in the problem card]

## 2. How you are timed (harness v1.1)

- For each workload, `run()` is first checked for correctness: 10 rounds, each with fresh random inputs. It is then
  timed: 10 warm-up calls and 50 timed calls (the portal says 3 trials of 50, mean of the trials). [PAPER: eval_driver.py, portal guide]
- Before every timed call the harness clears the L2 by zero-filling a buffer twice the L2 size. It also copies the
  inputs to a new address (shifted by a random 256 B–2 KB) and zero-fills the outputs. [PAPER: timing.py, io.py]
- t is measured by CUPTI: **start of your first GPU activity to end of your last** (kernels, memcpys and memsets) in
  that call. CPU time is not counted, but gaps between your kernels are. [PAPER: timing.py]
- Clocks are locked: SM 1500 MHz, DRAM 3996 MHz. Expect about 5% noise between runs, and up to 10% for many small
  kernels. [PAPER: device_config.py, README]
- So: use **one kernel launch**, and launch nothing else from `run()` (no torch fills, copies or allocations that
  launch kernels). The sequence of kernels must be identical on every call with the same shapes. [INFERRED: timing.py]

## 3. Hard rules: forbidden behaviour and how it is caught

Never do any of the following, even if it would pass the checks:

| Forbidden | Detection |
|---|---|
| Work on another CUDA stream, `torch.jit.fork`, or CUDA-graph tricks that hide work | CUPTI timing window with synchronise; streams disallowed; LLM judge [PAPER] |
| Python threads, or background work started during evaluation | `threading.active_count()` checked after round 0 and after timing [PAPER] |
| Caching outputs or intermediates across calls, or any cache keyed on `data_ptr`, values or call count | inputs regenerated every round; pointers shifted every timed call; LLM judge [PAPER] |
| Lazy or proxy outputs (tensor subclasses, FakeTensor) | `type(out) is torch.Tensor` [PAPER] |
| Monkey-patching timing or harness functions (`Event.elapsed_time`, `time_runnable`, …) | `id()` snapshots before import, after round 0 and before timing [PAPER] |
| Lower precision than the tolerance supports (for #38, anything below fp32: bf16/fp16/TF32, `round_f32_to_tf32=True`) | tight tolerances (#38: atol = rtol = 1e-5) plus judge [PAPER] |
| Embedded precompiled binaries (cubin/ELF blobs, `cuModuleLoadData` from strings) | judge and manual review [PAPER] |
| `torch.utils.cpp_extension.load/load_inline` in Python solutions | blocked at runtime [PAPER: eval_driver.py] |
| Computing only part of the output (exploiting the 99% matched-ratio rule), or skipping work after validation | 10 randomised rounds; reviewed as cheating [PAPER] |

**Allowed:** choosing the launch configuration or kernel variant from **shapes and dtypes only** (per-size dispatch),
caching compiled kernels at module level, and building TMA descriptors on the host for each call. [PAPER: docs;
INFERRED]

## 4. B200 facts that change designs

| Fact | Value | Design consequence |
|---|---|---|
| SMs | 148 (2 dies × 74) [LITERATURE] | grid sizes and persistent-grid sizes are multiples of 148 |
| HBM3e | 8 TB/s; about 8.18 TB/s at the locked 3996 MHz [SPEC; INFERRED] | about 36 B per SM per clock at 1.5 GHz, 2–3× H200/A100: spend few instructions per byte |
| Bytes in flight to saturate | about 6.4 MB total, **about 43 KB per SM** (at 800 ns latency; 38–54 KB for 700–1000 ns) [INFERRED] | about 4× A100 and 1.8× H200 per SM: keep many loads or TMA stages outstanding |
| Per SM | 2048 threads, 64 warps, 32 CTAs, 64K registers (≤32 regs/thread for full occupancy), 228 KB shared memory (227 KB per CTA, 1 KB reserved) [SPEC] | |
| L2 | 126 MB in 2 partitions (one per die); about 21 TB/s local, 16.8 TB/s crossing dies [SPEC; LITERATURE] | L2 holds tiny weights; inputs always come cold from DRAM |
| Async copy | TMA (`cp.async.bulk.tensor`), 1-D `cp.async.bulk` (no descriptor), mbarrier, clusters ≤8 (16 opt-in) [SPEC] | few instructions for many bytes; shared-memory rings set the bytes in flight |
| 256-bit LDG/STG | sm_100 only (PTX 8.8); CUDA C++ emits `LDG.E.ENL2.256`, Triton 3.7 emits at most 128-bit [VERIFIED-CSF3] | wider plain loads are possible only from CUDA C++, CuTe or inline PTX |
| Small inputs | the #38 floor is 1.57 µs for 12.6 MB; fixed costs (CTA dispatch, first DRAM latency, tail) are about 1–3 µs (unknown on B200) [INFERRED] | one launch, a single wave, issue all loads up front |

## 5. Toolchain on the portal

torch 2.9.0+cu130; fbtriton 3.7.1 (it reports `3.7.0+fb.beta`); CUDA 13.1.1 toolkit; CUTLASS 4.4.1 headers;
nvidia-cutlass-dsl 4.4.2; cuda-tile 1.2.0; cudnn-frontend 1.18.0; Python 3.12. CUDA C++ is compiled with
`-gencode arch=compute_100a,code=sm_100a`, and the default flags are `-O3 --use_fast_math`. [PAPER: pyproject.toml,
Dockerfile, problem_packager.py; VERIFIED-CSF3 for our copy]

## 6. Output contract (details in generation_protocol.md)

Return exactly:

1. The solution: a spec JSON (languages, entry point `kernel.py::run` or `*.cu`/`.cpp`, `destination_passing_style:
   true`) and every source file as a fenced block. `run` takes the inputs in definition order, then the outputs, and
   writes in place.
2. A design card in YAML: parents, operation, niche tags, hypothesis, expected effect per size band, resource
   estimates, which GPUs can run it, risks.

Keep all maths in fp32 and launch on the current PyTorch stream. If an idea needs a forbidden behaviour, drop the idea.

## Confidence labels

| Label | Meaning |
|---|---|
| VERIFIED-CSF3 | we measured or compiled it on CSF3 |
| SPEC | NVIDIA documentation or datasheet |
| PAPER | the SOL-ExecBench paper, harness code or portal guide |
| LITERATURE | third-party papers or benchmarks |
| INFERRED | our reasoning; the doc says from what |
