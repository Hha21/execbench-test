# Generation protocol for the inner LLM

The inner LLM is the loop's **mutation operator**. Each call proposes a small number of kernels (default 1, at most 3).
Each one comes with a design card stating what it changes and what it should do on B200. The loop compiles, checks,
measures and archives them. This document fixes what the LLM receives, what it may do, and the exact output format.

## 1. Inputs per call

The loop assembles the prompt in this order (template in §6):

| Block | Content | Always? |
|---|---|---|
| `core_brief.md` | objective, timing, rules, B200 facts, contract | yes |
| task header | call type, operation, target niche and size band, number of candidates, round id | yes |
| problem card | the problem card (`problems/<name>/card.md`), or its §1–3 + §6 only for knob mutations | yes |
| reference docs | chosen per call type (README table): `playbook_membound.md`, `b200_arch.md`, `harness_scoring.md` | per call type |
| archive summary | one row per niche: best kernel, measured and predicted results with uncertainty (format §4) | yes |
| parents | 1–2 kernels: full source, design card, per-band measurements, sm_100a static features | for mutation, crossover, port, repair |
| feedback | compiler output, correctness errors, or lint hits from the previous attempt | for repair |

## 2. Allowed operations

| Operation | What may change | Constraints |
|---|---|---|
| `knob_mutation` | only constexprs and launch parameters (ROWS, num_warps, STAGES, PROGS_PER_SM, band thresholds, hints) | niche tags unchanged; state the expected direction per band |
| `structural_mutation` | one design axis (memory path, grid, store path, reduction, tile scheme, launch structure) | changes 1–2 niche tags; one hypothesis |
| `crossover` | combine the axes of two parents (for example A's TMA ring and B's S-band dispatcher) | list which axis came from which parent |
| `specialisation` | a kernel tuned for one size band, plus or including a `dispatch:size` wrapper | `spec:` tag set; correct for **all** shapes (the dispatcher may route other bands elsewhere) |
| `port` | the same algorithm in another language (Triton → CUDA C++, Gluon, CuTe DSL) | niche tags unchanged except `lang:`; say what the port enables (for example `mem:ldg256`) |
| `repair` | fix compile, correctness or lint failures | minimal diff; no performance changes unless required |
| `new_design` | a fresh design for a niche not yet in the archive | must name the empty niche it fills |

Every candidate must:

- be correct for every workload shape;
- launch exactly one kernel per call, unless the card argues for `launch:split` + PDL;
- keep fp32 maths;
- launch on the current stream;
- hold no state between calls except compiled kernels and shape-keyed config tables;
- contain nothing on the forbidden list (`core_brief.md` §3).

If a requested operation can only be done by breaking a rule, return no candidate and say why in the rationale.

## 3. Output format (exact)

The response contains, in this order:

1. Optionally, a `### Rationale` heading with ≤ 150 words.
2. For each candidate, three kinds of fenced block, in this order:
   - one ```` ```json solution-spec ```` block: the solution JSON **without** `sources`;
   - one block per source file, with the info string `<lang> file=<relative path>` (for example
     ```` ```python file=kernel.py ````, ```` ```cuda file=kernel.cu ````, ```` ```cpp file=binding.cpp ````). The
     content is the complete file;
   - one ```` ```yaml design-card ```` block.

The loop builds the solution JSON from these, in the format of `poc/variants/v000.json`, which the harness and portal
accept:

```json
{"name": "<card.id>", "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": "<card.hypothesis>",
 "spec": {"languages": ["triton"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.py::run",
          "dependencies": ["torch", "triton"], "destination_passing_style": true},
 "sources": [{"path": "kernel.py", "content": "<file block>"}]}
```

(`definition` should match `definition.json`'s name, `038_flux_multi_head_rmsnorm_qk`. The PoC used
`flux_multi_head_rmsnorm_qk`, and the local CLI accepted it; the portal overwrites `name`, `definition`, `author` and
`target_hardware` anyway.)

Spec rules: `languages` is one of the harness names; Python and C++ languages cannot be mixed. `entry_point` is
`file::function`. `destination_passing_style: true`. For CUDA C++ add
`"compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}`. Do not set `-arch`/`-gencode`; the
packager injects sm_100a, plus the local arch when `LOCAL` is in `target_hardware`.

**CUDA C++ layout: two files.** Put the device code and a small `extern "C"` launcher in `kernel.cu` **with no
PyTorch headers** (`#include <cuda_runtime.h>` only), and the PyTorch glue (`torch/extension.h`, argument checks,
`CUDAGuard`, the current stream, `PYBIND11_MODULE`) in `binding.cpp`, which declares and calls the launcher; the entry
point is `binding.cpp::run`. The harness compiles and links every `.cu`/`.cpp` file, on the portal as on our B200.
Measured on the B200 test bench: a one-file build (nvcc parsing the PyTorch headers) takes 54–67 s; the two-file
build takes 18 s, and 9 s when `binding.cpp` is unchanged (it is cached). Same kernel, same timings.

### Design card schema (YAML)

```yaml
id: g2-tma-host-r16s4            # unique; lowercase, digits, dashes
parents: [v028]                  # ids from the archive; [] for new_design
operation: structural_mutation   # §2 names
language: triton                 # triton | gluon | tlx | cuda_cpp | cute_dsl
niche:                           # exactly one tag per axis (playbook §0)
  mem: tma-tensor                # ldg128 | ldg256 | cpasync | tma-tensor | bulk1d
  st: bulk                       # direct | bulk
  grid: persistent               # oneshot | persistent | persistent-wstat
  launch: fused                  # fused | split | pdl
  tile: rows16                   # rows<N> | token
  red: warp                      # warp | halfwarp | cta
  cache: default                 # default | stream
  spec: L                        # all | S | M | L | dispatch:size
hypothesis: >-
  One or two sentences: the mechanism and why B200 differs from where we can measure.
expected_effect:                 # predicted change in B200 time vs the first parent, per size band
  S: {pct: +5, confidence: low}  # pct > 0 means slower
  M: {pct: -2, confidence: low}
  L: {pct: -8, confidence: medium}
resources_sm100a:                # estimates; the loop overwrites them with the compiled values
  regs_per_thread: 46
  smem_per_cta_bytes: 106552
  threads_per_cta: 128
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 131072
  launches_per_call: 1
knobs: {ROWS: 16, STAGES: 4, PROGS_PER_SM: 2, num_warps: 4}
dispatch:                        # only for dispatch:size designs
  - {max_tokens: 600, kernel: _qk_rms, meta: {ROWS: 4, num_warps: 2}}
tests: [H8]                      # hypothesis ids from the ledger this candidate tests ([] if none)
findings:                        # REQUIRED in design sessions: what you measured or learned, one line each, with the
  - "probe: one wave of 1184 fat CTAs copies 805 MB in 117 us vs 120 us for 49k small CTAs [probe_b200]"
  - "dead end: prefetching the next tile into L2 evicts evict_last outputs at M sizes (+3%) [run_tests]"
                                 # source in brackets. Negative results count. They go into the lab notebook.
paths:                           # REQUIRED. What each code path does, for the emulator (portal = rented x correction)
  - {max_tokens: 600, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, x: ef, w: nc, st: el}
  - {max_tokens: null, lang: cuda, width: 256, threads: 256, rows: 16, grid: oneshot, st: el}
  # one entry per size range, smallest first; max_tokens is the largest B*S it serves (null = all above).
  # lang triton|cuda|cute; width 128|256; threads per CTA; rows per CTA; grid oneshot|persistent;
  # mem ldg|tma|cpasync; wstat true|false; launches 1|2; prefetch true|false;
  # x none|ef (Q/K loads); w none|el|nc (weight loads); st none|ef|el (output stores);
  # cluster 1|2|4 (thread-block cluster size at launch). Omitted = default. Every launch attribute you set
  # must appear here: the emulator cannot predict what it is not told.
runs_on:                         # can this exact code run, and is the timing representative of B200 behaviour?
  H200: {runs: true, representative: true}
  A100: {runs: true, representative: false, note: "Triton compiles a non-TMA fallback on sm_80 (runtime untested); correctness check only"}
  L40S: {runs: true, representative: false}
risks:
  - "106 KB smem limits to 2 CTAs/SM; in-flight bytes rely on STAGES"
measure_first: [H200]            # which cheap source is most informative for this candidate
compliance:                      # all must be true
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```

## 4. Archive summary format (what the LLM sees)

One row per niche, showing the best kernel in it, sorted by predicted B200 score. Times are geometric means per size
band in µs. Predicted values are the emulator's posterior mean ± 1 sd. `B200 meas` is filled in once the kernel has
been submitted.

| id | niche (mem/grid/spec) | lang | H200 S/M/L | A100 S/M/L | L40S S/M/L | pred B200 S/M/L ± sd | pred score ± sd | B200 meas (score) | status |
|---|---|---|---|---|---|---|---|---|---|
| v028 | ldg128/persistent/all | triton | pending | 16.5/81.0/322.1 | 37.2/223.0/860.3 | {{emulator}} | {{emulator}} | pending | queued for portal |

(The v028 A100/L40S numbers are real PoC band geomeans, CSF3; the other cells are placeholders.)

Below the table, the loop appends: the 3 most recent B200 results with the emulator's prior prediction (calibration
error), the niches that are empty, and any rule the last round broke.

## 5. Gates applied to every candidate (self-check before answering)

1. Parse: fences are present, the spec is valid JSON, the card validates (tags from `niche_tags.yaml`).
2. Lint (flags go to human review): `torch.cuda.Stream`, `torch.cuda.graph`/`CUDAGraph`, `threading`,
   `multiprocessing`, `concurrent.futures`, `torch.jit.fork`, `os.environ[...] =`, `load_inline`, `cuModuleLoadData`,
   `base64`, `ctypes` loads, `.half()`/`.bfloat16()`/`float16`/`bfloat16`/`allow_tf32`/`round_f32_to_tf32=True`, module
   globals holding tensors, `lru_cache`/dicts keyed on tensors or `data_ptr()`, torch ops inside `run()`
   (`torch.empty` is allowed; fills, copies and compute are not).
3. Compile for **sm_100a** without a GPU: Triton via `ASTSource` + `GPUTarget("cuda",100,32)` (see
   `probes/probe1.py`); CUDA via nvcc → PTX → `ptxas-blackwell`; CuTe DSL via `CUTE_DSL_ARCH=sm_100a`. Records regs,
   shared memory, spills and the SASS mix.
4. Correctness: the harness's `sol-execbench` CLI on an available GPU (H200 for TMA designs, any GPU otherwise), all 16
   workloads, 10 rounds each.
5. Timing: H200, plus A100/L40S where `runs_on` says representative. The emulator then predicts B200 per band, with an
   uncertainty.

A candidate that fails 1–4 is returned for `repair` with the log.

## 6. Prompt template

Fill the `{{…}}` slots; keep the section markers so the LLM can find things.

````text
=== CORE BRIEF ===
{{core_brief.md}}

=== TASK ===
Round: {{round_id}}   Call type: {{call_type}}   Operation: {{operation}}
Problem: {{problem_id}}   Target niche: {{target_niche or "any empty niche"}}   Target band: {{band or "all"}}
Return {{n_candidates}} candidate(s). Each must follow OUTPUT CONTRACT exactly.
{{extra_instructions, e.g. "Port parent g2-tma-host-r16s4 to CUDA C++ using a 1-D bulk ring (mem:bulk1d)."}}

=== PROBLEM CARD ===
{{the problem card or its sections 1-3,6}}

=== REFERENCE ===
{{selected docs per README table}}

=== ARCHIVE (best per niche) ===
{{archive table, section 4 format}}
Recent B200 results vs prior predictions: {{calibration lines}}
Empty niches: {{list}}

=== PARENTS ===
--- parent {{id}} ---
design card:
{{yaml}}
measurements (geomean µs per band; per-workload table optional): {{table}}
sm_100a static features: regs={{r}} smem={{s}} local={{l}} sass={{LDG/STG/UTMA counts}}
source:
{{full source}}
{{repeat for second parent}}

=== FEEDBACK FROM LAST ATTEMPT ===
{{compiler/correctness/lint output or "none"}}

=== OUTPUT CONTRACT ===
Optional "### Rationale" (<=150 words). Then, per candidate:
1) ```json solution-spec``` (solution JSON without "sources")
2) one ```<lang> file=<path>``` block per source file (complete files)
3) ```yaml design-card``` following the schema in the generation protocol.
No other code blocks. Do not use any forbidden behaviour; if the task needs one, return no candidate and explain.
````

### Call-type presets (which docs go into REFERENCE)

| Call type | REFERENCE contents | Parents |
|---|---|---|
| `new_design` (#38) | playbook (all), b200_arch §1–8 + §10 | 0–1 (best overall) |
| `knob_mutation` | playbook §0 + the axis being tuned | 1 |
| `structural_mutation` | playbook (all), b200_arch §3, §5, §8 | 1 |
| `crossover` | playbook §0, §9 | 2 |
| `specialisation` (S band) | playbook §2, §8, §9; b200_arch §8 | 1–2 |
| `port` to CUDA C++ / Gluon / CuTe DSL | harness_scoring §2; playbook §3b/3d/3e and §12; b200_arch §5 | 1 |
| `repair` | harness_scoring §2–4 | 1 (the failing one) |
| new memory-bound problem | harness_scoring (all), playbook (all), b200_arch §1–8, plus a new problem card | 0 |
