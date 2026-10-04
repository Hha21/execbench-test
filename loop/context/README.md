# Context documents for the inner LLM

These documents are what the inner LLM reads when it writes or mutates GPU kernels for the SOL-ExecBench B200
leaderboard (`poc/PLAN.md`: inner, middle and outer loops). The first target is L1 #38
(`038_flux_multi_head_rmsnorm_qk`). Material specific to #38 lives only in `problem_038.md`; everything else applies
to any memory-bound problem, and partly to later compute-bound ones.

## Files

| File | Purpose | ≈ tokens |
|---|---|---|
| `core_brief.md` | **Every call.** Objective (portal score), timing in brief, forbidden behaviour and its detection, B200 facts that change designs, toolchain, output contract, confidence labels | 1.9k |
| `problem_038.md` | #38 problem card: semantics and reference, numerics against 1e-5, the 16 workloads with bytes and floors, size bands, leaderboard anchors, score targets and sensitivity, PoC lessons, priorities, open questions | 3.3k |
| `playbook_membound.md` | design axes with niche tags; when each helps, effect on the S/M/L bands, compile-verified Triton 3.7 recipes, CUDA C++/Gluon/CuTe pointers, anti-patterns, porting notes | 6.2k |
| `b200_arch.md` | B200 for kernel writers: hierarchy numbers, Little's law, occupancy, TMA/mbarrier, clusters, cache hints, launch and small-input latency, compute-bound in brief, B200 vs H200/A100/L40S and what each can tell us | 4.3k |
| `harness_scoring.md` | pinned versions, calling convention and solution JSON, correctness checks, CUPTI timing and its consequences, score and aggregate (verified vs inferred), legitimate vs forbidden, portal rate limits | 3.6k |
| `generation_protocol.md` | per-call inputs, allowed operations, **exact output format**, design-card schema, archive table format, gates, the prompt template, call-type presets | 3.0k |
| `niche_tags.yaml` | machine-readable tag vocabulary and #38 size bands (for the archive and card validation) | 0.5k |
| `sources.md` | every source with path or URL and access date; disagreements between sources | 2.4k |
| `probes/` | compile-only probe scripts (Triton, Gluon, CUDA C++, CuTe DSL), raw results (`*.jsonl`), and `RESULTS.md` (summary, 1.5k) | — |

Token counts are estimated from bytes ÷ 3.7; tables and symbols run a little higher.

## What to include per call type

| Call type | Include (in this order) | ≈ tokens before archive/parents |
|---|---|---|
| New design for #38 | core_brief, problem_038, playbook (all), b200_arch §1–8 + §10, generation_protocol §2–3 | 17k |
| Knob mutation | core_brief, problem_038 §1–3 + §6, playbook §0 + the tuned axis, generation_protocol §3 | 6k |
| Structural mutation | core_brief, problem_038, playbook (all), b200_arch §3/§5/§8, generation_protocol §2–3 | 14k |
| Crossover | core_brief, problem_038 §3 + §6, playbook §0 + §9, generation_protocol §2–3 | 7k |
| Small-size specialist | core_brief, problem_038 §3 + §6 + §8, playbook §2 + §8 + §9, b200_arch §8, generation_protocol §3 | 7k |
| Port to CUDA C++ / Gluon / CuTe DSL | core_brief, harness_scoring §2, playbook §3b/3d/3e + §12, b200_arch §5, generation_protocol §3 | 9k |
| Repair (compile/correctness/lint) | core_brief, harness_scoring §2–4, generation_protocol §3 + §5, the error log | 6k |
| New memory-bound problem | core_brief, harness_scoring, playbook, b200_arch §1–8, a new problem card in the `problem_038.md` structure | 18k |
| Compute-bound problem (later) | core_brief, harness_scoring, b200_arch §1, §5, §9, a problem card; a compute playbook does not exist yet | 9k |

The template for assembling a prompt is in `generation_protocol.md` §6.

## Confidence labels

Every non-obvious fact carries one of these:

| Label | Meaning |
|---|---|
| VERIFIED-CSF3 | we measured or compiled it on CSF3. "PoC, CSF3" means the proof-of-concept measurements |
| SPEC | NVIDIA documentation or datasheet |
| PAPER | SOL-ExecBench paper, harness code, or the portal's public guide |
| LITERATURE | third-party papers or benchmarks |
| INFERRED | our reasoning; the text says from what |

"Compile-verified" means it compiled for sm_100a with the SASS inspected. It does not mean it ran: nothing here has
run on a B200.

## Keeping the documents current

| Trigger | Update |
|---|---|
| New B200 portal results | `problem_038.md` §7 (what we learned) and §9 (close questions); per-band efficiencies; recalibrated score targets if per-workload data appears. Update `b200_arch.md` §2/§8 if the results pin down bandwidth or fixed costs |
| Leaderboard refresh (weekly) | `problem_038.md` §4 table and date |
| Harness commit changes (`git -C SOL-ExecBench log`) | re-read `timing.py`, `eval_driver.py`, `io.py`, `correctness.py`; update `harness_scoring.md` §3–4 and the core brief §2–3 |
| Portal guide or evaluation-stack posts change | `harness_scoring.md` §5–7; `core_brief.md` §3 if a new rule or check appears |
| Toolchain pin changes (Dockerfile, pyproject) | re-run `probes/` on the CSF3 login node, update `probes/RESULTS.md` and every VERIFIED-CSF3 compiler claim |
| A recipe is run on H200 or B200 | upgrade "compile-verified" to measured in the playbook, with numbers and date |
| New problem | add `problem_<id>.md` modelled on `problem_038.md` and extend `niche_tags.yaml` size bands |

Keep `core_brief.md` under about 2,500 tokens. Move detail into the other documents rather than growing it.
