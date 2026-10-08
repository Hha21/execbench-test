# solx: evolving B200 kernels without a B200

An automated research loop in which LLM agents design, test and refine GPU kernels for NVIDIA's
[SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench) leaderboard. The target is problem
**#38 `038_flux_multi_head_rmsnorm_qk`**: per-head RMSNorm of the query and key tensors of FLUX attention, in fp32,
at 16 input sizes from 13 MB to 805 MB of traffic. Kernels are scored on the portal's B200 (clocks locked). We test
on a rented B200 (Modal, pay per second), correct its numbers with a multi-fidelity emulator trained on our own portal
results, and submit the best candidates to the portal by hand (about five results a day).

## Where we are

**0.6111**, faster than NVIDIA's hidden baseline at all 16 sizes (public #1: 0.628 when last checked).

```mermaid
xychart-beta
    title "B200 portal score (flat line: public #1, 0.628)"
    x-axis ["v028", "v039", "g2-os", "r3-cute", "r5-stel", "r6-s", "r10-rtok"]
    y-axis "SOL score" 0.40 --> 0.65
    line [0.451, 0.531, 0.577, 0.588, 0.609, 0.610, 0.611]
    line [0.628, 0.628, 0.628, 0.628, 0.628, 0.628, 0.628]
```

| Kernel | How it was made | B200 latency (geomean) | Portal score |
|---|---|---|---|
| Scoring baseline (hidden, NVIDIA) | - | 31.8 µs | 0.500 |
| `v028` | first submission: best of 72 Triton knob variants on A100 | 35.9 µs | 0.451 |
| `v039` | best knob variant on B200 (persistent grid) | 28.6 µs | 0.531 |
| `g2-os-r8w4` | generation 2: one-shot Triton kernel | 25.3 µs | 0.577 |
| `r3-cute-ldg256-os-r16` | round r3: CuTe DSL, 256-bit loads, no cache hints | 24.6 µs | 0.588 |
| `r5-ldg256-os-r16-stel` | round r5, first interactive session: outputs kept in L2 (`evict_last` stores) | 23.4 µs | 0.609 |
| `r6-s-ldef-nc-disp` / `c1` | small-input load policy; resubmitted unchanged as a control (0.6105) | 23.3 µs | 0.610 |
| `r10-rtok-m-disp` | exploratory round r10: 3 tokens per thread at 1024 tokens | **23.2 µs** | **0.611** |
| Leaderboard #1 (public) | - | 22.2 µs | 0.628 |

Large inputs run at about the B200's practical memory ceiling. What is left sits at small and medium sizes: a
clock-bound fixed cost of about 1 µs outside the kernel's blocks, and the write-back of dirty lines that the harness's
cache flush leaves in L2 (20-35% of in-kernel time). Details: [`loop/context/problem_038.md`](loop/context/problem_038.md)
and the hypothesis ledger [`loop/ledger.yaml`](loop/ledger.yaml).

## How the loop works

```mermaid
flowchart TB
    subgraph outer["Outer loop: the B200 portal (about 5 results a day)"]
        SL["round.py shortlist<br/>predicted portal score ± error,<br/>P(beats the best)"]
        SUB["you submit privately to #38"]
        ING["ingest_portal.py<br/>per-size latency, hidden baseline, score"]
        SL --> SUB --> ING
    end
    subgraph middle["Middle loop: research lead (each round)"]
        LEAD["research lead (Fable, headless Claude)<br/>reads the evidence, updates the ledger,<br/>designs experiments with success criteria"]
        LEDGER[("ledger.yaml<br/>hypotheses: open / supported / refuted")]
        EVID[("archive + lab notebook<br/>every kernel, portal vs bench outcome,<br/>findings incl. dead ends")]
        EVID --> LEAD
        LEDGER <--> LEAD
    end
    subgraph inner["Inner loop: design sessions (Opus / Sonnet, headless Claude, in parallel)"]
        SESS["design session<br/>hypothesis -> probe -> build -> measure -> revise"]
        TOOLS["MCP tools (mcp_tools.py)<br/>probe_b200 (1-6 s), compile_b200, run_tests (10-20 s)<br/>on a rented B200 (Modal), predict_score,<br/>get_kernel, read_example"]
        EMU["emulator.py<br/>portal = rented time x correction<br/>(size, design features, copy lag)"]
        SESS <-->|"tool calls"| TOOLS
        TOOLS --> EMU
    end
    LEAD -->|"experiments"| SESS
    SESS -->|"candidate + design card + findings"| EVID
    EVID --> SL
    ING --> EVID
    ING -->|"new training pair"| EMU
```

- **Research lead.** Once per round, a Fable agent reads the lab notebook (every portal result next to its bench
  result, and every finding recorded by earlier sessions, dead ends included), the archive, the hypothesis ledger and
  the emulator's learned portal-vs-bench effects. It rewrites the ledger and designs the round's experiments: each
  tests a hypothesis, states a success criterion and what would refute it, and names a model.
- **Design sessions.** Each experiment is a headless Claude Code session on the user's Claude plan, with our tools
  served over MCP. A session tests its idea's core assumption with quick probes on the rented B200 (an NVRTC kernel in
  about a second, timed exactly as NVIDIA's harness times it), builds the kernel, and runs NVIDIA's harness on it:
  correctness at all 16 sizes and a predicted portal score with error bars. It hands in one kernel with a design card
  (hypothesis, per-path design features, findings).
- **Emulator.** The rented B200 runs its SM clock at about 1940 MHz; the portal locks it at 1500 MHz. The emulator
  learns `log(portal / rented)` per size from every kernel measured on both (a Gaussian process over design features,
  plus the kernel's lag behind a plain copy, which measures how SM-bound it is). Typical error is ±0.005-0.009 in
  score; designs unlike anything measured get wider error bars.
- **Portal.** The shortlist ranks candidates by their chance of beating the best. Slots also go to deliberate
  experiments (A/B pairs, controls) because every portal result trains the emulator and settles hypotheses.

## Running a round

```bash
.venv/bin/python loop/round.py lead      --round r12 --backend claude --lead-model fable   # ledger + experiments
.venv/bin/python loop/round.py auto      --round r12 --use-plan --backend claude           # sessions + B200 test
.venv/bin/python loop/round.py shortlist --round r12                                        # -> rounds/r12/portal/
# after uploading: save each result page into html_results/, then
python3 poc/ingest_portal.py html_results/<page>.html
.venv/bin/python loop/emulator.py                                                           # refit + validation
```

`auto --lead` runs the lead and the sessions in one go; `--mode explore` uses the built-in idea list instead of the
lead; `--backend openrouter --interactive` runs the same sessions through OpenRouter. A round of four sessions takes
about 10-15 minutes, uses a few dollars of API-equivalent usage on the Claude plan and about $1-2 of B200 time.

## Repository layout

| Path | What it is |
|---|---|
| `loop/round.py` | the loop driver: lead, plan, propose, auto, test, collect, shortlist, table |
| `loop/ledger.yaml` | the hypothesis ledger (constraints, hypotheses with status and evidence) |
| `loop/mcp_tools.py`, `loop/designer.py` | the session tools (MCP server for headless Claude; OpenRouter tool loop) |
| `loop/b200_modal.py`, `loop/b200probe.py` | the rented B200 on Modal (NVIDIA's software stack) and the probe helpers |
| `loop/emulator.py`, `loop/emulator_features.yaml` | the multi-fidelity emulator and per-path design features |
| `loop/pair_timings.py`, `loop/calibrate_b200.py` | timing portal-measured kernels on the rented B200 |
| `loop/planner.py` | score model, task planning, explore ideas, shortlist |
| `loop/context/` | the agents' briefing: B200 architecture, public SOTA kernels, harness rules, problem card, playbook, protocol |
| `loop/archive.py`, `loop/archive/` | every kernel: card, timings per GPU, compile statistics, portal results |
| `loop/rounds/<r>/` | each round: plan, lead reply, session transcripts and tool logs, candidates, results, portal files |
| `loop/toolserver.py`, `loop/jobs/` | GPU tool server and Slurm jobs for CSF3 (A100/L40S/H200; optional now) |
| `poc/` | proof of concept: 72 variants, multi-GPU timing, the first emulator rehearsal, portal ingestion |
| `poc/results/b200_portal*.csv` | every B200 portal result, per submission and per size |

## Lessons so far

- **Keeping outputs in L2 is the biggest lever found.** `evict_last` output stores gained +0.021: output still in
  the 126 MB L2 when the kernel ends is written back after the timer stops.
- **The bench is not the portal.** A rented B200 boosts its SM clock; the portal locks it at 1500 MHz. Plain
  streaming transfers 1:1, but SM-bound designs lose more on the portal, and some changes flip sign (r8/r9: -1% on the
  bench, +3-4% on the portal). The emulator learns these corrections; changes below about 1% are only settled by the
  portal.
- **The portal is precise; the bench is noisy.** Identical code resubmitted moved at most 0.3% per size (times are
  reported in 0.1 µs steps). The rented B200 varies about 1% per size between runs.
- **Most structural ideas lose to the simple one-shot kernel.** Persistent grids, work stealing, TMA bulk pipelines,
  one wave of fat CTAs, FlashInfer's lane layout and die-affine placement were each measured and refuted.
- **Launch attributes matter at the smallest sizes.** A cluster-of-2 launch cut 128-256-token workloads by 2-6% on
  the portal; clustering larger sizes hurt.
- **Fast experiments change how the agents work.** With probes in seconds, sessions test assumptions before building,
  run controls, and record negative results, and the lab notebook stops later rounds repeating dead ends.
- **Portal arithmetic** (verified): latency is the geometric mean over sizes, the score is the arithmetic mean of
  per-size scores, and your own submission pages show the hidden baseline per size.

## Setup notes

- Local: `python3 -m venv --system-site-packages .venv && .venv/bin/pip install modal mcp`. Modal login:
  `.venv/bin/python -m modal setup`. The OpenRouter key (optional) is read from `.env` (git-ignored) and never printed.
- The B200 image follows SOL-ExecBench's Dockerfile (CUDA 13.1.1, CUTLASS 4.4.1, the uv-locked environment, fbtriton
  3.7.1). Our copy of the harness skips timing windows that hold none of the user's kernels (CPU/GPU timestamp skew
  on Modal; `loop/patch_harness_timing.py`); the portal never needs this.
- CUDA C++ solutions use two files (`kernel.cu` without PyTorch headers, `binding.cpp`), which cuts a B200 build
  from about a minute to 9-18 s.
- CSF3 (optional): everything lives in `/scratch/t95317ha/solx`; account `gpu-cdt-dmcs` for H200, `gpu-sk01` for
  A100/L40S.
- The portal has no official upload API, so submission stays manual.
