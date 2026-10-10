# Review of the solx kernel loop (2026-10-10)

Read-only review of the repository at `c5858e8`. Evidence is cited as `file:line`, data files, or commits. "Bench" means the rented Modal B200; "portal" means NVIDIA's scoring.

## Executive summary

1. **What works.** The probe-in-seconds loop, the hypothesis ledger with real refutations, controls (c1, c2) and the shape-split A/B on the portal, and the dirty-L2 / `evict_last` discovery are genuine research engineering. Nothing submitted breaks a rule I can find, and the legitimacy constraints the project set itself are stricter than the harness's.
2. **The emulator is calibrated only for #38.** On #38 it holds (leave-one-out rms z ~1.5 over 23 kernels, but -4.1 and -3.5 on structurally different kernels, and -2.8/-3.0 when r8+r9 are held out together, the realistic "new round" case). On the other three problems it trains on 1-3 kernels with #38-shaped features and still prints "P(beats best) 100%". Across the seven shortlist-to-portal pairs that exist, rms z = 2.7; the worst misses are #30 r1 (+5.8 sd) and #84 r1 (+3.2 sd).
3. **The bench cannot resolve the gains now being chased.** The x1 A/B showed the bench misranks 1-3% effects, which is the size of every remaining #38 gain (correctly paused) and of the #84 r2 experiments (0.05-0.2 µs on a 2 µs kernel). Compute-bound sizes are unrankable on the bench (power cap to ~1117 MHz, no clock lock), and no alternative B200 source has been explored.
4. **Sessions see one round of findings.** `prompts.lab_notebook` truncates at 40 lines sorted by round string: on #38 a session sees r14's 41 findings and none of the 162 from r10-r13; the lead sees r13-r14. The claim "the lab notebook stops later rounds repeating dead ends" holds for one round. The string sort ("r9" > "r14") will reorder #30/#84 wrongly from r10 on.
5. **Problem choice is structurally capped by the project's own findings.** #84: S band at the launch floor, L at 92% of DRAM, top 5 needs +0.015 and the ledger's remaining levers sum to about +0.01. #218: parked (stored Tb below any cuBLAS run). #30: the only identified lever worth +0.04-0.08, but its instrument is the portal, whose compute-band run-to-run drift (2-3% on #218) is as large as the gap to top 5, and no control resubmission has been done there.
6. **The cheap research phase (~$5, 15 min) is not used to screen boards.** The survey lists 20 boards with s5 < 0.60; four were picked before measuring whether a fused kernel can beat a torch.compile-class baseline there.
7. **Legitimacy process gaps.** The lint is #38-specific (`reply.py:34` flags the literal `bfloat16`), so agents now write `cutlass::bfloat##16_t` in submitted #30 code to evade it (`rounds/r1/portal/r1-oproj-cutlass-t224-dispatch.json`, kernel.cu:24). That is benign but looks evasive to a human reviewer. None of the ledger constraints (discard, applypriority, launch delay, PDL, persisting-L2 resets) is enforced in code, and `cmd_shortlist` copies anything into `portal/`.
8. **All 32 submissions are Private.** No public review (LLM judge, manual) has tested any design; "5th on the public board" is a comparison, not a rank.
9. **Prompt anchoring to #38.** Every problem gets 24-28k static tokens of RMSNorm-specific playbook and SOTA notes; `planner.EXPLORE_COMMON` appends "keep the kernel copy-like (memory-bound)" to GEMM tasks (#30 r2, #218 r2 `plan.json`); the design-card `paths` schema, emulator features and the example research card are all #38's.
10. **Budgets.** The probe-count cap (10) is the binding limit (9 of 44 sessions hit it; the #30 residual-in-TMEM session stopped before its profile and reported H15 "refuted" from a 255-register, spilling build), while B200 minutes are cheap. There is no turn cap in headless mode (one session: 23 turns, 46 min, $8.3). Money is not the scarce resource; portal slots and operator attention are.

## 1. Measurement validity and statistics

**Portal precision is problem-dependent.** "±0.3% per size" rests on one control pair (c1 vs r6) of a memory-bound kernel (`poc/results/b200_portal.csv`). #218 saw +2% then +2.6% drift on the compute band across r0, r1, r1b with identical cuBLAS code (card §7). For #30, where Tsol is 85% of Tb at L, 2% of time is 0.02-0.03 of per-workload S, i.e. the whole top-5 gap. No #30 control has been resubmitted.

**Emulator.** `loop/emulator.py` fits log(portal/rented) with a quadratic-in-size mean, copy lag and a GP over 17 design features. Issues:
- Hyperparameters sit on grid edges for #38 (`ll`=0.5, `ls`=0.5, `sf`=0.04; `emulator.py:125-126`), so the optimum is outside the grid.
- For #84 the `rented_exponents` k-correction clamps at 0.3 for L, for #218 at 0.3 for S and L (`planner.py:283`): the fallback is at its floor and meaningless there.
- `designer.emulator_line` rescales each workload by a single reference run in the same container (`designer.py:294-296`), injecting the reference's ~1% per-size noise; `planner.shortlist` does no rescaling (`planner.py:324-325`). The same kernel gets two different predictions.
- The shortlist prints `P(beats best)` with no floor on sd when n_train is 1-3 (`planner.py:300-342`): #30 r1 predicted 0.4258 ± 0.0047, actual 0.4529.

**Winner's curse.** Only seven predicted-vs-actual pairs exist; four landed above prediction, so no systematic over-selection shows yet. The directional misses (r8/r9: bench -1%, portal +3-4%; #218 CUTLASS: rented -1..-3%, portal +1..+2.5%) are what selecting the apparent best of noisy bench deltas produces. ABAB interleaving with 7-40 reps and medians (adopted from r10) is the right fix; it is not applied in `b200_test` or the shortlist, which use single runs.

**Shape-split A/B** (`loop/shape_ab.py`) is sound (historical offset of the shape pair, 2 sd rule) but assumes variants have no cross-workload side effects; H16 records exactly such an effect (clusters slowing later plain launches), and the harness workload order is fixed on the portal.

**Bench artefacts.** `toolserver.py:43-69` re-runs CUPTI-flaky workloads in a sub-problem copy; those timings come from a process that ran fewer workloads first. Flakes are frequent for #84's ~2 µs kernels, so #84 S-band bench numbers carry an order bias of unknown sign. `patch_harness_timing.py` drops windows (Modal only).

**Harness facts** check out against the clone: `iterations=50`, median (`benchmark_config.py:31`, `timing.py:288`); the "3 trials, mean" portal rule is inferred. `core_brief.md:29` still says "about 5% noise", contradicting the card's 0.3%; sessions read both.

## 2. Loop design

**Roles and prompts.** The lead (Fable) rewrites the whole ledger each round with no diff or validation (`round.py:256-260`); constraints have only grown (4 → 5 → 6 across r12-r14) but nothing checks. `LEAD_PROTOCOL` tells the lead run_tests predicts "with about ±0.01 error" (`round.py:193-195`), untrue on new problems. The static prompt is identical for GEMM and elementwise problems (`prompts.py:15`); `mcp_tools.py:88` still describes "#38 kernels".

**Anchoring.** Fresh-eyes sessions (r14) converged on the existing design, which is good evidence that #38 is at a local optimum for this approach, at $11.5 for two sessions. Normal sessions receive the full archive, ledger and (one round of) notebook, so refinement dominates.

**Tool budgets.** `--probes` defaults differ (10 in `round.py:661`, 12 in `mcp_tools.py:38`). The count cap bites before the minutes cap (`round.py:152`). No `--max-turns` is passed (`round.py:98`); the 60 min timeout is the only hard stop.

**Cost per round vs information.** A round is ~1 h wall-clock, $8-25 API-equivalent plus ~$2 B200 (`loop/runs/usage.jsonl`: $106 total over 179 calls). #38 r12-r14 (~$40, one portal slot) gained 0 score but closed H17-H21: good science, no leaderboard movement. #84 r2 ($6.7, one slot): +0.005. #30 r2 ($15, no slot): H14 refuted cleanly; H15 "refuted" by an unfinished implementation.

**Failure modes in the logs.** No session crashed or timed out. Parser losses of cards and ledgers (commits d840cd2, c9a4512, 3ba57e7) are fixed. Sessions hitting the probe cap mid-investigation is the recurring one. `b200_test` reuses session timings measured in whatever thermal state that container was in (`round.py:521-538`).

## 3. Problem selection and strategy

24 of 32 slots went to #38 (0.6125 vs #2 at 0.616, #1 at 0.6275). The pause is right. The other three were chosen from the survey's lowest-s5 list without a feasibility probe, and each research phase then found a structural cap: #84's Tb is torch.compile plus a launch delay the project refuses to use, so S is capped near 0.5 for a direct launch; #218's large-M Tb is 6-9% below any cuBLAS run; #30's only lever needs the portal as the instrument. The big teams' 0.56 on #84 therefore deserves a question: Python/torch.compile submissions get the +0.4 µs launch-delay advantage by default, and the survey cannot tell what language the top entries use.

Slots have been used well (every one a candidate or a control). Missing: a #30 control, a public submission, and screening.

## 4. Legitimacy

Submitted code uses: L2 eviction hints on the kernel's own buffers (CAKE and quack do the same), cluster launch attributes, per-shape dispatch, forced cublasLt algorithms chosen by shape, CUTLASS 2SM kernels. No PDL, streams, discard, applypriority, delays or resets appear in any submitted file (scan of every `rounds/*/portal/*.json`; the three r13/r14 files mentioning `applypriority`/`PersistingL2` do so only in comments and were never submitted). Points to settle before going public:
- The `bfloat##16_t` token-pasting (above). Fix the lint, not the code.
- The ledger H1 wording ("output parked past the timed window") reads as measurement gaming; r12 E1's finding (fewer in-window dirty write-backs) is the defensible description and should be the only one used publicly.
- Shape-split A/B packages dispatch different code for equal-work shapes; legitimate in private, but never publish one.
- #84's constraint list permits torch.compile-derived solutions while banning launch delays; a compiled path gets the delay by construction. State "launched directly".
- The applypriority gate is a ledger sentence, not a step: `cmd_shortlist` (`round.py:601-624`) copies candidates with no check, and `reply.lint` (`reply.py:28-38`) has no pattern for any ledger constraint.

## 5. Engineering debt and bugs

1. `loop/prompts.py:117` — notebook sorted by `str(round)` and cut at 40/80 lines; hides 80% of findings (section summary 4).
2. `loop/emulator.py:125-126` — hyperparameter grid too narrow; `planner.py:283` k clamp hit on #84/#218.
3. `loop/mcp_tools.py:50-52` and `round.py:94-97` — a 15 s GP refit at every session start and again in `--check`; cache it.
4. `loop/designer.py:294-296` vs `loop/planner.py:324-325` — inconsistent rescaling (section 1).
5. `loop/planner.py:300-342` — no sd floor or n_train shown.
6. `loop/reply.py:34` — `float16|bfloat16` lint is wrong for fp16/bf16-by-definition problems; the "operator whitelists" step in `problems/030_*/ledger.yaml` has no code.
7. `loop/round.py:661` vs `mcp_tools.py:38` — probe cap mismatch; `round.py:98` no turn cap.
8. `loop/planner.py:25-43,167-201` — #38-specific constants and explore ideas on live code paths (`--mode explore`, non-lead `plan`).
9. `loop/problem.py:89-92` — byte-tertile bands drive task bands and `predict_score` worth, but the #218 card says they "do not follow the physics".
10. `loop/round.py:362-376` — the #38 card is the example for every research phase, so new cards inherit the S/M/L memory-bound frame.
11. `loop/context/core_brief.md:29` — stale noise statement; `harness_scoring.md:189-191` inferred trial rule presented next to verified facts.
12. `problems/*/archive.json` on disk omits derived fields (status, portal) until the next `load`; a human reading the raw file sees r2 kernels as "proposed, no portal". Document or save after ingest.
13. Untracked scratch: `poc/results/leaderboards/{a,d,done,kernels,raw}.txt`.

## 6. Oversights

- **A clock-locked B200.** Only Modal was tried (`b200_modal.py:165`, H13). A bare-metal hour with `nvidia-smi -lgc 1500` would show whether portal/rented collapses to 1.00 and would make compute-bound problems testable; nothing in the repo prices this.
- **Portal order effects** (H16) rest on one observation, yet the harness runs locally: the 16 workloads can be permuted in `workload.jsonl` on the bench for a few dollars.
- **Public vs private**: unknown whether evaluation, Tb, or review differ; also whether private scores count for anything.
- **Baseline anatomy as a selection criterion**: for boards where Tb is a compiled single kernel, S-band scores are capped near 0.5 for every direct-launch entrant. The ratio reference/torch.compile/copy-floor per band (already measured by `research.measure()`) predicts reachable score before any round is spent.
- **The lead never sees the portal page's per-workload S directly** in a form that says "this workload is capped"; `predict_score` gives worth per band only.

## Prioritised improvements

Quick wins (under 2 h each):
1. Fix `lab_notebook` ordering and truncation (numeric round sort; all findings, or grouped by hypothesis with a per-round cap). Value: high; the project's main memory mechanism is currently one round deep.
2. Emulator honesty: sd floor of ~0.015 when n_train < 5, print n_train, one rescaling rule shared by `designer` and `planner`, widen the hp grid, cache the fit. Value: stops false 100% confidence on new problems.
3. Constraint lint: patterns for `discard.global`, `applypriority`, `nanosleep|clock64` loops, `griddepcontrol|ProgrammaticStreamSerialization|launch_pdl`, `cudaCtxResetPersistingL2Cache|accessPolicyWindow`, extra streams; make `cmd_shortlist` refuse or flag; derive the precision lint from the definition's dtypes so agents stop token-pasting. Value: insurance before the first public submission.
4. Raise `--probes` to ~30 (keep the minutes cap), pass `--max-turns`, align defaults.
5. Per-problem-class static docs: drop the #38 playbook/SOTA and the "copy-like" sentence for compute-bound problems; fix `core_brief.md:29`.

Portal slots (one each):
6. Resubmit `r1-oproj-cutlass-t224-dispatch` unchanged on #30 to measure compute-band portal noise before any more #30 rounds. Decides whether 2-3% deltas are readable there.
7. Make c2 public on #38 (and the #84 best) to hold a rank, trigger the review, and learn whether public = private.

Strategy (days):
8. Screen before committing: run `round.py research` on five boards chosen from the survey by s5 and by reference/compile/floor ratios (candidates: #10, #157, #162, #129, #124; skip the GEMM family, which repeats #218's stored-Tb trap). ~$30 plus five anchor slots.
9. Park #84 after r2's E1/E2 unless they found ≥ 0.05 µs; keep #218 parked; give #30 two more rounds only after item 6, starting with a finished residual-in-TMEM build (the H15 "refutation" came from a spilling 255-register kernel and an exhausted probe budget).
10. Price a bare-metal B200 with clock control and re-time the 23 #38 pairs there. If the ratio is flat, retire the emulator for memory-bound work and open compute-bound problems.
11. Validate lead ledger rewrites (constraints superset, hypothesis ids preserved) and store a diff per round.

## Open questions for the operator

1. Is private scoring identical to public, and has anything ever been reviewed by the judge? Do private results hold any rank?
2. What do the public top-5 on #84 and #38 submit (Python/compiled vs C++)? If compiled paths get the launch delay by default, is the self-imposed ban the right line, and has the planned question to NVIDIA been sent?
3. Who reads the code of a candidate before upload, and is the applypriority gate written down anywhere outside the ledger?
4. Is the goal top 5 on a few boards or presence on many? Breadth favours screening (item 8); depth favours #30 plus a clock-locked bench (item 10).
5. Has any non-Modal B200 (bare metal, clock control) been priced?
6. Should A/B diagnostic submissions keep using portal slots now that the bench is known to misrank 1-3% effects, or should those slots go to anchors on new boards?
