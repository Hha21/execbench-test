# Agentic kernel search: what the best systems do, and a critique of our loop (2026-10-11)

This is a read-only review of the repository at `bebea7b`, plus a web survey. "Rented" means the Modal B200, which runs
with unlocked clocks. "Portal" means NVIDIA's scorer, which locks the SM clock at 1500 MHz. Dollar figures for Claude
are API-equivalent estimates from `loop/runs/usage.jsonl`; the real cost is usage of the Claude plan. Modal and Verda
cost real money.

Three short definitions used throughout:
- **k-block**: one 64-wide slice of the GEMM's inner dimension K, the unit our tcgen05 kernels pipeline.
- **MMA ideal**: the clock count one k-block needs if the tensor cores never wait.
- **Ring**: the circular set of shared-memory stages that loader warps fill and the MMA warp drains.

---

## 1. Executive summary

1. **A wrong ceiling has steered #35 since r10.** `poc/leader_gap.py:35` and `loop/context/playbook_compute.md:18`
   treat 1.3 PF/s as the FP16 tensor-core ceiling at the portal clock. That number is our *achieved* TF32 rate × 2,
   not the hardware rate. Four independent checks put the real ceiling near **1.8 PF/s** (8,192 FLOP per clock per SM):
   - our own r6 ledger entry: an MMA ideal of 928 clk per k-block at N = 464, and 954 clk measured;
   - SOLAR's Tsol;
   - cuBLAS at 1.93 PF/s on the rented card;
   - FlashAttention-4's 1.61 PF/s at "71% utilisation".

   With the right ceiling, ledger H30 changes meaning. "The nb2 main loop is near its limit (1346 vs 1299 clk)" becomes
   "the main loop runs at about 69% of the MMA rate (1346 vs 928 clk)". r11 deprioritised exactly the two levers this
   reopens: the converter ring, and reading W once at nb2. On #30 the same constant makes the leader-gap report say the
   leader "needs FP8 tensor", which is a trap, because FP8 fails #30's tolerance.
2. **Our search is a single lineage, one candidate per round, under ±0.008 portal noise.** Since r6 every round has
   been one build session whose parent is the current best, which makes it a noisy hill-climb. Systems that escape
   plateaus keep several lineages and share conclusions between them. On KernelArc's own ablation, a multi-agent run
   reached 2.04× the single-agent result. We do not need that scale. We need 2-3 lineages and a rule that keeps an
   alternative alive for two rounds before judging it.
3. **The best-supported creativity trick is cheap: ideas in plain language first, then filter near-duplicates, then
   build a few far-apart ones** (PlanSearch, Stanford's fast-kernels study, ShinkaEvolve). "Encouragement" prompts,
   personas and temperature have weak or no evidence. Mixing models helps cost more than it helps diversity: different
   LLMs tend to give strikingly similar answers ("Artificial Hivemind").
4. **"Fresh eyes" is not fresh.** Fresh sessions still read `tricks.md` (`loop/prompts.py:31`), which describes our
   #35 design in words. The r11 paper task also handed over "settled facts" (including the wrong 1.3 PF/s) and spelled
   out "the obvious design", which is ours. That session's "independent" agreement with us is weak evidence.
5. **Verda tonight is the most valuable experiment available.** If a clock-locked B200 reproduces portal times within
   2-3%, it can replace the emulator for compute-bound problems. It would also make #30 workable: #30's whole puzzle is
   that our fused kernels slow down 1.36× from rented to portal, while cuBLAS (≈ Tb) slows down only 1.22×.
6. **#30, honestly.** KernelArc's 0.546 comes from making the output *alias* (share memory with) the residual input,
   so cuBLASLt computes C == D, plus a static per-size table of cuBLASLt "Expert API" settings. Writing into an input
   buffer breaks our own rule ("never touch memory we do not own"), so ask NVIDIA before counting on it. Without it, a
   realistic target is about 0.49-0.50, and top 5 needs 0.491. Two legitimate levers have never been tried on #30:
   - the Expert-API search, run at the locked clock;
   - the L2 `evict_first` lesson (tricks.md §3), applied to the residual and A loads.
7. **Cost.** Build sessions cost $6-8 mostly because they write a lot:
   - in r11, ~70k of the 183k output tokens were thinking;
   - about 40-45k were complete source files re-sent with every probe or test;
   - the final reply repeats the files again.

   Three changes would cut a round from ~$20 to ~$12-15 API-equivalent while testing more ideas:
   - a session workspace with diff edits;
   - a cheap idea/probe tier;
   - a one-page state card instead of 27k tokens of ledger plus notebook.
8. **Knowledge capture drifts.**
   - #30's ledger still lists H13, H14 and H15 as open, though r2 refuted two of them.
   - `playbook_compute.md` recommends the aliasing trick that our rules forbid.
   - `core_brief.md` still says "we cannot run a B200" and "keep all maths in fp32".
   - `read_example` offers only norm kernels, even to GEMM sessions.
9. **Rules risk.** NVIDIA's own paper lists "precision downgrade: computing in FP16 and upcasting to FP32" as the most
   common reward hack it caught (6.4% of submissions). Our *public* #35 entry uses FP16 operands that pass the
   tolerance. One email to NVIDIA could settle both this question and the aliasing question.
10. **Top three actions:**
    1. Fix the ceilings and re-plan #35's large-B band (no GPU, about an hour).
    2. Run the Verda calibration tonight (§7).
    3. Run rounds as "fresh lead + idea batch + 2-3 cheap explorers + 1 build", with two lineages kept alive.

---

## 2. What the best systems do

Words used in the tables:
- **Population**: a set of candidates kept at the same time, as opposed to a single "current best".
- **Islands**: several separate populations that rarely swap members, so they do not all converge on one idea.
- **MAP-Elites / quality-diversity**: keep the best candidate *of each kind*, for example the best per cell of a grid
  of features such as "library vs custom" × "persistent or not", not just the single overall best.
- **Novelty rejection**: discard a proposed change before testing it if it is too similar to something already tried.
- **UCB bandit**: a simple rule that sends more calls to whichever model has recently produced improvements, while
  still occasionally trying the others.

### 2a. General LLM-driven evolutionary search

| System | Search structure | Diversity / novelty | Noisy or costly evaluation | Compute |
|---|---|---|---|---|
| **FunSearch** (DeepMind, Nature 2023) | Many samplers feed islands. Periodically the worst half of the islands is wiped and reseeded from the best. Prompts show 2 earlier programs sorted by score. | Islands; programs clustered by score signature; shorter programs favoured | Deterministic maths evaluators | Millions of samples from a small, fast model |
| **AlphaEvolve** (DeepMind 2025) | Program database mixing MAP-Elites cells and islands. A prompt holds a parent, "inspiration" programs and rendered results. The model answers with SEARCH/REPLACE diffs. | MAP-Elites, islands, a Flash+Pro model ensemble (volume plus quality), and evolution of the prompt itself | Evaluation cascade: cheap tests first. Kernels were timed on real TPUs, on *held-out* shapes. | "On the order of 100 compute-hours" per problem; a Gemini matmul tiling heuristic gained 23% on average. Ablations: removing evolution, context or the big model each hurt. |
| **OpenEvolve** (open source) | AlphaEvolve clone: islands, a MAP-Elites feature grid (complexity, diversity), an elite archive, and migration between islands | Same | Your own evaluator | Cheap models, many iterations |
| **ShinkaEvolve** (Sakana, ICLR 2026) | Island archive. Parents are sampled by fitness, and down-weighted by how many children they already have. | Embeds each candidate's code and rejects it if cosine similarity > ~0.95 (an LLM judge gets the borderline cases). A UCB bandit picks among several LLMs. | Not a focus | Circle-packing record in ~150 evaluations, orders of magnitude fewer than earlier systems |
| **Darwin Gödel Machine** (Sakana 2025) | A growing archive of every working agent; any of them can be a parent. Selection weights score by a sigmoid and adds a bonus for having few children. | The archive keeps "stepping stones". Ablation: always building on the latest version does worse. | Benchmark subsets, run in stages | Expensive: every child is a full benchmark run |

### 2b. Kernel-specific systems

| System | Search structure | Diversity / novelty | Evaluation and noise | Compute |
|---|---|---|---|---|
| **KernelArc** (IMEC, arXiv 2608.17071; #1 on #30) | 2-4 agents in parallel, each seeded with strategy "skills": library, memory, compute, fusion, precision, reduction, scheduling, custom-sm100. A **deterministic guard** owns correctness, benchmarking and keep/revert. After r_draft benched non-improvements, the agent must "draft" a different algorithm, DSL or data layout. | **Conclusions-only shared memory**: 16 "wins" and 16 "traps" per kernel type, each with a "why" field (160/1200-character limits). Agents get a read-only view of the other agents' best kernels. | A candidate is kept only if it beats the incumbent by a set margin. Near-best "lateral" candidates can stay as working points. The GPU is locked during runs. | 100-candidate budget in the ablation. Two agents with unbounded shared memory reached 2.04× the single agent's speed-up and 1.59× the bounded-memory pair's (p ≈ 0.05, 5 runs each). Opus 4.6 for #30, Kimi K3 elsewhere. Kimi K3 was the cheapest model to reach near-cuBLAS on their GEMM study. |
| **Cursor + NVIDIA** (multi-agent kernels) | A planner agent splits and rebalances work across autonomous workers by performance. The whole protocol lives in one markdown file. | Breadth: all 235 problems | The SOL-ExecBench harness is the judge; results "beyond what a B200 can support" are thrown out | 27 B200s for 3 weeks. Median SOL score 0.56; their BF16 GEMM reached 86% of cuBLAS. Breadth alone did not beat the specialist teams. |
| **NVIDIA's baseline agent** (SOL-ExecBench paper) | Turn-based cohorts. After each round, every valid solution is shown to the next cohort, and the fastest kernel overall is kept. | Several independent agents per round | Sandbox plus an LLM judge; 14.5% of submissions were flagged as hacks | PyTorch-only; median SOL 0.732. Tb on #30 behaves like plain cuBLAS. |
| **Sakana AI CUDA Engineer → robust-kbench** | Evolutionary "meta-generation" with LLM verifiers | — | The 2025 failure: a memory exploit skipped correctness checks. Excluding contaminated tasks cut the claimed 3.13× to 1.49×. | — |
| **CUDA Agent** (ByteDance/Tsinghua 2026) | Large-scale agentic RL: a model *trained* to optimise kernels | Diversity comes from training data | Verified and profiled reward | A training budget far beyond ours; not directly applicable |
| **Kevin-32B** (Cognition) | Multi-turn RL. Earlier chain-of-thought is dropped; only kernels, results and a short self-summary are carried to the next turn. | 16 parallel trajectories × 4 refinement steps | Hacks seen: copying the reference, a try/except fallback to PyTorch, inheriting the reference class; fixed by format checks | Once steps × trajectories ≥ 8, more *serial refinement* beat more parallel samples |
| **METR KernelAgent** | Best-of-k attempts per task | k samples | A floor at the reference speed | 1.81× vs the 1.05× of the original KernelBench scaffold: the scaffold matters |
| **Stanford "surprisingly fast kernels"** | Each round: write optimisation *ideas* in plain language, conditioned on ideas already tried; each idea spawns several implementations; the best seed the next round | Idea-level branching; winners clustered into 6 strategy families | Fixed problem sizes | About 3M input + 4M output tokens over 5 rounds; most bests appeared in rounds 4-5 |
| **AMD GEAK** | Generator → reflector → evaluator → optimiser loop (Reflexion-style) | — | Cascade: correctness first, then speed | Inference-time scaling |
| **K-Search** (Berkeley 2026) | A "world model" search tree of hypotheses; high-level plans are kept separate from code | LLM-chosen exploration | Tolerates buggy intermediate attempts | 2.1× over earlier evolutionary search on FlashInfer kernels |
| **AccelOpt** (Stanford/AWS) | Beam search plus a memory of slow→fast kernel pairs | Beam width | — | Open models matched Claude Sonnet 4's gains at 26× lower cost |
| **MLSys 2026 FlashInfer contest** (Baidu harness paper; UW SyFI) | Claude Code or Codex working under human-written skill documents. Loop: read the profiler (NCU) → hypothesis → edit → re-measure. | Short human directions ("use tensor cores here", "try warp specialisation") | The contest harness | Baidu: agent-assisted runs beat fully autonomous ones on every operator. SyFI: a fully autonomous entry took 1st on one track. |

### What this means for us

- **Keep the judge deterministic and outside the agent.** We already do this, and we are stricter than most systems
  about legitimacy.
- **Several lineages with shared, compact conclusions beat one deep trajectory at a fixed budget** (KernelArc, Darwin
  Gödel Machine, AlphaEvolve). Our lab notebook and ledger are a shared memory, but there is only one trajectory.
- **Write ideas in plain language first, then branch.** Idea diversity predicts search gains (PlanSearch,
  Stanford, K-Search).
- **Switch strategy on a plateau, enforced by code, not by goodwill** (KernelArc's "draft on plateau"). Our plateau
  rule exists only in the lead prompt, and the lead has not run since r5.
- **Evaluate in the target configuration.** AlphaEvolve timed on real TPUs; KernelArc locked the GPU. Our rented card
  is unlocked.
- **Human direction still pays** (Baidu, SyFI). The operator is an asset, but should steer with questions and raw
  measurements, not "settled facts" that may be wrong.
- **These systems spend hundreds of candidates per problem; we spend 1-3 per round.** So every one of our candidates
  should be far from the others, with cheap tiers filtering ideas before an expensive build.

---

## 3. Getting creativity and diversity out of LLMs

### 3a. What has evidence

| Technique | Evidence | Effect | Fit for us |
|---|---|---|---|
| **Ideas first, in plain language, then code** | PlanSearch (ICLR 2025): searching over written plans gave 77.0% pass@200 on LiveCodeBench vs 60.6% for repeated sampling, and idea diversity predicted the gain. Stanford's fast-kernels study used the same pattern. | Large on code benchmarks | Very good. A no-GPU idea batch costs $0.5-2. |
| **Filtering near-duplicates** | Si, Yang and Hashimoto: of 4,000 generated research ideas, only ~200 were not duplicates. ShinkaEvolve's embedding-plus-judge rejection beat no rejection in its ablation. | Stops paying for the same idea twice | Very good. Embed one-line ideas and compare them with ledger entries; or ask a cheap model "is this the same mechanism as H#?" |
| **Verbalized sampling** (ask for k answers with a probability each, some from the tails) | 1.6-2.1× more diverse creative writing, with no loss of accuracy; stronger models benefit more | Untested on technical design | Cheap to try in the idea batch |
| **Several lineages / islands** | AlphaEvolve and Darwin Gödel Machine ablations; KernelArc's multi-agent runs | Escapes plateaus | Core suggestion (§5, item 3) |
| **Model ensembles with a bandit** | ShinkaEvolve: an adaptive ensemble beat a single model (the gain over a fixed ensemble was smaller). AlphaEvolve: Flash for volume, Pro for quality. | Moderate | Use OpenRouter models for *ideas* and cheap probes, Claude for builds |
| **Context helps, if it is the right context** | AlphaEvolve's "no context" ablation hurt. KernelArc stores *conclusions only*, with short "why" fields. | — | Replace the ledger and notebook dump with a compact state card plus retrieval (§5, item 6) |

### 3b. Folklore, or weak evidence

| Trick | What the evidence says |
|---|---|
| **"EmotionPrompt" / "this is important to my career" / tipping or threats** | The original EmotionPrompt paper (2023, older models) reported gains. Wharton's 2025 test on GPQA and MMLU-Pro found no overall effect from tips or threats, but swings of up to ±35% on single questions in both directions. That makes it noise, not a lever. Harmless but useless; do not spend tokens A/B-testing it. |
| **Personas ("you are a world-class CUDA engineer")** | 162 personas × 4 model families on 2,410 factual questions: no gain over no persona, and the per-question effects looked "largely random". A persona might shift *which* ideas appear; there is no evidence it improves them. |
| **Temperature** | Novelty is only weakly linked to temperature; incoherence is moderately linked to it. Claude Code does not expose temperature anyway. |
| **Multi-agent debate** | It does not reliably beat self-consistency (sampling several answers and voting) unless heavily tuned (Smit et al., ICML 2024). A 2025 follow-up found debate methods fail to beat chain-of-thought or self-consistency even with extra compute. |
| **Self-critique without new information** | LLMs mostly cannot fix their own reasoning without external feedback (Huang et al.). A critic helps when it can **check against data**. |
| **"Many different models = many different ideas"** | "Artificial Hivemind" (NeurIPS 2025): the similarity *between* models is even stronger than the repetition within one model. Mixing models helps less than people assume. |

### 3c. Cheap tricks worth trying

Each costs at most one cheap call per round.

1. **Idea batch with tails.** "List 10 designs for <sub-problem>. For each give the mechanism in one line, the binding
   limit with arithmetic, and the probability that a strong engineer would propose it. At least 4 must have a
   probability below 0.1." Then deduplicate against the ledger and let the lead pick 3 that use different strategy
   lenses (library, memory, compute, and so on).
2. **Premise audit.** "Here are the numeric claims in our ledger and tricks.md (ceilings, ideals, floors). Recompute
   each one from the datasheet rate and our raw measurements, and list contradictions." This would have caught
   1299 vs 928. It is a critic with *data* to check against, which is the form of critique that has evidence.
3. **Constraint audit.** "Which of *our own* constraints (not NVIDIA's) cost the most score? What evidence would
   justify relaxing each one, and whom would we ask?" The largest #35 gain (+0.042) came from exactly this: the
   precision policy change.
4. **Leader stories.** "Assume the leader is 25% faster than us at B = 919. Write three different stories of how,
   each consistent with our measurements, and the 10-minute probe that would rule each one out."
5. **Analogy.** "How do CUTLASS's sm100 pipeline, DeepGEMM and FlashAttention-4 solve <the same sub-problem>
   (feeding converted operands to tcgen05; hiding an epilogue)?" Pair this with a `read_source` tool (§5, item 6).
6. **A truly fresh session on another model family.** No tricks.md, no ledger, no "settled facts". Give it only the
   definition, the workload table, our raw per-workload times and the rules. Running it on GPT, Gemini or Kimi through
   OpenRouter also gives some model diversity.

---

## 4. Critical analysis of our loop

### 4.1 A wrong number steered four rounds

What the files say:
- `problems/035_.../ledger.yaml:242` (r6) gives FP16 MMA ideals of **768/1024/704/928 clk** per 64-wide k-block at
  384/512/691/919. The "unordered relaxed arrive" variant measured **954 clk at 919**. That means the hardware does
  about 4,096 multiply-adds per clock per SM.
- `ledger.yaml:267` (H30, r10) says: "nb2 about 1346 clk per k-block against **1299** ideal at N = 464". Note that
  1299 = 928 × 1.4, which is exactly the 1.3 PF/s premise.
- `rounds/r11/sessions/paper_task.txt:20` gave the paper session this as a settled fact: "kind::f16 ideal is about
  1.3 PF/s". The paper session itself flagged the doubt (`paper.reply.md:9`): "If the true issue rate is 4,096 MAC/clk
  (1.82 PF/s), every shared-memory budget below gets 1.4× tighter." Nobody followed up.
- `poc/leader_gap.py:35` sets `("FP16/BF16 tensor", 1.3e15)`. `playbook_compute.md:18` says "kind::tf32 ≈ 0.62-0.65 PF/s
  measured (#35); FP16 ≈ 2x that". The 0.62 figure includes padding, 144 of 148 SMs and overheads, so it is an
  *achieved* rate. The TF32 MMA rate itself is 0.91 PF/s at 1.5 GHz; r4's TF32 ring hit 1028 clk against a 1024 ideal.
- Independent checks of 1.8 PF/s for FP16:
  - SOLAR's Tsol back-solves to FLOPs / 1.75-1.80 PF/s (`ledger.yaml:71`, #30 H6);
  - the #30 card measured cuBLAS at 1.93 PF/s at 1965 MHz, which scales to 1.47 at 1500 MHz, already *above* the
    "ceiling";
  - FlashAttention-4 reports 1.61 PF/s at 71% utilisation.

What it cost:
- **r11 aimed at the time outside the main loop**, on the claim that "our main loop already runs at that efficiency".
  It explicitly ruled out building "W-in-TMEM (the paper's D4, which it called the most upside) ... cluster-4 DSMEM
  sharing or an L2 fp16 ring", reasoning that the main loop is not shared-memory-bound. At the true MMA rate, the
  shared-memory budget is 1.4× tighter and these ideas reopen.
- **The nb2 kernel reads W twice** (about 452 MB at 919; r11 plan, lever 4). At our measured 5.96 TB/s read rate that
  alone is ≥ 76 µs. So fixing the ring only helps if W is also read once. Both levers are needed, and r11 called the
  second "optional ... expect a small effect".
- **A rough size of the prize.** At 919 the main loop is 96 k-blocks; at 1346 clk and 1.5 GHz that is 86 µs, plus ~25 µs
  of overhead, which matches the portal's 111 µs. At ~1,050 clk (88% of the MMA rate), with W read once, 919 would be
  about 90 µs, moving S from 0.678 to about 0.77. Similar gains across 691-853 give **about +0.02 on the mean**. That is
  an estimate, not a promise, but it beats every round since r7.
- **What the leader must be doing.** If its 0.7945 is spread so that the small sizes score what ours do (~0.71), it
  needs ~73 µs at 919, about 1.43 PF/s. That is impossible under a 1.3 ceiling and comfortable under 1.8. The other way
  round (uniform S) needs 32.7 µs at B = 5, below our "measured floor" of 38 µs for streaming W. Either way the leader
  beats something we treated as a ceiling.
- **On #30,** leader_gap prints "FP8 tensor" as what the 8192-token sizes need. FP8's 3-bit mantissa cannot meet
  rtol 0.05 for D ≈ N(0, 50²) with K = 2560. A lead that follows its protocol ("if the leader is above what our
  approach can reach, the gap is a different precision") would chase a dead end.

The general lesson: a single constant, repeated in four files and in operator-written prompts, outranked our own
measurements. Nothing in the loop checks derived numbers against raw data.

### 4.2 Search structure: one lineage under noise

- Every kernel's parent is the previous best (archive: r6 ← r5-e2, r7 ← r6, ... r11 ← r10). The lead defaults parents
  to the best (`loop/round.py:289`). Since r6, `plan.json` has held exactly one task.
- With a per-submission speed factor of 0.965-1.045 (score ±0.008), one-sample hill-climbing becomes a random walk once
  real gains fall below the noise:

  | Round | Score change | Notes |
  |---|---|---|
  | r8 | +0.0005 | |
  | r9 | −0.008 | |
  | r10 | +0.0165 | +0.013 of this is the speed factor |
  | r11 | +0.0002 | |
- Radical alternatives cannot survive one round. A new design is usually slower at first, while the incumbent has had
  six rounds of tuning. Nothing keeps a second lineage alive long enough to mature.
- There is free quality-diversity structure we do not use. #35's kernel is a per-size dispatcher, so the natural
  "cells" are size bands × design family. Commit fb5b567 found that the best per workload across r6-r9 would have
  scored 0.6889 against the best single kernel's 0.6853, though part of that is the max of noisy draws.

### 4.3 The lead, and the operator who replaced it

- `LEAD_PROTOCOL` (`round.py:188-224`) is reasonable: leader gap first, strategy lenses, a plateau rule. But it has
  not run since r5. From r6 the operator (Claude in the main chat) wrote single-task plans.
- The operator has the deepest context of anyone, which is the most anchored position possible, and it writes its
  beliefs into prompts as "settled facts". The plateau rule never fired: r8-r11 were all refinements of `tc_h`.
- **Why rounds converge to refinements:**
  1. The parent is always the best.
  2. The ledger's live hypotheses (H28-H30) are all about the current kernel's internals.
  3. The success bar is "beat the best by ≥ 0.005 in one round".
  4. The budget is one session.
  5. A wrong premise made the main loop look finished.
  6. Noise hides real small gains, so "plateau" and "progress" look the same.
- The lead prompt is large and history-heavy (73 KB for r5; ledger + notebook + archive). It reads as "what do we do
  next with *this* design", not "what would a newcomer build".

### 4.4 Context size and anchoring

- **Size.** The r11 build session started at ~129k tokens of context and ended at 285k, over 23 model calls.
  - Static briefing, ~24k tokens: core_brief 2k, tricks 3k, card 3k, harness 4k, b200_arch 4.5k, playbook 2.4k,
    generation protocol 3.3k.
  - Dynamic prompt, ~46k tokens: ledger 13.7k, lab notebook 13.4k, parent source 16.6k.
  - The ledger and notebook largely repeat each other, because findings appear in both.
- **Anchoring.** Every session reads our conclusions as facts, and wrong ones spread (§4.1). `tricks.md` is in
  `FRESH_DOCS` (`prompts.py:31`), so "fresh" sessions learn our design from §5 ("Skinny GEMMs ... 144 CTAs × 128 rows",
  converter warps, 2SM pairs). The r11 paper session also saw tricks.md (`briefing_paper.txt`) and an operator-written
  description of "the obvious design".
- **What could be retrieved on demand instead:**
  - the full ledger: replace it with the 5-10 entries the task cites, and a `ledger(id)` tool for the rest;
  - the notebook: replace it with a `notebook(query)` tool;
  - harness and architecture details: send a 1-page digest and keep the full docs behind a tool;
  - the parent source: keep it, but as files in a workspace (§5, item 5) rather than inline text.
- **Wrong reference material for GEMMs.**
  - `read_example` serves only norm kernels (`loop/context/examples/`, 5 files).
  - `designer.PROTOCOL` still says "read_example shows annotated excerpts of the best public B200 norm kernels"
    (`designer.py:51`) and suggests checking for "LDG.256 / STG.256" (`designer.py:35`).
  - GEMM sessions read CUTLASS headers by spending B200 probes (#30 r2), which is slow and burns the probe budget.

### 4.5 Evaluation

**Good already:**
- interleaved A/B runs in one container, using medians;
- counting SM cycles (`clock64` with `%globaltimer`);
- shape-split A/B packages on the portal, which cancel the per-submission speed factor;
- recording portal/rented ratios per kernel type.

**Weak:**
- **Unlocked rented clock.** The portal/rented ratio ranges from 1.03 to 1.42 depending on the kind of kernel. On #30
  this hides the very effect we need to see: fused kernels scale 1.36-1.40, plain cuBLAS 1.22, and our "faster on
  rented" CUTLASS kernel ties Tb on the portal.
- **The emulator has the wrong features for tensor kernels.** `loop/emulator.py:26` uses memory-bound features
  (`width256`, `rows`, `x_ef`, ...). For tcgen05 kernels it mostly learns a per-kernel offset, and its error bars are
  smaller than the portal's own noise.
- **Winner's curse** (picking the best of several noisy measurements makes the winner look better than it is). We
  re-time interleaved, but we never confirm in a *fresh* container or at another time, which is what an independent
  replication would be.
- **The portal lock is not a guarantee under load.** NVIDIA's `clock_lock.py` checks clocks once, 3 s after locking,
  at idle. A power or thermal slowdown during tensor-heavy runs would look exactly like our "speed factor", which hits
  tensor-heavy sizes hardest. Nobody has recorded clock event reasons on a locked card under load.

**Statistical tricks, cheapest first:**
1. **Lock the clock (Verda).** It removes the biggest bias.
2. **Pair A and B in the same run, and average log-ratios over the 16 workloads.** Paired differences cancel shared
   drift; averaging over workloads shrinks the remaining noise.
3. **Keep a control path in every portal package when testing a change under 5%.** We already do this with shape-split.
4. **A confirmation run:** re-time the winner against the control in a new container, in a new random order, before
   using a portal slot.
5. **Report cycles for SM-bound parts and ns for DRAM- or latency-bound parts,** and predict portal time per part.
6. **Bring the GPU to a steady thermal state before timing either variant,** and record `clocks_event_reasons`.

### 4.6 Cost

#35 since the operator took over (API-equivalent, from `loop/runs/usage.jsonl`):

| Round | Portal change | Session cost |
|---|---|---|
| r6 | +0.042 (precision change) | $7.77 |
| r7 | +0.017 | $5.57 |
| r8 | +0.0005 | $6.53 |
| r9 | −0.008 (noise) | $7.94 |
| r10 | +0.0165 (≈ +0.003 net of noise) | $5.66 |
| r11 | +0.0002 | $6.31 build + $2.37 paper |

Where the r11 build session's output went:
- 183k output tokens in total, of which ~70k were thinking;
- ~147k characters (≈ 40-45k tokens) were tool arguments, mostly whole kernel files re-sent with each probe or test.
  `designer.py:53` says "Pass complete files every time; nothing persists between calls";
- ~64k characters of final reply, which repeats the files.

Cache reads (1.8M tokens) are cheap. Output and cache writes (285k) dominate. Trimming the prompt saves less money than
it saves *focus*; cutting re-sent code saves both.

Cheaper session types:

| Tier | What it does | Tools | Model | API-equivalent cost |
|---|---|---|---|---|
| Idea | 10 written ideas with arithmetic | `predict_score` only | Sonnet or OpenRouter | ~$0.3-2 |
| Probe | Answer one measurement question | `probe_b200` only, 30-minute cap | Sonnet | ~$1-2 + GPU |
| Build | Full design session | All | Opus | $6-8 |
| Lead | Plan the round | No GPU | Fresh session | $2-4 |

KernelArc found Kimi K3 the most cost-efficient model on its GEMM study, and AccelOpt matched Sonnet 4 with open models
at 26× lower cost. But Claude on the plan costs little at the margin, while OpenRouter is real money, so use OpenRouter
for diversity, not for savings.

### 4.7 Knowledge capture

It helps: tricks.md carried evict_first from #84/#85 to #35's W stream. It also fails and anchors:
- **No transfer to #30.** #30 never tried evict_first on any load (a grep finds only the old H8 about stores). Its r2
  predates tricks.md, and nothing re-applies new lessons to earlier boards.
- **Stale ledgers.** #30's `ledger.yaml` still says H13/H14/H15 are open. The r2 cards say H13 and H14 were refuted, and
  H15 was "refuted" by an incomplete build whose probe budget ran out before the profile. Ledgers are updated only by
  the lead or by hand, and the lead no longer runs.
- **No premises recorded.** H30's conclusion rests on a constant, and nothing links the two. When the constant is
  wrong, nothing re-opens H30.
- **Docs that contradict rules or reality:**
  - `playbook_compute.md` §2 recommends the KernelArc aliasing trick ("Not yet tested by us"), which writes into an
    input buffer;
  - `core_brief.md:4` says "a B200 that we cannot run ... We test on CSF3", and `core_brief.md:84` says "Keep all
    maths in fp32";
  - `mcp_tools.py:88` and `designer.py:51` talk about norm kernels.

### 4.8 Other issues

- **FP16 on fp32 problems.** NVIDIA's paper (Table: observed reward hacks) lists "Precision Downgrade: Computing in FP16
  and upcasting to FP32" as the most common exploit (259 kernels, 6.4% of submissions). The listed defence is "tightened
  numerical tolerances", which suggests that passing the tolerance is enough, but the LLM judge reviews public entries.
  #35 is public and depends on FP16. The leader almost certainly does too (leader_gap: FP16 is needed for 0.79). One
  email settles it.
- **The aliasing question on #30** is worth +0.05 if NVIDIA allows it, and nothing if not. Ask in the same email.
- **One-board focus.** It is fine, but the reopened #35 lever likely has a higher expected value per dollar than #30
  right now (see §6's go/no-go gate).

### 4.9 Previous review (2026-10-10): what is still open

| Item | Status |
|---|---|
| 1. Lab notebook ordering and truncation | Done (4cefe39) |
| 2. Emulator honesty | Partly done: sd floor and n_train were added. Still open: the hyperparameter grid edges (`emulator.py:125`), caching the fit, one rescaling rule shared by designer and planner. New: its features cannot describe tcgen05 kernels. |
| 3. Constraint lint | Done |
| 4. Probe and turn budgets | Done (30 probes, 40 turns) |
| 5. Per-kind docs | Partly done: the memory-bound docs are dropped for compute problems. Still open: core_brief, `designer.PROTOCOL` and `read_example` are norm-specific or stale. |
| 6. #30 control resubmission | Done: c1 scored 0.4490 vs r1's 0.4529, so #30 portal noise is about 0.004 |
| 10. Clock-controlled bare-metal B200 | Open: Verda tonight |
| 11. Validate the lead's ledger rewrites | Open (`round.py:282` writes without diff or validation) |
| 8. Screen boards before committing | Superseded by the operator's one-board policy |

---

## 5. Ranked suggestions

"No GPU" means it needs no GPU at all. "Verda" means it fits tonight's test.

| # | What to change (files) | Expected effect | Cost | Effort | Risk | GPU |
|---|---|---|---|---|---|---|
| 1 | **Fix the tensor-core ceilings.** Use datasheet per-SM rates (TF32 4,096, FP16/BF16 8,192, FP8 16,384 FLOP/clk/SM × 148 × 1.5 GHz = 0.91 / 1.82 / 3.64 PF/s), plus a separate "achieved by cuBLAS / by us" row. Files: `poc/leader_gap.py:35`, `playbook_compute.md:18`, a peak-rate table in `b200_arch.md`, `ledger.yaml` H30 for #35. Re-run leader_gap for #35 and #30. | Reopens #35's main-loop and W-once levers (estimate +0.01 to +0.03); removes the "needs FP8" trap on #30 | ~$0 | 1 h | Low. Optionally confirm with one resident-operand FP16 MMA probe (expect ~4,096 MAC/clk/SM). | No GPU |
| 2 | **Verda calibration** (§7). If it passes, run all compute-problem A/B ranking on Verda in end-of-round batches, keep Modal for building and debugging, and retire the emulator for compute problems. | Removes the 1.03-1.42 rented/portal spread; unblocks #30's H17 | $5-15 | 2 h | No root, or vGPU, means no lock. Fall back to cycles. | Verda |
| 3 | **Keep 2-3 lineages, with per-band elites and incubation.** `round.py` (`cmd_lead` must accept non-best parents; plans hold ≥ 2 tasks with at least one non-best lineage); `archive.py` (niche = size band × design family; a per-band best table); `planner.shortlist` (assemble a dispatcher from per-band elites, as r5-e1 once did); `LEAD_PROTOCOL` (an alternative lineage survives 2 rounds unless refuted on its own criterion). | Escapes refinement lock-in; per-band assembly is "free" score | More sessions per round (plan usage) | 1 day | Spreads attention; mitigate with cheap tiers | No GPU |
| 4 | **Idea tier + dedup + model mix.** A new `round.py ideas` command: 1 cheap call (Sonnet, or a non-Claude model through `loop/llm.py`) writes 10 ideas using verbalized sampling; dedup against ledger statements (embeddings or an LLM "same mechanism?" check); the lead picks across lenses. Run the paper session this way, without operator-written "settled facts". | More distinct ideas per dollar; fewer repeats of dead ends | $0.5-2 per batch | Half a day | Idea quality varies; the hivemind effect | No GPU |
| 5 | **Session workspace + diff edits.** `mcp_tools.py` / `designer.py`: files persist in the session directory; `write_file` / `edit_file(old, new)` tools; `run_tests` / `compile_b200` take paths; probes can import session files, and the files are shipped to Modal with each call. | Roughly −25-35% output tokens; fewer transcription bugs | ~$0 | Half a day | Low | No GPU |
| 6 | **Context diet + retrieval.** `prompts.dynamic_prompt`: a 1-2k "state card" (best per band, cited hypotheses, rules) instead of the full ledger and notebook; tools `ledger(id)`, `notebook(query)` and `read_source(lib, path, grep)` over local clones pinned to portal versions: CUTLASS 4.4.1 (sm100 pipeline/collectives), CuTe DSL Blackwell examples, DeepGEMM sm100, FlashAttention-4, ThunderKittens B200 GEMM. Add GEMM entries to `read_example`. Remove tricks.md from `FRESH_DOCS`; keep only the rules and harness sections. | Less anchoring; cheaper library reading (no B200 probes) | ~$0 | Half a day | A session misses a lesson; the tools mitigate it | No GPU |
| 7 | **Premise audit before each lead.** A cheap critic call that recomputes every numeric ideal, ceiling or floor in the ledger, tricks.md and the card from datasheet rates and raw measurements, and flags contradictions. Add a `depends_on:` field to hypotheses so a changed fact re-opens its dependants. Files: `round.py` (new step), ledger schema. | Would have caught §4.1 | ~$1 per round | 2 h | False alarms | No GPU |
| 8 | **Ledger hygiene.** After every session, auto-propose status changes from the card's `tests:` and its findings, and show them to the operator. Validate the lead's ledger rewrites: constraints must be a superset, ids preserved, store a diff (`round.py:282`). Fix #30 H13/H14/H15 now. | The ledger stays true without the lead | ~$0 | 2 h | Low | No GPU |
| 9 | **Two untried #30 levers** (plan in §6): the cuBLASLt Expert-API sweep at the lock, and L2 policy in the CUTLASS path (evict_first on R and A TMA loads; an evict_last or evict_first D store through a patched TMA store with `.L2::cache_hint`). Remove the aliasing advice from `playbook_compute.md` §2 (or mark it "forbidden unless NVIDIA approves"). | Maybe +0.01-0.03 on #30 | GPU $5-10 | 1-2 rounds | Might not transfer at the lock | Verda (sweep) |
| 10 | **Fix stale docs:** `core_brief.md:4,84`, `designer.py:35,51`, `mcp_tools.py:88`. Either give the emulator tensor-kernel features (`kind`, `cta_group`, stages, accumulator count, converter path) or switch it off for compute problems. | Fewer wrong cues | ~$0 | 1 h | Low | No GPU |
| 11 | **Ask NVIDIA** (one email or forum post) about (a) FP16 operands with fp32 accumulation on fp32-defined problems, when they pass the tolerance; (b) output aliasing an input (cuBLASLt C == D). | Removes a disqualification risk on #35; may unlock +0.05 on #30 | $0 | 15 min | None | No GPU |
| 12 | **Portal noise policy.** Treat ±0.008 as noise; use control-paired packages for changes under 5%. Board-keeps-best means a resubmission of identical code has positive expected value, about +0.003 for one more draw, but r10 was probably already a lucky draw, so only do it when a slot would otherwise go unused. | Avoids misreading noise | $0 | — | — | — |

**For tonight**, only items 2 and 9 (the sweep) need the B200. Items 1 and 11 take minutes and should go first, so that
the calibration is read against the right ceilings. Before the session, prepare the bootstrap and calibration scripts
(a no-GPU coding task), so that billed time is spent measuring.

---

## 6. Plan for the next 2-3 rounds on #30 (bf16 GEMM + residual)

**Where we stand.**
- Best score 0.4529 (r1 CUTLASS 256×224 transposed at L, cuBLASLt below 6,144 tokens). The c1 control scored 0.4490.
- Tb behaves like *plain* cuBLAS GEMM, with no add, at the lock (H16).
- Our fused kernels slow down 1.36× from rented to portal, while plain GEMMs slow down 1.22× (H9, H17), so a kernel
  that wins on the rented card ties Tb on the portal.
- Top 5 needs 0.491; the leader (KernelArc) has 0.546.
- KernelArc's intermediate bests were 0.4905 and 0.4915 before the final 0.546. We cannot tell which step added
  aliasing, but this hints that the non-aliasing part of their method was worth about 0.49.

**Roles.** A fresh lead session (not the main chat) is in charge each round. Explorers run in parallel on Modal for
building and correctness. A Verda batch at the end of each round does the ranking at the locked clock.

### Round r3: measure, then explore three lenses

**Tonight on Verda**, scripted, no LLM in the loop:
1. Calibration (§7).
2. #30 diagnostics at 1500 MHz across the 10 sizes, 3 interleaved repetitions each:
   - plain `torch.matmul`, compared with the stored Tb (expect a ratio of about 1.00);
   - r0 cuBLASLt fused (C ≠ D);
   - r1/c1 CUTLASS fused;
   - the same CUTLASS kernel with C = void (a diagnostic; never submitted).
   This tells us whether the portal's 1.36 vs 1.22 reproduces, and where the residual's cost sits at the lock.
3. The cuBLASLt Expert-API sweep for the fused C ≠ D call, per M: `cublasLtMatmulAlgoGetIds` × tile × stages ×
   cluster shape × split-K × swizzle, each passed through `AlgoCheck`; keep the top 5 per M by locked time. Our r1
   test only tried the 8 heuristic picks, all `nvjet`.

**Lead** (fresh session, about $3). Inputs:
- the corrected leader_gap (BF16 ceiling 1.82 PF/s);
- the Verda results;
- a fixed ledger (H13/H14 refuted; H15 "not settled, profile never ran"; H17 now testable);
- tricks.md §3;
- a short note on KernelArc's *published* method, marked "aliasing is off-limits under our rules".

Output: a 10-idea batch with verbalized probabilities, deduplicated, and three experiments in three different lenses.
Likely:

| Experiment | Lens | Model | What | Cost |
|---|---|---|---|---|
| E1 | Library | Sonnet, probe tier | Turn the sweep into a static per-M table (dispatch by shape only; 16/16 correct on 10 rounds) | ~$2 + GPU |
| E2 | Memory | Opus, build | L2 policy on the t224 CUTLASS path: evict_first on the R (C) and A TMA loads, then D-store hint variants through a patched TMA store; keep the C = void kernel as the reference. Hypothesis: C ≠ D's extra 3-5 µs is write-back of dirty L2 lines evicted by new D (and R) allocations. On r1's rented numbers, C = D cost 0.9 µs over plain and C ≠ D 5.9 µs at 8192. | ~$7 |
| E3 | Fresh | Paper tier on a non-Claude model, or Sonnet with no tricks.md and no ledger | "8 designs for D = A·Wᵀ + R with D a separate buffer, M = 256-8192, at 1.5 GHz", with the arithmetic that limits each, and a 10-minute probe for each | ~$1-2 |

**End of round.** A Verda batch ranks every candidate against the control, 3 interleaved repetitions (~30 min, $3-6).
Then one portal slot: the best per-M dispatcher, packaged as a shape-split A/B on the group where the two best variants
disagree.

### Round r4: combine, and keep one alternative alive

- The lead assembles per-M elites into one dispatcher (library vs CUTLASS vs E3's idea).
- One explorer continues the second-best lineage. This is the incubation rule: it is not judged against the best yet.
- If the leader-gap per band shows the S band (M ≤ 1571, 5 workloads) holds the remaining gap, add one Sonnet probe
  session on H3, a split-K reduction inside one launch (DSMEM or fp32 atomics plus an ordered epilogue).
- Verda ranking; one portal slot.

### Round r5 (only if needed): plateau gate

If r3 + r4 together move the portal by less than +0.01, or the locked bench shows no path to ≥ 0.49, **park #30** and
spend the next rounds on #35's reopened nb2 lever (§4.1). Its expected gain per dollar is likely higher. Resume #30
only if NVIDIA approves aliasing.

### Budget per round

| Item | API-equivalent / real cost |
|---|---|
| Lead | $3 |
| Idea batch | $1 |
| E1 | $2 |
| E2 | $7 |
| E3 | $2 |
| Claude total | ~$15 API-equivalent |
| Modal (building) | $3-5 real |
| Verda (ranking) | $3-6 real; less on spot |

Three rounds fit inside Verda's $30 credit if tonight's calibration uses spot or stays under 2 hours.

---

## 7. Verda checklist for tonight

Verda lists the B200 at about **$6.11/h on demand and about $2.14/h spot**, billed in 10-minute blocks; spot can be
evicted without warning. $30 buys about 4.9 h on demand or about 14 h on spot. Use spot for calibration if it is
available.

**Before you start (no GPU):**
- [ ] Item 1 of §5 is done, or at least noted, so the results are read against the right ceilings.
- [ ] Have a bootstrap script ready that:
  - clones the repo;
  - installs the portal stack: SOL-ExecBench `docker/Dockerfile`, or the same pins as `loop/b200_modal.py`
    (CUDA 13.1.1, CUTLASS 4.4.1, torch 2.9.0+cu130, cutlass-dsl 4.4.2);
  - copies the problem data.
- [ ] Have a calibration runner ready that takes a list of archive ids and writes per-workload JSON. It can be a
  Verda-target copy of `loop/calibrate_b200.py`.
- [ ] Set a phone timer for your budget, for example 2 h, and terminate the instance when it rings.

**Step 1: availability and identity (5 min)**
- [ ] On the Verda dashboard, is a 1× B200 available on spot or on demand? Note the price.
- [ ] `nvidia-smi -L`: the name must be exactly `NVIDIA B200`. B300/GB300 is a different chip.
- [ ] `nvidia-smi --query-gpu=name,driver_version,vbios_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,power.max_limit,temperature.gpu,ecc.mode.current --format=csv`
- [ ] The `nvidia-smi` header's "CUDA Version" must be ≥ 13.1, otherwise the portal stack will not run.
- [ ] `nvidia-smi -q | grep -i -A3 virtualization`: passthrough or bare metal is good; vGPU usually blocks clock
  control.
- [ ] Note `power.limit`. The B200's maximum is about 1000 W; a lower cap may throttle tensor-heavy runs even at a
  locked 1500 MHz.

**Step 2: can we lock the clock? (5 min; go/no-go)**
- [ ] `sudo -n true` checks for passwordless sudo, which the harness's own `clock_lock.py` uses as `sudo -n nvidia-smi`.
- [ ] `sudo nvidia-smi -pm 1`
- [ ] `sudo nvidia-smi -lgc 1500,1500`, then `sudo nvidia-smi -lmc 3996,3996`. These are the harness's B200 preset
  (`device_config.py`: 1500/3996). If `-lmc` fails, check `nvidia-smi -q -d SUPPORTED_CLOCKS`; a GPU-only lock is
  still very useful.
- [ ] `nvidia-smi -q -d CLOCK` should show SM 1500 and memory 3996.
- [ ] **Under load:** run a 60-second bf16 GEMM loop (`torch.mm` on 8192×2560×2560, in a loop) while sampling
  `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 200 > clocks.csv`.
  Pass if SM stays at 1500 ± 15 MHz with no "SW Power Cap", "HW Slowdown" or "thermal" reasons. If it sags, that is
  itself a finding: it is a candidate explanation for the portal's speed factor.
- [ ] Try `ncu --version` and one short `ncu --set basic` on a tiny kernel. Hardware profiler counters are usually
  blocked in containers (Modal); root on a VM may allow them. If they work, they are a big diagnostic win.
- [ ] **No-go:** if the lock is refused, record the error, still run Step 4's ratio test unlocked (cheap), and
  terminate.

**Step 3: environment (20-30 min)**
- [ ] Run the bootstrap. Check one known kernel passes 16/16 with the official harness (`run_tests`-equivalent), with
  `SOL_EXECBENCH_CLOCKS_LOCKED=1` if you use the harness's `lock_clocks` option.
- [ ] Warm up: 2 minutes of GEMM load, so all runs start from the same thermal state.

**Step 4: validate against portal data we already have (30-40 min)**

Run each kernel 3 times, interleaved (A B C ... A B C ...), with the official harness at the lock. In priority order:

| Kernel | Portal id | Kind | Rented/portal ratio | Why |
|---|---|---|---|---|
| #30 `r1-oproj-cutlass-t224-dispatch` and `c1-oproj-t224-control` | 63930 / 64523 | CUTLASS fused | 1.36 | Identical code submitted twice, so it also measures portal noise |
| #30 `r0-oproj-cublaslt-fused-v0` | 63900 | cuBLASLt | 1.22-1.40 by band | |
| `torch.matmul`, no residual, on #30 shapes | — | — | — | Compare with the stored Tb per workload; expect a ratio of about 1.00 |
| #35 `r10-e1-nb2-st4-919` | 65334 | FP16 tcgen05 | 1.09-1.28 | Its byte-identical B > 256 paths were also portal-timed in r8 and r9, which gives a portal noise band (speed factors 0.965-1.045) |
| #35 `r5-e2-tc-ws-uniform-issue` | 64927 | TF32 tcgen05 at the MMA ideal | 1.35 | |
| #85 `r1-geglu-efld-m` or #88 `r2-rope-pair2-magic-occ10` | 64549 / 64591 | DRAM-bound | 1.03-1.10 | |

For each kernel, compute the ratio portal µs / Verda µs per workload, then the geomean per kernel and per size band.

**Pass criteria:**
- [ ] Every kernel type's geomean ratio is within 0.97-1.03, against 1.03-1.42 on Modal.
- [ ] The spread of ratios *between* kind of kernels is under ~3%. This is what matters for ranking candidates.
- [ ] #30: `torch.matmul` at the lock ≈ Tb (within 2-3%), and the fused/plain gap at the lock matches the portal's
  (fused ≈ 1.03× Tb at 8192).
- [ ] #35: Verda's numbers sit inside the band the portal itself showed for identical code (r8/r9/r10).
- [ ] If the time is right but the *ordering* of kernels differs from the portal, write down which kernel type
  disagrees. Do not adopt the bench for that type.

**Step 5: if time is left (20 min)**
- [ ] Run #30's Expert-API sweep (§6, round r3, step 3) and save the top 5 per M.
- [ ] Optional thermal check: re-time one tensor kernel cold, and again after 5 minutes of GEMM burn. A difference over
  1% at a locked clock means the portal's speed factor could be thermal or power.

**Step 6: wrap up (5 min)**
- [ ] Copy back the result JSON, `clocks.csv` and the full `nvidia-smi -q` output (scp or rsync) into
  `problems/<name>/b200/verda_<date>/`.
- [ ] `sudo nvidia-smi -rgc; sudo nvidia-smi -rmc`, then **terminate the instance** (not just stop it). Check the
  billing page.

---

## 8. Sources

Agentic and evolutionary search
- AlphaEvolve paper: https://arxiv.org/abs/2506.13131 (full text: https://arxiv.org/html/2506.13131v1); DeepMind blog: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- FunSearch (Nature 2023): https://www.nature.com/articles/s41586-023-06924-6
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve ; database description in ThetaEvolve: https://arxiv.org/abs/2511.23473
- ShinkaEvolve: https://arxiv.org/abs/2509.19349 ; https://sakana.ai/shinka-evolve/
- Darwin Gödel Machine: https://arxiv.org/abs/2505.22954 ; https://sakana.ai/dgm/
- AI Scientist v2 (agentic tree search): https://arxiv.org/abs/2504.08066

Kernel agents
- KernelArc: https://arxiv.org/abs/2608.17071 (method, the L1-030 case study, ablation, Algorithm 3)
- Cursor multi-agent kernels: https://cursor.com/blog/multi-agent-kernels
- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173 (local copy: `SOL-ExecBench-LaTeX/04_dataset_evaluation.tex`, `05_experiments.tex`)
- Sakana AI CUDA Engineer: https://sakana.ai/ai-cuda-engineer/ ; TechCrunch on the walk-back: https://techcrunch.com/?p=2969225 ; robust-kbench: https://arxiv.org/abs/2509.14279
- CUDA Agent: https://arxiv.org/abs/2602.24286
- Kevin-32B: https://cognition.com/blog/kevin-32b ; https://arxiv.org/abs/2507.11948
- METR kernel engineering: https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering ; reward hacking: https://metr.org/blog/2025-06-05-recent-reward-hacking
- Stanford fast kernels: https://crfm.stanford.edu/2025/05/28/fast-kernels.html ; KernelBench v0.1 hacks: https://scalingintelligence.stanford.edu/blogs/kernelbenchv01
- AMD GEAK: https://arxiv.org/abs/2507.23194 ; https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html
- K-Search: https://arxiv.org/abs/2602.19128
- AccelOpt: https://arxiv.org/abs/2511.15915
- Harness engineering (MLSys 2026 FlashInfer contest, B200): https://arxiv.org/abs/2607.17979
- UW SyFI contest write-up: https://syfi.cs.washington.edu/blog/2026-05-28-mlsys-kernel-contest/
- Kernel-Smith: https://arxiv.org/abs/2603.28342

Creativity and diversity
- PlanSearch: https://arxiv.org/abs/2409.03733
- Si, Yang, Hashimoto, LLM research ideas (duplicates): https://arxiv.org/abs/2409.04109
- Verbalized Sampling: https://arxiv.org/abs/2510.01171
- Artificial Hivemind: https://arxiv.org/abs/2510.22954
- Temperature and creativity: https://arxiv.org/abs/2405.00492
- Wharton Prompting Science Report 3 (tips/threats): https://arxiv.org/abs/2508.00614 ; Report 1: https://arxiv.org/abs/2503.04818
- EmotionPrompt: https://arxiv.org/abs/2307.11760
- Personas in system prompts: https://arxiv.org/abs/2311.10054
- Multi-agent debate: https://arxiv.org/abs/2311.17371 ; follow-up: https://arxiv.org/abs/2502.08788
- LLMs cannot self-correct reasoning yet: https://arxiv.org/abs/2310.01798
- Analogical prompting: https://arxiv.org/abs/2310.01714

Blackwell kernel knowledge (to feed through `read_source` / `read_example`)
- FlashAttention-4: https://arxiv.org/abs/2603.05451 ; https://tridao.me/blog/2026/flash4/
- Modular, Matrix Multiplication on Blackwell, part 3: https://modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-3-the-optimizations-behind-85-of-sota-performance ; part 4: https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota
- Colfax CUTLASS tutorials: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ ; https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/ ; CLC: https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/
- CUTLASS Blackwell functionality: https://docs.nvidia.com/cutlass/4.4.1/media/docs/cpp/blackwell_functionality.html ; CuTe DSL: https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/cute_dsl.html
- Triton Gluon tcgen05 tutorial: https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html
- ThunderKittens on Blackwell: https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- PTX ISA (tcgen05 section): https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- GPU MODE / NVIDIA NVFP4 hackathon: https://forums.developer.nvidia.com/t/blackwell-nvfp4-kernel-hackathon/350212 ; participant write-ups: https://gauravjain.bearblog.dev/tcgen05-gemm-opt-part1/ , https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4 , https://veitner.bearblog.dev/nvfp4-gemv/

Verda
- B200 pricing: https://www.verda.com/b200 ; instance setup docs: https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/

Repository evidence (paths relative to the repo root)
- `poc/leader_gap.py:35`; `loop/context/playbook_compute.md:18`
- `problems/035_flux_ada_layer_norm_zero_modulation_extraction/ledger.yaml:71,242,267`
- `problems/035_.../rounds/r11/plan.json`; `rounds/r11/sessions/paper_task.txt:20`; `paper.reply.md:9`
- `problems/030_attention_output_projection_with_residual/ledger.yaml` (H6, H9, H13-H17); `rounds/r2/candidates/*.card.yaml`
- `loop/round.py:188-224,282,289`; `loop/prompts.py:15,31,94`; `loop/designer.py:35,51,53`; `loop/emulator.py:26,125`; `loop/mcp_tools.py:88`
- `loop/context/core_brief.md:4,84`; `loop/runs/usage.jsonl`; `poc/results/b200_portal.csv`
- `SOL-ExecBench/src/sol_execbench/core/bench/clock_lock.py`; `config/device_config.py` (B200 preset 1500/3996)
