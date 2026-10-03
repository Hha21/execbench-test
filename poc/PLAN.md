# Problem 038 on B200: where we are and the plan

Target: SOL-ExecBench L1 `038_flux_multi_head_rmsnorm_qk` (portal kernel #38): per-head RMSNorm of Q and K,
fp32, shapes `[B, S, 48, 128]`, 16 input sizes. Pure memory traffic: read Q and K, write two outputs.

## 1. Results so far (4 October 2026)

**Leaderboard anchors** (public, B200, v1.1):

| | Latency | Score |
|---|---|---|
| SOL bound (SOLAR) | 9.2 µs | 1.0 |
| Leader (Infinigence AI) | 22.2 µs | 0.628 |
| 20th place | 25.5 µs | 0.573 |
| Hidden scoring baseline | 31.8 µs | 0.5 |
| PyTorch reference | 240 µs | 0.09 |

Places 1–20 are within 15% of each other. Other teams' code cannot be downloaded (the portal returns 401
without a login, and logged in it shows only the reference), so real B200 numbers have to come from our own
submissions.

**What the proof of concept showed** (details in `README.md`):

- 72 Triton variants of one design, timed on A100 and L40S with NVIDIA's scorer: all correct, 0.15% noise.
- Rehearsal with one GPU hidden as the expensive target: on A100 the emulator finds the best variant in about
  8 measurements against about 20 for plain Bayesian optimisation; on L40S (different memory type) it only
  helps early. The simulator improves the overall ranking but misleads the top pick because it only saw
  small inputs.
- Our best A100 kernel reaches about 80% of peak bandwidth on large inputs and about 60% on the smallest,
  where fixed start-up cost dominates. Picking the best variant per input size gains only 1.4% (A100) and
  2.3% (L40S): the remaining gap is in the design, not the knobs.
- Triton 3.7 cross-compiles a TMA (bulk asynchronous copy) version of the kernel for `sm_100` without a B200:
  the machine code uses `UTMALDG`/`UTMASTG` and no ordinary loads. H200 has TMA too, so it can run and time
  such designs; A100 and L40S cannot.

**What it implies (estimates, to be confirmed by the first portal results):**

- Just moving the bytes at 8 TB/s takes about 16 µs (geometric mean over the 16 sizes), so SOLAR's 9.2 µs
  is not reachable and scores above roughly 0.75 are out of reach for anyone. It also suggests the portal's
  aggregate latency is a geometric mean: the arithmetic-mean floor would be 32 µs, slower than the leader.
- The leader is at about 72% of that floor. Our A100 kernel is at 76% of the A100's floor. If the same
  efficiency carried over to B200 we would be near 21 µs, a score around 0.65 (first place). It probably
  will not carry over unchanged: B200 needs about 4× more bytes in flight to stay busy, and fixed start-up
  cost is a bigger share of a B200 run on the small sizes.
- Rough score targets: about 4% faster than the leader gives 0.65, about 15% faster gives 0.70.

## 2. The loop

Three nested loops, each feeding the one outside it. The expensive outer loop runs on the portal's free
B200 slots; everything inside it runs on CSF3 for free.

```
 ┌─ Outer (daily, B200 portal, ~5 results/day without delay) ───────────────────────────┐
 │  pick 3 "best predicted" + 2 "most informative" candidates → submit privately        │
 │  results update the emulator (blend weights + GP correction) → re-rank the archive   │
 │ ┌─ Middle (hours, CSF3) ─────────────────────────────────────────────────────────┐   │
 │ │  emulator scores every candidate with uncertainty                               │   │
 │ │  archive keeps the best per design niche (diversity, not just the top few)      │   │
 │ │  simulator on a shortlist, now including large sizes                            │   │
 │ │ ┌─ Inner (minutes, CSF3) ───────────────────────────────────────────────────┐   │   │
 │ │ │  LLM proposes new designs / mutations, given the B200 brief + best so far │   │   │
 │ │ │  gate: compiles for sm_100a, passes the harness's correctness checks       │   │   │
 │ │ │  measure: H200 (+A100/L40S where portable), sm_100a code features, maths   │   │   │
 │ │ └───────────────────────────────────────────────────────────────────────────┘   │   │
 │ └─────────────────────────────────────────────────────────────────────────────────┘   │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

- **Generator.** The LLM is the mutation operator: it edits whole designs, not just knobs. Its prompt
  carries a short B200 brief (TMA / bulk copies and mbarriers, bytes in flight per SM, 148 SMs, 8 TB/s,
  cache hints, persistent vs one-shot grids, per-size dispatch is allowed) plus the current archive with
  measured and predicted results. Start with me in Claude Code as the operator; script it later.
- **Fitness.** Multi-fidelity, as in the proof of concept: cheap sources give a predicted B200 time and an
  uncertainty; real B200 results correct the predictor. Fitness is the predicted portal score (mean over
  sizes), not raw speed on any one chip.
- **Selection pressure without collapse.** Keep an archive of the best candidate per niche (memory path:
  plain loads vs TMA; launch: one-shot vs persistent; small-size vs large-size specialists). The rehearsal
  showed a cheap source can confidently favour the wrong family; niches stop the population collapsing onto
  the emulator's blind spot, and "most informative" submissions test exactly those blind spots.
- **Final kernel.** Dispatch by input size to the best specialist (legitimate: it chooses a launch
  configuration, it does not cache results).
- **Guardrails.** Correctness at the problem's 1e-5 tolerance; no extra streams, threads, caching or other
  reward hacks (the portal disqualifies them); private submissions until clearly ahead.

## 3. Submitting

- **Now: you submit by hand.** Five private uploads a day get results without delay. Record each result in
  `results/b200_portal.csv` (`vid,latency_ms,sol_score`) and rerun `b200_plan.py`.
- **Later, if NVIDIA agrees: automate.** The upload is one HTTP request authorised by your login token.
  A small script could upload from `b200_submit/`, poll for results and append to the CSV, respecting the
  rate limits. Ask solexecbench@nvidia.com first; it would use your token, which only you should handle.

## 4. Next steps

1. **You:** submit `b200_submit/v028.json` (the A100's best) privately to check Triton runs on the portal,
   then the other four (v039, v023, v037, v046). This gives the first real B200 numbers and tells us how
   far A100 efficiency carries over.
2. **Tonight (automatic):** H200 timings, then a three-chip rehearsal and a refreshed B200 plan.
3. **Me:** design generation 2, aimed at B200: a TMA version for large sizes (more bytes in flight), a
   latency-minimised one-launch version for small sizes, and a size dispatcher. Check them on H200.
4. **Me:** fix the simulator's blind spot (add large sizes) and make the emulator work on code features so
   it can score designs the LLM writes, not just knob settings.
5. **Together:** after 5–10 portal results, decide whether the emulator is earning its keep on B200 (does
   it rank the submitted candidates correctly?) before scripting the LLM loop.

## 5. Risks

- The portal's Triton build may hit the same `launch.h` packaging bug we patched locally (step 1 checks).
- The cheap sources may rank B200 candidates poorly, like L40S in the rehearsal. Steps 1 and 5 measure this
  early; the archive and the "most informative" submissions limit the damage.
- The field is tight and active (entries were submitted as recently as 3 October). A clear win needs a
  design change, not tuning.
