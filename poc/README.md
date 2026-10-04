# Multi-source B200 emulator: proof of concept

Question: can cheap information sources predict how GPU kernel variants rank on a B200 well
enough to steer the search, so that the expensive B200 measurements are spent only on
promising candidates?

The test kernel is SOL-ExecBench `L1/038_flux_multi_head_rmsnorm_qk` (per-head RMSNorm of Q and K, fp32,
memory-bound, 16 workloads). The leaderboard's best score is 0.628 at 0.022 ms; the SOL bound is 0.009 ms.

## Cheap sources

| Source | Script | Needs |
|---|---|---|
| Timing on other GPUs (A100, L40S, H200 on CSF3) | `run_timing.py` | a GPU of that type |
| Compiled-code features for each architecture, including B200's `sm_100a` | `static_features.py` | CPU only |
| Analytical model (bytes / achievable bandwidth, occupancy, launches) | `analytic.py` | nothing |
| Cycle-level simulation (FlashGPU-Sim, `SM90_H100` config) | `simulate.py` | CPU only |

`emulator.py` fits `log t_target = ridge blend of sources + GP correction(knobs, size, compiled-code features)`
and runs the rehearsal: pretend one GPU we *can* measure is the expensive target, reveal n of its
variants, and score the ranking of the rest against simpler alternatives (trust another GPU, trust
the maths model, or plain Bayesian optimisation on the target alone). `bo_race` replays a
sequential search to show how many target measurements each strategy needs to find the best variant.

## Layout on CSF3

Everything lives in `/scratch/t95317ha/solx` (visible from every node; `$HOME` is 93% full and
`~/h200-scratch` is not mounted on A100/L40S nodes).

```
env.sh                    source first: caches, venv, PROBLEM038
venv/                     matches NVIDIA's evaluation image (torch 2.9.0+cu130, fbtriton 3.7.1)
SOL-ExecBench/            harness at commit a9fa080, dataset in data/benchmark
flashgpusim/              FlashGPU-Sim built with CUDA 12.8.1
poc/                      this directory
jobs/                     sbatch scripts
logs/                     job output
```

## Running it

```bash
source /scratch/t95317ha/solx/env.sh && cd $SOLX
python poc/make_variants.py                                  # 72 variants -> poc/variants/
sbatch jobs/static.sbatch                                    # compiled-code features, CPU
sbatch -A gpu-sk01 -p gpuA jobs/timing.sbatch                # A100
sbatch -A gpu-sk01 -p gpuL jobs/timing.sbatch                # L40S
sbatch -A gpu-cdt-dmcs -p gpuH_short jobs/timing.sbatch      # H200 (long queue)
sbatch -a 0-71%16 jobs/sim.sbatch SM90_H100 sm90             # simulator, CPU
cd poc && python emulator.py                                 # rehearsal report
```

Only `gpu-cdt-dmcs` can use the H200 partitions; `gpu-sk01` gets A100/L40S immediately.

## Feeding in B200 results from the portal

1. Upload the files in `poc/b200_submit/` to the portal as **private** kernel submissions for
   `038_flux_multi_head_rmsnorm_qk` (one `.json` per submission, B200). Send one first and check it
   runs (see the Triton note below).
2. Add one line per result to `poc/results/b200_portal.csv`:

   ```
   vid,latency_ms,sol_score
   v050,0.0231,0.61
   ```

3. Rerun `python b200_plan.py` (or `sbatch jobs/b200_plan.sbatch`). Below 3 results it ranks by the
   average of the cheap sources; from 3 on it fits the blend + GP to the B200 numbers and picks the
   next batch by lowest optimistic estimate. It prints the fitted blend weights, which show how much
   each cheap source is trusted for B200.

## Results (3 October 2026, A100 + L40S, 72 variants)

Run-to-run noise is about 0.15% (log sd), so the differences between variants are real. Variants
range from 83 µs to 870 µs (geometric mean over the 16 workloads, A100). A100 and L40S agree on the
broad ranking (Spearman 0.77) but not on the winner.

**Search race**: a method decides which variant to measure next on the hidden target. The table shows
how much slower its best find so far is than the true best (mean of 20 runs, all start from the same
3 random variants).

| Target measurements | A100: emulator | random | plain BO | L40S: emulator | random | plain BO |
|---|---|---|---|---|---|---|
| 5 | **1.9%** | 17.6% | 18.8% | **4.0%** | 5.5% | 4.7% |
| 8 | **0.5%** | 15.2% | 13.2% | **2.1%** | 3.6% | 3.2% |
| 10 | **0.1%** | 12.8% | 8.2% | **1.3%** | 3.4% | 2.5% |
| 20 | 0.1% | 2.6% | 0.8% | 0.6% | 2.1% | **0.0%** |

**One-shot ranking** after n random target measurements (median Spearman / how much slower the
predicted best is than the true best):

| n | A100 emulator | A100 plain BO | L40S emulator | L40S plain BO |
|---|---|---|---|---|
| 0 | maths model 0.88 / 7.6%; L40S timing 0.77 / 31% | – | A100 timing 0.77 / 4.8% | – |
| 3 | 0.86 / 2.1% | 0.13 / 33% | 0.75 / 4.8% | 0.28 / 7.1% |
| 10 | 0.86 / 2.1% | 0.51 / 5.8% | 0.73 / 8.8% | 0.26 / 4.0% |
| 20 | 0.87 / 1.8% | 0.69 / 3.9% | 0.72 / 4.8% | 0.61 / 1.8% |

Reading: the emulator helps a lot when the cheap sources resemble the target (A100, HBM memory:
best variant found in about 8 measurements instead of about 20). It helps only early when they do
not (L40S, GDDR6 memory, where the maths model is poor and most variants are nearly tied). B200 has
HBM like the A100 and H200, which is encouraging but untested.

**Simulator** (`results/rehearsal_sim`): on its own the cold H100 config ranks A100 variants at 0.91,
better than the maths model, and adding it raises the emulator's A100 ranking to 0.90–0.92. But its
favourite variant is 16% slower than the A100's best, so one-shot picks get worse; in the search race
it still helps slightly (0.1% after 8 measurements). Cause: it only simulates the 4 smallest inputs,
and its favourites (fused, persistent, 2–16 rows per tile) are good on small inputs but rank 11th–32nd
on the large ones. Next step: simulate one or two large workloads too, or use the simulator only for
the workloads it covered.

## Three-chip rehearsal (4 October 2026, A100 + H200 + L40S)

`results/rehearsal_with_h200`. Search race, mean gap to the true best after k target measurements:

| Target | Best zero-shot source | Emulator k=5 / k=8 | Random k=8 | Plain BO k=8 / k=20 |
|---|---|---|---|---|
| A100 | H200 timing: exact winner | 0.9% / **0.0%** | 15.2% | 13.2% / 0.8% |
| H200 | A100 timing: exact winner | 4.6% / **0.0%** | 15.9% | 12.6% / 3.8% |
| L40S | H200 timing: 4.8% off | 4.5% / 3.5% | 4.2% | 3.0% / 0.6% |

The two HBM chips predict each other's winner exactly with no target data. The emulator reaches the best
variant in 8 measurements on either; plain BO is still 4% off on H200 after 20. Weakness: one-shot picks with
exactly 5 target points are unstable (H200 median 23% off) because the non-negative blend can lean on L40S or the
maths model when data are scarce. Next fix: start the blend from "trust the closest chip" and only move away
as target data accumulate (a prior on the weights).

## Queued and next steps

Queued on CSF3 at the end of 3 October:

- H200 timing (`solx_timing`, gpuH_short, estimated start 01:30 on 4 October), then automatically a
  three-chip rehearsal (`results/rehearsal_with_h200`) and a refreshed B200 plan (`solx_b200plan`).

Next:

1. Submit one variant from `b200_submit/` to the portal (private) to confirm Triton runs there.
2. Submit the rest of the batch, record results in `results/b200_portal.csv`, rerun `b200_plan.py`.
3. Simulator: add one or two large workloads (or per-workload simulator features).
4. Widen the variant space (more designs, not just knobs) once the loop works on B200.

## Model details that mattered

Three choices in `Emulator.fit` came out of the rehearsal; the first versions got them wrong.

1. **Workload fixed effects.** Most of the variance in log-latency is input size, which every
   source gets right and which does not affect the ranking. Each workload gets its own level; the
   blend is fitted on how variants differ.
2. **Non-negative weights on standardised sources.** Ridge shrank the useful source toward zero and
   occasionally gave it a negative weight. A cheap source should never vote the wrong way, so the
   blend is a non-negative least-squares fit.
3. **Fit the blend on per-variant means.** Per-row fitting let a few very slow variants on large
   inputs dominate the weights. With L40S as target, ranking accuracy at n=10 went from 0.43 to 0.72.

## Simulator caveats

- FlashGPU-Sim's PTX parser rejects `createpolicy` (cache-eviction hints), so `EVICT=True` variants
  are simulated without the hint. The simulator cannot see that knob.
- `-gpgpu_perf_sim_memcpy 1` (the shipped default) pre-loads inputs into L2, unlike the harness,
  which flushes L2 before every timed run. `SM90_H100_cold` and `SM100_B200_approx` turn it off and
  flush L2 between kernels.
- With L2 flushing on, about 1 in 20 runs loops on "Dirty lines flushed" and never prints its
  cycle count. Those workloads are dropped; the per-variant simulator score is built from
  per-workload deviations so a missing workload does not bias it.
- `SM100_B200_approx` is the H100 config with 148 SMs and 190 memory channels (about 8 TB/s and
  124 MB of L2). It does not model tcgen05/TMEM, which this kernel does not use.
- One simulated variant takes about 6 minutes on 4 cores for the four smallest workloads.

## Environment fix

The `fbtriton==3.7.1` wheel that NVIDIA's Dockerfile installs ships `triton/runtime/launch.h` but its
NVIDIA driver only looks in `triton/backends/nvidia/`, so every Triton launch fails with
`launch.h not found`. Fixed locally with a symlink:

```bash
ln -sf ../../runtime/launch.h $SOLX/venv/lib/python3.12/site-packages/triton/backends/nvidia/launch.h
```

If the portal image has the same wheel, Triton submissions may fail there too, so the first portal
submission should be a single Triton variant to confirm it runs.
