"""Time every variant on the current GPU with the official sol-execbench CLI.

Writes one JSONL trace file per variant (one line per workload) under
results/timing/<gpu_tag>/ and skips variants that already have a complete trace,
so the job can be resubmitted after a timeout. The first few variants are timed
twice more at the end to measure run-to-run noise on this GPU.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def gpu_tag():
    name = torch.cuda.get_device_name(0)
    for tag in ("B200", "H200", "H100", "A100", "L40S"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def n_workloads(problem_dir):
    return sum(1 for line in open(problem_dir / "workload.jsonl") if line.strip())


def statuses(trace):
    if not trace.exists():
        return []
    return [(json.loads(line).get("evaluation") or {}).get("status") for line in open(trace) if line.strip()]


def complete(trace, expected):
    """A trace is finished when every workload has a result (passed or not)."""
    return len(statuses(trace)) == expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=Path, required=True)
    ap.add_argument("--variants", type=Path, default=HERE / "variants")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "timing")
    ap.add_argument("--replicates", type=int, default=4, help="variants re-timed to estimate noise")
    args = ap.parse_args()

    tag = gpu_tag()
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    expected = n_workloads(args.problem)
    sols = sorted(args.variants.glob("v*.json"))
    jobs = [(s, out / f"{s.stem}.jsonl") for s in sols]
    jobs += [(s, out / f"{s.stem}_rep{r}.jsonl") for s in sols[: args.replicates] for r in (1, 2)]

    print(f"GPU {torch.cuda.get_device_name(0)} -> {out}; {len(jobs)} runs", flush=True)
    for sol, trace in jobs:
        if complete(trace, expected):
            continue
        t0 = time.time()
        cmd = ["sol-execbench", str(args.problem), "--solution", str(sol), "-o", str(trace), "--timeout", "900"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        st = statuses(trace)
        passed = sum(s == "PASSED" for s in st)
        status = "ok" if passed == expected else f"FAILED rc={proc.returncode} {sorted(set(st))}"
        print(f"{trace.stem}: {status} ({passed}/{expected} passed) in {time.time() - t0:.0f}s", flush=True)
        if status != "ok":
            (out / f"{trace.stem}.log").write_text(proc.stdout[-20000:] + "\n---\n" + proc.stderr[-20000:])
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
