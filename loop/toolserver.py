"""GPU tool server for interactive design sessions. Runs on a CSF3 GPU node, inside the CUDA 13.1 container.

Polls <queue>/req/*.json and handles one request at a time (so timings never overlap), writing <queue>/res/<id>.json:
  compile  {"id", "solution"} -> sm_100a registers, shared memory, spills, CTAs per SM and load/store widths
  test     {"id", "solution"} -> NVIDIA's harness on this GPU: per-workload status, errors and latency
At start it times the reference solution (--ref) once and writes <queue>/ref.json, so tests compare on one GPU.
It touches <queue>/alive on every poll, and exits when <queue>/stop appears or after --idle seconds without requests.

  python loop/toolserver.py --queue $SOLX/toolq/<name> --ref loop/rounds/r3/candidates/<best>.json --problem $PROBLEM038
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def harness(problem, sol_path, trace):
    """Run NVIDIA's harness on one solution; returns (records, console tail)."""
    cmd = ["sol-execbench", str(problem), "--solution", str(sol_path), "-o", str(trace), "--timeout", "900",
           "--compile-timeout", "900"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    recs = []
    if trace.exists():
        for line in open(trace):
            if line.strip():
                t = json.loads(line)
                ev, a = t.get("evaluation") or {}, t["workload"]["axes"]
                lat = (ev.get("performance") or {}).get("latency_ms")
                corr = ev.get("correctness") or {}
                recs.append(dict(workload=f"{a['batch_size']},{a['seq_len']}", status=ev.get("status"),
                                 latency_us=round(lat * 1e3, 2) if lat else None,
                                 max_abs_err=corr.get("max_absolute_error"), max_rel_err=corr.get("max_relative_error"),
                                 log=(ev.get("log") or "")[-1500:]))
    return recs, (p.stdout[-4000:] + "\n---\n" + p.stderr[-4000:]).strip()


def handle(req, work, problem):
    kind, sol = req["kind"], req["solution"]
    d = work / req["id"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    sol_path = d / f"{req['id']}.json"
    sol_path.write_text(json.dumps(sol))
    t0 = time.time()
    if kind == "compile":
        out = d / "static.jsonl"
        p = subprocess.run([sys.executable, str(HERE / "static_any.py"), "--candidates", str(d), "--problem",
                            str(problem), "--out", str(out)], capture_output=True, text=True, timeout=1200)
        rec = json.loads(out.read_text().splitlines()[0]) if out.exists() and out.read_text().strip() else {}
        res = dict(kernels=rec.get("kernels", []), error=rec.get("error") or ("" if rec else p.stderr[-3000:]))
    elif kind == "test":
        recs, console = harness(problem, sol_path, d / "trace.jsonl")
        res = dict(workloads=recs, console_tail="" if recs and all(r["status"] == "PASSED" for r in recs) else console)
    else:
        res = dict(error=f"unknown request kind {kind!r}")
    res.update(id=req["id"], kind=kind, seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True, help="reference solution JSON, timed once at start")
    ap.add_argument("--problem", type=Path, required=True)
    ap.add_argument("--idle", type=int, default=2400, help="exit after this many seconds without a request")
    a = ap.parse_args()
    q = a.queue
    for sub in ("req", "res", "work"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    (q / "alive").write_text(gpu)
    print(f"tool server on {gpu}, queue {q}", flush=True)
    if not (q / "ref.json").exists():
        recs, console = harness(a.problem, a.ref, q / "work" / "ref_trace.jsonl")
        write_json(q / "ref.json", dict(ref=a.ref.stem, gpu=gpu, workloads=recs, console_tail=console if not recs else ""))
        print(f"reference {a.ref.stem}: {sum(r['status'] == 'PASSED' for r in recs)}/{len(recs)} passed", flush=True)
    last = time.time()
    while not (q / "stop").exists() and time.time() - last < a.idle:
        (q / "alive").touch()
        reqs = sorted((q / "req").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not reqs:
            time.sleep(2)
            continue
        path = reqs[0]
        req = json.loads(path.read_text())
        path.unlink()
        print(f"{time.strftime('%H:%M:%S')} {req['kind']} {req['id']}", flush=True)
        try:
            res = handle(req, q / "work", a.problem)
        except Exception as e:      # report and keep serving
            res = dict(id=req["id"], kind=req["kind"], error=f"{type(e).__name__}: {e}")
        write_json(q / "res" / f"{req['id']}.json", res)
        print(f"{time.strftime('%H:%M:%S')} done {req['id']} in {res.get('seconds', '?')} s", flush=True)
        last = time.time()
    print("stopping", flush=True)


if __name__ == "__main__":
    main()
