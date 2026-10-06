"""Compile any Triton candidate for sm_100a (B200) on a non-B200 GPU node and read its resource use.

For each candidate, a fresh subprocess runs the solution's run() once per distinct input size with Triton's
current target patched to sm_100 and a private TRITON_CACHE_DIR, so every compiler pass targets B200 (the
TRITON_OVERRIDE_ARCH knob only changes the last stage and breaks TMA kernels on an A100 host: "Cannot select
intrinsic elect.sync"). Loading the B200 code on this GPU then fails, which is expected and caught. The cubins left in the cache give registers, shared memory, spills and the
load/store mix of the code B200 would run, whatever kernels the candidate defines.

  python loop/static_any.py --candidates loop/rounds/<r>/candidates --problem $PROBLEM038 --out <file.jsonl>
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

B200 = dict(threads=2048, regs=65536, smem=233472)

RUNNER = r"""
import importlib.util, json, sys, torch
import triton
from triton.backends.compiler import GPUTarget
# Make every compiler pass see a B200 (sm_100), not the GPU this job runs on. Launches then fail, as intended.
triton.runtime.driver.active.get_current_target = lambda: GPUTarget("cuda", 100, 32)
spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
entry = sys.argv[2]
fn = getattr(mod, entry)
sizes = json.loads(sys.argv[3])
for b, s in sizes:
    q = torch.randn(b, s, 48, 128, device="cuda"); k = torch.randn_like(q)
    wq = torch.randn(48, 128, device="cuda"); wk = torch.randn_like(wq)
    try:
        fn(q, k, wq, wk, 1e-6, torch.empty_like(q), torch.empty_like(k))
        torch.cuda.synchronize()
    except Exception as e:
        print(f"{b},{s}: {type(e).__name__}", file=sys.stderr)
"""


def sizes(problem):
    seen, out = set(), []
    for line in open(Path(problem) / "workload.jsonl"):
        a = json.loads(line)["axes"]
        if a["batch_size"] * a["seq_len"] not in seen:
            seen.add(a["batch_size"] * a["seq_len"])
            out.append((a["batch_size"], a["seq_len"]))
    return out


def tools():
    import triton
    return Path(triton.__file__).parent / "backends" / "nvidia" / "bin"


def inspect_cubin(path, meta):
    t = tools()
    res = subprocess.run([t / "cuobjdump", "-res-usage", str(path)], capture_output=True, text=True).stdout
    sass = subprocess.run([t / "cuobjdump", "-sass", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    ops = collections.Counter(mm[1].split(".")[0] for mm in
                              re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", sass))
    regs, local = (int(m[1]), int(m[4])) if m else (None, None)
    smem = meta.get("shared", 0)
    warps = meta.get("num_warps", 4)
    fit = None
    if regs:
        by_regs = (B200["regs"] // (-(-regs // 8) * 8 * 32)) // warps
        fit = min(32, B200["threads"] // (32 * warps), by_regs, B200["smem"] // (smem + 1024) if smem else 32)
    return dict(kernel=meta.get("name", path.stem), num_warps=warps, regs=regs, smem=smem, local=local,
                resident_per_sm=fit, target=meta.get("target"),
                ops={k: v for k, v in ops.items() if k.startswith(("LDG", "STG", "UTMA", "LDGSTS", "UBLKCP"))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--problem", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sz = json.dumps(sizes(args.problem))
    recs = []
    for sol_path in sorted(args.candidates.glob("*.json")):
        sol = json.loads(sol_path.read_text())
        langs = sol["spec"].get("languages", [])
        if not any(l in ("triton", "pytorch", "cute_dsl", "cutile") for l in langs):
            recs.append(dict(id=sol_path.stem, error=f"static capture only supports Python languages, got {langs}"))
            continue
        work = Path(tempfile.mkdtemp())
        for s in sol["sources"]:
            (work / s["path"]).parent.mkdir(parents=True, exist_ok=True)
            (work / s["path"]).write_text(s["content"])
        file, func = sol["spec"]["entry_point"].split("::")
        cache = work / "triton_cache"
        env = {**{k: v for k, v in os.environ.items() if k != "TRITON_OVERRIDE_ARCH"}, "TRITON_CACHE_DIR": str(cache)}
        p = subprocess.run([sys.executable, "-c", RUNNER, str(work / file), func, sz], env=env, cwd=work,
                           capture_output=True, text=True, timeout=900)
        kernels = []
        for cubin in cache.rglob("*.cubin"):
            meta_path = cubin.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            kernels.append(inspect_cubin(cubin, meta))
        rec = dict(id=sol_path.stem, kernels=kernels)
        if not kernels:
            rec["error"] = "no sm_100 cubin produced: " + (p.stderr[-1500:] or p.stdout[-500:])
        recs.append(rec)
        print(json.dumps({"id": rec["id"], "kernels": [(k["kernel"], k["regs"], k["smem"], k["resident_per_sm"])
                                                        for k in kernels], "error": rec.get("error", "")[:200]}),
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


if __name__ == "__main__":
    main()
