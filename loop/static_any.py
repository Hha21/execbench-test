"""Compile any Triton, CuTe DSL or CUDA C++ candidate for sm_100a (B200) on a non-B200 GPU node and read its resource use.

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


def op_key(name):
    """SASS opcode -> base name, cache-hint modifiers and access width, e.g. LDG.E.EF.128 -> LDG.EF.128.

    Kept modifiers: EF/EL (evict first/last), LU (last use), NA (no L1 allocate), CONSTANT (read-only path).
    """
    parts = name.split(".")
    mods = [m for m in parts[1:] if m in ("EF", "EL", "LU", "NA", "CONSTANT")]
    w = re.search(r"\.(64|128|256)(?=\.|$)", name)
    return ".".join([parts[0], *mods] + ([w[1]] if w else []))


def inspect_cubin(path, meta):
    """One record per kernel function in the cubin (Triton cubins hold one; CUDA C++ ones may hold several)."""
    t = tools()
    res = subprocess.run([t / "cuobjdump", "-res-usage", str(path)], capture_output=True, text=True).stdout
    sass = subprocess.run([t / "cuobjdump", "-sass", str(path)], capture_output=True, text=True).stdout
    ops = collections.Counter(op_key(mm[1]) for mm in
                              re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", sass))
    ops = {k: v for k, v in ops.items() if k.startswith(("LDG", "STG", "UTMA", "LDGSTS", "UBLKCP"))}
    out = []
    for fn, regs, smem_static, local in re.findall(
            r"Function ([^\s:]+):\s*\n\s*REG:(\d+) STACK:\d+ SHARED:(\d+) LOCAL:(\d+)", res):
        regs, local = int(regs), int(local)
        smem = meta.get("shared", int(smem_static))      # Triton: total incl. dynamic; CUDA: static only
        warps = meta.get("num_warps")
        fit = None
        if warps:
            by_regs = (B200["regs"] // (-(-regs // 8) * 8 * 32)) // warps
            fit = min(32, B200["threads"] // (32 * warps), by_regs, B200["smem"] // (smem + 1024) if smem else 32)
        out.append(dict(kernel=meta.get("name", fn), num_warps=warps, regs=regs, smem=smem, local=local,
                        resident_per_sm=fit, ops=ops))
    return out


def cuda_cubins(work, sol):
    """Compile each .cu source for sm_100a with nvcc (device code only) against PyTorch's headers."""
    import sysconfig
    import torch
    from torch.utils.cpp_extension import include_paths
    incs = include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]
    flags = [f for f in sol["spec"].get("compile_options", {}).get("cuda_cflags", ["-O3", "--use_fast_math"])
             if not f.startswith(("-gencode", "-arch"))]
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    cubins, errors = [], []
    for s in sol["sources"]:
        if not s["path"].endswith(".cu"):
            continue
        out = work / (Path(s["path"]).stem + ".sm100a.cubin")
        cmd = ["nvcc", "-arch=sm_100a", "-cubin", *flags, f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
               "-DTORCH_EXTENSION_NAME=cand", *[f"-I{i}" for i in incs], str(work / s["path"]), "-o", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=work, timeout=900)
        (cubins if p.returncode == 0 else errors).append(out if p.returncode == 0 else p.stderr[-1500:])
    return cubins, errors


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
        work = Path(tempfile.mkdtemp())
        for s in sol["sources"]:
            (work / s["path"]).parent.mkdir(parents=True, exist_ok=True)
            (work / s["path"]).write_text(s["content"])
        if any(l in ("cuda_cpp", "cutlass", "cudnn", "cublas") for l in langs):
            cubins, errors = cuda_cubins(work, sol)
            kernels = [k for c in cubins for k in inspect_cubin(c, {})]
            rec = dict(id=sol_path.stem, kernels=kernels, **({"error": " | ".join(errors)} if errors else {}))
            recs.append(rec)
            print(json.dumps({"id": rec["id"], "kernels": [(k["kernel"][:40], k["regs"], k["smem"]) for k in kernels],
                              "error": rec.get("error", "")[:200]}), flush=True)
            continue
        file, func = sol["spec"]["entry_point"].split("::")
        cache, dump = work / "triton_cache", work / "cute_dump"
        env = {**{k: v for k, v in os.environ.items() if k != "TRITON_OVERRIDE_ARCH"}, "TRITON_CACHE_DIR": str(cache),
               # CuTe DSL: compile for B200 and keep each cubin (and PTX, for the block size) in a private directory.
               "CUTE_DSL_ARCH": "sm_100a", "CUTE_DSL_KEEP_CUBIN": "1", "CUTE_DSL_KEEP_PTX": "1",
               "CUTE_DSL_DUMP_DIR": str(dump), "CUTE_DSL_NO_CACHE": "1", "CUTE_DSL_CACHE_DIR": str(work / "cute_cache")}
        dump.mkdir()
        p = subprocess.run([sys.executable, "-c", RUNNER, str(work / file), func, sz], env=env, cwd=work,
                           capture_output=True, text=True, timeout=900)
        kernels = []
        for cubin in cache.rglob("*.cubin"):
            meta_path = cubin.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            kernels.extend(inspect_cubin(cubin, meta))
        for cubin in dump.rglob("*.cubin"):
            ptx = cubin.with_suffix(".ptx")
            ntid = re.search(r"\.(?:reqntid|maxntid)\s+(\d+)", ptx.read_text()) if ptx.exists() else None
            kernels.extend(inspect_cubin(cubin, {"num_warps": int(ntid[1]) // 32} if ntid else {}))
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
