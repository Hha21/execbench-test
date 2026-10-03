"""Compile every variant for several GPU architectures (no GPU needed) and record what the
compiler produced: registers, spills, shared memory and the machine-instruction mix.

The B200 entry (sm_100a) is the point: these numbers describe the exact code a B200
would run, even though we cannot run it.
"""

import argparse
import collections
import csv
import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

HERE = Path(__file__).resolve().parent
ARCHS = {"sm_80": 80, "sm_89": 89, "sm_90": 90, "sm_100a": 100}
TOOLS = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"

POINTERS = ["q_ptr", "k_ptr", "wq_ptr", "wk_ptr", "qo_ptr", "ko_ptr"]
CONSTEXPRS = ["ROWS", "D", "H", "EVICT", "FUSE", "PERSIST"]


def load_variant(sol_path):
    sol = json.loads(Path(sol_path).read_text())
    src = sol["sources"][0]["content"]
    tmp = Path(tempfile.mkdtemp()) / "kernel.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(f"variant_{Path(sol_path).stem}", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_variant(mod, cc):
    fn = mod._qk_norm_kernel
    names = fn.arg_names
    signature = {n: "*fp32" for n in POINTERS}
    signature.update({"n_rows": "i32", "eps": "fp32", "n_prog": "i32"})
    signature.update({n: "constexpr" for n in CONSTEXPRS})
    values = {"ROWS": mod.ROWS, "D": mod.D, "H": mod.H, "EVICT": mod.EVICT, "FUSE": mod.FUSE, "PERSIST": mod.PERSIST}
    constexprs = {(names.index(n),): v for n, v in values.items()}
    # Match the runtime specialisation: torch allocations are 16-byte aligned and
    # n_rows = batch * seq * 48 is always a multiple of 16.
    attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in POINTERS + ["n_rows"]}
    src = ASTSource(fn=fn, signature={k: signature[k] for k in names}, constexprs=constexprs, attrs=attrs)
    opts = {"num_warps": mod.NUM_WARPS, "num_stages": mod.NUM_STAGES}
    return triton.compile(src, target=GPUTarget("cuda", cc, 32), options=opts)


def sass_features(cubin):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        res = subprocess.run([TOOLS / "cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
        sass = subprocess.run([TOOLS / "cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    feats = {}
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    feats.update(regs=int(m[1]), stack=int(m[2]), shared=int(m[3]), local=int(m[4]))
    ops = collections.Counter()
    for line in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", line)
        if m:
            ops[m[1]] += 1
    full = collections.Counter()
    for op, n in ops.items():
        base = op.split(".")[0]
        full[base] += n
        if base in ("LDG", "STG"):
            width = next((w for w in ("128", "64") if f".{w}" in op), "32")
            full[f"{base}_{width}"] += n
    feats["sass_total"] = sum(ops.values())
    for k in ("LDG", "LDG_128", "LDG_64", "LDG_32", "STG", "STG_128", "STG_64", "STG_32",
              "LDS", "STS", "SHFL", "MUFU", "FFMA", "FMUL", "FADD", "BAR", "BRA", "ISETP"):
        feats[f"n_{k}"] = full.get(k, 0)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=Path, default=HERE / "variants")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "static_features.csv")
    args = ap.parse_args()

    rows = []
    for sol in sorted(args.variants.glob("v*.json")):
        mod = load_variant(sol)
        for arch, cc in ARCHS.items():
            row = {"vid": sol.stem, "arch": arch}
            try:
                ck = compile_variant(mod, cc)
                row.update(sass_features(ck.asm["cubin"]))
                row["triton_shared"] = ck.metadata.shared
                row["ok"] = 1
            except Exception as e:  # record the failure, keep going
                row["ok"] = 0
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
        print(sol.stem, " ".join(f"{r['arch']}:{r.get('regs', 'ERR')}" for r in rows[-len(ARCHS):]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("vid", "arch", "ok"), k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
