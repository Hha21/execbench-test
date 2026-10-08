"""Compile every generation-2 candidate for sm_100a (B200) and sm_90a (H200) without a GPU.

Records registers, shared memory, spills, the load/store instruction mix and how many programs fit per SM on B200.
  python loop/gen2/static_sm100.py   (needs triton; run on CSF3)  -> loop/gen2/results/static.jsonl
"""

import collections
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

HERE = Path(__file__).resolve().parent
TOOLS = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
B200 = dict(sms=148, threads=2048, regs=65536, smem=233472)
P6 = ["q_ptr", "k_ptr", "wq_ptr", "wk_ptr", "qo_ptr", "ko_ptr"]
DESC = ["qd", "kd", "qod", "kod"]


def load(sol_path):
    src = json.loads(Path(sol_path).read_text())["sources"][0]["content"]
    tmp = Path(tempfile.mkdtemp()) / "kernel.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(Path(sol_path).stem.replace("-", "_"), tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def signature(mod, family, meta):
    D, H = mod.D, mod.H
    if family == "oneshot":
        return (mod._qk_oneshot, {**{p: "*fp32" for p in P6}, "n_rows": "i32", "eps": "fp32"},
                dict(ROWS=meta["ROWS"], D=D, H=H, EVICT=meta["EVICT"]), P6 + ["n_rows"])
    if family == "wstat":
        return (mod._qk_wstat, {**{p: "*fp32" for p in P6}, "n_tok": "i32", "eps": "fp32", "n_prog": "i32"},
                dict(HB=meta["HB"], D=D, H=H, EVICT=meta["EVICT"], STAGES=meta["STAGES"]), P6)
    rows = meta["ROWS"] if family == "tma" else meta["HB"]
    td = f"tensordesc<fp32[{rows},128]>"
    common = {**{d: td for d in DESC}, "wq_ptr": "*fp32", "wk_ptr": "*fp32"}
    if family == "tma":
        return (mod._qk_tma, {**common, "n_rows": "i32", "eps": "fp32", "n_prog": "i32"},
                dict(ROWS=rows, D=D, H=H, STAGES=meta["STAGES"]), ["wq_ptr", "wk_ptr", "n_rows"])
    return (mod._qk_tma_wstat, {**common, "n_tok": "i32", "eps": "fp32", "n_prog": "i32"},
            dict(HB=rows, D=D, H=H, STAGES=meta["STAGES"]), ["wq_ptr", "wk_ptr"])


def compile_one(fn, sig, constexprs, div16, cc, num_warps):
    names = fn.arg_names
    full = {n: sig.get(n, "constexpr") for n in names}
    src = ASTSource(fn=fn, signature=full, constexprs={(names.index(k),): v for k, v in constexprs.items()},
                    attrs={(names.index(n),): [["tt.divisibility", 16]] for n in div16})
    return triton.compile(src, target=GPUTarget("cuda", cc, 32), options={"num_warps": num_warps})


def sass_info(cubin):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        res = subprocess.run([TOOLS / "cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
        sass = subprocess.run([TOOLS / "cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
    ops = collections.Counter(mm[1].split(".")[0] for mm in
                              re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", sass))
    keep = {k: v for k, v in ops.items() if k.startswith(("LDG", "STG", "UTMA", "LDGSTS", "UBLKCP", "LDS", "STS"))}
    return int(m[1]), int(m[4]), keep


def resident(regs, smem, num_warps):
    threads = 32 * num_warps
    by_regs = (B200["regs"] // (-(-regs // 8) * 8 * 32)) // num_warps
    by_smem = B200["smem"] // (smem + 1024) if smem else 32
    return min(32, B200["threads"] // threads, by_regs, by_smem)


def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    recs = []
    for sol in sorted((HERE / "candidates").glob("*.json")):
        mod = load(sol)
        for limit, family, meta in mod.BANDS:
            fn, sig, ce, div16 = signature(mod, family, meta)
            for arch, cc in (("sm_100a", 100), ("sm_90a", 90)):
                rec = dict(id=sol.stem, family=family, meta=meta, arch=arch)
                try:
                    ck = compile_one(fn, sig, ce, div16, cc, meta["num_warps"])
                    regs, local, ops = sass_info(ck.asm["cubin"])
                    smem = ck.metadata.shared
                    rec.update(regs=regs, smem=smem, local=local, ops=ops,
                               resident_per_sm=resident(regs, smem, meta["num_warps"]))
                except Exception as e:  # keep going; the failure is the result
                    rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                recs.append(rec)
                print(json.dumps({k: rec.get(k) for k in ("id", "arch", "regs", "smem", "local", "resident_per_sm",
                                                          "ops", "error")}), flush=True)
    (out / "static.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"wrote {len(recs)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
