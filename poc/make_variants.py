"""Sample knob settings for the rmsnorm_qk kernel and write one solution JSON per variant.

Each JSON is a complete SOL-ExecBench solution: it runs locally with the sol-execbench CLI
and can be uploaded to the portal unchanged.
"""

import argparse
import csv
import itertools
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "kernels" / "rmsnorm_qk.py"
DEFINITION = "flux_multi_head_rmsnorm_qk"

SPACE = {
    "ROWS": [1, 2, 4, 8, 16, 32],
    "NUM_WARPS": [1, 2, 4, 8],
    "NUM_STAGES": [1, 2, 3, 4],
    "FUSE": [False, True],
    "PERSIST": [False, True],
    "PROGS_PER_SM": [1, 2, 4, 8],
    "EVICT": [False, True],
}


def canonical(knobs):
    """Pin knobs that have no effect so equivalent variants collapse to one."""
    k = dict(knobs)
    if not k["PERSIST"]:
        k["PROGS_PER_SM"] = 0
        k["NUM_STAGES"] = 1
    return k


def render(knobs):
    src = TEMPLATE.read_text()
    for name, value in knobs.items():
        src, n = re.subn(rf"^{name} = .*?(  #.*)?$", lambda m: f"{name} = {value!r}{m.group(1) or ''}", src, count=1, flags=re.M)
        assert n == 1, name
    return src


def solution_json(vid, knobs):
    return {
        "name": f"rmsnorm_qk_triton_{vid}",
        "definition": DEFINITION,
        "author": "solx-poc",
        "description": "Triton per-head RMSNorm for Q and K; " + ", ".join(f"{k}={v}" for k, v in knobs.items()),
        "spec": {
            "languages": ["triton"],
            "target_hardware": ["B200", "LOCAL"],
            "entry_point": "kernel.py::run",
            "dependencies": ["torch", "triton"],
            "destination_passing_style": True,
        },
        "sources": [{"path": "kernel.py", "content": render(knobs)}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "variants")
    args = ap.parse_args()

    grid = []
    seen = set()
    for values in itertools.product(*SPACE.values()):
        k = canonical(dict(zip(SPACE, values)))
        key = tuple(k.values())
        if key not in seen:
            seen.add(key)
            grid.append(k)
    rng = random.Random(args.seed)
    picked = rng.sample(grid, args.n)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "variants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vid", *SPACE])
        w.writeheader()
        for i, k in enumerate(picked):
            vid = f"v{i:03d}"
            (args.out / f"{vid}.json").write_text(json.dumps(solution_json(vid, k), indent=1))
            w.writerow({"vid": vid, **k})
    print(f"{len(grid)} distinct variants in the space; wrote {len(picked)} to {args.out}")


if __name__ == "__main__":
    main()
