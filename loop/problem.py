"""A SOL-ExecBench problem: its definition, workloads, per-workload sizes and size bands, and where its loop state lives.

Everything problem-specific in the loop goes through this module, so the same code serves any problem. The current
problem is chosen with `round.py --problem <name>` (or SOLX_PROBLEM in the environment; default: #38) and inherited by
every subprocess (MCP tool servers, session runners).

Layout per problem, problems/<name>/:
    card.md         problem card: semantics, numerics, workloads, bounds, results, lessons (part of the briefing)
    sol.yaml        optional: `flops: <expression in axis names>`, so the portal's Tsol can be modelled as a roofline
    ledger.yaml     hypothesis ledger
    features.yaml   per-path design features of kernels with portal results (emulator)
    archive.json    every kernel: design card, timings per GPU, compile statistics, portal results
    rounds/<r>/     plans, lead and session transcripts, candidates, results, portal files
    b200/           rented-B200 timings of portal kernels (emulator pairs), copy reference, calibration

Workload keys are the variable-axis values joined with commas in definition order ("2,128" for #38's
batch_size=2, seq_len=128). Sizes are the bytes the workload's inputs and outputs occupy (its compulsory traffic), and
the S/M/L bands are size tertiles.
"""

import ast
import json
import math
import operator
import os
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "SOL-ExecBench" / "data" / "benchmark"
PROBLEMS = ROOT / "problems"
DEFAULT = "038_flux_multi_head_rmsnorm_qk"
DTYPE_BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4, "int16": 2, "int8": 1,
               "uint8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1, "float4_e2m1fn_x2": 1}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _eval(expr, env):
    """Safe arithmetic on axis names (definition 'expr' axes such as 'num_heads * head_dim')."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError(f"unsupported expression {expr!r}")
    return ev(ast.parse(str(expr), mode="eval"))


def find(name):
    """'L1/038', '038_flux_...' or 'L1/038_flux_...' -> the problem's data directory (numbers repeat across levels)."""
    name = str(name).strip("/")
    if (DATA / name / "definition.json").exists():
        return DATA / name
    level, _, name = name.rpartition("/")                 # 'L1/038' narrows the search to one level
    hits = [d for d in DATA.glob(f"{level or '*'}/*") if (d / "definition.json").exists()
            and (d.name == name or d.name.startswith(name + "_") or d.name.split("_", 1)[0] == name)]
    if len(hits) != 1:
        raise SystemExit(f"problem {name!r}: {len(hits)} matches in {DATA}" +
                         (f": {', '.join(h.parent.name + '/' + h.name for h in hits[:6])}" if hits else ""))
    return hits[0]


class Problem:
    def __init__(self, name=DEFAULT):
        self.data = find(name)
        self.name = self.data.name
        self.level = self.data.parent.name
        self.definition = json.loads((self.data / "definition.json").read_text())
        self.workloads = [json.loads(l) for l in (self.data / "workload.jsonl").read_text().splitlines() if l.strip()]
        axes = self.definition["axes"]
        self.var_axes = [a for a, v in axes.items() if v["type"] == "var"]
        self.dir = PROBLEMS / self.name
        self.card = self.dir / "card.md"
        self.ledger = self.dir / "ledger.yaml"
        self.features = self.dir / "features.yaml"
        self.archive = self.dir / "archive.json"
        self.rounds = self.dir / "rounds"
        self.b200 = self.dir / "b200"
        sol = self.dir / "sol.yaml"
        self.sol = (__import__("yaml").safe_load(sol.read_text()) or {}) if sol.exists() else {}
        sizes = sorted(self.mbytes(self.key(w["axes"])) for w in self.workloads)   # tertiles over workloads (5/5/6 for #38)
        n = len(sizes)
        self.s_max = sizes[max(0, n // 3 - 1)]
        self.m_max = sizes[max(0, 2 * n // 3 - 1)]

    # --- workloads ---------------------------------------------------------------------------------------------
    def key(self, axes):
        return ",".join(str(axes[a]) for a in self.var_axes)

    def keys(self):
        return list(dict.fromkeys(self.key(w["axes"]) for w in self.workloads))

    def key_from_label(self, label):
        """Portal workload label ('batch_size=2, seq_len=128') -> key."""
        axes = dict(re.findall(r"(\w+)=(\d+)", label))
        return self.key({a: int(axes[a]) for a in self.var_axes})

    def axes(self, key):
        """Resolved values of every axis for a workload key."""
        env = dict(zip(self.var_axes, (int(x) for x in key.split(","))))
        pending = dict(self.definition["axes"])
        for _ in range(len(pending) + 1):
            for a, v in list(pending.items()):
                try:
                    if v["type"] == "const":
                        env[a] = v["value"]
                    elif v["type"] == "expr":
                        env[a] = _eval(v["expression"], env)
                    elif a not in env:
                        continue
                    pending.pop(a)
                except KeyError:
                    pass
        return env

    def tokens(self, key):
        """Product of the variable axes (B*S for #38); designs dispatch on it (`max_tokens` in a card's paths)."""
        return math.prod(int(x) for x in key.split(","))

    @lru_cache(maxsize=None)
    def mbytes(self, key):
        """MB of compulsory traffic: every input and output tensor once."""
        env, total = self.axes(key), 0
        for sect in ("inputs", "outputs"):
            for t in self.definition[sect].values():
                n = math.prod(env[d] if isinstance(d, str) and not d.isdigit() else int(d) for d in (t.get("shape") or []))
                total += n * DTYPE_BYTES.get(t["dtype"], 4)
        return total / 1e6

    @property
    def kind(self):
        """'compute' for problems with a FLOP formula in sol.yaml (GEMM-like), else 'memory' (elementwise/norm)."""
        return "compute" if self.sol.get("flops") else "memory"

    def gflop(self, key):
        """GFLOP of the workload from sol.yaml's `flops` expression; 0 when the problem has none."""
        return _eval(self.sol["flops"], self.axes(key)) / 1e9 if self.sol.get("flops") else 0.0

    def band(self, key):
        mb = self.mbytes(key)
        return "S" if mb <= self.s_max else "M" if mb <= self.m_max else "L"

    def in_band(self, key, band):
        return self.band(key) == band

    def summary(self):
        lines = [f"{self.level}/{self.name}: {len(self.workloads)} workloads, variable axes {self.var_axes}; "
                 f"bands by size: S <= {self.s_max:.1f} MB < M <= {self.m_max:.1f} MB < L"]
        for k in sorted(self.keys(), key=self.mbytes):
            lines.append(f"  {k:>16}  {self.mbytes(k):9.2f} MB  band {self.band(k)}")
        if self.sol.get("note"):                    # problem-specific rulings (e.g. which precision the tolerance allows)
            lines.append(f"note: {self.sol['note']}")
        return "\n".join(lines)


@lru_cache(maxsize=None)
def get(name=None):
    return Problem(name or os.environ.get("SOLX_PROBLEM") or DEFAULT)


def current():
    return get(os.environ.get("SOLX_PROBLEM") or DEFAULT)


if __name__ == "__main__":
    import sys
    print(get(sys.argv[1] if len(sys.argv) > 1 else None).summary())
