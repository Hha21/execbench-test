"""Turn an inner-LLM reply into candidate solutions, and lint them against the forbidden-behaviour list."""

import json
import re

from archive import parse_card

FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.S | re.M)

# generation_protocol.md §5: anything matching goes to human review instead of the GPU.
LINT = [
    (r"torch\.cuda\.Stream|torch\.cuda\.stream\(", "extra CUDA stream"),
    (r"torch\.cuda\.graph|CUDAGraph", "CUDA graph"),
    (r"\bthreading\b|\bmultiprocessing\b|concurrent\.futures|torch\.jit\.fork", "threads or forks"),
    (r"os\.environ\[[^]]+\]\s*=|os\.environ\.(update|setdefault)|os\.putenv", "environment mutation"),
    (r"load_inline|cpp_extension\.load\b|cuModuleLoadData|\bbase64\b|ctypes\.(CDLL|cdll)", "runtime binary loading"),
    (r"\.half\(\)|\.bfloat16\(\)|float16|bfloat16|allow_tf32|round_f32_to_tf32\s*=\s*True", "reduced precision"),
    (r"lru_cache|functools\.cache", "caching keyed on tensors or pointers"),
    (r"data_ptr\(\)", "data_ptr() in Python (possible pointer-keyed cache)"),
    (r"\.copy_\(|\.fill_\(|\.zero_\(|torch\.(zeros|ones|full|cat|stack)\(|\.clone\(\)", "torch op inside run()"),
]


def candidates(text):
    """Split a reply into candidates: a json solution-spec block, file blocks, then a yaml design-card block."""
    out, cur = [], None
    for m in FENCE.finditer(text):
        info, body = m.group(1).strip(), m.group(2)
        if info.startswith("json") and "solution-spec" in info:
            cur = dict(spec=body, files=[], card=None)
            out.append(cur)
        elif "file=" in info and cur is not None:
            cur["files"].append((info.split("file=", 1)[1].strip(), body))
        elif info.startswith("yaml") and "design-card" in info and cur is not None:
            cur["card"] = body
    return [c for c in out if c["files"] and c["card"]]


def build(c, round_id, taken):
    """Candidate dict -> (id, solution JSON, card text, problems)."""
    problems = []
    try:
        spec = json.loads(c["spec"])
    except json.JSONDecodeError as e:
        return None, None, c["card"], [f"solution-spec is not valid JSON: {e}"]
    spec = spec.get("spec", spec)
    card = parse_card(c["card"])
    cid = re.sub(r"[^a-z0-9-]+", "-", str(card.get("id") or f"{round_id}-cand").lower()).strip("-")
    if not cid.startswith(round_id):
        cid = f"{round_id}-{cid}"
    base, i = cid, 2
    while cid in taken:
        cid, i = f"{base}-{i}", i + 1
    entry = spec.get("entry_point", "")
    if entry.split("::")[0] not in [p for p, _ in c["files"]]:
        problems.append(f"entry point {entry!r} is not one of the files")
    spec.setdefault("target_hardware", ["B200", "LOCAL"])
    spec.setdefault("destination_passing_style", True)
    sol = {"name": cid, "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop",
           "description": str(card.get("hypothesis") or "")[:500], "spec": spec,
           "sources": [{"path": p, "content": body} for p, body in c["files"]]}
    return cid, sol, c["card"], problems


def lint(sol):
    hits = []
    for s in sol["sources"]:
        code = s["content"]
        # Only look inside run() for torch ops; elsewhere (e.g. module-level setup) they are not timed.
        run_body = code.split("def run(", 1)[1] if "def run(" in code else ""
        for pat, why in LINT:
            if why.startswith("data_ptr()") and not s["path"].endswith(".py"):
                continue  # data_ptr<T>() is how C++/CUDA extensions pass tensors to kernels
            target = run_body if why == "torch op inside run()" else code
            if re.search(pat, target):
                hits.append(f"{s['path']}: {why} ({pat})")
    return hits
