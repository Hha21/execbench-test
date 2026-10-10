"""Turn an inner-LLM reply into candidate solutions, and lint them against the forbidden-behaviour list."""

import json
import re

from archive import parse_card

FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.S | re.M)
NAMED = re.compile(r"^```((?:markdown|yaml|json)\s+[\w-]+|\w+\s+file=\S+)\s*$")


def named_blocks(text):
    """{info: body} for our named fenced blocks (```markdown card, ```yaml ledger, ```cuda file=kernel.cu, ...).

    Unlike FENCE, a named block may contain other fenced blocks (a problem card with ```python examples): it runs from
    its opening line to the last bare ``` line before the next named opening line (or the end of the text)."""
    lines = text.splitlines()
    opens = [i for i, l in enumerate(lines) if NAMED.match(l)]
    out = {}
    for n, i in enumerate(opens):
        end = opens[n + 1] if n + 1 < len(opens) else len(lines)
        close = next((j for j in range(end - 1, i, -1) if lines[j].strip() == "```"), None)
        if close is not None:
            out[NAMED.match(lines[i]).group(1).strip()] = "\n".join(lines[i + 1:close]) + "\n"
    return out

# generation_protocol.md §5: anything matching goes to human review instead of the GPU.
LINT = [
    (r"torch\.cuda\.Stream|torch\.cuda\.stream\(", "extra CUDA stream"),
    (r"torch\.cuda\.graph|CUDAGraph", "CUDA graph"),
    (r"\bthreading\b|\bmultiprocessing\b|concurrent\.futures|torch\.jit\.fork", "threads or forks"),
    (r"os\.environ\[[^]]+\]\s*=|os\.environ\.(update|setdefault)|os\.putenv", "environment mutation"),
    (r"load_inline|cpp_extension\.load\b|cuModuleLoadData|\bbase64\b|ctypes\.(CDLL|cdll)", "runtime binary loading"),
    (r"\.half\(\)|\.bfloat16\(\)|float16|bfloat16|allow_tf32|round_f32_to_tf32\s*=\s*True", "reduced precision"),
    # Project rules (ledger constraints), checked in code as well as stated in prompts:
    (r"discard\.global|discard\.L2|createpolicy\.[\w.]*evict_unchanged|\binvalidate\b", "cache-line discard/invalidate"),
    (r"applypriority|accessPolicyWindow|cudaStreamAttrValue|cudaLimitPersistingL2CacheSize", "L2 priority/persistence change"),
    (r"cudaCtxResetPersistingL2Cache|cudaDeviceSetLimit", "device/context state change"),
    (r"griddepcontrol|ProgrammaticStreamSerialization|ProgrammaticEvent|launch_pdl|cudaGridDependencySynchronize",
     "programmatic dependent launch (overlap with the harness's kernels)"),
    (r"cudaStreamCreate|cudaStreamCreateWithFlags|cudaStreamCreateWithPriority|getStreamFromPool|cudaStreamPerThread",
     "extra CUDA stream"),
    (r"\bnanosleep\b|__nanosleep|usleep|this_thread::sleep|time\.sleep|std::chrono::[\w:]*sleep",
     "deliberate delay (launch timing)"),
    (r"while\s*\([^)]*(clock64|clock\(\)|globaltimer)", "spin-wait on a clock (deliberate delay)"),
    (r"(b?float)\s*##\s*(16|8)|half\s*##", "token-pasted precision type name (looks like lint evasion)"),
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
    import problem
    sol = {"name": cid, "definition": problem.current().name, "author": "solx-loop",
           "description": str(card.get("hypothesis") or "")[:500], "spec": spec,
           "sources": [{"path": p, "content": body} for p, body in c["files"]]}
    return cid, sol, c["card"], problems


def low_precision_problem():
    """True if the current problem's own inputs/outputs are fp16/bf16 (then those types are not a precision cut)."""
    try:
        import problem
        d = problem.current().definition
        return any(t.get("dtype") in ("float16", "bfloat16")
                   for sect in ("inputs", "outputs") for t in d[sect].values())
    except Exception:
        return False


def strip_comments(code, path):
    """Code without comments, so a rule named in a comment ("no applypriority here") is not a hit."""
    if path.endswith(".py"):
        return re.sub(r"(?m)#(?!include|define|pragma|if|endif|else).*$", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return re.sub(r"//.*$", "", code, flags=re.M)


def lint(sol):
    hits = []
    low = low_precision_problem()
    for s in sol["sources"]:
        code = strip_comments(s["content"], s["path"])
        # Only look inside run() for torch ops; elsewhere (e.g. module-level setup) they are not timed.
        run_body = code.split("def run(", 1)[1] if "def run(" in code else ""
        for pat, why in LINT:
            if why.startswith("data_ptr()") and not s["path"].endswith(".py"):
                continue  # data_ptr<T>() is how C++/CUDA extensions pass tensors to kernels
            if why == "reduced precision" and low:
                pat = r"allow_tf32|round_f32_to_tf32\s*=\s*True"   # fp16/bf16 is this problem's own dtype
            target = run_body if why == "torch op inside run()" else code
            if re.search(pat, target):
                hits.append(f"{s['path']}: {why} ({pat})")
    return hits
