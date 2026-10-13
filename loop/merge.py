"""Per-workload merge: one solution that runs, for each workload, the kernel that was fastest there.

Each source kernel ("leaf") keeps its files unchanged. Every compiled file of a leaf becomes a top-level translation
unit `<label>__<file>` that wraps the original text in `namespace <label>` (leaves from a common ancestor share struct,
template and inline-function names, which would otherwise silently COMDAT-merge across translation units) and renames
the leaf's extern "C" symbols to `<label>_<name>` with #defines (C linkage ignores namespaces). The leaf's own
PYBIND11_MODULE becomes an unused static function. A generated binding.cpp reads the variable axes from the input
tensors and calls the routed leaf. Its first lines carry a manifest, so a merged kernel can be merged again: its
leaves are recovered verbatim and re-routed (no nesting).

  .venv/bin/python loop/merge.py --problem L2/035_convnextv2_block_with_grn --name my-merge --out merged.json \\
      --leaf a=path/a.json --leaf d=path/d.json --route d=2,768,7,7 --route d=8,384,14,14 --default a
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import problem

MARK = "// MERGE-MANIFEST "
COMPILED = (".cu", ".cpp", ".cc", ".cxx", ".c")


def label_of(leaf_id):
    return "lf_" + re.sub(r"\W", "_", leaf_id)


def _strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)


def extern_c_names(text):
    """Names given C linkage in a source file (single declarations and extern "C" { ... } blocks)."""
    t = _strip_comments(text)
    names = set()
    for m in re.finditer(r'extern\s+"C"\s*\{', t):        # block form: the identifier before each top-level '('
        i, depth, decl = m.end(), 1, ""
        while i < len(t) and depth:
            c = t[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 1 and c not in "{}":
                decl += c
            if depth == 1 and c in ";}":
                hit = re.search(r"\b([A-Za-z_]\w*)\s*(?:\(|\[|=|;|$)", decl.split("(")[0] + "(")
                if hit:
                    names.add(hit.group(1))
                decl = ""
            i += 1
    for m in re.finditer(r'extern\s+"C"\s+(?!\{)([^;{(=\[]*?)\b([A-Za-z_]\w*)\s*[(;=\[]', t):
        names.add(m.group(2))
    return names


def entry(sol):
    file, sym = sol["spec"]["entry_point"].split("::")
    return file, sym


def run_params(sol):
    """[(type, name)] of the leaf's entry function."""
    file, sym = entry(sol)
    text = _strip_comments(next(s["content"] for s in sol["sources"] if s["path"] == file))
    m = re.search(rf"\bvoid\s+{re.escape(sym)}\s*\(([^)]*)\)\s*\{{", text)
    if not m:
        raise SystemExit(f"{sol['name']}: no `void {sym}(...) {{` definition in {file}")
    out = []
    for p in m.group(1).split(","):
        p = re.sub(r"\s+", " ", p.split("=")[0]).strip()
        mm = re.fullmatch(r"(.*?)\b([A-Za-z_]\w*)", p)
        out.append((mm.group(1).strip(), mm.group(2)))
    return out


def preamble(text, own_files):
    """Lines to emit before the namespace: the file's leading preprocessor block (when its #if's balance) plus every
    #include that is not one of the leaf's own files. The originals repeat them inside the namespace, where the
    include guards make them no-ops."""
    lead, depth = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("//"):
            lead.append(line)
            continue
        if not s.startswith("#"):
            break
        depth += s.startswith("#if") - s.startswith("#endif")
        lead.append(line)
    if depth != 0:
        lead = []
    incs = [l for l in text.split("\n") if re.match(r"\s*#\s*include\s*[<\"]([^>\"]+)", l)
            and re.match(r"\s*#\s*include\s*[<\"]([^>\"]+)", l).group(1) not in own_files and l not in lead]
    if any(re.match(r"\s*#\s*include\s*\"([^\"]+)\"", l) and re.match(r"\s*#\s*include\s*\"([^\"]+)\"", l).group(1)
           in own_files for l in text.split("\n")):
        raise SystemExit("leaf includes one of its own files; merge.py handles self-contained translation units only")
    return "\n".join(lead + incs)


def wrap_file(label, leaf_id, path, text, sol, cnames, params):
    own = {s["path"] for s in sol["sources"]}
    efile, sym = entry(sol)
    out = [f"// ===== merge leaf {label} ({leaf_id}): {path} =====", preamble(text, own), ""]
    out += [f"#define {n} {label}_{n}" for n in sorted(cnames)]
    if path == efile:
        out += ["#undef PYBIND11_MODULE",
                "#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void merge_unused_module_("
                "::pybind11::module_& variable)"]
    out += [f"namespace {label} {{", f"// ----- BEGIN {path} -----", text.rstrip("\n"), f"// ----- END {path} -----"]
    if path == efile:
        out.append(f"void merge_entry_({', '.join(f'{t} {n}' for t, n in params)}) "
                   f"{{ {sym}({', '.join(n for _, n in params)}); }}")
    out += [f"}}  // namespace {label}", ""]
    return "\n".join(out)


def unwrap(merged):
    """{leaf id: solution} and the route table of a merged solution; None if it is not a merge.py merge."""
    efile, _ = entry(merged)
    head = next(s["content"] for s in merged["sources"] if s["path"] == efile).split("\n", 3)
    line = next((l for l in head if l.startswith(MARK)), None)
    if not line:
        return None
    man = json.loads(line[len(MARK):])
    files = {s["path"]: s["content"] for s in merged["sources"]}
    leaves = {}
    for lid, info in man["leaves"].items():
        srcs = []
        for path in info["files"]:
            text = files[f"{info['label']}__{path}"]
            a, b = f"// ----- BEGIN {path} -----\n", f"\n// ----- END {path} -----"
            srcs.append(dict(path=path, content=text[text.index(a) + len(a):text.rindex(b)] + "\n"))
        leaves[lid] = dict(merged, name=lid, spec=info["spec"], sources=srcs)
    return dict(leaves=leaves, routes=man["routes"], default=man["default"])


def flatten(sol, sol_id):
    """(leaves {id: sol}, route function key -> leaf id) for any solution, merged or not."""
    m = unwrap(sol)
    if m is None:
        return {sol_id: sol}, (lambda k: sol_id)
    return m["leaves"], (lambda k: m["routes"].get(k, m["default"]))


def axis_reads(params):
    """C++ expressions for each variable axis, read from the input tensors (definition input order = entry
    parameter order; outputs follow)."""
    d = problem.current().definition
    names = list(d["inputs"])
    if len(params) != len(d["inputs"]) + len(d["outputs"]):
        raise SystemExit(f"entry has {len(params)} parameters; the definition has {len(d['inputs'])} inputs and "
                         f"{len(d['outputs'])} outputs")
    reads = []
    for ax in problem.current().var_axes:
        hit = next(((i, dim) for i, n in enumerate(names) for dim, s in enumerate(d["inputs"][n].get("shape") or [])
                    if s == ax), None)
        if hit is None:
            raise SystemExit(f"axis {ax} is not a plain dimension of any input tensor")
        reads.append((ax, f"{params[hit[0]][1]}.size({hit[1]})"))
    return reads


def build(leaves, routes, default, name, description=""):
    """leaves {leaf id: solution}; routes {workload key: leaf id}; default leaf id for every other workload."""
    used = [lid for lid in leaves if lid == default or lid in routes.values()]
    if default not in leaves or any(v not in leaves for v in routes.values()):
        raise SystemExit("a route or the default names a leaf that was not given")
    first = leaves[used[0]]
    params = [("torch::Tensor" if "Tensor" in t else t, n) for t, n in run_params(first)]
    specs = [leaves[l]["spec"] for l in used]
    for l in used:
        if len(run_params(leaves[l])) != len(params):
            raise SystemExit(f"{l}: entry signature has a different parameter count")
        if leaves[l]["spec"].get("destination_passing_style") != first["spec"].get("destination_passing_style"):
            raise SystemExit(f"{l}: destination_passing_style differs")
    union = lambda key: list(dict.fromkeys(f for s in specs for f in ((s.get("compile_options") or {}).get(key) or [])))
    cflag_sets = {tuple((s.get("compile_options") or {}).get("cuda_cflags") or []) for s in specs}
    if len(cflag_sets) > 1:
        print(f"warning: leaves use different cuda_cflags; the merge compiles every file with {union('cuda_cflags')}")
    spec = dict(first["spec"])
    spec["entry_point"] = "binding.cpp::run"
    spec["languages"] = list(dict.fromkeys(x for s in specs for x in s["languages"]))
    spec["target_hardware"] = [h for h in first["spec"]["target_hardware"] if all(h in s["target_hardware"] for s in specs)]
    spec["dependencies"] = list(dict.fromkeys(x for s in specs for x in s.get("dependencies") or []))
    spec["compile_options"] = {k: union(k) for k in ("cflags", "cuda_cflags", "ld_flags") if union(k)}
    sources, manifest = [], {}
    for lid in used:
        sol, label = leaves[lid], label_of(lid)
        compiled = [s for s in sol["sources"] if s["path"].endswith(COMPILED)]
        if any("/" in s["path"] for s in sol["sources"]) or len(compiled) != len(sol["sources"]):
            raise SystemExit(f"{lid}: only flat leaves of compiled files are supported")
        cnames = set().union(*(extern_c_names(s["content"]) for s in compiled))
        lp = run_params(sol)
        for s in compiled:
            text = wrap_file(label, lid, s["path"], s["content"], sol, cnames, lp if s["path"] == entry(sol)[0] else [])
            sources.append(dict(path=f"{label}__{s['path']}", content=text))
        manifest[lid] = dict(label=label, files=[s["path"] for s in sol["sources"]], spec=sol["spec"])
    keys = problem.current().keys()
    reads = axis_reads(params)
    man = dict(leaves=manifest, routes={k: v for k, v in routes.items() if v != default}, default=default)
    b = [MARK + json.dumps(man, separators=(",", ":")),
         f"// {name}: per-workload dispatch over {len(used)} unchanged kernels (loop/merge.py).",
         "//   " + "; ".join(f"{lid}: {' '.join(k for k in keys if routes.get(k, default) == lid) or '-'}"
                            + (" (+ any other shape)" if lid == default else "") for lid in used),
         "#include <torch/extension.h>", "#include <cstdint>", ""]
    sig = ", ".join(f"{t} {n}" for t, n in params)
    b += [f"namespace {label_of(l)} {{ void merge_entry_({sig}); }}" for l in used] + [""]
    b += [f"void run({sig}) {{"] + [f"  const int64_t {ax} = {expr};" for ax, expr in reads]
    args = ", ".join(n for _, n in params)
    var = [ax for ax, _ in reads]
    for l in used:
        if l == default:
            continue
        conds = [" && ".join(f"{ax} == {v}" for ax, v in zip(var, k.split(","))) for k in keys if routes.get(k) == l]
        b += [f"  if ({' || '.join(f'({c})' for c in conds)}) {{", f"    {label_of(l)}::merge_entry_({args});",
              "    return;", "  }"]
    b += [f"  {label_of(default)}::merge_entry_({args});", "}", "",
          f'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{ m.def("run", &run, "{name} (per-workload merge)"); }}', ""]
    return dict(first, name=name, description=description or f"Per-workload merge of {', '.join(used)}.", spec=spec,
                sources=[dict(path="binding.cpp", content="\n".join(b))] + sources)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leaf", action="append", default=[], help="id=solution.json (a merged solution adds its leaves)")
    ap.add_argument("--route", action="append", default=[], help="leaf_id=workload key, e.g. d=2,768,7,7")
    ap.add_argument("--default", required=True, help="leaf for every workload without a route")
    a = ap.parse_args()
    os.environ["SOLX_PROBLEM"] = a.problem
    leaves = {}
    for spec in a.leaf:
        lid, path = spec.split("=", 1)
        sub, _ = flatten(json.loads(Path(path).read_text()), lid)
        leaves.update(sub)
    routes = {}
    for r in a.route:
        lid, key = r.split("=", 1)
        routes[key] = lid
    sol = build(leaves, routes, a.default, a.name)
    a.out.write_text(json.dumps(sol, indent=1))
    print("wrote", a.out, [(s["path"], len(s["content"])) for s in sol["sources"]])


if __name__ == "__main__":
    main()
