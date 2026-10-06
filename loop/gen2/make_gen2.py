"""Render generation-2 candidates in the generation-protocol format: a solution JSON and a design card each.

  python3 loop/gen2/make_gen2.py          -> loop/gen2/candidates/<id>.json and <id>.card.yaml

Round 1 tests each family on its own across all sizes (spec: all), so the per-size portal and CSF3 results show
where each design wins; dispatchers that combine the winners come in round 2.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "kernel_template.py"
OUT = HERE / "candidates"
ALL = 1 << 62

E = dict(EVICT=True)
CANDIDATES = [
    # id, bands, niche, hypothesis, expected effect vs v039 per band (pct > 0 = slower), risks
    ("g2-os-r16w8", [(ALL, "oneshot", dict(ROWS=16, num_warps=8, **E))],
     dict(mem="ldg128", grid="oneshot", tile="rows16", spec="S"),
     "v039's tile without the persistent loop: small inputs already run one tile per program, so dropping the loop, "
     "the idle extra programs and the per-program setup should cut fixed cost.",
     dict(S=(-10, "low"), M=(+3, "low"), L=(+5, "low")), ["more CTAs to schedule on large inputs"]),
    ("g2-os-r16w4", [(ALL, "oneshot", dict(ROWS=16, num_warps=4, **E))],
     dict(mem="ldg128", grid="oneshot", tile="rows16", spec="S"),
     "Half the threads per tile: 4 x LDG.128 per thread issued back to back, fewer threads to launch.",
     dict(S=(-8, "low"), M=(+3, "low"), L=(+5, "low")), ["fewer resident warps to hide latency"]),
    ("g2-os-r8w4", [(ALL, "oneshot", dict(ROWS=8, num_warps=4, **E))],
     dict(mem="ldg128", grid="oneshot", tile="rows8", spec="S"),
     "Playbook 3a recipe: 8-row tiles give 1,536 CTAs on the smallest input, about 10 per SM in one wave.",
     dict(S=(-8, "low"), M=(+5, "low"), L=(+8, "low")), ["CTA launch rate on B200 is unknown; v040 showed tiny CTAs hurt"]),
    ("g2-os-r8w2", [(ALL, "oneshot", dict(ROWS=8, num_warps=2, **E))],
     dict(mem="ldg128", grid="oneshot", tile="rows8", spec="S"),
     "Smaller CTAs (64 threads) launch faster and balance better on small inputs.",
     dict(S=(-6, "low"), M=(+5, "low"), L=(+10, "low")), ["as above"]),
    ("g2-ws-hb16w8", [(ALL, "wstat", dict(HB=16, num_warps=8, STAGES=1, PROGS_PER_SM=8, **E))],
     dict(mem="ldg128", grid="persistent-wstat", tile="rows16", spec="L"),
     "v039 with weights loaded once per program: halves load instructions per tile and removes the weight "
     "address arithmetic, which matters at B200's 36 B/clk/SM.",
     dict(S=(0, "low"), M=(-4, "low"), L=(-6, "medium")), ["programs are tied to a head group; tail imbalance"]),
    ("g2-ws-hb16w8s3", [(ALL, "wstat", dict(HB=16, num_warps=8, STAGES=3, PROGS_PER_SM=8, **E))],
     dict(mem="cpasync", grid="persistent-wstat", tile="rows16", spec="L"),
     "Weight-stationary plus a 3-stage cp.async pipeline keeps the next tiles in flight while one is reduced.",
     dict(S=(+3, "low"), M=(-6, "low"), L=(-10, "low")), ["shared-memory ring lowers resident programs"]),
    ("g2-tma-r8s3", [(ALL, "tma", dict(ROWS=8, num_warps=4, STAGES=3, PROGS_PER_SM=4))],
     dict(mem="tma-tensor", grid="persistent", tile="rows8", spec="L"),
     "Playbook 3d: TMA moves tiles with few instructions and a 3-deep ring keeps bytes in flight.",
     dict(S=(+10, "low"), M=(-3, "low"), L=(-8, "low")), ["weights go through cp.async and grow shared memory",
                                                          "host descriptor path not yet run on a GPU"]),
    ("g2-tma-r16s4", [(ALL, "tma", dict(ROWS=16, num_warps=4, STAGES=4, PROGS_PER_SM=4))],
     dict(mem="tma-tensor", grid="persistent", tile="rows16", spec="L"),
     "Bigger TMA tiles and a deeper ring: more bytes in flight per CTA; about 2 CTAs per SM by shared memory.",
     dict(S=(+15, "low"), M=(-3, "low"), L=(-8, "low")), ["107 KB shared memory caps residency at 2 CTAs/SM"]),
    ("g2-tmaws-hb16s3", [(ALL, "tma_wstat", dict(HB=16, num_warps=4, STAGES=3, PROGS_PER_SM=4))],
     dict(mem="tma-tensor", grid="persistent-wstat", tile="rows16", spec="L"),
     "TMA ring with weights held in registers once per program, so the ring carries only input tiles.",
     dict(S=(+10, "low"), M=(-5, "low"), L=(-12, "low")), ["per-program head group may leave SMs uneven on small inputs"]),
    ("g2-tmaws-hb8s4", [(ALL, "tma_wstat", dict(HB=8, num_warps=4, STAGES=4, PROGS_PER_SM=8))],
     dict(mem="tma-tensor", grid="persistent-wstat", tile="rows8", spec="L"),
     "Smaller weight-stationary TMA tiles with a 4-deep ring and more programs per SM.",
     dict(S=(+8, "low"), M=(-5, "low"), L=(-10, "low")), ["more programs, more descriptor traffic"]),
]


def render(bands):
    src = TEMPLATE.read_text()
    new = "BANDS = " + repr(bands).replace(str(ALL), "1 << 62")
    src, n = re.subn(r"^BANDS = .*$", new, src, count=1, flags=re.M)
    assert n == 1
    return src


def yaml_lines(obj, indent=0):
    pad = "  " * indent
    out = []
    for k, v in obj.items():
        if isinstance(v, dict) and v and not all(isinstance(x, (int, float, str, bool)) for x in v.values()):
            out.append(f"{pad}{k}:")
            out += yaml_lines(v, indent + 1)
        elif isinstance(v, dict):
            out.append(f"{pad}{k}: {{{', '.join(f'{a}: {json.dumps(b)}' for a, b in v.items())}}}")
        elif isinstance(v, list):
            out.append(f"{pad}{k}:")
            out += [f"{pad}  - {json.dumps(x) if not isinstance(x, dict) else '{' + ', '.join(f'{a}: {json.dumps(b)}' for a, b in x.items()) + '}'}" for x in v]
        else:
            out.append(f"{pad}{k}: {json.dumps(v)}")
    return out


def card(cid, bands, niche, hypothesis, effect, risks):
    fam = {f for _, f, _ in bands}
    tma = any(f.startswith("tma") for f in fam)
    meta = bands[0][2]
    return {
        "id": cid, "parents": ["v039"], "operation": "structural_mutation" if cid.startswith(("g2-ws", "g2-tma")) else "specialisation",
        "language": "triton",
        "niche": {"mem": niche["mem"], "st": "bulk" if tma else "direct", "grid": niche["grid"], "launch": "fused",
                  "tile": niche["tile"], "red": "warp", "cache": "stream" if meta.get("EVICT") else "default",
                  "spec": niche["spec"]},
        "hypothesis": hypothesis,
        "expected_effect": {b: {"pct": p, "confidence": c} for b, (p, c) in effect.items()},
        "resources_sm100a": "filled by static_sm100.py",
        "knobs": meta,
        "dispatch": [{"max_tokens": "all" if lim == ALL else lim, "kernel": f, **m} for lim, f, m in bands],
        "runs_on": {"H200": {"runs": True, "representative": True},
                    "A100": {"runs": True, "representative": not tma,
                             **({"note": "TMA compiles to a non-TMA fallback on sm_80; correctness only"} if tma else {})},
                    "L40S": {"runs": True, "representative": False}},
        "risks": risks,
        "measure_first": ["H200"] if tma else ["A100", "H200"],
        "compliance": {k: True for k in ("single_stream", "no_cross_call_state", "fp32_math", "no_threads_or_fork",
                                         "no_precompiled_binaries")},
    }


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for cid, bands, niche, hyp, eff, risks in CANDIDATES:
        c = card(cid, bands, niche, hyp, eff, risks)
        sol = {"name": cid, "definition": "038_flux_multi_head_rmsnorm_qk", "author": "solx-loop", "description": hyp,
               "spec": {"languages": ["triton"], "target_hardware": ["B200", "LOCAL"], "entry_point": "kernel.py::run",
                        "dependencies": ["torch", "triton"], "destination_passing_style": True},
               "sources": [{"path": "kernel.py", "content": render(bands)}]}
        (OUT / f"{cid}.json").write_text(json.dumps(sol, indent=1))
        (OUT / f"{cid}.card.yaml").write_text("\n".join(yaml_lines(c)) + "\n")
        rows.append(cid)
    print(f"wrote {len(rows)} candidates to {OUT}: {rows}")


if __name__ == "__main__":
    main()
