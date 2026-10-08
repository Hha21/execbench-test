"""Assemble inner-LLM prompts in the generation-protocol layout.

The static part (core brief, problem card, reference docs, protocol) is identical for every call, so it is sent as
a cached system prompt. The dynamic part (task, archive, parents, feedback) changes per call.
"""

import csv
import json
import re
from pathlib import Path

from archive import ROOT, band_geomeans, fmt, table

CTX = ROOT / "loop" / "context"
STATIC_DOCS = ["core_brief.md", "problem_038.md", "harness_scoring.md", "b200_arch.md", "b200_sota.md",
               "playbook_membound.md", "generation_protocol.md"]
CONTRACT = """=== OUTPUT CONTRACT ===
Optional "### Rationale" (<=150 words). Then, per candidate:
1) ```json solution-spec``` (the solution JSON without "sources")
2) one ```<lang> file=<path>``` block per source file (complete files)
3) ```yaml design-card``` following the schema in the generation protocol (block style, one key per line).
No other fenced code blocks. Never use a forbidden behaviour; if the task needs one, return no candidate and explain."""


def static_prompt():
    parts = ["You are the kernel-design step of an automated optimisation loop for NVIDIA's SOL-ExecBench B200 "
             "leaderboard. The documents below are your briefing; follow the output contract exactly."]
    for name in STATIC_DOCS:
        parts.append(f"=== {name} ===\n{(CTX / name).read_text()}")
    parts.append(CONTRACT)
    return "\n\n".join(parts)


def b200_workload_table(vid):
    """Per-workload B200 results for a parent, with the hidden baseline, from the ingested portal pages."""
    path = ROOT / "poc" / "results" / "b200_portal_workloads.csv"
    if not path.exists():
        return ""
    rows = [r for r in csv.DictReader(open(path)) if r["vid"] == vid]
    if not rows:
        return ""
    out = ["| workload | latency µs | baseline µs | score |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: [int(x) for x in re.findall(r"=(\d+)", r["workload"])][0] *
                    [int(x) for x in re.findall(r"=(\d+)", r["workload"])][1]):
        out.append(f"| {r['workload']} | {float(r['latency_ms']) * 1e3:.1f} | {float(r['baseline_ms']) * 1e3:.1f} | "
                   f"{float(r['sol_score']):.3f} |")
    return "\n".join(out)


def parent_block(rec):
    sol = json.loads((ROOT / rec["solution"]).read_text())
    src = "\n\n".join(f"--- file {s['path']} ---\n{s['content']}" for s in sol["sources"])
    meas = "; ".join(f"{g} {fmt(band_geomeans(t))}" for g, t in rec.get("timings", {}).items() if t)
    b200 = rec.get("b200") or {}
    static = json.dumps(rec.get("static_sm100") or [])
    return (f"--- parent {rec['id']} ---\n"
            f"design card:\n{rec.get('card', '')}\n"
            f"measurements (geomean µs per band S/M/L): {meas or 'none yet'}\n"
            f"B200 portal: score {b200.get('score', 'not submitted')}, geomean {b200.get('geomean_us', '-')} µs\n"
            f"{b200_workload_table(rec['id'])}\n"
            f"sm_100a static features: {static}\n"
            f"source:\n{src}\n")


def lab_notebook(arc, limit=40):
    """What earlier rounds learned: portal outcomes against the bench, and findings recorded in design cards."""
    import math
    import planner
    from archive import BANDS, parse_card, tokens
    lines = []
    best = planner.best_kernel(arc)
    bp, br = best["b200"]["timings"], (best.get("timings") or {}).get("B200r", {})

    def rel(t, ref, lo, hi):
        ks = [k for k in t if lo < tokens(k) <= hi and k in ref and t[k] and ref[k]]
        return 100 * (math.exp(sum(math.log(t[k] / ref[k]) for k in ks) / len(ks)) - 1) if ks else None
    out = []
    for r in arc.values():
        p_t, r_t = (r.get("b200") or {}).get("timings"), (r.get("timings") or {}).get("B200r")
        if p_t and r_t and r["id"] != best["id"] and str(r.get("round", "")).startswith(("r", "d")):
            cells = []
            for b, lo, hi in BANDS:
                a, c = rel(p_t, bp, lo, hi), rel(r_t, br, lo, hi)
                if a is not None and c is not None:
                    cells.append(f"{b} bench {c:+.1f}% / portal {a:+.1f}%")
            out.append(f"- {r['id']} (portal {r['b200']['score']:.4f}) vs {best['id']}: " + "; ".join(cells))
    if out:
        lines.append("Portal outcomes vs the bench (change vs the current best, same kernel pair):")
        lines.extend(sorted(out))
    notes = []
    for r in arc.values():
        card = parse_card(r.get("card", "")) if r.get("card") else {}
        for f in card.get("findings") or []:
            notes.append((str(r.get("round", "")), f"- [{r['id']}] {str(f).strip()}"))
    if notes:
        lines.append("Findings recorded by earlier design sessions (most recent first):")
        lines.extend(n for _, n in sorted(notes, key=lambda x: x[0], reverse=True)[:limit])
    return "\n".join(lines) or "empty"


def dynamic_prompt(arc, round_id, operation, parents, n=1, band="all", niche="any empty niche", extra="",
                   feedback="none"):
    measured = [r for r in arc.values() if r.get("b200")]
    calib = "\n".join(f"- {r['id']}: B200 score {r['b200']['score']:.3f}, geomean {r['b200']['geomean_us']:.1f} µs, "
                      f"B200 S/M/L {fmt(band_geomeans(r['b200'].get('timings', {})))} µs" for r in measured)
    return "\n\n".join([
        f"=== TASK ===\nRound: {round_id}   Operation: {operation}\nProblem: #38 038_flux_multi_head_rmsnorm_qk   "
        f"Target niche: {niche}   Target band: {band}\nReturn {n} candidate(s). Each must follow OUTPUT CONTRACT "
        f"exactly. Give every candidate a new unique id starting with '{round_id}-'.\n{extra}",
        "=== ARCHIVE (best per kernel; times are geomean µs per size band S/M/L; S = B·S ≤ 600, M ≤ 2100, L above) "
        f"===\n{table(arc)}\nB200 results so far:\n{calib or 'none'}",
        f"=== HYPOTHESIS LEDGER ===\n{(CTX.parent / 'ledger.yaml').read_text() if (CTX.parent / 'ledger.yaml').exists() else 'none'}",
        f"=== LAB NOTEBOOK ===\n{lab_notebook(arc)}",
        "=== PARENTS ===\n" + ("\n".join(parent_block(arc[p]) for p in parents) if parents else "none"),
        f"=== FEEDBACK FROM LAST ATTEMPT ===\n{feedback}",
    ])
