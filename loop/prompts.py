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
        "=== PARENTS ===\n" + ("\n".join(parent_block(arc[p]) for p in parents) if parents else "none"),
        f"=== FEEDBACK FROM LAST ATTEMPT ===\n{feedback}",
    ])
