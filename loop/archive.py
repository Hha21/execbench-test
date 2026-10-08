"""The loop's archive for the current problem: one record per kernel, with its design card, measurements and portal
results. Stored as problems/<name>/archive.json (id -> record); see loop/problem.py.

Record fields
  id, round, parents, operation, language, niche {mem, grid, spec, ...}, card (raw YAML), solution (repo path),
  status (proposed | rejected: ... | failed: ... | passed), timings {gpu: {workload key: latency_us}},
  static_sm100 [ {kernel, regs, smem, local, resident_per_sm, ops} ], b200 {score, geomean_us, timings {key: us}}
Workload keys are the variable-axis values joined by commas ("2,128" for #38's batch_size=2, seq_len=128).
"""

import csv
import json
import math
import re
from pathlib import Path

import problem

ROOT = problem.ROOT
GPUS = ("B200r", "H200", "A100", "L40S")    # B200r: rented B200 (Modal), unlocked clocks
PORTAL = ROOT / "poc" / "results" / "b200_portal.csv"
PORTAL_WL = ROOT / "poc" / "results" / "b200_portal_workloads.csv"


def P():
    return problem.current()


def load():
    path = P().archive
    return refresh(json.loads(path.read_text())) if path.exists() else {}


def save(arc):
    """Write the archive, keeping records another process added since this copy was loaded (records are never
    deleted, so a long-running round must not drop kernels registered while it ran)."""
    path = P().archive
    path.parent.mkdir(parents=True, exist_ok=True)
    disk = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**disk, **arc}, indent=1, sort_keys=True))


def parse_card(text):
    """Design card -> dict. PyYAML when available (cards nest, e.g. runs_on.H200.representative); else a tiny reader."""
    try:
        import yaml
        card = yaml.safe_load(text)
        if isinstance(card, dict):
            return card
    except Exception:
        pass
    return _parse_card_simple(text)


def _parse_card_simple(text):
    """Tiny YAML reader: top-level scalars, lists and one level of mapping (block or flow)."""
    card, key = {}, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.split(" #")[0].rstrip()
        indent = len(line) - len(line.lstrip())
        if indent == 0 and ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if val.startswith("{") and val.endswith("}"):
                card[key] = {k: v.strip('"') for k, v in re.findall(r'([\w-]+):\s*("[^"]*"|[^,}]+)', val[1:-1])}
                card[key] = {k: v.strip().strip('"') for k, v in card[key].items()}
            elif val.startswith("[") and val.endswith("]"):
                card[key] = [x.strip().strip('"') for x in val[1:-1].split(",") if x.strip()]
            elif val in ("", ">-", "|", ">"):
                card[key] = None
            else:
                card[key] = val.strip('"')
        elif key is not None and line.lstrip().startswith("- "):
            card[key] = (card[key] if isinstance(card.get(key), list) else []) + [line.lstrip()[2:].strip().strip('"')]
        elif key is not None and ":" in line and indent > 0:
            k, _, v = line.strip().partition(":")
            if not isinstance(card.get(key), dict):
                card[key] = {}
            card[key][k.strip()] = v.strip().strip('"')
        elif key is not None and card.get(key) is None:
            card[key] = line.strip()
        elif key is not None and isinstance(card.get(key), str):
            card[key] += " " + line.strip()
    return card


def tokens(key):
    """Product of the variable axes of a workload key (B*S for #38)."""
    return P().tokens(key)


def band_geomeans(timings):
    out = {}
    for name in "SML":
        xs = [v for k, v in timings.items() if v and P().band(k) == name]
        out[name] = math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None
    xs = [v for v in timings.values() if v]
    out["all"] = math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None
    return out


def read_traces(path):
    """Harness JSONL trace -> ({workload key: latency_us}, set of statuses)."""
    times, statuses = {}, set()
    for line in open(path):
        if not line.strip():
            continue
        t = json.loads(line)
        ev = t.get("evaluation") or {}
        statuses.add(ev.get("status"))
        lat = (ev.get("performance") or {}).get("latency_ms")
        times[P().key(t["workload"]["axes"])] = lat * 1e3 if lat else None
    return times, statuses


def portal_rows(path):
    """Rows of a portal CSV that belong to the current problem."""
    if not path.exists():
        return []
    name = P().name
    return [r for r in csv.DictReader(open(path)) if r.get("definition", name) == name]


def ingest_portal(arc):
    """Attach portal results (poc/results/b200_portal*.csv, current problem only) to matching archive ids."""
    per = {}
    for r in portal_rows(PORTAL_WL):
        per.setdefault(r["vid"], {})[P().key_from_label(r["workload"])] = float(r["latency_ms"]) * 1e3
    for r in portal_rows(PORTAL):
        vid = r["vid"]
        if vid in arc:
            arc[vid]["b200"] = dict(score=float(r["sol_score"]), geomean_us=float(r["latency_ms"]) * 1e3,
                                    submission=r.get("submission_id"), timings=per.get(vid, {}))


def refresh(arc):
    """Rebuild every record's measurements from the result files on disk (idempotent).

    Timings, statuses, failure logs, sm_100a statistics and portal results are derived data, so they are re-read on
    every load. Concurrent writers (e.g. a background proposal round) can then never lose them by saving an older copy.
    """
    d = P().dir
    timing_dirs = [d / "gen2" / "results" / "timing", d / "b200" / "timing"] + \
                  sorted(P().rounds.glob("*/results/timing"))
    verdicts = {}                                     # id -> {gpu: None (passed) or failure text}
    for tdir in timing_dirs:
        for gdir in sorted(p for p in tdir.glob("*") if p.is_dir()):
            for tr in gdir.glob("*.jsonl"):
                cid = tr.stem
                if "_rep" in cid or cid not in arc or not tr.read_text().strip():
                    continue
                times, statuses = read_traces(tr)
                arc[cid].setdefault("timings", {})[gdir.name] = times
                ok = statuses == {"PASSED"}
                verdicts.setdefault(cid, {})[gdir.name] = None if ok else f"failed on {gdir.name}: {sorted(statuses)}"
                log = gdir / f"{cid}.log"
                if log.exists() and not ok:
                    arc[cid]["failure_log"] = log.read_text()[-3000:]
            for log in gdir.glob("*.log"):            # build failures leave a log but no (or an empty) trace
                cid = log.stem
                trace = gdir / f"{cid}.jsonl"
                if cid in arc and (not trace.exists() or not trace.read_text().strip()):
                    verdicts.setdefault(cid, {})[gdir.name] = f"failed: build or run error on {gdir.name}"
                    arc[cid]["failure_log"] = log.read_text()[-3000:]
    for cid, v in verdicts.items():                   # a failure on any GPU (above all the B200) beats a pass elsewhere
        fails = [x for x in v.values() if x]
        arc[cid]["status"] = "; ".join(fails) if fails else "passed"
    for sf in sorted(P().rounds.glob("*/results/static_*.jsonl")) + sorted((d / "b200").glob("static_*.jsonl")):
        for line in open(sf):
            s = json.loads(line)
            if s["id"] in arc and s.get("kernels"):
                arc[s["id"]]["static_sm100"] = [{k: v for k, v in kk.items() if k != "target"} for kk in s["kernels"]]
            elif s["id"] in arc and s.get("error") and not arc[s["id"]].get("static_sm100"):
                arc[s["id"]]["static_error"] = s["error"][:500]
    ingest_portal(arc)
    return arc


def fmt(bands):
    return "/".join("-" if bands.get(b) is None else f"{bands[b]:.1f}" for b in ("S", "M", "L"))


def table(arc, limit=40):
    """Archive summary in the generation-protocol §4 format (times are geomean µs per size band S/M/L)."""
    rows = []
    for rec in arc.values():
        b200 = rec.get("b200") or {}
        a100 = band_geomeans(rec.get("timings", {}).get("A100", {})).get("all")
        key = (-(b200.get("score") or 0), a100 or 1e9)
        rows.append((key, rec))
    rows.sort(key=lambda x: x[0])
    head = ("| id | niche (mem/grid/spec) | lang | rented B200 S/M/L | H200 S/M/L | A100 S/M/L | L40S S/M/L | "
            "portal B200 S/M/L | B200 score | B200 regs/smem/fit | status |\n|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for _, rec in rows[:limit]:
        n = rec.get("niche") or {}
        cells = [fmt(band_geomeans(rec.get("timings", {}).get(g, {}))) for g in GPUS]
        b200 = rec.get("b200") or {}
        st = rec.get("static_sm100") or []
        res = "; ".join(f"{k.get('regs')}/{k.get('smem')}/{k.get('resident_per_sm')}" for k in st[:2]) or "-"
        lines.append(f"| {rec['id']} | {n.get('mem', '?')}/{n.get('grid', '?')}/{n.get('spec', '?')} | "
                     f"{rec.get('language', '?')} | {' | '.join(cells)} | {fmt(band_geomeans(b200.get('timings', {})))} | "
                     f"{b200.get('score', 'pending') if b200 else 'pending'} | {res} | {rec.get('status', '?')} |")
    return "\n".join(lines)


def seed():
    """#38 only, historical: build the archive from the proof of concept and generation 2 (idempotent)."""
    arc = load()
    # Generation 1: kernels that reached the B200 portal, with their CSF3 timings.
    table_csv = ROOT / "poc" / "results" / "rehearsal_with_h200" / "table.csv"
    gen1 = {}
    for r in csv.DictReader(open(table_csv)):
        rec = gen1.setdefault(r["vid"], {"A100": {}, "H200": {}, "L40S": {}})
        for g in GPUS:
            rec[g][f"{r['batch_size']},{r['seq_len']}"] = float(r[f"t_{g}"]) * 1e3
    static1 = {}
    for r in csv.DictReader(open(ROOT / "poc" / "results" / "static_features.csv")):
        if r["arch"] == "sm_100a" and r["ok"] == "1":
            static1[r["vid"]] = [dict(kernel="_qk_norm_kernel", regs=int(r["regs"]), smem=int(r["shared"]),
                                      local=int(r["local"]))]
    knobs = {r["vid"]: r for r in csv.DictReader(open(ROOT / "poc" / "variants" / "variants.csv"))}
    for vid in ("v028", "v039", "v040", "v050"):
        k = knobs[vid]
        arc.setdefault(vid, {}).update(
            id=vid, round="gen1", parents=[], operation="knob_variant", language="triton",
            niche=dict(mem="ldg128", grid="persistent" if k["PERSIST"] == "True" else "oneshot", spec="all",
                       launch="fused" if k["FUSE"] == "True" else "split", tile=f"rows{k['ROWS']}"),
            card=f"gen-1 knob variant: {', '.join(f'{a}={b}' for a, b in k.items() if a != 'vid')}",
            solution=f"poc/variants/{vid}.json", status="passed", timings=gen1[vid], static_sm100=static1.get(vid, []))
    # Generation 2.
    g2 = P().dir / "gen2"
    static2 = {}
    sp = g2 / "results" / "static.jsonl"
    if sp.exists():
        for line in open(sp):
            s = json.loads(line)
            if s["arch"] == "sm_100a" and "error" not in s:
                static2.setdefault(s["id"], []).append(dict(kernel=s["family"], regs=s["regs"], smem=s["smem"],
                                                            local=s["local"], resident_per_sm=s["resident_per_sm"],
                                                            ops=s["ops"]))
    for sol in sorted((g2 / "candidates").glob("*.json")):
        cid = sol.stem
        text = (g2 / "candidates" / f"{cid}.card.yaml").read_text()
        card = parse_card(text)
        rec = arc.setdefault(cid, {})
        rec.update(id=cid, round="gen2-r1", parents=card.get("parents") or [], operation=card.get("operation"),
                   language=card.get("language", "triton"), niche=card.get("niche") or {}, card=text,
                   solution=str(sol.relative_to(ROOT)), static_sm100=static2.get(cid, []))
        rec.setdefault("timings", {})
        for g in GPUS:
            tr = g2 / "results" / "timing" / g / f"{cid}.jsonl"
            if tr.exists():
                times, statuses = read_traces(tr)
                rec["timings"][g] = times
                rec["status"] = "passed" if statuses == {"PASSED"} else f"failed: {sorted(statuses)}"
        rec.setdefault("status", "proposed")
    ingest_portal(arc)
    save(arc)
    return arc


if __name__ == "__main__":
    print(table(seed()))
