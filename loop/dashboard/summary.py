"""Print one problem's authoritative numbers as JSON, using the loop's own modules.
Usage: SOLX_PROBLEM=<name> .venv/bin/python loop/dashboard/summary.py <name>   (run from the repo root)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "loop"))
if len(sys.argv) > 1:
    os.environ["SOLX_PROBLEM"] = sys.argv[1]
os.chdir(ROOT)
import archive, planner, problem  # noqa: E402


def main():
    pr = problem.current()
    arc = archive.load()
    keys = sorted(pr.keys(), key=pr.mbytes)
    try:
        anc = planner.anchors()
    except Exception as e:  # no portal data yet
        anc = {}
    best = planner.best_kernel(arc)
    kernels = []
    for kid, r in arc.items():
        b = r.get("b200") or {}
        rt = (r.get("timings") or {}).get("B200r") or {}
        bands = archive.band_geomeans(rt) if rt else {}
        kernels.append(dict(id=kid, round=r.get("round"), parents=r.get("parents"), operation=r.get("operation"),
                            language=r.get("language"), status=r.get("status"), score=b.get("score"),
                            geomean_us=b.get("geomean_us"), submission=b.get("submission"),
                            portal=b.get("timings"), rented=rt or None, bands=bands))
    out = dict(name=pr.name, level=pr.level, keys=keys,
               workloads=[dict(key=k, band=pr.band(k), mbytes=pr.mbytes(k),
                               tb=(anc.get(k) or [None, None])[0], tsol=(anc.get(k) or [None, None])[1]) for k in keys],
               best=best["id"] if best else None, best_score=best["b200"]["score"] if best else None,
               kernels=kernels, summary=pr.summary())
    print(json.dumps(out, default=str))


main()
