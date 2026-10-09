"""Read a shape-split portal A/B: workloads with equal B*S do identical work, so within one submission a variant on one
shape is compared with the control on another, corrected by that shape pair's usual offset in earlier submissions.

    SOLX_PROBLEM=L1/038 .venv/bin/python loop/shape_ab.py x1-shape-ab
"""
import statistics
import sys
from collections import defaultdict

import archive
import planner
import problem

ARMS = {   # vid -> {group: (control shape, [(arm shape, label)])}
    "x1-shape-ab": {"256 tok": ("2,128", [("1,256", "r7: prefetch, no cluster")]),
                    "1024 tok": ("1,1024", [("8,128", "r7: R=1 EF+nc + L2 prefetch"), ("4,256", "F1 simple kernel")]),
                    "2048 tok": ("4,512", [("8,256", "c2 with R=3")]),
                    "4096 tok": ("16,256", [("32,128", "c2 with EF x + nc w"), ("4,1024", "F1 simple kernel")])},
}


def main(vid):
    P = problem.current()
    anc = planner.anchors()
    per = defaultdict(dict)
    for r in archive.portal_rows(archive.PORTAL_WL):
        per[r["vid"]][P.key_from_label(r["workload"])] = float(r["latency_ms"]) * 1e3
    if vid not in per:
        sys.exit(f"{vid}: no portal result ingested yet")
    hist = {v: t for v, t in per.items() if v != vid and len(t) == len(anc) and v not in ARMS}
    t = per[vid]
    print(f"{vid}: portal score {planner.score(t, anc):.4f}; offsets from {len(hist)} earlier single-kernel submissions")
    for group, (ctl, arms) in ARMS[vid].items():
        print(f"{group}: control {ctl} {t[ctl]:.1f} us")
        for k, label in arms:
            off = [(h[k] / h[ctl] - 1) * 100 for h in hist.values()]
            raw = (t[k] / t[ctl] - 1) * 100
            eff = raw - statistics.mean(off)
            sd = statistics.stdev(off)
            verdict = "faster" if eff < -2 * sd else "slower" if eff > 2 * sd else "no clear difference"
            print(f"   {k:>7} {label:30} {t[k]:6.1f} us  raw {raw:+5.1f}%  usual offset {statistics.mean(off):+5.2f}%"
                  f"  -> effect {eff:+5.1f}% (+-{sd:.1f}%): {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "x1-shape-ab")
