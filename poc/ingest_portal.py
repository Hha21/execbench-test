"""Record SOL-ExecBench portal results from saved submission pages.

Save the submission's detail page from the browser (File > Save Page As, "HTML only" is enough), or paste the page's
text (select all, copy; works from a phone) into a .txt file, then:

  python3 poc/ingest_portal.py "path/to/#61441_results.html" [more pages ...]
  python3 poc/ingest_portal.py page.html --vid v039      # override the variant ID (one page only)

The variant ID defaults to the uploaded file's name as shown on the page (v039.json -> v039).
Re-ingesting a submission replaces its rows, so running it twice is harmless.

Writes
  results/b200_portal.csv            one row per submission (what b200_plan.py reads)
  results/b200_portal_workloads.csv  one row per submission x workload, including the hidden baseline
  results/portal_html/<id>_<vid>.html  a copy of each page, for provenance

Portal arithmetic (checked against submission #61441): the overall latency is the geometric mean of the
per-workload latencies, the overall score is the arithmetic mean of the per-workload scores, and the
"Avg Speedup" is the geometric mean of baseline/latency (checked on #61435: 0.885 vs 0.89 shown).
Standard library only.
"""

import argparse
import csv
import math
import re
import shutil
import statistics
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUMMARY = RESULTS / "b200_portal.csv"
WORKLOADS = RESULTS / "b200_portal_workloads.csv"
SUMMARY_FIELDS = ["vid", "latency_ms", "sol_score", "fast_1", "avg_speedup", "submission_id", "kernel", "device",
                  "eval_stack", "mode", "result", "source_html"]
WORKLOAD_FIELDS = ["vid", "submission_id", "workload", "latency_ms", "baseline_ms", "speedup", "sol_score",
                   "t_sol_ms_derived"]
BYTES_PER_BS_038 = 48 * 128 * 4 * 4  # #38: read Q and K, write two outputs, fp32 -> bytes per (batch x seq)


class Page(HTMLParser):
    """Collects the visible text outside tables, and every table as rows of cell strings."""

    def __init__(self):
        super().__init__()
        self.text, self.tables = [], []
        self._skip = 0
        self._row = self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "svg", "noscript"):
            self._skip += 1
        elif tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "svg", "noscript"):
            self._skip -= 1
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._row is None:
            d = " ".join(data.split())
            if d:
                self.text.append(d)


def field_after(text, label):
    """Value of the text chunk that follows a label such as 'Device' or 'SOL Score:'."""
    for i, chunk in enumerate(text[:-1]):
        if chunk == label:
            return text[i + 1]
    return None


def parse_page(path):
    p = Page()
    p.feed(Path(path).read_text(encoding="utf-8", errors="replace"))
    sub = next((m[1] for c in p.text if (m := re.fullmatch(r"Submission #(\d+)", c))), None)
    table = next((t for t in p.tables if t and t[0][:2] == ["Workload", "Latency (ms)"]), None)
    if sub is None or table is None:
        sys.exit(f"{path}: not a submission page with a 'Benchmark Results' table (did the page finish loading?)")
    header, rows = table[0], table[1:]
    col = {name: i for i, name in enumerate(header)}
    workloads = []
    for r in rows:
        t, tb = float(r[col["Latency (ms)"]]), float(r[col["Baseline (ms)"]])
        s = float(r[col["SOL Score"]])
        # S = (Tb - Ts) / ((t - Ts) + (Tb - Ts))  =>  Ts = (Tb - S*t - S*Tb) / (1 - 2S); ill-conditioned near S = 0.5
        ts = (tb - s * t - s * tb) / (1 - 2 * s) if abs(1 - 2 * s) > 0.1 else ""
        workloads.append(dict(workload=r[col["Workload"]], latency_ms=t, baseline_ms=tb,
                              speedup=r[col["Speedup"]].rstrip("x"), sol_score=s,
                              t_sol_ms_derived=f"{ts:.5f}" if ts != "" else ""))
    submitted = field_after(p.text, "Submitted File:")
    info = dict(submission_id=sub, kernel=field_after(p.text, "Kernel"), device=field_after(p.text, "Device"),
                mode=field_after(p.text, "Mode"), result=field_after(p.text, "Result:"),
                eval_stack=next((c.removeprefix("Evaluation Stack ") for c in p.text
                                 if c.startswith("Evaluation Stack ")), ""),
                page_score=field_after(p.text, "SOL Score:"), submitted_file=submitted)
    return info, workloads


LABEL = r"[a-z_]+=\d+(?:,\s*[a-z_]+=\d+)*"
NUM = r"(\d+(?:\.\d+)?)"


def parse_text(path):
    """Like parse_page, from the page's text (select all + copy, e.g. on a phone) instead of the saved HTML.
    Cells may arrive separated by tabs, spaces or newlines."""
    s = Path(path).read_text(encoding="utf-8", errors="replace")
    after = lambda label: (m[1] if (m := re.search(label + r"\s*(\S+)", s)) else None)
    sub = after(r"Submission\s*#")
    rows = re.findall(rf"({LABEL})\s+{NUM}\s+{NUM}\s+{NUM}\s*x\s+{NUM}", s)
    if sub is None or not rows:
        sys.exit(f"{path}: no 'Submission #' or no workload rows (label, latency, baseline, speedup x, score) in the text")
    workloads = []
    for label, t, tb, speed, score in rows:
        t, tb, sc = float(t), float(tb), float(score)
        ts = (tb - sc * t - sc * tb) / (1 - 2 * sc) if abs(1 - 2 * sc) > 0.1 else ""
        workloads.append(dict(workload=" ".join(label.split()), latency_ms=t, baseline_ms=tb, speedup=speed,
                              sol_score=sc, t_sol_ms_derived=f"{ts:.5f}" if ts != "" else ""))
    kernel = re.search(r"\bKernel\s+(\d+_\w+)", s)                # not the title's "GPU Kernel Performance"
    info = dict(submission_id=sub, kernel=kernel and kernel[1], device=after(r"\bDevice"), mode=after(r"\bMode"),
                result=after(r"Result:"), eval_stack=(after(r"Evaluation Stack") or ""),
                page_score=after(r"SOL Score:"), submitted_file=after(r"Submitted File:"))
    return info, workloads


def is_text(path):
    return "<table" not in Path(path).read_text(encoding="utf-8", errors="replace")[:2_000_000]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Older hand-written rows used a " #ID" column with values like "#61435".
    for r in rows:
        for k in list(r):
            if k is not None and k.strip() in ("#ID", "ID", "id"):
                r["submission_id"] = (r.pop(k) or "").strip().lstrip("#")
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", type=Path)
    ap.add_argument("--vid", help="variant ID, if it should differ from the uploaded file's name (one page only)")
    args = ap.parse_args()
    if args.vid and len(args.pages) > 1:
        sys.exit("--vid only works with a single page")

    summary, per_wl = read_csv(SUMMARY), read_csv(WORKLOADS)
    (RESULTS / "portal_html").mkdir(parents=True, exist_ok=True)
    for page in args.pages:
        text = is_text(page)
        info, wls = parse_text(page) if text else parse_page(page)
        vid = args.vid or Path(info["submitted_file"] or "").stem
        if not vid:
            sys.exit(f"{page}: no uploaded file name on the page; pass --vid")
        lat = [w["latency_ms"] for w in wls]
        speed = [w["latency_ms"] and w["baseline_ms"] / w["latency_ms"] for w in wls]
        geo = math.exp(statistics.mean(math.log(x) for x in lat))
        mean_s = statistics.mean(w["sol_score"] for w in wls)
        if info["page_score"] and abs(float(info["page_score"]) - mean_s) > 1e-4:
            print(f"  warning: page score {info['page_score']} != mean of workload scores {mean_s:.6f}")
        for check, want in (("device", "B200"), ("result", "AC"), ("eval_stack", "v1.1")):
            if info[check] != want:
                print(f"  warning: {check} is {info[check]!r}, expected {want!r}")

        sid = info["submission_id"]
        summary = [r for r in summary if r.get("submission_id") != sid]
        summary.append(dict(vid=vid, latency_ms=f"{geo:.6f}", sol_score=f"{mean_s:.6f}",
                            fast_1=sum(x > 1 for x in speed), avg_speedup=f"{statistics.geometric_mean(speed):.3f}",
                            source_html=f"portal_html/{sid}_{vid}.{'txt' if text else 'html'}",
                            **{k: info[k] for k in ("submission_id", "kernel", "device", "eval_stack", "mode", "result")}))
        per_wl = [r for r in per_wl if r.get("submission_id") != sid]
        per_wl += [dict(vid=vid, submission_id=sid, **w) for w in wls]
        shutil.copy(page, RESULTS / "portal_html" / f"{sid}_{vid}.{'txt' if text else 'html'}")

        print(f"#{sid} {vid}: score {mean_s:.4f}, geomean latency {geo * 1e3:.2f} us, "
              f"faster than baseline on {sum(x > 1 for x in speed)}/{len(wls)} workloads ({info['kernel']}, {info['device']})")
        for w in sorted(wls, key=lambda w: w["latency_ms"]):
            note = ""
            if (info["kernel"] or "").startswith("038_"):
                b, s = (int(x) for x in re.findall(r"=(\d+)", w["workload"])[:2])
                note = f"  {b * s * BYTES_PER_BS_038 / (w['latency_ms'] * 1e-3) / 1e12:5.2f} TB/s"
            print(f"   {w['workload']:30s} {w['latency_ms'] * 1e3:7.1f} us  baseline {w['baseline_ms'] * 1e3:7.1f} us"
                  f"  score {w['sol_score']:.3f}{note}")

    summary.sort(key=lambda r: int(r.get("submission_id") or 0))
    write_csv(SUMMARY, summary, SUMMARY_FIELDS)
    write_csv(WORKLOADS, per_wl, WORKLOAD_FIELDS)
    print(f"updated {SUMMARY.relative_to(HERE.parent)} ({len(summary)} submissions) and "
          f"{WORKLOADS.relative_to(HERE.parent)} ({len(per_wl)} workload rows)")


if __name__ == "__main__":
    main()
