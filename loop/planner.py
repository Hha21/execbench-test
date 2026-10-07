"""Decide what a round should attempt, and which tested candidates deserve the B200 portal.

Both decisions use the B200 per-workload portal data: the hidden baseline Tb per workload, and Tsol recovered from
the per-workload scores, so a predicted set of B200 times can be turned into a predicted portal score exactly.

  plan(arc)        -> tasks for the next round: exploit the band with the most score to gain on the best kernel,
                      the next band, one exploration of an untried niche, and repairs of recent failures.
  shortlist(arc,r) -> candidates of round r ranked by predicted B200 score, plus B200-only designs that cheap GPUs
                      cannot predict (exploration slots).
"""

import csv
import math
import re
from collections import defaultdict

import archive
from archive import ROOT

S_MAX, M_MAX = 600, 2100            # band edges in B*S tokens (S = 12.6-57.6 MB, M = 101-201 MB)
FIXED_TARGET_US = 2.0               # public B200 floor for a tiny streaming kernel is ~2.2-2.5 µs in total (b200_sota.md)
BW_TARGET_TBS = 7.1                 # best public B200 read+write streams fit 7.05-7.10 TB/s marginal (b200_sota.md)
EXPLORE_NICHES = [                  # (axis, tag, instruction) tried in order until one is missing from the archive
    # Dropped: launch:pdl (one-kernel designs gain nothing) and red:halfwarp (the r3 CuTe winner already does it).
    ("lang", "cutedsl", "Write the design in CuTe DSL (Python), which compiles for sm_100a; use it to reach a memory "
                        "path Triton cannot express."),
    ("tile", "token", "Process one token's 48 heads per program (tile:token) so the weight tile is constant."),
    ("lang", "gluon", "Write the design in Gluon (triton.experimental.gluon) for explicit layout and memory control."),
    ("grid", "capped4", "Small/medium specialist after FlashInfer's CAKE B200 kernels (b200_sota.md section 4): a persistent "
                        "grid capped at 4 CTAs per SM, each thread keeping 2-4 rows' 256-bit loads in flight and issuing "
                        "the next row's loads before the current row's reduction; parent's kernel for large sizes."),
    ("tile", "fat", "Cut the CTA count 4-16x: an empty grid of 49,152 CTAs already spans 26 us on B200 "
                    "(problem_038.md section 7). Give each thread several rows (loads for all of them issued first, then "
                    "the reductions), or larger CTAs, keeping registers low enough for full occupancy. Measure the "
                    "CTA-count effect with probe_b200 before writing the full kernel."),
    ("st", "l2order", "Order the work so that the outputs written last (the ones still in L2 when the kernel ends) are "
                      "as large a share as possible, e.g. reverse or interleave the tile order; keep the parent's "
                      "evict_last stores."),
]


def bs_of(workload):
    b, s = (int(x) for x in re.findall(r"=(\d+)", workload)[:2])
    return b * s


def anchors():
    """Per B*S: (Tb, Tsol) in µs, recovered from every ingested portal page."""
    path = ROOT / "poc" / "results" / "b200_portal_workloads.csv"
    tb, ts = {}, defaultdict(list)
    for r in csv.DictReader(open(path)):
        bs, t, b, s = bs_of(r["workload"]), float(r["latency_ms"]), float(r["baseline_ms"]), float(r["sol_score"])
        tb[bs] = b * 1e3
        if abs(1 - 2 * s) > 0.1:
            ts[bs].append((b - s * t - s * b) / (1 - 2 * s) * 1e3)
    # Sizes without a well-conditioned score: Tsol measured at about 0.53x the 8 TB/s floor (problem card).
    return {bs: (tb[bs], sorted(ts[bs])[len(ts[bs]) // 2] if ts.get(bs) else 0.53 * mb(bs) / 8) for bs in tb}


def score(times_by_bs, anc):
    """Mean per-workload score; times_by_bs maps B*S -> µs (shapes with equal B*S time identically)."""
    wl = [r for r in csv.DictReader(open(ROOT / "poc" / "results" / "b200_portal_workloads.csv"))]
    shapes = sorted({r["workload"] for r in wl})
    vals = []
    for w in shapes:
        bs = bs_of(w)
        tb, ts = anc[bs]
        t = times_by_bs[bs]
        vals.append((tb - ts) / ((t - ts) + (tb - ts)))
    return sum(vals) / len(vals)


def b200_times(rec):
    """B200 µs per B*S for an archive record."""
    out = {}
    for k, v in ((rec.get("b200") or {}).get("timings") or {}).items():
        out[archive.tokens(k)] = v
    return out


def mb(bs):
    return bs * 98304 / 1e6


def fit_fixed_bw(times):
    """Least-squares t = a + bytes/bw over S and M sizes; returns (fixed µs, TB/s)."""
    pts = [(mb(bs), t) for bs, t in times.items() if bs <= M_MAX]
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return my - slope * mx, 1 / slope            # µs per MB -> MB/µs = TB/s


def best_kernel(arc):
    measured = [r for r in arc.values() if r.get("b200")]
    return max(measured, key=lambda r: r["b200"]["score"])


def band_tbs(times, lo, hi):
    xs = [mb(bs) / t for bs, t in times.items() if lo < bs <= hi]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def gains(best, anc):
    t = b200_times(best)
    base = score(t, anc)
    fixed, bw = fit_fixed_bw(t)
    s_t = {bs: (min(v, FIXED_TARGET_US + mb(bs) / bw) if bs <= S_MAX else v) for bs, v in t.items()}
    # M/L: the best kernel already beats the physical read+write rate (outputs left in L2 at kernel end are written
    # back after the timed window), so a bandwidth target no longer fits; value a further 5% instead.
    l_t = {bs: (min(v, fixed + mb(bs) / BW_TARGET_TBS, 0.95 * v) if bw < BW_TARGET_TBS else 0.95 * v)
           if bs > S_MAX else v for bs, v in t.items()}
    return dict(base=base, fixed=fixed, bw=bw, S=score(s_t, anc) - base, ML=score(l_t, anc) - base,
                tbs_M=band_tbs(t, S_MAX, M_MAX), tbs_L=band_tbs(t, M_MAX, 1 << 30))


def worse_in_band(arc, best, lo, hi):
    """Kernels already measured on B200 that are slower than the best in this band (what not to repeat)."""
    bt = b200_times(best)
    out = []
    for r in arc.values():
        t = b200_times(r)
        if not t or r["id"] == best["id"]:
            continue
        rel = [t[bs] / bt[bs] for bs in t if lo < bs <= hi and bs in bt]
        if rel and sum(rel) / len(rel) > 1.02:
            n = r.get("niche") or {}
            out.append(f"{r['id']} ({n.get('mem', '?')}/{n.get('grid', '?')}): {100 * (sum(rel) / len(rel) - 1):+.0f}%")
    return out


def in_flight(arc):
    """Proposed or tested ideas with no B200 result yet, so the LLM does not propose them again."""
    out = []
    for r in arc.values():
        if not r.get("b200") and str(r.get("round", "")).startswith("r") and r.get("status") in ("proposed", "passed"):
            n = r.get("niche") or {}
            out.append(f"{r['id']} ({n.get('mem', '?')}/{n.get('grid', '?')}, {r.get('language', '?')})")
    return out


def plan(arc, max_tasks=3, last_round=None):
    anc = anchors()
    best = best_kernel(arc)
    g = gains(best, anc)
    tasks = []
    pending = "; ".join(in_flight(arc)) or "none"
    note = f" Ideas already proposed but not yet measured on B200 (do not repeat them): {pending}."
    # Repairs first: candidates that failed correctness or build in the last round.
    for r in arc.values():
        if last_round and r.get("round") == last_round and str(r.get("status", "")).startswith("failed") \
                and r.get("failure_log") and len(tasks) < max_tasks:
            tasks.append(dict(operation="repair", parents=[r["id"]], band="all",
                              instructions="Fix this candidate so it passes the harness; keep its design.",
                              feedback=r["failure_log"]))
    s_task = dict(operation="specialisation", parents=[best["id"]], band="S", gain=g["S"], instructions=(
        f"Goal: cut the fixed cost on small inputs (B*S <= {S_MAX}). On B200 the parent fits {g['fixed']:.1f} us fixed "
        f"+ bytes / {g['bw']:.1f} TB/s on small and medium sizes; reaching {FIXED_TARGET_US} us fixed would add about "
        f"{g['S']:+.3f} to the score. Write a small-input kernel behind a dispatch:size wrapper that routes every larger "
        f"size to the parent's kernel with the parent's exact configuration, so the B200 result isolates the change. "
        f"The best public B200 streaming kernels take about 2.2-2.5 us in total at the smallest sizes, so about 1 us is "
        f"recoverable; see b200_sota.md section 4 for what they do. "
        f"Keep the answer focused: one candidate, and say in the card what you think the fixed cost consists of. "
        f"Slower on B200 than the parent in this band already: {'; '.join(worse_in_band(arc, best, 0, S_MAX)) or 'none'}." + note))
    l_task = dict(operation="structural_mutation", parents=[best["id"]], band="M,L", gain=g["ML"], instructions=(
        f"Goal: cut medium and large input latency (B*S > {S_MAX}) on B200. The parent reaches {g['tbs_M']:.2f} TB/s (M) "
        f"and {g['tbs_L']:.2f} TB/s (L) effective; a further 5% would add about {g['ML']:+.3f} to the score. Its fit is "
        f"{g['bw']:.1f} TB/s, above the physical read+write rate: outputs still in L2 when the kernel ends are written "
        f"back after the timed window (see problem_038.md section 7), so think about what occupies L2 at the end, not "
        f"only bandwidth. Propose the single change most likely to help, correct for every shape, one launch. "
        f"Slower on B200 than the parent in these bands already: "
        f"{'; '.join(worse_in_band(arc, best, S_MAX, 1 << 30)) or 'none'}." + note))
    for t in sorted([s_task, l_task], key=lambda t: -t["gain"]):
        if len(tasks) < max_tasks:
            tasks.append(t)
    tried = {(k, str(v)) for r in arc.values() for k, v in (r.get("niche") or {}).items()} | \
            {("lang", str(r.get("language", "")).replace("cuda_cpp", "cuda").replace("cute_dsl", "cutedsl"))
             for r in arc.values()} | \
            {tuple(str(r["task"]["niche"]).split(":", 1)) for r in arc.values()     # the niche a task asked for
             if ":" in str((r.get("task") or {}).get("niche", ""))}
    for axis, tag, text in EXPLORE_NICHES:
        if (axis, tag) not in tried and len(tasks) < max_tasks:
            tasks.append(dict(operation="new_design", parents=[best["id"]], band="all", niche=f"{axis}:{tag}",
                              instructions=f"Explore the empty niche {axis}:{tag}. {text} It must be correct for every "
                                           f"shape; aim to beat the parent somewhere and say where in the card."))
            break
    return dict(best=best["id"], best_score=best["b200"]["score"], gains=g, tasks=tasks)


def band_of(bs):
    return "S" if bs <= S_MAX else "M" if bs <= M_MAX else "L"


def rented_exponents(arc, best=None):
    """Per size band, k such that (portal time ratio) ~ (rented-B200 time ratio) ** k, both relative to the best kernel.

    Least squares through the origin in log space over every kernel timed on both. The rented B200 runs unlocked
    clocks, which exaggerated S/M differences about 1.5-2x in the 7 October calibration (k < 1); L matched (k ~ 1).
    """
    best = best or best_kernel(arc)
    toks = lambda t: {archive.tokens(k): v for k, v in t.items()}
    bp, br = b200_times(best), toks((best.get("timings") or {}).get("B200r", {}))
    num, den = defaultdict(float), defaultdict(float)
    for r in arc.values():
        if r["id"] == best["id"]:
            continue
        p, q = b200_times(r), toks((r.get("timings") or {}).get("B200r", {}))
        for bs in set(p) & set(q) & set(bp) & set(br):
            x, y = math.log(q[bs] / br[bs]), math.log(p[bs] / bp[bs])
            num[band_of(bs)] += x * y
            den[band_of(bs)] += x * x
    return {b: min(1.5, max(0.3, num[b] / den[b])) if den[b] > 1e-6 else 1.0 for b in "SML"}


def representative_gpu(rec):
    """(gpu with timings, gpu the card names but without timings yet), H200 preferred."""
    card = archive.parse_card(rec.get("card", ""))
    runs = card.get("runs_on") or {}
    def says_representative(v):
        if isinstance(v, dict):
            return str(v.get("representative")).lower() == "true"
        return "representative:true" in str(v).replace(" ", "").replace("'", "").replace('"', "").lower()

    named = [g for g in ("H200", "A100") if says_representative(runs.get(g, ""))]
    have = [g for g in named if rec.get("timings", {}).get(g)]
    return (have[0] if have else None), (named[0] if named and not have else None)


def shortlist(arc, round_id, k_predicted=3, k_explore=2):
    anc = anchors()
    best = best_kernel(arc)
    bt = b200_times(best)
    ranked, explore, awaiting = [], [], []
    for r in arc.values():
        if r.get("round") != round_id or r.get("status") != "passed" or r.get("b200"):
            continue
        gpu, waiting = representative_gpu(r)
        if (r.get("timings") or {}).get("B200r") and (best.get("timings") or {}).get("B200r"):
            gpu, waiting = "B200r", None          # a rented B200 beats any cheap GPU (calibration 7 October)
        if waiting:
            awaiting.append(dict(id=r["id"], gpu=waiting))
            continue
        ref = (best.get("timings") or {}).get(gpu or "", {})
        if gpu and ref:
            ct = r["timings"][gpu]
            rel = defaultdict(list)
            for key, v in ct.items():
                if v and ref.get(key):
                    rel[archive.tokens(key)].append(v / ref[key])
            k = rented_exponents(arc, best) if gpu == "B200r" else {"S": 1.0, "M": 1.0, "L": 1.0}
            pred = {bs: bt[bs] * ((sum(rel[bs]) / len(rel[bs])) ** k[band_of(bs)] if rel.get(bs) else 1.0)
                    for bs in bt}
            ranked.append(dict(id=r["id"], gpu=gpu, predicted_score=score(pred, anc),
                               rel=f"{fmt_rel(rel, 0, S_MAX)}/{fmt_rel(rel, S_MAX, M_MAX)}/{fmt_rel(rel, M_MAX, 1 << 30)}"))
        else:
            explore.append(dict(id=r["id"], gpu=None, predicted_score=None,
                                rel="no representative cheap GPU (B200-only features)"))
    ranked.sort(key=lambda x: -x["predicted_score"])
    return dict(best=best["id"], best_score=best["b200"]["score"], awaiting=awaiting,
                picks=ranked[:k_predicted] + explore[:k_explore], ranked=ranked, explore=explore)


def fmt_rel(rel, lo, hi):
    xs = [x for bs, v in rel.items() if lo < bs <= hi for x in v]
    return f"{100 * (math.exp(sum(map(math.log, xs)) / len(xs)) - 1):+.0f}%" if xs else "-"
