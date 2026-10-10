"""Multi-fidelity emulator: predict portal B200 times (clocks locked at SM 1500 MHz) from rented-B200 timings
(unlocked, ~1940 MHz under load) and design features, with uncertainty.

Per workload, y = log(portal time / rented time) is modelled as
    m(size)                       quadratic in log2(B*S): the clock gap matters more for small inputs
  + f(features, size)             Gaussian process over the features of the code path serving that size
  + u_kernel                      per-kernel offset: structure the features do not explain
  + noise
A new kernel gets f's posterior plus the full variance of u_kernel, so a design unlike anything measured on the portal
comes with wide error bars. Training pairs: every kernel with portal per-workload results, rented timings (two reps,
loop/pair_timings.py) and features (loop/emulator_features.yaml, or a design card's `paths:` block).

  .venv/bin/python loop/emulator.py            # fit, show hyperparameters, leave-one-kernel-out validation
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import planner  # noqa: E402

FEATS = ["width256", "threads", "rows", "persistent", "tma", "cpasync", "wstat", "launches2", "x_ef", "w_el", "w_nc",
         "st_ef", "st_el", "prefetch", "triton", "cute", "cluster"]
import problem


def features_file():
    return problem.current().features


def copy_file():
    return problem.current().b200 / "copy_reference.json"   # plain read+write copy, rented B200, per workload
LAG_SCALE = 0.2


def copy_reference():
    import json
    f = copy_file()
    return json.loads(f.read_text()) if f.exists() else {}


def encode(path):
    g = path.get
    return np.array([
        1.0 if g("width", 128) == 256 else 0.0,
        (math.log2(g("threads") or 256) - 8) / 2,
        (math.log2(g("rows") or 16) - 4) / 2,
        1.0 if g("grid", "oneshot") == "persistent" else 0.0,
        1.0 if g("mem", "ldg") == "tma" else 0.0,
        1.0 if g("mem", "ldg") == "cpasync" else 0.0,
        1.0 if g("wstat") else 0.0,
        1.0 if g("launches", 1) == 2 else 0.0,
        1.0 if g("x", "none") == "ef" else 0.0,
        1.0 if g("w", "none") == "el" else 0.0,
        1.0 if g("w", "none") == "nc" else 0.0,
        1.0 if g("st", "none") == "ef" else 0.0,
        1.0 if g("st", "none") == "el" else 0.0,
        1.0 if g("prefetch") else 0.0,
        1.0 if g("lang") == "triton" else 0.0,
        1.0 if g("lang") == "cute" else 0.0,
        math.log2(g("cluster") or 1),                 # thread-block cluster size (1 = none)
    ])


def path_for(paths, tokens):
    """The path serving a workload; max_tokens bounds the product of the variable axes (B*S for #38)."""
    for p in paths:
        if p.get("max_tokens") is None or tokens <= p["max_tokens"]:
            return p
    return paths[-1]


def size_x(mb):
    """Workload size feature: log2 of its MB of compulsory traffic, centred at 100 MB (1024 tokens for #38)."""
    return (math.log2(mb) - math.log2(100)) / 3


def load_features(arc=None):
    """Curated features file, plus `paths:` blocks from design cards of archive kernels not in the file."""
    import yaml
    f = features_file()
    feats = (yaml.safe_load(f.read_text()) or {}) if f.exists() else {}
    for cid, rec in (arc or {}).items():
        if cid not in feats:
            card = archive.parse_card(rec.get("card", "")) if rec.get("card") else {}
            if isinstance(card.get("paths"), list) and card["paths"]:
                feats[cid] = card["paths"]
    return feats


def rented_times(rec):
    """Geometric mean over rented reps, per workload key."""
    t = rec.get("timings") or {}
    reps = [t[g] for g in ("B200r", "B200r2") if t.get(g)]
    out = {}
    for key in set().union(*reps) if reps else ():
        vals = [r[key] for r in reps if r.get(key)]
        if vals:
            out[key] = math.exp(sum(map(math.log, vals)) / len(vals))
    return out


def design_rows(paths, keys, rented):
    """Features per workload: the serving path's design, its lag behind a plain copy on the rented B200 (a kernel
    at copy speed is memory-limited; one that lags is partly limited by the SMs, whose clock the portal lowers),
    and the input size."""
    copy = copy_reference()
    rows = []
    for k in keys:
        lag = math.log(rented[k] / copy[k]) / LAG_SCALE if k in copy else 0.0
        rows.append(np.concatenate([encode(path_for(paths, archive.tokens(k))), [lag, size_x(problem.current().mbytes(k))]]))
    return np.array(rows)


def mean_basis(X):
    s, lag = X[:, -1], X[:, -2]
    return np.stack([np.ones_like(s), s, s * s, lag, lag * s], axis=1)


class Emulator:
    GRID = dict(lf=(0.75, 1.5, 3.0), ll=(0.5, 1.0, 2.0), ls=(0.5, 1.0, 2.0), sf=(0.01, 0.02, 0.04),
                sk=(0.005, 0.01, 0.02, 0.04), sn=(0.005, 0.01, 0.02))

    def __init__(self, arc=None, feats=None, exclude=()):
        self.arc = arc if arc is not None else archive.load()
        self.feats = feats if feats is not None else load_features(self.arc)
        ids, X, y, wkeys = [], [], [], []
        for cid, paths in self.feats.items():
            rec = self.arc.get(cid) or {}
            portal = (rec.get("b200") or {}).get("timings") or {}
            rented = rented_times(rec)
            keys = [k for k in portal if k in rented]
            if cid in exclude or not keys:
                continue
            X.append(design_rows(paths, keys, rented))
            y.extend(math.log(portal[k] / rented[k]) for k in keys)
            ids.extend([cid] * len(keys))
            wkeys.extend(keys)
        if not ids:
            raise ValueError("no kernel has both portal and rented-B200 timings yet for this problem")
        self.ids, self.X, self.y, self.keys = np.array(ids), np.vstack(X), np.array(y), wkeys
        self.kernels = sorted(set(ids))
        B = mean_basis(self.X)
        self.beta = np.linalg.lstsq(B, self.y, rcond=None)[0]
        self.r = self.y - B @ self.beta
        self.same = (self.ids[:, None] == self.ids[None, :]).astype(float)
        self.hp, self.lml = max(((hp, self._lml(hp)) for hp in self._grid()), key=lambda t: t[1])
        # The kernel-to-kernel spread of log(portal/rented) cannot be learned from a handful of kernels: with 2-3
        # portal kernels the fit puts it at 0.5-1%, while real misses on new problems were 2-6 sd (review
        # 2026-10-10). Leave-one-kernel-out on #38 (23 kernels) still has rms z ~1.5. Floor it accordingly.
        self.n_kernels = len(self.kernels)
        self.sk_floor = 0.03 if self.n_kernels < 5 else 0.015
        K = self._K(self.X, self.X, self.hp) + self.hp["sk"] ** 2 * self.same + self.hp["sn"] ** 2 * np.eye(len(self.y))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.r))

    def _grid(self):
        g = self.GRID
        for lf in g["lf"]:
            for ll in g["ll"]:
                for ls in g["ls"]:
                    for sf in g["sf"]:
                        for sk in g["sk"]:
                            for sn in g["sn"]:
                                yield dict(lf=lf, ll=ll, ls=ls, sf=sf, sk=sk, sn=sn)

    @staticmethod
    def _K(A, B, hp):
        scale = np.r_[np.full(A.shape[1] - 2, 1 / hp["lf"]), 1 / hp["ll"], 1 / hp["ls"]]
        a, b = A * scale, B * scale
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return hp["sf"] ** 2 * np.exp(-0.5 * np.maximum(d2, 0))

    def _lml(self, hp):
        n = len(self.r)
        K = self._K(self.X, self.X, hp) + hp["sk"] ** 2 * self.same + hp["sn"] ** 2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.r))
        return -0.5 * self.r @ a - np.log(np.diag(L)).sum() - 0.5 * n * math.log(2 * math.pi)

    def predict_log_ratio(self, paths, keys, rented):
        """Mean vector and covariance of log(portal / rented) for a new kernel's workloads."""
        Xs = design_rows(paths, keys, rented)
        Ks = self._K(Xs, self.X, self.hp)
        mean = mean_basis(Xs) @ self.beta + Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        sk = max(self.hp["sk"], self.sk_floor)     # a new kernel's own offset: shared by all its workloads
        cov = self._K(Xs, Xs, self.hp) - v.T @ v + sk ** 2 + self.hp["sn"] ** 2 * np.eye(len(keys))
        return mean, cov

    def predict(self, paths, rented, n=2000, best_score=None, seed=0):
        """Portal score distribution for a kernel given its rented per-workload µs (all 16 workloads)."""
        keys = sorted(rented, key=problem.current().mbytes)
        mean, cov = self.predict_log_ratio(paths, keys, rented)
        t_mean = {k: rented[k] * math.exp(m) for k, m in zip(keys, mean)}
        anc = planner.anchors()
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(keys)))
        z = np.random.default_rng(seed).standard_normal((n, len(keys)))
        scores = []
        for draw in z @ L.T + mean:
            scores.append(planner.score({k: rented[k] * math.exp(v) for k, v in zip(keys, draw)}, anc))
        scores = np.array(scores)
        out = dict(score=float(scores.mean()), sd=float(scores.std()), times=t_mean, n_train=self.n_kernels,
                   band_sd={b: float(np.sqrt(np.mean([cov[i, i] for i, k in enumerate(keys)
                                                      if planner.band_of(k) == b]))) for b in "SML"})
        if best_score is not None:
            out["p_better"] = float((scores > best_score).mean())
        return out


TOGGLES = [("x: ef", {"x": "ef"}), ("w: nc", {"w": "nc"}), ("w: el", {"w": "el"}), ("st: ef", {"st": "ef"}),
           ("st: none (vs el)", {"st": "none"}), ("width 128", {"width": 128}), ("threads 512, rows 32",
           {"threads": 512, "rows": 32}), ("threads 128, rows 8", {"threads": 128, "rows": 8}),
           ("grid persistent", {"grid": "persistent"}), ("prefetch L2", {"prefetch": True}), ("mem tma", {"mem": "tma"}),
           ("lang triton", {"lang": "triton"}), ("lang cute", {"lang": "cute"})]


def effect_keys():
    ks = sorted(problem.current().keys(), key=problem.current().mbytes)
    return [ks[0], ks[len(ks) // 2], ks[-1]]


def feature_effects(emu, base=None):
    """How the portal shifts each feature's effect relative to the rented B200: the emulator's change in
    log(portal/rented) when one feature of a base path is toggled, at the smallest / middle / largest workload."""
    if base is None:                                  # the best kernel's main (last) code path
        best = planner.best_kernel(emu.arc)
        base = dict((emu.feats.get(best["id"]) or [{}])[-1])
        base.pop("max_tokens", None)
    keys = effect_keys()
    copy = copy_reference()
    rented = {k: copy.get(k, 1.0) for k in keys}          # effects at copy-like speed (lag 0)
    m0, c0 = emu.predict_log_ratio([base], keys, rented)
    rows = []
    for name, change in TOGGLES:
        m1, c1 = emu.predict_log_ratio([{**base, **change}], keys, rented)
        # sd of the difference: both share the kernel offset, which cancels; keep the GP parts
        sd = np.sqrt(np.maximum(np.diag(c1) + np.diag(c0) - 2 * (emu.hp["sk"] ** 2 + emu.hp["sn"] ** 2), 1e-12))
        rows.append((name, [(100 * (math.exp(d) - 1), 100 * s) for d, s in zip(m1 - m0, sd)]))
    return rows


def effects_markdown(emu):
    k = effect_keys()
    lines = [f"| change from the best kernel's main path | smallest ({k[0]}) | middle ({k[1]}) | largest ({k[2]}) |",
             "|---|---|---|---|"]
    for name, vals in feature_effects(emu):
        lines.append(f"| {name} | " + " | ".join(f"{d:+.1f}% ± {s:.1f}" for d, s in vals) + " |")
    return "\n".join(lines)


def validate(arc=None):
    """Leave one kernel out (and r8+r9 together, which were proposed before either had a portal result)."""
    arc = arc or archive.load()
    feats = load_features(arc)
    full = Emulator(arc, feats)
    print(f"fit on {len(full.kernels)} kernels, {len(full.y)} workloads; hyperparameters {full.hp}")
    print(f"mean log(portal/rented) by band: " + ", ".join(
        f"{b} {np.mean([y for y, k in zip(full.y, full.keys) if planner.band_of(k) == b]):+.3f}"
        for b in "SML"))
    print(f"\n{'held-out kernel':26} {'portal':>7} {'predicted':>16} {'z':>5}   naive (k-correction)")
    groups = [[c] for c in full.kernels]
    if {"r8-ldg256-l2mode-disp", "r9-xo-spf-l2mode-disp"} <= set(full.kernels):   # #38: proposed before either had a result
        groups.append(["r8-ldg256-l2mode-disp", "r9-xo-spf-l2mode-disp"])
    for held in groups:
        em = Emulator(arc, feats, exclude=held)
        for cid in held:
            rec = arc[cid]
            rented = rented_times(rec)
            p = em.predict(feats[cid], rented)
            actual = rec["b200"]["score"]
            tag = " (r8+r9 held out together)" if len(held) > 1 else ""
            print(f"{cid:26} {actual:7.4f} {p['score']:9.4f} ± {p['sd']:.4f} {(actual - p['score']) / p['sd']:+5.1f}{tag}")


if __name__ == "__main__":
    validate()
    print("\nportal minus rented, by feature (positive = the portal is slower than the rented B200 suggests):")
    print(effects_markdown(Emulator()))
