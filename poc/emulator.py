"""Multi-source emulator for kernel latency on a target GPU, plus a rehearsal harness.

Model (Kennedy-O'Hagan style): log t_target = linear blend of cheap sources + GP correction.
  cheap sources: log-latency on other GPUs, log of the analytical estimate for the target
  correction GP inputs: knobs, problem size, compiled-code features for the target's architecture

Rehearsal: pretend one GPU we can measure is the expensive target. Reveal n of its variants,
predict the rest, and score how well the emulator ranks them against simpler alternatives.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV

import analytic

warnings.filterwarnings("ignore", category=ConvergenceWarning)
HERE = Path(__file__).resolve().parent
KNOBS = ["ROWS", "NUM_WARPS", "NUM_STAGES", "FUSE", "PERSIST", "PROGS_PER_SM", "EVICT"]
STATIC = ["regs", "local", "sass_total", "n_LDG_128", "n_LDG_32", "n_STG_128", "n_STG_32", "n_SHFL", "n_BRA"]


# ---------------------------------------------------------------- data loading

def load_timings(timing_dir):
    recs = []
    for gdir in sorted(Path(timing_dir).iterdir()):
        for f in sorted(gdir.glob("v*.jsonl")):
            vid, _, rep = f.stem.partition("_rep")
            for line in open(f):
                t = json.loads(line)
                ev = t.get("evaluation") or {}
                perf = ev.get("performance") or {}
                recs.append(dict(gpu=gdir.name, vid=vid, rep=int(rep or 0), wid=t["workload"]["uuid"],
                                 batch_size=t["workload"]["axes"]["batch_size"], seq_len=t["workload"]["axes"]["seq_len"],
                                 status=ev.get("status"), lat_ms=perf.get("latency_ms")))
    return pd.DataFrame(recs)


def build_table(results=HERE / "results", variants=HERE / "variants"):
    """One row per (variant, workload): latency on every GPU, analytic estimates, static features."""
    knobs = pd.read_csv(variants / "variants.csv")
    for k in ("FUSE", "PERSIST", "EVICT"):
        knobs[k] = knobs[k].astype(str).eq("True").astype(int)
    tim = load_timings(results / "timing")
    static = pd.read_csv(results / "static_features.csv")
    gpus = sorted(tim.gpu.unique())

    main = tim[tim.rep == 0]
    passed = main.groupby("vid").status.apply(lambda s: (s == "PASSED").all())
    print("variants failing on some GPU/workload:", sorted(passed[~passed].index))
    wide = main[main.vid.isin(passed[passed].index)].pivot_table(
        index=["vid", "wid", "batch_size", "seq_len"], columns="gpu", values="lat_ms").reset_index()
    wide = wide.rename(columns={g: f"t_{g}" for g in gpus}).merge(knobs, on="vid")
    wide = wide.dropna(subset=[f"t_{g}" for g in gpus]).reset_index(drop=True)  # only variants timed everywhere
    wide["log_rows"] = np.log(wide.batch_size * wide.seq_len * analytic.H)

    for gpu, spec in analytic.GPUS.items():
        st = static[(static.arch == spec["arch"]) & (static.ok == 1)].set_index("vid")
        est = []
        for r in wide.itertuples():
            s = st.loc[r.vid] if r.vid in st.index else None
            e = analytic.estimate(gpu, {k: getattr(r, k) for k in KNOBS},
                                  {"batch_size": r.batch_size, "seq_len": r.seq_len},
                                  regs=None if s is None else s.regs, smem=None if s is None else s.shared)
            est.append(e["t_analytic"] * 1e3)
        wide[f"a_{gpu}"] = est
        for c in STATIC:
            wide[f"s_{gpu}_{c}"] = wide.vid.map(st[c]) if c in st else np.nan

    # Simulator: one number per variant (geometric mean over the small workloads it simulated).
    sim_cols = []
    for cfg_dir in sorted((results / "sim").glob("*")) if (results / "sim").exists() else []:
        recs = [r for f in cfg_dir.glob("v*.json") for r in json.loads(f.read_text()) if "sim_ms" in r]
        if not recs:
            continue
        r = pd.DataFrame(recs)
        r["dev"] = np.log(r.sim_ms) - np.log(r.sim_ms).groupby([r.batch_size, r.seq_len]).transform("mean")
        sim = np.exp(r.groupby("vid").dev.mean() + np.log(r.sim_ms).mean())
        if set(wide.vid) <= set(sim.index):
            col = f"sim_{cfg_dir.name}"
            wide[col] = wide.vid.map(sim)
            sim_cols.append(col)
        else:
            print(f"simulator {cfg_dir.name}: only {len(sim)} variants simulated, not used")

    noise = {}
    for g, df in tim[tim.status == "PASSED"].groupby("gpu"):
        reps = df.pivot_table(index=["vid", "wid"], columns="rep", values="lat_ms").dropna()
        if reps.shape[1] > 1:
            noise[g] = float(np.median(np.log(reps).std(axis=1)))
    wide.attrs["sim_cols"] = sim_cols
    return wide, gpus, noise


# ---------------------------------------------------------------- models

def _gp(d):
    k = ConstantKernel(1.0, (1e-3, 1e2)) * RBF(np.ones(d), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0)


class Emulator:
    """Ridge blend of cheap sources, then a GP on what the blend gets wrong."""

    def __init__(self, target, sources, use_gp=True, sim_cols=()):
        self.target, self.sources, self.use_gp = target, sources, use_gp
        self.blend_cols = [f"t_{g}" for g in sources] + [f"a_{target}"] + list(sim_cols)
        self.gp_cols = KNOBS + ["log_rows"] + [f"s_{target}_{c}" for c in STATIC]

    def _xb(self, df):
        return np.log(df[self.blend_cols].to_numpy(float))

    def _xg(self, df):
        x = df[self.gp_cols].to_numpy(float)
        return (x - self.mu) / self.sd

    def _blend(self, df):
        return self._xb(df) @ self.coef + df.wid.map(self.level).to_numpy(float)

    def fit(self, df):
        y = np.log(df[f"t_{self.target}"].to_numpy(float))
        x = self._xb(df)
        # Fixed effects: every workload gets its own level, and the blend weights are fitted only on
        # how variants differ *within* a workload. Otherwise the fit is dominated by the size effect
        # (big inputs are slow), which every source gets right and which does not affect the ranking.
        # Weights are non-negative (a variant that is slower on a cheap source should not be predicted
        # faster on the target), fitted on standardised sources with no shrinkage toward zero, and
        # fitted on each variant's mean over workloads: that is what gets ranked, and per-row fitting
        # lets a few very slow variants on large inputs dominate the weights.
        xv = pd.DataFrame(x).groupby(df.vid.to_numpy()).mean().to_numpy()
        yv = pd.Series(y).groupby(df.vid.to_numpy()).mean().to_numpy()
        xd, yd = xv - xv.mean(0), yv - yv.mean()
        scale = xd.std(0) + 1e-12
        b, _ = nnls(xd / scale, yd)
        if not b.any():  # nothing fits: fall back to an equal-weight average
            b = np.full(len(scale), yd.std() / ((xd / scale).mean(1).std() + 1e-12) / len(scale))
        self.coef = b / scale
        self.level = pd.Series(y - x @ self.coef).groupby(df.wid.to_numpy()).mean()
        resid = y - self._blend(df)
        xg = df[self.gp_cols].to_numpy(float)
        self.mu, self.sd = np.nanmean(xg, 0), np.nanstd(xg, 0) + 1e-9
        self.gp = _gp(len(self.gp_cols)).fit(np.nan_to_num(self._xg(df)), resid) if self.use_gp else None
        return self

    def predict(self, df):
        m = self._blend(df)
        if self.gp is None:
            return m, np.full(len(df), np.std(m) * 0.1 + 1e-3)
        r, s = self.gp.predict(np.nan_to_num(self._xg(df)), return_std=True)
        return m + r, s


def _demean(a, groups):
    """Subtract each group's mean (rows grouped by workload)."""
    return a - pd.DataFrame(a).groupby(np.asarray(groups)).transform("mean").to_numpy().reshape(a.shape)


class TargetOnlyGP:
    """Plain Bayesian optimisation: a GP from knobs to each variant's mean log-latency, target data only."""

    def __init__(self, target):
        self.target = target

    def fit(self, df):
        agg = df.groupby("vid").agg({**{k: "first" for k in KNOBS}, f"t_{self.target}": lambda s: np.log(s).mean()})
        x = agg[KNOBS].to_numpy(float)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        self.gp = _gp(len(KNOBS)).fit((x - self.mu) / self.sd, agg[f"t_{self.target}"].to_numpy(float))
        return self

    def predict(self, df):
        x = (df[KNOBS].to_numpy(float) - self.mu) / self.sd
        return self.gp.predict(x, return_std=True)  # same value for every workload of a variant


# ---------------------------------------------------------------- rehearsal

def per_variant(df, pred):
    """Aggregate per-workload log-latency predictions into one score per variant (geometric mean)."""
    return pd.Series(pred, index=df.index).groupby(df.vid).mean()


def zero_shot(df, target, sources, sim_cols=()):
    """Predictions available before any target measurement."""
    out = {f"only {g} timing": np.log(df[f"t_{g}"].to_numpy(float)) for g in sources}
    out.update({f"only simulator ({c[4:]})": np.log(df[c].to_numpy(float)) for c in sim_cols})
    out[f"maths model for {target}"] = np.log(df[f"a_{target}"].to_numpy(float))
    z = [(np.log(df[f"t_{g}"]) - np.log(df[f"t_{g}"]).mean()) / np.log(df[f"t_{g}"]).std() for g in sources]
    out["average of other GPUs"] = np.mean(z, axis=0)
    return out


def score(df, target, pred_log, train_vids):
    true = per_variant(df, np.log(df[f"t_{target}"].to_numpy(float)))
    pred = per_variant(df, pred_log)
    test = [v for v in true.index if v not in train_vids]
    rho = spearmanr(true[test], pred[test]).statistic if len(test) > 2 else np.nan
    # Pick the best-looking variant: measured ones count at their true time, the rest at the prediction.
    chosen = pred.where(~pred.index.isin(train_vids), true).idxmin()
    regret = float(np.exp(true[chosen] - true.min()) - 1)
    return rho, regret


def rehearse(df, target, sources, budgets, repeats, rng, sim_cols=()):
    vids = np.array(sorted(df.vid.unique()))
    rows = []
    for name, p in zero_shot(df, target, sources, sim_cols).items():
        rho, regret = score(df, target, p, set())
        rows.append(dict(method=name, n=0, rho=rho, regret=regret))
    for n in budgets:
        for rep in range(repeats):
            train = set(rng.choice(vids, n, replace=False))
            tr, te_all = df[df.vid.isin(train)], df
            models = [("emulator (blend + GP)", Emulator(target, sources)),
                      ("emulator (blend only)", Emulator(target, sources, use_gp=False)),
                      ("target-only BO surrogate", TargetOnlyGP(target))]
            if sim_cols:
                models.insert(0, ("emulator + simulator", Emulator(target, sources, sim_cols=sim_cols)))
            for name, model in models:
                if name.startswith("target-only") and n < 3:
                    continue
                mu, _ = model.fit(tr).predict(te_all)
                rho, regret = score(df, target, mu, train)
                rows.append(dict(method=name, n=n, rep=rep, rho=rho, regret=regret))
    return pd.DataFrame(rows)


def bo_race(df, target, sources, steps, repeats, rng, start=3, sim_cols=()):
    """Sequential search on the target: measure the variant with the lowest optimistic estimate next."""
    vids = np.array(sorted(df.vid.unique()))
    true = per_variant(df, np.log(df[f"t_{target}"].to_numpy(float)))
    out = []
    for rep in range(repeats):
        init = list(rng.choice(vids, start, replace=False))
        for name in ("emulator", "target-only BO", "random"):
            measured = list(init)
            for step in range(start, steps + 1):
                best = float(np.exp(true[measured].min() - true.min()) - 1)
                out.append(dict(method=name, rep=rep, evals=step, regret=best))
                left = [v for v in vids if v not in measured]
                if not left:
                    break
                if name == "random":
                    measured.append(rng.choice(left))
                    continue
                model = Emulator(target, sources, sim_cols=sim_cols) if name == "emulator" else TargetOnlyGP(target)
                model.fit(df[df.vid.isin(measured)])
                cand = df[df.vid.isin(left)]
                mu, sd = model.predict(cand)
                lcb = per_variant(cand, mu - 1.0 * sd)
                measured.append(lcb.idxmin())
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--bo-steps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "rehearsal")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df, gpus, noise = build_table()
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "table.csv", index=False)
    print(f"{df.vid.nunique()} variants x {df.wid.nunique()} workloads on {gpus}; run-to-run noise (log sd): {noise}")

    summary = {}
    for target in gpus:
        sources = [g for g in gpus if g != target]
        print(f"\n=== target {target}, cheap sources {sources} + maths model + {analytic.GPUS[target]['arch']} compiled-code features")
        sim_cols = df.attrs.get("sim_cols", [])
        r = rehearse(df, target, sources, budgets=[3, 5, 10, 20], repeats=args.repeats, rng=rng, sim_cols=sim_cols)
        r.to_csv(args.out / f"rehearsal_{target}.csv", index=False)
        tab = r.groupby(["n", "method"])[["rho", "regret"]].median().round(3)
        print(tab.to_string())
        b = bo_race(df, target, sources, steps=args.bo_steps, repeats=args.repeats, rng=rng, sim_cols=sim_cols)
        b.to_csv(args.out / f"bo_race_{target}.csv", index=False)
        race = b.groupby(["evals", "method"]).regret.mean().unstack().round(3)
        print("\nmean regret (how much slower than the true best) after k target measurements:")
        print(race.loc[[k for k in (3, 5, 8, 10, 15, 20) if k in race.index]].to_string())
        summary[target] = dict(rehearsal=tab.reset_index().to_dict("records"), race=race.reset_index().to_dict("records"))
    (args.out / "summary.json").write_text(json.dumps(dict(noise=noise, gpus=gpus, results=summary), indent=1, default=float))


if __name__ == "__main__":
    main()
