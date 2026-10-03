"""Choose which variants to send to the B200 portal next, and learn from the ones already sent.

Inputs
  everything emulator.build_table() uses (other-GPU timings, sm_100a compiled-code features,
  the analytical B200 model, the approximate-B200 simulator)
  results/b200_portal.csv   one row per portal submission: vid,latency_ms[,sol_score]
                            (latency_ms as shown on the portal's submission page)

Output
  results/b200_plan.csv      every variant with predicted B200 latency, uncertainty and rank
  b200_submit/<vid>.json     the solution files for the suggested batch, ready to upload

With fewer than 3 portal results there is nothing to calibrate against, so the ranking is the
average of the standardised cheap sources. From 3 results on, the same blend + GP correction as
the emulator is fitted per variant and the batch is chosen by lowest optimistic estimate.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import emulator

HERE = Path(__file__).resolve().parent


def per_variant_features(df):
    srcs = [c for c in df.columns if c.startswith("t_")] + ["a_B200"]
    agg = df.groupby("vid")[srcs].mean()  # portal latency is an average over workloads
    for c in df.attrs.get("sim_cols", []):
        agg[c] = df.groupby("vid")[c].first()
    agg = np.log(agg)
    static = df.groupby("vid")[[f"s_B200_{c}" for c in emulator.STATIC] + emulator.KNOBS].first()
    return agg, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--portal", type=Path, default=HERE / "results" / "b200_portal.csv")
    args = ap.parse_args()

    df, gpus, _ = emulator.build_table()
    src, static = per_variant_features(df)
    measured = pd.read_csv(args.portal).groupby("vid").latency_ms.mean() if args.portal.exists() else pd.Series(dtype=float)
    measured = measured[measured.index.isin(src.index)]
    print(f"cheap sources: {list(src.columns)}; B200 portal results: {len(measured)}")

    if len(measured) < 3:
        z = (src - src.mean()) / src.std()
        mu, sd = z.mean(axis=1), pd.Series(1.0, index=z.index)
        how = "average of standardised cheap sources (no B200 data yet)"
    else:
        from scipy.optimize import nnls

        y = np.log(measured)
        xs = src.loc[y.index]
        scale = xs.std() + 1e-12
        coef, _ = nnls(((xs - xs.mean()) / scale).to_numpy(), (y - y.mean()).to_numpy())  # non-negative blend
        coef = pd.Series(coef, index=src.columns) / scale
        blend = lambda x: y.mean() + (x - xs.mean()) @ coef  # noqa: E731
        resid = y - blend(xs)
        print("blend weights:", coef.round(3).to_dict())
        x = static.to_numpy(float)
        x = np.nan_to_num((x - x.mean(0)) / (x.std(0) + 1e-9))
        xi = pd.DataFrame(x, index=static.index)
        gp = emulator._gp(x.shape[1]).fit(xi.loc[y.index], resid)
        r, s = gp.predict(xi.loc[src.index], return_std=True)
        mu = pd.Series(blend(src).to_numpy() + r, index=src.index)
        sd = pd.Series(s, index=src.index)
        how = f"blend + GP fitted on {len(measured)} B200 results"

    plan = pd.DataFrame({"pred_log": mu, "sd": sd})
    if len(measured) >= 3:
        plan["pred_latency_ms"] = np.exp(plan.pred_log)
    plan["measured_latency_ms"] = measured
    plan["lcb"] = plan.pred_log - plan.sd
    plan = plan.sort_values("pred_log")
    plan["rank"] = range(1, len(plan) + 1)
    out = HERE / "results" / "b200_plan.csv"
    plan.join(static[emulator.KNOBS]).to_csv(out)

    nxt = plan[plan.measured_latency_ms.isna()].sort_values("lcb").head(args.batch).index
    sub = HERE / "b200_submit"
    shutil.rmtree(sub, ignore_errors=True)
    sub.mkdir()
    for v in nxt:
        shutil.copy(HERE / "variants" / f"{v}.json", sub / f"{v}.json")
    print(f"ranking: {how}")
    print(plan.head(10).join(static[emulator.KNOBS]).round(3).to_string())
    print(f"\nnext batch for the portal ({args.batch}): {list(nxt)} -> {sub}/")


if __name__ == "__main__":
    main()
