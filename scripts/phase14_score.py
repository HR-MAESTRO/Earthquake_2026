"""Score the free-background package ETAS against our gridded ETAS, our full
ETAS and the uniform-background package ETAS, on identical rows.

Post hoc validation on the development basins (lag-0 prediction files; ETAS uses
no injection data, so the lag does not affect it). bw_sq = 1 km^2 is primary and
25 km^2 is a sensitivity setting. "Stable" drops, per bandwidth, any test year
whose fit has rho at its 5.0 limit, a at its 0.01 limit, a > 8 or log10_k0 > 4.

    python ../scripts/phase14_score.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from induced import config, etas, multibasin as MB

R = config.RESULTS
OUT = R / "phase14_etas_freebg"
BW = (1, 25)
LAMS = {"full": "lam_full", "pkg": "lam_pkg", "fb1": "lam_fb1", "fb25": "lam_fb25"}
PAIRS = [("etas", "fb1"), ("etas", "fb25"), ("fb1", "pkg"), ("fb25", "pkg"), ("fb1", "fb25")]


def flagged_years(basin, bw):
    p = pd.read_csv(OUT / f"params_{basin}_bw{bw}.csv")
    bad = (p.rho >= 4.99) | (p.a <= 0.011) | (p.a > 8) | (p.log10_k0 > 4)
    return set(p.year[bad].astype(int))


def load(basin):
    d = pd.read_parquet(R / "phase9" / f"pred_lag0_{basin}.parquet")
    n = len(d)
    d = d.merge(pd.read_parquet(R / "phase9" / f"etas_full_{basin}.parquet"), on=["start", "cell"])
    d = d.merge(pd.read_parquet(R / "phase11_etas_package" / f"lam_pkg_{basin}.parquet"), on=["start", "cell"])
    for bw in BW:
        f = pd.read_parquet(OUT / f"lam_{basin}_bw{bw}.parquet").rename(columns={"lam_freebg": f"lam_fb{bw}"})
        d = d.merge(f, on=["start", "cell"])
    assert len(d) == n, (basin, n, len(d))
    for m, c in LAMS.items():
        for mag, t in ((3, "ex3"), (4, "ex4")):
            d[f"{m}_{t}"] = etas.exceed_prob(d[c].values, d.b.values, mag)
    return d.sort_values(["start", "cell"]).reset_index(drop=True)


def evaluate(d, basin, subset):
    models = ["etas", "full", "pkg", "fb1", "fb25"]
    lam = dict(etas="lam_etas", **LAMS)
    rows, diffs = [], []
    for t in ("ex3", "ex4"):
        s = MB.score(d, t, models=models)
        for r in s.to_dict("records"):
            c = lam[r["model"]]
            r.update(basin=basin, subset=subset, count_ratio=d[c].sum() / d.n_mc.sum(),
                     poisson_ll=MB.poisson_ll(d, c), spearman_vs_gridded=spearmanr(d[c], d.lam_etas)[0],
                     pred_over_obs=d[f"{r['model']}_{t}"].sum() / d[t].sum())
            rows.append(r)
        for a, b in PAIRS:
            x, lo, hi = MB.paired_diff(d, t, a, b)
            diffs.append(dict(basin=basin, subset=subset, target=t, a=a, b=b, diff=x, lo95=lo, hi95=hi))
    return rows, diffs


def main():
    rows, diffs = [], []
    for basin in ("oklahoma", "delaware"):
        d = load(basin)
        yr = pd.DatetimeIndex(d.start).year
        r, x = evaluate(d, basin, "all")
        rows += r
        diffs += x
        flag = {bw: flagged_years(basin, bw) for bw in BW}
        print(basin, "flagged years", flag, flush=True)
        keep = ~yr.isin(sorted(flag[1] | flag[25]))
        if d.ex3[keep].sum() >= 20:
            r, x = evaluate(d[keep].reset_index(drop=True), basin, "stable_both")
            rows += r
            diffs += x
    s, p = pd.DataFrame(rows), pd.DataFrame(diffs)
    s.to_csv(OUT / "scores.csv", index=False)
    p.to_csv(OUT / "paired.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(s[["basin", "subset", "target", "model", "n", "positives", "prauc", "prauc_lo", "prauc_hi", "bss", "count_ratio",
             "pred_over_obs", "poisson_ll", "spearman_vs_gridded"]].round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
