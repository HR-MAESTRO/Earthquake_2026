"""Compare the look-ahead-free development reruns (phase 16) with phase 9.

    python ../scripts/phase16_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import config, multibasin as MB

R = config.RESULTS
OUT = R / "phase16_clean_dev"
ALL4 = ["oklahoma", "kansas", "delaware", "midland"]
MODELS = ["clim", "si", "etas", "lgbm_cls_pooled", "lgbm_cls_lobo", "hybrid_count_pooled", "hybrid_count_lobo",
          "hybrid_count_local", "count_plain_gr", "hybrid_count_emp"]
PAIRS = [("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"), ("hybrid_count_pooled", "hybrid_count_local")]
LAMS = dict(etas="lam_etas", hybrid_count_lobo="lam_hybrid_lobo", hybrid_count_pooled="lam_hybrid_pooled")


def load(version, lag, b):
    p = R / "phase9" / f"pred_lag{lag}_{b}.parquet" if version == "original" else OUT / f"pred_lag{lag}_{b}.parquet"
    return pd.read_parquet(p).sort_values(["start", "cell"]).reset_index(drop=True)


def main():
    pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 400)
    for lag in (0, 3, 12):
        c = json.loads((OUT / f"control_check_lag{lag}.json").read_text())
        print("control lag", lag, {b: (v["rows"], v["rows_original"], max(x for x in v["max_abs"].values() if x == x)) for b, v in c.items()})
    sc, pa, ca = [], [], []
    for lag in (0, 3, 12):
        for b in ALL4:
            for version in ("original", "clean"):
                d = load(version, lag, b)
                for t in ("ex3", "ex4"):
                    s = MB.score(d, t, n_boot=200, models=MODELS)
                    s["basin"], s["lag"], s["version"] = b, lag, version
                    sc.append(s)
                if lag == 12:
                    for x, y in PAIRS:
                        for mode, kw in (("block6", {}), ("year", dict(by_year=True))):
                            di, lo, hi = MB.paired_diff(d, "ex3", x, y, **kw)
                            pa.append(dict(basin=b, version=version, a=x, b=y, resample=mode, diff=di, lo=lo, hi=hi,
                                           sig=bool(lo > 0 or hi < 0)))
                    for m, lc in LAMS.items():
                        sub = d[d[f"{m}_ex3"].notna()]
                        _, k = MB.calibration(sub.ex3.values, sub[f"{m}_ex3"].values)
                        _, k4 = MB.calibration(sub.ex4.values, sub[f"{m}_ex4"].values)
                        ca.append(dict(basin=b, version=version, model=m, pred_obs=k["expected"] / k["observed"],
                                       intercept=k["intercept"], slope=k["slope"],
                                       pred_obs_m4=k4["expected"] / max(k4["observed"], 1),
                                       count_ratio=sub[lc].sum() / sub.n_mc.sum(), ll=MB.poisson_ll(sub, lc)))
    s, p, k = pd.concat(sc), pd.DataFrame(pa), pd.DataFrame(ca)
    s.to_csv(OUT / "scores_compare.csv", index=False)
    p.to_csv(OUT / "paired_compare.csv", index=False)
    k.to_csv(OUT / "calibration_compare.csv", index=False)
    w = s.pivot_table(index=["lag", "basin", "target", "model"], columns="version", values="prauc")
    w["delta"] = w["clean"] - w["original"]
    print(w[["original", "clean", "delta"]].round(3).loc[12].to_string())
    print("\nLag 0 M>=4")
    print(w[["original", "clean"]].round(3).loc[0].xs("ex4", level="target").to_string())
    print("\nLargest absolute PR-AUC change by lag")
    print(w.delta.abs().groupby("lag").max().round(4).to_string())
    print("\nLag 3 and 0 M>=3 hybrid_lobo / si")
    print(w[["original", "clean"]].round(3).xs("ex3", level="target").query("model in ['hybrid_count_lobo','si']").to_string())
    print("\nPaired")
    print(p.round(3).to_string(index=False))
    print("\nCalibration")
    print(k.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
