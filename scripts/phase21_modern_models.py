"""Modern comparators on the 2023 task (central Oklahoma, half-months, 3 classes).

    python ../scripts/phase21_modern_models.py [--data PARQUET] [--out DIR] [--no-tabpfn]

--data defaults to the rebuilt dataset; pass results/phase20_original/original_input.parquet
for the original 2023 input matrix. Same walk-forward as phase 1 (test 2013-2017,
9-month gap, nested forward-chaining tuning, 2018+ holdout untouched). Models:
  logreg, svm_rbf, lgbm (as phase 1); ordinal logistic (Frank & Hall); TabPFN
  (pretrained, no tuning); climatology and GR-Poisson baselines; and
  etas_region: the look-ahead-free gridded ETAS (phase 16) summed over 0.2 deg cells
  whose centres fall in the central-Oklahoma box, giving P(M>=3), P(M>=4) per period.
  Gridded ETAS exists from 2014, so all models are also scored on 2014-2017.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import baselines, catalog, config, features as F, grid, harness, metrics, multibasin as MB


def etas_region(periods):
    p = pd.read_parquet(config.RESULTS / "phase16_clean_dev" / "pred_lag12_oklahoma.parquet")
    c = grid.make_cells(MB.BASINS["oklahoma"]["box"])
    b = config.CENTRAL_OK
    inside = c[(c.clat > b["lat_min"]) & (c.clat < b["lat_max"]) & (c.clon > b["lon_min"]) & (c.clon < b["lon_max"])]
    p = p[p.cell.isin(inside.cell)]
    # independent cells: P(any cell exceeds) = 1 - prod(1 - p_c)
    agg = p.groupby("start").agg(p3=("etas_ex3", lambda x: 1 - np.prod(1 - x)),
                                 p4=("etas_ex4", lambda x: 1 - np.prod(1 - x))).reset_index()
    m = periods[["start"]].merge(agg, on="start", how="left")
    p3, p4 = m.p3.values, np.minimum(m.p4.values, m.p3.values)
    return np.c_[1 - p3, p3 - p4, p4], len(inside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default=str(config.RESULTS / "phase21_modern"))
    ap.add_argument("--no-tabpfn", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.data or config.PROCESSED / "orig_dataset.parquet")
    df = df.dropna(subset=F.FEATURES).reset_index(drop=True)
    cat = catalog.in_box(catalog.load(), config.CENTRAL_OK)
    S = harness.DEFAULT_SPECS
    specs = {k: S[k] for k in ("climatology", "gr_poisson", "logreg", "svm_rbf", "lgbm")}
    specs["ordinal"] = ("ml", "ordinal", [("None", p) for p in harness.grid(C=[0.1, 1, 10], class_weight=["balanced"])])
    if not a.no_tabpfn:
        specs["tabpfn"] = ("ml", "tabpfn", [("None", {})])
    summ, preds, chosen = harness.evaluate(df, F.FEATURES, specs=specs, first_test_year=2013, last_test_year=2017,
                                           gap=18, holdout_start="2018-01-01", cat=cat)
    summ.to_csv(out / "walkforward_2013_2017.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    chosen.to_csv(out / "tuned_params.csv", index=False)

    # common window 2014-2017 with the regional ETAS forecast
    per = df[(df.start >= "2014-01-01") & (df.start < "2018-01-01")].reset_index(drop=True)
    proba, ncell = etas_region(per)
    tr = np.where(df.start < "2013-04-01")[0]   # prior for the decision rule: data before the first fold's gap
    prior = np.bincount(df.y.values[tr], minlength=3) / len(tr)
    rows = []
    s = metrics.summary(per.y.values, baselines.decide(proba, prior), proba)
    s["model"] = f"etas_region ({ncell} cells)"
    rows.append(s)
    pw = preds[(preds.start >= "2014-01-01") & (preds.start < "2018-01-01")]
    for name, g in pw.groupby("model", sort=False):
        pr = g[["p0", "p1", "p2"]].values
        s = metrics.summary(g.y.values, g.pred.values, None if np.isnan(pr).any() else pr)
        s["model"] = name
        rows.append(s)
    w = pd.DataFrame(rows)
    cols = ["model", "n", "mava", "mava_lo", "mava_hi", "pod2", "far2", "brier_m3", "prauc_m3", "brier_m4",
            "prauc_m4", "prauc_m4_lo", "prauc_m4_hi", "base_rate_m4"]
    w = w[[c for c in cols if c in w]]
    w.to_csv(out / "window_2014_2017.csv", index=False)
    pd.set_option("display.width", 220)
    txt = ("Walk-forward 2013-2017\n" + summ.round(3).to_string() + "\n\nCommon window 2014-2017 (with regional ETAS)\n"
           + w.round(3).to_string())
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
