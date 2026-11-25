"""Phases 3-4: probabilistic exceedance forecasting on the gridded dataset.

Walk-forward test years 2014-2022 (gap 9 months). Holdout 2023-2025 untouched.
    python ../scripts/phase34_grid_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import config, gridmodels as GM

OUT = config.RESULTS / "phase34"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["clim", "persist", "gr", "si", "logreg", "lgbm", "lgbm_smote", "lgbm_nosi"]


def main():
    df = pd.read_parquet(config.PROCESSED / "grid_dataset_v0.1.parquet")
    print("grid", df.shape)
    for target in ("ex3", "ex4"):
        print(f"\n=== {target} ===")
        pred, chosen = GM.walk_forward(df, target=target)
        pred.to_parquet(OUT / f"pred_{target}.parquet")
        chosen.to_csv(OUT / f"tuned_{target}.csv", index=False)
        sc = GM.score(pred, MODELS)
        sc.to_csv(OUT / f"scores_{target}.csv", index=False)
        print(sc.round(4).to_string())
        al = GM.alarm_table(pred, MODELS)
        al.to_csv(OUT / f"alarms_{target}.csv", index=False)
        print(al[al.model.isin(["clim", "gr", "si", "lgbm"])].round(3).to_string())
        # conformal: calibrate on even test years, evaluate coverage on odd years (both out-of-sample)
        yr = pred.start.dt.year
        cal, ev = pred[yr % 2 == 0], pred[yr % 2 == 1]
        sets = GM.conformal_sets(cal.y, cal.lgbm, ev.lgbm.values, alpha=0.1)
        cov = np.mean([str(int(t)) in s for t, s in zip(ev.y.values, sets)])
        cov1 = np.mean([("1" in s) for t, s in zip(ev.y.values, sets) if t == 1])
        pd.DataFrame([dict(target=target, alpha=0.1, coverage=cov, coverage_positive=cov1,
                           frac_singleton_1=np.mean(sets == "{1}"), frac_both=np.mean(sets == "{0,1}"))]
                     ).to_csv(OUT / f"conformal_{target}.csv", index=False)
        print(f"conformal 90%: coverage={cov:.3f}, coverage on positives={cov1:.3f}")

    # secondary task: 3-class, comparable with 2023 paper (MAvA), using ex3/ex4 models
    p3 = pd.read_parquet(OUT / "pred_ex3.parquet")
    p4 = pd.read_parquet(OUT / "pred_ex4.parquet")
    from induced import metrics
    rows = []
    for m in MODELS:
        P4 = p4[m].values
        P3 = np.maximum(p3[m].values, P4)
        proba = np.c_[1 - P3, P3 - P4, P4]
        prior = np.bincount(p3.y3.values, minlength=3) / len(p3)
        pred = np.argmax(proba / prior, 1)
        s = metrics.summary(p3.y3.values, pred, proba, n_boot=300)
        s["model"] = m
        rows.append(s)
    three = pd.DataFrame(rows)
    three.to_csv(OUT / "threeclass_from_exceedance.csv", index=False)
    print(three[["model", "mava", "mava_lo", "mava_hi", "pod2", "far2"]].round(3).to_string())


if __name__ == "__main__":
    main()
