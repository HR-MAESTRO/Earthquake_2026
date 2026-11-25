"""Phase 5: transfer of the Oklahoma model to Kansas and the Delaware Basin (TX).

For each target region and each test year k:
  zero-shot    : LightGBM trained on Oklahoma only (2010-2022, holdout excluded)
  zero-shot+cal: same, isotonic-recalibrated on the target region's years < k
  fine-tuned   : trained on Oklahoma + target region years < k (minus gap)
  local-only   : trained on target region years < k only
  baselines    : climatology (target region, years < k), GR-Poisson, SI physics
Fault features are excluded everywhere (OGS fault map covers Oklahoma only).

    python ../scripts/phase5_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from induced import basement, config, grid, gridmodels as GM, injection, metrics, regions

OUT = config.RESULTS / "phase5"
OUT.mkdir(parents=True, exist_ok=True)
FEATS = [f for f in grid.GRID_FEATURES if not f.startswith("fault")]
GAP = 18


def build_region(name, box, inj, cat, start):
    path = config.PROCESSED / f"grid_{name}_v0.1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    inj = injection.in_box(inj, pad)
    wells = basement.well_table(inj)
    g, geo = grid.build(cat, inj, wells, box, start, "2025-12-31", use_faults=False)
    g.to_parquet(path)
    geo.to_csv(OUT / f"cells_{name}.csv", index=False)
    return g


def fit(X, y, seed=0):
    return GM._fit_ml("lgbm", X, y, seed, dict(num_leaves=15, n_estimators=300, min_child_samples=50))


def main():
    ok = pd.read_parquet(config.PROCESSED / "grid_dataset_v0.1.parquet")
    ok = ok[(ok.start < "2023-01-01") & (ok.start >= "2010-01-01")]
    Xok, yok_all = GM.prep(ok, FEATS), ok
    targets = {
        "kansas": (regions.KANSAS_BOX, regions.kansas_injection, "kansas", "2015-01-01", 2017),
        "delaware": (regions.DELAWARE_BOX, regions.delaware_injection, "delaware", "2017-01-01", 2019),
    }
    allrows = []
    for name, (box, inj_fn, cc, start, first_test) in targets.items():
        print(f"\n=== {name} ===")
        g = build_region(name, box, inj_fn(), regions.comcat(cc), start)
        print("samples", g.shape, "ex3", int(g.ex3.sum()), "ex4", int(g.ex4.sum()))
        g = g.sort_values(["start", "cell"]).reset_index(drop=True)
        Xg = GM.prep(g, FEATS)
        yrs = g.start.dt.year.values
        uniq = np.sort(g.start.unique())
        pidx = np.searchsorted(uniq, g.start.values)
        for target in ("ex3", "ex4"):
            zs = fit(Xok, ok[target].values)
            preds = []
            for k in range(first_test, 2026):
                te = np.where(yrs == k)[0]
                if len(te) == 0:
                    continue
                tr = np.where(pidx < pidx[te].min() - GAP)[0]
                ytr, yte = g[target].values[tr], g[target].values[te]
                r = pd.DataFrame(dict(start=g.start.values[te], y=yte))
                r["zero_shot"] = GM._predict(zs, Xg.iloc[te])
                if ytr.sum() > 0:
                    iso = IsotonicRegression(out_of_bounds="clip").fit(GM._predict(zs, Xg.iloc[tr]), ytr)
                    r["zero_shot_cal"] = iso.predict(r["zero_shot"])
                else:
                    r["zero_shot_cal"] = r["zero_shot"]
                if ytr.sum() >= 5:
                    Xft = pd.concat([Xok, Xg.iloc[tr]])
                    yft = np.r_[ok[target].values, ytr]
                    r["fine_tuned"] = GM._predict(fit(Xft, yft), Xg.iloc[te])
                    r["local_only"] = GM._predict(fit(Xg.iloc[tr], ytr), Xg.iloc[te])
                else:
                    r["fine_tuned"] = r["zero_shot"]
                    r["local_only"] = ytr.mean() if len(ytr) else 0.0
                rate = pd.Series(ytr).groupby(g.cell.values[tr]).mean()
                r["clim"] = pd.Series(g.cell.values[te]).map(rate).fillna(ytr.mean() if len(ytr) else 0).values
                r["si"] = g["si_p3" if target == "ex3" else "si_p4"].values[te]
                preds.append(r)
            p = pd.concat(preds, ignore_index=True)
            p.to_parquet(OUT / f"pred_{name}_{target}.parquet")
            sc = GM.score(p, ["clim", "si", "zero_shot", "zero_shot_cal", "fine_tuned", "local_only"], n_boot=500)
            sc.insert(0, "target", target)
            sc.insert(0, "region", name)
            allrows.append(sc)
            print(sc[["target", "model", "positives", "prauc", "prauc_lo", "prauc_hi", "brier", "bss_vs_const"]]
                  .round(4).to_string())
    pd.concat(allrows).to_csv(OUT / "transfer_scores.csv", index=False)


if __name__ == "__main__":
    main()
