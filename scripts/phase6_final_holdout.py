"""Phase 6 (step 23): ONE run on the untouched final holdout (2023-01..2025-12).

Region-level (10 original features, 3 classes, MAvA) and gridded exceedance
(ex3/ex4, PR-AUC/Brier). Models are tuned only on data before the holdout.
Running this script again after looking at the results would break the
holdout; the output folder records a timestamp of the first run.

    python ../scripts/phase6_final_holdout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import catalog, config, features as F, gridmodels as GM, harness

OUT = config.RESULTS / "phase6_final"
OUT.mkdir(parents=True, exist_ok=True)
HOLD = "2023-01-01"


def main():
    stamp = OUT / "FIRST_RUN.txt"
    if stamp.exists():
        print("WARNING: holdout already evaluated at", stamp.read_text().strip())
    else:
        stamp.write_text(pd.Timestamp.now().isoformat())

    if "--grid-only" in sys.argv:          # used once to recover from a file-save crash
        return grid_part()

    # ---- region-level, original features ----
    reg =pd.read_parquet(config.PROCESSED / "region_dataset_v0.1.parquet").dropna(subset=F.FEATURES)
    cat = catalog.in_box(catalog.load(), config.CENTRAL_OK)
    print("Region-level, final holdout 2023-2025")
    s, p, ch = harness.evaluate(reg, F.FEATURES, gap=18, holdout_start=HOLD, final=True, cat=cat)
    s.to_csv(OUT / "region_holdout_summary.csv", index=False)
    p.to_csv(OUT / "region_holdout_predictions.csv", index=False)
    ch.to_csv(OUT / "region_holdout_tuned.csv", index=False)
    grid_part()


def grid_part():
    # ---- gridded exceedance ----
    g =pd.read_parquet(config.PROCESSED / "grid_dataset_v0.1.parquet")
    models = ["clim", "persist", "gr", "si", "hybrid", "logreg", "lgbm", "lgbm_smote", "lgbm_nosi"]
    for target in ("ex3", "ex4"):
        pred, chosen = GM.walk_forward(g, target=target, holdout_start=HOLD, final=True)
        final_model = pred.attrs.pop("final_model", None)
        final_X = pred.attrs.pop("final_X", None)
        pred.to_parquet(OUT / f"grid_holdout_pred_{target}.parquet")
        sc = GM.score(pred, models)
        sc.to_csv(OUT / f"grid_holdout_scores_{target}.csv", index=False)
        GM.alarm_table(pred, models).to_csv(OUT / f"grid_holdout_alarms_{target}.csv", index=False)
        print(f"\n{target}\n", sc.round(4).to_string())
        if final_model is not None:
            imp = GM.shap_importance(final_model, final_X)
            imp.to_csv(OUT / f"shap_importance_{target}.csv", header=["mean_abs_shap"])
            print(imp.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
