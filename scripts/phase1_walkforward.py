"""Phase 1: walk-forward evaluation of the original 10-feature setup.

Original window 2009-08..2019-12. Holdout = 2018-2019 (not touched here).
Walk-forward test years 2013..2017, 9-month gap, nested tuning.

    python ../scripts/phase1_walkforward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import catalog, config, features as F, harness

OUT = config.RESULTS / "phase1"
OUT.mkdir(parents=True, exist_ok=True)


def main(df=None, out=None):
    """df: optional dataset (e.g. the original 2023 input matrix); out: output folder."""
    global OUT
    if out is not None:
        OUT = Path(out)
        OUT.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_parquet(config.PROCESSED / "orig_dataset.parquet")
    df = df.dropna(subset=F.FEATURES).reset_index(drop=True)
    cat = catalog.in_box(catalog.load(), config.CENTRAL_OK)
    print("Walk-forward 2013-2017, gap 9 months, holdout 2018+ excluded")
    summ, preds, chosen = harness.evaluate(df, F.FEATURES, first_test_year=2013, last_test_year=2017,
                                           gap=18, holdout_start="2018-01-01", cat=cat)
    summ.to_csv(OUT / "walkforward_summary.csv", index=False)
    preds.to_csv(OUT / "walkforward_predictions.csv", index=False)
    chosen.to_csv(OUT / "walkforward_tuned_params.csv", index=False)
    # per-year MAvA
    from induced import metrics
    preds["year"] = pd.DatetimeIndex(preds.start).year
    py = preds.groupby(["model", "year"]).apply(lambda g: metrics.mava(g.y, g.pred)).unstack()
    py.round(3).to_csv(OUT / "walkforward_mava_by_year.csv")
    print(py.round(2))

    print("\nNo-gap variant (to measure the effect of the gap)")
    s0, _, _ = harness.evaluate(df, F.FEATURES, first_test_year=2013, last_test_year=2017, gap=0,
                                holdout_start="2018-01-01", cat=cat, verbose=False)
    s0.to_csv(OUT / "walkforward_summary_nogap.csv", index=False)
    print(s0[["model", "mava", "mava_lo", "mava_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
