"""Exploratory: Midland sensitivity to basement features (915 of 2,213 wells
lack basement data). Re-runs the LOBO hybrid with basement features removed.

    python ../scripts/phase9_midland_basement.py <lag>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import config, multibasin as MB

DROP = ("basement_depth_m", "d_basement_w")


def main(lag):
    MB.FEATS = [f for f in MB.FEATS if f not in DROP]
    MB.PLAIN_FEATS = [f for f in MB.PLAIN_FEATS if f not in DROP]
    data = {b: MB.load_basin(b, lag) for b in ["oklahoma", "kansas", "delaware", "midland"]}
    p = MB.run("midland", data, ["oklahoma", "kansas", "delaware", "midland"], MB.BASINS["midland"]["test"],
               local_ml=False, lobo=True, ablation=False)
    p.to_parquet(config.RESULTS / "phase9" / f"pred_lag{lag}_midland_nobasement.parquet")
    s = MB.score(p, "ex3", n_boot=500, models=["etas", "si", "hybrid_count_lobo", "hybrid_count_pooled"])
    s.to_csv(config.RESULTS / "phase9" / "tables" / f"midland_nobasement_lag{lag}.csv", index=False)
    print(s.round(4).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]))
