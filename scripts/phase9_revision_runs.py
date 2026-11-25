"""Revision runs (exploratory unless stated): reporting lag x basin, with
pooled, leave-one-basin-out (LOBO) and local ML, plus ablation models.

    python ../scripts/phase9_revision_runs.py <lag_months>
Writes $IEQ_WORK/results/phase9/pred_lag<L>_<basin>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from induced import config, multibasin as MB

OUT = config.RESULTS / "phase9"
OUT.mkdir(parents=True, exist_ok=True)
ALL4 = ["oklahoma", "kansas", "delaware", "midland"]


def main(lag):
    data = {b: MB.load_basin(b, lag) for b in ALL4}
    for b in ALL4:
        print(f"=== lag {lag} {b} ===", flush=True)
        pred = MB.run(b, data, ALL4, MB.BASINS[b]["test"], local_ml=True, lobo=True, ablation=True)
        pred.to_parquet(OUT / f"pred_lag{lag}_{b}.parquet")


if __name__ == "__main__":
    main(int(sys.argv[1]))
