"""Validate the simplified gridded ETAS against a full continuous space-time
ETAS fitted by maximum likelihood (etas_full.py), on the same cells/periods.

    python ../scripts/phase9_etas_validation.py <basin>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import config, etas, multibasin as MB
from induced.etas_full import FullETAS
from induced.grid import make_cells

OUT = config.RESULTS / "phase9"
OUT.mkdir(parents=True, exist_ok=True)


def main(basin):
    g, B, T, cat = MB.load_basin(basin, 0)
    cells = make_cells(MB.BASINS[basin]["box"], 0.2)
    y0, y1 = MB.BASINS[basin]["test"]
    rows, params = [], []
    uniq = np.sort(g.start.unique())
    for yr in range(y0, y1 + 1):
        te = np.where(pd.DatetimeIndex(g.start).year == yr)[0]
        p0 = np.searchsorted(uniq, g.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        model = FullETAS(MB.BASINS[basin]["box"])
        prm = model.fit(cat, cutoff - pd.Timedelta(days=365), cutoff)
        prm.update(basin=basin, year=yr)
        params.append(prm)
        print(basin, yr, {k: round(v, 4) if isinstance(v, float) else v for k, v in prm.items()}, flush=True)
        gy = g.iloc[te]
        for s, e in gy[["start", "end"]].drop_duplicates().itertuples(index=False):
            lam = model.expected_counts(cells, s, e)
            sub = gy[gy.start == s]
            rows.append(pd.DataFrame(dict(start=s, cell=sub.cell.values, lam_full=lam[sub.cell.values])))
    full = pd.concat(rows, ignore_index=True)
    full.to_parquet(OUT / f"etas_full_{basin}.parquet")
    pd.DataFrame(params).to_csv(OUT / f"etas_full_params_{basin}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
