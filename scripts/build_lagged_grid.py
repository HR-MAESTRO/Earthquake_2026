"""Build one basin's grid with an injection reporting lag.

    python ../scripts/build_lagged_grid.py <basin> <lag_months>
Writes $IEQ_WORK/data/processed/grid_<basin>_lag<L>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import basement, catalog, config, grid, injection, multibasin as MB, regions

STARTS = {"oklahoma": config.ORIG_START, "kansas": "2015-01-01", "delaware": "2017-01-01",
          "midland": "2017-01-01", "fortworth": "2010-01-01"}


def main(basin, lag):
    box = MB.BASINS[basin]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    if basin == "oklahoma":
        cat = catalog.load()
        inj = injection.in_box(injection.swd_only(injection.build()), pad)
        wells = pd.read_parquet(config.PROCESSED / "wells_grid_swd.parquet")
        faults = True
    else:
        cat = regions.comcat(basin)
        inj = injection.in_box(getattr(regions, f"{basin}_injection")(), pad)
        wells = basement.well_table(inj)
        faults = False
    g, _ = grid.build(cat, inj, wells, box, STARTS[basin], "2025-12-31", use_faults=faults,
                      inj_lag_months=lag)
    out = config.PROCESSED / f"grid_{basin}_lag{lag}.parquet"
    g.to_parquet(out)
    print(basin, lag, g.shape, "->", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
