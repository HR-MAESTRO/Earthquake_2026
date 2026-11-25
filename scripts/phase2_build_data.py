"""Phase 2: extended, versioned datasets.

1. Region-level 10-feature dataset extended to 2025-12 (comparable with 2023 paper)
2. Catalog quality report (Mc through time, magnitude sources, declustering)
3. Gridded cell x half-month dataset over north-central Oklahoma

    python ../scripts/phase2_build_data.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import basement, catalog, config, features as F, grid, injection

OUT = config.RESULTS / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
VERSION = "v0.1"
GRID_BOX = dict(lat_min=35.0, lat_max=37.0, lon_min=-99.0, lon_max=-96.0)
END = "2025-12-31"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    cat = catalog.load()
    inj_all = injection.build()
    inj = injection.swd_only(inj_all)

    # ---- 1. extended region-level dataset ----
    wells_c = pd.read_parquet(config.PROCESSED / "wells_central_swd.parquet")
    ext = F.build_dataset(cat, inj, wells_c, config.CENTRAL_OK, config.ORIG_START, END)
    ext.to_parquet(config.PROCESSED / f"region_dataset_{VERSION}.parquet")
    print("extended region dataset", ext.shape, "classes by year:")
    print(ext.groupby(ext.start.dt.year).y.apply(lambda s: np.bincount(s, minlength=3)).to_string())

    # ---- 2. catalog quality ----
    big = dict(lat_min=34.5, lat_max=37.5, lon_min=-99.5, lon_max=-95.5)
    c = catalog.in_box(cat, big)
    mc = catalog.mc_through_time(c[c.time >= "2009"], "QE")
    mc.to_csv(OUT / "catalog_mc_by_quarter.csv", index=False)
    srcs = c.groupby(c.time.dt.year).magnitude_source.value_counts().unstack(fill_value=0)
    srcs.to_csv(OUT / "catalog_magnitude_sources_by_year.csv")
    c2 = c[(c.mag >= 2.5) & (c.time >= "2009")]
    main_mask = catalog.decluster_gk(c2.reset_index(drop=True))
    dec = pd.DataFrame(dict(year=c2.time.dt.year.values, main=main_mask.values))
    dec.groupby("year").main.mean().rename("mainshock_fraction_M2.5").to_csv(OUT / "declustering_fraction.csv")
    print("Mc by quarter (tail):\n", mc.tail(12).round(2).to_string())

    # ---- 3. gridded dataset ----
    pad = dict(lat_min=GRID_BOX["lat_min"] - .3, lat_max=GRID_BOX["lat_max"] + .3,
               lon_min=GRID_BOX["lon_min"] - .3, lon_max=GRID_BOX["lon_max"] + .3)
    inj_g = injection.in_box(inj, pad)
    wells_g = basement.well_table(inj_g)
    wells_g.to_parquet(config.PROCESSED / "wells_grid_swd.parquet")
    g, geo = grid.build(cat, inj_g, wells_g, GRID_BOX, config.ORIG_START, END)
    g.to_parquet(config.PROCESSED / f"grid_dataset_{VERSION}.parquet")
    geo.to_csv(OUT / "grid_cells_geology.csv", index=False)
    print("grid dataset", g.shape, "ex3 rate", g.ex3.mean().round(4), "ex4 rate", g.ex4.mean().round(5),
          "classes", np.bincount(g.y))

    manifest = dict(
        version=VERSION, built=pd.Timestamp.now().isoformat(), end=END, grid_box=GRID_BOX,
        sources=dict(
            ogs_catalog=dict(url="https://ogsweb.ou.edu/eq_catalog/earthquake?start=200001010000&end=202610292359&mag=0&format=csv",
                             sha256_16=sha(config.OGS_CATALOG)),
            occ_uic="https://oklahoma.gov/occ/divisions/oil-gas/oil-gas-data.html (2006-2010 1012A annual; 2011-2025 monthly)",
            basement="Marshak et al. 2016, Illinois Data Bank doi:10.13012/B2IDB-7546972_V1 (CC0)",
            dem="USGS 3DEP ImageServer exportImage, ~250 m, EPSG:4326"),
        files={p.name: sha(p) for p in config.PROCESSED.glob(f"*{VERSION}.parquet")})
    (OUT / f"manifest_{VERSION}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
