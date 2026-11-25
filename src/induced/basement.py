"""Distance from injection-well bottom to the crystalline basement.

Basement surface: Marshak, Domrois, Abert & Larson (2016), "DEM of the Great
Unconformity, USA cratonic platform", Illinois Data Bank,
doi:10.13012/B2IDB-7546972_V1 (CC0). Raster pc52_clp, 500 m cells,
EPSG:26914, elevation of the Precambrian surface in FEET relative to sea level.

Surface elevation at each well: USGS Elevation Point Query Service (3DEP),
cached locally.

Sign convention follows the 2023 paper (Table 2 values are about -400 m):
    d_i = well_bottom_depth - basement_depth    [m]
so negative = well bottom is above the basement.

The 2023 paper used Weingarten et al. (2015) basement distances; this is an
independent reconstruction, so expect differences of tens to hundreds of m.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from . import config

FT = 0.3048
GDB = config.RAW / "basement" / "pcmap.gdb"
RASTER = f'OpenFileGDB:"{GDB.as_posix()}":pc52_clp'
FAULTS_LAYER = "structure"
ELEV_CACHE = config.PROCESSED / "surface_elevation_cache.json"


def basement_elev_m(lat, lon):
    """Precambrian surface elevation (m above sea level) at points."""
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(RASTER) as r:
        xs, ys = transform("EPSG:4326", r.crs, list(np.asarray(lon, float)),
                           list(np.asarray(lat, float)))
        vals = np.array([v[0] for v in r.sample(zip(xs, ys))], float)
        vals[(vals > 1e30) | (vals < -1e30)] = np.nan
    return vals * FT


def _epqs(lat, lon):
    url = "https://epqs.nationalmap.gov/v1/json"
    for _ in range(3):
        try:
            r = requests.get(url, params=dict(x=lon, y=lat, units="Meters", wkid=4326,
                                              includeDate="false"), timeout=30)
            v = float(r.json()["value"])
            if v > -1000:
                return v
        except Exception:
            pass
    return np.nan


DEM = config.RAW / "dem" / "ok_dem_250m.tif"


def surface_elev_m(lat, lon):
    """Ground elevation (m) from a statewide USGS 3DEP grid (~250 m cells),
    exported from elevation.nationalmap.gov ImageServer. Falls back to the
    point service for wells outside the grid."""
    import rasterio
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    v = np.full(len(lat), np.nan)
    for path in [DEM] + sorted(p for p in DEM.parent.glob("*.tif") if p != DEM):
        todo = np.isnan(v)
        if not todo.any():
            break
        with rasterio.open(path) as r:
            b = r.bounds
            inside = todo & (lon > b.left) & (lon < b.right) & (lat > b.bottom) & (lat < b.top)
            if inside.any():
                v[inside] = [x[0] for x in r.sample(zip(lon[inside], lat[inside]))]
    v[(v < -500) | (v > 5000)] = np.nan
    bad = np.isnan(v)
    if bad.any():
        v[bad] = surface_elev_points(np.asarray(lat)[bad], np.asarray(lon)[bad])
    return v


def surface_elev_points(lat, lon, workers=8):
    cache = json.loads(ELEV_CACHE.read_text()) if ELEV_CACHE.exists() else {}
    keys = [f"{a:.5f},{b:.5f}" for a, b in zip(lat, lon)]
    todo = sorted({k for k in keys if k not in cache})
    if todo:
        pts = [tuple(map(float, k.split(","))) for k in todo]
        with ThreadPoolExecutor(workers) as ex:
            vals = list(ex.map(lambda p: _epqs(*p), pts))
        for k, v in zip(todo, vals):
            cache[k] = None if np.isnan(v) else v
        ELEV_CACHE.write_text(json.dumps(cache))
    return np.array([np.nan if cache.get(k) is None else cache[k] for k in keys], float)


def well_table(inj, bottom="inj_bottom"):
    """One row per well: location, bottom depth, basement distance (m).

    bottom='inj_bottom': bottom of injection interval, else packer, else TD
    (default; closest to the paper's Table 2 scale).
    bottom='total_depth': TD, else injection bottom, else packer.
    """
    w = (inj.sort_values("year").groupby("api")
         .agg(lat=("lat", "last"), lon=("lon", "last"),
              total_depth_ft=("total_depth_ft", "last"), inj_bot_ft=("inj_bot_ft", "last"),
              packer_ft=("packer_ft", "last"), formation=("formation", "last"),
              well_type=("well_type", "last"))
         .dropna(subset=["lat", "lon"]).reset_index())
    td = w["total_depth_ft"].where(w["total_depth_ft"] > 0)
    ib = w["inj_bot_ft"].where(w["inj_bot_ft"] > 0)
    pk = w["packer_ft"].where(w["packer_ft"] > 0)
    order = [ib, pk, td] if bottom == "inj_bottom" else [td, ib, pk]
    bottom_ft = order[0].fillna(order[1]).fillna(order[2])
    w["bottom_m"] = bottom_ft * FT
    w["surf_elev_m"] = surface_elev_m(w["lat"].values, w["lon"].values)
    w["basement_elev_m"] = basement_elev_m(w["lat"].values, w["lon"].values)
    w["basement_depth_m"] = w["surf_elev_m"] - w["basement_elev_m"]
    w["d_basement_m"] = w["bottom_m"] - w["basement_depth_m"]
    return w


OGS_FAULTS = config.RAW / "ogs_faults" / "OF2-2016_Supplement" / "Interpretative.shp"


def faults(bbox=None, source="ogs"):
    """Mapped faults as a GeoDataFrame in EPSG:4326.

    source='ogs': Marsh & Holland (2016) OGS OF2-2016 interpretive fault map
    (5,877 faults; https://ou.edu/ogs/data/fault). source='marshak': the 28
    regional basement structures in the Marshak et al. geodatabase.
    """
    import geopandas as gpd
    if source == "ogs":
        gdf = gpd.read_file(OGS_FAULTS)
    else:
        import pyogrio
        gdf = pyogrio.read_dataframe(GDB, layer=FAULTS_LAYER)
    gdf = gdf.to_crs(4326)
    if bbox:
        gdf = gdf.cx[bbox["lon_min"]:bbox["lon_max"], bbox["lat_min"]:bbox["lat_max"]]
    return gdf
