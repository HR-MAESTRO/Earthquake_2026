"""OCC Class II UIC injection data -> monthly per-well table.

Sources (oklahoma.gov/occ, Oil & Gas Data Files):
  * all-2006-2010-uic-1012a.xls      annual volume per well (no monthly split)
  * {2011..2025}-uic-injection-volumes.xlsx   monthly volume + pressure per well

2006-2010 annual totals are spread evenly over 12 months. That is a known
deviation from the 2023 paper, which only affects features before ~2011.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from . import config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CACHE = config.PROCESSED / "injection_monthly.parquet"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _load_modern(path):
    df = pd.read_excel(path)
    year = int(re.search(r"(\d{4})-uic", path).group(1))
    form_col = "FormationName" if "FormationName" in df.columns else "PermitFormationName"
    keep = dict(api=_num(df["API"]).astype("Int64"), lat=_num(df["LAT"]), lon=_num(df["LON"]),
                total_depth_ft=_num(df["TotalDepth"]), inj_bot_ft=_num(df["InjBotDepth"]),
                inj_top_ft=_num(df["InjTopDepth"]), packer_ft=_num(df["PackerDepth"]),
                formation=df.get(form_col), well_type=df.get("WellType"))
    base = pd.DataFrame(keep)
    rows = []
    for i, m in enumerate(MONTHS, start=1):
        part = base.copy()
        part["year"], part["month"] = year, i
        part["vol_bbl"] = _num(df[f"{m} Vol"])
        part["psi"] = _num(df[f"{m} PSI"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _load_legacy(path):
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    api = _num(df["API_COUNTY"]) * 100000 + _num(df["API_NUMBER"]) + 3500000000
    base = pd.DataFrame(dict(api=api.astype("Int64"), lat=_num(df["DECIMAL_LAT"]),
                             lon=_num(df["DECIMAL_LONG"]), year=_num(df["YEAR"]),
                             packer_ft=_num(df["PACKERDPTH"]),
                             annual=_num(df["TOTAL VOLUME"])))
    base = base.dropna(subset=["year"])
    rows = []
    for i in range(1, 13):
        part = base.copy()
        part["month"] = i
        part["vol_bbl"] = part["annual"] / 12.0
        part["psi"] = np.nan
        rows.append(part.drop(columns="annual"))
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


def build(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    parts = [_load_legacy(str(config.OCC_DIR / "all-2006-2010-uic-1012a.xls"))]
    for p in sorted(glob.glob(str(config.OCC_DIR / "*-uic-injection-volumes.xlsx"))):
        parts.append(_load_modern(p))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["year"].between(2006, 2026)]
    df["vol_bbl"] = df["vol_bbl"].clip(lower=0)
    # Fill well depth for legacy rows from the most recent modern record
    static = (df.dropna(subset=["total_depth_ft"]).sort_values("year")
                .groupby("api")[["total_depth_ft", "inj_bot_ft", "inj_top_ft", "formation",
                                 "well_type"]].last())
    for c in static.columns:
        df[c] = df[c].where(df[c].notna(), df["api"].map(static[c]))
    # Some operators file duplicate rows for the same well-year; sum volumes.
    agg = dict(vol_bbl="sum", psi="max", lat="first", lon="first", total_depth_ft="first",
               well_type="first",
               inj_bot_ft="first", inj_top_ft="first", packer_ft="first", formation="first")
    df = df.groupby(["api", "year", "month"], as_index=False).agg(agg)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df["formation"] = df["formation"].astype("string")
    df["well_type"] = df["well_type"].astype("string")
    df.to_parquet(CACHE)
    return df


def swd_only(df):
    """Keep disposal wells: any year typed 2D* (2DNC, 2DCm, 2D) or SWD."""
    wt = df["well_type"].fillna("").str.upper()
    is_swd = (wt.str.startswith("2D") | wt.eq("SWD")).groupby(df["api"]).any()
    return df[df["api"].map(is_swd).fillna(False).astype(bool)]


def in_box(df, box, lat="lat", lon="lon"):
    return df[(df[lat] >= box["lat_min"]) & (df[lat] <= box["lat_max"]) &
              (df[lon] >= box["lon_min"]) & (df[lon] <= box["lon_max"])]
