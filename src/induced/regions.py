"""Phase 5: loaders for transfer regions, in the same schema as Oklahoma.

Kansas  : KGS Kansas Fluid Injection Database archive (KCC data), monthly per well
          https://www.kgs.ku.edu/Magellan/Qualified/class2_db.html
Delaware: RRC Form H-10 monthly volumes + RRC UIC well locations, Texas Open
          Data Portal (data.texas.gov datasets qq2j-f2zm, givw-z9t4)
Catalogs: USGS ComCat (includes KGS/ISMP and TexNet contributions)

Injection schema: api, lat, lon, date, year, month, vol_bbl, psi,
                  total_depth_ft, inj_bot_ft, inj_top_ft, packer_ft, formation, well_type
Catalog schema:   time, lat, lon, mag, depth, mag_type, net
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import config

KANSAS_BOX = dict(lat_min=37.0, lat_max=38.0, lon_min=-99.0, lon_max=-97.0)
DELAWARE_BOX = dict(lat_min=31.0, lat_max=32.4, lon_min=-104.6, lon_max=-103.0)


def comcat(prefix):
    files = sorted(glob.glob(str(config.RAW / "comcat" / f"{prefix}_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files if pd.read_csv(f, nrows=1).shape[0] >= 0],
                   ignore_index=True)
    df = df[df["type"] == "earthquake"]
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    out = df.rename(columns={"latitude": "lat", "longitude": "lon", "magType": "mag_type"})
    out = out[["time", "lat", "lon", "mag", "depth", "mag_type", "net"]].copy()
    for c in ("lat", "lon", "mag", "depth"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out = out.dropna(subset=["mag", "lat", "lon"])
    return out.sort_values("time").reset_index(drop=True)


def kansas_injection(swd_only=True):
    cache = config.PROCESSED / "kansas_injection_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["API", "LATITUDE", "LONGITUDE", "YEAR", "MONTH", "FLUID_INJECTED", "MAX_FLUID_PRESSURE",
            "INJECTION_ZONE", "WELL_DEPTH", "STATUS"]
    parts = []
    for f in [config.RAW / "kansas" / "KS_UIC_archive.txt", config.RAW / "kansas" / "KS_UIC_archive2025.txt"]:
        for ch in pd.read_csv(f, usecols=cols, chunksize=500_000, low_memory=False):
            ch = ch[(ch.LATITUDE.between(36.5, 38.5)) & (ch.LONGITUDE.between(-99.5, -96.5))]
            parts.append(ch)
    d = pd.concat(parts).drop_duplicates(["API", "YEAR", "MONTH"], keep="last")
    if swd_only:
        d = d[d.STATUS.astype(str).str.upper().str.contains("SWD")]
    out = pd.DataFrame(dict(
        api=d.API.astype(str), lat=pd.to_numeric(d.LATITUDE, errors="coerce"),
        lon=pd.to_numeric(d.LONGITUDE, errors="coerce"), year=d.YEAR.astype(int), month=d.MONTH.astype(int),
        vol_bbl=pd.to_numeric(d.FLUID_INJECTED, errors="coerce").clip(lower=0),
        psi=pd.to_numeric(d.MAX_FLUID_PRESSURE, errors="coerce"),
        total_depth_ft=pd.to_numeric(d.WELL_DEPTH, errors="coerce"), inj_bot_ft=np.nan, inj_top_ft=np.nan,
        packer_ft=np.nan, formation=d.INJECTION_ZONE.astype("string"), well_type="SWD"))
    out = out[(out.month >= 1) & (out.month <= 12) & out.year.between(2000, 2026)]
    out["date"] = pd.to_datetime(dict(year=out.year, month=out.month, day=1))
    out.to_parquet(cache)
    return out


def delaware_injection(disposal_only=True):
    return texas_injection("delaware", disposal_only)


def midland_injection(disposal_only=True):
    return texas_injection("midland", disposal_only)


def fortworth_injection(disposal_only=True):
    return texas_injection("fortworth", disposal_only)


def texas_injection(name, disposal_only=True):
    cache = config.PROCESSED / f"{name}_injection_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    loc = pd.read_csv(config.RAW / "texas" / f"uic_locations_{name}.csv", dtype={"uic_number": str})
    h = pd.read_csv(config.RAW / "texas" / f"h10_{name}_2010_2026.csv", dtype={"uic_no": str})
    loc["uic_number"] = loc.uic_number.str.zfill(9)
    h["uic_no"] = h.uic_no.str.zfill(9)
    if disposal_only:
        loc = loc[loc.uic_type_injection.astype(str) == "1"]
    h = h.merge(loc[["uic_number", "latitude_nad83", "longitude_nad83", "prod_casing_pkr_depth"]],
                left_on="uic_no", right_on="uic_number")
    t = pd.to_datetime(h.formatted_date)
    out = pd.DataFrame(dict(
        api=h.uic_no, lat=h.latitude_nad83.astype(float), lon=h.longitude_nad83.astype(float),
        year=t.dt.year, month=t.dt.month, vol_bbl=pd.to_numeric(h.vol_liq, errors="coerce").clip(lower=0),
        psi=pd.to_numeric(h.inj_press_avg, errors="coerce"), total_depth_ft=np.nan,
        inj_bot_ft=pd.to_numeric(h.boz, errors="coerce"), inj_top_ft=pd.to_numeric(h.toz, errors="coerce"),
        packer_ft=pd.to_numeric(h.prod_casing_pkr_depth, errors="coerce"), formation=pd.NA, well_type="SWD"))
    out = out.groupby(["api", "year", "month"], as_index=False).agg(
        dict(lat="first", lon="first", vol_bbl="sum", psi="max", total_depth_ft="first", inj_bot_ft="first",
             inj_top_ft="first", packer_ft="first", formation="first", well_type="first"))
    out["date"] = pd.to_datetime(dict(year=out.year, month=out.month, day=1))
    out["formation"] = out["formation"].astype("string")
    out.to_parquet(cache)
    return out
