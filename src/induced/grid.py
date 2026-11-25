"""Phase 2: spatially gridded dataset (cell x half-month).

Each sample is one grid cell in one half-month period. Features use only data
before the period start; labels use events inside the cell during the period.

Feature groups
  seismicity : counts above Mc in cell / neighbourhood (30, 90, 365 d), max M in
               past year, days since last M>=3, Aki ML b-value (2 y, 25 km)
  injection  : distance-weighted volume (exp(-r/L) kernel, r<=R) over 1/3/9/12
               months, cumulative volume since 2006, rate change (3 mo vs prior 6),
               volume-weighted wellhead pressure, volume-weighted basement
               distance, number of active wells
  geology    : basement depth at cell centre, distance to nearest mapped
               basement structure, angle between that structure and SHmax
  physics    : seismogenic-index forecast (Langenbruch & Zoback style) of
               expected M>=3 count in the period, fit on past data only
Labels
  mmax, y (3-class), ex3 = M>=3 occurred, ex4 = M>=4 occurred
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import basement, catalog, config
from .catalog import b_aki
from .features import half_month_periods, magnitude_class

KM_PER_DEG_LAT = 111.2
SHMAX_AZ = 85.0      # regional SHmax azimuth in Oklahoma (deg from N; Alt & Zoback 2017)


def make_cells(box, size_deg=0.2):
    lats = np.arange(box["lat_min"], box["lat_max"] - 1e-9, size_deg)
    lons = np.arange(box["lon_min"], box["lon_max"] - 1e-9, size_deg)
    cells = [(i, la + size_deg / 2, lo + size_deg / 2, la, lo)
             for i, (la, lo) in enumerate((a, b) for a in lats for b in lons)]
    c = pd.DataFrame(cells, columns=["cell", "clat", "clon", "lat0", "lon0"])
    c["size"] = size_deg
    return c


def _dist_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def assign_cell(df, cells, lat="lat", lon="lon"):
    s = cells["size"].iloc[0]
    lat0, lon0 = cells.lat0.min(), cells.lon0.min()
    nlon = cells.lon0.nunique()
    i = np.floor((df[lat] - lat0) / s).astype(int)
    j = np.floor((df[lon] - lon0) / s).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < cells.lat0.nunique()) & (j < nlon)
    out = pd.Series(-1, index=df.index)
    out[ok] = i[ok] * nlon + j[ok]
    return out


def geology(cells, use_faults=True):
    g = cells.copy()
    g["basement_elev_m"] = basement.basement_elev_m(g.clat, g.clon)
    g["surf_elev_m"] = basement.surface_elev_m(g.clat.values, g.clon.values)
    g["basement_depth_m"] = g.surf_elev_m - g.basement_elev_m
    if not use_faults:
        g["fault_dist_km"] = g["fault_shmax_angle"] = g["fault_density"] = np.nan
        return g
    try:
        from shapely.geometry import Point
        fl = basement.faults(dict(lat_min=g.clat.min() - 1, lat_max=g.clat.max() + 1,
                                  lon_min=g.clon.min() - 1, lon_max=g.clon.max() + 1))
        segs = []
        for geom in fl.geometry:
            for line in getattr(geom, "geoms", [geom]):
                xy = np.asarray(line.coords)
                segs += [(xy[k], xy[k + 1]) for k in range(len(xy) - 1)]
        segs = np.array(segs)                       # (n, 2, 2) lon/lat
        mid = segs.mean(1)
        az = (np.degrees(np.arctan2((segs[:, 1, 0] - segs[:, 0, 0]) * np.cos(np.radians(mid[:, 1])),
                                    segs[:, 1, 1] - segs[:, 0, 1])) + 180) % 180
        d_near, ang = [], []
        for la, lo in zip(g.clat, g.clon):
            d = _dist_km(la, lo, mid[:, 1], mid[:, 0])
            k = np.argmin(d)
            d_near.append(d[k])
            a = abs(az[k] - SHMAX_AZ) % 180
            ang.append(min(a, 180 - a))
        g["fault_dist_km"] = d_near
        g["fault_shmax_angle"] = ang      # ~30 deg = optimally oriented for slip
        g["fault_density"] = [(_dist_km(la, lo, mid[:, 1], mid[:, 0]) < 25).sum()
                              for la, lo in zip(g.clat, g.clon)]
    except Exception as e:                                   # pragma: no cover
        print("fault features skipped:", e)
        g["fault_dist_km"] = g["fault_shmax_angle"] = g["fault_density"] = np.nan
    return g


def monthly_kernel_injection(inj, wells, cells, L_km=10.0, R_km=30.0):
    """Distance-weighted monthly volume, pressure and basement distance per cell."""
    inj = inj[inj.vol_bbl > 0].merge(wells[["api", "d_basement_m"]], on="api", how="left")
    wloc = inj.groupby("api")[["lat", "lon"]].first()
    W = np.zeros((len(cells), len(wloc)))
    for ci, (la, lo) in enumerate(zip(cells.clat, cells.clon)):
        d = _dist_km(la, lo, wloc.lat.values, wloc.lon.values)
        W[ci] = np.where(d <= R_km, np.exp(-d / L_km), 0.0)
    api_idx = pd.Series(np.arange(len(wloc)), index=wloc.index)
    months = pd.date_range(inj.date.min(), inj.date.max(), freq="MS")
    m_idx = pd.Series(np.arange(len(months)), index=months)
    V = np.zeros((len(months), len(wloc)))
    P = np.zeros_like(V)
    D = np.zeros_like(V)
    ii, jj = m_idx[inj.date].values, api_idx[inj.api].values
    vv = inj.vol_bbl.to_numpy(dtype=float, na_value=0.0)
    pp = inj.psi.astype(float).fillna(0).to_numpy(dtype=float)
    dd = inj.d_basement_m.astype(float)
    dd = dd.fillna(dd.median()).to_numpy(dtype=float)
    np.add.at(V, (ii, jj), vv)
    np.add.at(P, (ii, jj), vv * pp)
    np.add.at(D, (ii, jj), vv * dd)
    vol = V @ W.T                                       # months x cells
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = (P @ W.T) / vol
        dbm = (D @ W.T) / vol
    nwell = (V > 0).astype(float) @ (W > 0).astype(float).T
    return months, vol, psi, dbm, nwell


def build(cat, inj, wells, box, start, end, size_deg=0.2, mc=2.5, neigh_km=25.0, use_faults=True,
          inj_lag_months=0):
    """inj_lag_months: months between the end of an injection report month and
    its availability. A forecast issued at period start s may use injection
    months up to month(s) - 1 - inj_lag_months. Catalog data are treated as
    available in real time."""
    cells = make_cells(box, size_deg)
    geo = geology(cells, use_faults)
    periods = half_month_periods(start, end)
    months, vol, psi, dbm, nwell = monthly_kernel_injection(inj, wells, cells)
    cumvol = np.cumsum(vol, axis=0)
    m_pos = {m: i for i, m in enumerate(months)}

    c = catalog.in_box(cat, dict(lat_min=box["lat_min"] - .5, lat_max=box["lat_max"] + .5,
                                 lon_min=box["lon_min"] - .5, lon_max=box["lon_max"] + .5)).copy()
    c["cell"] = assign_cell(c, cells)
    t = c.time.values
    tday = (t - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    # distances events <-> cell centres (for neighbourhood features)
    Dce = np.stack([_dist_km(la, lo, c.lat.values, c.lon.values) for la, lo in zip(cells.clat, cells.clon)])
    mags = c.mag.values
    cellv = c.cell.values

    rows = []
    for _, (s, e) in periods.iterrows():
        sd = (np.datetime64(s) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        ed = (np.datetime64(e) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        k = np.searchsorted(tday, sd)
        k_e = np.searchsorted(tday, ed)
        k30, k90, k365, k730 = (np.searchsorted(tday, sd - d) for d in (30, 90, 365, 730))
        mi = m_pos.get(s.replace(day=1) - pd.DateOffset(months=1 + inj_lag_months))  # last available month
        if mi is None:
            continue
        def vsum(n):
            lo = max(0, mi - n + 1)
            return vol[lo:mi + 1].sum(0)
        v1, v3, v9, v12 = vsum(1), vsum(3), vsum(9), vsum(12)
        v_prev6 = vol[max(0, mi - 8):max(0, mi - 2)].sum(0)
        rate_change = (v3 / 3 + 1) / (v_prev6 / 6 + 1)
        past = slice(k365, k)
        lab_sl = slice(k, k_e)
        for ci in range(len(cells)):
            inc = cellv[past] == ci
            near = Dce[ci, past] <= neigh_km
            m_past_near = mags[past][near]
            mm = mags[k30:k][Dce[ci, k30:k] <= neigh_km]
            m90 = mags[k90:k][Dce[ci, k90:k] <= neigh_km]
            m2y = mags[k730:k][Dce[ci, k730:k] <= neigh_km]
            b, _, nb = b_aki(m2y, mc) if len(m2y) >= 30 else (np.nan, np.nan, 0)
            big = np.where((Dce[ci, :k] <= neigh_km) & (mags[:k] >= 3))[0]
            lab = mags[lab_sl][cellv[lab_sl] == ci]
            mmax = lab.max() if len(lab) else np.nan
            rows.append(dict(
                cell=ci, start=s, end=e,
                n30_near=(mm >= mc).sum(), n90_near=(m90 >= mc).sum(),
                n365_near=(m_past_near >= mc).sum(), n365_cell=(mags[past][inc] >= mc).sum(),
                mmax365_near=m_past_near.max() if len(m_past_near) else 0.0,
                days_since_m3=(sd - tday[big[-1]]) if len(big) else 9999.0,
                b_ml=b,
                v1=v1[ci], v3=v3[ci], v9=v9[ci], v12=v12[ci], vcum=cumvol[mi, ci],
                v_rate_change=rate_change[ci], psi_w=psi[mi, ci], d_basement_w=dbm[mi, ci],
                n_wells=nwell[mi, ci],
                mmax=mmax, y=magnitude_class(mmax),
                ex3=int(np.isfinite(mmax) and mmax >= 3), ex4=int(np.isfinite(mmax) and mmax >= 4)))
    df = pd.DataFrame(rows).merge(
        geo[["cell", "clat", "clon", "basement_depth_m", "fault_dist_km", "fault_shmax_angle",
             "fault_density"]], on="cell")
    df = add_seismogenic_index(df, mc=mc)
    return df.sort_values(["start", "cell"]).reset_index(drop=True), geo


def add_seismogenic_index(df, mc=2.5, b_default=1.0, lookback_periods=72):
    """Seismogenic-index style forecast per cell, using past data only.

    Sigma_c = log10(N_c(>=Mc)) - log10(V_c) + b*Mc, estimated from the cell's
    previous `lookback_periods` half-months (counts in-cell, kernel volume).
    Expected M>=3 count in the next period:  10^(Sigma_c - b*3) * V_period,
    with V_period approximated by last month's kernel volume / 2.
    """
    df = df.sort_values(["cell", "start"]).copy()
    b = df["b_ml"].fillna(b_default).clip(0.6, 1.6)
    # counts above Mc that occurred in-cell per period: approximate from labels
    # (count >= Mc not stored); use n30_near difference-free proxy: ex3 count
    cnt = df.groupby("cell")["n365_cell"].shift(0)       # past-year in-cell count (already past-only)
    vol_year = df["v12"].replace(0, np.nan)
    sigma = np.log10(cnt + 0.5) - np.log10(vol_year) + b * mc
    df["si_sigma"] = sigma
    v_period = df["v1"] / 2.0
    lam3 = 10 ** (sigma - b * 3.0) * v_period
    lam4 = 10 ** (sigma - b * 4.0) * v_period
    df["si_lam3"] = lam3.fillna(0)
    df["si_p3"] = 1 - np.exp(-df["si_lam3"])
    df["si_p4"] = 1 - np.exp(-lam4.fillna(0))
    return df


GRID_FEATURES = ["n30_near", "n90_near", "n365_near", "n365_cell", "mmax365_near", "days_since_m3",
                 "b_ml", "v1", "v3", "v9", "v12", "vcum", "v_rate_change", "psi_w", "d_basement_w",
                 "n_wells", "basement_depth_m", "fault_dist_km", "fault_shmax_angle", "fault_density",
                 "si_sigma", "si_lam3"]
