"""OGS earthquake catalog loading and catalog-quality tools.

Source: Oklahoma Geological Survey catalog API
https://ogsweb.ou.edu/eq_catalog/earthquake?start=...&end=...&mag=0&format=csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load(path=None):
    df = pd.read_csv(path or config.OGS_CATALOG, na_values=["None"])
    df["time"] = pd.to_datetime(df["origintime"], errors="coerce")
    df = df.dropna(subset=["time", "magnitude", "latitude", "longitude"])
    df = df.rename(columns={"magnitude": "mag", "latitude": "lat", "longitude": "lon",
                            "depth_km": "depth"})
    return df.sort_values("time").reset_index(drop=True)


def in_box(df, box):
    return df[(df.lat >= box["lat_min"]) & (df.lat <= box["lat_max"]) &
              (df.lon >= box["lon_min"]) & (df.lon <= box["lon_max"])]


# ---------- magnitude of completeness and b-value ----------

def mc_maxc(mags, dm=0.1, correction=0.2):
    """Maximum-curvature Mc (Wiemer & Wyss 2000) plus the usual +0.2 correction."""
    m = np.round(np.asarray(mags) / dm) * dm
    if len(m) < 20:
        return np.nan
    vals, counts = np.unique(m, return_counts=True)
    return float(vals[np.argmax(counts)] + correction)


def b_aki(mags, mc, dm=0.1):
    """Aki (1965) / Utsu maximum-likelihood b-value with binning correction.
    Returns (b, sigma_b Shi & Bolt 1982, n)."""
    m = np.asarray(mags, float)
    m = m[m >= mc - dm / 2]
    n = len(m)
    if n < 10:
        return np.nan, np.nan, n
    mean = m.mean()
    b = np.log10(np.e) / (mean - (mc - dm / 2))
    sigma = 2.3 * b ** 2 * np.sqrt(((m - mean) ** 2).sum() / (n * (n - 1)))
    return float(b), float(sigma), n


def mc_through_time(cat, freq="YE", min_events=50):
    rows = []
    for t, g in cat.groupby(pd.Grouper(key="time", freq=freq)):
        mc = mc_maxc(g["mag"]) if len(g) >= min_events else np.nan
        b, sb, n = b_aki(g["mag"], mc) if np.isfinite(mc) else (np.nan, np.nan, 0)
        rows.append(dict(period=t, n_all=len(g), mc=mc, b_ml=b, b_sigma=sb, n_above_mc=n))
    return pd.DataFrame(rows)


# ---------- declustering: Zaliapin & Ben-Zion (2013) nearest neighbour ----------

def nn_distance(cat, b=1.0, df=1.6):
    """Nearest-neighbour proximity eta_j = min_i t_ij * r_ij^df * 10^(-b m_i), i before j.
    t in years, r in km (epicentral, floored at 0.1 km). Returns (log10 eta, parent index)."""
    t = cat["time"].values.astype("datetime64[s]").astype(np.int64) / (365.25 * 86400)
    lat = np.radians(cat["lat"].values)
    lon = np.radians(cat["lon"].values)
    m = cat["mag"].values
    n = len(cat)
    log_eta = np.full(n, np.nan)
    parent = np.full(n, -1)
    for j in range(1, n):
        i = np.arange(j)
        dt = t[j] - t[i]
        ok = dt > 0
        if not ok.any():
            continue
        i, dt = i[ok], dt[ok]
        a = (np.sin((lat[i] - lat[j]) / 2) ** 2 +
             np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[i] - lon[j]) / 2) ** 2)
        r = np.maximum(6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))), 0.1)
        le = np.log10(dt) + df * np.log10(r) - b * m[i]
        k = np.argmin(le)
        log_eta[j], parent[j] = le[k], i[k]
    return log_eta, parent


def nn_threshold(log_eta):
    """Split the bimodal log10(eta) distribution with a 2-component Gaussian mixture."""
    from sklearn.mixture import GaussianMixture
    x = log_eta[np.isfinite(log_eta)].reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(x)
    mu = np.sort(gm.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 400).reshape(-1, 1)
    p = gm.predict_proba(grid)
    lo = np.argmin(gm.means_.ravel())
    return float(grid[np.argmin(np.abs(p[:, lo] - 0.5)), 0])


def decluster_nn(cat, b=1.0, df=1.6, threshold=None):
    """Background (True) / clustered (False) mask, plus the threshold used."""
    c = cat.sort_values("time").reset_index(drop=True)
    le, _ = nn_distance(c, b, df)
    thr = nn_threshold(le) if threshold is None else threshold
    bg = ~(le < thr)                      # first event and isolated events are background
    return pd.Series(bg, index=cat.sort_values("time").index), thr


# ---------- declustering (Gardner & Knopoff 1974 windows) ----------

def _gk_window(m):
    dist_km = 10 ** (0.1238 * m + 0.983)
    days = np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))
    return dist_km, days


def decluster_gk(cat):
    """Return boolean mask of mainshocks (True) using Gardner-Knopoff windows."""
    c = cat.sort_values("mag", ascending=False)
    t = cat["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
    lat = np.radians(cat["lat"].values)
    lon = np.radians(cat["lon"].values)
    mags = cat["mag"].values
    pos = {ix: i for i, ix in enumerate(cat.index)}
    is_main = np.ones(len(cat), bool)
    order = [pos[ix] for ix in c.index]
    removed = np.zeros(len(cat), bool)
    for i in order:
        if removed[i]:
            continue
        r_km, w_days = _gk_window(mags[i])
        dt = t - t[i]
        cand = np.where((np.abs(dt) <= w_days) & ~removed)[0]
        if len(cand) <= 1:
            continue
        dlat = lat[cand] - lat[i]
        dlon = lon[cand] - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat[cand]) * np.sin(dlon / 2) ** 2
        d = 6371 * 2 * np.arcsin(np.sqrt(a))
        cl = cand[(d <= r_km) & (cand != i) & (mags[cand] <= mags[i])]
        removed[cl] = True
        is_main[cl] = False
    return pd.Series(is_main, index=cat.index)
