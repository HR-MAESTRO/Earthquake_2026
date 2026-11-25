"""ETAS-type rate forecast per grid cell and half-month, plus count targets.

lambda(cell, period) = a * B + K * T(alpha, p)

  B  background: (Zaliapin-Ben-Zion background events M>=Mc in the cell over
     the past 365 d + 0.1) / 365 * period length  (non-stationary background,
     cf. Kumazawa & Ogata 2013)
  T  triggering from all events M>=Mc in the past 365 d:
       sum_i 10^(alpha (m_i - Mc)) * [F(e - t_i) - F(s - t_i)] * S_ic
     F: integrated Omori-Utsu kernel (c = 0.01 d, exponent p)
     S_ic: fraction of a power-law spatial kernel (q = 1.5,
           d_i = 1 km * 10^(0.5 (m_i - Mc))) falling in the cell, approximated
           by min(1, area * f(r_ic))
a, K >= 0 by Poisson maximum likelihood on training rows; alpha, p from a grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import catalog
from .grid import _dist_km

MC = 2.5
C_DAYS = 0.01
Q = 1.5
ALPHAS = (0.6, 0.8, 1.0, 1.2)
PS = (1.05, 1.2, 1.4)


def _F(tau, p):
    tau = np.maximum(tau, 0.0)
    return ((tau + C_DAYS) ** (1 - p) - C_DAYS ** (1 - p)) / (1 - p)


def components(grid_df, cat, cells, mc=MC, size_deg=0.2):
    """Per row of grid_df (cell, start, end): background B, triggering T[alpha,p],
    and the observed count n_mc of M>=mc events in the cell during the period."""
    c = cat[cat.mag >= mc].sort_values("time").reset_index(drop=True)
    bg, _ = catalog.decluster_nn(c)
    bg = bg.sort_index().values
    tday = (c.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    mags = c.mag.values
    clat, clon = cells.set_index("cell").clat, cells.set_index("cell").clon
    cell_ids = cells.cell.values
    dlat = size_deg * 111.2
    area = dlat * dlat * np.cos(np.radians(cells.clat.values))                 # km^2
    # cell membership of events
    lat0, lon0 = cells.lat0.min(), cells.lon0.min()
    nlon = cells.lon0.nunique()
    ii = np.floor((c.lat.values - lat0) / size_deg).astype(int)
    jj = np.floor((c.lon.values - lon0) / size_deg).astype(int)
    ev_cell = np.where((ii >= 0) & (jj >= 0) & (ii < cells.lat0.nunique()) & (jj < nlon), ii * nlon + jj, -1)
    Dce = np.stack([_dist_km(la, lo, c.lat.values, c.lon.values) for la, lo in zip(cells.clat, cells.clon)])
    d_i = 1.0 * 10 ** (0.5 * (mags - mc))

    periods = grid_df[["start", "end"]].drop_duplicates().sort_values("start")
    out = {}
    for s, e in periods.itertuples(index=False):
        sd = (np.datetime64(s) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        ed = (np.datetime64(e) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        k0, k = np.searchsorted(tday, sd - 365), np.searchsorted(tday, sd)
        ke = np.searchsorted(tday, ed)
        idx = np.arange(k0, k)
        dt = ed - sd
        B = np.array([(bg[idx] & (ev_cell[idx] == cid)).sum() for cid in cell_ids], float)
        B = (B + 0.1) / 365.0 * dt
        if len(idx):
            r = Dce[:, idx]
            f = (Q - 1) / (np.pi * d_i[idx] ** 2) * (1 + (r / d_i[idx]) ** 2) ** (-Q)
            S = np.minimum(1.0, area[:, None] * f)                               # cells x events
            T = np.zeros((len(ALPHAS), len(PS), len(cell_ids)))
            for pi, p in enumerate(PS):
                tint = _F(ed - tday[idx], p) - _F(sd - tday[idx], p)
                for ai, al in enumerate(ALPHAS):
                    T[ai, pi] = S @ (10 ** (al * (mags[idx] - mc)) * tint)
        else:
            T = np.zeros((len(ALPHAS), len(PS), len(cell_ids)))
        n = np.bincount(ev_cell[k:ke][ev_cell[k:ke] >= 0], minlength=len(cell_ids))[: len(cell_ids)]
        out[pd.Timestamp(s)] = (B, T, n)
    # assemble in grid_df row order
    pos = {cid: j for j, cid in enumerate(cell_ids)}
    Brow = np.empty(len(grid_df))
    Trow = np.empty((len(ALPHAS), len(PS), len(grid_df)))
    nrow = np.empty(len(grid_df))
    for r_i, (cid, s) in enumerate(zip(grid_df.cell.values, pd.DatetimeIndex(grid_df.start))):
        B, T, n = out[s]
        j = pos[cid]
        Brow[r_i], Trow[:, :, r_i], nrow[r_i] = B[j], T[:, :, j], n[j]
    return Brow, Trow, nrow


def fit(B, T, n, rows):
    """Poisson MLE of (a, K) for each (alpha, p); return best params dict."""
    best = None
    for ai in range(len(ALPHAS)):
        for pi in range(len(PS)):
            b, t, y = B[rows], T[ai, pi, rows], n[rows]

            def nll(x):
                lam = np.exp(x[0]) * b + np.exp(x[1]) * t + 1e-12
                return lam.sum() - (y * np.log(lam)).sum()
            res = minimize(nll, x0=[0.0, -3.0], method="L-BFGS-B")
            if best is None or res.fun < best["nll"]:
                best = dict(nll=res.fun, ai=ai, pi=pi, a=float(np.exp(res.x[0])), K=float(np.exp(res.x[1])),
                            alpha=ALPHAS[ai], p=PS[pi])
    return best


def rate(B, T, params, rows=slice(None)):
    return params["a"] * B[rows] + params["K"] * T[params["ai"], params["pi"], rows]


def b_value(cat, until, mc=MC, years=5):
    c = cat[(cat.time < until) & (cat.time >= pd.Timestamp(until) - pd.DateOffset(years=years))]
    b, _, n = catalog.b_aki(c.mag.values, mc)
    return float(np.clip(b, 0.7, 1.4)) if np.isfinite(b) and n >= 100 else 1.0


def exceed_prob(lam_mc, b, m, mc=MC):
    return 1 - np.exp(-lam_mc * 10 ** (-b * (m - mc)))
