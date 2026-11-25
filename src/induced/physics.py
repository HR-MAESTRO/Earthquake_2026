"""Pore-pressure diffusion forecast per grid cell (plan step 17).

Each well's monthly injected volume is treated as an instantaneous volume
pulse in a laterally infinite, homogeneous reservoir layer (2-D diffusion).
The pressure change at distance r and lag t is proportional to

    G(r, t) = exp(-r^2 / (4 D t)) / (4 pi D t)

and pressures from all wells and months superpose. Units are relative (the
layer thickness, storage and viscosity are absorbed by the per-cell
calibration), so only D (hydraulic diffusivity, m^2/s) matters.

Forecast (seismogenic-index form, with pressure rate in place of volume,
after Langenbruch, Weingarten & Zoback 2018):
    sigma_p = log10(N_c(>=Mc, past year) + 0.5) - log10(sum of positive
              monthly pressure increments over the past year) + b*Mc
    lambda(M>=m) = 10^(sigma_p - b*m) * dP_next,
    dP_next ~ max(last monthly increment, 0) / 2      (half-month)
All quantities use data before the period start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import _dist_km

SEC_PER_MONTH = 30.44 * 86400


def pressure_matrix(inj, cells, D=0.5, r_max_km=60.0, max_lag=240):
    """Relative pore pressure at each cell centre at the END of each month.
    Returns (months, P[months, cells])."""
    inj = inj[inj.vol_bbl > 0]
    wloc = inj.groupby("api")[["lat", "lon"]].first()
    months = pd.date_range(inj.date.min(), inj.date.max(), freq="MS")
    m_idx = pd.Series(np.arange(len(months)), index=months)
    a_idx = pd.Series(np.arange(len(wloc)), index=wloc.index)
    V = np.zeros((len(months), len(wloc)))
    np.add.at(V, (m_idx[inj.date].values, a_idx[inj.api].values),
              inj.vol_bbl.to_numpy(dtype=float, na_value=0.0) * 0.159)        # m^3
    L = min(max_lag, len(months))
    t = (np.arange(L) + 0.5) * SEC_PER_MONTH                                    # lag 0 = same month
    P = np.zeros((len(months), len(cells)))
    for ci, (la, lo) in enumerate(zip(cells.clat, cells.clon)):
        r = _dist_km(la, lo, wloc.lat.values, wloc.lon.values) * 1000.0
        near = r <= r_max_km * 1000
        if not near.any():
            continue
        r = r[near]
        G = np.exp(-(r[None, :] ** 2) / (4 * D * t[:, None])) / (4 * np.pi * D * t[:, None])  # L x W
        Vn = V[:, near]
        for lag in range(L):
            P[lag:, ci] += Vn[:len(months) - lag] @ G[lag]
    return months, P


def add_pressure_features(df, months, P, mc=2.5, b_default=1.0):
    """Attach p_now, dp1, dp12 and the pressure-rate forecast pp_p3 / pp_p4."""
    m_pos = {m: i for i, m in enumerate(months)}
    dP = np.diff(P, axis=0, prepend=P[:1])
    dPpos = np.clip(dP, 0, None)
    cum12 = np.array([dPpos[max(0, i - 11):i + 1].sum(0) for i in range(len(months))])
    last = df["start"].apply(lambda s: m_pos.get(s.replace(day=1) - pd.DateOffset(months=1), -1)).values
    ok = last >= 0
    ci = df["cell"].values
    p_now = np.full(len(df), np.nan)
    dp1 = np.full(len(df), np.nan)
    dp12 = np.full(len(df), np.nan)
    p_now[ok] = P[last[ok], ci[ok]]
    dp1[ok] = dP[last[ok], ci[ok]]
    dp12[ok] = cum12[last[ok], ci[ok]]
    out = df.copy()
    out["p_now"], out["dp1"], out["dp12"] = p_now, dp1, dp12
    b = out["b_ml"].fillna(b_default).clip(0.6, 1.6)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.log10(out["n365_cell"] + 0.5) - np.log10(out["dp12"].where(out["dp12"] > 0)) + b * mc
    nxt = np.clip(out["dp1"], 0, None) / 2.0
    lam3 = (10 ** (sigma - b * 3.0) * nxt).fillna(0)
    lam4 = (10 ** (sigma - b * 4.0) * nxt).fillna(0)
    out["pp_sigma"] = sigma
    out["pp_p3"] = 1 - np.exp(-lam3)
    out["pp_p4"] = 1 - np.exp(-lam4)
    return out
