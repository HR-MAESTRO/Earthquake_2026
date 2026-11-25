"""Rebuild the 2023 paper's dataset: 10 features x half-month periods.

Seismicity indicators (Sect. 2) use the last n events with M >= m_thresh
before the period start, inside the study box:
    T, Mmean, dE^1/2, b (least squares, Eq. 7), eta (Eq. 8), dM (Eq. 9-10)
Injection inputs (Sect. 3) use the w calendar months before the period:
    vmean, vmax  of v(t) = sum_i v_i(t)                    [bbl/month]
    dmean, dmin  of d(t) = sum_i d_i v_i(t) / sum_i v_i(t)  [m]
Label: class of the largest event in [start, end) (0: M<3, 1: 3<=M<4, 2: M>=4).
Periods with no events are class 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SEIS_COLS = ["T", "Mmean", "dE", "b", "eta", "dM"]
INJ_COLS = ["vmean", "vmax", "dmean", "dmin"]
FEATURES = SEIS_COLS + INJ_COLS


def half_month_periods(start, end):
    out = []
    for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        mid = m + pd.Timedelta(days=15)
        nxt = m + pd.offsets.MonthBegin(1)
        out += [(m, mid), (mid, nxt)]
    df = pd.DataFrame(out, columns=["start", "end"])
    return df[(df.start >= pd.Timestamp(start)) & (df.start <= pd.Timestamp(end))].reset_index(drop=True)


def magnitude_class(m):
    if not np.isfinite(m) or m < config.CLASS_EDGES[0]:
        return 0
    return 1 if m < config.CLASS_EDGES[1] else 2


def seismicity_indicators(mags, times):
    """Eqs. 1-10 of the 2023 paper on a window of n events (time-ordered)."""
    mags = np.asarray(mags, float)
    n = len(mags)
    T = (times[-1] - times[0]) / np.timedelta64(1, "D")
    T = max(T, 1e-3)
    mmean = mags.mean()
    sqrtE = np.sqrt(10 ** (11.8 + 1.5 * mags))
    dE = sqrtE.sum() / T / 1e8                     # 1e8 erg^1/2 per day (Table 2 units)
    # Least-squares GR fit over the DISTINCT magnitudes in the window, with
    # N_i = #events with M >= M_i. This reproduces Table 2 of the paper exactly
    # (fitting over every event does not).
    M = np.unique(mags)
    Ni = np.array([(mags >= mi).sum() for mi in M], float)
    logN = np.log10(Ni)
    k = len(M)
    sM, sM2 = M.sum(), (M ** 2).sum()
    denom = sM ** 2 - k * sM2
    b = (k * (M * logN).sum() - sM * logN.sum()) / denom if (denom != 0 and k > 2) else np.nan
    a = (logN + b * M).sum() / k
    eta = ((logN - (a - b * M)) ** 2).sum() / (k - 1)
    # Table 2 values are a/b - Mmax (expected minus observed): the sign is
    # opposite to Eq. 9 as printed. We follow the table.
    dM = a / b - mags.max() if b else np.nan
    return dict(T=T, Mmean=mmean, dE=dE, b=b, eta=eta, dM=dM)


def build_seismicity(periods, cat, n_events=config.N_EVENTS, m_thresh=config.M_THRESH):
    ev = cat[cat["mag"] >= m_thresh]
    t = ev["time"].values
    m = ev["mag"].values
    rows = []
    for s in periods["start"].values:
        k = np.searchsorted(t, s)               # events strictly before start
        lo = max(0, k - n_events)
        if k - lo < 10:          # early periods use whatever history exists (as the paper did)
            rows.append({c: np.nan for c in SEIS_COLS})
            continue
        rows.append(seismicity_indicators(m[lo:k], t[lo:k]))
    return pd.DataFrame(rows, index=periods.index)


def build_labels(periods, cat):
    t = cat["time"].values
    m = cat["mag"].values
    mmax = []
    for s, e in zip(periods["start"].values, periods["end"].values):
        i, j = np.searchsorted(t, s), np.searchsorted(t, e)
        mmax.append(m[i:j].max() if j > i else np.nan)
    mmax = np.array(mmax)
    return pd.DataFrame(dict(mmax=mmax, y=[magnitude_class(x) for x in mmax]), index=periods.index)


def monthly_injection_series(inj, wells):
    """Region-level v(t) and d(t) per calendar month."""
    d = inj.merge(wells[["api", "d_basement_m"]], on="api", how="left")
    d = d[d["vol_bbl"] > 0]
    v = d.groupby("date")["vol_bbl"].sum()
    dd = d.dropna(subset=["d_basement_m"])
    dw = (dd["d_basement_m"] * dd["vol_bbl"]).groupby(dd["date"]).sum() / dd.groupby("date")["vol_bbl"].sum()
    s = pd.DataFrame(dict(v=v, d=dw)).sort_index()
    return s.asfreq("MS")


def build_injection(periods, monthly, window=config.INJ_WINDOW_MONTHS):
    rows = []
    for s in periods["start"]:
        m0 = s.replace(day=1)                   # first month NOT available
        win = monthly.loc[m0 - pd.DateOffset(months=window): m0 - pd.DateOffset(months=1)]
        rows.append(dict(vmean=win["v"].mean(), vmax=win["v"].max(),
                         dmean=win["d"].mean(), dmin=win["d"].min()))
    return pd.DataFrame(rows, index=periods.index)


def build_dataset(cat, inj, wells, box, start, end, n_events=config.N_EVENTS,
                  m_thresh=config.M_THRESH, window=config.INJ_WINDOW_MONTHS):
    from . import catalog, injection
    c = catalog.in_box(cat, box)
    i = injection.in_box(inj, box)
    periods = half_month_periods(start, end)
    monthly = monthly_injection_series(i, wells)
    df = pd.concat([periods,
                    build_seismicity(periods, c, n_events, m_thresh),
                    build_injection(periods, monthly, window),
                    build_labels(periods, c)], axis=1)
    return df
