"""Reference forecasts that any ML model must beat.

All return (y_pred, proba[n, 3]) for a set of test periods.

* persistence   - repeat last period's class
* climatology   - training-set class frequencies
* gr_poisson    - trailing rate of M>=Mc events, scaled to M>=3 / M>=4 with a
                  maximum-likelihood b-value, Poisson exceedance in the period

For probabilistic baselines the class decision is argmax(p / train_prior),
which is the Bayes rule for maximising macro-average recall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import b_aki


def decide(proba, prior):
    prior = np.clip(np.asarray(prior, float), 1e-6, None)
    return np.argmax(np.asarray(proba) / prior, axis=1)


def persistence(df, test_idx):
    y = df["y"].values
    prev = np.r_[y[0], y[:-1]]
    pred = prev[test_idx]
    proba = np.eye(3)[pred] * 0.9 + 0.1 / 3
    return pred, proba


def climatology(df, train_idx, test_idx, seed=0):
    prior = np.bincount(df["y"].values[train_idx], minlength=3) / len(train_idx)
    proba = np.tile(prior, (len(test_idx), 1))
    rng = np.random.default_rng(seed)
    pred = rng.choice(3, size=len(test_idx), p=prior)
    return pred, proba


def gr_poisson_proba(periods, cat, mc=2.5, rate_days=180, b_days=730, b_default=1.0):
    """P(M>=3), P(M>=4) per period from trailing seismicity only."""
    t = cat["time"].values
    m = cat["mag"].values
    out = []
    for s, e in zip(periods["start"].values, periods["end"].values):
        dt = (e - s) / np.timedelta64(1, "D")
        k = np.searchsorted(t, s)
        k_rate = np.searchsorted(t, s - np.timedelta64(rate_days, "D"))
        k_b = np.searchsorted(t, s - np.timedelta64(b_days, "D"))
        n_mc = (m[k_rate:k] >= mc).sum()
        b, _, nb = b_aki(m[k_b:k], mc)
        if not np.isfinite(b) or nb < 50:
            b = b_default
        rate = (n_mc + 0.5) / rate_days       # events/day above Mc (+0.5 avoids zero)
        lam3 = rate * 10 ** (-b * (3.0 - mc)) * dt
        lam4 = rate * 10 ** (-b * (4.0 - mc)) * dt
        out.append((1 - np.exp(-lam3), 1 - np.exp(-lam4), b))
    a = np.array(out)
    p3, p4 = a[:, 0], a[:, 1]
    proba = np.c_[1 - p3, p3 - p4, p4]
    return proba, a[:, 2]


def gr_poisson(df, train_idx, test_idx, cat, **kw):
    proba_all, _ = gr_poisson_proba(df, cat, **kw)
    prior = np.bincount(df["y"].values[train_idx], minlength=3) / len(train_idx)
    proba = proba_all[test_idx]
    return decide(proba, prior), proba
