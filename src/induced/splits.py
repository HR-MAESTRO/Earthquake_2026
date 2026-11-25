"""Train/test splitters.

All splitters take a DataFrame indexed in time order with a 'start' column
(period start timestamp) and return lists of (train_idx, test_idx) arrays of
integer positions.

* random_split     - what the 2023 paper did (random 75/25)
* chrono_split     - first 75% train, last 25% test
* walk_forward     - train on years <= k (minus a gap), test on year k+1
* holdout_mask     - final untouched holdout (last N years)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(df, test_size=0.25, seed=0, stratify=False):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    strat = df["y"].values if stratify else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                              stratify=strat)
    return [(np.sort(tr), np.sort(te))]


def chrono_split(df, test_size=0.25, gap=0):
    n = len(df)
    n_test = int(round(n * test_size))
    cut = n - n_test
    tr = np.arange(0, max(0, cut - gap))
    te = np.arange(cut, n)
    return [(tr, te)]


def walk_forward(df, first_test_year, last_test_year, gap_periods=18,
                 min_train=48):
    """Expanding-window yearly walk-forward.

    gap_periods: number of half-month periods dropped from the end of the
    training set before each test year (18 = 9 months, the feature window), so
    no training sample shares a look-back window with the first test sample.
    """
    years = pd.DatetimeIndex(df["start"]).year.values
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        te = np.where(years == y)[0]
        if len(te) == 0:
            continue
        tr = np.where(years < y)[0]
        if gap_periods:
            tr = tr[tr < te.min() - gap_periods]
        if len(tr) < min_train:
            continue
        folds.append((tr, te))
    return folds


def holdout_mask(df, holdout_start):
    """Boolean mask: True for periods in the final holdout."""
    return np.asarray(pd.DatetimeIndex(df["start"]) >= pd.Timestamp(holdout_start))


def inner_time_folds(n, n_splits=3, gap=0, min_train=30):
    """Forward-chaining folds on positions 0..n-1 for nested tuning."""
    folds = []
    edges = np.linspace(0, n, n_splits + 2).astype(int)[1:]
    for i in range(n_splits):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.arange(0, max(0, edges[i] - gap))
        if len(tr) >= min_train and len(te):
            folds.append((tr, te))
    return folds
