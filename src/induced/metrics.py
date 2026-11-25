"""Evaluation metrics with bootstrap confidence intervals.

MAvA (macro-average recall) and per-class POD / FAR / R follow the 2023 paper
(Eqs. 26-30). Probabilistic scores (Brier, PR-AUC) are added for Phase 3.

Bootstrap uses a moving-block bootstrap so that serial correlation between
neighbouring half-month periods is not ignored.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

N_CLASSES = 3


def confusion(y_true, y_pred, k=N_CLASSES):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        cm[t, p] += 1
    return cm


def per_class(y_true, y_pred, k=N_CLASSES):
    """POD (=recall), FAR (false alarm ratio, FP/(TP+FP)) and R = POD - FAR."""
    cm = confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        pod = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        far = np.where(tp + fp > 0, fp / (tp + fp), np.nan)
    return dict(pod=pod, far=far, r=pod - far, support=cm.sum(1))


def mava(y_true, y_pred, k=N_CLASSES):
    """Macro-average recall over classes present in y_true (Eq. 27)."""
    pod = per_class(y_true, y_pred, k)["pod"]
    return float(np.nanmean(pod))


def brier_binary(y, p):
    return float(brier_score_loss(np.asarray(y, int), np.asarray(p, float)))


def pr_auc(y, p):
    y = np.asarray(y, int)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def _block_indices(n, block, rng):
    """Moving-block bootstrap index sample of length n."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def bootstrap_ci(metric, *arrays, n_boot=2000, block=6, alpha=0.05, seed=0):
    """Point estimate and (lo, hi) percentile CI for metric(*arrays).

    Arrays must be in time order. block=6 half-months ~ 3 months.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    point = metric(*arrays)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        try:
            v = metric(*[a[idx] for a in arrays])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, (np.nan, np.nan)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, (float(lo), float(hi))


def summary(y_true, y_pred, proba=None, n_boot=2000, seed=0):
    """One-row dict of the standard metric set."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    m, (lo, hi) = bootstrap_ci(mava, y_true, y_pred, n_boot=n_boot, seed=seed)
    pc = per_class(y_true, y_pred)
    out = dict(n=len(y_true), mava=m, mava_lo=lo, mava_hi=hi,
               acc=float((y_true == y_pred).mean()))
    for c in range(N_CLASSES):
        out[f"pod{c}"] = pc["pod"][c]
        out[f"far{c}"] = pc["far"][c]
        out[f"n{c}"] = int(pc["support"][c])
    if proba is not None:
        proba = np.asarray(proba, float)
        y4 = (y_true == 2).astype(int)
        y3 = (y_true >= 1).astype(int)
        p4 = proba[:, 2]
        p3 = proba[:, 1] + proba[:, 2]
        out["brier_m4"], (out["brier_m4_lo"], out["brier_m4_hi"]) = bootstrap_ci(
            brier_binary, y4, p4, n_boot=n_boot, seed=seed)
        out["prauc_m4"], (out["prauc_m4_lo"], out["prauc_m4_hi"]) = bootstrap_ci(
            pr_auc, y4, p4, n_boot=n_boot, seed=seed)
        out["brier_m3"] = brier_binary(y3, p3)
        out["prauc_m3"] = pr_auc(y3, p3)
        out["base_rate_m4"] = float(y4.mean())
    return out
