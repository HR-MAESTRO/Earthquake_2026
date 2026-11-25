"""Reusable evaluation harness (Phase 1).

evaluate(df, features, spec, ...) runs an expanding-window yearly walk-forward:
  * outer folds: train on years < k (minus `gap` periods), test on year k
  * inner tuning: forward-chaining folds inside the outer training set only
  * resampling only inside training data (pipelines)
Predictions from all outer test years are pooled and scored with block
bootstrap CIs (metrics.summary). The final holdout is excluded unless
`final=True` is passed explicitly.

A "spec" is either
  ('ml', model_name, grid)  grid = list of (resampler_name, params) to tune over
  ('baseline', name)        name in {'persistence', 'climatology', 'gr_poisson'}
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from . import baselines, metrics, models, splits

warnings.filterwarnings("ignore")


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


DEFAULT_SPECS = {
    "persistence": ("baseline", "persistence"),
    "climatology": ("baseline", "climatology"),
    "gr_poisson": ("baseline", "gr_poisson"),
    "logreg": ("ml", "logreg", [("None", p) for p in grid(C=[0.1, 1, 10], class_weight=["balanced"])]),
    "svm_rbf": ("ml", "svm", [(rs, p) for rs in ["None", "SMOTE"] for p in
                              grid(C=[0.5, 4, 16], gamma=[0.5, 4, 16], kernel=["rbf"],
                                   class_weight=["balanced"], probability=[True])]),
    "pnn": ("ml", "pnn", [(rs, p) for rs in ["None", "SMOTE"] for p in grid(sigma=[0.05, 0.1, 0.2, 0.4])]),
    "ada_knn": ("ml", "ada_knn", [(rs, p) for rs in ["SMOTE", "ADASYN"] for p in
                                  grid(k=[10, 40], n_estimators=[50])]),
    "lgbm": ("ml", "lgbm", [(rs, p) for rs in ["None", "SMOTE"] for p in
                            grid(num_leaves=[3, 7], n_estimators=[100, 300], learning_rate=[0.05])]),
}


def _fit_predict(model, rs_name, params, X, y, tr, te, seed=0):
    rs = models.resamplers(seed)[rs_name]
    if rs_name != "None":
        params = {k: v for k, v in params.items() if k != "class_weight"}
        # small minority classes: shrink SMOTE neighbourhoods, or fall back
        min_count = np.bincount(y[tr]).astype(float)
        min_count = int(min_count[min_count > 0].min())
        if hasattr(rs, "k_neighbors") and isinstance(rs.k_neighbors, int):
            if min_count < 2:
                rs = models.resamplers(seed)["RandomOver"]
            else:
                rs.set_params(k_neighbors=min(rs.k_neighbors, min_count - 1))
        if hasattr(rs, "n_neighbors") and isinstance(getattr(rs, "n_neighbors"), int) and min_count < 2:
            rs = models.resamplers(seed)["RandomOver"]
    pipe = models.make(model, rs, seed=seed, **params)
    try:
        pipe.fit(X[tr], y[tr])
    except ValueError:
        pipe = models.make(model, models.resamplers(seed)["RandomOver"], seed=seed, **params)
        pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    try:
        proba = pipe.predict_proba(X[te])
        full = np.zeros((len(te), 3))
        full[:, pipe.classes_] = proba
        proba = full
    except Exception:
        proba = None
    return pred, proba


def tune(model, grid_, X, y, tr, inner_splits=3, gap=18, seed=0):
    """Pick (resampler, params) by forward-chaining CV inside `tr` only."""
    inner = splits.inner_time_folds(len(tr), inner_splits, gap=gap, min_train=40)
    if not inner or len(grid_) == 1:
        return grid_[0], np.nan
    best, best_s = grid_[0], -np.inf
    for rs, p in grid_:
        scores = []
        for itr, ite in inner:
            a, b = tr[itr], tr[ite]
            if len(np.unique(y[a])) < 2:
                continue
            try:
                pred, _ = _fit_predict(model, rs, p, X, y, a, b, seed)
                scores.append(metrics.mava(y[b], pred))
            except Exception:
                pass
        s = np.nanmean(scores) if scores else -np.inf
        if s > best_s:
            best, best_s = (rs, p), s
    return best, best_s


def evaluate(df, features, specs=None, first_test_year=2013, last_test_year=None,
             gap=18, holdout_start=None, final=False, cat=None, seed=0, n_boot=2000,
             verbose=True):
    """Return (summary DataFrame, per-period predictions DataFrame, chosen params)."""
    specs = specs or DEFAULT_SPECS
    df = df.reset_index(drop=True)
    X = df[features].values.astype(float)
    y = df["y"].values.astype(int)
    years = pd.DatetimeIndex(df["start"]).year
    if holdout_start is not None:
        hold = splits.holdout_mask(df, holdout_start)
    else:
        hold = np.zeros(len(df), bool)
    if final:
        # one fold: everything before the holdout (minus gap) -> holdout
        te = np.where(hold)[0]
        tr = np.where(~hold)[0]
        tr = tr[tr < te.min() - gap]
        folds = [(tr, te)]
    else:
        last = last_test_year or int(years[~hold].max())
        sub = df.loc[~hold]
        folds = splits.walk_forward(sub, first_test_year, last, gap_periods=gap)
    rows, preds, chosen = [], [], []
    for name, spec in specs.items():
        all_te, all_pred, all_proba = [], [], []
        for tr, te in folds:
            if spec[0] == "baseline":
                if spec[1] == "persistence":
                    pred, proba = baselines.persistence(df, te)
                elif spec[1] == "climatology":
                    pred, proba = baselines.climatology(df, tr, te, seed)
                else:
                    pred, proba = baselines.gr_poisson(df, tr, te, cat)
            else:
                _, model, grid_ = spec
                (rs, p), s = tune(model, grid_, X, y, tr, gap=gap, seed=seed)
                chosen.append(dict(model=name, test_year=int(years[te].min()), resampler=rs,
                                   params=str(p), inner_mava=s))
                pred, proba = _fit_predict(model, rs, p, X, y, tr, te, seed)
            all_te.append(te)
            all_pred.append(pred)
            all_proba.append(proba if proba is not None else np.full((len(te), 3), np.nan))
        te = np.concatenate(all_te)
        pred = np.concatenate(all_pred)
        proba = np.vstack(all_proba)
        has_p = not np.isnan(proba).any()
        s = metrics.summary(y[te], pred, proba if has_p else None, n_boot=n_boot, seed=seed)
        s["model"] = name
        rows.append(s)
        preds.append(pd.DataFrame(dict(model=name, start=df["start"].values[te], y=y[te], pred=pred,
                                       p0=proba[:, 0], p1=proba[:, 1], p2=proba[:, 2])))
        if verbose:
            print(f"  {name:12s} MAvA={s['mava']:.3f} [{s['mava_lo']:.2f},{s['mava_hi']:.2f}]"
                  f"  POD(M4)={s['pod2']:.2f} FAR(M4)={s['far2']:.2f}"
                  + (f"  PR-AUC(M4)={s['prauc_m4']:.2f} base={s['base_rate_m4']:.2f}" if has_p else ""))
    cols = ["model", "n", "mava", "mava_lo", "mava_hi", "acc", "pod0", "pod1", "pod2", "far0", "far1",
            "far2", "n0", "n1", "n2", "brier_m4", "brier_m4_lo", "brier_m4_hi", "prauc_m4",
            "prauc_m4_lo", "prauc_m4_hi", "base_rate_m4", "brier_m3", "prauc_m3"]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    return out, pd.concat(preds, ignore_index=True), pd.DataFrame(chosen)
