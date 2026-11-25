"""Phases 3-4 on the gridded dataset: probabilistic exceedance forecasts.

Targets: ex3 = P(M>=3 in cell during half-month), ex4 = P(M>=4).
Evaluation: yearly expanding walk-forward with a gap; the final holdout is only
used when final=True.

Models
  clim        per-cell historical exceedance frequency (training data only)
  persist     1 if the cell had an event above threshold in the past 30 days* (proxy)
  gr          trailing-rate Gutenberg-Richter Poisson (uses n90_near, b_ml)
  si          seismogenic-index physics forecast (feature si_p3 / si_p4)
  logreg      L2 logistic regression, balanced class weights
  lgbm        LightGBM, balanced weights (tuned in inner time split)
  lgbm_smote  LightGBM with SMOTE inside the training fold
  lgbm_nosi   LightGBM without the physics features (to measure hybrid value)
All ML probabilities are isotonic-calibrated on the last training year.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics
from .grid import GRID_FEATURES

warnings.filterwarnings("ignore")
LOG_COLS = ["n30_near", "n90_near", "n365_near", "n365_cell", "v1", "v3", "v9", "v12", "vcum",
            "n_wells", "days_since_m3", "si_lam3", "fault_density"]
PHYS = ["si_sigma", "si_lam3"]


def prep(df, feats):
    X = df[feats].astype(float).copy()
    for c in LOG_COLS:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def _lgbm(params, seed=0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=params.get("n_estimators", 300), learning_rate=0.03,
                          num_leaves=params.get("num_leaves", 15),
                          min_child_samples=params.get("min_child_samples", 50),
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, class_weight=params.get("class_weight", "balanced"),
                          random_state=seed, verbose=-1)


LGBM_GRID = [dict(num_leaves=nl, n_estimators=ne, min_child_samples=mcs)
             for nl in (7, 15) for ne in (200, 500) for mcs in (20, 100)]


def _fit_ml(name, Xtr, ytr, seed=0, params=None):
    params = params or {}
    if name == "logreg":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 0.3),
                                                               class_weight="balanced", max_iter=5000))
        m.fit(Xtr.fillna(Xtr.median()), ytr)
        return m
    if name == "lgbm_smote":
        from imblearn.over_sampling import SMOTE
        Xf = Xtr.fillna(Xtr.median())
        k = max(1, min(5, int(ytr.sum()) - 1))
        Xs, ys = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xf, ytr)
        m = _lgbm({**params, "class_weight": None}, seed)
        m.fit(Xs, ys)
        m._fill = Xtr.median()
        return m
    m = _lgbm(params, seed)
    m.fit(Xtr, ytr)
    return m


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


class OffsetLGBM:
    """Physics-offset hybrid: LightGBM boosts from logit(physics probability)
    and only learns corrections to it (no class weighting, so it stays
    probabilistic)."""

    def __init__(self, prior_col, params=None, seed=0):
        self.prior_col, self.params, self.seed = prior_col, params or {}, seed

    def fit(self, X, y, prior):
        from lightgbm import LGBMClassifier
        self.m = LGBMClassifier(n_estimators=self.params.get("n_estimators", 200), learning_rate=0.02,
                                num_leaves=self.params.get("num_leaves", 7),
                                min_child_samples=self.params.get("min_child_samples", 100),
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                random_state=self.seed, verbose=-1)
        self.m.fit(X, y, init_score=_logit(prior))
        return self

    def predict(self, X, prior):
        raw = self.m.predict(X, raw_score=True) + _logit(prior)
        return 1 / (1 + np.exp(-raw))


def _predict(m, X):
    if hasattr(m, "_fill"):
        X = X.fillna(m._fill)
    elif not hasattr(m, "booster_"):
        X = X.fillna(X.median())
    return m.predict_proba(X)[:, 1]


def _tune_lgbm(Xtr, ytr, years_tr, target_metric=metrics.pr_auc, seed=0):
    last = years_tr.max()
    a, b = years_tr < last - 0, years_tr == last
    a = a & ~b
    if ytr[b].sum() == 0 or ytr[a].sum() < 5:
        return LGBM_GRID[0]
    best, best_s = LGBM_GRID[0], -np.inf
    for p in LGBM_GRID:
        m = _fit_ml("lgbm", Xtr[a], ytr[a], seed, p)
        s = target_metric(ytr[b], _predict(m, Xtr[b]))
        if s > best_s:
            best, best_s = p, s
    return best


def _calibrate(name, Xtr, ytr, years_tr, params, seed=0):
    """Fit on years < last, isotonic-calibrate on last training year, refit on all."""
    last = years_tr.max()
    a, b = years_tr < last, years_tr == last
    iso = None
    if ytr[b].sum() > 0 and ytr[a].sum() > 5:
        m0 = _fit_ml(name, Xtr[a], ytr[a], seed, params)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(_predict(m0, Xtr[b]), ytr[b])
    m = _fit_ml(name, Xtr, ytr, seed, params)
    return m, iso


def walk_forward(df, target="ex3", first_test_year=2014, last_test_year=2022, gap=18,
                 holdout_start="2023-01-01", final=False, models=None, seed=0, verbose=True):
    """Returns per-sample prediction DataFrame (one column per model)."""
    models = models or ["clim", "persist", "gr", "si", "hybrid", "logreg", "lgbm", "lgbm_smote", "lgbm_nosi"]
    df = df.sort_values(["start", "cell"]).reset_index(drop=True)
    years = df.start.dt.year.values
    periods = df.start.values
    uniq = np.sort(np.unique(periods))
    pidx = np.searchsorted(uniq, periods)
    hold = df.start >= pd.Timestamp(holdout_start)
    if final:
        test_sets = [(np.where(hold)[0])]
    else:
        test_sets = [np.where((years == y) & ~hold.values)[0]
                     for y in range(first_test_year, last_test_year + 1)]
    feats = GRID_FEATURES
    feats_nosi = [f for f in feats if f not in PHYS]
    X_all, Xn_all = prep(df, feats), prep(df, feats_nosi)
    y_all = df[target].values
    out = []
    chosen = []
    for te in test_sets:
        if len(te) == 0:
            continue
        p0 = pidx[te].min()
        tr = np.where(pidx < p0 - gap)[0]
        ytr, yrs_tr = y_all[tr], years[tr]
        res = pd.DataFrame(dict(start=df.start.values[te], cell=df.cell.values[te], y=y_all[te],
                                y3=df.y.values[te]))
        # baselines
        rate = pd.Series(ytr).groupby(df.cell.values[tr]).mean()
        res["clim"] = pd.Series(df.cell.values[te]).map(rate).fillna(ytr.mean()).values
        thr = 3.0 if target == "ex3" else 4.0
        res["persist"] = ((df["mmax365_near"].values[te] >= thr) &
                          (df["days_since_m3"].values[te] <= 30)).astype(float) * 0.5 + 0.01
        mc, bb = 2.5, df["b_ml"].fillna(1.0).clip(0.6, 1.6).values[te]
        lam = (df["n90_near"].values[te] + 0.1) / 90 * 15 * (10 ** (-bb * (thr - mc)))
        lam *= 1.0 / (np.pi * 25 ** 2 / (0.2 * 111 * 0.2 * 90))   # neighbourhood -> cell area
        res["gr"] = 1 - np.exp(-lam)
        prior_col = "si_p3" if target == "ex3" else "si_p4"
        res["si"] = df[prior_col].values[te]
        if "hybrid" in models:
            prior_all = df[prior_col].clip(1e-5, 1 - 1e-5).values
            hm = OffsetLGBM(prior_col, seed=seed).fit(X_all.iloc[tr], ytr, prior_all[tr])
            res["hybrid"] = hm.predict(X_all.iloc[te], prior_all[te])
        pp_col = "pp_p3" if target == "ex3" else "pp_p4"
        if pp_col in df:
            res["pp"] = df[pp_col].values[te]
            if "hybrid_pp" in models:
                prior_pp = df[pp_col].clip(1e-5, 1 - 1e-5).values
                Xp = X_all.assign(**{c: np.log1p(np.clip(df[c].values, 0, None))
                                     for c in ("dp1", "dp12") if c in df})
                hm = OffsetLGBM(pp_col, seed=seed).fit(Xp.iloc[tr], ytr, prior_pp[tr])
                res["hybrid_pp"] = hm.predict(Xp.iloc[te], prior_pp[te])
        for name in [m for m in models if m in ("logreg", "lgbm", "lgbm_smote", "lgbm_nosi")]:
            Xtr = (Xn_all if name == "lgbm_nosi" else X_all).iloc[tr]
            Xte = (Xn_all if name == "lgbm_nosi" else X_all).iloc[te]
            base = "lgbm" if name == "lgbm_nosi" else name
            params = _tune_lgbm(Xtr, ytr, yrs_tr, seed=seed) if base.startswith("lgbm") else {}
            chosen.append(dict(model=name, test_year=int(years[te].min()), params=str(params)))
            m, iso = _calibrate(base, Xtr, ytr, yrs_tr, params, seed)
            p = _predict(m, Xte)
            res[name + "_raw"] = p
            res[name] = iso.predict(p) if iso is not None else p
            if name == "lgbm" and final:
                res.attrs["final_model"] = m
                res.attrs["final_X"] = Xte
        out.append(res)
        if verbose:
            r = res
            msg = " ".join(f"{m}={metrics.pr_auc(r.y, r[m]):.3f}" for m in models if m in r)
            print(f"  {target} test {int(years[te].min())}: pos={int(r.y.sum())} PR-AUC {msg}")
    return pd.concat(out, ignore_index=True), pd.DataFrame(chosen)


def score(pred, models, n_boot=1000, seed=0):
    rows = []
    # bootstrap by period blocks: aggregate per period to keep cells of a period together
    pred = pred.sort_values("start")
    periods = pred.start.values
    uniq, inv = np.unique(periods, return_inverse=True)
    for m in models:
        if m not in pred:
            continue
        y, p = pred.y.values, pred[m].values
        def boot(fn):
            point = fn(y, p)
            rng = np.random.default_rng(seed)
            vals = []
            groups = [np.where(inv == k)[0] for k in range(len(uniq))]
            for _ in range(n_boot):
                blk = metrics._block_indices(len(uniq), 6, rng)
                idx = np.concatenate([groups[k] for k in blk])
                v = fn(y[idx], p[idx])
                if np.isfinite(v):
                    vals.append(v)
            lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
            return point, lo, hi
        pr, prl, prh = boot(metrics.pr_auc)
        br, brl, brh = boot(metrics.brier_binary)
        clim_b = metrics.brier_binary(y, np.full_like(p, y.mean(), dtype=float))
        rows.append(dict(model=m, n=len(y), positives=int(y.sum()), base_rate=y.mean(),
                         prauc=pr, prauc_lo=prl, prauc_hi=prh, brier=br, brier_lo=brl, brier_hi=brh,
                         bss_vs_const=1 - br / clim_b))
    return pd.DataFrame(rows)


def cost_threshold(c_miss, c_false):
    """Bayes-optimal alarm threshold on a calibrated probability."""
    return c_false / (c_false + c_miss)


def alarm_table(pred, models, ratios=(5, 20, 100)):
    rows = []
    for r in ratios:
        t = cost_threshold(r, 1.0)
        for m in models:
            if m not in pred:
                continue
            a = pred[m].values >= t
            y = pred.y.values.astype(bool)
            tp, fp, fn = (a & y).sum(), (a & ~y).sum(), (~a & y).sum()
            rows.append(dict(cost_ratio=r, threshold=t, model=m, alarms=int(a.sum()), tp=int(tp),
                             fp=int(fp), fn=int(fn), pod=tp / max(1, tp + fn),
                             far=fp / max(1, tp + fp), expected_cost=fn * r + fp * 1.0))
    return pd.DataFrame(rows)


def conformal_sets(cal_y, cal_p, test_p, alpha=0.1):
    """Mondrian (class-conditional) split-conformal prediction sets for a binary
    probability. Returns array of sets as strings: '{0}', '{1}', '{0,1}', '{}'."""
    cal_y = np.asarray(cal_y, int)
    q = {}
    for c in (0, 1):
        s = 1 - np.where(c == 1, cal_p, 1 - cal_p)[cal_y == c]
        n = len(s)
        q[c] = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher") if n else 1.0
    inc1 = (1 - test_p) <= q[1]
    inc0 = test_p <= q[0]
    return np.where(inc0 & inc1, "{0,1}", np.where(inc1, "{1}", np.where(inc0, "{0}", "{}")))


def shap_importance(model, X):
    """Mean |SHAP| per feature using LightGBM's built-in TreeSHAP."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    vals = np.abs(contrib[:, :-1]).mean(0)
    return pd.Series(vals, index=X.columns).sort_values(ascending=False)
