"""Pooled multi-basin forecasting (protocol v1 / v2, docs/protocol_v*.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalog, config, etas, metrics, regions
from .grid import GRID_FEATURES, make_cells
from .gridmodels import prep

GAP = 18
FEATS = [f for f in GRID_FEATURES if not f.startswith("fault")]
PLAIN_FEATS = [f for f in FEATS if f not in ("si_sigma", "si_lam3")]
LGB = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, reg_lambda=5.0,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
OK_BOX = dict(lat_min=35.0, lat_max=37.0, lon_min=-99.0, lon_max=-96.0)
MIDLAND_BOX = dict(lat_min=31.0, lat_max=33.0, lon_min=-102.6, lon_max=-101.0)
FORTWORTH_BOX = dict(lat_min=32.0, lat_max=33.6, lon_min=-98.6, lon_max=-96.6)

BASINS = {
    "oklahoma": dict(box=OK_BOX, grid="grid_dataset_v0.1.parquet", test=(2014, 2022), dev=True),
    "kansas": dict(box=regions.KANSAS_BOX, grid="grid_kansas_v0.1.parquet", test=(2017, 2025), dev=True),
    "delaware": dict(box=regions.DELAWARE_BOX, grid="grid_delaware_v0.1.parquet", test=(2019, 2025), dev=True),
    "midland": dict(box=MIDLAND_BOX, grid="grid_midland_v0.1.parquet", test=(2020, 2025), dev=False),
    "fortworth": dict(box=FORTWORTH_BOX, grid="grid_fortworth_lag0.parquet", test=(2012, 2025), dev=False),
}


def basin_catalog(name):
    box = BASINS[name]["box"]
    pad = dict(lat_min=box["lat_min"] - .5, lat_max=box["lat_max"] + .5,
               lon_min=box["lon_min"] - .5, lon_max=box["lon_max"] + .5)
    if name == "oklahoma":
        c = catalog.in_box(catalog.load(), pad)
    else:
        c = catalog.in_box(regions.comcat(name), pad)
    return c[c.time >= "2005-01-01"].sort_values("time").reset_index(drop=True)


def _grid(name, lag):
    f = BASINS[name]["grid"] if lag == 0 else f"grid_{name}_lag{lag}.parquet"
    g = pd.read_parquet(config.PROCESSED / f).sort_values(["start", "cell"]).reset_index(drop=True)
    return g[g.start < "2026-01-01"].reset_index(drop=True)


def load_basin(name, lag=0):
    """Grid rows (with injection reporting lag `lag` months) + ETAS components.
    ETAS uses only the catalog, so its components are computed once on the
    lag-0 grid and re-indexed onto the lagged rows."""
    cache = config.PROCESSED / f"mb_{name}.npz"
    g0 = _grid(name, 0)
    cat = basin_catalog(name)
    if cache.exists():
        z = np.load(cache)
        B, T, n = z["B"], z["T"], z["n"]
    else:
        cells = make_cells(BASINS[name]["box"], 0.2)
        B, T, n = etas.components(g0, cat, cells)
        np.savez(cache, B=B, T=T, n=n)
    g = g0 if lag == 0 else _grid(name, lag)
    if lag:
        pos = pd.Series(np.arange(len(g0)), index=pd.MultiIndex.from_arrays([g0.cell, g0.start]))
        idx = pos.reindex(pd.MultiIndex.from_arrays([g.cell, g.start])).values
        keep = ~np.isnan(idx)
        g = g[keep].reset_index(drop=True)
        idx = idx[keep].astype(int)
        B, T, n = B[idx], T[:, :, idx], n[idx]
    g = g.copy()
    g["n_mc"] = n
    g["basin"] = name
    return g, B, T, cat


def _lgb_reg():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="poisson", random_state=0, **LGB)


def _lgb_cls():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=0, **LGB)


def _period_idx(g):
    return pd.DatetimeIndex(g.start)


_ETAS_MEMO = {}


def _etas_fit(name, data, cutoff):
    key = (name, str(cutoff))
    if key not in _ETAS_MEMO:
        gz, Bz, Tz, _ = data[name]
        tr = np.where(gz.start.values < cutoff)[0]
        _ETAS_MEMO[key] = etas.fit(Bz, Tz, gz.n_mc.values, tr) if len(tr) else None
    return _ETAS_MEMO[key]


def _empirical_ratio(names, data, cutoff, m, mc=etas.MC):
    """Pooled fraction of M>=mc events that are M>=m, from training-period catalogs."""
    num = den = 0
    for name in names:
        c = data[name][3]
        c = c[c.time < cutoff]
        den += (c.mag >= mc).sum()
        num += (c.mag >= m).sum()
    return (num + 0.5) / (den + 1.0)


def _pooled_set(names, data, cutoff):
    Xs, Xp, ys, offs, y3s, y4s = [], [], [], [], [], []
    for name in names:
        gz, Bz, Tz, _ = data[name]
        tr = np.where(gz.start.values < cutoff)[0]
        if len(tr) < 500:
            continue
        pz = _etas_fit(name, data, cutoff)
        lz = etas.rate(Bz, Tz, pz, tr)
        Xs.append(prep(gz.iloc[tr], FEATS).assign(etas_loglam=np.log(lz + 1e-6)))
        Xp.append(prep(gz.iloc[tr], PLAIN_FEATS))
        ys.append(gz.n_mc.values[tr])
        offs.append(np.log(lz + 1e-6))
        y3s.append(gz.ex3.values[tr])
        y4s.append(gz.ex4.values[tr])
    if not Xs:
        return None
    return dict(X=pd.concat(Xs), Xp=pd.concat(Xp), y=np.concatenate(ys), off=np.concatenate(offs),
                y3=np.concatenate(y3s), y4=np.concatenate(y4s), names=names)


def run(target, data, pooled_from, years, local_ml=True, lobo=True, ablation=True, verbose=True):
    """Yearly walk-forward on basin `target`.

    pooled_from: basins whose time-respecting past trains the pooled ML
                 (may include the target itself).
    lobo:        also train on pooled_from minus the target (leave-one-basin-out).
    ablation:    plain count model (no ETAS offset or physics inputs) and an
                 empirical magnitude conversion instead of Gutenberg-Richter.
    """
    g, B, T, cat = data[target]
    out = []
    for yr in range(years[0], years[1] + 1):
        t0 = pd.Timestamp(f"{yr}-01-01")
        te = np.where(_period_idx(g).year == yr)[0]
        if not len(te):
            continue
        uniq = np.sort(g.start.unique())
        p0 = np.searchsorted(uniq, g.start.values[te].min())
        cutoff = uniq[max(0, p0 - GAP)]
        tr_loc = np.where(g.start.values < cutoff)[0]
        if len(tr_loc) == 0:
            continue
        prm = _etas_fit(target, data, cutoff)
        b = etas.b_value(cat, t0)
        lam_all = etas.rate(B, T, prm)
        r = pd.DataFrame(dict(basin=target, start=g.start.values[te], cell=g.cell.values[te],
                              ex3=g.ex3.values[te], ex4=g.ex4.values[te], n_mc=g.n_mc.values[te],
                              days_since_m3=g.days_since_m3.values[te]))
        r["lam_etas"] = lam_all[te]
        for m, col in ((3, "ex3"), (4, "ex4")):
            yl = g[col].values[tr_loc]
            rate_c = pd.Series(yl).groupby(g.cell.values[tr_loc]).mean()
            r[f"clim_{col}"] = pd.Series(g.cell.values[te]).map(rate_c).fillna(yl.mean()).values
            r[f"const_{col}"] = yl.mean()
            r[f"si_{col}"] = g[f"si_p{m}"].values[te]
            r[f"etas_{col}"] = etas.exceed_prob(lam_all[te], b, m)
        Xte = prep(g.iloc[te], FEATS).assign(etas_loglam=np.log(lam_all[te] + 1e-6))
        Xte_p = prep(g.iloc[te], PLAIN_FEATS)
        sets = {"pooled": pooled_from}
        if lobo:
            sets["lobo"] = [n for n in pooled_from if n != target]
        for tag, names in sets.items():
            P = _pooled_set(names, data, cutoff)
            if P is None:
                continue
            reg = _lgb_reg().fit(P["X"], P["y"], init_score=P["off"])
            lam_h = np.exp(reg.predict(Xte, raw_score=True) + np.log(lam_all[te] + 1e-6))
            r[f"lam_hybrid_{tag}"] = lam_h
            for m, col, yy in ((3, "ex3", P["y3"]), (4, "ex4", P["y4"])):
                r[f"hybrid_count_{tag}_{col}"] = etas.exceed_prob(lam_h, b, m)
                if yy.sum() >= 5:
                    r[f"lgbm_cls_{tag}_{col}"] = _lgb_cls().fit(P["X"], yy).predict_proba(Xte)[:, 1]
                else:
                    r[f"lgbm_cls_{tag}_{col}"] = yy.mean()
            if ablation and tag == "pooled":
                # (a) plain count model: no ETAS offset, no ETAS or SI inputs; GR conversion
                plain = _lgb_reg().fit(P["Xp"], P["y"])
                lam_p = plain.predict(Xte_p)
                r["lam_count_plain"] = lam_p
                # (b) hybrid counts with an empirical magnitude ratio instead of GR
                for m, col in ((3, "ex3"), (4, "ex4")):
                    r[f"count_plain_gr_{col}"] = etas.exceed_prob(lam_p, b, m)
                    ratio = _empirical_ratio(names, data, cutoff, m)
                    r[f"hybrid_count_emp_{col}"] = 1 - np.exp(-lam_h * ratio)
        if local_ml and len(tr_loc) >= 500:
            Xl = prep(g.iloc[tr_loc], FEATS).assign(etas_loglam=np.log(lam_all[tr_loc] + 1e-6))
            reg = _lgb_reg().fit(Xl, g.n_mc.values[tr_loc], init_score=np.log(lam_all[tr_loc] + 1e-6))
            lam_l = np.exp(reg.predict(Xte, raw_score=True) + np.log(lam_all[te] + 1e-6))
            r["lam_hybrid_local"] = lam_l
            for m, col in ((3, "ex3"), (4, "ex4")):
                r[f"hybrid_count_local_{col}"] = etas.exceed_prob(lam_l, b, m)
        r["etas_alpha"], r["etas_p"], r["b"] = prm["alpha"], prm["p"], b
        out.append(r)
        if verbose:
            print(f"  {target} {yr}: ex3 pos={int(r.ex3.sum())} ex4 pos={int(r.ex4.sum())} "
                  f"alpha={prm['alpha']} p={prm['p']} b={b:.2f}", flush=True)
    return pd.concat(out, ignore_index=True)


MODELS = ["clim", "si", "etas", "lgbm_cls_pooled", "lgbm_cls_lobo", "hybrid_count_pooled",
          "hybrid_count_lobo", "hybrid_count_local", "count_plain_gr", "hybrid_count_emp"]


def _blocks(pred, rng, block=6):
    uniq, inv = np.unique(pred.start.values, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    blk = metrics._block_indices(len(uniq), block, rng)
    return np.concatenate([groups[k] for k in blk])


def _year_blocks(pred, rng):
    yrs = pd.DatetimeIndex(pred.start).year.values
    u = np.unique(yrs)
    pick = rng.choice(u, size=len(u), replace=True)
    return np.concatenate([np.where(yrs == y)[0] for y in pick])


def resample_indices(pred, n_boot=1000, seed=0, block=6, by_year=False):
    rng = np.random.default_rng(seed)
    if by_year:
        return [_year_blocks(pred, rng) for _ in range(n_boot)]
    return [_blocks(pred, rng, block) for _ in range(n_boot)]


def score(pred, col, n_boot=1000, seed=0, block=6, by_year=False, models=None):
    rows = []
    y = pred[col].values
    const = pred[f"const_{col}"].values
    bc = metrics.brier_binary(y, const)
    idxs = resample_indices(pred, n_boot, seed, block, by_year)
    for m in models or MODELS:
        c = f"{m}_{col}"
        if c not in pred:
            continue
        if pred[c].isna().any():          # model not available in some folds: score its own rows
            sub = pred[pred[c].notna()].reset_index(drop=True)
            if sub[col].sum() == 0:
                continue
            rows.append(score(sub, col, n_boot, seed, block, by_year, [m]).iloc[0].to_dict())
            continue
        p = pred[c].values
        pr = metrics.pr_auc(y, p)
        bss = 1 - metrics.brier_binary(y, p) / bc
        bs_pr = [metrics.pr_auc(y[i], p[i]) for i in idxs]
        bs_bss = [1 - metrics.brier_binary(y[i], p[i]) / metrics.brier_binary(y[i], const[i]) for i in idxs]
        rows.append(dict(model=m, target=col, n=len(y), positives=int(y.sum()), prauc=pr,
                         prauc_lo=np.nanpercentile(bs_pr, 2.5), prauc_hi=np.nanpercentile(bs_pr, 97.5),
                         bss=bss, bss_lo=np.nanpercentile(bs_bss, 2.5), bss_hi=np.nanpercentile(bs_bss, 97.5)))
    return pd.DataFrame(rows)


def paired_diff(pred, col, a, b, n_boot=1000, seed=0, block=6, by_year=False, level=0.95):
    """PR-AUC(a) - PR-AUC(b) with a paired bootstrap interval at `level`
    (on the rows where both models exist)."""
    pred = pred[pred[f"{a}_{col}"].notna() & pred[f"{b}_{col}"].notna()].reset_index(drop=True)
    y = pred[col].values
    pa, pb = pred[f"{a}_{col}"].values, pred[f"{b}_{col}"].values
    d = metrics.pr_auc(y, pa) - metrics.pr_auc(y, pb)
    vals = []
    for i in resample_indices(pred, n_boot, seed, block, by_year):
        vals.append(metrics.pr_auc(y[i], pa[i]) - metrics.pr_auc(y[i], pb[i]))
    q = (1 - level) / 2 * 100
    return d, np.nanpercentile(vals, q), np.nanpercentile(vals, 100 - q)


def equivalence(pred, col, a, b, margin, **kw):
    """TOST via the 90% paired interval: equivalent if it lies inside +-margin."""
    d, lo, hi = paired_diff(pred, col, a, b, level=0.90, **kw)
    return dict(diff=d, lo90=lo, hi90=hi, margin=margin, equivalent=bool(lo > -margin and hi < margin))


def poisson_ll(pred, lam_col):
    lam = np.clip(pred[lam_col].values, 1e-9, None)
    n = pred.n_mc.values
    from scipy.special import gammaln
    return float(np.mean(n * np.log(lam) - lam - gammaln(n + 1)))


def calibration(y, p, bins=(0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """Reliability table + logistic calibration intercept and slope
    (fit of y on logit(p); perfect = intercept 0, slope 1)."""
    import statsmodels.api as sm
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    cut = pd.cut(p, bins, include_lowest=True)
    rel = pd.DataFrame(dict(y=y, p=p, bin=cut)).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_p=("p", "mean"), obs_rate=("y", "mean"))
    lp = np.log(p / (1 - p))
    try:
        fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        icpt, slope = fit.params
    except Exception:
        icpt = slope = np.nan
    return rel.reset_index(), dict(intercept=float(icpt), slope=float(slope),
                                   expected=float(p.sum()), observed=float(y.sum()))
