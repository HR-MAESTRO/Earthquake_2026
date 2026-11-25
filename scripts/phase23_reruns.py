"""Second-round review reruns (look-ahead-free folds, as phase 16).

    python ../scripts/phase23_reruns.py offset     # review 3.3
    python ../scripts/phase23_reruns.py mc3        # review 4.2
    python ../scripts/phase23_reruns.py permian    # review 3.2 (optional item)

offset : pooled Poisson LightGBM with the hybrid's inputs (ETAS log-rate and SI
         included as features) but WITHOUT the ETAS offset ("count_feat"); Gutenberg-
         Richter conversion as for the hybrid. The pooled hybrid is refitted in the
         same fold as a control against phase 16. Lags 0 and 12, four development basins.
mc3    : the whole pooled pipeline with Mc = 3.0 for the ETAS components, background,
         count target, b-value and exceedance conversion, in all four development basins
         (so pooled and LOBO training are consistent). Predictions for Delaware
         (2019-2025) and Midland (2020-2025) at the 12-month lag. Seismicity input
         features and the SI baseline stay at M2.5 (grid as built).
permian: the ML part trained on ONE Permian basin only and applied to the other
         (Delaware -> Midland, Midland -> Delaware), 12-month lag.
Outputs: results/phase23_reruns/
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import phase13_leak_corrected as P13
import phase16_clean_dev as P16
from induced import catalog, config, etas, multibasin as MB
from induced.gridmodels import prep

OUT = config.RESULTS / "phase23_reruns"
OUT.mkdir(parents=True, exist_ok=True)
ALL4 = P16.ALL4


def folds(base, target, lag):
    """Yield (year, cutoff, clean data) for each walk-forward test year of `target`."""
    gt = base[target][0]
    y0, y1 = MB.BASINS[target]["test"]
    for year in range(y0, y1 + 1):
        te = gt.start.dt.year.values == year
        if not te.any():
            continue
        uniq = np.sort(gt.start.unique())
        p0 = np.searchsorted(uniq, gt.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        MB._ETAS_MEMO.clear()
        yield year, cutoff, P16.fold_data(base, year, lag, cutoff, True, {})


# ---------------------------------------------------------------- offset-only control
def offset(lag):
    base = {n: MB.load_basin(n, lag) for n in ALL4}
    res = []
    for target in ALL4:
        for year, cutoff, data in folds(base, target, lag):
            g, B, T, cat = data[target]
            te = np.where(g.start.dt.year.values == year)[0]
            prm = MB._etas_fit(target, data, cutoff)
            lam = etas.rate(B, T, prm)
            b = etas.b_value(cat, pd.Timestamp(f"{year}-01-01"))
            Xte = prep(g.iloc[te], MB.FEATS).assign(etas_loglam=np.log(lam[te] + 1e-6))
            P = MB._pooled_set([n for n in ALL4 if n in data], data, cutoff)
            r = pd.DataFrame(dict(basin=target, start=g.start.values[te], cell=g.cell.values[te],
                                  ex3=g.ex3.values[te], ex4=g.ex4.values[te], n_mc=g.n_mc.values[te]))
            hyb = MB._lgb_reg().fit(P["X"], P["y"], init_score=P["off"])
            lam_h = np.exp(hyb.predict(Xte, raw_score=True) + np.log(lam[te] + 1e-6))
            feat = MB._lgb_reg().fit(P["X"], P["y"])                     # same inputs, no offset
            lam_f = feat.predict(Xte)
            r["lam_hybrid_ctrl"], r["lam_count_feat"] = lam_h, lam_f
            for m, col in ((3, "ex3"), (4, "ex4")):
                r[f"hybrid_ctrl_{col}"] = etas.exceed_prob(lam_h, b, m)
                r[f"count_feat_{col}"] = etas.exceed_prob(lam_f, b, m)
            res.append(r)
            print("offset", lag, target, year, flush=True)
    p = pd.concat(res, ignore_index=True)
    p.to_parquet(OUT / f"offset_lag{lag}.parquet")


# ---------------------------------------------------------------- Mc = 3.0
MC3 = 3.0


def use_mc3():
    etas.b_value = functools.partial(_b_value, mc=MC3)
    etas.exceed_prob = functools.partial(_exceed, mc=MC3)
    P13.MC = MC3

    def nn_values(name, cat):
        path = OUT / f"nn3_{name}.npy"
        if path.exists():
            return np.load(path)
        c = cat[cat.mag >= MC3].sort_values("time").reset_index(drop=True)
        le, _ = catalog.nn_distance(c)
        np.save(path, le)
        return le

    P13.nn_values = nn_values


_b_value, _exceed = etas.b_value, etas.exceed_prob


def load_basin3(name, lag=12):
    g0 = MB._grid(name, 0)
    cat = MB.basin_catalog(name)
    cache = OUT / f"mb3_{name}.npz"
    if cache.exists():
        z = np.load(cache)
        B, T, n = z["B"], z["T"], z["n"]
    else:
        B, T, n = etas.components(g0, cat, MB.make_cells(MB.BASINS[name]["box"], 0.2), mc=MC3)
        np.savez(cache, B=B, T=T, n=n)
    g = MB._grid(name, lag)
    pos = pd.Series(np.arange(len(g0)), index=pd.MultiIndex.from_arrays([g0.cell, g0.start]))
    idx = pos.reindex(pd.MultiIndex.from_arrays([g.cell, g.start])).values
    keep = ~np.isnan(idx)
    g, idx = g[keep].reset_index(drop=True), idx[keep].astype(int)
    g = g.copy()
    g["n_mc"], g["basin"] = n[idx], name
    return g, B[idx], T[:, :, idx], cat


def mc3(targets=("delaware", "midland")):
    use_mc3()
    base = {n: load_basin3(n) for n in ALL4}
    for target in targets:
        res = []
        for year, cutoff, data in folds(base, target, 12):
            args = (target, data, [n for n in ALL4 if n in data], (year, year))
            try:
                r = MB.run(*args, local_ml=True, lobo=True, ablation=False, verbose=False)
            except Exception as e:   # no M>=3 events in the local training rows: skip the local model only
                print("mc3", target, year, "local model skipped:", str(e)[:60], flush=True)
                r = MB.run(*args, local_ml=False, lobo=True, ablation=False, verbose=False)
            res.append(r)
            print("mc3", target, year, flush=True)
        pd.concat(res, ignore_index=True).to_parquet(OUT / f"mc3_{target}.parquet")


# ---------------------------------------------------------------- single-basin Permian transfer
def permian():
    base = {n: MB.load_basin(n, 12) for n in ALL4}
    for target, source in (("midland", "delaware"), ("delaware", "midland")):
        res = []
        for year, cutoff, data in folds(base, target, 12):
            d = {k: v for k, v in data.items() if k in (target, source)}
            r = MB.run(target, d, [source], (year, year), local_ml=False, lobo=False, ablation=False,
                       verbose=False)
            res.append(r.rename(columns=lambda c: c.replace("_pooled", f"_from_{source}")))
            print("permian", source, "->", target, year, flush=True)
        pd.concat(res, ignore_index=True).to_parquet(OUT / f"single_source_{source}_to_{target}.parquet")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "offset":
        for lag in (0, 12):
            offset(lag)
    elif cmd == "mc3":
        mc3(tuple(sys.argv[2:]) or ("delaware", "midland"))
    else:
        permian()
