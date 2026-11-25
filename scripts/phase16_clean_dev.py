"""Look-ahead-free rerun of the phase 9 walk-forward runs (Oklahoma, Kansas,
Delaware, Midland; lags 0, 3, 12 months). Post hoc sensitivity analysis.

Same models, settings, pooled basins and test years as phase9_revision_runs.py.
For each test year, in every basin:
  * injection records after the last month usable in that year are removed
    before SWD selection, well attributes, medians and injection features
  * the Zaliapin-Ben-Zion threshold is fitted only on events before the fold's
    training cutoff
A control uses the original inputs through this same loop and must reproduce
the phase 9 forecasts.

    python ../scripts/phase16_clean_dev.py check
    python ../scripts/phase16_clean_dev.py run <lag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import phase13_leak_corrected as P13
from induced import config, multibasin as MB

OUT = config.RESULTS / "phase16_clean_dev"
OUT.mkdir(parents=True, exist_ok=True)
ALL4 = ["oklahoma", "kansas", "delaware", "midland"]
INJ = ["v1", "v3", "v9", "v12", "vcum", "v_rate_change", "psi_w", "d_basement_w", "n_wells", "si_sigma", "si_lam3"]


def check():
    """With nothing removed, the rebuilt injection features should match the grid."""
    rows = []
    for lag in (0, 12):
        for n in ALL4:
            g, _, _, _ = MB.load_basin(n, lag)
            last = pd.Timestamp("2030-01-01")
            p = P13.patch_injection(g, n, last, lag)
            m = g.merge(p, on=["start", "cell"], suffixes=("", "_p"))
            for c in INJ:
                if c in g and f"{c}_p" in m:
                    a, b = m[c].astype(float).values, m[f"{c}_p"].astype(float).values
                    ok = np.isfinite(a) & np.isfinite(b)
                    rows.append(dict(basin=n, lag=lag, feature=c, rows=len(m), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                                     max_abs=float(np.abs(a[ok] - b[ok]).max()) if ok.any() else np.nan,
                                     frac_diff=float((np.abs(a[ok] - b[ok]) > 1e-6 * (1 + np.abs(a[ok]))).mean()) if ok.any() else np.nan))
            print(n, lag, "checked", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "rebuild_check.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(r.round(6).to_string(index=False))


def fold_data(base, year, lag, cutoff, clean, cache):
    data = {}
    until = np.datetime64(f"{year + 1}-01-01")
    last_month = pd.Timestamp(year=year, month=12, day=1) - pd.DateOffset(months=1 + lag)
    for n in ALL4:
        g, B, T, cat = base[n]
        keep = g.start.values < until
        # A basin with no usable rows yet stays in `data` with an empty grid: phase 9
        # pooled its catalog into the empirical magnitude ratio.
        if not keep.any():
            data[n] = (g.iloc[:0], B[:0], T[:, :, :0], cat)
            continue
        g2, B2, T2 = g[keep].reset_index(drop=True), B[keep], T[:, :, keep]
        if clean:
            if pd.Timestamp(g2.start.min()) > last_month:
                data[n] = (g.iloc[:0], B[:0], T[:, :, :0], cat)
                continue
            if n not in cache:
                n_mc = g2.n_mc.values
                p = P13.patch_injection(g2, n, last_month, lag)
                p["n_mc"], p["basin"] = n_mc, n
                cache[n] = p
            g2 = cache[n]
            B2, used = P13.clean_background(g2, n, cat, cutoff)
            if not used:
                print("  no declustering", n, year, flush=True)
        data[n] = (g2, B2, T2, cat)
    return data


def run(lag):
    base = {n: MB.load_basin(n, lag) for n in ALL4}
    res = {n: dict(control=[], clean=[]) for n in ALL4}
    for year in range(2014, 2026):
        cache = {}
        for target in ALL4:
            y0, y1 = MB.BASINS[target]["test"]
            if not y0 <= year <= y1:
                continue
            gt = base[target][0]
            te = gt.start.dt.year.values == year
            uniq = np.sort(gt.start.unique())
            p0 = np.searchsorted(uniq, gt.start.values[te].min())
            cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
            for tag, clean in (("control", False), ("clean", True)):
                MB._ETAS_MEMO.clear()
                data = fold_data(base, year, lag, cutoff, clean, cache)
                use = [n for n in ALL4 if n in data]
                res[target][tag].append(MB.run(target, data, use, (year, year), local_ml=True, lobo=True,
                                               ablation=True, verbose=False))
            print(lag, target, year, "done", flush=True)
    checks = {}
    for n in ALL4:
        ctrl = pd.concat(res[n]["control"], ignore_index=True)
        clean = pd.concat(res[n]["clean"], ignore_index=True)
        ctrl.to_parquet(OUT / f"control_lag{lag}_{n}.parquet")
        clean.to_parquet(OUT / f"pred_lag{lag}_{n}.parquet")
        o = pd.read_parquet(config.RESULTS / "phase9" / f"pred_lag{lag}_{n}.parquet")
        m = ctrl.merge(o, on=["start", "cell"], suffixes=("", "_o"))
        cols = [c for c in ctrl.columns if c.endswith(("_ex3", "_ex4")) and f"{c}_o" in m]
        checks[n] = dict(rows=len(m), rows_original=len(o),
                         max_abs={c: float((m[c] - m[f"{c}_o"]).abs().max()) for c in cols})
        print(n, "control max difference", max(v for v in checks[n]["max_abs"].values() if v == v), flush=True)
    (OUT / f"control_check_lag{lag}.json").write_text(json.dumps(checks, indent=1))


def repair():
    """Recompute only hybrid_count_emp for Oklahoma 2014-2016 in files written before
    the empty-basin fix in fold_data. Everything else in those runs matched phase 9."""
    cats = {n: (None, None, None, MB.basin_catalog(n)) for n in ALL4}
    uniq = np.sort(MB.load_basin("oklahoma", 0)[0].start.unique())
    for lag in (0, 3, 12):
        o = pd.read_parquet(config.RESULTS / "phase9" / f"pred_lag{lag}_oklahoma.parquet")
        fixed = {}
        for name in (f"control_lag{lag}_oklahoma.parquet", f"pred_lag{lag}_oklahoma.parquet"):
            d = pd.read_parquet(OUT / name)
            yrs = pd.DatetimeIndex(d.start).year.values
            for year in (2014, 2015, 2016):
                r = yrs == year
                p0 = np.searchsorted(uniq, np.datetime64(f"{year}-01-01"))
                cutoff = uniq[max(0, p0 - MB.GAP)]
                for m, col in ((3, "ex3"), (4, "ex4")):
                    ratio = MB._empirical_ratio(ALL4, cats, cutoff, m)
                    d.loc[r, f"hybrid_count_emp_{col}"] = 1 - np.exp(-d.lam_hybrid_pooled.values[r] * ratio)
            fixed[name] = d
        c = fixed[f"control_lag{lag}_oklahoma.parquet"].merge(o, on=["start", "cell"], suffixes=("", "_o"))
        cols = [x for x in fixed[f"control_lag{lag}_oklahoma.parquet"].columns
                if x.endswith(("_ex3", "_ex4")) and f"{x}_o" in c]
        worst = max(float((c[x] - c[f"{x}_o"]).abs().max()) for x in cols)
        print("lag", lag, "control max difference after repair", worst, flush=True)
        if worst > 1e-12:
            raise SystemExit("control still differs; files not changed")
        for name, d in fixed.items():
            d.to_parquet(OUT / name)
        chk = json.loads((OUT / f"control_check_lag{lag}.json").read_text())
        chk["oklahoma"]["max_abs"] = {x: float((c[x] - c[f"{x}_o"]).abs().max()) for x in cols}
        (OUT / f"control_check_lag{lag}.json").write_text(json.dumps(chk, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "repair":
        repair()
    else:
        run(int(sys.argv[2]))
