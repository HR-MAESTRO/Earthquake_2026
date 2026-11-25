"""Temporal-cutoff audit (third-round review 4.3).

    python ../scripts/phase26_temporal_audit.py

Empirical causality checks. If a value for a period starting at s uses only
information available before s, recomputing it from inputs cut at s must give the
same number.
  1. Fold cutoffs: every training row starts before its cutoff, and the cutoff is at
     least 18 half-months before the first test period (all basins and years, as
     phase 16).
  2. ETAS triggering terms T: recomputed from a catalog cut at X, for periods
     starting before X.
  3. Corrected background B (phase 13): recomputed from a catalog cut at X, with the
     declustering threshold fitted before the cutoff X.
  4. Seismicity grid features: grid rebuilt from a catalog cut at X (Kansas,
     Delaware), against the stored 12-month-lag grid, for periods starting before X.
  5. Injection features as used in the corrected runs (patched to the test year's
     last usable month), against a patch cut at each row's own usable month: any
     difference measures within-year use of later well records.
Writes results/phase26_audit/audit.json and summary.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import build_lagged_grid as BLG
import phase13_leak_corrected as P13
import phase16_clean_dev as P16
from induced import basement, catalog, config, etas, grid, injection, multibasin as MB, regions

OUT = config.RESULTS / "phase26_audit"
OUT.mkdir(parents=True, exist_ok=True)
X = {"oklahoma": "2018-01-01", "kansas": "2020-07-01", "delaware": "2022-01-01", "midland": "2023-07-01"}
SEIS = ["n30_near", "n90_near", "n365_near", "n365_cell", "mmax365_near", "days_since_m3", "b_ml"]
INJ = ["v1", "v3", "v9", "v12", "vcum", "v_rate_change", "psi_w", "d_basement_w", "n_wells"]
res = {}


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    return (float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0), nan_mismatch


# 1. fold cutoffs ------------------------------------------------------------------------
recorded = []
_orig_pooled = MB._pooled_set


def spy(names, data, cutoff):
    for n in names:
        g = data[n][0]
        tr = g.start.values[g.start.values < cutoff]
        if len(tr):
            recorded.append(dict(basin=n, cutoff=str(cutoff)[:10], max_train_start=str(tr.max())[:10]))
    return _orig_pooled(names, data, cutoff)


MB._pooled_set = spy
fold_rows = []
base = {n: MB.load_basin(n, 12) for n in P16.ALL4}
for target in P16.ALL4:
    gt = base[target][0]
    uniq = np.sort(gt.start.unique())
    y0, y1 = MB.BASINS[target]["test"]
    for year in range(y0, y1 + 1):
        te = gt.start.dt.year.values == year
        if not te.any():
            continue
        p0 = np.searchsorted(uniq, gt.start.values[te].min())
        cutoff = uniq[max(0, p0 - MB.GAP)]
        gap = p0 - np.searchsorted(uniq, cutoff)
        tr = gt.start.values < cutoff
        fold_rows.append(dict(basin=target, year=year, cutoff=str(cutoff)[:10], gap_periods=int(gap),
                              train_before_cutoff=bool(gt.start.values[tr].max() < cutoff) if tr.any() else True))
folds = pd.DataFrame(fold_rows)
res["1_folds"] = dict(n_folds=len(folds), all_train_before_cutoff=bool(folds.train_before_cutoff.all()),
                      min_gap_periods=int(folds.gap_periods[folds.gap_periods > 0].min()),
                      folds_with_gap_below_18=int((folds.gap_periods < 18).sum()))
# the pooled training set used by MB.run (exercised on one fold per basin)
for target in P16.ALL4:
    y0 = MB.BASINS[target]["test"][0] + 1
    gt = base[target][0]
    uniq = np.sort(gt.start.unique())
    p0 = np.searchsorted(uniq, gt.start.values[gt.start.dt.year.values == y0].min())
    cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
    MB._ETAS_MEMO.clear()
    data = P16.fold_data(base, y0, 12, cutoff, True, {})
    MB.run(target, data, [n for n in P16.ALL4 if n in data], (y0, y0), local_ml=False, lobo=False,
           ablation=False, verbose=False)
rec = pd.DataFrame(recorded)
res["1_pooled_training_rows"] = dict(checked=len(rec), all_before_cutoff=bool((rec.max_train_start < rec.cutoff).all()))
MB._pooled_set = _orig_pooled
print("1", res["1_folds"], res["1_pooled_training_rows"], flush=True)


# 2 + 3. ETAS components and corrected background under catalog truncation ---------------
def nn_fresh(name, cat):
    c = cat[cat.mag >= P13.MC].sort_values("time").reset_index(drop=True)
    return catalog.nn_distance(c)[0]


P13.nn_values = nn_fresh
for n in P16.ALL4:
    g, _, _, cat = base[n]
    x = np.datetime64(X[n])
    rows = g[(g.start.values < x) & (g.start.values >= x - np.timedelta64(730, "D"))].reset_index(drop=True)
    cells = grid.make_cells(MB.BASINS[n]["box"], 0.2)
    cut = cat[cat.time.values < x]
    _, T_full, _ = etas.components(rows, cat, cells)
    _, T_cut, _ = etas.components(rows, cut, cells)
    B_full, _ = P13.clean_background(rows, n, cat, pd.Timestamp(X[n]))
    B_cut, _ = P13.clean_background(rows, n, cut, pd.Timestamp(X[n]))
    res[f"2_T_{n}"] = dict(rows=len(rows), max_abs_diff=maxdiff(T_full.ravel(), T_cut.ravel())[0])
    res[f"3_B_{n}"] = dict(rows=len(rows), max_abs_diff=maxdiff(B_full, B_cut)[0])
    print("2/3", n, res[f"2_T_{n}"], res[f"3_B_{n}"], flush=True)

# 4. seismicity grid features under catalog truncation -----------------------------------
for n in ("kansas", "delaware"):
    box = MB.BASINS[n]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    cat = regions.comcat(n)
    inj = injection.in_box(getattr(regions, f"{n}_injection")(), pad)
    wells = basement.well_table(inj)
    x = pd.Timestamp(X[n])
    gcut, _ = grid.build(cat[cat.time < x], inj, wells, box, BLG.STARTS[n], str((x - pd.Timedelta(days=1)).date()),
                         use_faults=False, inj_lag_months=12)
    stored = pd.read_parquet(config.PROCESSED / f"grid_{n}_lag12.parquet")
    m = gcut.merge(stored, on=["start", "cell"], suffixes=("_cut", ""))
    m = m[m.start < x]
    res[f"4_seis_{n}"] = {c: maxdiff(m[f"{c}_cut"], m[c]) for c in SEIS + ["si_sigma"]}
    res[f"4_seis_{n}"]["rows"] = len(m)
    print("4", n, res[f"4_seis_{n}"], flush=True)

# 5. injection features: test-year patch vs each row's own usable month -------------------
for n in P16.ALL4:
    g = base[n][0]
    year = int(X[n][:4])
    last_year = pd.Timestamp(year=year, month=12, day=1) - pd.DateOffset(months=13)
    yr_rows = g[g.start.dt.year == year].reset_index(drop=True)
    pa = P13.patch_injection(yr_rows, n, last_year, 12)
    diffs = {c: 0.0 for c in INJ}
    for month in (1, 4, 7):
        s = pd.Timestamp(year=year, month=month, day=1)
        own = s - pd.DateOffset(months=13)
        rows = yr_rows[yr_rows.start == s].reset_index(drop=True)
        pb = P13.patch_injection(rows, n, own, 12)
        a = pa[pa.start == s].sort_values("cell").reset_index(drop=True)
        b = pb.sort_values("cell").reset_index(drop=True)
        for c in INJ:
            diffs[c] = max(diffs[c], maxdiff(a[c], b[c])[0])
    res[f"5_inj_{n}"] = diffs
    print("5", n, diffs, flush=True)

(OUT / "audit.json").write_text(json.dumps(res, indent=1, default=str))
folds.to_csv(OUT / "folds.csv", index=False)
print("done", flush=True)
