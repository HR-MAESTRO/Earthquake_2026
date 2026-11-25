"""Look-ahead-corrected sensitivity rerun of the Midland (v1) and Fort Worth (v2) tests.

The pre-registered results stay as run. This script measures how much three
look-ahead sources changed them. For each test year, in every basin:
  * injection records after the last month usable in that year are removed
    before SWD selection, well attributes, medians and injection features
  * the Zaliapin-Ben-Zion threshold is fitted only on events before the fold's
    training cutoff (fewer than 50 prior events: no declustering in that fold)
Seismicity features, labels, model settings and decision rules are unchanged.
Texas well type still comes from the current RRC table.

A control run uses the original inputs through this same loop and must
reproduce the original predictions.

    python ../scripts/phase13_leak_corrected.py prep
    python ../scripts/phase13_leak_corrected.py run midland 0
    python ../scripts/phase13_leak_corrected.py run fortworth <lag>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import basement, catalog, config, etas, grid, injection, metrics, multibasin as MB, regions

OUT = config.RESULTS / "phase13_leak_corrected"
OUT.mkdir(parents=True, exist_ok=True)
ALL5 = ["oklahoma", "kansas", "delaware", "midland", "fortworth"]
DEV = ["oklahoma", "kansas", "delaware"]
MC = etas.MC
_RAW = {}


def nn_values(name, cat):
    path = OUT / f"nn_{name}.npy"
    if path.exists():
        return np.load(path)
    c = cat[cat.mag >= MC].sort_values("time").reset_index(drop=True)
    le, _ = catalog.nn_distance(c)
    np.save(path, le)
    return le


def raw_injection(name):
    if name not in _RAW:
        _RAW[name] = injection.build() if name == "oklahoma" else getattr(regions, f"{name}_injection")()
    return _RAW[name]


def clean_background(g, name, cat, cutoff):
    box = MB.BASINS[name]["box"]
    cells = grid.make_cells(box, 0.2)
    c = cat[cat.mag >= MC].sort_values("time").reset_index(drop=True)
    le = nn_values(name, cat)
    before = c.time.values < np.datetime64(cutoff)
    used = int(np.isfinite(le[before]).sum()) >= 50
    bg = ~(le < catalog.nn_threshold(le[before])) if used else np.ones(len(c), bool)
    tday = (c.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    nlat, nlon = cells.lat0.nunique(), cells.lon0.nunique()
    ii = np.floor((c.lat.values - cells.lat0.min()) / 0.2).astype(int)
    jj = np.floor((c.lon.values - cells.lon0.min()) / 0.2).astype(int)
    ev = np.where((ii >= 0) & (jj >= 0) & (ii < nlat) & (jj < nlon), ii * nlon + jj, -1)
    B = np.zeros(len(g))
    starts, ends, ids = g.start.values, g.end.values, g.cell.values.astype(int)
    for s in np.unique(starts):
        rows = np.where(starts == s)[0]
        sd = (s - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        ed = (ends[rows[0]] - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        k0, k = np.searchsorted(tday, sd - 365), np.searchsorted(tday, sd)
        e, b = ev[k0:k], bg[k0:k]
        counts = np.bincount(e[(e >= 0) & b], minlength=len(cells))
        B[rows] = (counts[ids[rows]] + 0.1) / 365.0 * (ed - sd)
    return B, used


def patch_injection(g, name, last_month, lag):
    box = MB.BASINS[name]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    raw = raw_injection(name)
    raw = raw[raw.date <= last_month]
    if name == "oklahoma":
        raw = injection.swd_only(raw)
    inj = injection.in_box(raw, pad)
    wells = basement.well_table(inj)
    cells = grid.make_cells(box, 0.2)
    months, vol, psi, dbm, nwell = grid.monthly_kernel_injection(inj, wells, cells)
    cum = np.cumsum(vol, axis=0)
    pos = {m: i for i, m in enumerate(months)}
    g = g.copy()
    cols = ["v1", "v3", "v9", "v12", "vcum", "v_rate_change", "psi_w", "d_basement_w", "n_wells"]
    out = {c: np.full(len(g), np.nan) for c in cols}
    ids = g.cell.values.astype(int)
    for s in g.start.unique():
        rows = np.where(g.start.values == np.datetime64(s))[0]
        mi = pos.get(pd.Timestamp(s).replace(day=1) - pd.DateOffset(months=1 + lag))
        if mi is None:                          # no SWD record yet in this window
            for c in ("v1", "v3", "v9", "v12", "vcum", "n_wells"):
                out[c][rows] = 0.0
            out["v_rate_change"][rows] = 1.0
            continue
        vs = {n: vol[max(0, mi - n + 1):mi + 1].sum(0) for n in (1, 3, 9, 12)}
        prev6 = vol[max(0, mi - 8):max(0, mi - 2)].sum(0)
        c_ = ids[rows]
        for n in (1, 3, 9, 12):
            out[f"v{n}"][rows] = vs[n][c_]
        out["vcum"][rows] = cum[mi, c_]
        out["v_rate_change"][rows] = ((vs[3] / 3 + 1) / (prev6 / 6 + 1))[c_]
        out["psi_w"][rows], out["d_basement_w"][rows], out["n_wells"][rows] = psi[mi, c_], dbm[mi, c_], nwell[mi, c_]
    for c in cols:
        g[c] = out[c]
    return grid.add_seismogenic_index(g).sort_values(["start", "cell"]).reset_index(drop=True)


def fold_data(base, names, year, lag, cutoff, clean):
    data, fallback = {}, []
    until = np.datetime64(f"{year + 1}-01-01")
    last_month = pd.Timestamp(year=year, month=12, day=1) - pd.DateOffset(months=1 + lag)
    for n in names:
        g, B, T, cat = base[n]
        keep = g.start.values < until
        if not keep.any():
            continue
        g2, B2, T2 = g[keep].reset_index(drop=True), B[keep], T[:, :, keep]
        if clean:
            if pd.Timestamp(g2.start.min()) > last_month:
                continue
            n_mc = g2.n_mc.values
            g2 = patch_injection(g2, n, last_month, lag)
            g2["n_mc"], g2["basin"] = n_mc, n
            B2, used = clean_background(g2, n, cat, cutoff)
            if not used:
                fallback.append(n)
        data[n] = (g2, B2, T2, cat)
    return data, fallback


def run_spec(base, target, lag, clean):
    y0, y1 = MB.BASINS[target]["test"]
    names = ALL5 if target == "fortworth" else DEV + ["midland"]
    pooled = ALL5 if target == "fortworth" else DEV
    is_fw = target == "fortworth"
    gt = base[target][0]
    uniq = np.sort(gt.start.unique())
    preds, log = [], []
    for year in range(y0, y1 + 1):
        te = gt.start.dt.year.values == year
        if not te.any():
            continue
        p0 = np.searchsorted(uniq, gt.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        MB._ETAS_MEMO.clear()
        data, fallback = fold_data(base, names, year, lag, cutoff, clean)
        use = [n for n in pooled if n in data]
        preds.append(MB.run(target, data, use, (year, year), local_ml=is_fw, lobo=is_fw, ablation=False, verbose=False))
        log.append(dict(year=year, cutoff=str(cutoff.date()), no_declustering=fallback))
        print(target, lag, "clean" if clean else "control", year, "done", flush=True)
    return pd.concat(preds, ignore_index=True), log


def decide(pred, target):
    model = "hybrid_count_lobo" if target == "fortworth" else "hybrid_count_pooled"
    s = MB.score(pred, "ex3", models=["etas", "si", model])
    b = s[s.model.isin(["etas", "si"])].sort_values("prauc", ascending=False).iloc[0]
    d, lo95, hi95 = MB.paired_diff(pred, "ex3", model, b.model, level=0.95)
    _, lo90, hi90 = MB.paired_diff(pred, "ex3", model, b.model, level=0.90)
    margin = 0.2 * b.prauc
    if target == "fortworth":
        verdict = ("superior" if lo95 > 0 else "equivalent" if lo90 > -margin and hi90 < margin
                   else "inferior" if hi95 < 0 else "inconclusive")
    else:
        verdict = "hybrid better" if lo95 > 0 else "hybrid worse" if hi95 < 0 else "no significant difference"
    return dict(model=model, best_baseline=b.model, prauc_baseline=float(b.prauc),
                prauc_model=float(s[s.model == model].prauc.iloc[0]),
                prauc_etas=float(s[s.model == "etas"].prauc.iloc[0]), prauc_si=float(s[s.model == "si"].prauc.iloc[0]),
                diff=d, ci95=[lo95, hi95], ci90=[lo90, hi90], margin=margin, verdict=verdict,
                positives=int(pred.ex3.sum()))


def original(target, lag):
    p = config.RESULTS / "phase8" / "pred_midland.parquet" if target == "midland" else \
        config.RESULTS / "phase10_fortworth" / f"pred_lag{lag}.parquet"
    return pd.read_parquet(p)


def main():
    if sys.argv[1] == "prep":
        for n in ALL5:
            nn_values(n, MB.basin_catalog(n))
            print("nn", n, flush=True)
        return
    target, lag = sys.argv[2], int(sys.argv[3])
    names = ALL5 if target == "fortworth" else DEV + ["midland"]
    base = {n: MB.load_basin(n, lag) for n in names}
    tag = OUT / f"{target}_lag{lag}"
    tag.mkdir(exist_ok=True)
    orig = original(target, lag)
    model = "hybrid_count_lobo" if target == "fortworth" else "hybrid_count_pooled"
    res = dict(original=decide(orig, target))
    ctrl, _ = run_spec(base, target, lag, False)
    ctrl.to_parquet(tag / "control.parquet")
    m = ctrl.merge(orig, on=["start", "cell"], suffixes=("", "_o"))
    res["control"] = decide(ctrl, target)
    res["control_check"] = dict(rows=len(m), rows_original=len(orig),
                                max_abs_etas=float((m.etas_ex3 - m.etas_ex3_o).abs().max()),
                                max_abs_model=float((m[f"{model}_ex3"] - m[f"{model}_ex3_o"]).abs().max()))
    clean, log = run_spec(base, target, lag, True)
    clean.to_parquet(tag / "corrected.parquet")
    res["corrected"] = decide(clean, target)
    res["fold_log"] = log
    (tag / "decision.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("original", "control", "control_check", "corrected")}, indent=1, default=float))


if __name__ == "__main__":
    main()
