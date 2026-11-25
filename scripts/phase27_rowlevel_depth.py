"""Row-level correction of the depth-to-basement feature (final review, highest priority).

The corrected runs (phase 13/16) and protocol v3 build injection features from records up
to the test year's last usable month. Volumes, pressure and well counts depend only on each
row's own usable month, but the volume-weighted depth to basement (d_basement_w) uses well
attributes -- and the median used to fill missing depths -- from all records up to that
last month. Here d_basement_w is recomputed for each usable month m from records <= m only,
and the primary comparisons are rerun.

    python ../scripts/phase27_rowlevel_depth.py depth      # per-basin monthly table
    python ../scripts/phase27_rowlevel_depth.py dev        # 12-month development runs
    python ../scripts/phase27_rowlevel_depth.py test2026   # protocol v3 test, sensitivity only
    python ../scripts/phase27_rowlevel_depth.py score
Outputs: results/phase27_rowlevel_depth/. The protocol results stay as run.
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
import phase16_clean_dev as P16
from induced import basement, config, grid, injection, multibasin as MB

OUT = config.RESULTS / "phase27_rowlevel_depth"
OUT.mkdir(parents=True, exist_ok=True)
LAG = 12
ALL5 = ["oklahoma", "kansas", "delaware", "midland", "fortworth"]
LAST = pd.Timestamp("2025-09-01")      # last month any grid in this study uses at a 12-month lag


def usable_month(starts, lag=LAG):
    s = pd.DatetimeIndex(starts)
    return s.to_period("M").to_timestamp() - pd.DateOffset(months=1 + lag)


def depth(name):
    """d_basement_w[month, cell] using only injection records up to that month."""
    f = OUT / f"dbm_{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    box = MB.BASINS[name]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    raw = P13.raw_injection(name)
    cells = grid.make_cells(box, 0.2)
    months = pd.date_range(raw.date.min(), min(raw.date.max(), LAST), freq="MS")
    rows = []
    for m in months:
        # same order as P13.patch_injection: cut at m first, so the Oklahoma disposal-well
        # classification (any record typed 2D*/SWD) also uses only records up to m
        inj = raw[raw.date <= m]
        if name == "oklahoma":
            inj = injection.swd_only(inj)
        inj = injection.in_box(inj, pad)
        if len(inj) == 0:
            continue
        wells = basement.well_table(inj)
        mm, vol, psi, dbm, nwell = grid.monthly_kernel_injection(inj, wells, cells)
        if mm[-1] != m:                      # no record in month m itself
            continue
        rows.append(pd.DataFrame(dict(month=m, cell=cells.cell.values, d_basement_w=dbm[-1])))
    t = pd.concat(rows, ignore_index=True)
    t.to_parquet(f)
    print("depth", name, len(months), "months", flush=True)
    return t


def apply(g, name, lag=LAG):
    """Replace d_basement_w in grid rows with the row-level value for each row's usable month."""
    t = depth(name)
    key = pd.DataFrame(dict(month=usable_month(g.start.values, lag), cell=g.cell.values.astype(int)))
    v = key.merge(t, on=["month", "cell"], how="left").d_basement_w.values
    g = g.copy()
    g["d_basement_w_orig"] = g["d_basement_w"].values
    g["d_basement_w"] = v
    return g


_patch = P13.patch_injection


def patched(g, name, last_month, lag):
    return apply(_patch(g, name, last_month, lag), name, lag)


def dev():
    P13.patch_injection = patched
    base = {n: MB.load_basin(n, LAG) for n in P16.ALL4}
    diffs, res = [], {n: [] for n in P16.ALL4}
    for year in range(2014, 2026):
        cache = {}
        for target in P16.ALL4:
            y0, y1 = MB.BASINS[target]["test"]
            if not y0 <= year <= y1:
                continue
            gt = base[target][0]
            te = gt.start.dt.year.values == year
            uniq = np.sort(gt.start.unique())
            p0 = np.searchsorted(uniq, gt.start.values[te].min())
            cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
            MB._ETAS_MEMO.clear()
            data = P16.fold_data(base, year, LAG, cutoff, True, cache)
            g = data[target][0]
            d = (g.d_basement_w - g.d_basement_w_orig).abs()
            diffs.append(dict(basin=target, year=year, max_abs_change_m=float(d.max()),
                              share_changed=float((d > 1e-6).mean())))
            res[target].append(MB.run(target, data, [n for n in P16.ALL4 if n in data], (year, year),
                                      local_ml=True, lobo=True, ablation=False, verbose=False))
            print("dev", target, year, flush=True)
    for n, r in res.items():
        pd.concat(r, ignore_index=True).to_parquet(OUT / f"pred_lag12_{n}.parquet")
    pd.DataFrame(diffs).to_csv(OUT / "feature_change.csv", index=False)


def test2026():
    import phase12_forward2026 as P12
    _load = P12.load

    def load(name, until):
        g, B, T, cat = _load(name, until)
        return apply(g, name), B, T, cat

    P12.load = load
    out = OUT / "test_2026"
    out.mkdir(exist_ok=True)
    print(json.dumps(P12.evaluate(2026, "2026-11-01", out)["primary"], default=float))


def score():
    rows = []
    pairs = (("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"), ("hybrid_count_pooled", "hybrid_count_local"),
             ("lgbm_cls_lobo", "etas"))
    for n in P16.ALL4:
        for tag, f in (("reported", config.RESULTS / "phase16_clean_dev" / f"pred_lag12_{n}.parquet"),
                       ("row-level depth", OUT / f"pred_lag12_{n}.parquet")):
            d = pd.read_parquet(f)            # stored row order, 500 resamples, as Table 6
            for a, c in pairs:
                x, lo, hi = MB.paired_diff(d, "ex3", a, c, n_boot=500)
                rows.append(dict(basin=n, version=tag, comparison=f"{a} - {c}", diff=x, lo=lo, hi=hi))
            s = MB.score(d.sort_values(["start", "cell"]).reset_index(drop=True), "ex3", n_boot=10,
                         models=["hybrid_count_lobo", "hybrid_count_pooled", "lgbm_cls_lobo", "etas"])
            for _, r in s.iterrows():
                rows.append(dict(basin=n, version=tag, comparison=f"PR-AUC {r.model}", diff=r.prauc))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "comparison.csv", index=False)
    pd.set_option("display.width", 220)
    w = t.assign(txt=t["diff"].map("{:+.3f}".format) + t.lo.map(lambda v: "" if pd.isna(v) else f" [{v:.3f}, ")
                 + t.hi.map(lambda v: "" if pd.isna(v) else f"{v:.3f}]"))
    txt = w.pivot_table(index=["basin", "comparison"], columns="version", values="txt", aggfunc="first").to_string()
    f26 = OUT / "test_2026" / "decision.json"
    if f26.exists():
        a = json.load(open(f26))["primary"]
        b = json.load(open(config.RESULTS / "phase12_forward2026" / "test_2026" / "decision.json"))["primary"]
        txt += f"\n\n2026 test: reported {b['diff']:+.4f} {b['ci90']} {b['verdict']}; row-level {a['diff']:+.4f} {a['ci90']} {a['verdict']}"
    fc = OUT / "feature_change.csv"
    if fc.exists():
        txt += "\n\nFeature change by basin (max |change| m, share of rows changed)\n" + \
               pd.read_csv(fc).groupby("basin").agg(max_m=("max_abs_change_m", "max"),
                                                    share=("share_changed", "mean")).round(3).to_string()
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "depth":
        for n in (sys.argv[2:] or ALL5):
            depth(n)
    elif cmd == "dev":
        dev()
    elif cmd == "test2026":
        test2026()
    else:
        score()
