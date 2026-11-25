"""Grid offset and cell-size sensitivity (post hoc, development basins).

Variants of the 0.2 deg grid, all at the primary 12-month injection lag and with
the look-ahead corrections of phase 16:
  offset : 0.2 deg cells, grid shifted by half a cell (+0.1 deg in lat and lon)
  size01 : 0.1 deg cells
  size04 : 0.4 deg cells
Same models, settings, pooled basins and test years as phase 16. Fort Worth is
not used (held out). PR-AUC is not comparable across cell sizes because base
rates change; the comparisons of interest are paired differences and rankings.

    python ../scripts/phase18_grid_sensitivity.py build <variant>
    python ../scripts/phase18_grid_sensitivity.py control
    python ../scripts/phase18_grid_sensitivity.py run <variant>
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

OUT = config.RESULTS / "phase18_grid"
GRIDS = OUT / "grids"
GRIDS.mkdir(parents=True, exist_ok=True)
LAG = 12
VARIANTS = {"base": (0.2, 0.0), "offset": (0.2, 0.1), "size01": (0.1, 0.0), "size04": (0.4, 0.0)}
BOX0 = {n: dict(MB.BASINS[n]["box"]) for n in P16.ALL4}
_make_cells = grid.make_cells
_clean_background = P13.clean_background


def shifted(box, sh):
    return {k: v + sh for k, v in box.items()}


def use_variant(tag):
    """Point every grid-dependent function at the variant's cell size and box."""
    size, sh = VARIANTS[tag]
    for n in P16.ALL4:
        MB.BASINS[n]["box"] = shifted(BOX0[n], sh)

    def make_cells(box, size_deg=0.2):
        return _make_cells(box, size)

    grid.make_cells = MB.make_cells = make_cells

    def basin_catalog(name):
        # The catalog keeps the original box plus 0.5 deg padding, which covers the
        # 0.1 deg shift; this keeps the stored nearest-neighbour values valid.
        box = MB.BASINS[name]["box"]
        MB.BASINS[name]["box"] = BOX0[name]
        try:
            return _basin_catalog(name)
        finally:
            MB.BASINS[name]["box"] = box

    MB.basin_catalog = basin_catalog

    def clean_background(g, name, cat, cutoff):
        # phase 13 version with the cell size as a parameter
        cells = grid.make_cells(MB.BASINS[name]["box"])
        c = cat[cat.mag >= P13.MC].sort_values("time").reset_index(drop=True)
        le = P13.nn_values(name, cat)
        before = c.time.values < np.datetime64(cutoff)
        used = int(np.isfinite(le[before]).sum()) >= 50
        bg = ~(le < catalog.nn_threshold(le[before])) if used else np.ones(len(c), bool)
        tday = (c.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        nlat, nlon = cells.lat0.nunique(), cells.lon0.nunique()
        ii = np.floor((c.lat.values - cells.lat0.min()) / size).astype(int)
        jj = np.floor((c.lon.values - cells.lon0.min()) / size).astype(int)
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

    P13.clean_background = clean_background

    def load_basin(name, lag=LAG):
        assert lag == LAG
        if tag == "base":
            return _orig_load_basin(name, lag)
        g = pd.read_parquet(GRIDS / f"grid_{tag}_{name}.parquet").sort_values(["start", "cell"])
        g = g[g.start < "2026-01-01"].reset_index(drop=True)
        cat = MB.basin_catalog(name)
        cache = GRIDS / f"etas_{tag}_{name}.npz"
        if cache.exists():
            z = np.load(cache)
            B, T, n = z["B"], z["T"], z["n"]
        else:
            B, T, n = etas.components(g, cat, grid.make_cells(MB.BASINS[name]["box"]), size_deg=size)
            np.savez(cache, B=B, T=T, n=n)
        g = g.copy()
        g["n_mc"], g["basin"] = n, name
        return g, B, T, cat

    MB.load_basin = load_basin


_orig_load_basin = MB.load_basin
_basin_catalog = MB.basin_catalog


def build(tag):
    """Same inputs as build_lagged_grid.py, on the variant grid."""
    size, sh = VARIANTS[tag]
    for n in P16.ALL4:
        box = shifted(BOX0[n], sh)
        pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
                   lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
        if n == "oklahoma":
            cat = catalog.load()
            inj = injection.in_box(injection.swd_only(injection.build()), pad)
            wells = pd.read_parquet(config.PROCESSED / "wells_grid_swd.parquet")
            faults = True
        else:
            cat = regions.comcat(n)
            inj = injection.in_box(getattr(regions, f"{n}_injection")(), pad)
            wells = basement.well_table(inj)
            faults = False
        g, _ = grid.build(cat, inj, wells, box, BLG.STARTS[n], "2025-12-31", size_deg=size, use_faults=faults,
                          inj_lag_months=LAG)
        g.to_parquet(GRIDS / f"grid_{tag}_{n}.parquet")
        print(tag, n, g.shape, flush=True)


def control():
    """1) the build path reproduces the stored 0.2 deg lag-12 grids;
    2) ETAS components computed directly on a lag-12 grid match the re-indexed cache;
    3) the patched loop reproduces phase 16 forecasts for one year per basin."""
    out = {}
    build("base")
    for n in P16.ALL4:
        a = pd.read_parquet(GRIDS / f"grid_base_{n}.parquet").sort_values(["start", "cell"]).reset_index(drop=True)
        b = pd.read_parquet(config.PROCESSED / f"grid_{n}_lag{LAG}.parquet").sort_values(["start", "cell"]).reset_index(drop=True)
        num = [c for c in b.columns if c in a.columns and pd.api.types.is_numeric_dtype(b[c])]
        out[f"grid_{n}"] = dict(rows=(len(a), len(b)),
                                max_abs=float(np.nanmax(np.abs(a[num].values.astype(float) - b[num].values.astype(float)))),
                                nan_mismatch=int((a[num].isna().values != b[num].isna().values).sum()))
        g, B0, T0, cat = _orig_load_basin(n, LAG)
        B, T, m = etas.components(g, cat, _make_cells(MB.BASINS[n]["box"], 0.2))
        out[f"etas_{n}"] = dict(B=float(np.abs(B - B0).max()), T=float(np.abs(T - T0).max()),
                                n=float(np.abs(m - g.n_mc.values).max()))
        print(n, out[f"grid_{n}"], out[f"etas_{n}"], flush=True)
    use_variant("base")
    preds = run("base", years=(2019, 2020), save=False)
    for n, p in preds.items():
        o = pd.read_parquet(config.RESULTS / "phase16_clean_dev" / f"pred_lag{LAG}_{n}.parquet")
        m = p.merge(o, on=["start", "cell"], suffixes=("", "_o"))
        cols = [c for c in p.columns if c.endswith(("_ex3", "_ex4")) and f"{c}_o" in m]
        out[f"loop_{n}"] = dict(rows=len(m), max_abs=max(float((m[c] - m[f"{c}_o"]).abs().max()) for c in cols))
        print(n, out[f"loop_{n}"], flush=True)
    (OUT / "control.json").write_text(json.dumps(out, indent=1))


def run(tag, years=(2014, 2025), save=True):
    base = {n: MB.load_basin(n, LAG) for n in P16.ALL4}
    res = {n: [] for n in P16.ALL4}
    for year in range(years[0], years[1] + 1):
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
            res[target].append(MB.run(target, data, P16.ALL4, (year, year), local_ml=True, lobo=True,
                                      ablation=False, verbose=False))
            print(tag, target, year, "done", flush=True)
    preds = {n: pd.concat(r, ignore_index=True) for n, r in res.items() if r}
    if save:
        for n, p in preds.items():
            p.to_parquet(OUT / f"pred_{tag}_{n}.parquet")
    return preds


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "control":
        control()
    else:
        use_variant(sys.argv[2])
        run(sys.argv[2])
