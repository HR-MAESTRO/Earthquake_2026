"""Protocol v3: out-of-time test on Jan-Oct 2026 (Delaware and Midland).

    python ../scripts/phase12_forward2026.py build <basin>   # inputs only, prints no 2026 outcomes
    python ../scripts/phase12_forward2026.py rehearse        # code check on 2025 only
    python ../scripts/phase12_forward2026.py test            # the single 2026 run

Differences from the earlier pipeline, all to remove look-ahead:
  * injection records after TRAIN/LAG limit (Sep 2025) are dropped before wells
    and features are built, so medians and well attributes cannot see later data
  * the Zaliapin-Ben-Zion threshold is fitted only on events before CUTOFF
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import basement, catalog, config, etas, grid, injection, multibasin as MB, regions

OUT = config.RESULTS / "phase12_forward2026"
OUT.mkdir(parents=True, exist_ok=True)
ALL5 = ["oklahoma", "kansas", "delaware", "midland", "fortworth"]
TARGETS = ["delaware", "midland"]
CUTOFF = pd.Timestamp("2025-04-01")     # 18 half-months before 2026-01-01
LAST_INJ = pd.Timestamp("2025-09-01")   # latest month usable by an Oct 2026 forecast at a 12-month lag
LAG = 12
END = "2026-10-31"
STARTS = {"oklahoma": config.ORIG_START, "kansas": "2015-01-01", "delaware": "2017-01-01",
          "midland": "2017-01-01", "fortworth": "2010-01-01"}
MODELS = ["clim", "si", "etas", "lgbm_cls_lobo", "hybrid_count_lobo", "hybrid_count_pooled", "hybrid_count_local"]


def _decluster_before_cutoff(cat, b=1.0, df=1.6, threshold=None):
    c = cat.sort_values("time").reset_index(drop=True)
    le, _ = catalog.nn_distance(c, b, df)
    thr = catalog.nn_threshold(le[c.time.values < np.datetime64(CUTOFF)])
    return pd.Series(~(le < thr), index=cat.sort_values("time").index), thr


def build(basin):
    box = MB.BASINS[basin]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    if basin == "oklahoma":
        raw = injection.build()
        raw = injection.swd_only(raw[raw.date <= LAST_INJ])
        cat = catalog.load()
    else:
        raw = getattr(regions, f"{basin}_injection")()
        raw = raw[raw.date <= LAST_INJ]
        cat = regions.comcat(basin)
    inj = injection.in_box(raw, pad)
    wells = basement.well_table(inj)
    g, _ = grid.build(cat, inj, wells, box, STARTS[basin], END, use_faults=False, inj_lag_months=LAG)
    g = g.sort_values(["start", "cell"]).reset_index(drop=True)
    g.to_parquet(config.PROCESSED / f"grid_v3_{basin}.parquet")
    catalog_nn = catalog.decluster_nn
    catalog.decluster_nn = _decluster_before_cutoff
    try:
        B, T, n = etas.components(g, MB.basin_catalog(basin), grid.make_cells(box, 0.2))
    finally:
        catalog.decluster_nn = catalog_nn
    np.savez(config.PROCESSED / f"mb3_{basin}.npz", B=B, T=T, n=n)
    print(basin, "built", g.shape, flush=True)


def load(name, until):
    g = pd.read_parquet(config.PROCESSED / f"grid_v3_{name}.parquet")
    z = np.load(config.PROCESSED / f"mb3_{name}.npz")
    keep = (g.start < pd.Timestamp(until)).values
    g = g[keep].reset_index(drop=True).copy()
    g["n_mc"], g["basin"] = z["n"][keep], name
    cat = MB.basin_catalog(name)
    return g, z["B"][keep], z["T"][:, :, keep], cat[cat.time < pd.Timestamp(until)].reset_index(drop=True)


def decide(pred, col="ex3"):
    base = MB.metrics.pr_auc(pred[col].values, pred[f"etas_{col}"].values)
    d, lo95, hi95 = MB.paired_diff(pred, col, "hybrid_count_lobo", "etas", level=0.95)
    _, lo90, hi90 = MB.paired_diff(pred, col, "hybrid_count_lobo", "etas", level=0.90)
    margin = 0.2 * base
    verdict = ("superior" if lo95 > 0 else "equivalent" if lo90 > -margin and hi90 < margin
               else "inferior" if hi95 < 0 else "inconclusive")
    pos = int(pred[col].sum())
    return dict(target=col, prauc_etas=base, diff=d, ci95=[lo95, hi95], ci90=[lo90, hi90], margin=margin,
                verdict=verdict, positives=pos, underpowered=pos < 20)


def evaluate(year, until, out):
    data = {b: load(b, until) for b in ALL5}
    pred = pd.concat([MB.run(b, data, ALL5, (year, year), local_ml=True, lobo=True, ablation=False, verbose=False)
                      for b in TARGETS], ignore_index=True)
    pred = pred.sort_values(["start", "basin", "cell"]).reset_index(drop=True)
    pred.to_parquet(out / "predictions.parquet")
    res = dict(primary=decide(pred, "ex3"), m4=decide(pred, "ex4") if pred.ex4.sum() else None)
    res["by_basin"] = {b: decide(pred[pred.basin == b].reset_index(drop=True), "ex3") for b in TARGETS
                       if pred.loc[pred.basin == b, "ex3"].sum()}
    res["block_sensitivity"] = {f"block{k}": MB.paired_diff(pred, "ex3", "hybrid_count_lobo", "etas", block=k)
                                for k in (3, 12)}
    res["poisson_ll"] = {k: MB.poisson_ll(pred, k) for k in ("lam_etas", "lam_hybrid_lobo", "lam_hybrid_pooled")}
    res["count_ratio"] = {k: float(pred[k].sum() / max(pred.n_mc.sum(), 1))
                          for k in ("lam_etas", "lam_hybrid_lobo", "lam_hybrid_pooled")}
    pd.concat([MB.score(pred, c, models=MODELS) for c in ("ex3", "ex4") if pred[c].sum()]).to_csv(out / "scores.csv", index=False)
    cal = []
    for c in ("ex3", "ex4"):
        for m in ("etas", "si", "hybrid_count_lobo"):
            if pred[c].sum():
                _, s = MB.calibration(pred[c], pred[f"{m}_{c}"])
                cal.append(s | dict(target=c, model=m))
    pd.DataFrame(cal).to_csv(out / "calibration.csv", index=False)
    (out / "decision.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main(mode, arg=None):
    if mode == "build":
        build(arg)
    elif mode == "rehearse":
        out = OUT / "rehearsal_2025"
        out.mkdir(exist_ok=True)
        print(json.dumps(evaluate(2025, "2026-01-01", out)["primary"], default=float))
    elif mode == "test":
        stamp = OUT / "RUN.txt"
        if stamp.exists():
            raise SystemExit(f"2026 test already run at {stamp.read_text().strip()}")
        stamp.write_text(pd.Timestamp.now().isoformat())
        out = OUT / "test_2026"
        out.mkdir(exist_ok=True)
        print(json.dumps(evaluate(2026, "2026-11-01", out), indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
