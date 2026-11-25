"""Follow-up analyses (after the holdout was used; walk-forward 2014-2022 only).

A. Declustering (plan step 10): features rebuilt from
   - full catalog (baseline, v0.1)
   - Gardner-Knopoff (1974) mainshocks
   - Zaliapin & Ben-Zion (2013) nearest-neighbour background events
   Labels always come from the FULL catalog (hazard includes aftershocks).
B. Pore-pressure diffusion forecast (plan step 17), diffusivity D chosen on
   2010-2013 only, added as baseline 'pp' and offset hybrid 'hybrid_pp'.

The 2023-2025 holdout is NOT used here.
    python ../scripts/phase7_decluster_pressure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import catalog, config, grid, gridmodels as GM, injection, metrics, physics

OUT = config.RESULTS / "phase7"
OUT.mkdir(parents=True, exist_ok=True)
GRID_BOX = dict(lat_min=35.0, lat_max=37.0, lon_min=-99.0, lon_max=-96.0)
PAD = dict(lat_min=34.5, lat_max=37.5, lon_min=-99.5, lon_max=-95.5)
LABELS = ["mmax", "y", "ex3", "ex4"]


def main():
    full = pd.read_parquet(config.PROCESSED / "grid_dataset_v0.1.parquet")
    cat = catalog.in_box(catalog.load(), PAD)
    cat = cat[(cat.time >= "2005-01-01") & (cat.mag >= 2.5)].sort_values("time").reset_index(drop=True)
    print("events M>=2.5 for declustering:", len(cat))

    # ---------------- A. declustering ----------------
    gk = catalog.decluster_gk(cat)
    nn, thr = catalog.decluster_nn(cat)
    frac = pd.DataFrame(dict(year=cat.time.dt.year, gk=gk.values, nn=nn.sort_index().values))
    fr = frac.groupby("year")[["gk", "nn"]].mean()
    fr.to_csv(OUT / "background_fraction_by_year.csv")
    print(f"NN threshold log10(eta0) = {thr:.2f}")
    print(fr.loc[2010:2025].round(2).to_string())
    pd.Series(dict(nn_log10_eta0=thr, n_events=len(cat), gk_background=float(gk.mean()),
                   nn_background=float(nn.mean()))).to_csv(OUT / "declustering_summary.csv")

    inj = injection.in_box(injection.swd_only(injection.build()), GRID_BOX | {
        "lat_min": GRID_BOX["lat_min"] - .3, "lat_max": GRID_BOX["lat_max"] + .3,
        "lon_min": GRID_BOX["lon_min"] - .3, "lon_max": GRID_BOX["lon_max"] + .3})
    wells = pd.read_parquet(config.PROCESSED / "wells_grid_swd.parquet")
    variants = {"full": full}
    for name, mask in (("gk", gk.values), ("nn", nn.sort_index().values)):
        path = config.PROCESSED / f"grid_dataset_v0.1_{name}.parquet"
        if path.exists():
            variants[name] = pd.read_parquet(path)
            continue
        g, _ = grid.build(cat[mask], inj, wells, GRID_BOX, config.ORIG_START, "2025-12-31")
        g = g.drop(columns=LABELS).merge(full[["cell", "start"] + LABELS], on=["cell", "start"])
        g.to_parquet(path)
        variants[name] = g

    # ---------------- B. pore pressure ----------------
    cells = grid.make_cells(GRID_BOX, 0.2)
    sel = []
    early = full[(full.start >= "2011-01-01") & (full.start < "2014-01-01")]
    for D in (0.05, 0.2, 0.5, 2.0):
        months, P = physics.pressure_matrix(inj, cells, D=D)
        e = physics.add_pressure_features(early, months, P)
        s = metrics.pr_auc(e.ex3, e.pp_p3)
        sel.append(dict(D=D, prauc_ex3_2011_2013=s))
        print(f"D={D} m2/s  PR-AUC(ex3, 2011-2013) = {s:.3f}")
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "diffusivity_selection.csv", index=False)
    D = float(sel.loc[sel.prauc_ex3_2011_2013.idxmax(), "D"])
    months, P = physics.pressure_matrix(inj, cells, D=D)
    print("chosen D =", D)

    # ---------------- walk-forward 2014-2022 ----------------
    rows = []
    for name, g in variants.items():
        g = physics.add_pressure_features(g, months, P)
        models = ["clim", "gr", "si", "pp", "hybrid", "hybrid_pp", "lgbm"]
        for target in ("ex3", "ex4"):
            pred, _ = GM.walk_forward(g, target=target, models=models, verbose=False)
            pred.to_parquet(OUT / f"pred_{name}_{target}.parquet")
            sc = GM.score(pred, models, n_boot=500)
            sc.insert(0, "target", target)
            sc.insert(0, "catalog", name)
            rows.append(sc)
            print(f"\n[{name}] {target}")
            print(sc[["model", "prauc", "prauc_lo", "prauc_hi", "brier", "bss_vs_const"]].round(4).to_string())
    pd.concat(rows).to_csv(OUT / "scores_decluster_pressure.csv", index=False)


if __name__ == "__main__":
    main()
