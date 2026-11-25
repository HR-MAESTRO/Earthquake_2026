"""Protocol v2: Fort Worth Basin held-out test. Run ONCE.

    python ../scripts/phase10_fortworth.py
Requires grid_fortworth_lag{0,3,12}.parquet (scripts/build_lagged_grid.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import catalog, config, multibasin as MB

OUT = config.RESULTS / "phase10_fortworth"
OUT.mkdir(parents=True, exist_ok=True)
ALL5 = ["oklahoma", "kansas", "delaware", "midland", "fortworth"]
MODELS = ["clim", "si", "etas", "hybrid_count_lobo", "hybrid_count_pooled", "lgbm_cls_lobo", "hybrid_count_local"]


def decide(pred, col="ex3"):
    s = MB.score(pred, col, models=["etas", "si"])
    best = s.sort_values("prauc", ascending=False).iloc[0]
    B, pb = best.model, best.prauc
    d, lo95, hi95 = MB.paired_diff(pred, col, "hybrid_count_lobo", B, level=0.95)
    _, lo90, hi90 = MB.paired_diff(pred, col, "hybrid_count_lobo", B, level=0.90)
    margin = 0.2 * pb
    if lo95 > 0:
        verdict = "superior"
    elif lo90 > -margin and hi90 < margin:
        verdict = "equivalent"
    elif hi95 < 0:
        verdict = "inferior"
    else:
        verdict = "inconclusive"
    pos = int(pred[col].sum())
    return dict(target=col, best_baseline=B, prauc_baseline=pb, diff=d, ci95=[lo95, hi95], ci90=[lo90, hi90],
                margin=margin, verdict=verdict, positives=pos, underpowered=pos < 20)


def main():
    stamp = OUT / "RUN.txt"
    if stamp.exists():
        print("WARNING: Fort Worth test already run at", stamp.read_text().strip())
    stamp.write_text(pd.Timestamp.now().isoformat())
    results = {}
    # completeness by year in the Fort Worth box
    c = catalog.in_box(MB.basin_catalog("fortworth"), MB.BASINS["fortworth"]["box"])
    comp = pd.DataFrame([dict(year=y, n=len(g), mc=catalog.mc_maxc(g.mag) if len(g) >= 50 else np.nan,
                              n_ge_2_5=int((g.mag >= 2.5).sum())) for y, g in c.groupby(c.time.dt.year)])
    comp.to_csv(OUT / "completeness.csv", index=False)
    print(comp.to_string())
    for lag in (12, 0, 3):                       # 12 = primary
        data = {b: MB.load_basin(b, lag) for b in ALL5}
        pred = MB.run("fortworth", data, ALL5, MB.BASINS["fortworth"]["test"], local_ml=True, lobo=True,
                      ablation=False)
        pred.to_parquet(OUT / f"pred_lag{lag}.parquet")
        sc = pd.concat([MB.score(pred, col, models=MODELS) for col in ("ex3", "ex4")])
        sc.to_csv(OUT / f"scores_lag{lag}.csv", index=False)
        dec = decide(pred, "ex3")
        results[f"lag{lag}"] = dec
        print(f"\n=== lag {lag} ===\n", sc.round(4).to_string(), "\n", dec, flush=True)
        if lag == 12:
            # secondary: years with Mc <= 2.5 only
            ok_years = comp[(comp.mc <= 2.5)].year.tolist()
            sub = pred[pd.DatetimeIndex(pred.start).year.isin(ok_years)].reset_index(drop=True)
            if sub.ex3.sum() > 0:
                results["lag12_complete_years"] = decide(sub, "ex3") | {"years": ok_years}
            # calibration + counts + block sensitivity
            cal = []
            for col in ("ex3", "ex4"):
                for m in ("etas", "si", "hybrid_count_lobo", "hybrid_count_pooled"):
                    k = f"{m}_{col}"
                    if k in pred and pred[col].sum() > 0:
                        _, s = MB.calibration(pred.loc[pred[k].notna(), col], pred.loc[pred[k].notna(), k])
                        cal.append(s | dict(target=col, model=m))
            pd.DataFrame(cal).to_csv(OUT / "calibration_lag12.csv", index=False)
            results["poisson_ll"] = {k: MB.poisson_ll(pred, k) for k in ("lam_etas", "lam_hybrid_lobo", "lam_hybrid_pooled")
                                     if k in pred}
            blocks = {}
            for blk in (3, 12, 24):
                blocks[f"block{blk}"] = MB.paired_diff(pred, "ex3", "hybrid_count_lobo", dec["best_baseline"], block=blk)
            blocks["by_year"] = MB.paired_diff(pred, "ex3", "hybrid_count_lobo", dec["best_baseline"], by_year=True)
            results["block_sensitivity"] = blocks
    (OUT / "decision.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
