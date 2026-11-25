"""Manuscript tables from the look-ahead-free development runs (phase 16).

1. Runs phase9_analysis.main unchanged, pointed at results/phase16_clean_dev.
2. Table 2: look-ahead-free gridded ETAS (phase 15) against the full ETAS.
3. Figure rows: M>=3 PR-AUC with 95% intervals at the 12-month lag.
4. Midland without basement features, corrected inputs, 12-month lag.

    python ../scripts/phase17_tables.py [all|nobasement]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import phase9_analysis as A
from induced import config, etas, metrics, multibasin as MB

R = config.RESULTS
SRC = R / "phase16_clean_dev"
OUT = SRC / "tables"
OUT.mkdir(parents=True, exist_ok=True)
FIG = {"clim": "Climatology", "si": "Seismogenic index", "etas": "ETAS", "lgbm_cls_lobo": "Classifier (LOBO)",
       "hybrid_count_local": "Count hybrid (local)", "hybrid_count_pooled": "Count hybrid (pooled)",
       "hybrid_count_lobo": "Count hybrid (LOBO)"}


def analysis():
    A.R, A.OUT = SRC, OUT
    A.main()


def table2():
    rows = []
    for b in ("oklahoma", "delaware"):
        d = pd.read_parquet(SRC / f"pred_lag0_{b}.parquet").merge(
            pd.read_parquet(R / "phase9" / f"etas_full_{b}.parquet"), on=["start", "cell"])
        d = d.sort_values(["start", "cell"]).reset_index(drop=True)
        for m, t in ((3, "ex3"), (4, "ex4")):
            d[f"full_{t}"] = etas.exceed_prob(d.lam_full.values, d.b.values, m)
            s = MB.score(d, t, models=["etas", "full"]).set_index("model")
            x, lo, hi = MB.paired_diff(d, t, "etas", "full")
            rows.append(dict(basin=b, target=t, gridded=s.prauc["etas"], full=s.prauc["full"], diff=x, lo=lo, hi=hi,
                             bss_gridded=s.bss["etas"], bss_full=s.bss["full"],
                             count_gridded=d.lam_etas.sum() / d.n_mc.sum(), count_full=d.lam_full.sum() / d.n_mc.sum(),
                             spearman=spearmanr(d.lam_etas, d.lam_full)[0]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table2_etas_validation.csv", index=False)
    print(t.round(4).to_string(index=False))


def figure_rows():
    out = []
    for b in ("oklahoma", "kansas", "delaware", "midland"):
        d = pd.read_parquet(SRC / f"pred_lag12_{b}.parquet").sort_values(["start", "cell"]).reset_index(drop=True)
        s = MB.score(d, "ex3", n_boot=500, models=list(FIG)).set_index("model")
        for m, lab in FIG.items():
            out.append(dict(basin=b, model=lab, mslug=m, v=s.prauc[m], lo=s.prauc_lo[m], hi=s.prauc_hi[m]))
        if b == "oklahoma":
            sub = d[d.hybrid_count_lobo_ex3.notna()].reset_index(drop=True)
            s2 = MB.score(sub, "ex3", n_boot=500, models=["etas"]).iloc[0]
            out.append(dict(basin=b, model="ETAS, 2017–2022", mslug="etas_sub", v=s2.prauc, lo=s2.prauc_lo, hi=s2.prauc_hi))
    f = pd.DataFrame(out)
    f.to_csv(OUT / "figure_rows_lag12.csv", index=False)
    print(f.round(3).to_string(index=False))


def nobasement():
    import phase16_clean_dev as P16
    drop = ("basement_depth_m", "d_basement_w")
    MB.FEATS = [f for f in MB.FEATS if f not in drop]
    MB.PLAIN_FEATS = [f for f in MB.PLAIN_FEATS if f not in drop]
    lag = 12
    base = {n: MB.load_basin(n, lag) for n in P16.ALL4}
    gt = base["midland"][0]
    uniq = np.sort(gt.start.unique())
    preds = []
    for year in range(2020, 2026):
        te = gt.start.dt.year.values == year
        p0 = np.searchsorted(uniq, gt.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        MB._ETAS_MEMO.clear()
        data = P16.fold_data(base, year, lag, cutoff, True, {})
        preds.append(MB.run("midland", data, P16.ALL4, (year, year), local_ml=False, lobo=True, ablation=False, verbose=False))
        print("nobasement", year, flush=True)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(SRC / "pred_lag12_midland_nobasement.parquet")
    s = MB.score(p, "ex3", n_boot=500, models=["etas", "si", "hybrid_count_lobo", "hybrid_count_pooled"])
    s.to_csv(OUT / "midland_nobasement_lag12.csv", index=False)
    print(s.round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["nobasement"]:
        nobasement()
    else:
        table2()
        figure_rows()
        analysis()
