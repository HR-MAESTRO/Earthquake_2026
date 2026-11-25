"""Revision analyses from phase9 predictions (all exploratory except where the
v1 Midland test is quoted).

    python ../scripts/phase9_analysis.py
Writes CSV tables to $IEQ_WORK/results/phase9/tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import catalog, config, etas, metrics, multibasin as MB

R = config.RESULTS / "phase9"
OUT = R / "tables"
OUT.mkdir(parents=True, exist_ok=True)
BASINS = ["oklahoma", "kansas", "delaware", "midland"]
LAGS = [0, 3, 12]
KEY = ["clim", "si", "etas", "lgbm_cls_pooled", "lgbm_cls_lobo", "hybrid_count_pooled",
       "hybrid_count_lobo", "hybrid_count_local", "count_plain_gr", "hybrid_count_emp"]


def load(lag, b):
    return pd.read_parquet(R / f"pred_lag{lag}_{b}.parquet")


def main():
    # 1. scores by lag x basin x model
    sc = []
    for lag in LAGS:
        for b in BASINS:
            p = load(lag, b)
            for col in ("ex3", "ex4"):
                s = MB.score(p, col, n_boot=500)
                s.insert(0, "basin", b)
                s.insert(0, "lag", lag)
                sc.append(s)
    sc = pd.concat(sc)
    sc.to_csv(OUT / "scores_lag_basin_model.csv", index=False)
    print(sc[(sc.target == "ex3")].pivot_table(index=["basin", "model"], columns="lag", values="prauc").round(3))

    # 2. paired differences, block-length sensitivity (lag 12 = operational)
    diffs = []
    pairs = [("hybrid_count_pooled", "etas"), ("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"),
             ("hybrid_count_pooled", "hybrid_count_local"), ("hybrid_count_lobo", "hybrid_count_local")]
    for lag in (0, 12):
        for b in BASINS:
            p = load(lag, b)
            for col in ("ex3", "ex4"):
                for a, c in pairs:
                    if f"{a}_{col}" not in p or f"{c}_{col}" not in p:
                        continue
                    for blk in (3, 6, 12, 24):
                        d, lo, hi = MB.paired_diff(p, col, a, c, n_boot=500, block=blk)
                        diffs.append(dict(lag=lag, basin=b, target=col, a=a, b=c, resample=f"block{blk}",
                                          diff=d, lo=lo, hi=hi))
                    d, lo, hi = MB.paired_diff(p, col, a, c, n_boot=500, by_year=True)
                    diffs.append(dict(lag=lag, basin=b, target=col, a=a, b=c, resample="by_year",
                                      diff=d, lo=lo, hi=hi))
    pd.DataFrame(diffs).to_csv(OUT / "paired_diffs_blocks.csv", index=False)

    # 3. equivalence (post hoc for Midland; margins reported as a sensitivity)
    eq = []
    for lag in (0, 12):
        p = load(lag, "midland")
        for a, c in (("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si")):
            for mg in (0.01, 0.02, 0.05):
                e = MB.equivalence(p, "ex3", a, c, mg, n_boot=1000)
                e.update(lag=lag, a=a, b=c)
                eq.append(e)
    pd.DataFrame(eq).to_csv(OUT / "equivalence_midland_posthoc.csv", index=False)

    # 4. M>=4 ablation + b sensitivity (lag 0)
    abl = []
    for b in BASINS:
        p = load(0, b)
        if p.ex4.sum() == 0:
            continue
        for m in ("lgbm_cls_pooled", "count_plain_gr", "hybrid_count_pooled", "hybrid_count_emp", "etas", "si"):
            c = f"{m}_ex4"
            if c in p:
                abl.append(dict(basin=b, variant=m, prauc=metrics.pr_auc(p.ex4, p[c]),
                                bss=1 - metrics.brier_binary(p.ex4, p[c]) / metrics.brier_binary(p.ex4, p.const_ex4)))
        for bv in (0.8, 1.0, 1.2):
            q = etas.exceed_prob(p.lam_hybrid_pooled.values, bv, 4)
            abl.append(dict(basin=b, variant=f"hybrid_pooled_b{bv}", prauc=metrics.pr_auc(p.ex4, q),
                            bss=1 - metrics.brier_binary(p.ex4, q) / metrics.brier_binary(p.ex4, p.const_ex4)))
    pd.DataFrame(abl).to_csv(OUT / "m4_ablation.csv", index=False)

    # 5. quiet cells (no M>=3 within 25 km for 12 / 24 months)
    qc = []
    for lag in (0, 12):
        for b in BASINS:
            p = load(lag, b)
            for days in (365, 730):
                s = p[p.days_since_m3 > days]
                if s.ex3.sum() == 0:
                    qc.append(dict(lag=lag, basin=b, quiet_days=days, n=len(s), positives=0))
                    continue
                for m in KEY:
                    c = f"{m}_ex3"
                    if c in s:
                        ss_ = s[s[c].notna()]
                        if ss_.ex3.sum() == 0:
                            continue
                        qc.append(dict(lag=lag, basin=b, quiet_days=days, n=len(ss_), positives=int(ss_.ex3.sum()),
                                       model=m, prauc=metrics.pr_auc(ss_.ex3, ss_[c]), base=ss_.ex3.mean()))
    pd.DataFrame(qc).to_csv(OUT / "quiet_cells.csv", index=False)

    # 6. calibration (lag 12) + count calibration
    cal, rel, cnt = [], [], []
    for b in BASINS:
        p = load(12, b)
        for col in ("ex3", "ex4"):
            for m in ("etas", "si", "hybrid_count_pooled", "hybrid_count_lobo", "lgbm_cls_pooled"):
                c = f"{m}_{col}"
                if c not in p:
                    continue
                ok = p[c].notna()
                r, s = MB.calibration(p.loc[ok, col], p.loc[ok, c])
                s.update(basin=b, target=col, model=m)
                cal.append(s)
                r.insert(0, "model", m)
                r.insert(0, "target", col)
                r.insert(0, "basin", b)
                rel.append(r)
        for lc in ("lam_etas", "lam_hybrid_pooled", "lam_hybrid_lobo"):
            if lc in p:
                q = p[p[lc].notna()]
                cnt.append(dict(basin=b, model=lc, n=len(q), expected=q[lc].sum(), observed=q.n_mc.sum(),
                                ratio=q[lc].sum() / max(q.n_mc.sum(), 1), loglik=MB.poisson_ll(q, lc)))
    pd.DataFrame(cal).to_csv(OUT / "calibration_lag12.csv", index=False)
    pd.concat(rel).astype({"bin": str}).to_csv(OUT / "reliability_lag12.csv", index=False)
    pd.DataFrame(cnt).to_csv(OUT / "count_calibration_lag12.csv", index=False)

    # 7. sample sizes
    ss = []
    for b in BASINS:
        g = MB._grid(b, 0)
        p = load(0, b)
        ss.append(dict(basin=b, data_from=str(g.start.min().date()), cells=g.cell.nunique(),
                       cell_periods=len(g), ex3_all=int(g.ex3.sum()), ex4_all=int(g.ex4.sum()),
                       test_years=f"{MB.BASINS[b]['test'][0]}-{MB.BASINS[b]['test'][1]}",
                       test_cell_periods=len(p), ex3_test=int(p.ex3.sum()), ex4_test=int(p.ex4.sum()),
                       ex3_rate_test=p.ex3.mean(), ex4_rate_test=p.ex4.mean()))
    pd.DataFrame(ss).to_csv(OUT / "sample_sizes.csv", index=False)
    print(pd.DataFrame(ss).to_string())

    # 8. completeness by basin and year (inside the basin box)
    comp = []
    for b in BASINS:
        c = catalog.in_box(MB.basin_catalog(b), MB.BASINS[b]["box"])
        for yr, gy in c.groupby(c.time.dt.year):
            if yr < 2009 or yr > 2025:
                continue
            comp.append(dict(basin=b, year=yr, n=len(gy), mc_maxc=catalog.mc_maxc(gy.mag) if len(gy) >= 50 else np.nan,
                             n_ge_2_5=int((gy.mag >= 2.5).sum())))
    pd.DataFrame(comp).to_csv(OUT / "completeness_by_basin_year.csv", index=False)


if __name__ == "__main__":
    main()
