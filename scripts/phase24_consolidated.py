"""Consolidated results on matched evaluation rows (second-round review 5.3).

For each basin, the rows where every listed model has a forecast (12-month lag;
development basins from the look-ahead-free reruns, Fort Worth as run). Reports
M>=3 PR-AUC, M>=3 Brier skill (vs the basin's past base rate), M>=4 PR-AUC and the
mean Poisson log-likelihood of M>=2.5 counts for the count models.

    python ../scripts/phase24_consolidated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import config, metrics, multibasin as MB

R = config.RESULTS
OUT = R / "phase24_consolidated"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["clim", "si", "etas", "lgbm_cls_lobo", "hybrid_count_local", "hybrid_count_pooled", "hybrid_count_lobo"]
LAM = {"etas": "lam_etas", "hybrid_count_local": "lam_hybrid_local", "hybrid_count_pooled": "lam_hybrid_pooled",
       "hybrid_count_lobo": "lam_hybrid_lobo"}
SRC = {b: R / "phase16_clean_dev" / f"pred_lag12_{b}.parquet" for b in ("oklahoma", "kansas", "delaware", "midland")}
SRC["fortworth"] = R / "phase10_fortworth" / "pred_lag12.parquet"

rows = []
for b, f in SRC.items():
    p = pd.read_parquet(f)
    cols = [f"{m}_{t}" for m in MODELS for t in ("ex3", "ex4") if f"{m}_{t}" in p]
    d = p.dropna(subset=cols).reset_index(drop=True)
    yrs = pd.DatetimeIndex(d.start).year
    for m in MODELS:
        if f"{m}_ex3" not in d:
            continue
        bc = metrics.brier_binary(d.ex3.values, d.const_ex3.values)
        rows.append(dict(basin=b, years=f"{yrs.min()}–{yrs.max()}", rows=len(d), m3=int(d.ex3.sum()),
                         m4=int(d.ex4.sum()), model=m,
                         prauc3=metrics.pr_auc(d.ex3.values, d[f"{m}_ex3"].values),
                         bss3=1 - metrics.brier_binary(d.ex3.values, d[f"{m}_ex3"].values) / bc,
                         prauc4=metrics.pr_auc(d.ex4.values, d[f"{m}_ex4"].values) if d.ex4.sum() else float("nan"),
                         loglik=MB.poisson_ll(d, LAM[m]) if m in LAM and LAM[m] in d else float("nan")))
t = pd.DataFrame(rows)
t.to_csv(OUT / "consolidated.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(4).to_string())
