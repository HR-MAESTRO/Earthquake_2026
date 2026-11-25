"""Third-round review: small computations on saved forecasts.

    python ../scripts/phase25_review3.py

1. 2026 test (review 4.1): bootstrap of the RELATIVE difference
   (PR-AUC hybrid - PR-AUC ETAS) / PR-AUC ETAS, with the same paired half-month
   block resampling (blocks 1, 3, 6), so comparator uncertainty is included.
2. Reliability bins (review 4.2): static table of the Fig. 3 bins with
   cell-periods, positives, mean forecast and observed frequency.
3. Appendix A (review 4.2): M>=4 Brier skill on matched rows for basins with
   at least 9 M>=4 positives.
Outputs: results/phase25_review3/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from induced import config, metrics, multibasin as MB

R = config.RESULTS
OUT = R / "phase25_review3"
OUT.mkdir(parents=True, exist_ok=True)


def relative_2026():
    p = pd.read_parquet(R / "phase12_forward2026" / "test_2026" / "predictions.parquet")
    p = p.sort_values(["start", "cell"]).reset_index(drop=True)
    y, a, b = p.ex3.values, p.hybrid_count_lobo_ex3.values, p.etas_ex3.values
    rel = (metrics.pr_auc(y, a) - metrics.pr_auc(y, b)) / metrics.pr_auc(y, b)
    rows = []
    for blk in (1, 3, 6):
        vals = []
        for i in MB.resample_indices(p, 1000, 0, blk):
            pb = metrics.pr_auc(y[i], b[i])
            vals.append((metrics.pr_auc(y[i], a[i]) - pb) / pb if pb > 0 else np.nan)
        lo90, hi90 = np.nanpercentile(vals, [5, 95])
        lo95, hi95 = np.nanpercentile(vals, [2.5, 97.5])
        rows.append(dict(block=blk, rel_diff=rel, lo90=lo90, hi90=hi90, lo95=lo95, hi95=hi95,
                         within_20pct=bool(lo90 > -0.2 and hi90 < 0.2)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "relative_2026.csv", index=False)
    return t


def bins():
    r = pd.DataFrame(json.load(open(R / "figures" / "fig3_reliability.json")))
    names = {"etas": "ETAS", "hybrid_count_lobo": "Count hybrid (LOBO)", "lgbm_cls_pooled": "Classifier (pooled)"}
    r["model"] = r["model"].map(names)
    t = r.rename(columns=dict(n="cell_periods", pos="positives", p="mean_forecast", obs="observed"))
    t = t[["basin", "model", "cell_periods", "positives", "mean_forecast", "observed"]]
    t.to_csv(OUT / "reliability_bins.csv", index=False)
    return t


def bss4():
    src = {b: R / "phase16_clean_dev" / f"pred_lag12_{b}.parquet" for b in ("oklahoma", "delaware", "midland")}
    models = ["clim", "si", "etas", "lgbm_cls_lobo", "hybrid_count_local", "hybrid_count_pooled", "hybrid_count_lobo"]
    rows = []
    for b, f in src.items():
        p = pd.read_parquet(f)
        cols = [f"{m}_{t}" for m in models for t in ("ex3", "ex4") if f"{m}_{t}" in p]
        d = p.dropna(subset=cols)
        bc = metrics.brier_binary(d.ex4.values, d.const_ex4.values)
        for m in models:
            rows.append(dict(basin=b, model=m, m4=int(d.ex4.sum()),
                             bss4=1 - metrics.brier_binary(d.ex4.values, d[f"{m}_ex4"].values) / bc))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "bss4_matched.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    a, b, c = relative_2026(), bins(), bss4()
    txt = ["Relative difference 2026", a.round(3).to_string(), "", "M>=4 Brier skill (matched rows)",
           c.round(3).to_string(), "", f"reliability bins: {len(b)} rows written"]
    (OUT / "summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
