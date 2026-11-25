"""Second-round review checks that reuse saved predictions (12-month lag, look-ahead-free).

    python ../scripts/phase22_review2.py

1. Historical recalibration (review 5.1). For each basin, model and test year Y, a
   logistic recalibration logit(P') = c0 + c1*logit(P) is fitted on that model's
   forecasts and outcomes from earlier test years only (all periods ended before Y
   starts), then applied to year Y. Years without at least 10 earlier positives are
   skipped. Reported: predicted/observed M>=3 ratio and Brier skill (against the
   basin's past base rate), raw vs recalibrated, on the same rows.
2. Delaware completeness (review 4.2). Mc is 2.7-2.8 in 2018-2020, above the M2.5
   threshold. Key Delaware comparisons are rescored on 2021-2025 only. Training
   still used the incomplete years.
Outputs: results/phase22_review2/{recalibration,delaware_2021}.csv and summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from induced import config, metrics, multibasin as MB

OUT = config.RESULTS / "phase22_review2"
OUT.mkdir(parents=True, exist_ok=True)
BASINS = ["oklahoma", "kansas", "delaware", "midland"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibration():
    rows = []
    for b in BASINS:
        p = pd.read_parquet(config.RESULTS / "phase16_clean_dev" / f"pred_lag12_{b}.parquet")
        p["year"] = pd.DatetimeIndex(p.start).year
        for m in ("etas", "hybrid_count_lobo", "si"):
            c = f"{m}_ex3"
            q = p[p[c].notna()]
            raw, cal, ys, const = [], [], [], []
            for y in sorted(q.year.unique()):
                tr, te = q[q.year < y], q[q.year == y]
                if tr.ex3.sum() < 10:
                    continue
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(tr[c].values)[:, None], tr.ex3.values)
                raw.append(te[c].values)
                cal.append(lr.predict_proba(logit(te[c].values)[:, None])[:, 1])
                ys.append(te.ex3.values)
                const.append(te.const_ex3.values)
            if not ys:
                continue
            y_, r_, c_, k_ = map(np.concatenate, (ys, raw, cal, const))
            bc = metrics.brier_binary(y_, k_)
            rows.append(dict(basin=b, model=m, n=len(y_), positives=int(y_.sum()),
                             ratio_raw=r_.sum() / y_.sum(), ratio_recal=c_.sum() / y_.sum(),
                             bss_raw=1 - metrics.brier_binary(y_, r_) / bc, bss_recal=1 - metrics.brier_binary(y_, c_) / bc,
                             prauc_raw=metrics.pr_auc(y_, r_), prauc_recal=metrics.pr_auc(y_, c_)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "recalibration.csv", index=False)
    return t


def delaware():
    p = pd.read_parquet(config.RESULTS / "phase16_clean_dev" / "pred_lag12_delaware.parquet")
    rows = []
    for label, d in (("2019-2025 (as reported)", p), ("2021-2025 (complete catalog)",
                                                      p[pd.DatetimeIndex(p.start).year >= 2021])):
        d = d.sort_values(["start", "cell"]).reset_index(drop=True)
        s = MB.score(d, "ex3", n_boot=500, models=["etas", "si", "hybrid_count_lobo", "hybrid_count_pooled",
                                                    "hybrid_count_local", "lgbm_cls_lobo", "clim"])
        for _, r in s.iterrows():
            rows.append(dict(window=label, what=r.model, value=r.prauc, lo=r.prauc_lo, hi=r.prauc_hi,
                             positives=r.positives))
        for a, c in (("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"), ("hybrid_count_pooled", "hybrid_count_local"),
                     ("lgbm_cls_lobo", "etas"), ("etas", "si")):
            x, lo, hi = MB.paired_diff(d, "ex3", a, c)
            rows.append(dict(window=label, what=f"{a} - {c}", value=x, lo=lo, hi=hi, positives=int(d.ex3.sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "delaware_2021.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    a, b = recalibration(), delaware()
    txt = "Historical recalibration (M>=3)\n" + a.round(3).to_string() + "\n\nDelaware completeness\n" + b.round(3).to_string()
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
