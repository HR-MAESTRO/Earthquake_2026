"""Score the phase 23 reruns.  python ../scripts/phase23_score.py
Writes results/phase23_reruns/summary.txt and scores_*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import config, multibasin as MB

R = config.RESULTS
OUT = R / "phase23_reruns"
BASINS = ["oklahoma", "kansas", "delaware", "midland"]


def fmt(x, lo, hi):
    return f"{x:+.3f} [{lo:.3f}, {hi:.3f}]" + ("*" if lo > 0 or hi < 0 else "")


def offset():
    lines, rows = [], []
    for lag in (0, 12):
        f = OUT / f"offset_lag{lag}.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f)
        for b in BASINS:
            p = pd.read_parquet(R / "phase16_clean_dev" / f"pred_lag{lag}_{b}.parquet")
            d = p.merge(o[o.basin == b].drop(columns=["basin", "ex3", "ex4", "n_mc"]), on=["start", "cell"])
            d = d.sort_values(["start", "cell"]).reset_index(drop=True)
            ctrl = float((d.hybrid_ctrl_ex4 - d.hybrid_count_pooled_ex4).abs().max())
            for t in ("ex3", "ex4"):
                s = MB.score(d, t, n_boot=500, models=["lgbm_cls_pooled", "count_plain_gr", "count_feat",
                                                       "hybrid_count_pooled", "etas"]).set_index("model")
                r = dict(lag=lag, basin=b, target=t, positives=int(d[t].sum()), control_max_diff=ctrl,
                         **{f"prauc_{m}": s.prauc[m] for m in s.index})
                for a, c in (("hybrid_count_pooled", "count_feat"), ("count_feat", "count_plain_gr"),
                             ("count_feat", "etas")):
                    r[f"{a} - {c}"] = fmt(*MB.paired_diff(d, t, a, c))
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scores_offset.csv", index=False)
    lines += ["Offset-only control (pooled; count_feat = ETAS/SI as features, no offset)", t.round(3).to_string(), ""]
    return lines


def mc3():
    lines, rows = [], []
    for b in ("delaware", "midland"):
        f = OUT / f"mc3_{b}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f).sort_values(["start", "cell"]).reset_index(drop=True)
        s = MB.score(d, "ex3", n_boot=500, models=["etas", "si", "hybrid_count_lobo", "hybrid_count_pooled",
                                                   "hybrid_count_local", "lgbm_cls_lobo", "clim"]).set_index("model")
        r = dict(basin=b, positives=int(d.ex3.sum()), **{f"prauc_{m}": s.prauc[m] for m in s.index})
        for a, c in (("hybrid_count_lobo", "etas"), ("hybrid_count_pooled", "hybrid_count_local"),
                     ("lgbm_cls_lobo", "etas"), ("etas", "si")):
            r[f"{a} - {c}"] = fmt(*MB.paired_diff(d, "ex3", a, c))
        e = MB.equivalence(d, "ex3", "hybrid_count_lobo", "etas", 0.02)
        r["lobo-etas 90%CI"] = f"[{e['lo90']:.3f}, {e['hi90']:.3f}]"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scores_mc3.csv", index=False)
    return ["Mc = 3.0 pipeline, 12-month lag (M>=3 PR-AUC)", t.round(3).T.to_string(), ""]


def permian():
    rows = []
    for source, target in (("delaware", "midland"), ("midland", "delaware")):
        f = OUT / f"single_source_{source}_to_{target}.parquet"
        if not f.exists():
            continue
        x = pd.read_parquet(f)
        p = pd.read_parquet(R / "phase16_clean_dev" / f"pred_lag12_{target}.parquet")
        c = f"hybrid_count_from_{source}_ex3"
        d = p.merge(x[["start", "cell", c]], on=["start", "cell"]).sort_values(["start", "cell"]).reset_index(drop=True)
        m = f"hybrid_count_from_{source}"
        s = MB.score(d, "ex3", n_boot=500, models=[m, "hybrid_count_lobo", "etas"]).set_index("model")
        e = MB.equivalence(d, "ex3", m, "etas", 0.02)
        rows.append(dict(source=source, target=target, prauc_single=s.prauc[m], prauc_lobo3=s.prauc["hybrid_count_lobo"],
                         prauc_etas=s.prauc["etas"], single_minus_etas=fmt(*MB.paired_diff(d, "ex3", m, "etas")),
                         single_minus_lobo3=fmt(*MB.paired_diff(d, "ex3", m, "hybrid_count_lobo")),
                         single_etas_90=f"[{e['lo90']:.3f}, {e['hi90']:.3f}]"))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scores_permian.csv", index=False)
    return ["Single-source Permian transfer, 12-month lag (M>=3)", t.round(3).T.to_string(), ""]


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 50)
    parts = sys.argv[1:] or ["offset", "mc3", "permian"]
    txt = []
    for p in parts:
        txt += globals()[p]()
    (OUT / f"summary_{'_'.join(parts)}.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
