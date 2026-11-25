"""Score the grid offset / cell-size variants against the main (phase 16) grid.

    python ../scripts/phase18_score.py
Writes results/phase18_grid/{scores,paired}.csv and summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import config, multibasin as MB

R = config.RESULTS
OUT = R / "phase18_grid"
BASINS = ["oklahoma", "kansas", "delaware", "midland"]
MODELS = ["clim", "si", "etas", "lgbm_cls_pooled", "lgbm_cls_lobo", "hybrid_count_local", "hybrid_count_pooled",
          "hybrid_count_lobo"]
PAIRS = [("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"), ("hybrid_count_pooled", "hybrid_count_local"),
         ("lgbm_cls_lobo", "etas"), ("etas", "si")]
M4 = ["etas", "si", "lgbm_cls_pooled", "hybrid_count_pooled"]


def preds(tag, b):
    f = R / "phase16_clean_dev" / f"pred_lag12_{b}.parquet" if tag == "main" else OUT / f"pred_{tag}_{b}.parquet"
    return pd.read_parquet(f).sort_values(["start", "cell"]).reset_index(drop=True) if f.exists() else None


def main():
    s_rows, p_rows = [], []
    for tag in ("main", "offset", "size01", "size04"):
        for b in BASINS:
            d = preds(tag, b)
            if d is None:
                continue
            for t, models in (("ex3", MODELS), ("ex4", M4)):
                s = MB.score(d, t, n_boot=500, models=models)
                s["variant"], s["basin"], s["base_rate"] = tag, b, d[t].mean()
                s_rows.append(s)
            for a, c in PAIRS:
                for by_year in (False, True):
                    x, lo, hi = MB.paired_diff(d, "ex3", a, c, by_year=by_year)
                    p_rows.append(dict(variant=tag, basin=b, a=a, b=c, resample="by_year" if by_year else "block6",
                                       diff=x, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0)))
            print(tag, b, flush=True)
    s, p = pd.concat(s_rows, ignore_index=True), pd.DataFrame(p_rows)
    s.to_csv(OUT / "scores.csv", index=False)
    p.to_csv(OUT / "paired.csv", index=False)
    summary(s, p)


def summary(s, p):
    pd.set_option("display.width", 250, "display.max_rows", 500)
    lines = []
    for t in ("ex3", "ex4"):
        x = s[s.target == t].pivot_table(index=["basin", "model"], columns="variant", values="prauc")
        lines += [f"PR-AUC {t}", x.round(3).to_string(), ""]
    x = s[s.target == "ex3"].groupby(["basin", "variant"])[["positives", "n", "base_rate"]].first()
    lines += ["Positives and base rates (M>=3)", x.to_string(), ""]
    p["txt"] = (p["diff"].map("{:+.3f}".format) + " [" + p.lo.map("{:.3f}".format) + ", " + p.hi.map("{:.3f}".format)
                + "]" + p.sig.map({True: "*", False: ""}))
    for r in ("block6", "by_year"):
        x = p[p["resample"] == r].pivot_table(index=["basin", "a", "b"], columns="variant", values="txt", aggfunc="first")
        lines += [f"Paired M>=3 PR-AUC differences ({r})", x.to_string(), ""]
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:  # rebuild summary.txt from the saved CSVs
        summary(pd.read_csv(OUT / "scores.csv"), pd.read_csv(OUT / "paired.csv"))
    else:
        main()
