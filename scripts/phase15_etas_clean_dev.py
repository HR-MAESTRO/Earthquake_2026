"""Gridded ETAS on Oklahoma and Delaware with the look-ahead-free background.

The declustering threshold is fitted only on events before each fold's training
cutoff (as in phase13). The external-ETAS comparisons are then repeated against
this corrected gridded ETAS. A control refit with the original background must
reproduce the stored forecasts.

    python ../scripts/phase15_etas_clean_dev.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import phase13_leak_corrected as P13
import phase14_score as P14
from induced import config, etas, multibasin as MB

OUT = config.RESULTS / "phase15_etas_clean"
OUT.mkdir(parents=True, exist_ok=True)


def run(basin):
    g, B, T, cat = MB.load_basin(basin, 0)
    uniq = np.sort(g.start.unique())
    y0, y1 = MB.BASINS[basin]["test"]
    out = []
    for yr in range(y0, y1 + 1):
        te = np.where(pd.DatetimeIndex(g.start).year == yr)[0]
        p0 = np.searchsorted(uniq, g.start.values[te].min())
        cutoff = uniq[max(0, p0 - MB.GAP)]
        tr = np.where(g.start.values < cutoff)[0]
        ctrl = etas.rate(B, T, etas.fit(B, T, g.n_mc.values, tr))
        Bc, used = P13.clean_background(g, basin, cat, pd.Timestamp(cutoff))
        lam = etas.rate(Bc, T, etas.fit(Bc, T, g.n_mc.values, tr))
        out.append(pd.DataFrame(dict(start=g.start.values[te], cell=g.cell.values[te],
                                     lam_ctrl=ctrl[te], lam_clean=lam[te], declustered=used)))
        print(basin, yr, "declustered", used, flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    rows, diffs = [], []
    for basin in ("oklahoma", "delaware"):
        c = run(basin)
        c.to_parquet(OUT / f"lam_{basin}.parquet")
        d = P14.load(basin).merge(c, on=["start", "cell"])
        print(basin, "control max abs diff", float((d.lam_ctrl - d.lam_etas).abs().max()), flush=True)
        for mag, t in ((3, "ex3"), (4, "ex4")):
            d[f"clean_{t}"] = etas.exceed_prob(d.lam_clean.values, d.b.values, mag)
            s = MB.score(d, t, models=["etas", "clean", "full", "pkg", "fb1", "fb25"])
            s["basin"] = basin
            s["pred_over_obs"] = [d[f"{m}_{t}"].sum() / d[t].sum() for m in s.model]
            rows.append(s)
            for b in ("etas", "full", "pkg", "fb1", "fb25"):
                x, lo, hi = MB.paired_diff(d, t, "clean", b)
                diffs.append(dict(basin=basin, target=t, a="clean", b=b, diff=x, lo95=lo, hi95=hi))
        print(basin, "clean count ratio", d.lam_clean.sum() / d.n_mc.sum(), "LL", MB.poisson_ll(d, "lam_clean"), flush=True)
    s, p = pd.concat(rows), pd.DataFrame(diffs)
    s.to_csv(OUT / "scores.csv", index=False)
    p.to_csv(OUT / "paired.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(s[["basin", "target", "model", "prauc", "prauc_lo", "prauc_hi", "bss", "pred_over_obs"]].round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
