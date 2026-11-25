"""Where does gridded-ETAS skill come from: triggering, the rolling background, or both?
(Second-round review, item 3.1.)

For every development basin and walk-forward test year (look-ahead-free background,
same cutoff and rows as phase 16):
  etas_roll     a*B_roll + K*T   (the paper's ETAS; control: must equal phase 16)
  bg_roll       a*B_roll         background only, updated at every issue time
  trig          K*T              triggering only (no background)
  etas_frozen   a*B_frz + K*T    background frozen at the training cutoff for the whole
                                 test year (the information the full and package ETAS use)
  bg_frozen     a*B_frz          frozen background only
B_roll: Zaliapin-Ben-Zion background events in the cell over the 365 days before each
period (threshold fitted before the cutoff), +0.1, scaled to period length.
B_frz: the same quantity evaluated at the cutoff, as a daily rate, times period length.
(a, K, alpha, p) are fitted by Poisson ML on training rows for each model separately.

    python ../scripts/phase22_etas_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import phase13_leak_corrected as P13
from induced import config, etas, multibasin as MB

OUT = config.RESULTS / "phase22_etas_attribution"
OUT.mkdir(parents=True, exist_ok=True)
BASINS = ["oklahoma", "kansas", "delaware", "midland"]
MODELS = ["etas_roll", "bg_roll", "trig", "etas_frozen", "bg_frozen"]


def fit_bg(B, n):
    return n.sum() / max(B.sum(), 1e-12)          # Poisson MLE of a in lambda = a*B


def fit_trig(T, n):
    best = None
    for ai in range(T.shape[0]):
        for pi in range(T.shape[1]):
            t = T[ai, pi] + 1e-9
            K = n.sum() / t.sum()
            ll = (n * np.log(K * t)).sum() - K * t.sum()
            if best is None or ll > best[0]:
                best = (ll, ai, pi, K)
    return dict(ai=best[1], pi=best[2], K=best[3])


def run():
    rows = []
    for name in BASINS:
        g, B0, T, cat = MB.load_basin(name, 12)
        days = ((g.end - g.start).dt.days).values.astype(float)
        uniq = np.sort(g.start.unique())
        y0, y1 = MB.BASINS[name]["test"]
        for yr in range(y0, y1 + 1):
            te = np.where(g.start.dt.year.values == yr)[0]
            if not len(te):
                continue
            p0 = np.searchsorted(uniq, g.start.values[te].min())
            cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
            tr = np.where(g.start.values < np.datetime64(cutoff))[0]
            if not len(tr):
                continue
            B, _ = P13.clean_background(g, name, cat, cutoff)
            # frozen background: per-cell daily rate at the cutoff period
            at_cut = g.start.values == np.datetime64(cutoff)
            rate_c = pd.Series(B[at_cut] / days[at_cut], index=g.cell.values[at_cut])
            Bf = g.cell.map(rate_c).values * days
            n = g.n_mc.values
            prm = etas.fit(B, T, n, tr)
            lam = {"etas_roll": etas.rate(B, T, prm)}
            lam["bg_roll"] = fit_bg(B[tr], n[tr]) * B
            pt = fit_trig(T[:, :, tr], n[tr])
            lam["trig"] = pt["K"] * T[pt["ai"], pt["pi"]] + 1e-9
            # frozen ETAS uses the same parameters, with the background frozen at the cutoff
            lam["etas_frozen"] = prm["a"] * Bf + prm["K"] * T[prm["ai"], prm["pi"]]
            lam["bg_frozen"] = fit_bg(B[tr], n[tr]) * Bf
            b = etas.b_value(cat, pd.Timestamp(f"{yr}-01-01"))
            r = pd.DataFrame(dict(basin=name, start=g.start.values[te], cell=g.cell.values[te],
                                  ex3=g.ex3.values[te], ex4=g.ex4.values[te], n_mc=n[te]))
            for k, v in lam.items():
                r[f"lam_{k}"] = v[te]
                r[f"{k}_ex3"] = etas.exceed_prob(v[te], b, 3)
                r[f"{k}_ex4"] = etas.exceed_prob(v[te], b, 4)
            rows.append(r)
            print(name, yr, "alpha", prm["alpha"], "p", prm["p"], flush=True)
    p = pd.concat(rows, ignore_index=True)
    p.to_parquet(OUT / "predictions.parquet")
    return p


def score(p):
    out, pairs, ctrl = [], [], {}
    for name in BASINS:
        d = p[p.basin == name].sort_values(["start", "cell"]).reset_index(drop=True)
        o = pd.read_parquet(config.RESULTS / "phase16_clean_dev" / f"pred_lag12_{name}.parquet")
        d = d.merge(o[["start", "cell", "const_ex3", "const_ex4"]], on=["start", "cell"], how="left")
        m = d.merge(o[["start", "cell", "etas_ex3", "etas_ex4"]], on=["start", "cell"])
        ctrl[name] = float(max((m.etas_roll_ex3 - m.etas_ex3).abs().max(), (m.etas_roll_ex4 - m.etas_ex4).abs().max()))
        for t in ("ex3", "ex4"):
            s = MB.score(d, t, n_boot=500, models=MODELS)
            s["basin"] = name
            out.append(s)
            for a in MODELS[1:]:
                x, lo, hi = MB.paired_diff(d, t, a, "etas_roll")
                pairs.append(dict(basin=name, target=t, model=a, diff=x, lo=lo, hi=hi))
            x, lo, hi = MB.paired_diff(d, t, "etas_frozen", "bg_frozen")
            pairs.append(dict(basin=name, target=t, model="etas_frozen - bg_frozen", diff=x, lo=lo, hi=hi))
        # Poisson log-likelihood per cell-period
        for k in MODELS:
            out.append(pd.DataFrame([dict(basin=name, model=k, target="loglik",
                                          prauc=MB.poisson_ll(d.assign(n_mc=d.n_mc), f"lam_{k}"))]))
    s = pd.concat(out, ignore_index=True)
    pr = pd.DataFrame(pairs)
    s.to_csv(OUT / "scores.csv", index=False)
    pr.to_csv(OUT / "paired_vs_rolling_etas.csv", index=False)
    pd.set_option("display.width", 200)
    txt = ["control: max |etas_roll - phase16 etas| per basin: " + str(ctrl), "",
           s.pivot_table(index=["basin", "target"], columns="model", values="prauc").round(3)[MODELS].to_string(), "",
           pr.assign(txt=pr["diff"].map("{:+.3f}".format) + " [" + pr.lo.map("{:.3f}".format) + ", "
                     + pr.hi.map("{:.3f}".format) + "]").pivot_table(index=["basin", "target"], columns="model",
                                                                     values="txt", aggfunc="first").to_string()]
    (OUT / "summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    f = OUT / "predictions.parquet"
    score(pd.read_parquet(f) if sys.argv[1:] == ["score"] and f.exists() else run())
