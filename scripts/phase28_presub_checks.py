"""Checks requested by the final pre-submission review (26 Nov 2026).

1. Brier skill after the row-level depth correction (development basins and 2026).
2. Midland post hoc equivalence (LOBO hybrid - ETAS, 90% CI, margin +-0.01) after the correction.
3. Implausible well depths: which records produce them and how many rows they affect.

    python ../scripts/phase28_presub_checks.py
Outputs: results/phase28_presub/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import phase13_leak_corrected as P13
from induced import basement, config, injection, multibasin as MB

OUT = config.RESULTS / "phase28_presub"
OUT.mkdir(parents=True, exist_ok=True)
REP = config.RESULTS / "phase16_clean_dev"
ROW = config.RESULTS / "phase27_rowlevel_depth"
DEV = ["oklahoma", "kansas", "delaware", "midland"]
MODELS = ["etas", "hybrid_count_lobo", "hybrid_count_pooled", "hybrid_count_local", "lgbm_cls_lobo", "si"]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# 1. Brier skill -------------------------------------------------------------------------
rows = []
for n in DEV:
    for tag, f in (("reported", REP / f"pred_lag12_{n}.parquet"), ("row-level", ROW / f"pred_lag12_{n}.parquet")):
        d = pd.read_parquet(f).sort_values(["start", "cell"]).reset_index(drop=True)
        for col in ("ex3", "ex4"):
            ms = [m for m in MODELS if f"{m}_{col}" in d]
            s = MB.score(d, col, n_boot=200, models=ms)
            for _, r in s.iterrows():
                rows.append(dict(basin=n, version=tag, target=col, model=r.model, positives=r.positives,
                                 prauc=r.prauc, bss=r.bss, bss_lo=r.bss_lo, bss_hi=r.bss_hi))
for tag, f in (("reported", config.RESULTS / "phase12_forward2026" / "test_2026" / "predictions.parquet"),
               ("row-level", ROW / "test_2026" / "predictions.parquet")):
    d = pd.read_parquet(f).sort_values(["start", "cell"]).reset_index(drop=True)
    ms = [m for m in MODELS if f"{m}_ex3" in d]
    s = MB.score(d, "ex3", n_boot=200, models=ms)
    for _, r in s.iterrows():
        rows.append(dict(basin="2026 (Delaware+Midland)", version=tag, target="ex3", model=r.model,
                         positives=r.positives, prauc=r.prauc, bss=r.bss, bss_lo=r.bss_lo, bss_hi=r.bss_hi))
b = pd.DataFrame(rows)
b.to_csv(OUT / "brier_skill.csv", index=False)
w = b.pivot_table(index=["basin", "target", "model"], columns="version", values="bss").round(3)
w["change"] = (w["row-level"] - w["reported"]).round(3)
pd.set_option("display.width", 200)
say("1. Brier skill (reported vs row-level depth)")
say(w.to_string())
say(f"largest |change| in Brier skill: {w['change'].abs().max():.3f}")

# 2. Midland equivalence -------------------------------------------------------------------
say("\n2. Midland LOBO hybrid - ETAS, 12-month lag, 90% CI (margin +-0.01)")
for tag, f in (("reported", REP / "pred_lag12_midland.parquet"), ("row-level", ROW / "pred_lag12_midland.parquet")):
    raw = pd.read_parquet(f)
    for order, d in (("stored order", raw), ("sorted", raw.sort_values(["start", "cell"]).reset_index(drop=True))):
        for nb in (500, 1000):
            x, lo, hi = MB.paired_diff(d, "ex3", "hybrid_count_lobo", "etas", n_boot=nb, level=0.90)
            say(f"  {tag:9s} {order:12s} n_boot={nb}: {x:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                f"equivalent within 0.01: {lo > -0.01 and hi < 0.01}")

# 3. Implausible well depths ---------------------------------------------------------------
say("\n3. Wells with implausible depths (bottom more than 7 km, or more than 3 km below basement)")
recs = []
for n in ["oklahoma", "kansas", "delaware", "midland", "fortworth"]:
    raw = P13.raw_injection(n)
    if n == "oklahoma":
        raw = injection.swd_only(raw)
    box = MB.BASINS[n]["box"]
    pad = dict(lat_min=box["lat_min"] - .3, lat_max=box["lat_max"] + .3,
               lon_min=box["lon_min"] - .3, lon_max=box["lon_max"] + .3)
    raw = injection.in_box(raw, pad)
    w = basement.well_table(raw)
    bad = w[(w.bottom_m > 7000) | (w.d_basement_m > 3000)]
    say(f"  {n}: {len(w)} wells, {len(bad)} flagged (latest record); max bottom {w.bottom_m.max():,.0f} m, "
        f"max depth below basement {w.d_basement_m.max():,.0f} m")
    # per-record depth fields over time for the flagged wells
    for api in bad.api.head(10):
        r = raw[raw.api == api].sort_values("date")
        for c in ("inj_bot_ft", "packer_ft", "total_depth_ft"):
            if c in r:
                vals = r.groupby("year")[c].last().dropna()
                if len(vals):
                    recs.append(dict(basin=n, api=api, field=c,
                                     by_year="; ".join(f"{int(y)}: {v:,.0f}" for y, v in vals.items())))
pd.DataFrame(recs).to_csv(OUT / "implausible_wells.csv", index=False)
say("  per-year depth fields of flagged wells: implausible_wells.csv")

(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
