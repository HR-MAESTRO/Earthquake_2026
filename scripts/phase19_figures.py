"""Figures for the manuscript (v0.8).

    python ../scripts/phase19_figures.py

Writes to results/figures/:
  fig1_map.png             study area: basin boxes, 0.2 deg cells, disposal wells, M>=3 epicentres
  fig3_reliability.json    M>=3 reliability rows, 12-month lag (look-ahead-free; Fort Worth as run)
  fig4_robustness.json     LOBO hybrid - ETAS, M>=3, across injection lags and grids (6-period blocks)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from induced import config, multibasin as MB

R = config.RESULTS
OUT = R / "figures"
OUT.mkdir(exist_ok=True)
NAMES = {"oklahoma": "Oklahoma", "kansas": "Kansas", "delaware": "Delaware Basin", "midland": "Midland Basin",
         "fortworth": "Fort Worth Basin"}
INJ = {"oklahoma": "injection_monthly.parquet", "kansas": "kansas_injection_monthly.parquet",
       "delaware": "delaware_injection_monthly.parquet", "midland": "midland_injection_monthly.parquet",
       "fortworth": "fortworth_injection_monthly.parquet"}


def inside(df, box):
    return df[(df.lat >= box["lat_min"]) & (df.lat < box["lat_max"]) & (df.lon >= box["lon_min"])
              & (df.lon < box["lon_max"])]


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=300)
    stats = {}
    # US Census 2018 cartographic boundary file, 1:20M (data/raw/census_states)
    import geopandas as gpd
    st = gpd.read_file(config.RAW / "census_states" / "cb_2018_us_state_20m.shp").to_crs(4326)
    st = st[st.STUSPS.isin(["OK", "KS", "TX", "NM", "CO", "MO", "AR", "LA", "NE"])]
    st.boundary.plot(ax=ax, color="#8a8a8a", lw=0.6, zorder=0.5)
    for code, (x, y) in {"KS": (-100.3, 38.35), "OK": (-101.6, 36.75), "TX": (-100.0, 32.3), "NM": (-104.6, 34.3),
                         "CO": (-104.6, 38.35)}.items():
        ax.text(x, y, code, fontsize=10, color="#8a8a8a", ha="center", va="center", zorder=0.6)
    for n, b in MB.BASINS.items():
        box = b["box"]
        if n == "oklahoma":  # Class II disposal wells used for the Oklahoma grid
            w = inside(pd.read_parquet(config.PROCESSED / "wells_grid_swd.parquet")[["lat", "lon"]].dropna(), box)
        else:  # basin files are already disposal-only
            inj = pd.read_parquet(config.PROCESSED / INJ[n], columns=["api", "lat", "lon", "vol_bbl"])
            w = inside(inj[inj.vol_bbl > 0].groupby("api")[["lat", "lon"]].first().dropna(), box)
        cat = MB.basin_catalog(n)
        q = inside(cat[cat.mag >= 3.0], box)
        stats[n] = dict(wells=len(w), m3=len(q), first=str(q.time.min())[:4], last=str(q.time.max())[:4])
        ax.scatter(w.lon, w.lat, s=0.7, c="#8a8a8a", lw=0, rasterized=True, zorder=1)
        ax.scatter(q.lon, q.lat, s=3.2, c="#1f5fbf", lw=0, alpha=0.7, rasterized=True, zorder=3)
        for x in np.arange(box["lon_min"], box["lon_max"] + 1e-9, 0.2):
            ax.plot([x, x], [box["lat_min"], box["lat_max"]], c="#d0d0d0", lw=0.25, zorder=0)
        for y in np.arange(box["lat_min"], box["lat_max"] + 1e-9, 0.2):
            ax.plot([box["lon_min"], box["lon_max"]], [y, y], c="#d0d0d0", lw=0.25, zorder=0)
        held = not b["dev"]
        ax.add_patch(plt.Rectangle((box["lon_min"], box["lat_min"]), box["lon_max"] - box["lon_min"],
                                   box["lat_max"] - box["lat_min"], fill=False, ec="#222222", lw=0.9,
                                   ls="--" if held else "-", zorder=4))
        y0, y1 = b["test"]
        label = f"{NAMES[n]}{' (held out)' if held else ''}\ntest {y0}–{y1}"
        if n == "oklahoma":  # Kansas sits directly above Oklahoma: label to the right
            ax.text(box["lon_max"] + 0.08, box["lat_max"] - 0.05, label, fontsize=8.5, va="top", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.6))
        else:
            ax.text(box["lon_min"] + 0.03, box["lat_max"] + 0.06, label, fontsize=8.5, va="bottom", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(-105.0, -94.4)
    ax.set_ylim(30.7, 38.6)
    ax.set_aspect(1 / np.cos(np.radians(34.5)))
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc="#1f5fbf", mec="none", label="M≥3 epicentre"),
         plt.Line2D([], [], ls="", marker="o", ms=2, mfc="#9a9a9a", mec="none", label="Disposal well"),
         plt.Line2D([], [], c="#222222", lw=0.9, label="Development basin"),
         plt.Line2D([], [], c="#222222", lw=0.9, ls="--", label="Held-out basin")]
    ax.legend(handles=h, fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_map.png", dpi=300)
    (OUT / "fig1_stats.json").write_text(json.dumps(stats, indent=1))
    print("fig1", stats, flush=True)


def fig3():
    rows = []
    src = {b: R / "phase16_clean_dev" / f"pred_lag12_{b}.parquet" for b in ("oklahoma", "kansas", "delaware", "midland")}
    src["fortworth"] = R / "phase10_fortworth" / "pred_lag12.parquet"
    for b, f in src.items():
        p = pd.read_parquet(f)
        for m in ("etas", "hybrid_count_lobo", "lgbm_cls_pooled"):
            c = f"{m}_ex3"
            ok = p[c].notna()
            r, _ = MB.calibration(p.loc[ok, "ex3"], p.loc[ok, c])
            for _, x in r.iterrows():
                rows.append(dict(basin=NAMES[b], bslug=b, model=m, n=int(x["n"]), pos=int(round(x["obs_rate"] * x["n"])),
                                 p=round(float(x["mean_p"]), 5), obs=round(float(x["obs_rate"]), 5)))
    (OUT / "fig3_reliability.json").write_text(json.dumps(rows))
    print("fig3", len(rows), flush=True)


def fig4():
    rows = []
    d = pd.read_csv(R / "phase16_clean_dev" / "tables" / "paired_diffs_blocks.csv")
    d = d[(d.target == "ex3") & (d.a == "hybrid_count_lobo") & (d.b == "etas") & (d["resample"] == "block6")]
    for _, x in d.iterrows():
        rows.append(dict(basin=NAMES[x.basin], bslug=x.basin, setting=f"{int(x.lag)}-month lag, 0.2°",
                         sslug=f"lag{int(x.lag)}", diff=round(x["diff"], 4), lo=round(x.lo, 4), hi=round(x.hi, 4)))
    g = pd.read_csv(R / "phase18_grid" / "paired.csv")
    g = g[(g.a == "hybrid_count_lobo") & (g.b == "etas") & (g["resample"] == "block6") & (g.variant != "main")]
    lab = {"offset": "12-month, shifted 0.1°", "size01": "12-month, 0.1° cells", "size04": "12-month, 0.4° cells"}
    for _, x in g.iterrows():
        rows.append(dict(basin=NAMES[x.basin], bslug=x.basin, setting=lab[x.variant], sslug=x.variant,
                         diff=round(x["diff"], 4), lo=round(x.lo, 4), hi=round(x.hi, 4)))
    order = ["lag0", "lag3", "lag12", "offset", "size01", "size04"]
    rows.sort(key=lambda r: (["oklahoma", "kansas", "delaware", "midland"].index(r["bslug"]), order.index(r["sslug"])))
    (OUT / "fig4_robustness.json").write_text(json.dumps(rows))
    print("fig4", len(rows), flush=True)


if __name__ == "__main__":
    fig3()
    fig4()
    fig1()
