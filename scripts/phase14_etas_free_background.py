"""External ETAS with a spatially varying background (flETAS mode of the ETH
Zurich `etas` package, Mizrahi et al.).

Same windows, constraints (alpha = beta, tau = 1e4 d, omega = -0.16) and
triggered-count calculation as phase11. Only the background changes: each
fitted event i contributes P_background_i, spread by a Gaussian kernel of
variance bw_sq (km^2) and divided by the fit-window length, as in the package's
E-step. The kernel is integrated exactly over each forecast cell. The background
map is fixed within a test year because it is fitted before the cutoff.

Primary setting: bw_sq = 1 km^2 (package default). Sensitivity: 25 km^2.

    python ../scripts/phase14_etas_free_background.py <basin> <bw_sq>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from etas.inversion import ETASParameterCalculation, expected_aftershocks
from scipy.special import erf

import phase11_etas_package as P11
from induced import config, multibasin as MB
from induced.etas_full import _xy
from induced.grid import make_cells

OUT = config.RESULTS / "phase14_etas_freebg"
OUT.mkdir(parents=True, exist_ok=True)
MC = 2.5
WINDOW = 365.0


def fit_year(cat, box, cutoff, bw_sq):
    c = cat[(cat.mag >= MC - 0.05)].copy()
    c = c[(c.time >= cutoff - pd.Timedelta(days=730)) & (c.time < cutoff)]
    c = c.rename(columns={"lat": "latitude", "lon": "longitude", "mag": "magnitude"})
    c["magnitude"] = (c["magnitude"] / 0.1).round() * 0.1
    c = c[["latitude", "longitude", "time", "magnitude"]].reset_index(drop=True)
    c.index.name = "id"
    shape = [[box["lat_min"], box["lon_min"]], [box["lat_max"], box["lon_min"]],
             [box["lat_max"], box["lon_max"]], [box["lat_min"], box["lon_max"]]]
    cfg = dict(catalog=c.copy(), data_path=str(OUT) + "/",
               auxiliary_start=str(cutoff - pd.Timedelta(days=730)),
               timewindow_start=str(cutoff - pd.Timedelta(days=365)), timewindow_end=str(cutoff),
               mc=MC, delta_m=0.1, coppersmith_multiplier=100, shape_coords=shape,
               free_background=True, bw_sq=bw_sq,
               theta_0={"log10_mu": -5.0, "log10_k0": -2.0, "a": 1.5, "log10_c": -2.5, "omega": -0.16,
                        "log10_tau": 4.0, "log10_d": -0.5, "gamma": 1.0, "rho": 0.6},
               fixed_parameters={"alpha": "beta", "log10_tau": 4, "omega": -0.16})
    calc = ETASParameterCalculation(cfg)
    calc.prepare()
    theta = calc.invert()
    te = calc.target_events
    pts = c.loc[te.index]
    return theta, calc.n_hat, calc.i, pts.latitude.values, pts.longitude.values, te["P_background"].fillna(0).values


def background_rate(lat, lon, pb, cells, lat0, lon0, bw_sq):
    """Background events per day in each cell."""
    x, y = _xy(lat, lon, lat0, lon0)
    cx, cy = _xy(cells.lat0.values, cells.lon0.values, lat0, lon0)
    cx2, cy2 = _xy(cells.lat0.values + 0.2, cells.lon0.values + 0.2, lat0, lon0)
    z = np.sqrt(2 * bw_sq)
    fx = 0.5 * (erf((cx2[:, None] - x[None, :]) / z) - erf((cx[:, None] - x[None, :]) / z))
    fy = 0.5 * (erf((cy2[:, None] - y[None, :]) / z) - erf((cy[:, None] - y[None, :]) / z))
    return (fx * fy * pb[None, :]).sum(1) / WINDOW


def main(basin, bw_sq, only_year=None):
    g, _, _, cat = MB.load_basin(basin, 0)
    box = MB.BASINS[basin]["box"]
    cells = make_cells(box, 0.2)
    lat0, lon0 = box["lat_min"], box["lon_min"]
    cx, cy = _xy(cells.lat0.values, cells.lon0.values, lat0, lon0)
    cx2, cy2 = _xy(cells.lat0.values + 0.2, cells.lon0.values + 0.2, lat0, lon0)
    ex, ey = _xy(cat.lat.values, cat.lon.values, lat0, lon0)
    tday = (cat.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    mags = cat.mag.values
    uniq = np.sort(g.start.unique())
    rows, prms = [], []
    y0, y1 = MB.BASINS[basin]["test"]
    for yr in ([only_year] if only_year else range(y0, y1 + 1)):
        te = np.where(pd.DatetimeIndex(g.start).year == yr)[0]
        p0 = np.searchsorted(uniq, g.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        theta, n_hat, iters, lat, lon, pb = fit_year(cat, box, cutoff, bw_sq)
        th = {k: v for k, v in dict(theta).items() if v is not None}
        bg_day = background_rate(lat, lon, pb, cells, lat0, lon0, bw_sq)
        prms.append(dict(basin=basin, year=yr, bw_sq=bw_sq, n_hat=n_hat, iterations=iters,
                         bg_in_cells=float(bg_day.sum() * WINDOW), **th))
        print(basin, yr, "bw", bw_sq, "iters", iters, "n_hat", round(float(n_hat), 1),
              "bg in cells", round(float(bg_day.sum() * WINDOW), 1),
              {k: round(float(v), 3) for k, v in th.items()}, flush=True)
        if only_year:
            return
        theta8 = [th["log10_k0"], th["a"], th["log10_c"], th["omega"], th["log10_tau"], th["log10_d"],
                  th["gamma"], th["rho"]]
        gy = g.iloc[te]
        for s, e in gy[["start", "end"]].drop_duplicates().itertuples(index=False):
            sd = (np.datetime64(s) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            ed = (np.datetime64(e) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            m = (tday < sd) & (tday >= sd - 365) & (mags >= MC)
            dm = mags[m] - MC
            n_off = expected_aftershocks((mags[m], sd - tday[m], ed - tday[m]), (theta8, MC))
            D = 10 ** th["log10_d"] * np.exp(th["gamma"] * dm)
            lam = bg_day * (ed - sd)
            for j in range(len(cells)):
                near = (ex[m] > cx[j] - 80) & (ex[m] < cx2[j] + 80) & (ey[m] > cy[j] - 80) & (ey[m] < cy2[j] + 80)
                if near.any():
                    fr = P11.frac_in_rect(ex[m][near], ey[m][near], D[near], th["rho"], cx[j], cx2[j], cy[j], cy2[j])
                    lam[j] += (n_off[near] * fr).sum()
            sub = gy[gy.start == s]
            rows.append(pd.DataFrame(dict(start=s, cell=sub.cell.values, lam_freebg=lam[sub.cell.values])))
    tag = f"{basin}_bw{bw_sq:g}"
    pd.concat(rows, ignore_index=True).to_parquet(OUT / f"lam_{tag}.parquet")
    pd.DataFrame(prms).to_csv(OUT / f"params_{tag}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
