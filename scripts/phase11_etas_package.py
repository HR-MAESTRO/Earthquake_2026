"""External ETAS validation with the ETH Zurich `etas` package (Mizrahi et al.
2021, MIT licence, https://github.com/lmizrahi/etas; core module only).

For each test year: invert standard ETAS (uniform background, tapered
Omori-Utsu, magnitude-scaled power-law space kernel) on the 365 days before the
walk-forward cutoff (auxiliary window: the 365 days before that), inside the
basin box, Mc = 2.5. Then convert the fitted parameters to expected M>=2.5
counts per grid cell and half-month with the package's own kernel:
    background  mu * cell area * period length
    triggered   etas.inversion.expected_aftershocks(...) (time + total productivity)
                x fraction of the spatial kernel inside the cell (QMC, 256 draws)

    python ../scripts/phase11_etas_package.py <basin>
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from etas.inversion import ETASParameterCalculation, expected_aftershocks

from induced import config, multibasin as MB
from induced.etas_full import U_RAD, U_TH, _xy
from induced.grid import make_cells

logging.disable(logging.INFO)
OUT = config.RESULTS / "phase11_etas_package"
OUT.mkdir(parents=True, exist_ok=True)
MC = 2.5


def frac_in_rect(xi, yi, D, rho, xlo, xhi, ylo, yhi):
    """Fraction of the kernel (r^2 + D)^(-1-rho) inside a rectangle (QMC)."""
    r = np.sqrt(D[:, None] * ((1 - U_RAD[None, :]) ** (-1 / rho) - 1))
    px = xi[:, None] + r * np.cos(U_TH)[None, :]
    py = yi[:, None] + r * np.sin(U_TH)[None, :]
    return ((px >= xlo) & (px < xhi) & (py >= ylo) & (py < yhi)).mean(1)


def fit_year(cat, box, cutoff):
    c = cat[(cat.mag >= MC - 0.05)].copy()
    c = c[(c.time >= cutoff - pd.Timedelta(days=730)) & (c.time < cutoff)]
    c = c.rename(columns={"lat": "latitude", "lon": "longitude", "mag": "magnitude"})
    c["magnitude"] = (c["magnitude"] / 0.1).round() * 0.1
    c = c[["latitude", "longitude", "time", "magnitude"]].reset_index(drop=True)
    c.index.name = "id"
    shape = [[box["lat_min"], box["lon_min"]], [box["lat_max"], box["lon_min"]],
             [box["lat_max"], box["lon_max"]], [box["lat_min"], box["lon_max"]]]
    cfg = dict(catalog=c, data_path=str(OUT) + "/",
               auxiliary_start=str(cutoff - pd.Timedelta(days=730)),
               timewindow_start=str(cutoff - pd.Timedelta(days=365)), timewindow_end=str(cutoff),
               mc=MC, delta_m=0.1, coppersmith_multiplier=100, shape_coords=shape,
               theta_0={"log10_mu": -5.0, "log10_k0": -2.0, "a": 1.5, "log10_c": -2.5, "omega": -0.16,
                        "log10_tau": 4.0, "log10_d": -0.5, "gamma": 1.0, "rho": 0.6},
               # The unconstrained inversion diverged on Delaware 2018 (a > 14 after 200 EM
               # iterations). We use the package's own constrained example
               # (config/invert_etas_with_constraints.json): alpha = beta, tau and omega fixed.
               fixed_parameters={"alpha": "beta", "log10_tau": 4, "omega": -0.16})
    calc = ETASParameterCalculation(cfg)
    calc.prepare()
    theta = calc.invert()
    return theta, calc.n_hat, calc.i


def main(basin):
    g, B, T, cat = MB.load_basin(basin, 0)
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
    for yr in range(y0, y1 + 1):
        te = np.where(pd.DatetimeIndex(g.start).year == yr)[0]
        p0 = np.searchsorted(uniq, g.start.values[te].min())
        cutoff = pd.Timestamp(uniq[max(0, p0 - MB.GAP)])
        theta, n_hat, iters = fit_year(cat, box, cutoff)
        th = {k: v for k, v in dict(theta).items() if v is not None}
        prms.append(dict(basin=basin, year=yr, n_hat=n_hat, iterations=iters, **th))
        print(basin, yr, {k: round(float(v), 3) for k, v in th.items()}, "iters", iters, flush=True)
        mu = 10 ** th["log10_mu"]
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
            lam = mu * (cx2 - cx) * (cy2 - cy) * (ed - sd)
            for j in range(len(cells)):
                near = (ex[m] > cx[j] - 80) & (ex[m] < cx2[j] + 80) & (ey[m] > cy[j] - 80) & (ey[m] < cy2[j] + 80)
                if near.any():
                    fr = frac_in_rect(ex[m][near], ey[m][near], D[near], th["rho"], cx[j], cx2[j], cy[j], cy2[j])
                    lam[j] += (n_off[near] * fr).sum()
            sub = gy[gy.start == s]
            rows.append(pd.DataFrame(dict(start=s, cell=sub.cell.values, lam_pkg=lam[sub.cell.values])))
    pd.concat(rows, ignore_index=True).to_parquet(OUT / f"lam_pkg_{basin}.parquet")
    pd.DataFrame(prms).to_csv(OUT / f"params_{basin}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
