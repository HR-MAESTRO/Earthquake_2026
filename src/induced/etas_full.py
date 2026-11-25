"""Reference continuous space-time ETAS (Ogata 1998 form), fitted by maximum
likelihood, for validating the simplified gridded ETAS in etas.py.

Conditional intensity (events per day per km^2), magnitudes m >= Mc:
    lambda(t, x) = mu * u(x) + sum_{t_i < t} k(m_i) g(t - t_i) f(x - x_i | m_i)
    k(m)   = K exp(a (m - Mc))                              productivity
    g(t)   = (p-1) c^(p-1) (t + c)^(-p)                     normalised Omori-Utsu
    f(r|m) = (q-1)/(pi s^2) (1 + r^2/s^2)^(-q),  s^2 = d^2 exp(gam (m - Mc))
    u(x)   = background density (per km^2): Gaussian-smoothed (5 km) density of
             events in the 365 d before the fit window (normalised over the region)
All 8 parameters (mu, K, a, c, p, d, gam, q) are free (log-parametrised) and
fitted with L-BFGS-B on the exact log-likelihood. The spatial integral of each
event's kernel over the region (and over each forecast cell) is computed with
quasi-Monte-Carlo draws from the kernel, so boundary losses are handled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MC = 2.5
NDRAW = 256


def _xy(lat, lon, lat0, lon0):
    return ((np.asarray(lon) - lon0) * 111.2 * np.cos(np.radians(lat0)),
            (np.asarray(lat) - lat0) * 111.2)


def _unit_draws(n=NDRAW, seed=0):
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.random(n)) / n
    th = 2 * np.pi * rng.permutation(np.arange(n) + rng.random(n)) / n
    return u, th


U_RAD, U_TH = _unit_draws()


def _frac_inside(xi, yi, s, q, xlo, xhi, ylo, yhi):
    """Fraction of each event's spatial kernel inside a rectangle (QMC)."""
    # inverse CDF of the power-law radial kernel: F(r) = 1 - (1 + r^2/s^2)^(1-q)
    r = s[:, None] * np.sqrt((1 - U_RAD[None, :]) ** (1 / (1 - q)) - 1)
    px = xi[:, None] + r * np.cos(U_TH)[None, :]
    py = yi[:, None] + r * np.sin(U_TH)[None, :]
    return ((px >= xlo) & (px < xhi) & (py >= ylo) & (py < yhi)).mean(1)


def _bg_density(x_ev, y_ev, xq, yq, area, bw=5.0):
    if len(x_ev) == 0:
        return np.full(len(xq), 1.0 / area)
    d2 = (xq[:, None] - x_ev[None, :]) ** 2 + (yq[:, None] - y_ev[None, :]) ** 2
    dens = np.exp(-d2 / (2 * bw ** 2)).sum(1) / (2 * np.pi * bw ** 2 * len(x_ev))
    return dens + 0.05 / area                                 # small floor everywhere


class FullETAS:
    def __init__(self, box):
        self.box = box
        self.lat0, self.lon0 = box["lat_min"], box["lon_min"]
        self.xhi, _ = _xy(box["lat_min"], box["lon_max"], self.lat0, self.lon0)
        _, self.yhi = _xy(box["lat_max"], box["lon_min"], self.lat0, self.lon0)
        self.xhi, self.yhi = float(self.xhi), float(self.yhi)
        self.area = self.xhi * self.yhi

    # ---------------- fitting ----------------
    def fit(self, cat, t_start, t_end, mc=MC):
        c = cat[(cat.mag >= mc)].sort_values("time")
        tday = (c.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        x, y = _xy(c.lat.values, c.lon.values, self.lat0, self.lon0)
        inbox = (x >= 0) & (x < self.xhi) & (y >= 0) & (y < self.yhi)
        ts = (np.datetime64(t_start) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        tE = (np.datetime64(t_end) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        pre = (tday >= ts - 365) & (tday < ts) & inbox
        win = (tday >= ts) & (tday < tE) & inbox
        hist = (tday >= ts - 365) & (tday < tE)                  # possible parents (incl. outside box)
        self.bg_x, self.bg_y = x[pre], y[pre]
        tw, xw, yw = tday[win], x[win], y[win]
        th, xh, yh, mh = tday[hist], x[hist], y[hist], c.mag.values[hist]
        uw = _bg_density(self.bg_x, self.bg_y, xw, yw, self.area)
        dt = tw[:, None] - th[None, :]
        valid = dt > 0
        dt = np.where(valid, dt, 1.0)
        r2 = (xw[:, None] - xh[None, :]) ** 2 + (yw[:, None] - yh[None, :]) ** 2
        dm = mh - mc
        T = tE - ts

        def unpack(z):
            mu, K, a, cc, p1, d, gam, q1 = np.exp(z)
            return mu, K, a, cc, 1 + p1, d, gam, 1 + q1

        def nll(z):
            mu, K, a, cc, p, d, gam, q = unpack(z)
            k = K * np.exp(a * dm)
            s2 = d ** 2 * np.exp(gam * dm)
            g = (p - 1) * cc ** (p - 1) * (dt + cc) ** (-p)
            f = (q - 1) / (np.pi * s2[None, :]) * (1 + r2 / s2[None, :]) ** (-q)
            trig = np.where(valid, k[None, :] * g * f, 0.0).sum(1)
            lam = mu * uw + trig
            # integral: background + each parent's expected offspring inside window and box
            a_lo = np.clip(ts - th, 0, None)
            a_hi = np.clip(tE - th, 0, None)
            Gint = (cc / (a_lo + cc)) ** (p - 1) - (cc / (a_hi + cc)) ** (p - 1)
            frac = _frac_inside(xh, yh, np.sqrt(s2), q, 0, self.xhi, 0, self.yhi)
            integral = mu * T + (k * Gint * frac).sum()
            return integral - np.log(lam + 1e-300).sum()

        z0 = np.log([max(len(tw), 1) / T * 0.3, 0.05, 1.5, 0.01, 0.2, 1.0, 1.0, 0.5])
        bounds = [(-12, 6), (-10, 2), (np.log(0.05), np.log(3.5)), (np.log(1e-4), np.log(1.0)),
                  (np.log(0.01), np.log(1.5)), (np.log(0.1), np.log(20)), (np.log(0.05), np.log(3)),
                  (np.log(0.05), np.log(3))]
        res = minimize(nll, z0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400))
        mu, K, a, cc, p, d, gam, q = unpack(res.x)
        self.params = dict(mu=mu, K=K, a=a, c=cc, p=p, d=d, gam=gam, q=q, nll=float(res.fun),
                           n_events=int(len(tw)), converged=bool(res.success))
        self.cat_x, self.cat_y, self.cat_t, self.cat_m = x, y, tday, c.mag.values
        return self.params

    # ---------------- forecasting ----------------
    def expected_counts(self, cells, s, e, mc=MC, size_deg=0.2):
        """Expected M>=mc count per cell over [s, e), using events before s."""
        P = self.params
        sd = (np.datetime64(s) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        ed = (np.datetime64(e) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
        m = (self.cat_t < sd) & (self.cat_t >= sd - 365)
        x, y, t, dm = self.cat_x[m], self.cat_y[m], self.cat_t[m], self.cat_m[m] - mc
        # background density from the past 365 d (time-varying via u only)
        inb = (x >= 0) & (x < self.xhi) & (y >= 0) & (y < self.yhi)
        cx, cy = _xy(cells.lat0.values, cells.lon0.values, self.lat0, self.lon0)
        cx2, cy2 = _xy(cells.lat0.values + size_deg, cells.lon0.values + size_deg, self.lat0, self.lon0)
        gx = np.linspace(0.1, 0.9, 5)
        bg = np.zeros(len(cells))
        for fx in gx:
            for fy in gx:
                xq = cx + fx * (cx2 - cx)
                yq = cy + fy * (cy2 - cy)
                bg += _bg_density(x[inb], y[inb], xq, yq, self.area)
        bg = bg / 25 * (cx2 - cx) * (cy2 - cy)
        lam = P["mu"] * (ed - sd) * bg
        k = P["K"] * np.exp(P["a"] * dm)
        G = (P["c"] / (sd - t + P["c"])) ** (P["p"] - 1) - (P["c"] / (ed - t + P["c"])) ** (P["p"] - 1)
        s_ = np.sqrt(P["d"] ** 2 * np.exp(P["gam"] * dm))
        for j in range(len(cells)):
            near = (x > cx[j] - 60) & (x < cx2[j] + 60) & (y > cy[j] - 60) & (y < cy2[j] + 60)
            if not near.any():
                continue
            fr = _frac_inside(x[near], y[near], s_[near], P["q"], cx[j], cx2[j], cy[j], cy2[j])
            lam[j] += (k[near] * G[near] * fr).sum()
        return lam
