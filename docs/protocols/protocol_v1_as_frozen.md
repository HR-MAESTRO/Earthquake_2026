# Pre-registered evaluation protocol v1 (frozen before any Midland Basin data is downloaded)

Frozen: 2026-11-03. Any change after Midland data is downloaded must be listed
under "Deviations" with a reason, and results must be reported both ways.

## Question

Does a physics-anchored ML model trained on several basins give better
probabilistic forecasts of M≥3 and M≥4 earthquakes per 0.2° cell per
half-month than physics or statistical seismology baselines, in a basin it
has never seen?

## Data

- Development basins (already examined): Oklahoma (35–37°N, 99–96°W), Kansas
  (37–38°N, 99–97°W), Delaware Basin (31.0–32.4°N, 104.6–103.0°W).
- **Untouched test basin: Midland Basin, Texas, 31.0–33.0°N, 102.6–101.0°W.**
  Injection: RRC Form H-10 monthly volumes, disposal wells (UIC type 1),
  RRC UIC well locations (data.texas.gov `qq2j-f2zm`, `givw-z9t4`). Catalog:
  USGS ComCat. Same pipeline and code as the Delaware Basin. Grid built from 2017-01.
- Grid: 0.2° cells, half-month periods. Features: `GRID_FEATURES` without the
  fault features (no fault map outside Oklahoma).

## Forecasts compared

| Name | Description |
|---|---|
| `clim` | per-cell historical exceedance frequency (local past only) |
| `si` | seismogenic-index forecast (local past only) |
| `etas` | ETAS-type rate. Background = Zaliapin–Ben-Zion background events in the cell over the past 365 d. Triggering: Omori–Utsu time kernel (c = 0.01 d), productivity 10^(α(m−2.5)), power-law spatial kernel (q = 1.5, d = 1 km·10^(0.5(m−2.5))). α and p chosen from a fixed grid, background scale and K by Poisson maximum likelihood, all on local past data only. |
| `lgbm_cls_pooled` | LightGBM classifier on exceedance, trained on pooled development basins |
| `hybrid_count_pooled` | **Primary model.** LightGBM Poisson regression of the count of M≥2.5 events per cell-period with offset log(λ_ETAS), trained on pooled development basins. Exceedance probabilities from P(M≥m) = 1 − exp(−λ·10^(−b(m−2.5))), with b the maximum-likelihood b-value of the basin's past catalog. |
| `hybrid_count_local` | same as primary, trained on the target basin's past only (not used for Midland zero-shot) |

Hyperparameters for the LightGBM models are fixed before the Midland run
(chosen on development basins). Each yearly fold uses only periods before
(test-year start − 9 months) for any fitting.

## Midland test

- Test years 2020–2025, yearly folds.
- ML components (`lgbm_cls_pooled`, `hybrid_count_pooled`) are trained on
  development-basin data only (zero-shot for Midland).
- `si`, `etas` and `clim` use Midland's own past (they are local physics or
  statistics models and need no labelled training on other basins).
- Run **once**.

## Metrics and decision rule

- Primary: PR-AUC and Brier skill score (vs constant rate = Midland
  past-period base rate) for **M≥3**, pooled over test years.
- Secondary: the same for M≥4; Poisson log-likelihood per cell-period for
  counts (`etas` vs `hybrid_count_pooled`).
- Uncertainty: moving-block bootstrap over periods (6-period blocks, 1000
  resamples), paired between models.
- **Claim "the hybrid beats physics"** only if the paired 95% CI of the
  PR-AUC difference (`hybrid_count_pooled` − best of {`si`, `etas`}) for M≥3
  excludes 0 on Midland. Otherwise report "no significant difference".

## Deviations

(none yet)
