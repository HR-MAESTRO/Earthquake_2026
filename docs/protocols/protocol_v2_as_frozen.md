# Pre-registered evaluation protocol v2: Fort Worth Basin

Status: FROZEN 2026-11-10, before any Fort Worth injection or catalog data
were downloaded. SHA-256 hashes of this file and of the source files that
define the models (`multibasin.py`, `etas.py`, `grid.py`, `gridmodels.py`,
`regions.py`) are recorded in `induced-eq-work/results/protocol/protocol_v2_frozen.txt`.
Changes after freezing are listed under "Deviations" with reasons, and
results are reported both ways.

## Scope of pre-registration (stated explicitly)

Only the Fort Worth evaluation below is pre-registered. Model development,
the choice of models and settings, and all analyses on Oklahoma, Kansas,
the Delaware Basin and the Midland Basin (including the earlier Midland test
under protocol v1) were done before this protocol and are exploratory with
respect to it.

## Question

Does the leave-one-basin-out hybrid (trained only on other basins) forecast
M≥3 and M≥4 cell-period exceedances in an unseen, geologically distinct basin
(Fort Worth Basin, Ellenburger disposal, north-central Texas) with ranking
skill equivalent to, or better than, locally fitted ETAS and seismogenic-index
forecasts?

## Data

- Box: 32.0–33.6°N, 98.6–96.6°W; 0.2° cells; half-month periods.
- Injection: RRC Form H-10 monthly volumes, disposal wells (UIC type 1), RRC
  UIC well locations (data.texas.gov `qq2j-f2zm`, `givw-z9t4`); 2010–2025.
- Catalog: USGS ComCat, M≥2.5 for features and targets.
- Grid from 2010-01. Test years 2012–2025, yearly walk-forward, 18-period gap.

## Forecast issue time and data availability

- A forecast for period [s, e) is issued at s.
- Catalog: all events before s (treated as real-time).
- Injection: **primary** analysis uses months up to month(s) − 13 (12-month
  reporting lag, matching annual H-10 reporting); secondary: 0- and 3-month lags.
- Geology: static.

## Forecasts compared

| Name | Description |
|---|---|
| `clim` | per-cell past exceedance frequency (Fort Worth past only) |
| `si` | seismogenic index (Fort Worth past only) |
| `etas` | gridded ETAS of protocol v1 (Fort Worth past only) |
| `hybrid_count_lobo` | **Primary.** LightGBM Poisson count model with log λ_ETAS offset, trained on Oklahoma, Kansas, Delaware and Midland (time-respecting), no Fort Worth data; probabilities via Gutenberg–Richter with Fort Worth's past b-value |
| `hybrid_count_pooled` | same, also trained on Fort Worth's own past |
| `lgbm_cls_lobo` | direct exceedance classifier, other basins only |

All settings exactly as in `multibasin.py` at freeze time (git-style hash of
the source file recorded with the protocol).

## Metrics

- Primary: M≥3 PR-AUC, pooled over test years.
- Secondary: M≥4 PR-AUC; Brier skill vs Fort Worth past base rate; calibration
  intercept and slope; Poisson log-likelihood of counts; reliability table.
- Uncertainty: moving-block bootstrap over periods, 6-period blocks, 1,000
  resamples, paired between models. Sensitivity: 3, 12, 24-period blocks and
  resampling by year.

## Decision rules (primary, M≥3, 12-month lag)

Let B = the better of `etas` and `si` by PR-AUC, and Δ = PR-AUC(`hybrid_count_lobo`) − PR-AUC(B).
Equivalence margin δ = 0.2 × PR-AUC(B) (relative, because PR-AUC scales with the base rate).

1. **Superior** if the 95% paired interval of Δ lies above 0.
2. **Equivalent** if the 90% paired interval of Δ lies within (−δ, +δ) (TOST, α = 0.05).
3. **Inferior** if the 95% interval lies below 0.
4. Otherwise **inconclusive**.

If the test years contain fewer than 20 M≥3 positive cell-periods, the result
is reported as **underpowered** regardless of the rule outcome.

Data quality: completeness (maximum curvature + 0.2) is reported for each
year in the Fort Worth box. Years with Mc > 2.5 are flagged; the primary
analysis is not changed, and a secondary analysis restricted to years with
Mc ≤ 2.5 is reported.

Known from development (stated before the test): the leave-one-basin-out
hybrid transferred poorly into Oklahoma (PR-AUC 0.25 vs 0.47 for ETAS on
overlapping years), and well into Kansas, Delaware and Midland.

## Deviations

(none yet)
