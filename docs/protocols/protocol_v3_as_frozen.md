# Pre-registered evaluation protocol v3: out-of-time test, January–October 2026

Status: FROZEN before any 2026 forecast was scored. The timestamp and SHA-256
hashes of this file, the runner, the model source and every input file are in
`induced-eq-work/results/protocol/protocol_v3_frozen.txt`. Any later change is
listed under "Deviations" and results are reported both ways.

## What this test is, and is not

It is a single out-of-time test of models fixed before their 2026 forecasts
were scored. It is **not** a live prospective forecast: the 2026 catalog and
injection files were already downloaded when this protocol was written.

Known before the freeze:

- 2026 event counts inside each basin box: Delaware 80 M≥3 and 1 M≥4; Midland
  18 M≥3 and 1 M≥4; Oklahoma 5 M≥3; Kansas and Fort Worth none. These are event
  counts, not positive cell-periods.
- All results through December 2025, including the Midland (v1) and Fort Worth
  (v2) tests.
- No forecast for any 2026 period has been computed or scored. A rehearsal of
  the runner used 2025 only.

## Question

For M≥3 per 0.2° cell and half-month in the Delaware and Midland basins during
2026, does a count hybrid trained only on *other* basins rank cells better than,
equivalently to, or worse than the gridded ETAS fitted on the local past?

## Data

- Test periods: the 20 half-months from 2026-01-01 through 2026-10-31.
- Test basins: Delaware (56 cells) and Midland (80 cells); 2,720 cell-periods.
- Training basins for the data-driven models: Oklahoma, Kansas, Delaware,
  Midland and Fort Worth, using only rows before the cutoff.
- Sources: the snapshot files hashed in the freeze record. They are **not**
  refreshed after the freeze, so later catalog or injection revisions cannot
  affect the test.

## Forecast issue time and look-ahead controls

- A forecast for [s, e) is issued at s.
- Training cutoff for every model: **2025-04-01** (18 half-months before the
  first test period).
- Catalog: events before s.
- Injection: months up to month(s) − 13 (12-month lag). Records after September
  2025 are removed before wells and features are built.
- The Zaliapin–Ben-Zion declustering threshold is fitted only on events before
  the cutoff. Earlier runs (including v1 and v2) fitted it on the full catalog.

Remaining limitations: catalog magnitudes and locations are the versions in the
snapshot, not those available at each issue time. Texas well type and location
come from the current RRC well table.

## Forecasts compared

| Name | Role |
|---|---|
| `hybrid_count_lobo` | **Primary.** Poisson LightGBM with log λ_ETAS offset, trained on the other four basins |
| `etas` | **Primary comparator.** Gridded ETAS fitted on the target basin's past |
| `si`, `clim` | secondary baselines |
| `lgbm_cls_lobo`, `hybrid_count_pooled`, `hybrid_count_local` | secondary data-driven models |

Model settings are unchanged from protocol v2. The runner is
`scripts/phase12_forward2026.py`; its hash and those of the model source files
are in the freeze record. The comparator is fixed as `etas` in advance, so no
baseline is selected after seeing the test.

## Metrics

- **Primary:** M≥3 PR-AUC over both basins' test cell-periods.
- **Secondary:** results per basin; M≥4 PR-AUC; Brier skill against each basin's
  past base rate; calibration intercept and slope; predicted-to-observed counts;
  Poisson log-likelihood.
- **Uncertainty:** paired moving-block bootstrap over periods, 6-period blocks,
  1,000 resamples. Sensitivity: 3- and 12-period blocks.

## Decision rule (primary)

Let Δ = PR-AUC(`hybrid_count_lobo`) − PR-AUC(`etas`) and δ = 0.2 × PR-AUC(`etas`).

1. **Superior** if the 95% paired interval of Δ lies above 0.
2. Otherwise **equivalent** if the 90% paired interval lies within (−δ, +δ).
3. Otherwise **inferior** if the 95% interval lies below 0.
4. Otherwise **inconclusive**.

If the test contains fewer than 20 positive M≥3 cell-periods, the result is
reported as **underpowered** whatever the rule returns. M≥4 is descriptive only,
since at most a handful of positives are expected.

## Execution

The runner writes a timestamp and refuses a second run. If it crashes after
scoring starts, only the failing step may be fixed, and the fix is logged here.

## Deviations

(none yet)
