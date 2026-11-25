# Row-level depth-to-basement correction

Script: `scripts/phase27_rowlevel_depth.py`. Supporting Information, Text S4 and Table S5.

## Why

The temporal audit (`scripts/phase26_temporal_audit.py`) found one residual. In the
corrected runs, the volume-weighted depth to basement (`d_basement_w`) was built from
records up to the test year's last usable month, which can be up to 11 months after a
row's own usable month. Three things were taken from those records: the well
attributes, the median used to fill missing depths and, in Oklahoma, the
disposal-well classification. Every other injection feature already used only each
row's own usable month.

## What was done

`d_basement_w` was recomputed for each usable month from records up to that month
only. The development-basin comparisons and the protocol v3 test were then rerun
with otherwise identical code. The protocol results in the paper are still reported
as run.

## Feature change

| Basin | Rows changed | Size of change |
| --- | --- | --- |
| Kansas | none | — |
| Delaware | none | — |
| Midland | about half | at most 29 m |
| Oklahoma | about 80% | 99% under 11 m (2022 fold); a few cells up to about 25 km |

The largest Oklahoma changes are cells where the reported value came from a later,
implausible well record.

## Effect on the primary comparisons

M≥3 PR-AUC differences at the 12-month lag, with 95% intervals (6-period blocks,
500 resamples, stored row order).

| Basin | Comparison | As reported | Row-level depth |
| --- | --- | --- | --- |
| Oklahoma | LOBO hybrid − ETAS | −0.028 [−0.042, −0.007] | −0.027 [−0.042, −0.006] |
| | LOBO hybrid − SI | +0.003 [−0.024, 0.029] | +0.004 [−0.024, 0.030] |
| | Pooled − local hybrid | +0.000 [−0.001, 0.002] | −0.001 [−0.002, 0.001] |
| | LOBO classifier − ETAS | −0.082 [−0.119, −0.048] | −0.082 [−0.118, −0.049] |
| Kansas | LOBO hybrid − ETAS | +0.028 [−0.014, 0.076] | +0.025 [−0.014, 0.068] |
| | LOBO hybrid − SI | +0.069 [0.010, 0.144] | +0.065 [0.010, 0.136] |
| | Pooled − local hybrid | +0.044 [0.012, 0.101] | +0.050 [0.010, 0.104] |
| | LOBO classifier − ETAS | +0.054 [0.006, 0.102] | +0.067 [0.009, 0.136] |
| Delaware | LOBO hybrid − ETAS | +0.004 [−0.008, 0.020] | +0.008 [−0.003, 0.023] |
| | LOBO hybrid − SI | +0.013 [−0.015, 0.041] | +0.017 [−0.012, 0.045] |
| | Pooled − local hybrid | +0.043 [0.016, 0.067] | +0.046 [0.018, 0.070] |
| | LOBO classifier − ETAS | +0.026 [0.008, 0.048] | +0.021 [0.003, 0.042] |
| Midland | LOBO hybrid − ETAS | +0.001 [−0.008, 0.009] | −0.001 [−0.011, 0.007] |
| | LOBO hybrid − SI | +0.001 [−0.047, 0.047] | −0.001 [−0.050, 0.044] |
| | Pooled − local hybrid | +0.028 [0.006, 0.058] | +0.025 [0.002, 0.053] |
| | LOBO classifier − ETAS | +0.009 [−0.017, 0.024] | +0.013 [−0.010, 0.031] |

No comparison changes sign or significance.

**2026 test (protocol v3):** LOBO hybrid − ETAS is +0.007 (90% CI −0.035, 0.044) as
reported and +0.006 (90% CI −0.036, 0.043) with the row-level feature. Both are
equivalent under the prespecified 20% rule.

## Reproducing

Run from `src/`:

```
python ../scripts/phase27_rowlevel_depth.py depth
python ../scripts/phase27_rowlevel_depth.py dev
python ../scripts/phase27_rowlevel_depth.py test2026
python ../scripts/phase27_rowlevel_depth.py score
```

Outputs go to `results/phase27_rowlevel_depth/` under the work folder. The revised
forecasts are not in data archive version 1.0.0.
