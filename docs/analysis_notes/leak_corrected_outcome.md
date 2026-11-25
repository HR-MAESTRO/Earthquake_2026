# Look-ahead-corrected reruns and external ETAS comparison (2026-11-22)

The pre-registered Midland (v1) and Fort Worth (v2) results stand as run. This
is a post hoc sensitivity analysis (`scripts/phase13_leak_corrected.py`).

## What changed in the corrected runs

For each test year and basin:

- injection records after the last month usable in that year are removed before
  disposal-well selection, well attributes, medians and injection features;
- the declustering threshold is fitted only on events before that fold's
  training cutoff. With fewer than 50 prior events, the fold is not declustered
  (Fort Worth 2012–2014).

Not corrected: Texas well type and location come from the current RRC table, and
catalog values are the current versions. Within a test year, well attributes can
use records up to that year's last usable month.

## Results (M≥3 PR-AUC)

| Test | Version | Model | ETAS | SI | Model − best baseline [95% CI] | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Midland, 0-month lag (primary v1) | original | 0.154 | 0.154 | 0.157 | −0.003 [−0.055, 0.047] | no significant difference |
| | corrected | 0.147 | 0.151 | 0.157 | −0.010 [−0.060, 0.041] | no significant difference |
| Fort Worth, 12-month lag (primary v2) | original | 0.165 | 0.240 | 0.095 | −0.076 [−0.176, 0.019] | inconclusive |
| | corrected | 0.185 | 0.229 | 0.095 | −0.044 [−0.134, 0.059] | inconclusive |
| Fort Worth, 0-month lag (secondary) | original | 0.195 | 0.240 | 0.056 | −0.046 [−0.084, −0.001] | inferior |
| | cache fixed | 0.199 | 0.241 | 0.056 | −0.042 [−0.078, −0.003] | inferior |
| | corrected | 0.199 | 0.234 | 0.056 | −0.034 [−0.072, 0.014] | **inconclusive** |
| Fort Worth, 3-month lag (secondary) | original | 0.231 | 0.240 | 0.057 | −0.010 [−0.036, 0.020] | equivalent |
| | cache fixed | 0.232 | 0.241 | 0.057 | −0.009 [−0.033, 0.023] | equivalent |
| | corrected | 0.240 | 0.233 | 0.057 | +0.007 [−0.022, 0.043] | equivalent |

Both primary verdicts are unchanged. One secondary verdict changes.

## Controls

- Midland and the Fort Worth 12-month run were reproduced exactly from the
  original inputs (maximum forecast difference 0.0).
- **Bug found in the original Fort Worth secondary runs.** The script ran the
  12-, 0- and 3-month lags in one process, and an in-memory cache reused the
  12-month ETAS fits for the other two. Recomputing with the reused fits matches
  the original forecasts exactly. The "cache fixed" rows give the values without
  this bug. The primary 12-month result is unaffected.

## External ETAS (ETH Zurich `etas` package, Mizrahi et al. 2021)

The package's default inversion diverged on Delaware, so we used its documented
constrained form: α = β, τ = 10⁴ days, ω = −0.16, with a uniform background rate.

| Basin | Target | Gridded ETAS | Our full ETAS | Package ETAS | Gridded − package [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Oklahoma | M≥3 | 0.466 | 0.464 | 0.414 | +0.052 [0.027, 0.076] |
| Oklahoma | M≥4 | 0.084 | 0.062 | 0.036 | +0.049 [0.006, 0.118] |
| Delaware | M≥3 | 0.601 | 0.591 | 0.466 | +0.135 [0.076, 0.189] |
| Delaware | M≥4 | 0.103 | 0.101 | 0.064 | +0.039 [0.015, 0.073] |

Predicted ÷ observed M≥2.5 counts: package 0.62 (Oklahoma) and 0.70 (Delaware);
gridded 1.25 and 1.22. Rank correlation with gridded ETAS is 0.74 and 0.77.

**Interpretation.** The package does not confirm that our gridded ETAS and a
standard ETAS are interchangeable. Our version ranks significantly better than
this constrained, uniform-background configuration. A likely reason is that our
background follows recent activity in each cell, but this has not been tested.
The package also supports a spatially varying background, which would be the
fair next comparison.

**Update (2026-11-23).** That comparison was run. A fitted spatial background did
not close the gap: see `docs/etas_free_background_outcome.md`.
