# External ETAS with a spatially varying background (2026-11-23)

Post hoc validation, not pre-registered. Development basins only: Oklahoma
2014–2022 and Delaware 2019–2025. Scripts: `scripts/phase14_etas_free_background.py`
and `scripts/phase14_score.py`. Outputs: `induced-eq-work/results/phase14_etas_freebg/`.

## Setup

ETH Zurich `etas` package, free-background mode, with the same windows and
constraints as the uniform-background run (α = β, τ = 10⁴ days, ω = −0.16). The
background is a Gaussian-smoothed map of the fitted background probabilities.
Smoothing variance: 1 km² (package default, primary) and 25 km² (sensitivity).
All 32 yearly fits completed, and all models are scored on identical rows. The
gridded and uniform-package scores reproduce the earlier values.

## M≥3 and M≥4 PR-AUC

| Basin | Target | Gridded ETAS | Our full ETAS | Package, uniform | Package, free 1 km² | Package, free 25 km² |
| --- | --- | --- | --- | --- | --- | --- |
| Oklahoma | M≥3 | 0.466 | 0.464 | 0.414 | 0.355 | 0.391 |
| Oklahoma | M≥4 | 0.084 | 0.062 | 0.036 | 0.029 | 0.033 |
| Delaware | M≥3 | 0.601 | 0.591 | 0.466 | 0.484 | 0.487 |
| Delaware | M≥4 | 0.103 | 0.101 | 0.064 | 0.056 | 0.063 |

## Paired differences in PR-AUC [95% CI]

| Basin | Target | Gridded − free 1 km² | Gridded − free 25 km² | Free 1 km² − uniform | Free 25 km² − uniform |
| --- | --- | --- | --- | --- | --- |
| Oklahoma | M≥3 | +0.111 [0.069, 0.163] | +0.075 [0.041, 0.117] | −0.059 [−0.093, −0.031] | −0.023 [−0.045, −0.009] |
| Oklahoma | M≥4 | +0.055 [0.011, 0.125] | +0.051 [0.007, 0.121] | −0.007 [−0.013, −0.002] | −0.002 [−0.006, 0.000] |
| Delaware | M≥3 | +0.117 [0.066, 0.151] | +0.113 [0.063, 0.158] | +0.018 [−0.039, 0.071] | +0.022 [−0.006, 0.055] |
| Delaware | M≥4 | +0.047 [0.020, 0.082] | +0.040 [0.016, 0.072] | −0.008 [−0.033, 0.011] | −0.001 [−0.011, 0.006] |

## Other scores (M≥3 unless stated)

| Basin | Quantity | Gridded | Full | Uniform | Free 1 km² | Free 25 km² |
| --- | --- | --- | --- | --- | --- | --- |
| Oklahoma | Brier skill | 0.240 | 0.287 | 0.246 | 0.212 | 0.237 |
| Oklahoma | Predicted ÷ observed positives | 1.52 | 0.81 | 0.77 | 0.94 | 0.79 |
| Oklahoma | Predicted ÷ observed M≥2.5 count | 1.25 | 0.55 | 0.62 | 0.71 | 0.63 |
| Oklahoma | Mean Poisson log-likelihood (M≥2.5) | −0.456 | −0.507 | −0.577 | −0.685 | −0.615 |
| Delaware | Brier skill | 0.379 | 0.396 | 0.314 | 0.338 | 0.328 |
| Delaware | Predicted ÷ observed positives | 1.45 | 0.79 | 0.89 | 0.85 | 0.83 |
| Delaware | Predicted ÷ observed M≥2.5 count | 1.22 | 0.52 | 0.70 | 0.60 | 0.64 |
| Delaware | Mean Poisson log-likelihood (M≥2.5) | −0.371 | −0.407 | −0.474 | −0.511 | −0.492 |

## Fit stability

Nine of the 32 fits have `rho` at its 5.0 limit, `a` at its 0.01 limit, `a` > 8
or `log10_k0` > 4. They affect Oklahoma 2014, 2015 and 2022 and Delaware
2019–2022. Dropping those years from every model (a rule set after seeing the
parameters but before scoring) leaves Oklahoma with 6 years and Delaware with 3.

- Gridded ETAS still ranks above the free-background package at M≥3: Oklahoma
  +0.044 [0.025, 0.062] (1 km²) and +0.037 [0.021, 0.054] (25 km²); Delaware
  +0.081 [0.003, 0.106] and +0.093 [0.022, 0.121].
- Free and uniform backgrounds are then not significantly different at M≥3.

## Recheck with the look-ahead-free gridded ETAS

The gridded ETAS above used a declustering threshold fitted on each basin's full
catalog. `scripts/phase15_etas_clean_dev.py` refits it with the threshold fitted
only before each fold's training cutoff. A control with the original background
reproduced the stored forecasts exactly (maximum difference 0.0).

| Basin | Target | Gridded, original | Gridded, corrected | Corrected − uniform package | Corrected − free 1 km² | Corrected − free 25 km² |
| --- | --- | --- | --- | --- | --- | --- |
| Oklahoma | M≥3 | 0.466 | 0.463 | +0.049 [0.024, 0.074] | +0.108 [0.067, 0.161] | +0.072 [0.038, 0.114] |
| Oklahoma | M≥4 | 0.084 | 0.081 | +0.046 [0.004, 0.113] | +0.052 [0.010, 0.119] | +0.048 [0.005, 0.116] |
| Delaware | M≥3 | 0.601 | 0.590 | +0.124 [0.062, 0.176] | +0.106 [0.056, 0.140] | +0.103 [0.050, 0.146] |
| Delaware | M≥4 | 0.103 | 0.096 | +0.032 [0.009, 0.061] | +0.040 [0.013, 0.073] | +0.033 [0.010, 0.063] |

The conclusion is unchanged. Corrected and original differ by −0.003 [−0.004,
−0.001] in Oklahoma and −0.011 [−0.024, 0.001] in Delaware at M≥3. The corrected
gridded and full ETAS are not distinguishable. Oklahoma, Kansas and Delaware
hybrid models have not been rerun with this correction.

## Interpretation

- **A spatially varying background alone does not close the gap.** The earlier
  suggestion that it would explain our advantage is not supported.
- **The package and gridded forecasts still differ in another untested way.** The
  package's background is fixed from a fit ending before the 9-month gap, while
  the gridded background is recomputed at each forecast from the prior 365 days.
  This is a candidate explanation and has not been tested.
- **The gridded ETAS ranks best but is not best calibrated.** It over-predicts
  positive cell-periods by about 50% in both basins, and our full ETAS has higher
  Brier skill.
- **Parameter values from unstable fits should not be interpreted physically.**
- For the manuscript: describe the main baseline as a gridded, ETAS-style
  benchmark with a rolling background. Do not say an independent implementation
  reproduced it.
