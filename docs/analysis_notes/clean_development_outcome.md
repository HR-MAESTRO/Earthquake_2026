# Look-ahead-free development reruns (2026-11-23)

Post hoc sensitivity analysis of the phase 9 runs: Oklahoma, Kansas, Delaware and
Midland at 0-, 3- and 12-month lags. Scripts: `scripts/phase16_clean_dev.py`,
`scripts/phase16_compare.py`. Outputs: `induced-eq-work/results/phase16_clean_dev/`.

## Controls

- With nothing removed, rebuilt injection features match the original grids exactly.
- Control runs through the same loop reproduce all phase 9 forecasts (maximum
  difference 0.0) after one fix.
- **The fix.** The first run dropped basins with no grid rows yet from the pooled
  empirical magnitude ratio. This changed only `hybrid_count_emp` in Oklahoma
  2014–2016. The script was fixed, and that column was recomputed in the stored
  control and corrected files with the phase 9 pooling.

## M≥3 PR-AUC at the 12-month lag (original → corrected)

| Basin | ETAS | LOBO hybrid | LOBO classifier | LOBO hybrid − ETAS, corrected | Pooled − local, corrected |
| --- | --- | --- | --- | --- | --- |
| Oklahoma | 0.466 → 0.463 | 0.254 → 0.251 | 0.192 → 0.197 | −0.028 [−0.043, −0.007] | +0.000 [−0.001, 0.002] |
| Kansas | 0.126 → 0.114 | 0.138 → 0.142 | 0.153 → 0.167 | +0.028 [−0.014, 0.077] | +0.044 [0.012, 0.099] |
| Delaware | 0.601 → 0.590 | 0.605 → 0.594 | 0.617 → 0.616 | +0.004 [−0.008, 0.020] | +0.043 [0.016, 0.069] |
| Midland | 0.154 → 0.151 | 0.152 → 0.151 | 0.155 → 0.159 | +0.001 [−0.008, 0.009] | +0.028 [0.005, 0.058] |

Oklahoma LOBO values cover 2017–2022; ETAS covers 2014–2022.

## Claims changed

1. **"ETAS is best or tied in every basin" now applies among baselines only.**
   LOBO classifier − ETAS at 12 months: Kansas +0.054 [0.005, 0.103] (43 positives),
   Delaware +0.026 [0.009, 0.050]. With 6-period blocks neither is significant at 0
   or 3 months. Whole-year resampling makes Delaware significant at 3 months.
   These are exploratory comparisons among many models.
2. **"M≥4 skill comes from ETAS anchoring" now applies to Oklahoma only.** In
   Delaware at lag 0, the plain count model minus ETAS is +0.028 [−0.029, 0.105].
3. **Pooling under whole-year resampling.** At 12 months the gains are significant
   in Kansas and Delaware after correction, and borderline in Midland.

## Manuscript switched to corrected runs (2026-11-24, draft v0.6)

Tables 2–5, the figure and all text numbers from the development runs now use
these corrected runs (`scripts/phase17_tables.py`, outputs in
`results/phase16_clean_dev/tables/`). Fort Worth rows and the pre-registered
Midland v1 result stay as run. Table 8 keeps original → corrected values.

Changes beyond the table values:

- Midland post hoc equivalence (LOBO hybrid vs ETAS) tightens: within ±0.01 at
  12 months (90% CI [−0.007, 0.007]); within ±0.02 at 0 months ([−0.014, 0.004]).
- Whole-year resampling: pooling gains stay significant in Kansas, Delaware and
  Midland, with lower limits of 0.001–0.004. With 12- and 24-period blocks the
  Midland pooling gain is not significant. Delaware LOBO − SI is never significant.
- Oklahoma ETAS predicts 3.6× (was 3.7×) the observed M≥4 cell-periods; LOBO count
  ratios span 0.60–1.58 (was 0.60–1.55).
- Midland without basement features: LOBO hybrid 0.151 → 0.150.
- Stable-year free-background gaps (M≥3): 0.036–0.043 Oklahoma, 0.077–0.089 Delaware.
- Table 2: gridded vs full ETAS differences all within ±0.02 at M≥3; Delaware
  rate-map Spearman 0.76 (was 0.80).
