# Grid offset and cell-size sensitivity (2026-11-24)

Post hoc check of the development-basin results (Oklahoma, Kansas, Delaware,
Midland) at the primary 12-month injection lag, with the phase 16 look-ahead
corrections. Fort Worth is not used (held out).

Scripts: `scripts/phase18_grid_sensitivity.py` (build, control, run),
`scripts/phase18_score.py` (scores; `summary` rebuilds `summary.txt` from the CSVs).
Outputs: `induced-eq-work/results/phase18_grid/` (`scores.csv`, `paired.csv`,
`summary.txt`, `midland_tost.csv`).

## Variants

| Variant | Cell | Shift | M≥3 base rate (OK / KS / DE / Mid) |
| --- | --- | --- | --- |
| main | 0.2° | none | 4.3% / 0.40% / 4.2% / 1.0% |
| offset | 0.2° | +0.1° lat and lon | 4.3% / 0.33% / 3.6% / 1.0% |
| size01 | 0.1° | none | 1.2% / 0.10% / 1.3% / 0.25% |
| size04 | 0.4° | none | 12.1% / 1.3% / 10.9% / 3.8% |

Same models, settings, pooled basins and test years as phase 16. PR-AUC is not
comparable across cell sizes (base rates change), so the comparisons are paired
differences within a variant.

## Controls

All maximum differences are 0.0:

- the rebuilt 0.2° lag-12 grids match the stored grids in all four basins;
- ETAS components computed directly match the cache;
- the patched loop reproduces the phase 16 forecasts for 2019–2020.

## Results (M≥3, paired PR-AUC differences, 6-period blocks, 95% CI)

| Comparison | Main | Offset | 0.1° | 0.4° |
| --- | --- | --- | --- | --- |
| OK LOBO hybrid − ETAS | −0.028* | −0.024* | −0.071* | −0.057* |
| KS LOBO hybrid − ETAS | +0.028 | +0.014 | −0.008 | −0.007 |
| DE LOBO hybrid − ETAS | +0.004 | +0.012 | −0.007 | +0.009 |
| Mid LOBO hybrid − ETAS | +0.001 | **−0.022*** | +0.003 | −0.027 [−0.064, 0.000] |
| OK LOBO classifier − ETAS | −0.082* | −0.106* | −0.156* | −0.086* |
| KS LOBO classifier − ETAS | +0.054* | +0.069* | +0.004 | +0.033 |
| DE LOBO classifier − ETAS | +0.026* | +0.024 | −0.004 | −0.003 |
| KS pooled − local | +0.044* | +0.061* | +0.013 | +0.033 |
| DE pooled − local | +0.043* | +0.030 | +0.041* | +0.026* |
| Mid pooled − local | +0.028* | +0.007 | +0.022* | +0.047 |
| OK pooled − local | +0.000 | +0.000 | +0.003 | +0.001 |
| ETAS − SI | never significantly negative in any basin or variant | | | |

`*` = 95% CI excludes zero.

## What holds

1. **Oklahoma.** LOBO hybrid and LOBO classifier are significantly below ETAS on
   every grid.
2. **ETAS among the baselines.** ETAS is best or tied in every basin and grid. SI
   is never significantly better.
3. **Kansas and Delaware.** LOBO hybrid − ETAS is not significant on any grid.
4. **Pooling.** Pooled − local is positive in all 12 Kansas/Delaware/Midland cases
   and about zero in Oklahoma. It is significant in 7 of 12.
5. **Oklahoma M≥4.** ETAS has the highest PR-AUC on every grid. The pooled
   classifier stays far below it (0.007–0.033 vs 0.050–0.081).

## What does not hold

1. **Midland equivalence at ±0.01 / ±0.02 depends on the grid.** The 90% CIs for
   LOBO hybrid − ETAS:

   | Grid | 90% CI | Within ±0.01 / ±0.02 |
   | --- | --- | --- |
   | Main | [−0.007, 0.007] | yes |
   | 0.1° | [−0.003, 0.008] | yes |
   | Offset | [−0.036, −0.007] | no |
   | 0.4° | [−0.058, −0.003] | no |

   On every grid, the difference stays inside ±0.08, which is the margin used in
   the 2026 test. The pre-registered Midland v1 result was on the main grid and is
   unchanged.
2. **LOBO classifier > ETAS (Kansas, Delaware) depends on the grid.**
   - Kansas: holds under the offset, but not at 0.1° or 0.4° with blocks.
   - Delaware: holds only on the main grid.

   Both were already labelled exploratory.
3. **Pooling-gain significance in individual basins depends on the grid.** Only
   the direction is robust.
4. **Midland M≥4 pooled classifier.** The main-grid value of 0.051 is not
   reproduced (0.004, 0.004, 0.025).
5. **Kansas M≥4.** There are no M≥4 positives in the offset grid's test years.

## Limits

- Post hoc; one offset (half a cell) and two sizes.
- No ablation runs for the variants.
- 0.4° boxes extend up to 0.2° beyond the original box where the box is not a
  multiple of 0.4°.
- The offset keeps the catalog padding of the original box.
