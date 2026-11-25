# Protocol v3 outcome: out-of-time test, January–October 2026

Kept separate from `protocol_v3.md` so that the frozen file still matches its
recorded hash. Run once on 2026-11-22 at 20:13 after all frozen hashes were
verified. No deviations.

## Primary result (M≥3, both basins, 12-month injection lag)

**Verdict: equivalent.**

| Quantity | Value |
| --- | --- |
| Test cell-periods / positives | 2,720 / 61 |
| ETAS PR-AUC | 0.402 [0.268, 0.546] |
| LOBO hybrid PR-AUC | 0.410 [0.284, 0.539] |
| Difference (hybrid − ETAS) | +0.007 |
| 95% paired interval | −0.044 to +0.051 |
| 90% paired interval | −0.035 to +0.044 |
| Equivalence margin (20% of ETAS) | ±0.080 |

With 3- and 12-period blocks the 95% intervals are [−0.051, 0.060] and
[−0.008, 0.047]. Both include zero.

## Secondary results

| Model | M≥3 PR-AUC | M≥3 Brier skill |
| --- | --- | --- |
| LOBO count hybrid | 0.410 | 0.226 |
| Pooled count hybrid | 0.407 | 0.212 |
| ETAS | 0.402 | 0.217 |
| Local count hybrid | 0.399 | 0.222 |
| Seismogenic index | 0.376 | 0.185 |
| LOBO classifier | 0.347 | 0.189 |
| Climatology | 0.322 | 0.204 |

- **Delaware:** 46 positives, difference +0.011 [−0.056, 0.068], equivalent.
- **Midland:** 15 positives, difference +0.003 [−0.011, 0.016]. Underpowered
  under the protocol (fewer than 20 positives).
- **M≥4:** 2 positives, underpowered; descriptive only.
- **Calibration (M≥3):** predicted ÷ observed positive cell-periods is 1.41 for
  ETAS (intercept −0.51, slope 0.98) and 1.25 for the LOBO hybrid (−0.23, 1.03).
- **Calibration (M≥4):** ETAS predicts 8.9 and the hybrid 7.6 positive
  cell-periods; 2 occurred.
- **Counts (M≥2.5):** predicted ÷ observed is 1.05 for ETAS and 0.89 for the
  LOBO hybrid. Mean Poisson log-likelihood is −0.2349 and −0.2391, so ETAS fits
  counts slightly better.

## Limits on interpretation

- Equivalence is defined by a fairly wide pre-set margin (±0.080 PR-AUC).
- 46 of the 61 positives are in Delaware, so the pooled result mostly reflects Delaware.
- Delaware and Midland were both used earlier in the project. This tests new
  *time*, not a new basin, and each basin's LOBO model trains on its Permian neighbour.
- Climatology already scores 0.322, so much of the ranking skill reflects where
  seismicity was already persistent.
- It is a retrospective out-of-time test, not a live forecast.
