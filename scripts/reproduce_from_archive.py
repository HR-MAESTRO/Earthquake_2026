"""Reproduce headline numbers from the Zenodo data archive alone.

    python scripts/reproduce_from_archive.py <path to Earthquake_2026_data>

Needs only this repository's package and the extracted archive
(doi:10.5281/zenodo.24541351). No raw data or model fitting. Recomputes, with the
same seeds as the paper:
  Table 5  paired M>=3 PR-AUC differences (12-month lag, 6-period blocks)
  Table 6  ETAS predicted / observed M>=3 ratio
  Table 4  M>=4 PR-AUC of the pooled hybrid and the offset-free count model (0-month lag)
  Table 7  protocol v3 2026 decision
and compares each with the value printed in the manuscript (to its printed precision).
Exit code 1 if any value differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import multibasin as MB

A = Path(sys.argv[1]) / "forecasts"
DEV = A / "development_lookahead_free"
checks = []


def check(label, got, expected, nd=3):
    ok = round(got, nd) == round(expected, nd) or abs(got - expected) < 0.5 * 10 ** -nd + 1e-12
    checks.append(ok)
    print(f"{'OK ' if ok else 'BAD'} {label}: {got:+.{nd}f} (paper {expected:+.{nd}f})")


def pred(b, lag=12, sort=True):
    d = pd.read_parquet(DEV / f"pred_lag{lag}_{b}.parquet")
    return d.sort_values(["start", "cell"]).reset_index(drop=True) if sort else d


T6 = {  # basin: (LOBO-ETAS, LOBO-SI, pooled-local) as (diff, lo, hi)
    "oklahoma": ((-0.028, -0.042, -0.007), (0.003, -0.024, 0.029), (0.000, -0.001, 0.002)),
    "kansas": ((0.028, -0.014, 0.076), (0.069, 0.010, 0.144), (0.044, 0.012, 0.101)),
    "delaware": ((0.004, -0.008, 0.020), (0.013, -0.015, 0.041), (0.043, 0.016, 0.067)),
    "midland": ((0.001, -0.008, 0.009), (0.001, -0.047, 0.047), (0.028, 0.006, 0.058)),
}
PAIRS = (("hybrid_count_lobo", "etas"), ("hybrid_count_lobo", "si"), ("hybrid_count_pooled", "hybrid_count_local"))
for b, exp in T6.items():
    d = pred(b, sort=False)              # Table 5 used the stored row order and 500 resamples
    for (a, c), e in zip(PAIRS, exp):
        got = MB.paired_diff(d, "ex3", a, c, n_boot=500)
        for k, name in enumerate(("diff", "lo", "hi")):
            check(f"Table 5 {b} {a}-{c} {name}", got[k], e[k])

for b, e in {"oklahoma": 1.51, "kansas": 1.45, "delaware": 1.48, "midland": 1.05}.items():
    d = pred(b)
    check(f"Table 6 {b} ETAS predicted/observed", d.etas_ex3.sum() / d.ex3.sum(), e, 2)

off = pd.read_parquet(A / "review_reruns" / "offset_lag0.parquet")
for b, (eh, ef) in {"oklahoma": (0.077, 0.015), "delaware": (0.099, 0.102), "midland": (0.012, 0.013)}.items():
    d = pred(b, 0).merge(off[off.basin == b][["start", "cell", "count_feat_ex4"]], on=["start", "cell"])
    s = MB.score(d.sort_values(["start", "cell"]).reset_index(drop=True), "ex4", n_boot=10,
                 models=["hybrid_count_pooled", "count_feat"]).set_index("model")
    check(f"Table 4 {b} hybrid M>=4", s.prauc["hybrid_count_pooled"], eh)
    check(f"Table 4 {b} no-offset count M>=4", s.prauc["count_feat"], ef)

p = pd.read_parquet(A / "test_2026_protocol_v3" / "predictions.parquet")
d95 = MB.paired_diff(p, "ex3", "hybrid_count_lobo", "etas", level=0.95)
d90 = MB.paired_diff(p, "ex3", "hybrid_count_lobo", "etas", level=0.90)
for label, got, e in (("diff", d95[0], 0.007), ("95% lo", d95[1], -0.044), ("95% hi", d95[2], 0.051),
                      ("90% lo", d90[1], -0.035), ("90% hi", d90[2], 0.044)):
    check(f"Table 7 2026 LOBO-ETAS {label}", got, e)

print(f"\n{sum(checks)} of {len(checks)} values reproduced")
sys.exit(0 if all(checks) else 1)
