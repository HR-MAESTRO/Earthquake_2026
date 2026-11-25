"""Protocol v1: pooled multi-basin hybrid vs physics/ETAS baselines.

    python ../scripts/phase8_multibasin.py dev       # development basins
    python ../scripts/phase8_multibasin.py midland   # untouched test, run ONCE
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from induced import config, multibasin as MB

OUT = config.RESULTS / "phase8"
OUT.mkdir(parents=True, exist_ok=True)
DEV = ["oklahoma", "kansas", "delaware"]


def report(pred, tag):
    rows, diffs = [], []
    for col in ("ex3", "ex4"):
        rows.append(MB.score(pred, col))
        for base in ("si", "etas"):
            for a in ("hybrid_count_pooled", "hybrid_count_local", "lgbm_cls_pooled"):
                if f"{a}_{col}" in pred:
                    d, lo, hi = MB.paired_diff(pred, col, a, base)
                    diffs.append(dict(target=col, model=a, minus=base, diff=d, lo=lo, hi=hi))
    sc = pd.concat(rows)
    df = pd.DataFrame(diffs)
    ll = {c: MB.poisson_ll(pred, c) for c in ("lam_etas", "lam_hybrid_pooled", "lam_hybrid_local") if c in pred}
    sc.to_csv(OUT / f"scores_{tag}.csv", index=False)
    df.to_csv(OUT / f"paired_diffs_{tag}.csv", index=False)
    pd.Series(ll).to_csv(OUT / f"poisson_ll_{tag}.csv")
    print(sc.round(4).to_string())
    print(df.round(4).to_string())
    print("Poisson log-lik per cell-period:", {k: round(v, 4) for k, v in ll.items()})


def main(mode):
    data = {}
    for name in DEV + (["midland"] if mode == "midland" else []):
        print("loading", name, flush=True)
        data[name] = MB.load_basin(name)
    if mode == "dev":
        for name in DEV:
            print(f"\n=== {name} ===", flush=True)
            pred = MB.run(name, data, DEV, MB.BASINS[name]["test"])
            pred.to_parquet(OUT / f"pred_dev_{name}.parquet")
            report(pred, f"dev_{name}")
    else:
        stamp = OUT / "MIDLAND_RUN.txt"
        if stamp.exists():
            print("WARNING: Midland test already run at", stamp.read_text().strip())
        stamp.write_text(pd.Timestamp.now().isoformat())
        print("\n=== midland (untouched, zero-shot ML) ===", flush=True)
        pred = MB.run("midland", data, DEV, MB.BASINS["midland"]["test"], local_ml=False)
        pred.to_parquet(OUT / "pred_midland.parquet")
        report(pred, "midland")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
