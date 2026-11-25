"""Rerun the 2023 evaluations on the ORIGINAL input matrix used in
Rashidi & Ghassemieh (2023), instead of our rebuilt version.

    python ../scripts/phase20_original_rerun.py <path to original file> [--sheet NAME]

Accepts .csv, .xlsx/.xls or .parquet. The file needs the 10 inputs and the
class (or the period's maximum magnitude). Column names are matched loosely
(e.g. "Mmean", "M_mean", "mean magnitude"); unmatched columns are listed and the
script stops, so nothing is guessed silently. If the file has no dates, rows are
taken in time order and aligned with the 250 half-month periods of 2009-08..2019-12
(an error is raised if the row count differs).

Writes to results/phase20_original/:
  input_check.csv          per-feature agreement with the rebuilt dataset
  phase0/  random-split reproduction (Tables 3/5), chronological split, headline
  phase1/  walk-forward 2013-2017 with 9-month gap (holdout 2018+ excluded)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from induced import config, features as F

OUT = config.RESULTS / "phase20_original"
ALIASES = {
    "T": ["t", "time", "elapsed time", "t days"],
    "Mmean": ["mmean", "m mean", "mean magnitude", "mean m"],
    "dE": ["de", "de1 2", "energy", "sqrt energy", "rate of square root of seismic energy"],
    "b": ["b", "b value", "bvalue"],
    "eta": ["eta", "η", "mean square deviation"],
    "dM": ["dm", "δm", "delta m", "magnitude deficit"],
    "vmean": ["vmean", "v mean", "mean volume", "average volume"],
    "vmax": ["vmax", "v max", "max volume", "maximum volume"],
    "dmean": ["dmean", "d mean", "mean depth", "average distance", "mean distance"],
    "dmin": ["dmin", "d min", "min depth", "minimum distance", "min distance"],
    "y": ["y", "class", "label", "target", "magnitude class"],
    "mmax": ["mmax", "m max", "max magnitude", "maximum magnitude"],
    "start": ["start", "date", "period start", "period"],
}


def norm(s):
    return re.sub(r"[^a-zα-ωηδ0-9]+", " ", str(s).lower()).strip()


def load(path, sheet=None):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(p, sheet_name=sheet or 0)
    elif p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p)
    cols = {}
    for c in raw.columns:
        for k, al in ALIASES.items():
            if norm(c) in al or norm(c) == k.lower():
                cols.setdefault(k, c)
    missing = [k for k in F.FEATURES if k not in cols]
    if missing or not ({"y", "mmax"} & cols.keys()):
        raise SystemExit(f"Could not match columns {missing or ['y or mmax']}.\nFile columns: {list(raw.columns)}\n"
                         "Rename them to T, Mmean, dE, b, eta, dM, vmean, vmax, dmean, dmin and y (or mmax).")
    df = pd.DataFrame({k: raw[c] for k, c in cols.items()})
    if "y" not in df:
        df["y"] = [F.magnitude_class(m) for m in df.mmax]
    df["y"] = df.y.astype(int)
    rebuilt = pd.read_parquet(config.PROCESSED / "orig_dataset.parquet").reset_index(drop=True)
    if "start" in df:
        df["start"] = pd.to_datetime(df.start)
    elif len(df) == len(rebuilt):
        df["start"] = rebuilt.start.values
    else:
        raise SystemExit(f"No date column and {len(df)} rows (rebuilt has {len(rebuilt)}); add a period start column.")
    df = df.sort_values("start").reset_index(drop=True)
    df["end"] = df.start.shift(-1).fillna(df.start.iloc[-1] + pd.Timedelta(days=16))
    return df, rebuilt


def check(df, rebuilt):
    m = df.merge(rebuilt, on="start", suffixes=("_orig", "_rebuilt"))
    rows = []
    for c in F.FEATURES + ["y"]:
        a, b = m[f"{c}_orig"].astype(float), m[f"{c}_rebuilt"].astype(float)
        ok = a.notna() & b.notna()
        rows.append(dict(feature=c, n=int(ok.sum()), pearson=float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                         median_ratio=float((a[ok] / b[ok]).replace([np.inf, -np.inf], np.nan).median()),
                         exact_share=float((np.isclose(a[ok], b[ok], rtol=1e-3)).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "input_check.csv", index=False)
    cls = pd.crosstab(m.y_orig, m.y_rebuilt)
    cls.to_csv(OUT / "class_crosstab.csv")
    print(t.round(3).to_string(), "\n\nclass: original (rows) vs rebuilt (cols)\n", cls, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sheet")
    ap.add_argument("--out", help="output folder (default results/phase20_original)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    df, rebuilt = load(a.path, a.sheet)
    df.to_parquet(OUT / "original_input.parquet")
    print("original input", df.shape, "classes", np.bincount(df.y), flush=True)
    check(df, rebuilt)
    import phase0_reproduce as P0
    import phase1_walkforward as P1
    if "mmax" not in df:
        df["mmax"] = np.nan
    P0.main(df=df.copy(), out=OUT / "phase0")
    P1.main(df=df.copy(), out=OUT / "phase1")


if __name__ == "__main__":
    main()
