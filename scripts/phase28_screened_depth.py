"""Row-level depth to basement with a plausibility screen on reported well depths.

Operator reports contain mistyped depths (for example a packer at 2,318,429 ft in a
2,992 ft well). Before the well table is built, any injection-bottom, packer or total
depth over 25,000 ft (about 7.6 km, deeper than any disposal well in these basins), and
any injection-bottom or packer depth more than 1.5 times the well's reported total depth,
is set to missing. The well table then falls back to the next depth field or to the well's
last plausible value. Everything else is as in phase27 (row-level, records up to each
usable month). Sensitivity analysis only; the reported results stay as run.

The screen changes only Oklahoma. No Kansas, Delaware or Fort Worth record fails it, and
the one Midland well with a mistyped packer depth always reports its injection bottom,
which the well table prefers. The screened Kansas table was checked to be identical to the
phase27 table; for the Texas basins the phase27 tables were copied into the output folder
before running `dev test2026 score`, since `depth()` reuses existing tables.

    python ../scripts/phase28_screened_depth.py           # depth, dev, test2026, score
Outputs: results/phase28_screened_depth/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import phase27_rowlevel_depth as P27
from induced import basement, config

MAX_FT = 25000.0
_well_table = basement.well_table


def screen(inj):
    inj = inj.copy()
    cols = [c for c in ("inj_bot_ft", "packer_ft", "total_depth_ft") if c in inj]
    for c in cols:
        v = inj[c].astype(float)
        inj[c] = v.where(~(v > MAX_FT))
    if "total_depth_ft" in inj:
        td = inj["total_depth_ft"].astype(float)
        for c in ("inj_bot_ft", "packer_ft"):
            if c in inj:
                v = inj[c].astype(float)
                inj[c] = v.where(~((td > 0) & (v > 1.5 * td)))
    return inj


def well_table(inj, bottom="inj_bottom"):
    return _well_table(screen(inj), bottom)


basement.well_table = well_table
P27.OUT = config.RESULTS / "phase28_screened_depth"
P27.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    steps = sys.argv[1:] or ["depth", "dev", "test2026", "score"]
    for s in steps:
        if s == "depth":
            for n in P27.ALL5:
                P27.depth(n)
        elif s == "dev":
            P27.dev()
        elif s == "test2026":
            P27.test2026()
        elif s == "score":
            P27.score()
        print("done", s, flush=True)
