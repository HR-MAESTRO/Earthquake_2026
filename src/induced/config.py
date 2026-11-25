"""Paths and study constants.

Code lives in the project folder. Data and outputs live in a separate work
folder (default: ~/induced-eq-work), because Windows Controlled Folder Access
blocks Python from writing inside Documents. Override with IEQ_WORK.
"""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("IEQ_WORK", Path.home() / "induced-eq-work"))
RAW = WORK / "data" / "raw"
PROCESSED = WORK / "data" / "processed"
RESULTS = WORK / "results"
for _p in (PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

OGS_CATALOG = RAW / "ogs" / "ogs_catalog_2000_2026_m0.csv"
OCC_DIR = RAW / "occ"

# Central Oklahoma box from the 2023 paper (Sect. 8, Fig. 4)
CENTRAL_OK = dict(lat_min=35.3, lat_max=36.4, lon_min=-97.6, lon_max=-96.4)

# Original study window: Aug 2009 .. Dec 2019 -> 125 months -> 250 half-months
ORIG_START = "2009-08-01"
ORIG_END = "2019-12-31"

# Original feature settings
N_EVENTS = 100          # events used for seismicity indicators
M_THRESH = 3.0          # magnitude threshold for those events
INJ_WINDOW_MONTHS = 9   # injection look-back window

CLASS_EDGES = (3.0, 4.0)  # M<3 -> 0, 3<=M<4 -> 1, M>=4 -> 2
CLASS_NAMES = ("M<3", "3<=M<4", "M>=4")

SEED = 20231
