# Earthquake_2026: a multi-basin benchmark for forecasting injection-induced earthquakes

[![DOI](https://zenodo.org/badge/1472213291.svg)](https://doi.org/10.5281/zenodo.24541277)

Code archive: version 1.0.0 is doi:[10.5281/zenodo.24541278](https://doi.org/10.5281/zenodo.24541278);
all versions are doi:[10.5281/zenodo.24541277](https://doi.org/10.5281/zenodo.24541277).

Code, pre-registration protocols and analysis notes for the paper *Evaluating the
transferability of probabilistic forecasts of injection-induced earthquakes: a
multi-basin benchmark of statistical, injection-based and machine-learning models*
(manuscript in preparation). It follows Rashidi & Ghassemieh (2023), *Natural
Hazards* 118:545–570, doi:10.1007/s11069-023-06018-6.

**Authors:** Javad N. Rashidi and Hadi Rashidi (contact). Questions and issues go to
the [issue tracker](https://github.com/HR-MAESTRO/Earthquake_2026/issues).

## What is here

The benchmark covers five U.S. basins: Oklahoma, south-central Kansas, and the
Delaware, Midland and Fort Worth basins of Texas. Each is split into 0.2° cells and
half-month periods. Every forecast gives P(M≥3) and P(M≥4) per cell-period and is
built only from inputs assumed available at its issue time. Injection volumes have a
12-month reporting lag by default, so the evaluation is pseudo-prospective.

Models:

- **Statistical:** climatology and a gridded ETAS with a rolling background; a
  full space–time ETAS is used for validation.
- **Injection-based:** the seismogenic index.
- **Data-driven:** LightGBM classifiers and Poisson count models with an ETAS offset,
  trained pooled, leave-one-basin-out or local.

Three tests were pre-registered. Protocol v1 (Midland) and v2 (Fort Worth) were each
frozen before that basin's data were downloaded. Protocol v3 (January–October 2026)
was frozen before the test was run. All three are retrospective.

```
src/induced/      Python package (data loaders, gridding, ETAS, models, metrics)
scripts/          one script per analysis step (see the map below)
docs/protocols/   frozen protocol texts, SHA-256 hash records, outcomes, run markers
docs/analysis_notes/  write-ups of the post hoc sensitivity analyses
```

## Installation

Python 3.13. With a virtual environment:

```
pip install -r requirements.txt
```

Two optional packages:

- **ETH Zurich `etas` package**, used only by the external ETAS comparison
  (`phase11`, `phase14`): `pip install git+https://github.com/lmizrahi/etas`
  (version 3.0.0 was used).
- **TabPFN**, an optional comparator in `phase21`. Its model weights need a Prior
  Labs account; run `phase21` with `--no-tabpfn` to skip it.

Data and outputs go to a work folder, `~/induced-eq-work` by default. Set the
`IEQ_WORK` environment variable to change it. Run scripts from `src/`:

```
cd src
python ../scripts/phase2_build_data.py
```

## Data

All inputs are public. Raw downloads are not redistributed here. The gridded
datasets, saved forecasts and result tables are archived on Zenodo: version 1.0.0 is
doi:[10.5281/zenodo.24541351](https://doi.org/10.5281/zenodo.24541351); all versions are
doi:[10.5281/zenodo.24541350](https://doi.org/10.5281/zenodo.24541350).

- [OGS earthquake catalog](https://ogsweb.ou.edu/eq_catalog/)
- [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/1/), which includes Kansas and
  TexNet contributions
- [OCC UIC injection volumes](https://oklahoma.gov/occ/divisions/oil-gas/oil-gas-data.html)
- [KGS Kansas Class II injection database](https://www.kgs.ku.edu/Magellan/Qualified/class2_db.html)
- [RRC UIC H-10 injection monitoring](https://catalog.data.gov/dataset/uic-h10-injection-monitoring)
- [Marshak et al. (2016) Great Unconformity DEM](https://databank.illinois.edu/datasets/IDB-7546972)
  and USGS 3DEP elevation
- [OGS Oklahoma fault database](https://ou.edu/ogs/data/fault) (Marsh & Holland 2016)
- [US Census 2018 state boundaries, 1:20M](https://www2.census.gov/geo/tiger/GENZ2018/shp/), used for
  the map only

## Script map

| Paper item | Script(s) |
| --- | --- |
| Data download, gridding, injection lags | `phase2_build_data.py`, `build_lagged_grid.py` |
| Section 6.1: the 2023 task (random vs chronological vs walk-forward) | `phase0_reproduce.py`, `phase1_walkforward.py` |
| Section 6.1: modern comparators on the 2023 task | `phase21_modern_models.py` |
| Rerun on the original 2023 input matrix (when available) | `phase20_original_rerun.py` |
| Protocol v1 (Midland) | `phase8_multibasin.py` |
| Lags, pooling and LOBO runs, ablations | `phase9_revision_runs.py`, `phase9_analysis.py` |
| Midland basement sensitivity | `phase9_midland_basement.py` |
| Protocol v2 (Fort Worth) | `phase10_fortworth.py` |
| Protocol v3 (2026 test), Table 8 | `phase12_forward2026.py` |
| Look-ahead corrections, Table 9 | `phase13_leak_corrected.py` |
| Look-ahead-free development reruns, Table 10 | `phase16_clean_dev.py`, `phase16_compare.py` |
| Tables 1, 2, 5–7; Fig. 2 rows | `phase17_tables.py` (full ETAS: `phase9_etas_validation.py`) |
| Table 3 (external `etas` package) | `phase11_etas_package.py`, `phase14_etas_free_background.py`, `phase14_score.py`, `phase15_etas_clean_dev.py` |
| Table 4 (sources of ETAS skill) | `phase22_etas_attribution.py` |
| Table 5 offset-only row; Mc = 3.0 rerun; single-source transfer | `phase23_reruns.py`, `phase23_score.py` |
| Grid offset and cell size; Fig. 4 grid rows | `phase18_grid_sensitivity.py`, `phase18_score.py` |
| Recalibration, Delaware completeness, 2026 block lengths | `phase22_review2.py` |
| Appendix Table A1 | `phase24_consolidated.py` |
| Figs. 1, 3, 4 | `phase19_figures.py` |
| Earlier single-basin work (2023 holdout, declustering, pore pressure, transfer) | `phase34_grid_models.py`, `phase5_transfer.py`, `phase6_final_holdout.py`, `phase7_decluster_pressure.py` |

## Pre-registration records

`docs/protocols/` holds:

- the exact frozen protocol texts and their SHA-256 hash records;
- the deviation logs and outcomes;
- the run markers.

`HASH_CHECK.md` reports which hashed files have changed since each freeze.

## Licences

- **Code:** MIT (`LICENSE`).
- **Documentation, protocols and derived data:** CC BY 4.0 (`LICENSE-DATA.md`).
- **Input data:** remain under their providers' terms.

## Citation

See `CITATION.cff`.
