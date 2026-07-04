# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoWISP is a Python pipeline for extracting high-precision photometry from astronomical observations, especially consumer-grade color cameras (DSLRs). It wraps the lower-level AstroWISP C++/Python library (`astrowisp >= 1.5`) into a full end-to-end pipeline with database management, a Django web UI, and CLI tools.

## Build & Install

Uses Meson build system via `meson-python` backend:

```bash
pip install .                    # Install from source
pip install autowisp             # Install from PyPI
```

## Running Tests

Tests use Python `unittest` (pytest-compatible). They download test data automatically and run pipeline steps sequentially:

```bash
python -m autowisp.tests <failed_test_dir> -v    # Run all tests
python -m autowisp.tests failed_test -v           # CI convention

# Run a single test class
python -m autowisp.tests failed_test -v -k TestCalibrate
```

The `<failed_test_dir>` argument is **required** — it's where artifacts from failed tests are preserved for debugging. Tests run in a temporary directory, copy test data there, and clean up on success.

Test classes (in order of pipeline dependency): `TestCalibrate` → `TestStackToMaster` → `TestFindStars` → `TestSolveAstrometry` → `TestFitStarShape` → `TestMeasureAperturePhotometry` → `TestFitSourceExtractedPSFMap` → `TestFitMagnitudes` → `TestCreateLightcurves` → `TestEPD` → `TestTFA` → `TestDetrendingStat`

Base test class: `AutoWISPTestCase` (extends `astrowisp.tests.utilities.FloatTestCase`). Use `self.run_step(command)` to invoke pipeline CLI commands within tests.

## Linting

```bash
pylint autowisp/                  # Uses .pylintrc config
```

Formatting: Black with 80-char line length. Pylint disables: `duplicate-code`, `fixme`. Constants use a relaxed regex (`[a-z_][a-z0-9_]{2,30}$`). Accepted short variable names include `x`, `y`, `xi`, `eta`, `ra`, `dec` (astronomical conventions).

## Architecture

### Pipeline Steps (`autowisp/processing_steps/`)

Each step is a standalone CLI tool and Python module with a `main()` entry point. The pipeline processes FITS images through these stages:

1. **calibrate** — Generate master bias/dark/flat frames and calibrate raw images
2. **stack_to_master** / **stack_to_master_flat** — Stack calibration frames
3. **find_stars** — Source extraction from images
4. **solve_astrometry** — Plate-solve to map sky coordinates (RA/Dec) to pixel positions
5. **fit_star_shape** — PSF/PRF fitting across the image
6. **measure_aperture_photometry** — Extract flux measurements using PSF-informed apertures
7. **fit_source_extracted_psf_map** — Store PSF model for reuse
8. **fit_magnitudes** — Ensemble photometric calibration across frames
9. **create_lightcurves** — Transpose per-image photometry into per-star time series
10. **epd** / **tfa** — Post-processing detrending (External Parameter Decorrelation, Trend Filtering Algorithm)

CLI tools are prefixed `wisp-*` (e.g., `wisp-calibrate`, `wisp-fit-magnitudes`). All defined in `pyproject.toml` `[project.scripts]`.

### Database Layer (`autowisp/database/`)

- **SQLAlchemy ORM** with SQLite backend (`autowisp.db` in project home directory)
- `interface.py` — Global engine/session management via `set_project_home()` and `start_db_session()` context manager
- `image_processing.py` / `lightcurve_processing.py` — Orchestrate pipeline step execution with dependency tracking
- `data_model/` — 25+ ORM models (Image, Target, ObservingSession, PipelineRun, HDF5 products, provenance tracking for telescope/camera/instrument)
- Database is auto-initialized on first access when `autowisp.db` doesn't exist

### Data Flow

- **Input**: Raw FITS images (bias, dark, flat, object frames)
- **Intermediate storage**: HDF5 files (`hdf5_file.py`, `data_reduction/`) and SQLite database
- **Output**: Light curve files, detrending statistics
- **Catalog**: GAIA catalog queries via `catalog.py` (extended WISPGaia class using `astroquery`)

### Processor Pattern (`processor.py`)

Base class `Processor` enforces a uniform interface for configuration, with `__init__` for setup and `__call__` for execution. Processing steps inherit from this.

### Browser Interface (`autowisp/browser_interface/`)

Django 5 web application (under development). Launch with `wisp-bui [port]`. Django apps: `home`, `core`, `configuration`, `processing`, `results`. Uses separate SQLite database (`bui_db.sqlite3`).

### Key Modules

- `catalog.py` — GAIA catalog queries with POLYGON-based spatial filtering
- `astrometry/` — Coordinate transformations, gnomonic projections, plate solving
- `magnitude_fitting/` — Linear ensemble photometry calibration with iterative refinement
- `fit_expression/` — ANTLR4-based custom expression parser for user-defined fitting terms
- `image_calibration/` — Master frame generation (HAT/HATSouth-style implementation)
- `source_finder.py` — Star detection using brightness thresholds (0.999 quantile)
- `evaluator.py` — Expression evaluation for user-defined processing parameters

### Relationship to AstroWISP

AstroWISP (`/home/kpenev/projects/git/AstroWISP/`) is the lower-level C++/Python library providing core PSF/PRF fitting and aperture photometry algorithms. AutoWISP depends on it (`astrowisp >= 1.5`) and wraps it into the full pipeline. The test base class chain is: `AutoWISPTestCase` → `astrowisp.tests.utilities.FloatTestCase`.

## Key Constraints

- Both **NumPy 1 and NumPy 2** are supported (`numpy >= 1.21`), via
  `astrowisp >= 2.0.1` which is compatible with both. Note NumPy **2.4** made
  `float()`/`int()` of a 1-element array a hard error (2.3 only warned) — use
  `.item()` / explicit indexing, and validate under the numpy the CI installs
  (2.4.x), not an older one.
- Python **3.11+**: 3.11 floor for `ProcessPoolExecutor`'s
  `max_tasks_per_child` (used by `run_pool`); no upper ceiling. CI runs a grid
  of numpy 1/2 × {Linux, Windows, macOS-arm, macOS-intel} × Python 3.11–3.14
  (numpy 1 excluded on 3.13/3.14, which have no numpy-1.26 wheels).
- Cross-platform: Linux, macOS, Windows. Windows-specific gotchas: `numpy.uint`/
  `numpy.int_` are 32-bit there (use `numpy.uint64` for source IDs), and
  serialize paths with `Path.as_posix()`.
