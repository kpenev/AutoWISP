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

**Adding or removing a `*.py` file requires editing the matching `meson.build`.**
Each `meson.build` enumerates every source by name in `py.install_sources([...])`
— there is no globbing. Deleting a module without removing its line breaks the
build; adding one without a line silently omits it from the installed package.
Update both in the same change, then sanity-check with a build/import.

*Auditing what is installed:* compare each directory's `*.py` / `*.html` / `*.js`
/ `*.css` against the `'name'` entries in its `meson.build`, and check that
`subdir(...)` covers every child directory. A file tracked by git but absent
from its `meson.build` is silently not installed, and it bites much later. Ask
per file before adding or deleting — a module is not dead merely because nothing
imports it, and a commit subject is not evidence about its contents.

*Deliberate exclusions — do not re-raise these unasked:* `fake_image/` and
`magnitude_fitting/tests/` have no `meson.build` at all and are not installed
(both are also in `.coveragerc`'s omit list). `tests/generate_catalog_test_data.py`
and `tests/update_hdf5_contents.py` are test-data tooling rather than suite
members, and are excluded on purpose — there is a comment saying so at the top of
`autowisp/tests/meson.build`, because an audit flags them otherwise.

**Pipeline steps run the *installed* package, not your working tree.** Tests
invoke each step as a separate `wisp-*` console-script subprocess whose cwd is a
temp directory, so it imports `autowisp` from site-packages, while in-process
unit tests import the working tree. The two diverge silently after an edit,
which can produce a false pass. Use `pip install -e .` for local iteration, or
re-run `pip install .` after every source change. To check: `which wisp-<step>`,
then run its interpreter from outside the repo and inspect
`autowisp.<module>.__file__`.

**Exception — the browser interface:** do *not* use `-e` for BUI work; the
editable install breaks BUI styling (meson-python lays down no
`autowisp/browser_interface/` tree in site-packages, so the `static/` and
`templates/` files go missing while imports still succeed). Use a plain
`pip install .`, repeated after every change, then hard-refresh the browser
(Cmd-Shift-R).

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

**Not unit-tested, by choice.** No Django test-client tests for views, no
template or JS tests — the BUI keeps changing and exposes no contract worth
pinning, so such tests would cost more in churn than they catch. Verify BUI
behaviour by running it and looking at it. Test the layers *below* the views,
where the rules live, and keep decisions separable from rendering so they stay
testable there (e.g. `plan_spare_row` returns the row entry and the caller
renders it).

**Configuration view redesign pending.** The orgchart decision tree
(`configuration/config_tree.html` +
`static/configuration/js/autowisp.config.tree.js`) is slated for a redesign.
Implement config-view features against the existing tree without gold-plating
its styling; raise the pending redesign before any substantial rework of its
presentation.

**Configuration conditions vs versions.** Conditions (several values per
parameter, each guarded by header expressions, first match wins) are exercised
regularly and work as intended. Versions (`Configuration.version`, the
"Version: N" dropdown) have never really been used and are probably not fully
implemented — don't document them as something to rely on. A single project's
`autowisp.db` showing one value per parameter is not evidence that conditions
are unused. Both are gaps in the test suite.

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

## Documentation

Sphinx sources live in `documentation/source/`; the published `docs/` folder is
build output (~1181 tracked files) served by GitHub Pages.

Regenerate `documentation/source/wisp_options.rst` **first** by running
`documentation/source/document_options.py` — it is gitignored and is built from
a throwaway project, so it needs the current code installed. Skipping it does
not fail the build: you get a "toctree contains reference to nonexisting
document" warning lost among ~56 pre-existing ones, no options page, and every
`:option:` link silently unresolved. Then `rm -rf docs/` and
`sphinx-build -b html documentation/source docs`.

The wipe matters: commit 07817f54 deleted the sources for eleven pages whose
built HTML is still tracked, and a plain rebuild does not remove them — they
stay live, unreachable from the nav but reachable by URL and search. Wiping is
safe: `sphinx.ext.githubpages` recreates `.nojekyll`, which is essential, since
without it Pages runs Jekyll and ignores `_static/`, `_sources/` and `_images/`.
Keep the rebuild as its own commit; it rewrites every page.

## Issue Tracking

AutoWISP and AstroWISP work is tracked in the **SuperPhot (`SUP`)** Jira project
on `https://kaloyanpenev.atlassian.net` (cloud id
`02881f8d-fafe-49ff-87bb-4e99c57ee4d1`). The site has thirteen projects and
several plausible-sounding names — `APH` "Amateur Photometry", `TP` "TESS
Photometry" — but none of them hold this repo's issues; `SUP` does. File new
issues there as **Task** unless there is a reason to pick another type.

**Every issue must have a parent.** Stories, tasks and bugs are parented to an
epic; sub-tasks are parented to the story, task or bug they belong to. Set the
parent when the issue is created rather than filing it loose and fixing it
afterwards. The epics available in `SUP`:

| Epic | Scope |
| --- | --- |
| `SUP-1` | Image Calibration |
| `SUP-2` | Astrometry |
| `SUP-3` | Photometry |
| `SUP-4` | Magnitude Fitting |
| `SUP-5` | Generating Raw Lightcurves |
| `SUP-6` | Lightcurve Post-processing (EPD/TFA) |
| `SUP-37` | Automated processing — pipeline engine and cross-cutting internals |
| `SUP-49` | AstroWISP/AutoWISP under Windows |
| `SUP-128` | User Interface — everything BUI-facing |
| `SUP-132` | Package for Major Operating Systems — packaging and installation |
| `SUP-160` | Documentation |
| `SUP-167` | Tests |
| `SUP-200` | Non-development related tasks |

Note that `SUP-37` and `SUP-128` divide by *surface*, not by subject: the
BUI-facing half of a concern goes under `SUP-128` and its engine half under
`SUP-37`, so one body of work can legitimately span both.

Its two Done-category statuses do not mean what Jira's stock descriptions
suggest, and the difference matters:

- **Resolved** (transition id `31`) — the work was actually completed. This is
  what "mark it done" means here. The workflow sets resolution `Done` on its
  own; the transition needs no extra fields.
- **Closed** (transition id `51`) — the issue **will not** be done: dropped,
  obsolete, won't fix. It is not a "finished and verified" state.

Jira describes Resolved as provisional ("awaiting verification by reporter") and
Closed as final, which reads backwards for this project and would lead you to
Close completed work. Transition finished work to Resolved; reserve Closed for
work being abandoned.

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

## Working Conventions

- **Make changes with the Edit tool, not scripts.** Piping python/sed through
  Bash hides the before/after that makes a change reviewable. A script is
  defensible only for a bulk *mechanical* transform — re-indenting a block after
  wrapping it, or the same substitution across 10+ files — and even then, show
  `git diff -w` afterwards as the reviewable artifact and say why.

- **Don't commit unless asked.** Leave work uncommitted. It gets several rounds
  of edits and corrections on top, and committing each intermediate state makes
  noise that then has to be squashed. Wait for an explicit "commit".

- **Plans are drafted as a file, then filed in Jira — never committed.** While
  a design is still being argued over, keep it as an uncommitted Markdown plan
  in the working tree: rereading and editing a local file beats reloading a
  Jira page for every change. Once it settles and implementation is about to
  start, create it as a SUP story with the implementation stages as sub-tasks
  (see *Issue Tracking*), transition the ones in scope to Selected For
  Development, leave deferred ideas Open, and delete the file. Jira then holds
  the design, the reasoning and the rejected alternatives, and nothing has to
  be committed and later removed.

- **Don't revert incidental Black reformatting.** The repo is not uniformly
  Black-clean at 80 columns, so a directory-wide run touches unrelated files.
  Split the commits instead — functional change in one, formatting-only files in
  another. For a file carrying both, leave the formatting in with the fix.

- **Durable rules about this project belong in this file**, or in
  `.claude/skills/`, not in per-machine assistant memory. This repo is worked on
  from more than one machine; anything recorded only in memory applies on one of
  them and silently diverges from the other.
