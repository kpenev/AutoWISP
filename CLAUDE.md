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
(both are also in `.coveragerc`'s omit list). `tests/generate_catalog_test_data.py`,
`tests/update_hdf5_contents.py` and `tests/compare_h5.py` are test-data tooling
rather than suite members, and are excluded on purpose — there is a comment saying so at the top of
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

*Checking working-tree code without reinstalling:* a scratch script run with
`PYTHONPATH=<repo>` imports the tree rather than site-packages, which is how
a function can be exercised straight after editing it while the installed
copy stays the non-editable one the BUI needs. Without it the import comes
from site-packages and silently tests the last install — an `ImportError`
for something just added is the friendly version of that; a stale function
that merely passes is the other one. A browser check still needs the
install.

## Running Tests

Tests use Python `unittest` (pytest-compatible). They download test data automatically and run pipeline steps sequentially:

```bash
conda activate autowisp          # or whatever the environment is called
pip install .                    # the suite runs the *installed* package
python -m autowisp.tests <failed_test_dir> -v    # Run all tests
python -m autowisp.tests failed_test -v           # CI convention

# Run a single test class
python -m autowisp.tests failed_test -v -k TestCalibrate
```

**Activate the environment; do not reach into it.** Calling
`~/miniforge3/envs/autowisp/bin/python -m autowisp.tests` runs the right
interpreter but leaves the environment's `bin` off `PATH`, so every test that
shells out to a `wisp-*` console script dies with `FileNotFoundError:
'wisp-calibrate'`. That is 21 errors that look like the pipeline is broken and
are not, and they cost a full 16-minute run to find out. Non-interactive
shells need `source ~/miniforge3/etc/profile.d/conda.sh` before
`conda activate`.

**Import the checkout, not site-packages.** `TestUpgradeFromRelease` migrates
each tagged release's schema forward, which needs the repository's history, so
it asks `git -C os.path.dirname(__file__) rev-parse --show-toplevel` whether
the test file it is running from is inside a checkout. An installed copy is
not, so it skips — correctly, there being no history to export, but three
migration tests then vanish into the skip count. Run from the repository root,
or set `PYTHONPATH=<repo>`; the working directory matters only because
`python -m` puts it on `sys.path`. Installing first is what keeps this honest:
the tree and site-packages then agree, so the in-process tests and the `wisp-*`
subprocesses exercise the same code. A skip count above 1 is the tell — the one
expected skip is the server-only backup check, which the MariaDB jobs run.

**Select tests with `-k`, rather than running a module directly.** The full
suite is too slow to run after every edit, but `python -m autowisp.tests
failed_test -v -k TestCalibrate` still imports `__main__`, which is where the
suite collects from and the first thing CI trips over. `python -m unittest
autowisp.tests.test_x` costs about the same and skips that entirely, so it
passes happily while the suite cannot even start.

**Leave `TestFullPipeline` out while iterating.** It runs every step through
the engine and takes far longer than the rest, while the per-step tests
exercise the same step code on the same data. It belongs to the full-suite
run before a merge.

**Run the whole suite, or dispatch CI, before merging into master.** CI is
`workflow_dispatch` only, so a push runs nothing. On a feature branch the
tests covering the change are enough before a push; the full suite, slow as
it is, gates the merge.

**Run the full suite locally under Python 3.11 as well as 3.14 before
dispatching the grid** (`conda activate wisp-py311`, then install and run as
above). Run the two at the same time, each with its own failed-test
directory. Syntax newer than the 3.11 floor passes everything a 3.14
environment runs -- the suite, pylint and Black alike -- and fails only on
the grid, where it breaks every import: `except A, B:` without parentheses
(PEP 758, 3.14 only) did exactly that, costing a full grid run.

The `<failed_test_dir>` argument is **required** — it's where artifacts from failed tests are preserved for debugging. Tests run in a temporary directory, copy test data there, and clean up on success.

**The test data comes from Zenodo**, downloaded afresh on every run from the
record named in `tests/get_test_data.py`. `--test-data <zip or directory>`
runs against a local copy instead, which is how a regenerated bundle is
checked before it is published. Zenodo records are permanent: they cannot be
unpublished or replaced. So batch bundle changes, publish one new version
once they are final, and then point `get_test_data.py` at it.

**Pass `--test-data` while iterating, whichever tests are selected.** The
bundle is a 226 MB zip, and the download happens at start-up, before `-k` is
looked at, so even tests that never open it, such as the migration ones, wait
several minutes for it. Keep an unzipped copy and point every run at it. A
run that prints nothing for minutes is downloading, not testing.

**Regenerating expected outputs cascades.** Each step test reads its inputs
from the bundle and compares its outputs with it, and one step's outputs are
the next step's inputs: the DR fits feed `TestCreateLightcurves`, whose
lightcurves feed EPD, then TFA, then the statistics. After a change to a
step's output, go down the chain one step at a time:

1. Run the step's test against the bundle as updated so far (`--test-data`).
   It fails and keeps its output in `<failed_test_dir>`.
2. Check that only the step's own groups differ from the bundle, with
   `tests/compare_h5.py <failed output dir> <bundle dir>`. The test's own
   comparison stops at the first mismatch.
3. Copy those groups into the bundle with `tests/update_hdf5_contents.py`.
4. Rerun the test to see it pass, then move to the next step.

Differences outside the step's groups that lie within the test tolerance,
such as floating-point noise in `SkyPosition`, are left alone.

Test classes (in order of pipeline dependency): `TestCalibrate` → `TestStackToMaster` → `TestFindStars` → `TestSolveAstrometry` → `TestFitStarShape` → `TestMeasureAperturePhotometry` → `TestFitSourceExtractedPSFMap` → `TestFitMagnitudes` → `TestCreateLightcurves` → `TestEPD` → `TestTFA` → `TestDetrendingStat`

Base test class: `AutoWISPTestCase` (extends `astrowisp.tests.utilities.FloatTestCase`). Use `self.run_step(command)` to invoke pipeline CLI commands within tests.

**A new test class must be imported into `autowisp/tests/__main__.py`**, which
is where the suite collects from — an unimported class is never run and nothing
says so. `test_suite_registration` compares what the runner reaches with what
the modules define and fails naming whatever is unreachable, so this is caught
rather than remembered. Two test classes may not share a name across modules:
the imports land in one namespace, where the second silently replaces the
first.

**Writing tests:**

- *Mix the cases in one population.* Rather than one test per rule over data
  built to trip only that rule, give the fixture entries that fail different
  rules side by side and assert that exactly the right ones survive: each is
  then a control for the others.
- *Include the rare case that would go unnoticed*, e.g. a frame bound to
  different photometric references in two channels. It is what breaks long
  after the code was written, and what nobody thinks to try by hand.
- *Don't pin behaviour for states real use cannot reach*, e.g. an image
  without astrometry reaching magnitude fitting. Asserting what happens there
  requires behaviour that is better left undefined.
- *Compare every output a step writes with the bundle.* A step test that
  checks only the DR or lightcurve groups leaves the step's other outputs
  (masters, statistics files) unchecked, and the bundle's copies go stale
  without anyone noticing: magfit's masters sat in a format from before
  SUP-532 until SUP-553 added the comparison. Passing tests say nothing
  about an output no test compares.
- *Keep a fixture's data private* (`_` prefix on class attributes the tests
  read), and keep data only one method uses local to it rather than global.
- *Never let one name mean two things* in a fixture: references named `A` and
  `B` next to a channel `B` make every assertion ambiguous to read. Use
  `ref1`, `ref2`, etc.

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

**Changing what project creation writes needs a revision for existing
projects.** Creating a project fills its database with definitions: the
steps, their parameters with their help and defaults, the dependencies and
processing sequence, the master types, and the layout of the HDF5 products
(`initialize_database.py`, `initialize_*_structure.py`, and the steps'
command-line parsers, whose options become the parameters). A project
created earlier keeps what it was given, so adding, removing or rewording
any of these is not done until a revision in `database/migrations/versions/`
does the same to existing projects.

- *The test that enforces it* is
  `test_every_release_ends_up_holding_what_a_new_project_does`
  (`tests/test_upgrade_from_release.py`): each released version creates a
  project with its own code, and after migration every table must hold what
  a new project's does. It compares all tables, so nothing needs registering
  for a new one.
- *A changed default is the one exception*, because the value a project
  stores is its own. Either migrate it, or decide that existing projects keep
  theirs and add the parameter to `stored_values_kept` in that test, with the
  reason.
- *A revision that changes rows carries its own copy* of the names, help
  texts and row definitions it writes, reflects tables from the database
  instead of importing the models, looks before each insert so that it can
  be run twice, and has a downgrade. `0012`–`0014` are the examples.
- *List the revision in `versions/meson.build`.* `TestRevisionChain` checks
  that from a checkout; an installed package that lacks the newest revision
  looks valid and stamps projects at the wrong head.
- *An unreleased revision may be renamed or rewritten freely* while its
  branch is in development; avoiding the break is not worth extra code or
  commits. A project already migrated by the earlier version is stamped
  with an id the code no longer has, and fails to open with "Can't locate
  revision identified by ...". Re-stamp it to the previous revision
  (`UPDATE alembic_version SET version_num = '<previous>'` in its
  `autowisp.db`) and reopen it: revisions look before they change anything,
  so the new version runs over whatever the old one did.

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

*Don't pin presentation even there.* Labels, headings, tooltips and other text
a user reads are checked by looking at the page, even when a pure function
builds them: what matters is whether they confuse, and fixing one and
redesigning it are the same edit, so a test only freezes the current wording.
Test what decides which data is read, counted or drawn.

**Configuration view redesign pending.** The orgchart decision tree
(`configuration/config_tree.html` +
`static/configuration/js/autowisp.config.tree.js`) is slated for a redesign.
Implement config-view features against the existing tree without gold-plating
its styling; raise the pending redesign before any substantial rework of its
presentation. Until then the tree is a standalone page rather than an
`lcars_app.html` one, so it shows no Django messages: one added while it is in
use (e.g. the exclusion-rule warnings `save_config` gives) appears on the next
LCARS page instead.

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

**The build is warning-free, and is meant to stay that way.** Any warning
`make html` prints was introduced by the change in hand — there is no
background noise left to lose it in, so read the output rather than the exit
code, which is 0 either way.

**Build with `make html` from `documentation/`** — never by calling
`sphinx-build` yourself. The Makefile does five things in order that a bare
build does not, each guarding a failure that is silent rather than loud:

- runs `document_options.py`, which regenerates the gitignored
  `wisp_options.rst` from a throwaway project, so it needs the current code
  installed. Skipping it does not fail the build: you get a "toctree contains
  reference to nonexisting document" warning, no options page, and every
  `:option:` link silently unresolved.
- wipes `source/implementation` before `sphinx-apidoc`, which overwrites the
  pages it generates but never deletes the ones whose module is gone.
- passes `sphinx-apidoc` the exclusions in `APIDOCSKIP`: the modules that are
  deliberately not installed, which would otherwise get a page autodoc cannot
  fill, and the `data_model` submodules, whose classes the package page
  already documents by way of its `__all__`.
- wipes `build/`, because an incremental build only writes the pages it
  re-reads while the whole of `build/html` replaces `docs/`, so anything
  skipped goes missing from the published site.
- wipes `docs/` before moving the new build in. That matters: commit 07817f54
  deleted the sources for eleven pages whose built HTML is still tracked, and
  a plain rebuild does not remove them — they stay live, unreachable from the
  nav but reachable by URL and search. Wiping is safe: `sphinx.ext.githubpages`
  recreates `.nojekyll`, which is essential, since without it Pages runs Jekyll
  and ignores `_static/`, `_sources/` and `_images/`.

Keep the rebuild as its own commit; it rewrites every page.

**Rebuild once per branch, when its development is finished** — just before
merging into master, not after each story or sub-task. Every rebuild is a
huge commit, so rebuilding along the way scatters several of them through
the branch's history and buries the real changes. In Jira, the rebuild is a
single sub-task of the branch's main story, not a sub-task of each story
that changes documented code. Editing the documentation *sources*
(`documentation/source/`, docstrings) alongside the code is fine. It's the
generated `docs/` that waits.

The toolchain is an extra, `pip install .[docs]`, and belongs in **the same
environment as autowisp** — `sphinx-build` imports every module it documents,
so a system-wide Sphinx whose interpreter cannot import `autowisp` produces a
full set of API pages with every `automodule` silently empty. Graphviz (`dot`)
must be on PATH for the inheritance diagrams.

`conf.py` calls `django.setup()`, because the browser interface is Django
applications and importing one of its modules needs the application registry.
That works only while the applications name each other by their full import
path; a short name reintroduces the empty pages, and worse, a second module
identity for a model Django has already registered.

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

**Check for `do-first` issues before starting any new work.** Query
`project = SUP AND labels = do-first AND statusCategory != Done` and take
each issue it returns before anything else, unless its description says it
waits for something that has not happened yet (e.g. a branch being merged).
This is how work deferred to "the next development" is remembered across
machines: label the issue, and it surfaces here.

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

- **One edit per tool call, never a batch of them.** Each edit is reviewed as
  it is proposed, and rejecting one must stop everything after it; edits sent
  together are all presented anyway, including those built on the rejected
  one.

- **Say which function an edit touches, and where, before proposing it.** The
  edit preview numbers lines relative to the snippet, so on its own it does not
  show where in the file the change lands — give `file:line` and the function
  name, as they are *now* (an earlier edit in the same file moves them).

- **Don't commit unless asked.** Leave work uncommitted. It gets several rounds
  of edits and corrections on top, and committing each intermediate state makes
  noise that then has to be squashed. Wait for an explicit "commit".

- **Push before updating Jira about a commit.** A comment written before the
  push can only say the work is not on the remote yet, and is stale as soon
  as it is. Push first, then comment and transition the issue.

- **A commit need not be a working version on its own.** Committing code that
  a later commit fixes is fine. And when finished work is split into a series
  of commits at once, to make it readable, don't build or test the
  intermediate ones: the tests of the final state already cover them. Keep
  running tests *during* development, though -- a break found soon after the
  edit that caused it is far easier to pinpoint.

- **Plans are drafted as a file, then filed in Jira — never committed.** While
  a design is still being argued over, keep it as an uncommitted Markdown plan
  in the working tree: rereading and editing a local file beats reloading a
  Jira page for every change. Once it settles and implementation is about to
  start, create it as a SUP story with the implementation stages as sub-tasks
  (see *Issue Tracking*), transition the ones in scope to Selected For
  Development, leave deferred ideas Open, and delete the file. Jira then holds
  the design, the reasoning and the rejected alternatives, and nothing has to
  be committed and later removed.

- **Run Black itself on the files you touch** (`black -l 80 <files>`), not
  `--check` or `--diff` followed by applying its changes by hand. Black never
  changes behaviour and files are meant to be Black-clean, so its output needs
  no review hunk by hunk; `git diff` shows what it did. It is an exception to
  *Make changes with the Edit tool*. The one thing to fix up afterwards is a
  trailing `# pylint:` comment it has moved off the line it covered.

- **Don't revert incidental Black reformatting.** The repo is not uniformly
  Black-clean at 80 columns, so a directory-wide run touches unrelated files.
  Split the commits instead — functional change in one, formatting-only files in
  another. For a file carrying both, leave the formatting in with the fix.

  This holds for lines the change never touched. Black reformatting old code
  in a file being edited means a previous commit missed it, so the fix is to
  let Black have it, not to preserve the old spelling — reverting it only
  leaves the next person the same decision.

- **Durable rules about this project belong in this file**, or in
  `.claude/skills/`, not in per-machine assistant memory. This repo is worked on
  from more than one machine; anything recorded only in memory applies on one of
  them and silently diverges from the other.
