# AutoWISP Error Handling Plan

## Motivation

AutoWISP currently raises a mix of stdlib exceptions and a handful of
domain-specific ones (`autowisp/pipeline_exceptions.py`, plus ad-hoc
classes scattered through `database/`, `astrometry/`, etc.). When
something fails:

- **Users** rarely get a clear statement of *what* failed, *where* in
  the pipeline, *which artifact* (image, DR file, lightcurve, master)
  is implicated, or *what to do next*.
- **Developers** rarely get the full provenance needed to reproduce a
  failure: which host the run was on, which Python process, which
  multiprocessing worker, when the run started, what configuration was
  in effect.

The goal of this plan is to make errors reliably useful for both
audiences, with the same exception object carrying enough context to
power CLI messages, BUI surfacing, and post-mortem debugging.

### Why persist errors in a database table

The eventual `Error` table (phase 4) is not just "logging done twice."
It follows from what this pipeline already is:

- **Failures happen out of sight.** `run_pipeline.py` detaches into the
  background and redirects each process's stdout/stderr and logs to
  per-PID files (`{task}_{now}_{pid}.outerr`) under the app data dir,
  and work fans out across `multiprocessing.Pool` workers that each get
  their *own* redirected log. A failure is therefore buried in one of N
  files keyed by PID and timestamp, with nobody watching a terminal.
  The BUI — the intended primary UI — cannot reasonably scrape those to
  answer "what failed for this image?"

- **The DB is already the coordination layer.** Run state already lives
  in SQLite (`PipelineRun`, `ProcessedImages`,
  `ImageProcessingProgress`, master selection, photref binding). Errors
  *are* run state; storing them anywhere else creates a second source
  of truth that can drift from `ProcessedImages.status`.

- **Errors are relational, and the relations are the point.** An error
  is *about* an artifact and *belongs to* a run. A table carries real
  foreign keys to `PipelineRun` + `Image` / `DRFile` / `Lightcurve` /
  `MasterFile`, which is exactly what lets the BUI answer "what failed
  for this lightcurve?", a developer filter "every failure in run 88 by
  step", and either ask "one-off, or did this `SolveAstrometryError`
  hit 200 frames?" (`GROUP BY`). A flat log answers none of these
  without re-parsing.

- **One source of truth, many projections.** CLI stderr, BUI views, and
  the post-mortem detail are all projections of the *same* exception
  (see below). A persisted row *is* that object — its structured fields
  survive as columns + a JSON blob, instead of being flattened into a
  string the instant they are logged.

- **It outlives the process that raised it.** The raising worker is
  gone by the time anyone looks (possibly segfaulted); the snapshotting
  in phase 1 (a `FrozenRow` of the run, plus `code_version`) exists so
  the row remains reproducible after the session — and the rotated log
  — is gone.

This does **not** replace logging: logs keep the full traceback firehose
for live debugging. `ProcessedImages.status` likewise still records
pass/fail. The `Error` table is the curated, queryable, human- and
developer-readable record that says *what* and *why*, not just *that*.

## Audiences and what they need

| Audience  | What they need from an error |
| --------- | --------------------------- |
| End user (BUI) | A short human-readable description, the affected artifact, suggested remediation if available, a link to the structured detail for support. |
| End user (CLI / `wisp-*`) | Same as BUI but rendered to stderr. |
| Developer | Component, step, sub-process ID, host, PID, timestamps, related file paths, original traceback, configuration snapshot. |

Both views are projections of the *same* exception object — there is
exactly one source of truth.

## Phase overview

1. ✅ **[Exception hierarchy.](#phase-1--exception-hierarchy)**
2. ✅ [Context-collection mechanism](#phase-2--context-collection)
   (decorators / context managers) so steps and the pipeline driver
   automatically populate the exception fields without each call site
   having to remember.
3. ✅ [Propagation across multiprocessing boundaries](#phase-3--propagation-across-process-boundaries)
   (worker stamps subprocess ID + host before re-raising; parent
   re-raises wrapped, preserving the original `__cause__`).
4. ✅ [Persistence layer](#phase-4--persistence): an `Error`
   table linked to `PipelineRun` + `Image` / `DRFile` / `Lightcurve` /
   `MasterFile` rows, plus a JSON blob for the structured ``details``.
5. ✅ [User-facing rendering](#phase-5--user-facing-rendering-bui--cli):
   BUI views and CLI formatter, both reading from the structured record
   (other channels such as an email/Slack notifier come later).
6. ✅ **[Crash-report bundler](#phase-6--crash-report-bundler-implemented)**:
   on demand, collect everything needed to debug a failure into a single
   zip the user can send to the maintainers — the `Error` row(s) +
   sidecar(s), the relevant per-process logs/stdout-stderr, the
   configuration snapshot, a scrubbed database copy, the `code_version`,
   and environment/provenance — with a scrubbing pass for credentials.
7. ✅ [Migrate existing call sites](#phase-7--call-site-migration) to
   raise the new exception types, apply the CLI error boundary to each
   step `main()`, and fold the legacy ad-hoc classes into the hierarchy.
8. ⏳ [Deferred-site migration + BUI-specific raises](#phase-8--deferred-site-migration--bui-specific-raises):
   retype the worthwhile subset of the Phase 7 "deferred" raise sites,
   and introduce a few new exceptions raised specifically so the BUI can
   handle them. *(section pending)*
9. ⏳ [Silent-worker-death diagnostics + crash-report completeness](#phase-9--silent-worker-death-diagnostics--crash-report-completeness):
   close the gaps a real `WorkerCrashedError` crash report exposed — the
   step link that log-collection depends on, lightcurve-step log
   selection, the specific culprit input, a native/OS-level cause for the
   death, and run-time (not report-time) provenance.

Phases 1–7 are implemented and have their own sections below; phases 8–9
get their sections when we start them (phase 9's is written below,
motivated by a real crash report).

## Phase 1 — Exception hierarchy

### Component classification

Every error belongs to exactly one AutoWISP **component**:

- **`step`** — failure inside a processing step (calibrate,
  find_stars, solve_astrometry, fit_star_shape,
  measure_aperture_photometry, fit_source_extracted_psf_map,
  fit_magnitudes, create_lightcurves, epd, tfa,
  detrending_stat). Step errors carry the step name so the
  reporter can match it back to a `Step` row.
- **`pipeline`** — failure in the orchestration layer:
  `run_pipeline.py`, `ImageProcessingManager`,
  `LightCurveProcessingManager`, the database interface, master
  selection, photref binding, etc. Anything that is not "the
  algorithm" but is "running the algorithm."
- **`bui`** — failure in the Django views, forms, or templates of
  `autowisp/browser_interface/`.

The component is recorded as a string enum on the base class. Tests
verify that every concrete exception sets it (no `None`).

### Base class

```python
# autowisp/exceptions.py
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

from autowisp.database.frozen_row import FrozenRow


class Component(str, Enum):
    STEP = "step"
    PIPELINE = "pipeline"
    BUI = "bui"


class FileKind(str, Enum):
    RAW_IMAGE = "raw_image"
    CALIBRATED_IMAGE = "calibrated_image"
    MASTER_BIAS = "master_bias"
    MASTER_DARK = "master_dark"
    MASTER_FLAT = "master_flat"
    MASTER_PHOTREF = "master_photref"
    DR_FILE = "dr_file"
    LIGHTCURVE = "lightcurve"
    CATALOG = "catalog"
    CONFIG = "config"
    OUTPUT = "output"
    OTHER = "other"


@dataclass(frozen=True)
class RelatedFile:
    """A file the error is about (input, output, or intermediate).

    Attributes:
        kind(FileKind):    What sort of file this is.

        path(Path):    Location of the file.

        role(str):    How the file relates to the failure, e.g.
            ``"input"`` / ``"intermediate"`` / ``"expected_output"``.
    """

    kind: FileKind
    path: Path
    role: str = ""
```

`FrozenRow` is a dependency-free dataclass, so it gets its own tiny
module with **no SQLAlchemy import in any form** — which is exactly what
lets `exceptions.py` import it directly (above) without dragging the ORM
into the exception hierarchy:

```python
# autowisp/database/frozen_row.py
from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenRow:
    """Immutable, picklable snapshot of an ORM row's column values.

    Holds the column values of a SQLAlchemy row detached from any
    session, so it stays usable after the session that produced it is
    closed and survives pickling across process/host boundaries (where a
    live ORM instance would not). Column values are reached by attribute
    (``snapshot.host``) or via :attr:`columns`.

    Built from a live instance with :func:`snapshot_row` (parent side,
    where the row exists) or directly from a dict (a worker, which only
    has primitives threaded through its config).

    This is a *general* utility — not specific to errors. Any place that
    needs a durable, picklable copy of a row (related-artifact context,
    provenance, caching a row past its session) can use it.

    Attributes:
        table(str):    Name of the source table, for display/debugging.

        columns(dict):    ``{column_key: value}`` for every mapped
            column captured. Treated as read-only.
    """

    table: str
    columns: dict

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails, so ``table``
        # and ``columns`` are unaffected.
        try:
            return self.columns[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
```

`snapshot_row` is the only piece that touches SQLAlchemy, so it lives in
`autowisp/database/interface.py` — the module that already manages the
session lifecycle and already imports `inspect as sa_inspect`.
Detaching a row's values so they survive past its session is exactly the
concern `interface.py` owns (it is the generalized form of the
`pipeline_run = pipeline_run.id` move `run_pipeline.main` already makes):

```python
# autowisp/database/interface.py  (sa_inspect already imported here)
from autowisp.database.frozen_row import FrozenRow


def snapshot_row(orm_obj, *, exclude=()) -> FrozenRow:
    """Freeze all mapped columns of a live ORM instance into a FrozenRow.

    Must be called while ``orm_obj`` is still attached/loaded (i.e.
    inside the ``start_db_session()`` block that produced it).

    Args:
        orm_obj:    A SQLAlchemy ORM instance.

        exclude(Iterable[str]):    Column keys to omit (e.g. large or
            sensitive columns).

    Returns:
        FrozenRow:    Snapshot of the instance's column values.
    """

    mapper = sa_inspect(orm_obj).mapper
    return FrozenRow(
        table=mapper.local_table.name,
        columns={
            attr.key: getattr(orm_obj, attr.key)
            for attr in mapper.column_attrs
            if attr.key not in exclude
        },
    )
```

Splitting them this way keeps `FrozenRow` importable with zero
dependencies, so `exceptions.py` imports it directly — no `TYPE_CHECKING`
dance and no SQLAlchemy pulled into the exception hierarchy — while
`snapshot_row`, the only SQLAlchemy-touching piece, sits with the session
machinery it generalizes (the decoupling point in "Why a snapshot, not
the ORM object?" below).

The pipeline-run context an exception carries is then **just a
``FrozenRow`` of the ``PipelineRun`` row** — no hand-maintained field
list, so it tracks the table definition automatically (and picks up the
``code_version`` column added in phase 4 with zero changes here):

```python
# Parent side (run_pipeline.main), inside the session:
pipeline_run_context = snapshot_row(pipeline_run)   # FrozenRow

# Worker side, no ORM object — build from config primitives:
pipeline_run_context = FrozenRow(
    "pipeline_run",
    {"id": config["pipeline_run_id"],
     "host": config.get("host") or socket.gethostname(),
     "started": config.get("pipeline_started"),
     "code_version": config.get("code_version") or get_code_version_str()},
)


def _rebuild_autowisp_error(cls, args, state):
    """Reconstruct an :class:`AutoWISPError` subclass for unpickling.

    Bypasses ``__init__`` (so keyword-only arguments on subclasses do not
    block unpickling) and restores both ``BaseException.args`` and the
    instance ``__dict__`` directly. ``args`` *must* be restored
    separately: the message lives in ``BaseException.args``, which is
    C-level storage **outside** ``__dict__``, so restoring only
    ``__dict__`` would silently drop the message (``str(exc)`` becomes
    empty). See :meth:`AutoWISPError.__reduce__`.

    Args:
        cls(type):    The concrete exception class to rebuild.

        args(tuple):    The original ``self.args`` (carrying the
            message).

        state(dict):    The instance ``__dict__`` captured at pickle
            time.

    Returns:
        AutoWISPError:    The reconstructed exception.
    """

    obj = cls.__new__(cls)
    obj.args = args
    obj.__dict__.update(state)
    return obj


class AutoWISPError(Exception):
    """Base class for every AutoWISP-raised exception.

    Every concrete subclass selects a :class:`Component`.

    Attributes:
        component(Component):    Set on each subclass; verified by tests.

        related_files(tuple):    The :class:`RelatedFile` entries this
            error is about.

        pipeline_run(FrozenRow or None):    Snapshot of the
            ``PipelineRun`` row (see :class:`FrozenRow`), set by the
            pipeline driver when it wraps a step's exception, or by
            ``wisp-*`` entry points. ``None`` for runs with no DB row.

        crashed(datetime or None):    When the failure surfaced, filled
            in by the top-level handler. (This is error-level, not a
            column of ``PipelineRun``, so it lives here rather than in
            the row snapshot.)

        subprocess_id(int or None):    PID of the multiprocessing worker
            that raised, when the exception travelled out of a Pool;
            ``None`` for errors raised in the main process.

        user_message(str):    Short, free of jargon, suitable for the
            BUI.

        details(dict):    Arbitrary key/value pairs giving extra context
            about the failure (e.g. shape mismatches, expected/actual
            values, parsed config). Useful to both users and developers.
    """

    component: Component  # set on each subclass; verified by tests

    def __init__(
        self,
        message: str,
        *,
        related_files: Sequence[RelatedFile] = (),
        pipeline_run: Optional[FrozenRow] = None,
        crashed: Optional[datetime] = None,
        subprocess_id: Optional[int] = None,
        user_message: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Store the context attributes (see class ``Attributes``)."""

        super().__init__(message)
        self.related_files = tuple(related_files)
        self.pipeline_run = pipeline_run
        self.crashed = crashed
        self.subprocess_id = subprocess_id
        self.user_message = user_message or message
        self.details = dict(details or {})

    def __reduce__(self):
        """Pickle by restoring ``args`` + ``__dict__`` rather than
        re-running ``__init__``.

        Subclasses (e.g. :class:`StepError`) take keyword-only arguments
        and carry context attributes that are not part of ``self.args``,
        so the default exception unpickler — which calls
        ``cls(*self.args)`` — would drop those fields (and, for any
        required kwarg, raise ``TypeError``). Reconstructing through
        ``__new__`` while restoring ``args`` *and* ``__dict__`` keeps
        every field intact — including the message, which lives in
        ``args``, not ``__dict__`` — and lets the exception travel back
        out of a multiprocessing Pool (see phase 3).

        Returns:
            tuple:    ``(callable, args)`` per the pickle protocol.
        """

        return (
            _rebuild_autowisp_error,
            (type(self), self.args, self.__dict__.copy()),
        )

    def stamp_subprocess(self) -> None:
        """Record the current PID as the raising sub-process.

        Called by the worker before re-raising out of a multiprocessing
        Pool. Idempotent: a value already set in a deeper worker wins.

        Returns:
            None
        """

        if self.subprocess_id is None:
            self.subprocess_id = os.getpid()

    def with_pipeline_run(
        self, run: Optional[FrozenRow], *, crashed: Optional[datetime] = None
    ) -> "AutoWISPError":
        """Attach a :class:`FrozenRow` snapshot of the ``PipelineRun``.

        Args:
            run(FrozenRow or None):    Row snapshot to attach. Built by
                ``snapshot_row`` (parent) or from config primitives
                (worker), so host/started are already populated; nothing
                is reconstructed here.

            crashed(datetime or None):    Failure time; defaults to now
                if not already set on the exception.

        Returns:
            AutoWISPError:    ``self``, so it can be used inline before
                re-raising.
        """

        self.pipeline_run = run
        self.crashed = self.crashed or crashed or datetime.utcnow()
        return self
```

### Why a snapshot, not the ORM object?

The exception carries a `FrozenRow` snapshot rather than the live
`PipelineRun` instance because the ORM object is **session-bound and not
durable**, while the exception must stay useful long after, and far
from, the session that produced it:

- **It does not outlive its session.** Everything goes through
  `start_db_session()` (scoped sessions, `NullPool`). Once the `with`
  block exits the instance is detached and touching an unloaded
  attribute raises `DetachedInstanceError`. An exception propagates
  through many such blocks and is persisted much later. Tellingly,
  `run_pipeline.main` already discards the object today
  (`pipeline_run = pipeline_run.id`) because it cannot hold it past the
  block — `snapshot_row` is that same move, keeping *all* the fields.
- **It must pickle across processes (phase 3).** An ORM instance drags
  a session reference and `InstanceState`; it does not pickle cleanly
  and could not be reattached in the parent. A `FrozenRow` of primitives
  pickles trivially.
- **Workers have no ORM object at all.** They rebuild the snapshot from
  config primitives (`ErrorContext.from_config`), never having queried
  the row.
- **Snapshot semantics + decoupling.** A frozen copy freezes the values
  as they were at failure (the row can't mutate or be deleted under it),
  and keeps `exceptions.py` and the deep algorithmic code free of a hard
  SQLAlchemy dependency.

Generalising this to `FrozenRow` / `snapshot_row` (rather than a
bespoke, hand-listed `PipelineRunContext`) means the snapshot tracks the
table definition automatically, and the same utility is reusable for any
other row we need to detach and carry (related artifacts, provenance).

### Component bases and subclasses

```python
class StepError(AutoWISPError):
    component = Component.STEP

    def __init__(
        self, message: str, *, step_name: Optional[str] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.step_name = step_name


class PipelineError(AutoWISPError):
    component = Component.PIPELINE


class BUIError(AutoWISPError):
    component = Component.BUI
```

`step_name` is **optional**, not required. A deep call site must be able
to `raise SolveAstrometryError("no WCS solution")` without repeating the
step name (the whole point of phase 2, which fills it from the ambient
context in `_stamp`), and the re-rooted legacy classes (below) are still
raised as `raise BadImageError("...")` with only a message. A required
`step_name` would break both. It is therefore `None` at raise time and
stamped in later.

Concrete step-level exceptions are defined per pipeline stage so
catch sites can be specific without having to string-match step names:

```python
class CalibrationError(StepError): ...
class StackToMasterError(StepError): ...
class FindStarsError(StepError): ...
class SolveAstrometryError(StepError): ...
class FitStarShapeError(StepError): ...
class MeasurePhotometryError(StepError): ...
class FitPSFMapError(StepError): ...
class FitMagnitudesError(StepError): ...
class CreateLightCurvesError(StepError): ...
class EPDError(StepError): ...
class TFAError(StepError): ...
class DetrendingStatError(StepError): ...
```

Pipeline-level exceptions:

```python
class ConfigurationError(PipelineError): ...
class DatabaseError(PipelineError): ...
class ResourceError(PipelineError): ...        # disk / memory / CPU
class MasterSelectionError(PipelineError): ...
class PhotrefBindingError(PipelineError): ...
class DependencyResolutionError(PipelineError): ...
```

The worker-death wrapper is a `StepError`, not a pipeline error — the
failure is in the algorithm running inside a step (see phase 9, item 1);
the parent only synthesises and reports it:

```python
class WorkerCrashedError(StepError):
    """Re-raise wrapper used when a multiprocessing worker dies in a
    way that does not preserve the original exception (segfault,
    OOM-killer). A single generic StepError: the parent has the ambient
    step *name* but not the failing step's exception type."""
```

BUI-level exceptions:

```python
class ViewError(BUIError): ...
class FormValidationError(BUIError): ...
class ProjectStateError(BUIError): ...
```

### Migrating existing exceptions

Existing classes get re-rooted (signatures unchanged, so call sites
don't break in step 1):

| Existing                       | New parent                  |
| ------------------------------ | --------------------------- |
| `OutsideImageError`            | `StepError` (calibration)   |
| `ImageMismatchError`           | `StepError`                 |
| `BadImageError`                | `StepError`                 |
| `ConvergenceError`             | `StepError`                 |
| `HDF5LayoutError`              | `PipelineError`             |
| `NoMasterError`                | `MasterSelectionError`      |
| `ProcessingInProgress`         | `PipelineError`             |
| `MalformedResponse`            | `StepError` (solve_astrometry) |
| `RequestError`                 | `StepError` (solve_astrometry) |

The migration keeps the old names as aliases so external callers (and
tests) keep working until phase 7.

### How the run snapshot gets attached

This is mostly the job of phase 2, but the hierarchy needs to support
it now:

- `run_pipeline.py` snapshots the `PipelineRun` ORM row at startup with
  `snapshot_row(...)` (while the session is open) and stashes the
  resulting `FrozenRow` in the ambient `ErrorContext` (via
  `set_pipeline_run`) so any code on the call stack can grab it.
- A top-level `except AutoWISPError as exc:` in `run_pipeline.main`
  calls `exc.with_pipeline_run(get_error_context().pipeline_run)` if the
  field is unset (which also fills `crashed`), and re-raises.
- `wisp-*` CLI entry points do the same but with `pipeline_run` left
  `None` (no `PipelineRun` row).

### How subprocess_id gets attached

Also mostly phase 2/3, but constraints on the hierarchy:

- Workers use a top-level `try/except AutoWISPError` that calls
  `exc.stamp_subprocess()` and re-raises. The Pool's pickling carries
  the stamped value back to the parent.
- For non-`AutoWISPError` exceptions raised inside a worker, the
  worker wraps them in a `WorkerCrashedError(component=PIPELINE)`
  whose `__cause__` is the original, then stamps and re-raises.
- Nested pools (worker spawning sub-workers) leave the innermost
  stamp in place — `stamp_subprocess()` is idempotent for that
  reason.

### Tests

Phase 1 ships with:

- `tests/test_exception_hierarchy.py` that imports every concrete
  exception class, instantiates it with a minimal payload, and
  asserts:
  - `component` is a `Component`, not `None`.
  - `related_files`, `pipeline_run`, `crashed`, `subprocess_id`,
    `user_message`, `details` round-trip through
    pickling (needed for Pool propagation).
  - `stamp_subprocess()` is idempotent and sets `subprocess_id` to
    `os.getpid()` when previously `None`.
  - `with_pipeline_run()` attaches the `FrozenRow` and sets `crashed`.
  - `FrozenRow` / `snapshot_row` round-trip through pickling and expose
    columns by attribute.

## Open questions to revisit before phase 2

- ~~Should `RelatedFile` carry a hash or only a path?~~ **Resolved:
  path only.** Codebase identity is captured once per run as a git
  hash (`code_version` on the run snapshot, via
  `get_code_version_str()` in `miscellaneous.py`), which
  identifies the *entire* working tree — including its dirty state —
  without an extra per-file read. Per-file content hashes would cost a
  read each and still not capture the code that produced the file, so
  they are dropped.
- ~~`details` size vs. retention.~~ **Resolved in phases 4–5.**
  Capturing rich per-failure context (and, for BUI errors, the full
  request context) makes a single `details` payload large, and keeping
  every payload forever would balloon the SQLite file. Phases 4–5 settle
  this with three levers, each now specified:
  - **Keep the heavy part separate** — Phase 4 "The sidecar file": small
    queryable fields inline on the `Error` row, the large
    `details`/request snapshot spilled to a per-error sidecar so routine
    queries and the BUI list never touch it.
  - **Bounded persistence + user-driven cleanup** — Phase 4 "Retention &
    cleanup": `wisp-cleanup-errors --older-than` (and a BUI action), an
    orphan/dangling sweep, and an optional opportunistic prune at
    startup.
  - **BUI request snapshot to the sidecar** — Phase 5 "Request snapshot
    for BUI errors": request context (keys only, never values) captured
    into `details` so it reaches the sidecar rather than inline,
    precisely because it is the largest contributor to row size.

## Phase 2 — Context collection

### Problem

Phase 1 gave the exception object fields (`pipeline_run`,
`subprocess_id`, `step_name`, `component`, `related_files`). Phase 2 is
about *filling them in without burdening every raise site*. A call deep
inside `solve_astrometry` should be able to write
`raise SolveAstrometryError("no WCS solution")` and have the step name,
the pipeline-run snapshot, the worker PID, and the image/DR file it was
working on attached automatically by the time it surfaces.

The mechanism must satisfy three constraints that come straight from
the existing code:

1. **It rides on the bootstrap that already exists.**
   `setup_process_map()` (`autowisp/multiprocessing_util.py`) is already
   called once per process — in the main process from
   `ProcessingManager.__init__` / `_prepare_processing`, and again
   inside every worker — and it already receives a `config` dict
   carrying `processing_step`, `image_type`, `parent_pid`, and
   `project_home`. That dict is the one cross-process transport we
   already trust, so it is where context gets (re)established.

2. **Context does not cross process boundaries on its own.** The
   default start method is `spawn` on macOS/Windows, and even under
   `fork` the `Pool` sites re-run `setup_process`. So `contextvars`
   state set in the parent is *not* visible in a worker — each process
   must rebuild it from the config dict. Phase 2 therefore threads the
   pipeline-run identity (`pipeline_run_id`, `host`, `started`) through
   `config` alongside the keys already present.

3. **The wrapping must be opt-in at a few boundaries, not sprinkled.**
   We add the capture at: each step `main()`, the manager's per-image
   dispatch, and each `Pool` worker entry — and nowhere else.

### New module: `autowisp/error_context.py`

The ambient context is a single immutable `ErrorContext` bundle held in
one `contextvars.ContextVar`, plus helpers to read it, set it, and scope
a finer version of it.

Bundling the fields into one frozen dataclass (rather than three loose
`ContextVar`s) keeps related state cohesive — a deep raise site, the
capture decorators, and the worker hooks all read one consistent
snapshot in a single `get_error_context()` — while a *single* ContextVar
holding a *frozen* object preserves exactly the `contextvars` semantics
the scoping relies on: entering a scope `set`s a new value and exiting
`reset`s the token, so nothing is mutated in place and thread/asyncio
isolation is intact. (A mutable object mutated in `__enter__`/`__exit__`
would lose both.)

```python
import contextvars
import functools
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from traceback import format_exc
from typing import Optional, Sequence

from autowisp.database.frozen_row import FrozenRow
from autowisp.miscellaneous import get_code_version_str
from autowisp.exceptions import (
    AutoWISPError,
    Component,
    PipelineError,
    RelatedFile,
    StepError,
    ViewError,
    WorkerCrashedError,
    # ... plus the concrete StepError subclasses _wrap maps step names to.
)


@dataclass(frozen=True)
class ErrorContext:
    """Immutable bundle of the ambient context attached to errors.

    Held in a single :data:`_context` ContextVar so any code on the call
    stack can read a consistent snapshot without it being threaded
    through every call. Frozen so that establishing or scoping context
    replaces the ContextVar value (preserving its set/reset semantics and
    thread/async isolation) rather than mutating shared state.

    Attributes:
        pipeline_run(FrozenRow or None):    Snapshot of the
            ``PipelineRun`` row, or ``None`` for runs with no DB row.

        step_name(str or None):    The processing step currently
            executing.

        related_files(tuple):    The :class:`RelatedFile` entries in
            scope.

        in_worker(bool):    True inside a multiprocessing worker process;
            used by the phase-3 nesting guard.
    """

    pipeline_run: Optional[FrozenRow] = None
    step_name: Optional[str] = None
    related_files: tuple = ()
    in_worker: bool = False

    @classmethod
    def from_config(cls, config):
        """Rebuild context inside a freshly-started process from config.

        A worker has no ORM instance, so the pipeline-run snapshot is
        built from the primitives threaded through the per-process config
        dict rather than via ``snapshot_row``. Also picks up the step
        name already present in ``config``, and infers ``in_worker`` from
        ``parent_pid`` — the key the parent threads in for workers (and
        which is absent in the main process; see
        ``get_log_outerr_filenames``).

        Args:
            config(dict):    The per-process config dict, carrying the
                pipeline-run keys, ``processing_step``, and (for workers)
                ``parent_pid`` threaded through by the parent.

        Returns:
            ErrorContext:    The rebuilt context. ``pipeline_run`` is
                ``None`` when the keys are absent (e.g. a unit test
                calling a step directly).
        """

        pipeline_run = None
        if "pipeline_run_id" in config:
            pipeline_run = FrozenRow(
                "pipeline_run",
                {
                    "id": config["pipeline_run_id"],
                    "host": config.get("host") or socket.gethostname(),
                    "started": config.get("pipeline_started"),
                    "code_version": (
                        config.get("code_version") or get_code_version_str()
                    ),
                },
            )
        step_name = config.get("processing_step")
        if step_name in (None, "none", "init_processing"):
            step_name = None
        return cls(
            pipeline_run=pipeline_run,
            step_name=step_name,
            in_worker=bool(config.get("parent_pid")),
        )


_context: contextvars.ContextVar[ErrorContext] = contextvars.ContextVar(
    "autowisp_error_context", default=ErrorContext()
)


def get_error_context() -> ErrorContext:
    return _context.get()


def in_worker() -> bool:
    return _context.get().in_worker
```

#### Establishing pipeline-run context

Two entry points set it once, near the top of the call stack:

- **`run_pipeline.main`** already builds the `PipelineRun` row. While
  the session is still open, snapshot it and stash the snapshot (and
  thread its parts into the processing config so workers can rebuild
  it):

  ```python
  set_pipeline_run(snapshot_row(pipeline_run))
  ```

  (Before phase 4 adds the `code_version` column, attach it to the
  snapshot's `columns` here; afterwards `snapshot_row` captures it
  automatically.)

- **`wisp-*` CLI entry points** (steps run standalone) have no
  `PipelineRun` row; they leave `pipeline_run` as ``None`` (and the
  `Error` row's `pipeline_run_id` ends up null). A tiny decorator
  `@cli_entry_point(component=...)` wraps each `main()` and runs the
  capture/handler on the way out (see below).

`set_pipeline_run` replaces the whole bundle with a copy carrying the new
snapshot, keeping any step/files already in scope. The from-config
constructor is `ErrorContext.from_config` (above); `set_error_context`
installs a fully-built bundle (used by the bootstrap).

```python
def set_error_context(ctx: ErrorContext) -> contextvars.Token:
    return _context.set(ctx)


def set_pipeline_run(run: Optional[FrozenRow]) -> contextvars.Token:
    current = _context.get()
    return _context.set(
        ErrorContext(
            pipeline_run=run,
            step_name=current.step_name,
            related_files=current.related_files,
            in_worker=current.in_worker,
        )
    )
```

#### Hooking the existing bootstrap

`setup_process_map()` gains one line at the end (it already has
`project_home`, `processing_step`, etc. in `config`):

```python
    # ... existing logging / IO / project-home setup ...
    set_error_context(ErrorContext.from_config(config))
```

One call rebuilds the whole bundle — pipeline-run snapshot, step name,
and the `in_worker` flag (inferred from `parent_pid`) — from `config`.
Because this code path runs in every process (`setup_process` in the
main process, `setup_process_map` as the Pool `initializer` in workers),
it gives both the main process and the workers their context for free —
the main process gets `in_worker=False` since it has no `parent_pid`.
`run_pipeline.main` / the CLI decorator additionally set the run snapshot
directly (so context exists even before the first `setup_process` call).

#### Scoping finer context: `error_context`

Inside a step, the input/output filenames are known per image. A
context manager pushes them so a deeper raise picks them up, and pops
on exit:

```python
@contextmanager
def error_context(*, step_name=None, related_files: Sequence[RelatedFile] = ()):
    """Scope additional context for any error raised inside the block.

    Builds a new :class:`ErrorContext` (step and files supplied at
    construction, not by mutating the current one), installs it for the
    duration of the block, and resets the token on exit.

    Args:
        step_name(str or None):    Override the ambient step name for
            the duration of the block.

        related_files(Sequence[RelatedFile]):    Files appended to the
            ambient related-files list for the duration of the block.

    Yields:
        None
    """

    current = _context.get()
    token = _context.set(
        ErrorContext(
            pipeline_run=current.pipeline_run,
            step_name=step_name or current.step_name,
            related_files=current.related_files + tuple(related_files),
            in_worker=current.in_worker,
        )
    )
    try:
        yield
    finally:
        _context.reset(token)
```

The manager wraps each per-image dispatch in
`with error_context(related_files=[RelatedFile(kind, path, role="input")])`
using the filename it already computes in `get_step_input`
(`image_processing.py`). Steps that produce an output add the expected
output path the same way.

### The capture boundary: `@capture_errors`

A single decorator applied to step `main()` functions and to the
manager's per-image worker function. On the way out it stamps whatever
the ambient context knows onto the exception, then re-raises:

```python
def capture_errors(*, component: Component, wrap_unknown=True):
    """Stamp ambient context onto errors leaving the wrapped callable.

    Args:
        component(Component):    Which component the wrapped callable
            belongs to, used when wrapping unknown exceptions.

        wrap_unknown(bool):    If True, wrap non-:class:`AutoWISPError`
            exceptions in the appropriate concrete class (preserving
            ``__cause__``); if False, let them propagate untouched.

    Returns:
        Callable:    A decorator for the step/dispatch callable.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AutoWISPError as exc:
                _stamp(exc)
                raise
            except Exception as exc:  # noqa: BLE001
                if not wrap_unknown:
                    raise
                wrapped = _wrap(exc, component)
                _stamp(wrapped)
                raise wrapped from exc
        return wrapper
    return decorate


def _stamp(exc: AutoWISPError) -> None:
    """Fill any unset context fields on ``exc`` from the ambient context.

    Args:
        exc(AutoWISPError):    The exception to stamp in place. Already
            populated fields are left untouched.

    Returns:
        None
    """

    ctx = get_error_context()
    if isinstance(exc, StepError) and not getattr(exc, "step_name", None):
        exc.step_name = ctx.step_name
    if not exc.related_files:
        exc.related_files = ctx.related_files
    if exc.pipeline_run is None and ctx.pipeline_run is not None:
        exc.with_pipeline_run(ctx.pipeline_run)
```

`_wrap` maps an unknown exception to the right concrete class: inside a
step it becomes the step's `StepError` subclass (looked up from
`component`/`get_error_context().step_name`); in the orchestration layer
it becomes a `PipelineError`. The original is preserved as `__cause__` —
never swallowed.

Note `_stamp` mutates `exc.step_name` / `exc.related_files` /
`exc.pipeline_run` / `exc.crashed`. Phase 1 declares these
`__init__`-assigned instance attributes (not the frozen dataclasses), so
they remain writable; only the `RelatedFile` / `FrozenRow` payloads are
frozen. This is the one place that writes them post-construction.

### Worker propagation hook

Phase 3 owns full multiprocessing propagation, but Phase 2 defines the
hook the `Pool` sites will adopt, so we can retire the current
stringify-to-`RuntimeError` workarounds in `epd_correction.py`,
`iterative_refit.py`, and `apply_correction.py`. A worker entry point is
wrapped with:

```python
def worker_entry(func, component: Component):
    """Wrap a Pool worker callable so errors come back picklable + stamped.

    Args:
        func(Callable):    The worker callable to wrap.

        component(Component):    Component used to pick the wrapper class
            for a non-AutoWISP exception (see ``_stamp_worker_error``).

    Returns:
        Callable:    The wrapped callable, suitable to hand to a Pool.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            stamped = _stamp_worker_error(exc, component)
            if stamped is exc:   # already an AutoWISPError, stamped in place
                raise
            raise stamped from exc
    return wrapper
```

The shared `_stamp_worker_error(exc, component)` (also used by Phase 3's
`capture_for_queue`) stamps an `AutoWISPError` in place, or wraps any
other exception via `_wrap` into the step's concrete `StepError` subclass
— **not** a `WorkerCrashedError`, which is reserved for a worker that
dies *without* producing an error object (synthesised by the parent;
Requirement 3). It also records the worker traceback in
`details["original_traceback"]`, the only copy that survives back to the
parent (Scheme A's `RemoteTraceback` is on the live object only, Scheme B
has none, and neither transport pickles `__cause__`).

This is safe to pickle back to the parent precisely because Phase 1
made every field round-trip through pickle. The parent's
`capture_errors` then sees an already-stamped `AutoWISPError` and only
fills the pipeline-run snapshot it has locally.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/database/frozen_row.py` | **New.** Dependency-free `FrozenRow` (no SQLAlchemy import in any form). |
| `autowisp/database/interface.py` | Add `snapshot_row` (uses the `inspect as sa_inspect` it already imports). |
| `autowisp/error_context.py` | **New.** `ErrorContext` bundle (incl. an `in_worker` flag) in a single ContextVar, `get_error_context` / `set_error_context` / `set_pipeline_run` / `in_worker`, `ErrorContext.from_config`, `error_context`, `capture_errors`, `worker_entry`. |
| `autowisp/miscellaneous.py` | `get_code_version_str` moved here from `multiprocessing_util.py` — it is general code-provenance, not multiprocessing-specific, and as a dependency-free leaf it lets `error_context` import it at top level (no cycle, no lazy import). `multiprocessing_util` re-imports it for its `__main__`. |
| `autowisp/multiprocessing_util.py` | `setup_process_map` ends with `set_error_context(ErrorContext.from_config(config))` — one call sets step + pipeline-run context **and the `in_worker` flag** (inferred from `parent_pid`). |
| `autowisp/database/processing.py` | `ProcessingManager.__init__` looks up the `PipelineRun` row once and stores `pipeline_run_id` / `host` / `pipeline_started` / `code_version` (computed once here, not per-worker); `get_config` injects these into every per-step config so each worker rebuilds the snapshot. |
| `autowisp/run_pipeline.py` | Set the run snapshot (`snapshot_row` via `set_pipeline_run`) right after creating the `PipelineRun` row; top-level handler in `main` fills `crashed`/run on any escaping `AutoWISPError` (body extracted to `_run_pipeline`). |
| `autowisp/database/image_processing.py` | Per-batch dispatch split into `_process_batch` (scopes the step via `error_context(step_name=...)`) wrapping `_run_step` (`@capture_errors(component=Component.STEP)`), so the step name is still in scope when the inner capture stamps. |
| each `processing_steps/*.py` `main()` | Decorate with `@capture_errors` / `@cli_entry_point` for the standalone path. **Deferred to phase 7**: `cli_entry_point` was built in phase 5, but applying it to the step `main()`s is part of the phase-7 call-site migration. |
| Pool worker sites (`epd_correction.py`, `iterative_refit.py`, `apply_correction.py`, and others) | `worker_entry` is defined and unit-tested here; **adoption at the call sites is phase 3** (full inventory + the Process+Queue case in `solve_astrometry.py`). |

### Tests (Phase 2)

- A step `main()` that raises a bare `ValueError` surfaces as the
  correct `StepError` subclass with `step_name` set and `__cause__`
  preserved.
- With a run snapshot (`FrozenRow`) set in the ambient context, a raised
  `AutoWISPError` comes out carrying it (and `crashed` set); with none
  set (unit-test path), `pipeline_run` stays `None` and nothing crashes.
- `error_context(related_files=...)` nesting attaches/detaches
  correctly and the innermost set wins for `step_name`.
- A `setup_process` call rebuilds context from a config dict (simulating
  a worker) and a subsequent raise is fully stamped.
- `ErrorContext.from_config` infers `in_worker` from `parent_pid`: true
  when the key is present, false (main process) when it is absent.
- `worker_entry` stamps `subprocess_id` and the result pickles
  round-trip (feeds directly into the Phase 3 Pool tests).

## Phase 3 — Propagation across process boundaries

### The two parallelism schemes (inventory)

There is **no single parallel idiom** in the codebase. An audit of the
multiprocessing call sites turns up two distinct schemes, and Phase 3
has to handle both. (The migration step must re-run this audit — the
grep below is the starting point, not a guarantee of completeness.)

**Scheme A — `Pool` + `map`/`imap`.** A pool is created with
`initializer=setup_process_map`, work is mapped, and *an exception
raised in the worker is automatically pickled and re-raised in the
parent* at the `.map()` / iteration site:

```python
with Pool(n, initializer=setup_process_map, initargs=(config,)) as pool:
    pool.map(callable, items)        # or imap_unordered(...)
```

Sites: `processing_steps/find_stars.py`,
`processing_steps/fit_star_shape.py` (also uses `Manager`,
`maxtasksperchild=1`), `processing_steps/measure_aperture_photometry.py`,
`light_curves/apply_correction.py`,
`magnitude_fitting/iterative_refit.py`. (The bare `Pool(3)` in
`fit_expression/interface.py` is leftover debugging scaffolding, not a
real parallel site — ignore it here; it should just be removed.)

**Scheme B — `Process` + `Queue`, manual marshalling.**
`solve_astrometry.py` spawns long-lived `Process` workers that bootstrap
with `setup_process(task="solve", ...)`, consume tasks from a
`task_queue`, and **do not raise out of the process** — instead the
worker catches, and *puts the error onto `result_queue`*:

```python
# worker (astrometry_process):
except Exception:
    result_queue.put({"error": RuntimeError(format_exc()), "dr_fname": dr_fname})
    return
# parent (manage_astrometry):
if "error" in result:
    raise result["error"]                       # re-raise on the parent side
...
if workers and not any(p.is_alive() for p in workers):
    raise RuntimeError(...)                      # all workers died
```

So in Scheme B the queue is the transport, the parent explicitly
re-raises, and dead-worker detection is hand-rolled via `is_alive()`.

**Out of scope:** `browser_interface/processing/views.py` uses a
`threading.Thread(target=proc.wait)` purely to reap a detached process —
no data parallelism, no error to propagate.

### What both schemes share

Despite the different transports, four invariants are common, and
Phase 3 implements each once:

1. **Bootstrap.** Both call into `setup_process`/`setup_process_map`,
   which (after phase 2) rebuilds the pipeline-run + step context from
   `config`. So every worker — Pool or Process — already knows *which
   run* and *which step* it is in.
2. **The error object must pickle faithfully** — to be re-raised across
   the Pool boundary (A) *or* to be put on and pulled off a `Queue`
   (B). Same requirement, two transports.
3. **The error must be stamped** (`subprocess_id` + context) before it
   leaves the worker.
4. **A worker can die without producing an error object at all**
   (segfault / OOM / `os._exit`), and the parent must synthesise one.

### Requirement 1 — exceptions must pickle faithfully

Three things must survive the round-trip:

- **The type and all phase-1 fields.** This is *not* automatic. The
  default exception unpickler reconstructs via `cls(*self.args)`, which
  preserves only the message and drops every context field
  (`subprocess_id`, `pipeline_run`, `step_name`, `details`, …) set after
  construction. Phase 1's base class handles this with the `__reduce__` /
  `_rebuild_autowisp_error` pair defined above, which reconstructs
  through `__new__` while restoring both `args` (the message) and
  `__dict__` (the context fields) instead of re-running `__init__`.
  Phase 1's pickling test (`tests/test_exception_hierarchy.py`) is what
  guards this; Phase 3 depends on it.

- **The `__cause__` chain.** When an unknown exception is wrapped (into
  the step's `StepError` subclass), the original may itself be
  unpicklable (e.g. a third-party C-extension error). So
  `_stamp_worker_error` stores a *formatted string* of the original
  (`traceback.format_exc()`) in `details["original_traceback"]` rather
  than relying on a live `__cause__` reference — which would not survive
  pickling anyway (our `__reduce__` carries only `args` + `__dict__`, not
  `__cause__`) and, if kept alive and unpicklable, could turn the whole
  error into an opaque `PicklingError`. This matters for *both* schemes:
  a `Queue.put` of an unpicklable object fails just as a Pool re-raise
  does.

- **The traceback.** In Scheme A, `multiprocessing` attaches the
  worker's traceback to the re-raised exception as a `RemoteTraceback`
  (`__cause__`) automatically. Scheme B has no such machinery — the
  worker is the one calling `Queue.put`, so it must capture the
  traceback string itself (into `details`). The stamping helpers below
  do this so both schemes end up with the worker traceback visible in
  the parent.

### Requirement 2 — stamp before it leaves the worker (both schemes)

The stamping logic is identical; only *where it runs* differs.

- **Scheme A (process pool):** `run_pool` wraps the mapped callable with
  `worker_entry` (phase 2). On the way out it calls `stamp_subprocess()`
  (idempotent) and `_stamp()`, then **re-raises**, letting the executor
  pickle it back to the parent.

- **Scheme B (Process+Queue):** the worker does not re-raise, so
  `worker_entry` does not fit. Phase 3 adds a sibling helper
  `capture_for_queue(exc, *, component) -> AutoWISPError` that performs
  the *same* `stamp_subprocess()` + `_stamp()` + traceback-capture and
  *returns* the stamped, picklable exception for the worker to
  `result_queue.put(...)`. So `astrometry_process`'s
  `except Exception as exc:` becomes:

  ```python
  except Exception as exc:  # noqa: BLE001
      result_queue.put(
          {"error": capture_for_queue(exc, component=Component.STEP),
           "dr_fname": dr_fname}
      )
      return
  ```

  replacing today's `RuntimeError(format_exc())`.

### Requirement 3 — workers that die without an error object

Segfault, OOM-killer, `os._exit` — nothing gets pickled or queued.

- **Scheme A:** a bare `multiprocessing.Pool` **hangs** on a worker that
  dies mid-task — verified empirically on CPython 3.12: `pool.map`
  neither returns nor raises. A hung pipeline step is exactly the opaque,
  confusing failure this plan exists to eliminate, so `run_pool` is built
  on `concurrent.futures.ProcessPoolExecutor`, which raises
  `BrokenProcessPool` when a worker dies. `run_pool` catches that (and
  any other non-`AutoWISPError`) and synthesises a `WorkerCrashedError`
  (a `StepError` — component `step`; see phase 9, item 1) carrying what
  the *parent* knows — step, the inputs in flight (`num_inputs` + a
  sample), pipeline-run context — plus the underlying `pool_error`.

- **Scheme B:** `manage_astrometry` *already* detects this
  (`not any(p.is_alive())`) and raised a bare `RuntimeError`. Phase 3
  upgrades that to a `WorkerCrashedError` and reads `process.exitcode`
  (negative → killed by signal `-exitcode`) for the OS clue.

This "the worker cannot describe its own death, so the parent owns it"
case is the one place the parent synthesises the error in both schemes.

> Why `ProcessPoolExecutor` for Scheme A: it is the only option that
> turns a silent worker death into a prompt, catchable error rather than
> a hang. `run_pool` mirrors the previous `Pool` usage — `executor.map`
> for the ordered/eager sites, `submit` + `as_completed` for the one
> streaming site (`iterative_refit`'s stat collector, replacing
> `imap_unordered`), and `max_tasks_per_child` replacing
> `maxtasksperchild` (CPython ≥ 3.11; the pipeline targets 3.12). Scheme
> B keeps its hand-rolled `Process`/`Queue` loop, with `run_pool` and the
> `manage_astrometry` loop the two places that own worker-death handling.

### Requirement 4 — re-raise in the parent with the full chain

- **Scheme A:** the exception arrives at the parent's `capture_errors`
  boundary already stamped; the parent only fills the pipeline-run
  snapshot if it is somehow still unset.

- **Scheme B:** the parent pulls `result["error"]` off the queue and
  must re-raise it itself. Phase 3 provides `reraise_from_worker(exc)`
  which fills the pipeline-run snapshot from the parent's context if
  unset, then `raise exc`. `manage_astrometry`'s `raise result["error"]`
  becomes `reraise_from_worker(result["error"])`.

Either way the error then propagates to the top-level handler that
records it (phase 4) and renders it (phase 5).

### No nested workers (policy)

Nested multiprocessing is **disallowed**, not merely absent. The reason
is resource control, not error handling: every parallel site is sized
by the user's `num_parallel_processes`, and a worker that spawns its own
pool/process would multiply that out (`N` workers each spawning `N`
gives `N²` live processes), silently blowing past the limit the user
set. So a pool/process worker callable must never create another
pool/process.

This holds today — audited: every caller of a pool-creating function
(`find_stars`, `fit_star_shape`, `measure_aperture_photometry`,
`apply_parallel_correction`, `single_iteration`, and
`solve_astrometry`'s `Process` spawn) is a step `main()` or the
in-process orchestration loop, never another worker — and the plan
keeps it that way:

- `setup_process_map` / `setup_process` set an `in_worker` flag in the
  error context when they bootstrap a worker process.
- The Phase 3 `run_pool` helper (and, for Scheme B, the
  `Process`-spawning path) assert `not in_worker()` before creating
  workers, raising a `PipelineError` if a nested launch is ever
  introduced. This turns an accidental `N²` regression into an
  immediate, clearly-attributed failure instead of a silent resource
  blow-up.

Because nesting cannot occur, `subprocess_id` is set exactly once, by
the single worker layer. `stamp_subprocess()` is still written
idempotently (phase 1) only as a cheap guard against an exception
passing through more than one `except` on its way out of the *same*
worker — not to arbitrate between nested layers.

### Retiring the existing workarounds

The faithful-pickling guarantee plus the two helpers let the bespoke
hacks go. Note the audit found more Pool sites than the detrending ones:

| Site | Scheme | Today | After phase 3 |
| ---- | ------ | ----- | ------------- |
| `epd_correction.py` (~L399–423) | A | Catches all, builds a giant string, raises `RuntimeError` "to avoid pickling error". | Let the wrapped `EPDError` propagate; diagnostics move into `details`. Delete the stringify. |
| `iterative_refit.py` | A | Bare `pool_magfit` mapped (`map` / `imap_unordered`). | Call `run_pool(pool_magfit, …)`; the stat-collector case passes `stream_consumer=`. |
| `apply_correction.py` | A | Bare `correct` mapped. | `numpy.concatenate(run_pool(correct, …))`. |
| `find_stars.py`, `fit_star_shape.py`, `measure_aperture_photometry.py` | A | Bare callable mapped on a `Pool`. | Call `run_pool(callable, …)` (it wraps with `worker_entry` and runs the executor); `fit_star_shape`/`measure` pass `max_tasks_per_child=1`. |
| `solve_astrometry.py` | B | Worker puts `RuntimeError(format_exc())` on `result_queue`; parent `raise result["error"]`; dead workers → bare `RuntimeError`. | Worker puts `capture_for_queue(exc, ...)`; parent `reraise_from_worker(...)`; dead workers → `WorkerCrashedError` with `exitcode`. |

The Scheme-A sites need no change to `setup_process_map` beyond
phase 2 — they already use it as the initializer.

The three new helpers (`run_pool`, `capture_for_queue`,
`reraise_from_worker`) live alongside `worker_entry` in
`autowisp/error_context.py`.

### Tests (Phase 3)

- **Scheme A:** a worker raising a `StepError` → the parent receives the
  *same* type with `subprocess_id` == the worker PID (≠ parent PID),
  `step_name` preserved, `pipeline_run` set, and the worker traceback
  visible via the `RemoteTraceback` cause.
- **Scheme A:** a worker raising a bare `ValueError` → the parent
  receives the mapped `StepError` subclass (wrapped), original traceback
  string in `details`, `__cause__` preserved.
- **Scheme B:** a queue worker that fails → the object pulled off the
  queue is a stamped `AutoWISPError` (subprocess_id set, traceback in
  `details`), and `reraise_from_worker` raises it with the pipeline-run
  snapshot attached.
- Pickling round-trip of every concrete exception with a fully
  populated payload (shared with the phase-1 test) — covers both the
  Pool re-raise and the `Queue.put` transport.
- A worker that hard-exits (test helper calling `os._exit`) → the parent
  raises `WorkerCrashedError` naming the step and in-flight inputs (one
  test per scheme), not a hang; Scheme B asserts `exitcode` is recorded.
- `subprocess_id` idempotency when an exception passes through more
  than one `except` in the same worker.
- The nesting guard: calling `run_pool` (or the `Process` spawn path)
  with the `in_worker` flag set raises a `PipelineError` instead of
  creating workers, so an accidental `N²` launch fails loudly.

## Phase 4 — persistence

Errors are persisted as **a queryable row in the database plus a
file-based sidecar holding the heavy detail**. The row carries the small
fields the BUI/CLI list views and developer queries need; the sidecar
carries the full projection of the exception. This split is what keeps
the SQLite file small while still capturing rich per-error context (see
the connected open question above).

### `PipelineRun` gains a `code_version` column

The git hash is carried on the run snapshot (`FrozenRow`) in
phase 1 / phase 2. To survive past the in-memory snapshot, add
`code_version = Column(String(...), nullable=True)` to the `PipelineRun`
model and populate it in `run_pipeline.main` where the row is created
(the value is already on hand from `get_code_version_str()`). Once it is
a real column, `snapshot_row` captures it automatically — the explicit
hand-off in phase 2 can be dropped. The `Error` table references the run
rather than duplicating the hash per error.

### The `Error` table (inline, queryable fields)

A new `error` table with one row per persisted error. Columns are the
*queryable subset* of the exception — the fields list views and
aggregate queries need, so they never have to open a sidecar:

- `id` — PK, also names the sidecar file.
- `pipeline_run_id` — FK to `PipelineRun` (nullable; CLI/BUI errors
  have no run). The `code_version`, host, and PID come from the run row,
  not duplicated here.
- `component` — `component.value` (`"step"` / `"pipeline"` / `"bui"`).
- `step_name` — nullable; set for `StepError`.
- `exception_class` — concrete class name, for filtering "all
  `SolveAstrometryError`s".
- artifact FKs — nullable `image_id` / `dr_file_id` / `lightcurve_id` /
  `master_file_id`, set from the `related_files` when one maps to a
  known artifact row. This is what powers "what failed for *this*
  lightcurve?".
- `subprocess_id` — nullable.
- `user_message` — the short, jargon-free string for the BUI/CLI.
- `created` — timestamp (the exception's `crashed` value).
- `sidecar_path` — path to the detail file, **relative to
  `project_home`**, resolved through `get_project_home()` on read;
  nullable so a row remains valid even if the sidecar write failed.

### The sidecar file (full detail)

One file per error, not a shared append log — so per-row pruning is a
single `unlink` and the row↔blob reference stays 1:1. Location, under
the project home (next to `autowisp.db`, so the project dir stays
self-contained and movable):

```
<project_home>/errors/<pipeline_run_id|cli|bui>/<error_id>.json[.gz]
```

Bucketing by run keeps directories small and makes "drop everything
from run 88" a directory `rmtree`. Naming by `Error.id` ties the file
unambiguously to its row (and requires the id before the write — see
the write path).

Contents — **only the fields that are not already columns on the
`Error` row (or reachable from it).** The sidecar is always opened
*through* `Error.sidecar_path`, so the row's fields are already in hand;
repeating `component` / `step_name` / `subprocess_id` / `user_message`
or the pipeline-run snapshot (reachable via `pipeline_run_id` →
`PipelineRun`) would gain nothing. The file therefore holds the heavy,
non-queryable remainder, produced by a `to_detail_dict()` method on
`AutoWISPError`:

```json
{
  "schema_version": 1,
  "message": "...full technical message (not the short user_message)...",
  "related_files": [{"kind": "dr_file", "path": "...", "role": "input"}],
  "details": { "...": "the big, arbitrary dict" },
  "traceback": "...full formatted __cause__ chain...",
  "bui_request": { "...": "only for BUIError; the largest contributor" }
}
```

Notes on the kept fields:

- `related_files` stays in full here even though the row carries
  artifact FKs — the FKs are a *subset* (only files that map to a known
  `Image` / `DRFile` / `Lightcurve` / `MasterFile` row), whereas the
  list also includes config / catalog / output files with their
  `kind` / `role`. It is a superset, not a duplicate.
- `message` is the raw technical string; only the short `user_message`
  is a column, so the technical message lives here.
- `schema_version` allows the format to evolve.

The row and the sidecar are thus **complementary, not overlapping** —
`persist_error()` writes both, but each field lives in exactly one
place, so there is nothing to keep in sync.

### Serialization: `details` is not plain JSON

`details` will carry numpy scalars/arrays, `Path`, `datetime`, sets,
etc., so `json.dump` needs a total (never-raising) `default=` sanitizer:

- numpy scalar → `.item()`; small ndarray → `.tolist()`; **large**
  ndarray → a summary (`{"__ndarray__": {"shape": ..., "dtype": ...,
  "head": [...]}}`) so a stray full-frame array cannot write hundreds of
  MB.
- `Path` → str, `datetime` → ISO 8601, `set` → list, anything else →
  `repr()` as the last resort.

### Write path (atomic, best-effort)

A single `persist_error(exc, db_session)` is called from the top-level
handlers defined in phase 2 (`run_pipeline.main`, the CLI entry
decorator, the BUI middleware):

1. Insert the `Error` row with the inline fields; `flush()` to obtain
   `error.id`.
2. Build the sidecar path from `id`; `makedirs` the bucket.
3. `json.dump` (with the sanitizer) to `…/<id>.json.tmp`, then
   `os.replace` to the final name — atomic, so a reader never sees a
   half-written file. Write `.json.gz` instead when the payload exceeds
   a threshold (≈64 KB); the stored filename records which.
4. Set `error.sidecar_path` and `commit`.

Two hard rules:

- **Persistence never raises.** The whole body is wrapped in
  `try/except` that logs and continues — an error while recording an
  error must not crash the pipeline or mask the original.
- **Only the parent writes.** By phase 3 the exception is already
  marshalled back to the main process before the handler runs, so
  workers never touch the errors dir — no write contention, and
  id-based names are unique regardless.

A crash between steps 1 and 4 leaves a row whose `sidecar_path` is null
or points at a missing file; readers treat "no sidecar" as "inline
fields only," never as an error.

### Read path

Lazy. BUI list views and CLI summaries use only the inline columns. The
sidecar is opened *only* when drilling into a single error (the BUI
detail view, or a `wisp-show-error <id>` that pretty-prints the dict). A
missing or corrupt sidecar degrades to "inline fields + detail
unavailable," never a crash.

### Retention & cleanup

Because the heavy data is in files, retention is mostly filesystem work:

- `wisp-cleanup-errors --older-than <duration>` (and a BUI action):
  delete `Error` rows older than the cutoff and their sidecars in one
  pass.
- A sweep for **orphans** (files with no row, from write-path crashes)
  and **dangling rows** (row, no file).
- Optionally an opportunistic prune at pipeline startup so an unattended
  deployment does not grow unbounded.
- **Project deletion** removes the sidecars too:
  `delete_all_error_sidecars` unlinks exactly the files persistence wrote
  (one per `Error` row), leaving any unrelated file under the `errors`
  directory; `delete_projects` then prunes the emptied directory.

### Row/sidecar boundary

The inline columns and the sidecar are **disjoint**: a field is a column
when list views or aggregate queries need it (so those never touch the
filesystem), and lives in the sidecar otherwise. Nothing is stored in
both, so there is no synchronization burden and no risk of the two
drifting. The only near-overlap is artifact FKs vs. `related_files`, and
that is a deliberate subset/superset relationship (queryable FKs for the
artifacts we recognize; the complete file list in the sidecar), not a
copy.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/database/data_model/pipeline_run.py` | Add `code_version` column. |
| `autowisp/database/data_model/error.py` | **New.** The `Error` ORM model above. |
| `autowisp/exceptions.py` | Add `AutoWISPError.to_detail_dict()` and the JSON sanitizer. |
| `autowisp/error_persistence.py` (or extend `error_context.py`) | **New.** `persist_error()`, sidecar path/compression helpers, cleanup sweep. |
| `autowisp/run_pipeline.py`, CLI entry decorator, BUI middleware | Call `persist_error()` in the top-level handler. |
| `pyproject.toml` | Add the `wisp-cleanup-errors` script. |

### Tests (Phase 4)

- `to_detail_dict()` round-trips through `json.dumps` with the
  sanitizer for payloads containing numpy scalars/arrays, `Path`,
  `datetime`, and sets; a large ndarray is summarized, not dumped whole.
- `persist_error()` writes an atomic sidecar and an `Error` row, with
  the queryable fields on the row and the remainder in the sidecar (no
  field in both); artifact FKs are set from `related_files`.
- A forced sidecar-write failure still leaves a valid row
  (`sidecar_path` null) and does not propagate.
- Reading an error with a missing sidecar degrades gracefully.
- `wisp-cleanup-errors` removes aged rows + files and clears orphans and
  dangling rows.

## Phase 5 — user-facing rendering (BUI + CLI)

Both front-ends are *projections of the persisted `Error` record* (phase
4) — the BUI and CLI never format an exception directly, they render the
row (and, on demand, its sidecar). Other channels (email/Slack/webhook
notifier) are deliberately out of scope here and will be specified
later.

### Shared formatter

One module, `autowisp/error_render.py`, owns the human-readable
projection so the two front-ends cannot drift:

- `error_summary(error_row) -> str` — one-line: component/step, the
  affected artifact (from the artifact FK), and `user_message`.
- `error_detail(error_row, *, developer=False) -> dict|str` — the full
  human view, lazily loading the sidecar. With `developer=False` it
  shows `user_message`, affected artifact, and remediation if available;
  with `developer=True` it adds the technical `message`, `traceback`,
  `details`, `subprocess_id`, host/PID, and `code_version`.
- Remediation is optional: surfaced when the exception provides one
  (`details["remediation"]`); otherwise omitted, never faked.

The CLI calls these and prints; the BUI calls these and puts the result
in a template context. No formatting logic lives in either front-end.

### CLI rendering

The top-level handlers from phase 2 (the `wisp-*` entry decorator and
`run_pipeline.main`) catch `AutoWISPError` and render to **stderr**:

- Default: `error_summary(...)` plus a pointer line — the `Error.id`,
  the sidecar path, and `Run 'wisp-crash-report <id>' for a shareable
  report` (phase 6).
- `-v` / `--traceback`: switch to `error_detail(developer=True)` for the
  full technical view inline.
- Exit code is non-zero (a small map from `Component` so scripts can
  distinguish step vs. pipeline vs. config failures).

Caveat for the detached run: `run_pipeline.py` redirects stdout/stderr
to `run_pipeline.out` under the app data dir, so its rendered summary
lands there, not a terminal — which is exactly why the persisted row +
BUI surfacing matter. Standalone `wisp-*` invocations render to the real
stderr.

### BUI rendering

Errors fold into the existing processing UI rather than a bolted-on
page:

- **Error list view** (new, in the `processing` app): a table over the
  `Error` rows for the current project, newest first, columns =
  `error_summary` fields + timestamp, filterable by run / step /
  component. Backed by the inline columns only (no sidecar reads), so
  the list stays cheap.
- **Error detail view** (new): renders `error_detail`. Shows the
  user-facing view by default with a "show developer detail" toggle that
  reveals the sidecar-backed technical fields, the related-file list,
  and the traceback. Carries the phase-6 "Download crash report" button.
- **Progress grid integration**: `progress_view.py` already renders
  per-channel/step status cells, with failures as non-positive status
  codes. Make a failed cell **link to the matching `Error`
  detail/list** (joined on run + step + image/channel), so a user who
  sees a red cell is one click from the explanation. This is the
  primary discovery path.
- **Log cross-link** *(delivered in phase 6)*: the error detail links to
  the matching per-process log (the existing `review` / `review_single`
  pages). It reuses the same "find the right log(s) for this error"
  resolution the crash-report bundler builds (`find_error_progress`:
  run + step → `ImageProcessingProgress`), so the zip and the "View Log"
  link share one helper rather than duplicating it.

### Request snapshot for BUI errors

This is where the phase-1/phase-5 question lands: a `BUIError` (or any
`AutoWISPError` escaping a Django view) should capture the request
context — path, view name, query/POST *keys* (not values), session ID —
into `details` so it reaches the sidecar (the large-payload reason it
goes to the sidecar, not inline, per phase 4). A small middleware /
decorator on the BUI views does this capture and routes the error
through `persist_error`, mirroring the pipeline's top-level handler.
Values are omitted (only keys) to avoid persisting user-entered secrets.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/error_render.py` | **New.** `error_summary`, `error_detail`, `format_detail_text`, `error_list_rows`, the open-error counts, remediation lookup. |
| `autowisp/error_cli.py` | **New.** `report_error` (persist + render to stderr), `exit_code_for`, and the `cli_entry_point` decorator. `run_pipeline.main` reports via it. |
| `browser_interface/processing/error_views.py` (+ urls, templates) | **New** error list + detail views (and the resolve/delete actions). |
| `browser_interface/processing/progress_view.py` (+ template) | Per-step red-alert marker on the image-type cell; the Start Processing button reflects open errors (two-click confirm). |
| `browser_interface/core/...` | Global "Errors (N)" badge via the context processor + base template. |
| `browser_interface/error_capture_middleware.py` | **New.** Capture request context (keys only) + route view errors through `persist_error`. |

### Status

Phase 5 is complete. Of the two items carried elsewhere, the **log
cross-link** (above) was delivered in phase 6; a **component filter** on
the list remains a later nicety (run + step filters exist). The discovery surfaces ended up as the per-step grid marker, the
global badge, and the start-processing gate (a two-click red button); a
manual **resolve/reopen** state and a **delete** action were added on top
of the original plan.

### Tests (Phase 5)

- `error_summary` / `error_detail` render from a persisted row, and
  `developer=True` adds the sidecar-backed fields while `False` omits
  them; remediation appears only when present.
- The CLI handler prints the summary + pointer to stderr and exits
  non-zero; `-v` prints the developer detail.
- The BUI list view renders without opening any sidecar; the detail view
  loads the sidecar and degrades gracefully when it is missing.
- A failed progress cell links to the correct `Error`.
- The BUI middleware captures request *keys* (never values) into
  `details` and persists the error.

## Phase 6 — crash-report bundler (implemented)

When the structured detail is still not enough — or the user simply
wants to hand the problem off — they should be able to produce a single
self-contained zip to send to the maintainers, with no manual
file-hunting across the per-PID logs.

**Status: complete.** Delivered in `autowisp/crash_report.py`
(scrubbing: `scrub_text` / `scrub_mapping` / `scrub_config_values`;
log-selection: `find_error_progress` / `select_error_logs`;
`collect_provenance`; `build_crash_report`; `crash_report_main`), the
`wisp-crash-report` CLI, and the BUI error-detail "Download crash report"
button plus the "View log" cross-link (the deferred phase-5 item).

### Trigger

- CLI: `wisp-crash-report <project_home> <error_id> [--out report.zip]
  [--last] [--max-log-bytes N]` (`--last` targets the most recent error;
  `--max-log-bytes` raises the per-log truncation cap for a more
  thorough report).
- BUI: a "Download crash report" button on an error's detail view.

Both call one
`build_crash_report(error_id, out_path, *, max_log_bytes=...)`.

### Contents

Everything needed to reproduce/diagnose, gathered from sources the
earlier phases already populate:

- The `Error` row — serialized to JSON — and its **sidecar** file
  (phase 4). *(Refinement still open: for a `WorkerCrashedError`,
  include the whole failed batch's errors, not just one. The current
  builder bundles the single requested error.)*
- The relevant **logs / stdout-stderr**: the per-process
  `{task}_{now}_{pid}.outerr` and `.log` files written by
  `setup_process_map` (`multiprocessing_util.py`), selected by the
  failure's `pipeline_run_id` / `subprocess_id` / time window rather
  than dumping the whole log directory.
- The **configuration snapshot** in effect for the run/step.
- The **project database** — a *scrubbed copy* (the SQLite file copied
  then scrubbed, or a dump for a server DB), since it is often the most
  useful artifact for reproducing a failure.
- **Provenance**: `code_version` (git hash), Python / OS / key package
  versions (`astrowisp`, `numpy`), hostname.
- A short **`manifest.json`** describing what was collected and the
  report's own schema version.

### Credential scrubbing (required)

The config, logs, **and database** can contain secrets — e.g.
`gaia_user` / `gaia_password` are threaded through the process config
(`multiprocessing_util.py`) *and* stored as `Configuration` rows in the
project database. The scrubbing pass is mandatory and tested; nothing
enters the zip unscrubbed:

- **Text artifacts** (logs, config snapshot): `scrub_text` /
  `scrub_mapping` in `crash_report.py` redact values of any secret-named
  key, plus anything matching credential-like patterns.
- **The database**: a binary `.db` / SQL dump cannot be text-scrubbed
  line-by-line, so `scrub_config_values` redacts the values row-by-row in
  a *copy* of the DB (every `Configuration` whose parameter name names a
  secret) before that copy enters the zip — never the live database.

### Constraints

- Read-only with respect to pipeline state; never mutates the DB or
  deletes logs.
- Bounded size: reuse the phase-4 large-payload handling and optionally
  truncate very large logs (head+tail) with a note in the manifest.
- Best-effort per source: a missing log or unreadable sidecar becomes a
  noted gap in the manifest, not a failure of the whole report.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/crash_report.py` | **New.** `build_crash_report()`, the log-selection helpers (`find_error_progress` / `select_error_logs`, reusing the configured log naming via `find_processing_outputs`), `collect_provenance`, the scrubbers, and `crash_report_main`. |
| `pyproject.toml` | Added the `wisp-crash-report` script. |
| BUI error detail view (`error_views.py`, `urls.py`, `error_detail.html`) | "Download crash report" action (header bar) streaming the built zip, **and** a "View log" link (side bar) to the matching `review` page when `find_error_progress` resolves one. |

### Tests (Phase 6)

- `build_crash_report()` produces a zip containing the error JSON,
  sidecar, the matching (and only the matching) logs, config, and
  `manifest.json`.
- The scrubber removes `gaia_password` and pattern-matched secrets from
  every artifact in the zip.
- A missing log / sidecar is recorded as a manifest gap, and the report
  still builds.
- The shared log-selection helper resolves a step error to the matching
  `ImageProcessingProgress` (run + step [+ image type]) so the detail
  page's "View log" link targets the right log; an error with no
  resolvable progress (pipeline/BUI) yields no link rather than a wrong
  one.

## Phase 7 — call-site migration

Phases 1–6 built the machinery; nothing *forced* the rest of the code to
use it. Phase 7 is the migration that makes the hierarchy the actual,
exclusive way AutoWISP signals failure:

1. **Apply the CLI error boundary** (`cli_entry_point`, built in phase 5)
   to every standalone step `main()` — the one phase-2 deferral that was
   explicitly parked for here (see the phase-2 "What changes" table).
2. **Fold the legacy ad-hoc classes** into `autowisp/exceptions.py`,
   dropping the stdlib-compat multiple inheritance and the separate
   `pipeline_exceptions.py` module, and fix the handful of `except`
   sites that relied on the stdlib bases.
3. **Retype the high-value raise sites** so the error carries a precise
   class, a jargon-free `user_message`, and `related_files` — without
   churning every deep `raise` (auto-wrapping already covers those; see
   below).

### What "migration" does *not* mean: auto-wrapping is already in place

A deliberate scoping point, because it shrinks phase 7 a lot.
`@capture_errors(component=Component.STEP)` already wraps `_run_step`
(`image_processing.py`), and phase 3's `run_pool` / `capture_for_queue`
already wrap the worker callables. So **any** bare `ValueError` /
`RuntimeError` / `KeyError` raised deep inside a step *already* surfaces
as that step's concrete `StepError` subclass, stamped with `step_name`,
the pipeline-run snapshot, `subprocess_id`, the ambient `related_files`,
and the original traceback in `details["original_traceback"]` (with
`__cause__` preserved). The standalone path gets the same once
`cli_entry_point` is applied (item 1).

Therefore phase 7 does **not** rewrite all ~119 raise sites. Converting a
deep `raise ValueError(...)` to `raise SomeStepError(...)` is worthwhile
*only* when it buys something the auto-wrapper cannot infer:

- a **more specific class** than the step default (e.g. a convergence
  failure inside `fit_magnitudes` that callers want to catch as
  `ConvergenceError` rather than the generic `FitMagnitudesError`);
- a **`user_message`** materially clearer than the technical message;
- **`related_files`** the ambient context does not already carry; or
- a site **outside** any capture boundary (orchestration code in
  `run_pipeline.py` / `database/`, and BUI views), where there is no
  auto-wrapper to lean on.

Everything else is left to auto-wrapping. This keeps the diff focused on
sites where an explicit type changes behaviour (a `catch` becomes
possible) or the user-facing message, rather than mechanical churn.

### Item 1 — apply `cli_entry_point` to step `main()`s

Each `wisp-*` step entry in `pyproject.toml` points at a
`processing_steps/<step>.py:main`. Today those `main()`s run
`setup_process(task=...)` and call the step function directly, so an
exception escapes as a raw traceback with no persisted `Error` row and no
rendered summary. Decorating each with
`@cli_entry_point(component=Component.STEP)` routes the escape through
`capture_errors` → `report_error` (persist + stderr render + non-zero
exit), exactly as the in-pipeline path already does via `_run_step`.

Scope: the step modules with a `main()` wired to a `wisp-*` script —
`calibrate`, `stack_to_master`, `stack_to_master_flat`, `find_stars`,
`solve_astrometry`, `fit_star_shape`, `measure_aperture_photometry`,
`fit_source_extracted_psf_map`, `fit_magnitudes`, `create_lightcurves`,
`epd`, `tfa`, plus `generate_epd_statistics` / `generate_tfa_statistics`.

Two wrinkles to handle during the migration:

- **Component for the stat generators.** `generate_*_statistics` are
  detrending-stat steps → `Component.STEP`. Anything that turns out to be
  pure orchestration (no step semantics) takes `Component.PIPELINE`.
- **Return value vs. exit code.** Several `main()`s `return 0` / `return
  -1`. `cli_entry_point` only intervenes on an *exception* (it calls
  `sys.exit(report_error(...))`); a normal return is passed through
  untouched, so the existing `return 0` contract is preserved. Any
  `main()` that signals failure with a *return value* rather than an
  exception (e.g. `calibrate`'s `return -1`) should raise the appropriate
  `StepError` instead, so the failure is actually recorded — otherwise
  `cli_entry_point` never sees it.

`run_pipeline.main` already has its own top-level handler (phase 2/5) and
is **not** a `wisp-*` step entry, so it is left as-is.

### Item 2 — fold the legacy classes into the hierarchy

Phase 1 re-rooted the legacy classes but kept them dual-based on a stdlib
exception (`BadImageError(StepError, ValueError)`, …) and in their old
locations, "so existing `except ValueError/RuntimeError/IndexError`
handlers keep catching them … until phase 7." Phase 7 removes that shim.

Current legacy inventory and target:

| Legacy class | Where defined now | Stdlib mix-in | Target |
| ------------ | ----------------- | ------------- | ------ |
| `OutsideImageError` | `pipeline_exceptions.py` | `IndexError` | `exceptions.py`, drop mix-in (a `CalibrationError`) |
| `ImageMismatchError` | `pipeline_exceptions.py` | `ValueError` | `exceptions.py`, drop mix-in (a `StepError`) |
| `BadImageError` | `pipeline_exceptions.py` | `ValueError` | `exceptions.py`, drop mix-in (a `StepError`) |
| `ConvergenceError` | `pipeline_exceptions.py` | `RuntimeError` | `exceptions.py`, drop mix-in (a `StepError`) |
| `HDF5LayoutError` | `pipeline_exceptions.py` | `RuntimeError` | `exceptions.py`, drop mix-in (a `PipelineError`) |
| `NoMasterError` | `database/image_processing.py` | `ValueError` | keep where it is, drop the `ValueError` mix-in |
| `ProcessingInProgress` | `database/processing.py` | — (already pure `PipelineError`) | leave as-is |
| `MalformedResponse` | `astrometry/astrometry_net_client.py` | — (already pure `SolveAstrometryError`) | leave as-is |
| `RequestError` | `astrometry/astrometry_net_client.py` | — (already pure `SolveAstrometryError`) | leave as-is |

These classes are **not deleted** — they carry real domain meaning
(`BadImageError`, `ImageMismatchError`, `ConvergenceError`,
`OutsideImageError`, `HDF5LayoutError`). What is deleted is the *ad-hoc
shape*: the stdlib multiple inheritance and the standalone
`pipeline_exceptions.py` module. The five classes move into
`exceptions.py` as first-class concrete subclasses next to the other
domain exceptions; `pipeline_exceptions.py` is removed (or reduced to a
thin re-export for one release if external code imports it — decided when
we check for outside importers).

**The mix-in audit (the only risk in item 2).** Dropping `ValueError` /
`RuntimeError` / `IndexError` from these classes means any handler that
caught the *stdlib* type in order to catch the *domain* error stops
catching it. The `except (ValueError|RuntimeError|IndexError)` sites are
enumerable (≈ a dozen, mostly in `catalog.py`, `lc_data_io.py`,
`user_interface.py`, `magnitude_fitting/`, and some BUI views). The audit
classifies each:

- **Catches a genuine stdlib error** (e.g. `int(x)` raising `ValueError`,
  a missing list index) → leave untouched; it never relied on the domain
  class.
- **Relied on the mix-in** to catch a re-rooted domain error → change it
  to catch the domain class explicitly (e.g. `except ConvergenceError`),
  which is strictly clearer.

Each call site that *raises* a legacy class simply updates its import
from `autowisp.pipeline_exceptions` to `autowisp.exceptions`
(`calibrator.py`, `mask_utilities.py`, `fits_utilities.py`,
`image_utilities.py`, `iterative_rejection_util.py`,
`overscan_methods.py`, `hdf5_file.py`). Signatures are unchanged, so the
raises themselves do not move.

### Item 3 — retype the high-value sites

Guided by the "does it buy something" test above. Concretely, the sites
worth an explicit type are:

- **Orchestration / `database/` raises** — these run *outside* any
  `capture_errors` boundary, so a bare `raise ValueError`/`RuntimeError`
  there would surface untyped. Convert to the matching `PipelineError`
  subclass (`ConfigurationError`, `DatabaseError`,
  `DependencyResolutionError`, `MasterSelectionError`,
  `PhotrefBindingError`), attaching `related_files` / `details` where the
  artifact is known.
- **`run_pipeline.py`** config/argument validation → `ConfigurationError`
  with a `user_message` aimed at the operator.
- **BUI view raises** → the `BUIError` subclasses (`ViewError`,
  `FormValidationError`, `ProjectStateError`); the phase-5 middleware
  already persists these, but several views still raise stdlib types.
- **A few step-internal sites** where a *narrower* class than the step
  default is useful to catch — notably `ConvergenceError` from the
  iterative fitters (`magnitude_fitting/`, `iterative_rejection_util.py`)
  and `BadImageError` / `ImageMismatchError` from calibration, which some
  callers branch on.

Deep, single-use `raise ValueError("…")` lines inside a step that no one
catches specifically are **left to auto-wrapping** — converting them adds
churn without changing behaviour or the rendered message.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/exceptions.py` | Add the five folded domain classes (`OutsideImageError`, `ImageMismatchError`, `BadImageError`, `ConvergenceError`, `HDF5LayoutError`) as pure `StepError`/`PipelineError` subclasses (no stdlib mix-in). |
| `autowisp/pipeline_exceptions.py` | Removed outright (no deprecation shim -- the outside-importer check found only in-tree importers, all re-pointed to `autowisp.exceptions`). |
| `autowisp/database/image_processing.py` | `NoMasterError` drops its `ValueError` mix-in. |
| each `processing_steps/*.py` `main()` (the `wisp-*` entries) | Decorate with `@cli_entry_point(component=Component.STEP)`; convert failure-by-return-value (`return -1`) to a raised `StepError`. |
| raise/​catch sites importing the legacy classes (`calibrator.py`, `mask_utilities.py`, `fits_utilities.py`, `image_utilities.py`, `iterative_rejection_util.py`, `overscan_methods.py`, `hdf5_file.py`) | Re-point imports to `autowisp.exceptions`. |
| the audited `except (ValueError|RuntimeError|IndexError)` sites | Switch the ones that relied on a mix-in to catch the domain class; leave genuine-stdlib catches alone. |
| orchestration / `database/` / `run_pipeline.py` / BUI view raises (high-value subset) | Convert to the matching `PipelineError` / `BUIError` subclass with `user_message` (+ `related_files`/`details` where known). |

### Tests (Phase 7)

- A `wisp-*` step `main()` that raises (or whose step function raises)
  surfaces through `cli_entry_point`: an `Error` row + sidecar is
  persisted, a summary is rendered to stderr, and the process exits with
  the `Component.STEP` code — mirroring the existing
  `test_error_cli.py` coverage but at a real step entry point.
- The folded classes are importable from `autowisp.exceptions`, are
  `AutoWISPError` subclasses with the right `component`, and are **not**
  `ValueError`/`RuntimeError`/`IndexError` subclasses anymore (guards
  against an accidental mix-in creeping back).
- `tests/test_exception_hierarchy.py` is extended to cover the folded
  classes (component set, pickling round-trip).
- For each `except` site changed in the audit, a regression test (or an
  existing test) confirms the domain error is still caught.
- A raise in orchestration/`database`/BUI converted in item 3 surfaces as
  the intended typed exception with its `user_message`.
- A grep-style test (or a lint check) asserts `pipeline_exceptions` is no
  longer imported anywhere.

### Status

Phase 7 is complete. All three items landed: the five legacy classes were
folded into `autowisp/exceptions.py` as pure subclasses (`OutsideImageError`
under `CalibrationError`; `ImageMismatchError` / `BadImageError` /
`ConvergenceError` as cross-cutting `StepError`s; `HDF5LayoutError` under
`PipelineError`), `NoMasterError` lost its `ValueError` base,
`pipeline_exceptions.py` was deleted and all importers re-pointed; the
mix-in audit found **no** handler relied on the stdlib bases, so no
`except` site changed. `@cli_entry_point(component=Component.STEP)` is on
all 14 step `main()`s, and the high-value orchestration / BUI raises were
retyped. The deep, auto-wrapped `raise` sites were left for phase 8 (see
the backlog below).

Notes on what differed from / extended the plan:

- **`get_step_names()`** was extracted in `processing_steps/__init__.py`
  as the single source of truth for the step list, and the phase-7 test
  drives the "every step `main()` is a CLI boundary" check from it (so a
  newly-added step is checked automatically).
- **`pipeline_exceptions.py` was removed outright** (no deprecation shim).

Two follow-on fixes surfaced while exercising phase 7 through the BUI and
were delivered alongside it:

- **Stored-configuration parse errors are now recordable.** The pipeline
  builds each step's config by feeding stored configuration through the
  step's `ManualStepArgumentParser` (`ProcessingManager.get_config`); a
  bad value made argparse `sys.exit()`, which (as a `BaseException`)
  escaped `capture_errors` / `run_pipeline.main` and silently ended the
  detached run with no `Error` row. The parser's `error()` now raises a
  catchable `ConfigurationError` while parsing stored config (scoped by
  `raise_config_parse_errors`), leaving interactive CLI parsing with
  argparse's usual usage-and-exit.
- **BUI errors stay in the BUI.** `ErrorCaptureMiddleware` previously
  recorded the error then returned `None`, letting Django render its
  technical exception page. It now records the error, queues a dismissible
  LCARS banner (linking to the error-detail page), and redirects the user
  back to where they were; `Http404` / `PermissionDenied` are passed
  through unchanged. This supersedes the phase-5 middleware's "leave
  Django's 500 untouched" behaviour.

## Phase 8 — deferred-site migration + BUI-specific raises

*(In progress. Below: the two strands, the per-site design as each is
tackled, and the "Deferred raise sites" backlog inventory carried over
from Phase 7.)*

Two strands:

1. **Migrate the worthwhile deferred raise sites.** Phase 7 left the
   bare-stdlib raises inside step / library code to auto-wrapping; the
   catalogued backlog is the "Deferred raise sites" subsection below.
   Phase 8 retypes the subset that actually benefits — a more specific
   class callers can `except`, a clearer `user_message`, or explicit
   `related_files` — rather than the whole list.

2. **New BUI-specific raises.** Introduce a few new exceptions raised
   precisely so the BUI can detect and handle them (distinct presentation
   / recovery, not just the generic error surfacing from phase 5). The
   exact set, their `Component`/parent, and the BUI handling are to be
   designed here.

### Strand 1a — `CatalogError` (done)

Catalog trouble is **not astrometry-specific** — a live Gaia query can
fail, and a cached catalog can fail to cover the frames or mismatch the
required epoch / magnitude range / FOV, during solve_astrometry,
find_stars, fit_star_shape, etc. So `CatalogError` is a cross-cutting
`StepError` (component `step`, `step_name` stamped from context), giving
callers one `except CatalogError` regardless of which step tripped it.

- **Class** added in `exceptions.py` (a `StepError`).
- **`catalog.py` migrated** — all 12 coverage/consistency raises (`FOV
  with no consistent pointing`, `FOV > 40°`, epoch / magnitude-expression
  / magnitude-limit / width / height / RA / Dec mismatches, missing cached
  fixture with live query disabled) go from bare `RuntimeError`/
  `ValueError` to `CatalogError`.
- **The retry-exhaustion case** (`WISPGaia.get_result`) now raises a
  `CatalogError` (chaining the underlying error via `from`) after the last
  of its 10 attempts, instead of re-raising the raw transport error —
  exactly the "ran out of retries" failure.
- **Not** migrated: `solve_astrometry`'s "catalog coverage seems to be in
  an infinite loop" — despite the wording that is a *convergence* failure
  (the astrometry solution shifts as the catalog is re-fetched), a
  `ConvergenceError` candidate, not a `CatalogError`.
- Tests: `test_catalog_error_is_cross_cutting_step_error` and
  `test_get_result_raises_catalog_error_after_retries` (mocked query +
  no-op sleep) in `test_exception_hierarchy.py`; plus the auto-coverage
  from the hierarchy's "every concrete class" + pickle round-trip tests.

### Strand 1b — cohesive library clusters (done)

Retyped the backlog clusters that sit in **library** code (so a specific
class lets callers `except` regardless of which step reached them), each to
an *existing* class -- no new types beyond `CatalogError`:

- `image_calibration/calibrator.py` — the 4 config/data raises (bad area
  dimension, invalid gain, invalid / malformatted leak directions) →
  `CalibrationError`.
- `source_finder_util.py` — unrecognized source-extraction tool →
  `ConfigurationError` (a bad config value).
- `astrometry/astrometry.py` — too few equations to solve for the
  transformation coefficients → `SolveAstrometryError`.
- `magnitude_fitting/master_photref_collector_{grcollect,zarr}.py` —
  "failed to generate master photometric reference" → `FitMagnitudesError`,
  **and** the one catch of it (`iterative_refit.py`, "no new master photref
  this iteration") narrowed from the far-too-broad `except RuntimeError` to
  `except FitMagnitudesError`, so an unrelated `RuntimeError` in
  `generate_master` now surfaces instead of being swallowed.

**Retyping a deferred site requires auditing its catch sites** — these
sites were deferred precisely because callers `except` the *stdlib* type,
and the AutoWISP classes deliberately do not subclass stdlib, so a blind
swap silently breaks control flow. Each cluster above was checked: the
calibrator / source-finder / astrometry raises have no specific-type
handler wrapping them; the master-photref one did (handled above).

- **`hdf5_file.py` — reverted, deliberately deferred.** Its ~14
  `IOError`/`KeyError` raises are **control-flow**, not just errors:
  `DataReductionFile.check_for_dataset(must_exist=True)` and
  `get_attribute` raise `IOError` for "absent", and `DataReductionFile` /
  `LightCurveFile` / `magnitude_fitting` `except IOError` all over to detect
  absence (loop termination, `return False`, existence checks). Retyping to
  `HDF5LayoutError` needs a coordinated update of every such catch site — a
  careful task of its own, not a mechanical swap; left for later.

These are otherwise mechanical class swaps; the classes are covered by the
hierarchy tests, and a smoke import verifies each module. The step-internal
backlog sites (below) are deliberately left to auto-wrapping -- the step
boundary already stamps the step name, so retyping them mostly buys a
narrower `user_message`, lower value.

### Deferred raise sites (Phase 7 backlog)

These are the `raise <stdlib exception>` sites that Phase 7 **leaves to
auto-wrapping** (see "What 'migration' does *not* mean" under Phase 7).
All sit *inside* a step / library call graph, so when reached through the
pipeline they are already wrapped by `@capture_errors` / `run_pool` /
`capture_for_queue` into the right `StepError` subclass, stamped with
`step_name`, the pipeline-run snapshot, `subprocess_id`, ambient
`related_files`, and the original traceback.

They are catalogued here because several **may be worth migrating** for a
more precise class (so callers can `except` it), a clearer
`user_message`, or explicit `related_files`. This list excludes the
"high-value" sites Phase 7 migrates (orchestration / `database/` /
`run_pipeline.py` / BUI), already-typed `AutoWISPError` raises, and the
crash-report / error-persistence tooling. Suggested target classes are
advisory — pick when actually migrating.

#### Not actually an error (leave as-is)

- `processing_steps/lc_detrending_argument_parser.py:25` — `raise
  StopIteration`. Iterator protocol, not a failure. **Do not migrate.**
- `light_curves/tfa_correction.py:1158` — `raise NotImplementedError`
  ("Adding extra templates is not implemented yet."). Genuine
  not-implemented marker; leave unless we want a typed `TFAError`.

#### Catalog (`catalog.py`) — candidate: a `CatalogError` domain class, else `SolveAstrometryError`

Mostly catalog-coverage / consistency validation reached from
solve_astrometry / find_stars. A dedicated `CatalogError` would let
callers catch catalog problems specifically.

- `:803` `ValueError` — FOV requested for frames with no consistent pointing.
- `:940` `RuntimeError` — FOV > 40 degrees not supported.
- `:1068` `RuntimeError` — DR files to be covered by one catalog have different epochs.
- `:1124` `RuntimeError` — catalog epoch mismatch vs. required.
- `:1134` `RuntimeError` — catalog magnitude-expression mismatch.
- `:1148` `RuntimeError` — catalog excludes sources brighter than required.
- `:1159` `RuntimeError` — catalog excludes sources fainter than required.
- `:1169` `RuntimeError` — catalog width less than required.
- `:1177` `RuntimeError` — catalog height less than required.
- `:1190` `RuntimeError` — catalog center RA too far from required.
- `:1199` `RuntimeError` — catalog center Dec too far from required.

#### HDF5 I/O (`hdf5_file.py`) — candidate: `HDF5LayoutError` (layout) / a DR-file IO error, with `related_files`

Shared infra under almost every DR-touching step. Layout/config problems
map to `HDF5LayoutError`; missing-dataset/attribute reads are closer to a
DR-file IO error.

- `:249` `KeyError` — dataset key not in configured file entries.
- `:258` `KeyError` — dataset key does not identify a dataset/link.
- `:268` `IOError` — required dataset does not exist in file.
- `:294` `KeyError` — unrecognized element id.
- `:370` `TypeError` — element exists but is of the wrong type.
- `:613` `ValueError` — argument to `hdf5_class_string` is not an h5py class.
- `:764` `IOError` — link already exists pointing elsewhere.
- `:770` `IOError` — non-link object already exists at link path.
- `:924` `IOError` — dump dataset not created though deletion requested.
- `:959` `KeyError` — attribute key not in configured structure.
- `:964` `KeyError` — attribute key not an attribute.
- `:977` `IOError` — attribute requested from non-existent path.
- `:985` `IOError` — attribute not defined for path.
- `:1165` `IOError` — dataset already exists and overwrite not allowed.

#### Lightcurve I/O — candidate: `CreateLightCurvesError` / a lightcurve-IO error, with `related_files` (the LC)

- `light_curves/light_curve_file.py:115` `IOError` — (message var).
- `light_curves/light_curve_file.py:172` `ValueError` — need ≥1 identifier to create LC file.
- `light_curves/light_curve_file.py:285` `RuntimeError` — dataset shape mismatch within LC.
- `light_curves/light_curve_file.py:509` `IOError` — failed to read LC dataset (length mismatch).
- `light_curves/light_curve_file.py:554` `IOError` — LC length smaller than confirmed length.
- `light_curves/light_curve_file.py:561` `IOError` — LC actual vs expected length mismatch.
- `light_curves/light_curve_file.py:569` `IOError` — unexpected LC length resolution mode.
- `light_curves/lc_data_io.py:479` `Exception` — wraps conversion error (chained). **Broad `Exception`; good migration candidate.**
- `light_curves/lc_data_io.py:675` `IOError` — adding frame-independent dataset.
- `light_curves/lc_data_io.py:1257` `IOError` — `prepare_for_writing()` not called.
- `light_curves/lc_data_io.py:1369` `IOError` — while reading frame (chained).
- `light_curves/lc_data_io.py:1410` `IOError` — while writing source (chained).
- `light_curves/lc_data_slice.py:78` `TypeError` — unrecognized dtype.

#### Astrometry (`astrometry/astrometry.py`) — candidate: `SolveAstrometryError`

- `:762` `ValueError` — too few equations to solve for transformation coefficients.

#### Calibration (`image_calibration/calibrator.py`) — candidate: `CalibrationError`

(`BadImageError` / `ImageMismatchError` / `OutsideImageError` already typed here.)

- `:244` `ValueError` — area has bad dimension (`area_name`/`direction`).
- `:341` `ValueError` — invalid gain specified.
- `:351` `ValueError` — invalid leak direction.
- `:355` `ValueError` — malformatted list of leak directions (chained).

#### Master photometric reference — candidate: `FitMagnitudesError` / `MasterSelectionError`

- `magnitude_fitting/master_photref_collector_grcollect.py:610` `RuntimeError` — failed to generate master photref.
- `magnitude_fitting/master_photref_collector_zarr.py:689` `RuntimeError` — failed to generate master photref.

#### Source finding (`source_finder_util.py`) — candidate: `ConfigurationError` / `FindStarsError`

- `:80` `KeyError` — unrecognized source-extraction tool.

#### Step modules (`processing_steps/`)

Step-internal sites. The step boundary already stamps the step name, so a
migration here mainly buys a clearer `user_message` or a narrower class.

- `add_images_to_db.py:77` `ValueError` — unrecognized image type → `ConfigurationError`.
- `calibrate.py:57` `ValueError` — malformatted channel specification → `ConfigurationError`/`CalibrationError`.
- `calibrate.py:132` `ValueError` — malformatted overscan specification (chained) → `ConfigurationError`/`CalibrationError`.
- `fit_magnitudes.py:222` `ValueError` — master photref filename doesn't match format → `FitMagnitudesError`.
- `fit_magnitudes.py:363` `RuntimeError` — cleanup of interrupted magfit failed (file should not exist) → `FitMagnitudesError`.
- `fit_magnitudes.py:436` `ValueError` — inconsistent interrupted status values → `FitMagnitudesError`.
- `fit_source_extracted_psf_map.py:264` `IOError` — matched sources lack full PSF parameter set → `FitPSFMapError`.
- `fit_star_shape.py:602` `RuntimeError` — images in multi-image fit have inconsistent pointing → `FitStarShapeError`.
- `solve_astrometry.py:452` `RuntimeError` — catalog-coverage loop appears infinite → `SolveAstrometryError`.
- `stack_to_master_flat.py:493` `RuntimeError` — mismatched master filenames during cleanup → `StackToMasterError`.
- `lc_detrending.py:31` `ValueError` — none of the lightcurves is for the target → `ConfigurationError`/detrending error.
- `manual_util.py:626` `ValueError` — non-string default with no type specified → `ConfigurationError`.
- `manual_util.py:650` `ValueError` — could not convert default value for DB → `ConfigurationError`.

#### Other

- `diagnostics/calibrate.py:88` `ValueError` — no observing session with the given label → `ConfigurationError`.
- `bui_util.py:55` `RuntimeError` — requested FITS file does not exist. Reached from the BUI; candidate `ViewError`/`ResourceError` (or leave if only used as a helper).

## Phase 9 — silent-worker-death diagnostics + crash-report completeness

Phases 3 and 6 built the machinery to catch a silent worker death
(`WorkerCrashedError`) and to bundle a shareable crash report. A *real*
crash report — a BUI-generated `crash_report_error_4.zip` for a
`WorkerCrashedError` during `tfa` — put that machinery to the test and
exposed a set of gaps that, together, made the report say little beyond
"the pool crashed." Phase 9 closes them. Every item here is motivated by
what that report did and did not yield.

### Evidence: what the real report proved and what it lost

The failure: on an Intel-mac / conda-forge **numpy 1.26.4 / CPython
3.13** host (a combination outside the CI matrix — no numpy-1.26 wheels
for cp313), a `tfa` worker died with `BrokenProcessPool` and no
worker-reported error, i.e. a hard death (native segfault or an OS kill /
macOS jetsam), 1718 lightcurves in flight.

What was *still* recoverable — but only by hand-querying the bundled
`autowisp.db` — was genuinely useful: `light_curve_processing_progress`
pinned the death to **~25 s into `tfa` on the 3rd of 4 photometric
references** (EPD + epd-statistics complete for all four, `tfa` complete
for photref 2, dead early into photref 3), i.e. in the parallel
`apply_correction` load/template-build phase, not deep in the fit. That
the report *contained* this but did not *surface* it is itself a finding.

What was lost — the two artifacts a silent death needs most:

- **The worker log** — `manifest.json` recorded the single gap
  `logs: "no matching logs found"`. Not bad luck: log-collection
  **cannot** succeed for this error class (items 1–2 below).
- **The culprit input** — the sidecar carried only `num_inputs: 1718`
  and the first-20 `inputs_sample`, which for a silent death names
  nothing (item 3).
- **The nature of the death** — `BrokenProcessPool` alone does not
  distinguish SIGKILL (OOM/jetsam) from SIGSEGV (native crash); items
  4–6 add that.

### Item 1 — a `WorkerCrashedError` must carry its queryable step link

**Root cause of "no matching logs found."** `select_error_logs` →
`find_error_progress` (`crash_report.py`) bails immediately on
`not error_row.step_name`, and for this error the `step_name` column is
empty. Yet the step name *is* known at crash time: `_worker_crashed`
(`error_context.py`) uses `get_error_context().step_name` to *build the
message* ("...died during step 'tfa'...") — it just never lands in a
queryable field. The reason is that `_stamp` copies the ambient
`step_name` only for a `StepError` (`isinstance(exc, StepError)`), and
`WorkerCrashedError` was a `PipelineError`.

**Fix: reclassify `WorkerCrashedError` as a `StepError`** (component
`step`). Although the *parent* synthesises it — the worker cannot describe
its own death — the failure is in the algorithm running *inside* a step,
and the error belongs to that step; the parent merely reports it.
Reclassifying is both more accurate and mechanically cleaner:

- It inherits the `StepError` `step_name` slot and is **auto-stamped by
  the existing `_stamp` machinery**. That covers *both* worker-death
  sites for free: the `run_pool` synthesis (`_worker_crashed`) *and* the
  Scheme-B `manage_astrometry` raise (`solve_astrometry.py`), which
  previously constructed a bare `WorkerCrashedError` with no step.
- `_worker_crashed` additionally passes `step_name=ctx.step_name`
  explicitly (the message already computes it) and stores it in
  `details["step_name"]` as a belt-and-braces copy for the sidecar.
- **No persistence change is needed:** the phase-4 write already does
  `step_name=getattr(exc, "step_name", None)` (`error_persistence.py`),
  so the column populates for any error carrying the attribute —
  `WorkerCrashedError` simply never set one.
- It stays a *single generic* class (not one of the per-stage
  `StepError` subclasses) because the parent has only the ambient step
  *name*, not the failing step's exception type.
- Bonus: `open_error_count_for_steps` (`error_render.py`) counts a
  `step`-component error as relevant only to launches whose steps include
  its `step_name`; as a former `PipelineError` a worker crash was flagged
  as relevant to *every* launch. Reclassifying scopes it correctly.

This restores the DB link the whole log-collection path depends on. (It
is the necessary condition; item 2 is the sufficient one.)

### Item 2 — log selection must cover lightcurve steps

Even with item 1, logs for `tfa` would still not be found:
`find_error_progress` / `select_error_logs` query **`ImageProcessingProgress`**
and call **`ImageProcessingManager.find_processing_outputs`**. But the LC
steps (`create_lightcurves`, `epd`, `generate_epd_statistics`, `tfa`,
`detrending_stat`) record progress in `light_curve_processing_progress`
and are driven by `LightCurveProcessingManager`, which has **no**
`find_processing_outputs`. So an LC-step error can never resolve to a
progress row or its logs — a second, independent cause of the empty
`logs` gap.

The fix is *not* a second, parallel implementation.
`ImageProcessingManager.find_processing_outputs` is already almost
entirely generic: it reads only `progress.run.process_id`,
`progress.step.name`, `progress.image_type.name`, and
`self._processing_config`, then globs the per-process
`logging_fname` / `std_out_err_fname`. Crucially, the LC manager writes
its logs through the *same* naming scheme — `_prepare_processing` calls
`setup_process(processing_step=step_name, image_type=self._current_image_type, ...)`
(`lightcurve_processing.py`), i.e. keyed on **step + image type** exactly
like image processing. So the method already works for LC logs; it just
hard-codes two image-only assumptions. **Promote it to the base
`ProcessingManager`** and abstract only those two:

1. **Which progress table the int→row resolution queries.** The
   `isinstance(processing_progress, ImageProcessingProgress)` /
   `select(ImageProcessingProgress).filter_by(id=...)` becomes a
   subclass-supplied progress model — a `_progress_model` class attribute
   (`ImageProcessingProgress` on the image manager,
   `LightCurveProcessingProgress` on the LC one). The base already
   branches on exactly this pair in `_create_current_processing`, so the
   precedent and the imports are in place.
2. **How to get the `image_type` name for a progress row.** Image
   progress exposes `.image_type` directly; LC progress carries
   `single_photref_id` instead. A tiny overridable hook —
   `_progress_image_type(progress, db_session)` — returns
   `progress.image_type.name` on the image manager, and on the LC manager
   derives it from the single photref the same way the manager already
   does at run time when it sets `self._current_image_type` (resolve the
   sphotref's source `Image` → `ImageType`). *(Check whether the cheaper
   `MasterType.maker_image_type_id` link on the photref is equivalent; if
   so the hook is a one-liner with no DR read. It must reproduce the
   exact string used in the log name, so verify before relying on it.)*

Everything else in `find_processing_outputs` — the `run.process_id`
glob key, the `step.name`, the two `get_log_outerr_filenames` calls, the
`self._processing_config` spread, the main-vs-worker split — is shared
unchanged.

Then the two crash-report entry points stop being image-only:

- `find_error_progress` resolves the step's progress by trying the LC
  progress table and the image progress table for the run+step and taking
  whichever has a row (no hard-coded step-name list; robust to new
  steps). Pipeline/BUI errors with no step still yield `None`.
- `select_error_logs` instantiates the matching manager subclass for the
  resolved progress kind and calls the now-inherited
  `find_processing_outputs`.

This also fixes the BUI error-detail "View log" cross-link for LC-step
errors, which is broken today for the same root cause.

**Implemented.** `find_processing_outputs` and the `_progress_image_type`
hook live on the base `ProcessingManager`; the two managers set
`_progress_model` and their `_progress_image_type`; `crash_report`
dispatches across both progress tables and instantiates the matching
manager. The LC `_progress_image_type` uses the DR-read derivation (the
`MasterType.maker_image_type_id` shortcut was left unverified, so not
relied on). One extra change the design did not anticipate:
`LightCurveProcessingManager.__init__` unconditionally ran `set_pending`
(which opens every pending photref's DR file), so a review-only
`pipeline_run_id=None` instance — how `crash_report` and the BUI use it —
must skip it; guarded on `self._pipeline_run_id is not None`
(`ImageProcessingManager` already had no such scan). Regression:
`test_resolves_lightcurve_step` in `test_crash_report.py`.

### Item 3 — track what each worker is processing (a shared in-flight map)

The parent must be able to say *which* inputs were in flight when a worker
died — the sidecar today has only `num_inputs: 1718` and a blind first-20
sample. The obvious fix (switch the eager `list(executor.map(...))` to
`submit` + a `{future: item}` map) does **not** actually work, because of
two hard limits in `concurrent.futures.process`:

- On a broken pool, `_terminate_broken` sets the *same*
  `BrokenProcessPool` on **every** entry of `pending_work_items`
  (`process.py`, the `for work_id, work_item in ...: set_exception(bpe)`
  loop) — there is no per-future distinction between the culprit and the
  merely-pending.
- `executor.map` submits *all* items upfront, so for an early crash (the
  real `tfa` case died ~25 s in) almost nothing has completed and the
  "not done" set is ≈ the whole input. `submit` + `{future: item}` would
  hand back ~1718 candidates — no better than the head sample.

So the association we need — *which worker is running which item right
now* — does not exist anywhere in the executor: workers pull items off a
shared queue themselves, and the parent is only notified when one
*finishes*, never when one *starts*. We record it ourselves.

- **A shared in-flight map.** A `multiprocessing.Manager().dict()` is
  created in `run_pool` and **carried to each worker on the pickled
  `_WorkerEntry` wrapper itself** — the executor already pickles the
  wrapper for every call item, and a `Manager` proxy pickles/reconnects
  across that boundary, so no `config`/`initargs` plumbing or ambient-
  context field is needed. `_WorkerEntry.__call__` writes
  `self.inflight[os.getpid()] = item` (the raw item, so Item 9 can rebuild
  its related file) immediately before invoking the wrapped callable and
  clears it (`pop(pid, None)`) in a `finally`. At any instant the
  non-empty entries are exactly the items executing; a hard `os._exit`
  skips the `finally`, leaving the culprit behind.
- On `BrokenProcessPool`, `_worker_crashed(items, exc, inflight, ...)`
  reads the map's values into `details["crashed_inputs"]` (as
  `repr(item)`) — a candidate set bounded by `num_processes`, not a blind
  slice of the inputs. The `Manager` is shut down in a `finally` in
  `run_pool`, so nothing leaks.
- A `Manager().dict()` (server process, one small IPC per task start/end)
  is chosen over a `multiprocessing.Array` in shared memory because the
  Array holds only C scalars (so it would store item *indices*, not the
  items) and, worse, there is no stable ``0..N-1`` worker index to key it
  by — the executor never numbers its workers, so a slot would have to be
  claimed via an atomic counter in the initializer. The per-task IPC is
  negligible against `tfa`/`epd` task cost; revisit only if a hot,
  tiny-task site ever adopts `run_pool`.

**Implemented.** `_WorkerEntry`/`worker_entry` gained an optional
`inflight` proxy; `run_pool` owns the `Manager` and passes the map to both
the wrapper and `_worker_crashed`. Verified end-to-end: a hard-exiting
worker leaves its item in the map (`crashed_inputs`), and the proxy write
survives the death. Tests: `test_inflight_map_tracks_then_clears_item`,
`test_inflight_map_cleared_on_error`, `test_worker_crashed_names_inflight_input`
in `test_error_context.py`.

**Honest scope: this yields a *candidate set*, not the unique culprit.**
When worker D segfaults on item X, the executor force-`terminate()`s the
still-busy innocents B and C mid-item too (all their futures get the same
`BrokenProcessPool`), so the map shows `{B:Y, C:Z, D:X}` — the guilty item
plus up to `num_processes-1` innocents, with nothing to mark which is
which. The executor even discards *which* worker died: it waits on all
worker sentinels together (`mp.connection.wait(readers + worker_sentinels)`
in `wait_result_broken_or_wakeup`) but never inspects which sentinel
fired, then terminates the rest — erasing the liveness difference. Pinning
the culprit *within* this set is Item 4's job.

### Item 4 — `faulthandler` in every worker (the culprit's self-report)

A segfault produces no Python exception — which is exactly why the death
is "silent" — and, per Item 3, the parent cannot attribute the death to a
specific worker. The fix is to make the dying worker **incriminate
itself** before it goes: `faulthandler` turns its fatal signal into a
native stack dump in its own log.

- `setup_process_map` (`multiprocessing_util.py`) calls
  `faulthandler.enable(file=<the worker's redirected stderr>)` during
  bootstrap, so a SIGSEGV/SIGABRT/SIGFPE dumps a C-level traceback into
  the per-process `.outerr` file — the file items 1–2 make collectable.
- **This is what isolates the culprit within Item 3's candidate set.**
  The worker that actually faulted (D) dumps a native traceback into *its*
  log; the innocents (B, C) receive a clean `SIGTERM` from the executor's
  `terminate()` and dump nothing. So the collected log carrying a
  faulthandler stack identifies the guilty worker, and Item 3's in-flight
  entry for that same pid names the guilty item — together, the unique
  ``(worker, item, native stack)``. Neither item alone suffices: Item 3
  narrows to ≤ `num_processes`, Item 4 singles out one within it.
- Also register a fault handler on a signal (e.g. `SIGUSR1`, POSIX only)
  so a *hung* — not crashed — worker can be prodded to dump where it is
  stuck; ties into the phase-3 "no nested workers" resource story.
- No-op on platforms/streams where `faulthandler` can't attach; never
  fails bootstrap. (An OOM/jetsam `SIGKILL` cannot be caught by
  `faulthandler` either — that death stays attributable only to Item 3's
  candidate set plus Item 5's exit-signal; a `SIGKILL` with no native dump
  is itself the tell that it was a kill, not a crash.)

**Implemented** as `_enable_faulthandler(sys.stderr)` in
`setup_process_map`, called right after the stderr redirect (and armed for
`SIGUSR1` on POSIX). Verified end-to-end: a real `SIGSEGV` in a `run_pool`
worker writes a "Fatal Python error" native traceback into that worker's
collected `.outerr`. Test: `test_faulthandler_dumps_native_traceback` in
`test_error_context.py`.

### Item 5 — record the OS-level cause of the death

Phase 3 already reads `process.exitcode` for Scheme B (negative →
killed by signal `-exitcode`) but Scheme A (`ProcessPoolExecutor`) hides
the dead worker behind `BrokenProcessPool`, so the report cannot tell
SIGKILL (OOM / macOS jetsam) from SIGSEGV (native crash) — the single
most diagnostic bit for this failure.

- On catching `BrokenProcessPool`, best-effort scan the executor's
  worker processes for a terminated one and record its `exitcode` /
  decoded signal name. This reaches into executor internals (`_processes`),
  so it is strictly best-effort and guarded — an empty list when the API
  shifts, never a secondary failure.
- Normalise the Scheme-A and Scheme-B representations so a report reads
  the same `details["exit_signal"]` regardless of transport.

**Implemented.** `decode_exit_signals(exitcodes)` turns a collection of
`Process.exitcode` values into a portable, structured list — one
`{"exitcode", ...}` entry per *abnormal* exit (dropping `None` = running
and `0` = clean). Decoding is **OS-aware**, which was the subtle part: on
POSIX a negative code is a kill by signal `-code` (so `SIGKILL` →
OOM/jetsam, `SIGSEGV` → native crash, decoded to the name); on **Windows**
there are no POSIX signals — a negative/large code is an NTSTATUS crash
status (e.g. `0xC0000005` access violation), so it is reported in hex
rather than mis-read as a signal. Scheme A: `run_pool` synthesises the
error *inside* the `with` (the executor clears `_processes` on shutdown)
and passes `_pool_exit_signals(executor)` — verified end-to-end that a
segfault → `SIGSEGV`, a `SIGKILL` → `SIGKILL`, a `os._exit(7)` → a plain
code. Scheme B: `solve_astrometry` now records the same
`details["exit_signal"]` via `decode_exit_signals`. Tests:
`TestExitSignalDecode` (POSIX + mocked-Windows branches, and the
running/clean drop) and `test_worker_crashed_records_exit_signal` (POSIX
segfault e2e), plus the updated Scheme-B `test_all_workers_dead_...`.

### Item 6 — a resource snapshot to confirm/deny OOM

Given macOS jetsam and `tfa`'s large in-memory template matrix over 1718
lightcurves, "was it memory?" is the first question and the report
currently cannot answer it.

- `_worker_crashed` records a resource snapshot into `details`:
  `num_parallel_processes`, system total/available RAM (via `psutil` if
  present, else `os.sysconf` / platform fallbacks, best-effort), and the
  parent's peak RSS. A dead worker's own peak RSS is gone, but the
  parent's and the system pressure at crash time are strong signal.
- `collect_provenance` gains the same machine-resource fields so a report
  built later still shows the box's memory ceiling.

**Implemented.** `collect_resource_snapshot()` (in `miscellaneous.py`, the
leaf both `error_context` and `crash_report` already import) returns
`ram_total` / `ram_available` (bytes), `ram_percent_used`, and the
process `process_rss` via `psutil` (a hard dependency, so cross-OS without
the sysconf fallbacks) — best-effort, never raising (empty dict if it
can't). `_worker_crashed` records it as `details["resources"]` together
with `num_processes` (N workers vs. total RAM is the OOM tell, paired with
item 5's `SIGKILL` and item 4's *absent* native dump).
`collect_provenance` adds the same snapshot so a report built later still
shows the box's RAM ceiling. (Peak RSS was dropped in favour of current
RSS + system available: `ru_maxrss` units differ by platform — KiB on
Linux, bytes on macOS — and `psutil` gives a clean portable current RSS;
the system-available figure is the more directly diagnostic number.)
Tests: `TestResourceSnapshot` (fields + the psutil-missing degradation),
`test_worker_crashed_records_resources`, and the provenance test.

### Item 7 — crash-time provenance (scope reduced)

The plan originally framed this as "the report may be built on a
*different machine* than the run" — but that premise is essentially
**false**. To build a report you need the project home (DB + sidecars +
logs), which lives on the machine that ran the pipeline; the BUI shares
that DB. So the report builder is virtually always the same box.

The apparent "two hosts" in the real report were **one machine reported
two ways**: `PipelineRun.host` used `socket.getfqdn()` (→
`1.0.0.127.in-addr.arpa` on a loopback-only laptop) while
`collect_provenance` used `socket.gethostname()` (→
`Shashanks-MacBook-Pro.local`). A recording inconsistency, not a second
machine.

That leaves one *real* concern — **time, not place**: the environment on
that one box can change *between* the crash and building the report (the
user upgrades numpy / astrowisp, then downloads the report, and
`collect_provenance`'s live `importlib.metadata` now shows the *fixed*
versions, hiding the fragile combo that actually crashed). So the reduced
scope, with the `PipelineRun` migration dropped:

- **Host consistency.** A single `get_hostname()` helper (using
  `gethostname`, the clean name) used by *both* `run_pipeline` (for
  `PipelineRun.host`) and `collect_provenance`, so the run host and report
  host agree instead of looking like two boxes.
- **Crash-time environment in the sidecar.** `collect_environment()`
  (platform + key package versions) is captured by `error_persistence`
  *at record time, in the process that hit the error* — the correct
  crash-time versions, no migration. `collect_provenance` records the same
  shape for the report builder, so the two are directly comparable and a
  difference is the tell that packages drifted between failure and report.

**Implemented.** `get_hostname` / `collect_environment` (and, moved for
cohesion, `collect_resource_snapshot` from item 6) live in `exceptions.py`
— the leaf error module (home of `sanitize_for_json`) that
`error_context`, `error_persistence`, `crash_report`, and `run_pipeline`
all import cycle-free. `run_pipeline` records `host=get_hostname()`;
`error_persistence._write_sidecar` adds `detail["environment"]`;
`collect_provenance` reuses the shared helpers. Tests: the sidecar
`environment` capture (`test_error_persistence`), `collect_environment`'s
package filtering (`test_crash_report`), and the existing provenance test.

### Item 8 — record the resolved config on the error

The original framing here ("the bundled DB holds the resolved
configuration, but reading it means opening SQLite") was **wrong**: the DB
holds only the *raw* config rows (`parameter × condition × value ×
version`); the **resolved** config a step actually ran with is a runtime
derivation (`ProcessingManager.get_config` picks each parameter's value by
matching the image's evaluated conditions) and is persisted **nowhere**.
So it is the one config-related thing worth capturing — and, because it
lives only in memory, worth capturing for *every* error, not just a crash.

- The resolved config is a **per-step context item**, set the same way as
  `step_name` / `related_files`: `ErrorContext` gains a `config` field;
  `from_config` snapshots it (so each worker rebuilds it from the config
  it was handed); the managers scope it at each step's dispatch via
  `error_context(config=...)`. `_stamp` then carries it into
  `details["config"]` on any error — so a worker error gets it from its
  own bootstrap and a parent-side error (including a synthesised
  `WorkerCrashedError`, which runs inside the manager's scope) gets the
  *failing step's* config, not the base `add_images_to_db` config the
  parent bootstrapped with. Scrubbed at report time by the existing
  sidecar `scrub_text`.
- **Dropped** (were phase-6 refinements): a `progress.json` timeline and a
  `related_errors.json` summary. Both only *duplicate the bundled DB* (the
  `*_processing_progress` and `error` tables are in the scrubbed copy), and
  after items 1–9 the facts that once needed a hand-query — which step,
  which photref, which input — are already on the error itself
  (`step_name`, `related_files`, `crashed_inputs`). Not worth the code.

**Implemented.** `ErrorContext.config` + `from_config` snapshot;
`error_context(config=...)` scoped in `_process_batch` (image) and
`__call__` (lightcurve); `_stamp` writes `details["config"]`. Tests:
`test_worker_error_carries_config` (worker path) and
`test_worker_crashed_carries_scoped_config` (parent/crash path, proving it
is the failing step's config, not the base).

### Item 9 — populate `related_files` (finish the deferred Phase-2 scoping)

`related_files` is plumbed end to end but **never fed**: the
`ErrorContext.related_files` field, the `error_context(related_files=...)`
manager, the `_stamp` copy onto the exception, the sidecar serialization,
the artifact-FK resolution (`_resolve_artifact_fks` → `image_id` / `dr` /
`lightcurve` / `master`), and the BUI render all exist, but **no
production code ever constructs a `RelatedFile` or passes one to
`error_context()`** (only tests do). The manager scopes
`error_context(step_name=...)` in `_process_batch` and nothing else. So
every persisted error carries `related_files = ()`, and the FK-resolution
and render run on an empty list. This is the half of the Phase-2 per-image
dispatch scoping that was deferred and never done.

The point is not cosmetic: many failures are a mismatch between the
configuration and a *specific file's* contents (FITS header keywords vs.
config, a DR/LC layout, a wrong master), so the file being processed is
the single most useful thing to attach — and today it is absent from every
error, crash or not.

**The natural home is the per-item boundary, driven by a per-call-site
classifier** (the items are heterogeneous — LC paths, DR paths, image
records, image *sets* — so the generic wrapper cannot classify them):

- **`run_pool` gains an optional `related_files` classifier**: either a
  `FileKind` (for plain path-string items) or a picklable callable
  ``item -> RelatedFile | Iterable[RelatedFile] | None`` (module-level or a
  `functools.partial`, since it rides the pickled wrapper to the workers).
  `_WorkerEntry.__call__` wraps the call in
  `error_context(related_files=[...])` for that item, so **any** error the
  worker raises — including a deep config-vs-file mismatch — carries the
  file, is FK-resolved to the real row, and shows in the error detail. The
  five call sites each pass their kind: `apply_correction` → `LIGHTCURVE`,
  `iterative_refit` → `DR_FILE`, `measure_aperture_photometry` /
  `find_stars` / `fit_star_shape` → the image/DR they map over.
- **The crash case falls out** (the ask that motivated this): the
  in-flight item *is* the related file, so a `WorkerCrashedError` links
  straight to the offending lightcurve. `_worker_crashed` also promotes
  the in-flight items (Item 3's map) to `related_files`, not just
  `details["crashed_inputs"]`, so a reviewer never has to dig for it. This
  unifies with Item 3: the same `item` the in-flight map records is the
  one scoped as a related file.
- **Main-process / Scheme-B paths** scope the same way at their per-item
  point using the filename they already compute (`get_step_input` in the
  manager dispatch; the `dr_fname` in `solve_astrometry`'s worker) — the
  literal deferred Phase-2 line. *(Superseded in part: where the per-item
  file is an HDF5 product it now attaches itself, so several of these
  scopes were written and then removed again — see "HDF5 products attach
  themselves" below.)*

**More than the per-item file: per-step auxiliary inputs.** The item being
mapped over is not the only file a failure implicates — most steps also
consume auxiliary inputs that are exactly what a config-vs-file mismatch is
usually *about*, and they are known at dispatch time.

**Every step, and the files each one implicates.** The per-item column is
what the step maps over (the natural `role="input"`); the auxiliary column
is everything else the step reads or is about to write, all of it known at
dispatch time. `→ out` marks a file the step *produces*, attached as
`role="expected_output"` — worth attaching because a failure mid-write is
exactly when you want to know which output is now suspect.

| step | per-item file | auxiliary files | how the per-item file is attached |
| ---- | ------------- | --------------- | --------------------------------- |
| `add_images_to_db` | raw image | — | explicit dispatch scope (FITS) |
| `calibrate` | raw image | master bias / dark / flat applied | explicit, `_calibration_related_files` (FITS) |
| `stack_to_master` | calibrated frame | the master being stacked `→ out` | explicit, `stacking_related_files` (FITS) |
| `stack_to_master_flat` | calibrated frame | the high / low master flats `→ out` | explicit, `stacking_related_files` (FITS) |
| `find_stars` | calibrated image | DR file `→ out` | explicit classifier (image); DR **automatic** once opened |
| `solve_astrometry` | DR file | the Gaia catalog queried | **automatic** (`solve_image` opens the DR as its first act) |
| `fit_star_shape` | the frame *set* (simultaneous fit) | each frame's DR; the catalog | explicit `_frame_set_related_files`; DRs automatic |
| `measure_aperture_photometry` | calibrated image | DR file | explicit classifier (image); DR automatic |
| `fit_source_extracted_psf_map` | DR file | — | **automatic** |
| `calculate_photref_merit` | DR file | — | **automatic** |
| `fit_magnitudes` | DR file | single photref DR; master photref; the catalog | explicit `_magfit_related_files` (needed for crash promotion) |
| `create_lightcurves` | the DR being read, **or** the one LC being written | single photref DR; the lightcurve catalog; the Gaia catalog; the catalog source-list filter | **automatic** for both; catalogs/filter explicit at the step |
| `epd` | lightcurve | single photref DR; output statistics `→ out` | explicit `_detrending_related_files` (crash promotion); LC also automatic |
| `tfa` | lightcurve | single photref DR; the template lightcurves; output statistics `→ out` | as `epd`; templates not attached |
| `generate_epd_statistics`, `generate_tfa_statistics` | lightcurve | single photref DR; the detrending catalog; output statistics `→ out` | **automatic** for the LC; catalog/output explicit at the step |

### HDF5 products attach themselves

Most of the "per-item file" column above needs no code in the step at
all. `HDF5File.__enter__` / `__exit__` scope the product they opened, so
**any DR or lightcurve opened through a `with` block names itself on
every error raised while it is open**, at whatever depth. Each subclass
declares one class attribute (`related_file_kind`); the role comes from
the open mode (`r` → `input`, else `output`), and a nameless in-memory
product attaches nothing. Entering the same object twice pushes a stack
of scopes rather than a single slot.

That covers roughly 67 `with` sites and made the explicit DR/LC scoping
in `fit_source_extracted_psf_map`, `calculate_photref_merit`,
`collect_light_curves`, `recalculate_correction_statistics` and
`_add_catalog_info` redundant; all of it was removed. It cannot cover
FITS images, masters, catalogs, statistics outputs or the source-list
filter, which is why the explicit scopes above remain.

Two limits worth knowing:

- **The open itself is not covered.** `h5py.File` opens in the
  *constructor*, before `__enter__` runs, so "could not open this DR"
  carries no related file (the filename is in the `HDF5LayoutError`
  message instead). The `run_pool` sites keep their explicit classifiers
  regardless, because crash promotion needs one when a worker dies without
  raising: no `__enter__` ever runs, so there is nothing to attach. Scheme
  B does *not* need one -- its dead-worker `WorkerCrashedError` is raised
  in the parent, where a worker-side scope could never have reached it.
- **Coverage follows the `with`, not the step.** Narrowing a block so the
  work happens after the file closes silently drops the file from any
  error. A generic test of the hook cannot see that, so each step relying
  solely on the automatic path has a test that runs it against a real but
  empty product and asserts the natural failure names it.

### The scoping trap: a scope must outlive the stamp

`related_files` used to be lost on every main-process scope, and the
tests did not notice. `error_context` resets its ContextVar as the
exception unwinds — *before* any enclosing `except` runs — so by the time
`capture_errors` (at `ProcessingManager._run_step`) stamped the error,
the ambient context was empty. Only the `run_pool` and Scheme-B sites
worked, because they stamp *inside* the scope on purpose.

The fix keeps the files with the exception rather than with the context:
`error_context` records its own contribution onto the exception passing
through it, `capture_errors` / `_stamp_worker_error` carry that over when
they wrap it in a `StepError` subclass, and `_stamp` merges the recorded
entries with any scopes still in force. Two consequences to preserve:

- **A test that reads the ambient context from inside a scope proves
  nothing.** It passes whether or not anything reaches the error. Assert
  on the related files of the *stamped exception* instead.
- Entries accumulate innermost-first, but **order carries no meaning** —
  the renderer lists them and the artifact-FK lookup is an SQL `IN`. Tests
  compare with `assertCountEqual`, which ignores order while still
  catching a missing, extra or duplicated entry.

Two distinct shapes fall out of the table, and they are wired differently:

- **Per-item** files vary with the work item and must be classified inside
  the worker boundary (the `run_pool` classifier, or the dispatch scope for
  main-process/Scheme-B steps).
- **Auxiliary** files are **batch-constant** (the LC/magfit manager
  processes one single photref at a time; the masters and the catalog come
  from the step's resolved config), so they are bound once at the call site.

**Never attach a collection — attach the file in hand.** `related_files`
is a *diagnostic pointer*, not an inventory of what the step touched, and
the two must not be confused where the collection is large.
`create_lightcurves` is the case that forces the rule: it writes one
lightcurve per catalog source, which is **tens of thousands** of files in
a large field. Attaching them all would bloat every sidecar, drown the
rendered list, and still not say which one failed. The scope must
therefore sit at the innermost point that handles a *single* file, so the
ambient context is never more than one item deep. In
`collect_light_curves` that is three distinct points, each already a loop
over one file:

| loop | file to scope | role |
| ---- | ------------- | ---- |
| `data_io.read` over the DR chunk | that DR file | `input` |
| `data_io.write` over `sources_lc_fnames` | that one LC | `expected_output` |
| the `confirm_lc_length` pass | that one LC | `output` |

So a failure writing source *N*'s lightcurve names *that* lightcurve plus
the step's auxiliaries, and a failure reading a DR names that DR — while
a failure in the surrounding setup (catalog, photref, memory planning)
names only the auxiliaries, which is the correct answer for it.

The `create_lightcurves` auxiliaries are worth spelling out, because
"the catalog" is ambiguous here — the step has **two**:

- **the single photref DR** (`single_photref_dr_fname`) — the reference
  the whole step is bound to;
- **the lightcurve catalog** (`--lightcurve-catalog-fname`, the
  `lc_catalog_{TARGETID}_{CLRCHNL}_{EXPTIME}.fits` master): read as an
  `input` when it already exists, attached as `expected_output` on the
  run that creates it (it is registered as a `MasterFile` of type
  `lightcurve_catalog`, so it FK-resolves);
- **the Gaia query catalog** (`--lc-catalog`, the cached
  `MASTERS/Gaia/{checksum}.fits`) — `FileKind.CATALOG`, `input`;
- **the catalog source-list filter** (`--lc-catalog-source-list`), when
  given: a plain text list of GAIA source IDs that restricts the catalog
  at read time. **Only present once this branch is merged with master** —
  it arrived with the lc-filter work (`read_source_id_list` in
  `catalog.py`) and does not exist on this branch yet. It belongs in the
  list because a wrong or malformed filter file silently changes which
  sources get lightcurves at all, which is exactly the kind of
  config-vs-file mismatch this item exists to surface.

`FileKind` needs no new member for these: both catalogs are
`FileKind.CATALOG`, told apart by `role` (`"lc_catalog"` vs
`"query_catalog"`) rather than by kind, and the source-list filter is
`FileKind.CONFIG` — it is a user-supplied control file, not catalog data.
A dedicated ``LIGHTCURVE_CATALOG`` kind would only be worth it if the BUI
wants to render the two differently. Rather than a second `run_pool` argument, they are folded into
the *same* `related_files` classifier: the call site binds the auxiliary
files into a picklable `functools.partial` of a module-level function that
returns ``[item_file, *auxiliaries]``. One argument, and both the
normal-error scope and the crash promotion (which already run every
in-flight item through the classifier) pick up the auxiliaries for free;
`_worker_crashed` dedups so a shared auxiliary (e.g. the single photref)
appears once, not once per crashed input.

The step builds them from what it already holds: the sphotref /
master-photref / stat filenames and the master bias/dark/flat paths in its
config. Main-process steps that do not use `run_pool` (`calibrate`,
`create_lightcurves`, the two statistics generators) attach them in their
dispatch scope alongside the per-item file.

`FileKind`: the enum already covers the item kinds (`LIGHTCURVE`,
`DR_FILE`, `CALIBRATED_IMAGE`, `RAW_IMAGE`) and the masters
(`MASTER_BIAS` / `MASTER_DARK` / `MASTER_FLAT` / `MASTER_PHOTREF`); the
single photref is a `MasterFile` row, so `DR_FILE` (or a new
``SINGLE_PHOTREF`` kind, if the single-vs-master distinction is worth
surfacing) suffices. Note the masters and the photref are `MasterFile`
rows, so they FK-resolve via `MasterFile.filename` — but
`_resolve_artifact_fks` returns a *single* `master_file_id`, so with
several masters attached only the first becomes an FK while all appear by
path in the rendered list; broadening that to multiple master FKs is a
separate, optional follow-up.

Sequence within the item: land the **lightcurve path
(`apply_correction`) + the `_worker_crashed` promotion** first — that is
the path the real crash hit and the smallest end-to-end slice — then the
remaining `run_pool` sites, then the main-process/Scheme-B dispatch.

**Implemented (the table's last column is the authority on what is
explicit, what is automatic, and what is left).**
`run_pool`/`worker_entry`/`_WorkerEntry` gained an optional `related_files`
classifier (a `FileKind` or an ``item -> RelatedFile | Iterable | None``
callable, resolved by `_resolve_related_files`); `_WorkerEntry.__call__`
scopes `error_context(related_files=...)` around the call so the stamping
`except` (inside the scope) copies the files onto the error before it
pickles back. `_worker_crashed` promotes the in-flight items through the
same classifier (deduping shared auxiliaries) so a silent death carries
them too. Wired at every site:

- **`run_pool` sites** — `apply_correction` →
  `partial(_detrending_related_files, single_photref_dr_fname=...)` (LC +
  single photref); `iterative_refit` →
  `partial(_magfit_related_files, single_photref=..., master_photref=...)`
  (DR + single/master photref); `measure_aperture_photometry` /
  `find_stars` → `FileKind.CALIBRATED_IMAGE`; `fit_star_shape` →
  `_frame_set_related_files` (every frame in the simultaneous-fit set).
- **calibrate** (main-process) — scopes `_calibration_related_files`
  (the raw image + the channel-keyed master bias/dark/flat applied)
  around each image.
- **Lightcurve steps** (`create_lightcurves`, `epd`, `tfa`,
  `generate_epd_statistics`, `generate_tfa_statistics`) — the
  `LightCurveProcessingManager.__call__` dispatch scopes the single
  photref once for the whole step (covering the main-process paths); the
  epd/tfa parallel workers additionally scope each LC via
  `apply_correction`.
- **solve_astrometry** (Scheme B) — nothing explicit: `solve_image`
  opens the DR as its first act and works inside that block, so the file
  records itself on the exception and `capture_for_queue` stamps it onto
  the queued error.

**Still to wire.**

1. ~~Steps with no scope at all~~ — **done.** `add_images_to_db` took the
   per-item dispatch scope `calibrate` already used;
   `fit_source_extracted_psf_map` and `calculate_photref_merit` need none
   at all, since the DR they open attaches itself. The two stackers are
   the one shape that legitimately attaches a *collection*: stacking has
   no per-item boundary — the whole set is averaged in a single
   `MasterMaker`/`MasterFlatMaker` call — so a failure (too few valid
   frames, mismatched geometry) is about the set, not one frame.
   `stacking_related_files` (in `stack_to_master`, reused by the flat
   variant) attaches every input frame plus the master(s) being written as
   `expected_output`. That does not contradict the "never attach a
   collection" rule above: the rule applies where a single-file boundary
   exists, and a stack is bounded by what one master consumes, unlike the
   per-source lightcurves `create_lightcurves` writes.
2. ~~Per-item scope missing under a manager-level scope~~ — **done.**
   `create_lightcurves` and the two statistics generators now name the
   individual DR or lightcurve, all of it automatic: the reading,
   writing and confirmation passes each open exactly one product at a
   time, which is also what keeps a large field's tens of thousands of
   lightcurves out of any single error. What the steps do scope
   explicitly is what is *not* an HDF5 product — the lightcurve catalog
   (`MasterCatalog.as_related_file`, whose role depends on whether this
   run creates it), the Gaia catalog, the source-list filter, and the
   detrending catalog plus statistics output.
3. **Auxiliaries not yet attached where the item already is** — the
   catalog for `solve_astrometry` / `fit_star_shape` / `fit_magnitudes`,
   the TFA template lightcurves, and the `→ out` products throughout. The
   DR files that used to head this list are now automatic. Cheapest of
   the three (the call sites already build a classifier; these only
   extend what it returns) and the most useful for catalog-coverage and
   layout-mismatch failures. The catalog is best attached inside
   `ensure_catalog`, which resolves the `{checksum}` filename every caller
   would otherwise have to reproduce.

Note `_resolve_artifact_fks` only FK-links images/masters — the raw
image, masters, single/master photref all resolve to their rows, while
LC / DR / calibrated files (no row) surface by path in the rendered
related-files list. Tests: the classifier logic in
`test_related_files.py`; the scope/promotion mechanism (including the
multi-file and dedup cases) in `test_error_context.py`.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/error_context.py` | `run_pool` creates a `Manager().dict()` in-flight map and passes it to `worker_entry`; `_WorkerEntry.__call__` writes/clears `self.inflight[os.getpid()]` around the callable (item 3). `_worker_crashed` sets `step_name` on the `WorkerCrashedError`, records `crashed_inputs` (from the map), and records `exit_signal` from `_pool_exit_signals(executor)` (item 5; `decode_exit_signals` is OS-aware — POSIX signal vs. Windows NTSTATUS). `run_pool`/`worker_entry` gain an optional `related_files` classifier (FileKind or picklable callable returning one or many); `_WorkerEntry.__call__` scopes `error_context(related_files=[...])` per item and `_worker_crashed` promotes the in-flight items through it (deduped). Auxiliary files (single photref, masters) are folded into the classifier via `partial`, not a second argument (item 9). `_worker_crashed` also records `details["resources"]` (memory snapshot + `num_processes`) via `collect_resource_snapshot` (item 6). `ErrorContext` gains a `config` field (`from_config` snapshots it) and `_stamp` writes `details["config"]` (item 8). **Scopes also record their own files onto the exception passing through them** (`_remember_related_files`), since the ContextVar is reset before any enclosing `except` runs; `capture_errors` / `_stamp_worker_error` carry those over when they wrap (`_inherit_related_files`), and `_stamp` merges them with the scopes still in force (item 9). |
| `autowisp/miscellaneous.py` | **New** `collect_resource_snapshot()` — cross-OS memory snapshot via `psutil`, best-effort (items 6). |
| run_pool call sites (`apply_correction.py`, `iterative_refit.py`, `measure_aperture_photometry.py`, `find_stars.py`, `fit_star_shape.py`) ✓ | Each passes its `related_files` classifier — a `FileKind` for item-only sites, or a `partial`/module function returning the item plus batch-constant auxiliaries (item 9, done). |
| `calibrate.py` ✓ | Scopes `_calibration_related_files` (raw image + master bias/dark/flat) around each image (item 9, done). |
| `lightcurve_processing.py` (`LightCurveProcessingManager.__call__`) ✓ | Scopes the step's config and the single photref for the whole LC step, covering `create_lightcurves` / `epd` / `tfa` / the statistics generators (items 8, 9). |
| `image_processing.py` (`_process_batch`) ✓ | Its per-step `error_context` now also scopes `config` (item 8). |
| `solve_astrometry.py` ✓ | Explicit DR scoping *removed*: the file attaches itself inside `solve_image`, and the queued error picks it up from the exception (item 9). |
| `autowisp/hdf5_file.py` ✓ | **New behaviour.** `HDF5File.__enter__` / `__exit__` scope the product they open, so every DR / lightcurve names itself on errors raised while it is open; `DataReductionFile` / `LightCurveFile` declare `related_file_kind` (item 9). |
| `collect_light_curves.py` ✓ | Split into `_prepare_lc_collection` (catalog, sources, destinations) and `_write_lightcurves` (the chunk loop), which also gives the writing pass a testable seam. No explicit related-files scoping: every DR and lightcurve it touches is opened through a `with`. |
| `create_lightcurves.py`, `lc_detrending.py` ✓ | Scope what is *not* an HDF5 product: the lightcurve catalog (`MasterCatalog.as_related_file`), the Gaia catalog, the source-list filter, the detrending catalog and the statistics output (item 9). |
| `autowisp/tests/test_related_files.py` ✓ | **New.** The per-step classifiers, the main-process dispatch scopes (asserted on the *stamped exception*), and HDF5 self-attachment -- including two steps run against a real but empty DR, so a later narrowing of the `with` is caught. |
| `autowisp/exceptions.py` | `WorkerCrashedError` is re-parented from `PipelineError` to `StepError` (component `step`, inheriting the `step_name` slot); no change to persistence, which already reads `getattr(exc, "step_name", None)`. |
| `autowisp/multiprocessing_util.py` | `setup_process_map` enables `faulthandler` against the worker's redirected stderr and registers the SIGUSR1 dump (item 4). |
| `autowisp/database/processing.py` | Promote `find_processing_outputs` to the base `ProcessingManager`, parameterized by a `_progress_model` class attribute and a `_progress_image_type(progress, db_session)` hook (item 2). |
| `autowisp/database/image_processing.py` | Drop the now-inherited `find_processing_outputs`; set `_progress_model = ImageProcessingProgress` and `_progress_image_type` → `progress.image_type.name` (item 2). |
| `autowisp/database/lightcurve_processing.py` | Set `_progress_model = LightCurveProcessingProgress` and `_progress_image_type` deriving the type from the single photref (as `_current_image_type` is set at run time); skip `set_pending` when `pipeline_run_id is None` so a review-only manager is cheap (item 2). |
| `autowisp/crash_report.py` | `find_error_progress` resolves against either progress table and `select_error_logs` instantiates the matching manager (item 2); `collect_provenance` records `resources` + reuses the shared `get_hostname`/`collect_environment` helpers (items 6, 7); sidecar gains the resolved step config + the progress timeline, and a `WorkerCrashedError` report bundles the whole batch (item 8). |
| `autowisp/exceptions.py` | New leaf provenance helpers `get_hostname` / `collect_environment` / `collect_resource_snapshot`, imported cycle-free by every layer (items 6, 7). |
| `autowisp/error_persistence.py` | `_write_sidecar` records `detail["environment"]` (crash-time, in the failing process) (item 7). |
| `autowisp/run_pipeline.py` | `PipelineRun.host = get_hostname()` (was `getfqdn()`), so the run host matches the report host (item 7). |

### Tests (Phase 9)

- A synthesised `WorkerCrashedError` (via `_worker_crashed` with an
  ambient step context) is a `step`-component error carrying `step_name`,
  and the phase-4 write populates the `error.step_name` column —
  regression test for the exact "no matching logs found" cause.
  *(Implemented: `test_worker_crashed_carries_step_name` in
  `test_error_context.py`; `test_worker_crashed_persists_step_name` in
  `test_error_persistence.py`.)*
- `find_error_progress` / `select_error_logs` resolve an **LC-step**
  error (e.g. `tfa`) to its `light_curve_processing_progress` row and its
  logs; an image-step error still resolves via `ImageProcessingProgress`;
  a pipeline/BUI error still yields none. *(Implemented:
  `test_resolves_lightcurve_step` in `test_crash_report.py`; image-step
  and stepless cases already covered there.)*
- `_WorkerEntry` writes the current item into the shared in-flight map
  while the callable runs and clears it on return (and on error), so a
  normal `run_pool` finishes with an empty map. *(Implemented:
  `test_inflight_map_tracks_then_clears_item`,
  `test_inflight_map_cleared_on_error`.)*
- A worker that hard-exits (`os._exit`) mid-task → the parent's
  `WorkerCrashedError` records the still-in-flight item(s) in
  `details["crashed_inputs"]` (a set bounded by `num_processes`), not a
  blind head sample. *(Implemented:
  `test_worker_crashed_names_inflight_input`.)*
- With `faulthandler` enabled in the worker bootstrap, a real `SIGSEGV` in
  a test worker writes a native traceback to its redirected stderr file
  (the log items 1–2 collect), so the faulted worker is distinguishable
  from executor-terminated innocents. *(Implemented:
  `test_faulthandler_dumps_native_traceback`.)*
- `build_crash_report` for a `WorkerCrashedError` includes the resolved
  step config, the progress timeline, the resource snapshot, and (when
  present) `exit_signal` — and, for the batch, every sibling error, not
  just the requested one.
- Provenance distinguishes run host/versions from report host/versions
  when they differ.
- A `run_pool` worker that raises for a given item produces an error
  carrying that item as a `related_file` of the call site's `FileKind`
  (and `_resolve_artifact_fks` sets the FK for an image/master item); a
  `WorkerCrashedError` likewise carries the in-flight item(s) as
  `related_files` (item 9). *(Implemented for the lightcurve path:
  `test_worker_error_carries_related_file`,
  `test_worker_crashed_promotes_related_files`, plus
  `_resolve_related_files` unit tests.)*

### Suggested sequencing

Items **1 + 2** are the highest-value, lowest-risk pair — together they
are the difference between a report with logs and one without — and are
**done**. **3** (in-flight map → candidate set) and **4** (native
self-report → the one culprit within it) are **coupled** — Item 3 narrows,
Item 4 isolates — and are now **done** together. **5–8** are enrichment
that make the report self-explaining without a maintainer hand-querying
the DB, and can follow independently; **5** (OS-level exit signal) and
**6** (memory snapshot) are **done** — together with items 4 (native dump)
they triangulate OOM vs. crash. **7** (crash-time provenance) is **done**
at reduced scope (the "different machine" premise was a `getfqdn` vs.
`gethostname` artifact, so the `PipelineRun` migration was dropped in
favour of host-consistency + crash-time env in the sidecar). **8** is
**done** at reduced scope too — only the resolved config is worth
recording (the rest of the original framing just duplicated the bundled
DB). **9** (populate `related_files`)
stands somewhat apart — it improves *every* error, not just crashes — and
is now **done for every step**, by two complementary routes: HDF5 products
(DR files, lightcurves) attach themselves whenever they are open, and the
steps explicitly scope what is not an HDF5 product (raw and calibrated
frames, masters, catalogs, the source-list filter, output products). The
`run_pool` classifiers stay regardless, since crash promotion has no open
file to draw on. What remains is the tier-3 list of auxiliaries.
