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
  in phase 1 (`PipelineRunContext`, `code_version`) exists precisely so
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

1. **[Define the exception hierarchy.](#phase-1--exception-hierarchy)**
   ← *this document, current focus.*
2. [Define a context-collection mechanism](#phase-2--context-collection)
   (decorators / context managers) so steps and the pipeline driver
   automatically populate the exception fields without each call site
   having to remember.
3. [Define propagation across multiprocessing boundaries](#phase-3--propagation-across-process-boundaries)
   (worker stamps subprocess ID + host before re-raising; parent
   re-raises wrapped, preserving the original `__cause__`).
4. [Define the persistence layer](#phase-4--persistence): an `Error`
   table linked to `PipelineRun` + `Image` / `DRFile` / `Lightcurve` /
   `MasterFile` rows, plus a JSON blob for the structured ``details``.
5. [Define the user-facing rendering](#phase-5--user-facing-rendering-bui--cli):
   BUI views and CLI formatter, both reading from the structured record
   (other channels such as an email/Slack notifier come later).
6. Define a **[crash-report bundler](#phase-6--crash-report-bundler-early-notes)**:
   on demand, collect everything needed to debug a failure into a single
   zip the user can send to the maintainers — the `Error` row(s) +
   sidecar(s), the relevant per-process logs/stdout-stderr, the
   configuration snapshot, the `code_version`, and environment/provenance
   — with a scrubbing pass for credentials.
7. Migrate existing call sites to raise the new exception types and
   delete the legacy ad-hoc classes. *(section pending)*

Phases 2–7 will get their own sections in this file as we get to them.

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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Sequence


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


@dataclass(frozen=True)
class PipelineRunContext:
    """Snapshot of the ``PipelineRun`` row at the moment of failure.

    Snapshotted rather than holding the ORM instance directly so the
    exception remains useful after the session is closed.

    Attributes:
        pipeline_run_id(int or None):    The ``PipelineRun.id`` this
            failure belongs to, or ``None`` for standalone ``wisp-*``
            invocations.

        host(str):    Fully qualified host name the run executed on.

        process_id(int):    PID of the main pipeline process.

        started(datetime or None):    When the run started.

        crashed(datetime or None):    When the failure surfaced, filled
            in by the top-level handler.

        code_version(str):    Git hash of the working tree that produced
            the failure (HEAD SHA, suffixed ``:dirty`` for an unclean
            tree), from ``get_code_version_str()``. Identifies the whole
            codebase so a developer can check out the exact code without
            any per-file read.
    """

    pipeline_run_id: Optional[int]
    host: str
    process_id: int
    started: Optional[datetime]
    crashed: Optional[datetime]
    code_version: str = ""


def _rebuild_autowisp_error(cls, state):
    """Reconstruct an :class:`AutoWISPError` subclass for unpickling.

    Bypasses ``__init__`` (so required keyword-only arguments on
    subclasses do not block unpickling) and restores the instance
    ``__dict__`` directly. See :meth:`AutoWISPError.__reduce__`.

    Args:
        cls(type):    The concrete exception class to rebuild.

        state(dict):    The instance ``__dict__`` captured at pickle
            time.

    Returns:
        AutoWISPError:    The reconstructed exception.
    """

    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


class AutoWISPError(Exception):
    """Base class for every AutoWISP-raised exception.

    Every concrete subclass selects a :class:`Component`.

    Attributes:
        component(Component):    Set on each subclass; verified by tests.

        related_files(tuple):    The :class:`RelatedFile` entries this
            error is about.

        pipeline_run(PipelineRunContext or None):    Snapshot set by the
            pipeline driver when it wraps a step's exception, or by
            ``wisp-*`` entry points.

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
        pipeline_run: Optional[PipelineRunContext] = None,
        subprocess_id: Optional[int] = None,
        user_message: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Store the context attributes (see class ``Attributes``)."""

        super().__init__(message)
        self.related_files = tuple(related_files)
        self.pipeline_run = pipeline_run
        self.subprocess_id = subprocess_id
        self.user_message = user_message or message
        self.details = dict(details or {})

    def __reduce__(self):
        """Pickle by restoring ``__dict__`` rather than re-running
        ``__init__``.

        Subclasses (e.g. :class:`StepError`) take required keyword-only
        arguments, so the default exception unpickler — which calls
        ``cls(*self.args)`` — would raise ``TypeError``. Reconstructing
        through ``__new__`` + ``__dict__`` keeps every field intact and
        lets the exception travel back out of a multiprocessing Pool
        (see phase 3).

        Returns:
            tuple:    ``(callable, args)`` per the pickle protocol.
        """

        return (_rebuild_autowisp_error, (type(self), self.__dict__.copy()))

    def stamp_subprocess(self) -> None:
        """Record the current PID as the raising sub-process.

        Called by the worker before re-raising out of a multiprocessing
        Pool. Idempotent: a value already set in a deeper worker wins.

        Returns:
            None
        """

        if self.subprocess_id is None:
            self.subprocess_id = os.getpid()

    def with_pipeline_run(self, ctx: PipelineRunContext) -> "AutoWISPError":
        """Attach a :class:`PipelineRunContext` snapshot.

        Args:
            ctx(PipelineRunContext):    The snapshot to attach. If its
                ``host`` is blank it is filled from the current host and
                ``crashed`` is set to now.

        Returns:
            AutoWISPError:    ``self``, so it can be used inline before
                re-raising.
        """

        # Snapshot host even if ctx omitted it.
        if ctx.host == "":
            ctx = PipelineRunContext(
                pipeline_run_id=ctx.pipeline_run_id,
                host=socket.gethostname(),
                process_id=ctx.process_id,
                started=ctx.started,
                crashed=ctx.crashed or datetime.utcnow(),
                code_version=ctx.code_version,
            )
        self.pipeline_run = ctx
        return self
```

### Component bases and subclasses

```python
class StepError(AutoWISPError):
    component = Component.STEP

    def __init__(self, message: str, *, step_name: str, **kwargs):
        super().__init__(message, **kwargs)
        self.step_name = step_name


class PipelineError(AutoWISPError):
    component = Component.PIPELINE


class BUIError(AutoWISPError):
    component = Component.BUI
```

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
class WorkerCrashedError(PipelineError):
    """Re-raise wrapper used when a multiprocessing worker dies in a
    way that does not preserve the original exception (segfault,
    OOM-killer)."""
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

### How PipelineRunContext gets attached

This is mostly the job of phase 2, but the hierarchy needs to support
it now:

- `run_pipeline.py` constructs a `PipelineRunContext` from the
  `PipelineRun` ORM row at startup, snapshotting `host`, `process_id`,
  `started`. It stashes it on a `contextvars.ContextVar` so any code
  on the call stack can grab it.
- A top-level `except AutoWISPError as exc:` in `run_pipeline.main`
  calls `exc.with_pipeline_run(get_pipeline_run_context())` if the
  field is unset, fills in `crashed=datetime.utcnow()`, and re-raises.
- `wisp-*` CLI entry points do the same but with `pipeline_run_id =
  None`.

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
  - `related_files`, `pipeline_run`, `subprocess_id`,
    `user_message`, `details` round-trip through
    pickling (needed for Pool propagation).
  - `stamp_subprocess()` is idempotent and sets `subprocess_id` to
    `os.getpid()` when previously `None`.
  - `with_pipeline_run()` snapshots `host` when blank.

## Open questions to revisit before phase 2

- ~~Should `RelatedFile` carry a hash or only a path?~~ **Resolved:
  path only.** Codebase identity is captured once per run as a git
  hash (`code_version` on :class:`PipelineRunContext`, via
  `get_code_version_str()` in `multiprocessing_util.py`), which
  identifies the *entire* working tree — including its dirty state —
  without an extra per-file read. Per-file content hashes would cost a
  read each and still not capture the code that produced the file, so
  they are dropped.
- **`details` size vs. retention (these two questions are connected).**
  For debugging it is genuinely useful to capture rich context on a
  failure — and for BUI errors specifically, the full request context
  (path, view, query/POST keys, session ID) would help a lot. But that
  makes a single `details` payload potentially large, and an `Error`
  table that keeps every such payload *forever* will balloon the SQLite
  file. The resolution combines three levers, to be finalized in
  phases 4–5:
  - **Bounded persistence.** Error rows (or at least their heavy
    payloads) are retained for a limited period rather than
    indefinitely.
  - **Keep the heavy part separate.** Store the small, queryable fields
    (component, step, artifact FKs, `user_message`, timestamps) inline
    in the table, but spill the large `details`/request snapshot to a
    sidecar artifact (file or a separate, prunable blob), referenced by
    the row — so routine queries and the BUI list stay cheap and the
    bulk can be aged out or dropped independently.
  - **User-driven cleanup.** Provide an explicit mechanism for users to
    purge old error info (CLI command and/or BUI action), so retention
    is not solely automatic.
  - *Implication for phase 5:* BUI request snapshots are worth
    capturing **into the sidecar `details`, not inline**, precisely
    because they are the largest contributor to row size.

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

Holds the ambient context in `contextvars` plus helpers to set, read,
and scope it.

```python
import contextvars
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Sequence

from autowisp.exceptions import (
    AutoWISPError,
    Component,
    PipelineRunContext,
    RelatedFile,
    StepError,
    WorkerCrashedError,
)

_pipeline_run: contextvars.ContextVar[Optional[PipelineRunContext]] = (
    contextvars.ContextVar("autowisp_pipeline_run", default=None)
)
_step_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "autowisp_step_name", default=None
)
_related_files: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "autowisp_related_files", default=()
)


def get_pipeline_run_context() -> Optional[PipelineRunContext]:
    return _pipeline_run.get()


def get_step_name() -> Optional[str]:
    return _step_name.get()


def get_related_files() -> tuple:
    return _related_files.get()
```

#### Establishing pipeline-run context

Two entry points set it once, near the top of the call stack:

- **`run_pipeline.main`** already builds the `PipelineRun` row. Right
  after the commit, build the snapshot and stash it (and put its parts
  into the processing config so workers can rebuild it):

  ```python
  ctx = PipelineRunContext(
      pipeline_run_id=pipeline_run,
      host=getfqdn(),
      process_id=os.getpid(),
      started=started,
      crashed=None,
      code_version=get_code_version_str(),
  )
  set_pipeline_run_context(ctx)
  ```

- **`wisp-*` CLI entry points** (steps run standalone) set the same
  thing with `pipeline_run_id=None`. A tiny decorator
  `@cli_entry_point(component=...)` wraps each `main()` and does this
  before dispatch, then runs the capture/handler on the way out (see
  below).

```python
def set_pipeline_run_context(ctx: PipelineRunContext) -> contextvars.Token:
    return _pipeline_run.set(ctx)


def pipeline_run_context_from_config(config: dict) -> Optional[PipelineRunContext]:
    """Rebuild the run snapshot inside a freshly-started process.

    Args:
        config(dict):    The per-process config dict, carrying the
            pipeline-run keys threaded through by the parent.

    Returns:
        PipelineRunContext or None:    The rebuilt snapshot, or ``None``
            when the keys are absent (e.g. a unit test calling a step
            directly).
    """

    if "pipeline_run_id" not in config:
        return None
    return PipelineRunContext(
        pipeline_run_id=config["pipeline_run_id"],
        host=config.get("host") or socket.gethostname(),
        process_id=os.getpid(),
        started=config.get("pipeline_started"),
        crashed=None,
        code_version=config.get("code_version") or get_code_version_str(),
    )
```

#### Hooking the existing bootstrap

`setup_process_map()` gains a few lines at the end (it already has
`project_home`, `processing_step`, etc. in `config`):

```python
    # ... existing logging / IO / project-home setup ...
    ctx = pipeline_run_context_from_config(config)
    if ctx is not None:
        set_pipeline_run_context(ctx)
    if config.get("processing_step") not in (None, "none", "init_processing"):
        _step_name.set(config["processing_step"])
```

Because `setup_process` runs in every worker, this single change gives
both the main process and the workers their pipeline-run + step context
for free. `run_pipeline.main` / the CLI decorator additionally set the
`PipelineRunContext` directly (so context exists even before the first
`setup_process` call).

#### Scoping finer context: `error_context`

Inside a step, the input/output filenames are known per image. A
context manager pushes them so a deeper raise picks them up, and pops
on exit:

```python
@contextmanager
def error_context(*, step_name=None, related_files: Sequence[RelatedFile] = ()):
    """Scope additional context for any error raised inside the block.

    Args:
        step_name(str or None):    Override the ambient step name for
            the duration of the block.

        related_files(Sequence[RelatedFile]):    Files to append to the
            ambient related-files list; removed again on exit.

    Yields:
        None
    """

    tokens = []
    if step_name is not None:
        tokens.append((_step_name, _step_name.set(step_name)))
    if related_files:
        merged = get_related_files() + tuple(related_files)
        tokens.append((_related_files, _related_files.set(merged)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
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

    if isinstance(exc, StepError) and not getattr(exc, "step_name", None):
        exc.step_name = get_step_name()
    if not exc.related_files:
        exc.related_files = get_related_files()
    if exc.pipeline_run is None:
        ctx = get_pipeline_run_context()
        if ctx is not None:
            exc.with_pipeline_run(ctx)
```

`_wrap` maps an unknown exception to the right concrete class: inside a
step it becomes the step's `StepError` subclass (looked up from
`component`/`get_step_name()`); in the orchestration layer it becomes a
`PipelineError`. The original is preserved as `__cause__` — never
swallowed.

Note `_stamp` mutates `exc.step_name` / `exc.related_files` /
`exc.pipeline_run`. Phase 1 declares these `__init__`-assigned instance
attributes (not the frozen dataclasses), so they remain writable; only
the `RelatedFile` / `PipelineRunContext` payloads are frozen. This is
the one place that writes them post-construction.

### Worker propagation hook

Phase 3 owns full multiprocessing propagation, but Phase 2 defines the
hook the `Pool` sites will adopt, so we can retire the current
stringify-to-`RuntimeError` workarounds in `epd_correction.py`,
`iterative_refit.py`, and `apply_correction.py`. A worker entry point is
wrapped with:

```python
def worker_entry(func, component: Component):
    """Wrap a Pool worker callable so errors return picklable + stamped.

    Args:
        func(Callable):    The worker callable to wrap.

        component(Component):    Component to assign when wrapping an
            unknown exception in a :class:`WorkerCrashedError`.

    Returns:
        Callable:    The wrapped callable, suitable to hand to a Pool.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AutoWISPError as exc:
            exc.stamp_subprocess()
            _stamp(exc)
            raise
        except Exception as exc:  # noqa: BLE001
            wrapped = WorkerCrashedError(str(exc), ...)  # phase 3 details
            wrapped.stamp_subprocess()
            _stamp(wrapped)
            raise wrapped from exc
    return wrapper
```

This is safe to pickle back to the parent precisely because Phase 1
made every field round-trip through pickle. The parent's
`capture_errors` then sees an already-stamped `AutoWISPError` and only
fills the pipeline-run snapshot it has locally.

### What changes, concretely

| File | Change |
| ---- | ------ |
| `autowisp/error_context.py` | **New.** ContextVars (incl. an `in_worker` flag), getters/setters, `pipeline_run_context_from_config`, `error_context`, `capture_errors`, `worker_entry`. |
| `autowisp/multiprocessing_util.py` | `setup_process_map` sets step + pipeline-run context from `config` **and the `in_worker` flag**; thread `pipeline_run_id` / `host` / `pipeline_started` / `code_version` into the default keys (the parent computes `code_version` once so workers reuse it instead of each shelling out to git). |
| `autowisp/run_pipeline.py` | Set `PipelineRunContext` after creating the `PipelineRun` row; fill `crashed` in the top-level handler. |
| `autowisp/database/processing.py`, `image_processing.py` | Pass pipeline-run keys into the config dicts handed to `setup_process`; wrap per-image dispatch in `error_context` using the already-computed input/output paths. |
| each `processing_steps/*.py` `main()` | Decorate with `@capture_errors(component=Component.STEP)` (and `@cli_entry_point` for the standalone path). |
| Pool worker sites (`epd_correction.py`, `iterative_refit.py`, `apply_correction.py`, and others) | Begin replacing bespoke try/except-stringify with `worker_entry`; full inventory and the Process+Queue case (`solve_astrometry.py`) are handled in phase 3. |

### Tests (Phase 2)

- A step `main()` that raises a bare `ValueError` surfaces as the
  correct `StepError` subclass with `step_name` set and `__cause__`
  preserved.
- With a `PipelineRunContext` set in the ambient context, a raised
  `AutoWISPError` comes out carrying it; with none set (unit-test
  path), `pipeline_run` stays `None` and nothing crashes.
- `error_context(related_files=...)` nesting attaches/detaches
  correctly and the innermost set wins for `step_name`.
- A `setup_process` call rebuilds context from a config dict (simulating
  a worker) and a subsequent raise is fully stamped.
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

- **The type and all phase-1 fields.** This is *not* automatic. A
  subclass such as `StepError` has a required keyword-only `step_name`
  in its `__init__`, but the default exception unpickler reconstructs
  via `cls(*self.args)` — which would call `StepError(message)` and
  raise `TypeError`. Phase 1's base class handles this with the
  `__reduce__` / `_rebuild_autowisp_error` pair defined above, which
  reconstructs through `__new__` + `__dict__` instead of `__init__`.
  Phase 1's pickling test (`tests/test_exception_hierarchy.py`) is what
  guards this; Phase 3 depends on it.

- **The `__cause__` chain.** When an unknown exception is wrapped in a
  `WorkerCrashedError`, the original may itself be unpicklable (e.g. a
  third-party C-extension error). So the wrapper stores a *formatted
  string* of the original (`traceback.format_exc()`) in
  `details["original_traceback"]` and only keeps a picklable cause —
  never a live reference to something that may fail to pickle and turn
  the whole error into an opaque `PicklingError`. This matters for
  *both* schemes: a `Queue.put` of an unpicklable object fails just as
  a Pool re-raise does.

- **The traceback.** In Scheme A, `multiprocessing` attaches the
  worker's traceback to the re-raised exception as a `RemoteTraceback`
  (`__cause__`) automatically. Scheme B has no such machinery — the
  worker is the one calling `Queue.put`, so it must capture the
  traceback string itself (into `details`). The stamping helpers below
  do this so both schemes end up with the worker traceback visible in
  the parent.

### Requirement 2 — stamp before it leaves the worker (both schemes)

The stamping logic is identical; only *where it runs* differs.

- **Scheme A (Pool):** wrap the mapped callable with `worker_entry`
  (phase 2). On the way out it calls `stamp_subprocess()` (idempotent)
  and `_stamp()`, then **re-raises**, letting Pool pickle it.

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

- **Scheme A:** on supported Python a broken pool surfaces as an error
  from the `.map()` call (older versions could hang). Phase 3 wraps the
  parent-side `.map()` / `.imap*()` in a helper `run_pool` that, on such
  a failure, synthesises a `WorkerCrashedError` (component `PIPELINE`)
  carrying what the *parent* knows — step, the batch of inputs in
  flight, pipeline-run context — plus any OS-level clue (exit
  code/signal).

- **Scheme B:** `manage_astrometry` *already* detects this
  (`not any(p.is_alive())`) and raises a bare `RuntimeError`. Phase 3
  upgrades that to a `WorkerCrashedError` and reads `process.exitcode`
  (negative → killed by signal `-exitcode`) for the OS clue.

This "the worker cannot describe its own death, so the parent owns it"
case is the one place the parent synthesises the error in both schemes.

> Note: `concurrent.futures.ProcessPoolExecutor` reports the Scheme-A
> case more cleanly (`BrokenProcessPool`). We keep `Pool`/`Process` to
> minimise churn at the existing call sites and just detect + wrap;
> `run_pool` / the `manage_astrometry` loop are the single places that
> would change if we ever switch.

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
| `iterative_refit.py` | A | Bare `pool_magfit` mapped. | Wrap `pool_magfit` in `worker_entry`; map via `run_pool`. |
| `apply_correction.py` | A | Bare `correct` mapped. | Wrap `correct` in `worker_entry`; map via `run_pool`. |
| `find_stars.py`, `fit_star_shape.py`, `measure_aperture_photometry.py` | A | Bare callable mapped (already use `setup_process_map` initializer). | Wrap callable in `worker_entry`; map via `run_pool`. |
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

The git hash lives on :class:`PipelineRunContext` (phase 1 / phase 2).
To survive past the in-memory snapshot, add
`code_version = Column(String(...), nullable=True)` to the `PipelineRun`
model and populate it in `run_pipeline.main` where the row is created
(the value is already on hand from `get_code_version_str()`). The
`Error` table references the run rather than duplicating the hash per
error.

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
- `created` — timestamp (the `crashed` time from the run context).
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
- **Log cross-link**: the existing `review` / `review_single` log pages
  already show per-process logs; the error detail links to the matching
  log (selected by the failure's `subprocess_id` / run / time), reusing
  that machinery instead of duplicating it.

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
| `autowisp/error_render.py` | **New.** `error_summary`, `error_detail`, remediation lookup. |
| `wisp-*` entry decorator, `run_pipeline.py` | Render via `error_render` on the way out; set exit code. |
| `browser_interface/processing/views.py` (+ urls, templates) | **New** error list + detail views and templates. |
| `browser_interface/processing/progress_view.py` (+ template) | Link failed status cells to the matching error. |
| `browser_interface/.../middleware.py` | **New.** Capture request context + route `BUIError` through `persist_error`. |

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

## Phase 6 — crash-report bundler (early notes)

When the structured detail is still not enough — or the user simply
wants to hand the problem off — they should be able to produce a single
self-contained zip to send to the maintainers, with no manual
file-hunting across the per-PID logs.

### Trigger

- CLI: `wisp-crash-report <error_id> [--out report.zip]` (and
  `--last` for the most recent error).
- BUI: a "Download crash report" button on an error's detail view.

Both call one `build_crash_report(error_id, out_path)`.

### Contents

Everything needed to reproduce/diagnose, gathered from sources the
earlier phases already populate:

- The `Error` row(s) — serialized to JSON — and their **sidecar**
  file(s) (phase 4). For a `WorkerCrashedError`, include the whole
  failed batch's errors, not just one.
- The relevant **logs / stdout-stderr**: the per-process
  `{task}_{now}_{pid}.outerr` and `.log` files written by
  `setup_process_map` (`multiprocessing_util.py`), selected by the
  failure's `pipeline_run_id` / `subprocess_id` / time window rather
  than dumping the whole log directory.
- The **configuration snapshot** in effect for the run/step.
- **Provenance**: `code_version` (git hash), Python / OS / key package
  versions (`astrowisp`, `numpy`), hostname.
- A short **`manifest.json`** describing what was collected and the
  report's own schema version.

### Credential scrubbing (required)

The config and logs can contain secrets — e.g. `gaia_user` /
`gaia_password` are threaded through the process config
(`multiprocessing_util.py`). The bundler runs a scrubbing pass over
every text artifact before zipping, redacting known secret keys and
anything matching credential-like patterns. This pass is mandatory and
tested; nothing enters the zip unscrubbed.

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
| `autowisp/crash_report.py` | **New.** `build_crash_report()`, log selection, scrubber. |
| `pyproject.toml` | Add the `wisp-crash-report` script. |
| BUI error detail view | "Download crash report" action calling the same builder. |

### Tests (Phase 6)

- `build_crash_report()` produces a zip containing the error JSON,
  sidecar, the matching (and only the matching) logs, config, and
  `manifest.json`.
- The scrubber removes `gaia_password` and pattern-matched secrets from
  every artifact in the zip.
- A missing log / sidecar is recorded as a manifest gap, and the report
  still builds.
