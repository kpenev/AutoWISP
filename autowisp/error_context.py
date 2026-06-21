"""Ambient error context and the capture boundaries that stamp it.

Phase 1 (:mod:`autowisp.exceptions`) gave each error fields for the
pipeline run, the raising worker, the step, and the related files. This
module is phase 2: it fills those fields in *without burdening every
raise site*. A call deep inside a step can ``raise SolveAstrometryError(
"no WCS solution")`` and have the step name, the pipeline-run snapshot,
and the files it was working on attached automatically by the time the
exception surfaces.

The ambient context is a single immutable :class:`ErrorContext` bundle
held in one ``contextvars.ContextVar``. A *single* var holding a *frozen*
object keeps related state cohesive (one ``get_error_context()`` returns
a consistent snapshot) while preserving exactly the contextvars
set/reset semantics — and thread/asyncio isolation — the scoping relies
on. See ``error_handling_plan.md`` (Phase 2) for the design.
"""

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
    CalibrationError,
    Component,
    CreateLightCurvesError,
    DetrendingStatError,
    EPDError,
    FindStarsError,
    FitMagnitudesError,
    FitPSFMapError,
    FitStarShapeError,
    MeasurePhotometryError,
    PipelineError,
    SolveAstrometryError,
    StackToMasterError,
    StepError,
    TFAError,
    ViewError,
    WorkerCrashedError,
)

git_id = "$Id$"


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

        related_files(tuple):    The ``RelatedFile`` entries in scope.

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
        ``parent_pid`` -- the key the parent threads in for workers (and
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


_context: contextvars.ContextVar = contextvars.ContextVar(
    "autowisp_error_context", default=ErrorContext()
)


def get_error_context() -> ErrorContext:
    """Return the current ambient :class:`ErrorContext`."""

    return _context.get()


def in_worker() -> bool:
    """Whether the current process is a multiprocessing worker."""

    return _context.get().in_worker


def set_error_context(ctx: ErrorContext) -> contextvars.Token:
    """Install ``ctx`` as the ambient context, returning the reset token."""

    return _context.set(ctx)


def set_pipeline_run(run: Optional[FrozenRow]) -> contextvars.Token:
    """Replace the bundle with a copy carrying ``run``, keeping the rest.

    Args:
        run(FrozenRow or None):    The pipeline-run snapshot to attach.

    Returns:
        contextvars.Token:    The reset token for the previous value.
    """

    current = _context.get()
    return _context.set(
        ErrorContext(
            pipeline_run=run,
            step_name=current.step_name,
            related_files=current.related_files,
            in_worker=current.in_worker,
        )
    )


@contextmanager
def error_context(*, step_name=None, related_files: Sequence = ()):
    """Scope additional context for any error raised inside the block.

    Builds a new :class:`ErrorContext` (step and files supplied at
    construction, not by mutating the current one), installs it for the
    duration of the block, and resets the token on exit.

    Args:
        step_name(str or None):    Override the ambient step name for the
            duration of the block.

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


# Maps a step name to its concrete StepError subclass, so an unknown
# exception raised inside a step is wrapped in a type catch sites can be
# specific about. Steps with no dedicated subclass fall back to
# ``StepError`` (still carrying ``step_name``).
_STEP_ERROR_BY_NAME = {
    "calibrate": CalibrationError,
    "stack_to_master": StackToMasterError,
    "stack_to_master_flat": StackToMasterError,
    "find_stars": FindStarsError,
    "solve_astrometry": SolveAstrometryError,
    "fit_star_shape": FitStarShapeError,
    "measure_aperture_photometry": MeasurePhotometryError,
    "fit_source_extracted_psf_map": FitPSFMapError,
    "fit_magnitudes": FitMagnitudesError,
    "create_lightcurves": CreateLightCurvesError,
    "epd": EPDError,
    "tfa": TFAError,
    "generate_epd_statistics": DetrendingStatError,
    "generate_tfa_statistics": DetrendingStatError,
}


def _stamp(exc: AutoWISPError) -> None:
    """Fill any unset context fields on ``exc`` from the ambient context.

    Already-populated fields are left untouched. This is the one place
    that writes ``step_name`` / ``related_files`` / ``pipeline_run`` /
    ``crashed`` after construction (phase 1 leaves them writable).

    Args:
        exc(AutoWISPError):    The exception to stamp in place.

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


def _wrap(exc: Exception, component: Component) -> AutoWISPError:
    """Wrap a non-AutoWISP exception in the right concrete class.

    Inside a step it becomes the step's :class:`StepError` subclass
    (looked up from the ambient step name); in the BUI it becomes a
    :class:`ViewError`; otherwise a :class:`PipelineError`. The original
    is preserved as ``__cause__`` by the caller (``raise ... from exc``).

    Args:
        exc(Exception):    The original, non-AutoWISP exception.

        component(Component):    Component of the wrapping callable.

    Returns:
        AutoWISPError:    The wrapping exception (not yet stamped).
    """

    message = str(exc) or exc.__class__.__name__
    if component is Component.STEP:
        step_name = get_error_context().step_name
        cls = _STEP_ERROR_BY_NAME.get(step_name, StepError)
        return cls(message, step_name=step_name)
    if component is Component.BUI:
        return ViewError(message)
    return PipelineError(message)


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
            except Exception as exc:  # pylint: disable=broad-except
                if not wrap_unknown:
                    raise
                wrapped = _wrap(exc, component)
                _stamp(wrapped)
                raise wrapped from exc

        return wrapper

    return decorate


def worker_entry(func, component: Component):
    """Wrap a Pool worker callable so errors return picklable + stamped.

    On the way out it stamps ``subprocess_id`` (the worker PID) and the
    ambient context, then re-raises, letting the Pool pickle the
    exception back to the parent. Non-:class:`AutoWISPError` exceptions
    are wrapped in a :class:`WorkerCrashedError` whose ``__cause__`` is
    the original and whose ``details`` carry the worker traceback (phase
    3 enriches this further).

    Args:
        func(Callable):    The worker callable to wrap.

        component(Component):    Component to assign when wrapping an
            unknown exception.

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
        except Exception as exc:  # pylint: disable=broad-except
            wrapped = WorkerCrashedError(
                str(exc) or exc.__class__.__name__,
                details={
                    "original_traceback": format_exc(),
                    "worker_component": component.value,
                },
            )
            wrapped.stamp_subprocess()
            _stamp(wrapped)
            raise wrapped from exc

    return wrapper
