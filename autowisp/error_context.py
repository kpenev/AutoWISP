"""Ambient error context and the capture boundaries that stamp it.

:class:`~autowisp.exceptions.AutoWISPError` carries fields for the
pipeline run, the raising worker, the step, and the related files. This
module fills those fields in *without burdening every raise site*: a call
deep inside a step can ``raise SolveAstrometryError("no WCS solution")``
and have the step name, the pipeline-run snapshot, and the files it was
working on attached automatically by the time the exception surfaces.

The ambient context is a single immutable :class:`ErrorContext` bundle
held in one ``contextvars.ContextVar``. A *single* var holding a *frozen*
object keeps related state cohesive (one ``get_error_context()`` returns
a consistent snapshot) while preserving exactly the contextvars
set/reset semantics -- and thread/asyncio isolation -- the scoping relies
on.
"""

import contextvars
import functools
import os
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
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
            used by the nested-worker guard.
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


def _stamp(exc: AutoWISPError) -> None:
    """Fill any unset context fields on ``exc`` from the ambient context.

    Already-populated fields are left untouched. This is the one place
    that writes ``step_name`` / ``related_files`` / ``pipeline_run`` /
    ``crashed`` after construction (they are mutable instance attributes).

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

    # A step name's concrete StepError subclass, so an unknown exception
    # raised inside a step is wrapped in a type catch sites can be specific
    # about. Steps with no dedicated subclass fall back to ``StepError``
    # (still carrying ``step_name``).
    step_error_by_name = {
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

    message = str(exc) or exc.__class__.__name__
    if component is Component.STEP:
        step_name = get_error_context().step_name
        cls = step_error_by_name.get(step_name, StepError)
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


def _stamp_worker_error(exc: Exception, component: Component) -> AutoWISPError:
    """Turn an error raised in a worker into a stamped, picklable one.

    Shared by :func:`worker_entry` (Scheme A: ``Pool``, which re-raises)
    and :func:`capture_for_queue` (Scheme B: ``Process`` + ``Queue``,
    which puts the returned object on a queue). An :class:`AutoWISPError`
    is stamped in place; any other exception is wrapped via :func:`_wrap`
    into the step's concrete :class:`StepError` subclass (so a worker's
    bare ``ValueError`` surfaces as e.g. ``FindStarsError``), *not* a
    :class:`WorkerCrashedError` -- that type is reserved for a worker that
    dies without producing an error object at all (synthesised by the
    parent).

    The worker traceback is captured into ``details["original_traceback"]``
    because it is the only durable record that crosses back: Scheme A's
    ``RemoteTraceback`` lives only on the live re-raised object, Scheme B
    has none, and neither transport pickles ``__cause__``.

    Args:
        exc(Exception):    The exception raised in the worker.

        component(Component):    Component of the worker callable, used to
            pick the wrapper class for a non-AutoWISP exception.

    Returns:
        AutoWISPError:    The stamped exception, safe to pickle.
    """

    stamped = exc if isinstance(exc, AutoWISPError) else _wrap(exc, component)
    stamped.stamp_subprocess()
    _stamp(stamped)
    stamped.details.setdefault("original_traceback", format_exc())
    return stamped


class _WorkerEntry:  # pylint: disable=too-few-public-methods
    """Picklable wrapper that stamps errors leaving a Pool worker.

    Scheme A (``Pool`` + ``map``/``imap``): on the way out an error is
    stamped with ``subprocess_id`` + ambient context (see
    :func:`_stamp_worker_error`) and re-raised, letting the Pool pickle it
    back to the parent.

    This is a class, not a closure, because ``Pool.map`` pickles the
    mapped callable to send it to the worker (under both ``fork`` and
    ``spawn``); a closure is not picklable, whereas an instance holding a
    picklable ``func`` (e.g. a ``functools.partial`` of a module-level
    function) and an enum ``component`` is.

    Attributes:
        func(Callable):    The wrapped per-item worker callable.

        component(Component):    Component for wrapping unknown errors.
    """

    def __init__(self, func, component: Component):
        self.func = func
        self.component = component

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            stamped = _stamp_worker_error(exc, self.component)
            if stamped is exc:
                raise
            raise stamped from exc


def worker_entry(func, component: Component):
    """Wrap a Pool worker callable so errors come back picklable + stamped.

    Args:
        func(Callable):    The worker callable to wrap (must itself be
            picklable, e.g. a module-level function or a ``partial`` of
            one).

        component(Component):    Component to assign when wrapping an
            unknown exception.

    Returns:
        _WorkerEntry:    A picklable callable suitable to hand to a Pool.
    """

    return _WorkerEntry(func, component)


def capture_for_queue(exc: Exception, *, component: Component) -> AutoWISPError:
    """Stamp a worker error and return it for a result queue (Scheme B).

    Sibling of :func:`worker_entry` for ``Process`` + ``Queue`` workers
    that catch and *return* their error (to ``result_queue.put(...)``)
    rather than re-raising it. Performs the same stamping + traceback
    capture and returns the picklable exception.

    Args:
        exc(Exception):    The exception raised in the worker.

        component(Component):    Component of the worker callable.

    Returns:
        AutoWISPError:    The stamped exception, safe to put on a queue.
    """

    return _stamp_worker_error(exc, component)


def reraise_from_worker(exc: AutoWISPError) -> None:
    """Re-raise in the parent an error pulled off a worker result queue.

    Fills the pipeline-run snapshot from the parent's ambient context if
    the worker did not already carry one, then raises. The error then
    flows up to the parent's ``capture_errors`` boundary like any other.

    Args:
        exc(AutoWISPError):    The stamped exception from the queue.

    Returns:
        None
    """

    if isinstance(exc, AutoWISPError) and exc.pipeline_run is None:
        ctx = get_error_context()
        if ctx.pipeline_run is not None:
            exc.with_pipeline_run(ctx.pipeline_run)
    raise exc


def forbid_nested_workers() -> None:
    """Enforce the no-nested-workers policy (resource control).

    Every parallel site is sized by ``num_parallel_processes``; a worker
    that spawned its own pool/process would multiply that out to ``N^2``
    live processes. Called before any worker launch so an accidental
    nested launch fails loudly instead of silently blowing the limit.

    Returns:
        None
    """

    if in_worker():
        raise PipelineError(
            "Nested multiprocessing is not allowed: a worker attempted to "
            "create its own pool/process, which would multiply "
            "num_parallel_processes out to N^2 live processes."
        )


def _worker_crashed(items, exc: Exception) -> "WorkerCrashedError":
    """Synthesise the parent-side error for a worker that died silently.

    Used when a worker dies without producing an error object (segfault,
    OOM-killer, ``os._exit``), so the parent must describe the failure
    from what *it* knows: the step, the in-flight inputs, and the
    underlying pool error.

    Args:
        items:    The work items that were in flight.

        exc(Exception):    The error the pool surfaced for the death.

    Returns:
        WorkerCrashedError:    Stamped with the ambient context.
    """

    ctx = get_error_context()
    err = WorkerCrashedError(
        f"A worker process died during step {ctx.step_name!r} without "
        f"reporting an error ({exc!r}).",
        step_name=ctx.step_name,
    )
    # ``step_name`` also goes in ``details`` so it survives into the
    # sidecar even if the queryable column is ever dropped; the attribute
    # above is what the persistence layer writes to ``error.step_name``,
    # which crash-report log-collection resolves the run/step logs from.
    err.details["step_name"] = ctx.step_name
    err.details["pool_error"] = repr(exc)
    try:
        items_list = list(items)
        err.details["num_inputs"] = len(items_list)
        err.details["inputs_sample"] = [repr(i) for i in items_list[:20]]
    except Exception:  # pylint: disable=broad-except
        pass
    _stamp(err)
    return err


def _stream_as_completed(executor, wrapped, items):
    """Yield worker results as they finish (unordered streaming).

    The ``ProcessPoolExecutor`` analogue of ``Pool.imap_unordered``:
    submit every item, then surface results via ``as_completed`` so a
    consumer can process them lazily. ``future.result()`` re-raises a
    worker error (a stamped :class:`AutoWISPError`) or, on a worker death,
    a ``BrokenProcessPool`` -- both then handled by :func:`run_pool`.

    Args:
        executor(ProcessPoolExecutor):    The live executor.

        wrapped(Callable):    The :func:`worker_entry`-wrapped worker.

        items(iterable):    Work items to submit.

    Yields:
        The return value of ``wrapped`` for each item, in completion
        order.
    """

    futures = [executor.submit(wrapped, item) for item in items]
    for future in as_completed(futures):
        yield future.result()


# The keyword-only options each map an existing call-site knob; a config
# object would just be a thin shim over the same set.
# pylint: disable=too-many-arguments
def run_pool(
    worker,
    items,
    *,
    config,
    num_processes,
    component: Component = Component.STEP,
    max_tasks_per_child=None,
    stream_consumer=None,
):
    """Map ``worker`` over ``items`` in a process pool, stamping errors.

    Single entry point for the ``Pool``-style parallel sites. It enforces
    the no-nested-workers policy, bootstraps each worker with
    ``setup_process_map``, wraps ``worker`` with :func:`worker_entry` so
    any error is stamped + picklable before it crosses back, and
    synthesises a :class:`WorkerCrashedError` if a worker dies without
    surfacing one.

    Built on :class:`concurrent.futures.ProcessPoolExecutor` rather than
    ``multiprocessing.Pool`` specifically so that a worker that dies
    mid-task (segfault / OOM-killer / ``os._exit``) raises
    ``BrokenProcessPool`` instead of hanging the pipeline forever -- the
    silent-death case ``Pool`` cannot report.

    Args:
        worker(Callable):    The per-item callable (already bound, e.g.
            via ``functools.partial``); must be picklable.

        items(iterable):    Work items to map over.

        config(dict):    Per-process config passed to
            ``setup_process_map``; ``parent_pid`` is set here so workers
            know they are workers.

        num_processes(int):    Number of worker processes.

        component(Component):    Component for wrapping unknown errors.

        max_tasks_per_child(int or None):    Recycle each worker after
            this many tasks (memory control); ``None`` keeps workers for
            the whole run.

        stream_consumer(Callable or None):    If given, it is called with
            an iterator yielding results as they complete (consumed inside
            the pool block) instead of returning a materialised, ordered
            result list.

    Returns:
        list or None:    The ordered results, or ``None`` when a
            ``stream_consumer`` is used.
    """

    forbid_nested_workers()
    # Lazy import breaks a genuine cycle: multiprocessing_util imports
    # this module for the setup_process_map bootstrap hook.
    # pylint: disable=import-outside-toplevel
    from autowisp.multiprocessing_util import setup_process_map

    # pylint: enable=import-outside-toplevel

    config["parent_pid"] = os.getpid()
    wrapped = worker_entry(worker, component)
    executor_kwargs = {
        "max_workers": num_processes,
        "initializer": setup_process_map,
        "initargs": (config,),
    }
    if max_tasks_per_child is not None:
        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

    try:
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            if stream_consumer is None:
                return list(executor.map(wrapped, items))
            stream_consumer(_stream_as_completed(executor, wrapped, items))
            return None
    except AutoWISPError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise _worker_crashed(items, exc) from exc
