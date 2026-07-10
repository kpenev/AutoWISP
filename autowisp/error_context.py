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
import signal
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Manager
from traceback import format_exc
from typing import Optional, Sequence

from autowisp.database.frozen_row import FrozenRow
from autowisp.miscellaneous import (
    collect_resource_snapshot,
    get_code_version_str,
)
from autowisp.exceptions import (
    AutoWISPError,
    CalibrationError,
    Component,
    CreateLightCurvesError,
    DetrendingStatError,
    EPDError,
    FileKind,
    FindStarsError,
    FitMagnitudesError,
    FitPSFMapError,
    FitStarShapeError,
    MeasurePhotometryError,
    PipelineError,
    RelatedFile,
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


def _resolve_related_files(related_files, item):
    """Build the :class:`RelatedFile`\\ s for a work item, best-effort.

    ``related_files`` is the call site's classifier for the items it maps
    over -- either a :class:`FileKind` (the item is a path) or a callable
    ``item -> RelatedFile | Iterable[RelatedFile] | None`` (for items that
    are not a bare path, e.g. an image set). Never raises: a classifier
    that does not fit the item simply yields no related files, so error
    handling is never itself a source of errors.

    Args:
        related_files(FileKind, Callable, or None):    The classifier, or
            ``None`` to attach nothing.

        item:    The work item handed to the worker.

    Returns:
        tuple:    Zero or more :class:`RelatedFile` entries.
    """

    if related_files is None or item is None:
        return ()
    try:
        if isinstance(related_files, FileKind):
            return (RelatedFile(related_files, item, role="input"),)
        result = related_files(item)
    except Exception:  # pylint: disable=broad-except
        return ()
    if result is None:
        return ()
    if isinstance(result, RelatedFile):
        return (result,)
    try:
        return tuple(result)
    except TypeError:
        return ()


class _WorkerEntry:  # pylint: disable=too-few-public-methods
    """Picklable wrapper that stamps errors leaving a Pool worker.

    Scheme A (``Pool`` + ``map``/``imap``): on the way out an error is
    stamped with ``subprocess_id`` + ambient context (see
    :func:`_stamp_worker_error`) and re-raised, letting the Pool pickle it
    back to the parent.

    Around the wrapped call it does two things with the item:

    - **In-flight tracking.** The item is recorded in the shared in-flight
      map (``{pid: item}``) and cleared on return. The executor never
      records which worker is running which item -- workers self-pull, the
      parent only hears back on *completion*, and a broken pool collapses
      every pending future to the same ``BrokenProcessPool`` -- so this
      map is the only place the culprit input of a silent death can be
      recovered from. A hard ``os._exit`` (segfault/OOM) skips the
      ``finally``, leaving the culprit behind, which is exactly the case
      we need it for.
    - **Related-file context.** The item is scoped as the ambient
      ``related_files`` (via ``related_files``, the call site's classifier),
      so *any* error the callable raises -- e.g. a config-vs-file-content
      mismatch deep inside the step -- carries the file it was about, which
      then FK-resolves / renders in the error record.

    Both ride on the wrapper: the executor already pickles ``_WorkerEntry``
    to each worker, and a ``Manager().dict()`` proxy pickles/reconnects
    across that boundary, so no separate plumbing is needed.

    This is a class, not a closure, because ``Pool.map`` pickles the
    mapped callable to send it to the worker (under both ``fork`` and
    ``spawn``); a closure is not picklable, whereas an instance holding a
    picklable ``func`` (e.g. a ``functools.partial`` of a module-level
    function), an enum ``component``, and a picklable proxy is.

    Attributes:
        func(Callable):    The wrapped per-item worker callable.

        component(Component):    Component for wrapping unknown errors.

        inflight(DictProxy or None):    Shared ``{pid: item}`` map, or
            ``None`` to disable tracking (non-``run_pool`` callers).

        related_files(FileKind, Callable, or None):    Classifier turning
            the item into related file(s); see :func:`_resolve_related_files`.
    """

    def __init__(
        self, func, component: Component, inflight=None, related_files=None
    ):
        self.func = func
        self.component = component
        self.inflight = inflight
        self.related_files = related_files

    def __call__(self, *args, **kwargs):
        item = args[0] if args else None
        pid = os.getpid()
        if self.inflight is not None:
            try:
                self.inflight[pid] = item if args else kwargs
            except Exception:  # pylint: disable=broad-except
                pass  # tracking is best-effort; never fail a task over it
        try:
            # The stamping ``except`` is *inside* the related-files scope so
            # ``_stamp`` copies the item onto the error before it is pickled
            # back (the parent's context no longer has it).
            with error_context(
                related_files=_resolve_related_files(self.related_files, item)
            ):
                try:
                    return self.func(*args, **kwargs)
                except Exception as exc:  # pylint: disable=broad-except
                    stamped = _stamp_worker_error(exc, self.component)
                    if stamped is exc:
                        raise
                    raise stamped from exc
        finally:
            if self.inflight is not None:
                try:
                    self.inflight.pop(pid, None)
                except Exception:  # pylint: disable=broad-except
                    pass


def worker_entry(func, component: Component, inflight=None, related_files=None):
    """Wrap a Pool worker callable so errors come back picklable + stamped.

    Args:
        func(Callable):    The worker callable to wrap (must itself be
            picklable, e.g. a module-level function or a ``partial`` of
            one).

        component(Component):    Component to assign when wrapping an
            unknown exception.

        inflight(DictProxy or None):    Shared in-flight map (see
            :class:`_WorkerEntry`); ``None`` disables tracking.

        related_files(FileKind, Callable, or None):    Per-item related-file
            classifier (see :func:`_resolve_related_files`).

    Returns:
        _WorkerEntry:    A picklable callable suitable to hand to a Pool.
    """

    return _WorkerEntry(func, component, inflight, related_files)


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


def _signal_name(signum):
    """POSIX signal name for a number (e.g. 9 -> ``"SIGKILL"``), or None."""

    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return None


def _exit_signal_entry(code):
    """Decode one process exit code into a portable death descriptor.

    ``None`` (still running) and ``0`` (clean) yield ``None``. The meaning
    of a non-zero code is OS-specific, so decode accordingly:

    - **POSIX**: a *negative* code is a kill by signal ``-code`` (``SIGKILL``
      -> OOM / macOS jetsam, ``SIGSEGV`` -> native crash), whose name is
      added; a positive code is a plain ``exit(code)``.
    - **Windows**: there are no POSIX signals -- the code is a process /
      NTSTATUS exit status (e.g. ``0xC0000005`` = access violation), so its
      conventional hex form is added for abnormal values rather than being
      (mis)read as a signal.

    Never raises.

    Args:
        code(int or None):    A ``multiprocessing.Process.exitcode``.

    Returns:
        dict or None:    ``{"exitcode": code[, "signal"|"status": ...]}``.
    """

    if code in (None, 0):
        return None
    entry = {"exitcode": code}
    try:
        if os.name == "posix":
            if code < 0:
                entry["signal"] = _signal_name(-code)
        elif code < 0 or code > 0xFFFF:
            # Windows crash/NTSTATUS codes read best in hex.
            entry["status"] = f"0x{code & 0xFFFFFFFF:08X}"
    except Exception:  # pylint: disable=broad-except
        pass
    return entry


def decode_exit_signals(exitcodes):
    """Decode a collection of process exit codes (best-effort, portable).

    Returns one :func:`_exit_signal_entry` per *abnormal* exit (dropping
    ``None`` = still running and ``0`` = clean), so an empty list means no
    abnormal termination was observed. Shared by both parallel schemes so a
    crash report reads the same ``details["exit_signal"]`` regardless of
    transport. Never raises.

    Args:
        exitcodes(iterable):    ``Process.exitcode`` values.

    Returns:
        list[dict]:    The decoded abnormal exits.
    """

    result = []
    try:
        for code in exitcodes:
            entry = _exit_signal_entry(code)
            if entry is not None:
                result.append(entry)
    except Exception:  # pylint: disable=broad-except
        pass
    return result


def _pool_exit_signals(executor):
    """Decode a broken pool's worker exit codes (best-effort, private API).

    ``ProcessPoolExecutor`` hides a worker death behind
    ``BrokenProcessPool`` and clears its process table on shutdown, so this
    must be read at the moment of the break (see :func:`run_pool`). Reaches
    into the executor's private ``_processes``; returns ``[]`` if the
    attribute is absent or anything goes wrong.

    Args:
        executor(ProcessPoolExecutor):    The broken executor.

    Returns:
        list[dict]:    Decoded abnormal worker exits.
    """

    try:
        processes = getattr(executor, "_processes", None) or {}
        return decode_exit_signals(
            proc.exitcode for proc in list(processes.values())
        )
    except Exception:  # pylint: disable=broad-except
        return []


def _worker_crashed(
    items,
    exc: Exception,
    inflight=None,
    related_files=None,
    exit_signal=None,
    num_processes=None,
) -> "WorkerCrashedError":
    """Synthesise the parent-side error for a worker that died silently.

    Used when a worker dies without producing an error object (segfault,
    OOM-killer, ``os._exit``), so the parent must describe the failure
    from what *it* knows: the step, the in-flight inputs, and the
    underlying pool error.

    Args:
        items:    The work items that were in flight.

        exc(Exception):    The error the pool surfaced for the death.

        inflight(DictProxy or None):    The shared ``{pid: item}``
            in-flight map (see :class:`_WorkerEntry`). Its values are the
            items being executed at the moment of death -- the culprit
            plus any innocents the executor force-terminated, a set
            bounded by the worker count. ``None`` if tracking was off.

        related_files(FileKind, Callable, or None):    The call site's
            related-file classifier, used to promote the in-flight items
            to structured ``related_files`` on the error (so a crash links
            straight to the offending file, not just a ``details`` string).

        exit_signal(list or None):    Decoded OS-level exit info for the
            dead worker(s) (see :func:`decode_exit_signals`) -- the tell
            for SIGKILL/OOM vs. a native crash. Recorded when non-empty.

        num_processes(int or None):    The pool's worker count, recorded
            alongside the memory snapshot so ``N`` workers vs. total RAM
            makes an OOM death easy to judge.

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
    if exit_signal:
        err.details["exit_signal"] = exit_signal
    # Machine memory at crash time (+ the worker count): the tell for an
    # OOM/jetsam kill, especially paired with a SIGKILL and no native dump.
    resources = collect_resource_snapshot()
    if num_processes is not None:
        resources["num_processes"] = num_processes
    if resources:
        err.details["resources"] = resources
    if inflight is not None:
        try:
            in_flight = list(inflight.values())
        except Exception:  # pylint: disable=broad-except
            in_flight = []
        if in_flight:
            err.details["crashed_inputs"] = [repr(i) for i in in_flight]
            # Promote to structured related files so the crash links to the
            # actual artifact (rendered / FK-resolved), not just a string.
            # ``dict.fromkeys`` dedups (keeping order) so a batch-constant
            # file the classifier returns for every item -- e.g. the single
            # photref -- appears once, not once per crashed input.
            related = []
            for crashed_item in in_flight:
                related.extend(
                    _resolve_related_files(related_files, crashed_item)
                )
            related = list(dict.fromkeys(related))
            if related:
                err.related_files = tuple(related)
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
    related_files=None,
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

        related_files(FileKind, Callable, or None):    Classifier that turns
            each item into the file it is about (a :class:`FileKind` when
            items are paths, else an ``item -> RelatedFile`` callable), so
            errors -- including a silent worker death -- carry the artifact
            they were processing. ``None`` attaches nothing.

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
    executor_kwargs = {
        "max_workers": num_processes,
        "initializer": setup_process_map,
        "initargs": (config,),
    }
    if max_tasks_per_child is not None:
        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

    # The in-flight map lets a silent worker death name its culprit
    # input(s). It lives on a Manager server process, and the proxy rides
    # to each worker on the pickled ``worker_entry`` wrapper; the Manager
    # is torn down when the pool is done, so nothing leaks.
    manager = Manager()
    try:
        inflight = manager.dict()
        wrapped = worker_entry(worker, component, inflight, related_files)
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            try:
                if stream_consumer is None:
                    return list(executor.map(wrapped, items))
                stream_consumer(_stream_as_completed(executor, wrapped, items))
                return None
            except AutoWISPError:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                # Synthesise *inside* the ``with`` so the dead worker's OS
                # exit code is still readable -- ``ProcessPoolExecutor``
                # clears its process table on shutdown, which the enclosing
                # ``with`` triggers on the way out.
                raise _worker_crashed(
                    items,
                    exc,
                    inflight,
                    related_files,
                    _pool_exit_signals(executor),
                    num_processes,
                ) from exc
    finally:
        manager.shutdown()
