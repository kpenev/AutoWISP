"""Unit tests for the ambient error context and capture layer.

The in-process tests need no pipeline fixtures or database; the
cross-process tests spin up a real ``multiprocessing.Pool`` (through
``run_pool``) and let its worker bootstrap with the real
``setup_process_map``, which initialises a throwaway project home in a
temporary directory. Each test starts from a clean ambient context (reset
in ``setUp``).
"""

import glob
import os
import pickle
import tempfile
import unittest
from multiprocessing import Process, Queue
from unittest import mock

import autowisp.error_context as ecmod
from autowisp.multiprocessing_util import setup_process
from autowisp.error_context import (
    ErrorContext,
    _resolve_related_files,
    _worker_crashed,
    capture_errors,
    capture_for_queue,
    error_context,
    forbid_nested_workers,
    get_error_context,
    in_worker,
    reraise_from_worker,
    run_pool,
    set_error_context,
    set_pipeline_run,
    worker_entry,
)
from autowisp.exceptions import (
    Component,
    FileKind,
    FindStarsError,
    MeasurePhotometryError,
    PipelineError,
    RelatedFile,
    SolveAstrometryError,
    StackToMasterError,
    StepError,
    ViewError,
    WorkerCrashedError,
    collect_resource_snapshot,
)
from autowisp.processing_steps.solve_astrometry import manage_astrometry
from autowisp.database.frozen_row import FrozenRow


def _dr_file(path="/tmp/x.h5", role="input"):
    """A small RelatedFile for related-files assertions."""

    return RelatedFile(FileKind.DR_FILE, path, role=role)


def _lc_with_reference(item):
    """A ``related_files`` classifier: the item plus a batch-constant ref.

    Module-level (so it is picklable to workers) and returns *multiple*
    files -- the per-item lightcurve and the single photometric reference
    the whole batch shares -- exactly the shape the detrending call site
    builds via ``functools.partial``.
    """

    return [
        RelatedFile(FileKind.LIGHTCURVE, item, role="input"),
        RelatedFile(FileKind.DR_FILE, "/dr/ref.h5", role="single_photref"),
    ]


def _raise_find_stars_error(item):
    """Module-level Pool worker that raises a StepError.

    Must be top-level (not a closure/lambda) so ``Pool.map`` can pickle
    it to the worker.
    """

    raise FindStarsError(f"boom in worker for {item!r}")


def _raise_value_error(item):
    """Module-level Pool worker that raises a bare (non-AutoWISP) error."""

    raise ValueError(f"bad value in worker for {item!r}")


def _hard_exit_worker(_):
    """Module-level Pool worker that hard-exits without an exception."""

    os._exit(7)  # pylint: disable=protected-access


def _segfault_worker(_):
    """Module-level Pool worker that crashes with a real SIGSEGV.

    Used to prove ``faulthandler`` (armed in ``setup_process_map``) dumps a
    native traceback into the worker's own redirected log before it dies.
    """

    try:
        import resource  # pylint: disable=import-outside-toplevel

        # No core file, so a CI box is not littered with them.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:  # pylint: disable=broad-except
        pass
    import ctypes  # pylint: disable=import-outside-toplevel

    ctypes.string_at(0)  # dereference NULL -> SIGSEGV


def _nested_run_pool_worker(item):
    """Worker that (illegally) tries to launch its own pool.

    Used to prove the no-nested-workers guard end-to-end: this runs inside
    a real worker (so ``setup_process_map`` has set ``in_worker``), and the
    inner ``run_pool`` must refuse before spawning anything.
    """

    return run_pool(_noop, [item], config={}, num_processes=1)


def _instant_exit():
    """Module-level Process target that hard-exits without an exception."""

    # os._exit skips cleanup -- exactly how a segfault/OOM-killed worker
    # dies, which is what we are simulating.
    os._exit(7)  # pylint: disable=protected-access


def _noop(*args, **kwargs):  # pylint: disable=unused-argument
    """Do-nothing stand-in for the per-image mark_start / mark_end hooks."""


def _process_queue_worker(result_queue, config):
    """Process worker that bootstraps, fails, and queues the stamped error.

    Mirrors ``solve_astrometry.astrometry_process``: bootstrap the ambient
    context from ``config``, catch the failure, and put the result of
    ``capture_for_queue`` (a stamped, picklable error) on the queue rather
    than raising out of the process.
    """

    setup_process(**config)
    try:
        raise ValueError("no wcs in worker")
    except Exception as exc:  # pylint: disable=broad-except
        result_queue.put(capture_for_queue(exc, component=Component.STEP))


def _pool_config(project_home, *, run_id, step="find_stars"):
    """Minimal per-process config for ``run_pool``'s bootstrap.

    ``setup_process_map`` runs in each worker; pointing it at a temporary
    ``project_home`` keeps its side effects (a throwaway SQLite DB and the
    per-worker log / stdout-stderr files) inside the temp dir.
    """

    return {
        "project_home": project_home,
        "pipeline_run_id": run_id,
        "host": "testhost",
        "pipeline_started": None,
        "code_version": "testver",
        "processing_step": step,
        "std_out_err_fname": os.path.join(
            project_home, "worker_{pid:d}.outerr"
        ),
        "logging_fname": os.path.join(project_home, "worker_{pid:d}.log"),
    }


class _ContextTestCase(unittest.TestCase):
    """Base resetting the ambient context to empty before each test."""

    def setUp(self):
        set_error_context(ErrorContext())


class TestErrorContextDataclass(_ContextTestCase):
    """The ErrorContext bundle itself."""

    def test_defaults(self):
        """A bare ErrorContext has empty/None fields."""

        ctx = ErrorContext()
        self.assertIsNone(ctx.pipeline_run)
        self.assertIsNone(ctx.step_name)
        self.assertEqual(ctx.related_files, ())
        self.assertFalse(ctx.in_worker)

    def test_frozen(self):
        """ErrorContext is immutable."""

        ctx = ErrorContext()
        with self.assertRaises(Exception):
            ctx.step_name = "find_stars"  # frozen dataclass -> error


class TestFromConfig(_ContextTestCase):
    """ErrorContext.from_config rebuilds context from a config dict."""

    def test_empty_config(self):
        """No run keys -> no snapshot, no step, main process."""

        ctx = ErrorContext.from_config({})
        self.assertIsNone(ctx.pipeline_run)
        self.assertIsNone(ctx.step_name)
        self.assertFalse(ctx.in_worker)

    def test_full_config(self):
        """Run keys produce a populated pipeline-run snapshot."""

        ctx = ErrorContext.from_config(
            {
                "pipeline_run_id": 88,
                "host": "node3",
                "pipeline_started": "2026-06-21T00:00:00",
                "code_version": "abc123",
                "processing_step": "solve_astrometry",
                "parent_pid": 4321,
            }
        )
        self.assertIsInstance(ctx.pipeline_run, FrozenRow)
        self.assertEqual(ctx.pipeline_run.id, 88)
        self.assertEqual(ctx.pipeline_run.host, "node3")
        self.assertEqual(ctx.pipeline_run.code_version, "abc123")
        self.assertEqual(ctx.step_name, "solve_astrometry")
        self.assertTrue(ctx.in_worker)

    def test_code_version_fallback(self):
        """A missing code_version falls back to get_code_version_str()."""

        ctx = ErrorContext.from_config({"pipeline_run_id": 1})
        self.assertIsInstance(ctx.pipeline_run.code_version, str)
        self.assertTrue(ctx.pipeline_run.code_version)

    def test_sentinel_step_names_become_none(self):
        """Bootstrap sentinels are not treated as real step names."""

        for sentinel in ("init_processing", "none", None):
            ctx = ErrorContext.from_config({"processing_step": sentinel})
            self.assertIsNone(ctx.step_name)

    def test_in_worker_from_parent_pid(self):
        """in_worker is True iff a truthy parent_pid is present."""

        self.assertTrue(ErrorContext.from_config({"parent_pid": 999}).in_worker)
        # The main process threads parent_pid="" -> falsy -> not a worker.
        self.assertFalse(ErrorContext.from_config({"parent_pid": ""}).in_worker)
        self.assertFalse(ErrorContext.from_config({}).in_worker)


class TestAmbientAccessors(_ContextTestCase):
    """get/set helpers and in_worker()."""

    def test_set_and_get(self):
        """set_error_context installs a bundle get_error_context returns."""

        ctx = ErrorContext(step_name="epd", in_worker=True)
        set_error_context(ctx)
        self.assertIs(get_error_context(), ctx)
        self.assertTrue(in_worker())

    def test_set_pipeline_run_keeps_rest(self):
        """set_pipeline_run swaps only the run, preserving other fields."""

        set_error_context(
            ErrorContext(
                step_name="tfa",
                related_files=(_dr_file(),),
                in_worker=True,
            )
        )
        run = FrozenRow("pipeline_run", {"id": 7})
        set_pipeline_run(run)

        ctx = get_error_context()
        self.assertIs(ctx.pipeline_run, run)
        self.assertEqual(ctx.step_name, "tfa")
        self.assertEqual(len(ctx.related_files), 1)
        self.assertTrue(ctx.in_worker)


class TestErrorContextManager(_ContextTestCase):
    """The error_context() scoping context manager."""

    def test_scopes_and_restores(self):
        """Step + files apply inside the block and are removed after."""

        with error_context(step_name="find_stars", related_files=[_dr_file()]):
            ctx = get_error_context()
            self.assertEqual(ctx.step_name, "find_stars")
            self.assertEqual(len(ctx.related_files), 1)

        ctx = get_error_context()
        self.assertIsNone(ctx.step_name)
        self.assertEqual(ctx.related_files, ())

    def test_nesting_innermost_step_wins_and_files_accumulate(self):
        """Inner step overrides; related files from both levels stack."""

        with error_context(step_name="outer", related_files=[_dr_file("/a")]):
            with error_context(
                step_name="inner", related_files=[_dr_file("/b")]
            ):
                ctx = get_error_context()
                self.assertEqual(ctx.step_name, "inner")
                self.assertEqual(
                    [rf.path.as_posix() for rf in ctx.related_files],
                    ["/a", "/b"],
                )
            # Back to the outer scope.
            self.assertEqual(get_error_context().step_name, "outer")

    def test_keeps_pipeline_run_and_worker_flag(self):
        """The block inherits run + in_worker from the current context."""

        run = FrozenRow("pipeline_run", {"id": 3})
        set_error_context(ErrorContext(pipeline_run=run, in_worker=True))
        with error_context(step_name="epd"):
            ctx = get_error_context()
            self.assertIs(ctx.pipeline_run, run)
            self.assertTrue(ctx.in_worker)


class TestCaptureErrors(_ContextTestCase):
    """The capture_errors decorator stamps + optionally wraps."""

    def test_bare_value_error_wrapped_to_step_subclass(self):
        """A bare ValueError becomes the ambient step's StepError type."""

        @capture_errors(component=Component.STEP)
        def boom():
            raise ValueError("bad pixels")

        with error_context(step_name="find_stars"):
            with self.assertRaises(FindStarsError) as ctx:
                boom()

        exc = ctx.exception
        self.assertEqual(exc.step_name, "find_stars")
        self.assertEqual(str(exc), "bad pixels")
        self.assertIsInstance(exc.__cause__, ValueError)

    def test_unknown_step_falls_back_to_base_steperror(self):
        """An unmapped step name yields a plain StepError."""

        @capture_errors(component=Component.STEP)
        def boom():
            raise ValueError("x")

        with error_context(step_name="mystery_step"):
            with self.assertRaises(StepError) as ctx:
                boom()
        self.assertIs(type(ctx.exception), StepError)
        self.assertEqual(ctx.exception.step_name, "mystery_step")

    def test_pipeline_and_bui_wrapping(self):
        """Component selects the wrapper class."""

        @capture_errors(component=Component.PIPELINE)
        def boom_pipeline():
            raise KeyError("k")

        @capture_errors(component=Component.BUI)
        def boom_bui():
            raise KeyError("k")

        with self.assertRaises(PipelineError):
            boom_pipeline()
        with self.assertRaises(ViewError):
            boom_bui()

    def test_existing_autowisp_error_passes_through_stamped(self):
        """An AutoWISPError keeps its type and gets stamped from context."""

        run = FrozenRow("pipeline_run", {"id": 9, "host": "h"})

        @capture_errors(component=Component.STEP)
        def boom():
            raise StackToMasterError("cannot stack")

        set_pipeline_run(run)
        with error_context(
            step_name="stack_to_master", related_files=[_dr_file()]
        ):
            with self.assertRaises(StackToMasterError) as ctx:
                boom()

        exc = ctx.exception
        self.assertEqual(exc.step_name, "stack_to_master")
        self.assertIs(exc.pipeline_run, run)
        self.assertIsNotNone(exc.crashed)
        self.assertEqual(len(exc.related_files), 1)

    def test_no_run_leaves_pipeline_run_none(self):
        """With no run in context, pipeline_run/crashed stay unset."""

        @capture_errors(component=Component.PIPELINE)
        def boom():
            raise PipelineError("x")

        with self.assertRaises(PipelineError) as ctx:
            boom()
        self.assertIsNone(ctx.exception.pipeline_run)
        self.assertIsNone(ctx.exception.crashed)

    def test_preset_step_name_not_overwritten(self):
        """A step name set at the raise site survives stamping."""

        @capture_errors(component=Component.STEP)
        def boom():
            raise FindStarsError("x", step_name="explicit")

        with error_context(step_name="find_stars"):
            with self.assertRaises(FindStarsError) as ctx:
                boom()
        self.assertEqual(ctx.exception.step_name, "explicit")

    def test_wrap_unknown_false_passes_through(self):
        """With wrap_unknown=False the original exception is untouched."""

        @capture_errors(component=Component.STEP, wrap_unknown=False)
        def boom():
            raise ValueError("raw")

        with error_context(step_name="find_stars"):
            with self.assertRaises(ValueError):
                boom()

    def test_success_returns_value(self):
        """The happy path returns the wrapped callable's result."""

        @capture_errors(component=Component.STEP)
        def ok():
            return 123

        self.assertEqual(ok(), 123)


class TestWorkerEntry(_ContextTestCase):
    """worker_entry stamps subprocess_id and stays picklable."""

    def test_bare_exception_wrapped_to_step_subclass(self):
        """A bare worker exception becomes the mapped, picklable StepError."""

        def boom():
            raise ValueError("kaboom")

        with error_context(step_name="find_stars"):
            with self.assertRaises(FindStarsError) as ctx:
                worker_entry(boom, Component.STEP)()

        exc = ctx.exception
        self.assertEqual(exc.subprocess_id, os.getpid())
        self.assertEqual(exc.step_name, "find_stars")
        self.assertIn("original_traceback", exc.details)
        self.assertIsInstance(exc.__cause__, ValueError)

        restored = pickle.loads(pickle.dumps(exc))
        self.assertEqual(restored.subprocess_id, os.getpid())
        self.assertEqual(restored.step_name, "find_stars")
        self.assertIn("original_traceback", restored.details)

    def test_autowisp_error_passes_through_stamped(self):
        """An AutoWISPError keeps its type and gains subprocess_id."""

        def boom():
            raise SolveAstrometryError("no wcs", step_name="solve_astrometry")

        with self.assertRaises(SolveAstrometryError) as ctx:
            worker_entry(boom, Component.STEP)()

        self.assertEqual(ctx.exception.subprocess_id, os.getpid())
        self.assertEqual(ctx.exception.step_name, "solve_astrometry")

    def test_success_returns_value(self):
        """worker_entry is transparent on the happy path."""

        self.assertEqual(worker_entry(lambda: 7, Component.STEP)(), 7)

    def test_inflight_map_tracks_then_clears_item(self):
        """The item is in the in-flight map while running, gone after.

        (A plain dict stands in for the ``Manager().dict()`` proxy; the
        write/clear logic is the same.)
        """

        tracker = {}
        seen = {}

        def fn(item):
            seen["during"] = dict(tracker)  # snapshot while executing
            return item * 2

        result = worker_entry(fn, Component.STEP, tracker)(21)

        self.assertEqual(result, 42)
        self.assertEqual(seen["during"], {os.getpid(): 21})
        self.assertEqual(dict(tracker), {})  # cleared on return

    def test_inflight_map_cleared_on_error(self):
        """A failing task still clears its in-flight entry."""

        tracker = {}

        def boom(_):
            raise ValueError("boom")

        with error_context(step_name="find_stars"):
            with self.assertRaises(FindStarsError):
                worker_entry(boom, Component.STEP, tracker)(7)

        self.assertEqual(dict(tracker), {})

    def test_resolve_related_files_from_kind_and_callable(self):
        """A FileKind wraps a path item; a callable is used as given."""

        (rf,) = _resolve_related_files(FileKind.LIGHTCURVE, "/lc/x.h5")
        self.assertEqual(rf.kind, FileKind.LIGHTCURVE)
        self.assertEqual(rf.path.as_posix(), "/lc/x.h5")
        self.assertEqual(rf.role, "input")

        made = _dr_file("/data/y.h5")
        self.assertEqual(
            _resolve_related_files(lambda item: made, "anything"), (made,)
        )

    def test_resolve_related_files_callable_returning_many(self):
        """A classifier may return several files (item + batch constants)."""

        result = _resolve_related_files(_lc_with_reference, "/lc/x.h5")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].kind, FileKind.LIGHTCURVE)
        self.assertEqual(result[0].path.as_posix(), "/lc/x.h5")
        self.assertEqual(result[1].kind, FileKind.DR_FILE)
        self.assertEqual(result[1].path.as_posix(), "/dr/ref.h5")

    def test_resolve_related_files_is_total(self):
        """No classifier, no item, or a bad classifier -> empty, no raise."""

        self.assertEqual(_resolve_related_files(None, "/lc/x.h5"), ())
        self.assertEqual(_resolve_related_files(FileKind.LIGHTCURVE, None), ())
        # A FileKind on a non-path item (Path(42) raises) degrades to empty.
        self.assertEqual(_resolve_related_files(FileKind.LIGHTCURVE, 42), ())
        # A classifier that raises degrades to empty.
        self.assertEqual(_resolve_related_files(lambda _: 1 / 0, "x"), ())

    def test_subprocess_id_not_overwritten_by_later_stamp(self):
        """An already-stamped subprocess_id survives a second boundary.

        Models an exception stamped by one ``except`` that then passes
        through another on its way out of the *same* worker: the second
        stamp is a no-op (``stamp_subprocess`` is idempotent), so the first
        PID wins rather than being overwritten.
        """

        def boom():
            exc = FindStarsError("boom", step_name="find_stars")
            exc.subprocess_id = 4242  # stamped by a deeper except already
            raise exc

        with self.assertRaises(FindStarsError) as ctx:
            worker_entry(boom, Component.STEP)()

        self.assertEqual(ctx.exception.subprocess_id, 4242)


class TestPoolPropagation(_ContextTestCase):
    """Errors raised in real Pool workers cross back stamped."""

    def test_step_error_propagates_with_context(self):
        """A worker StepError returns same-typed, stamped, with traceback.

        Exercises the full ``run_pool`` round trip: the worker bootstraps
        its context from ``config`` (via the real ``setup_process_map``),
        raises a ``FindStarsError``, and the parent receives the *same*
        type carrying the worker's ``subprocess_id`` (a different PID), the
        step name and pipeline-run snapshot rebuilt in the worker, and the
        worker traceback both in ``details`` and as the Pool's
        ``RemoteTraceback`` cause.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(FindStarsError) as ctx:
                run_pool(
                    _raise_find_stars_error,
                    [1],
                    config=_pool_config(project_home, run_id=77),
                    num_processes=1,
                )

        exc = ctx.exception
        # Stamped inside the worker, which is a different process:
        self.assertIsNotNone(exc.subprocess_id)
        self.assertNotEqual(exc.subprocess_id, os.getpid())
        # Step + run context rebuilt in the worker from config:
        self.assertEqual(exc.step_name, "find_stars")
        self.assertIsNotNone(exc.pipeline_run)
        self.assertEqual(exc.pipeline_run.id, 77)
        # Worker traceback survives both ways:
        self.assertIn("original_traceback", exc.details)
        self.assertIn("boom in worker", exc.details["original_traceback"])
        self.assertIsNotNone(exc.__cause__)
        self.assertIn("boom in worker", str(exc.__cause__))

    def test_bare_exception_wrapped_to_mapped_step_error(self):
        """A worker's bare exception returns as the step's StepError subclass.

        The worker raises a plain ``ValueError``; ``_wrap`` maps the
        ambient step name to its concrete subclass, so the parent receives
        a ``MeasurePhotometryError``. The original is not kept as a live
        ``__cause__`` across the Pool (our pickling carries only the
        fields, and multiprocessing replaces ``__cause__`` with its
        ``RemoteTraceback``); the durable record is the formatted traceback
        in ``details``.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(MeasurePhotometryError) as ctx:
                run_pool(
                    _raise_value_error,
                    [1],
                    config=_pool_config(
                        project_home,
                        run_id=88,
                        step="measure_aperture_photometry",
                    ),
                    num_processes=1,
                )

        exc = ctx.exception
        self.assertEqual(exc.step_name, "measure_aperture_photometry")
        self.assertEqual(str(exc), "bad value in worker for 1")
        self.assertIsNotNone(exc.subprocess_id)
        self.assertNotEqual(exc.subprocess_id, os.getpid())
        # The original ValueError is recorded as a traceback string:
        self.assertIn("ValueError", exc.details["original_traceback"])
        self.assertIn("bad value in worker", exc.details["original_traceback"])
        # ... and is visible via the Pool's RemoteTraceback cause:
        self.assertIsNotNone(exc.__cause__)
        self.assertIn("bad value in worker", str(exc.__cause__))

    def test_worker_hard_exit_raises_worker_crashed(self):
        """A worker that hard-exits surfaces as WorkerCrashedError, no hang.

        ``ProcessPoolExecutor`` reports the death as ``BrokenProcessPool``
        (a bare ``multiprocessing.Pool`` would hang here), which
        ``run_pool`` converts into a ``WorkerCrashedError`` describing the
        in-flight inputs instead of blocking the pipeline forever.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(WorkerCrashedError) as ctx:
                run_pool(
                    _hard_exit_worker,
                    [1, 2],
                    config=_pool_config(project_home, run_id=55),
                    num_processes=1,
                )

        exc = ctx.exception
        self.assertEqual(exc.details["num_inputs"], 2)
        self.assertIn("pool_error", exc.details)

    def test_worker_crashed_carries_step_name(self):
        """The synthesised WorkerCrashedError records the ambient step.

        Regression for the "no matching logs found" crash-report gap: the
        step name must reach the queryable ``step_name`` attribute (which
        the persistence layer writes to ``error.step_name`` and
        crash-report log-collection resolves the run/step logs from), not
        only the free-text message. The parent stamps it from its ambient
        context -- in the pipeline it is inside
        ``error_context(step_name=...)`` when it launches the pool.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _hard_exit_worker,
                        [1, 2],
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=1,
                    )

        exc = ctx.exception
        # The attribute persistence reads (getattr(exc, "step_name", ...)):
        self.assertEqual(exc.step_name, "tfa")
        # And the belt-and-braces copy that reaches the sidecar:
        self.assertEqual(exc.details["step_name"], "tfa")
        # A worker death is a step failure, not an orchestration failure:
        self.assertEqual(exc.component, Component.STEP)

    def test_worker_crashed_names_inflight_input(self):
        """A silent death records the in-flight item(s), not a head sample.

        The wrapped worker writes its item into the shared in-flight map
        before running; a hard ``os._exit`` leaves it there, so
        ``_worker_crashed`` can name the culprit in
        ``details["crashed_inputs"]`` (bounded by the worker count).
        """

        items = ["/lc/AAA.h5", "/lc/BBB.h5"]
        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _hard_exit_worker,
                        items,
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=1,
                    )

        crashed = ctx.exception.details.get("crashed_inputs")
        self.assertTrue(crashed, "expected a non-empty crashed_inputs")
        self.assertTrue(set(crashed) <= {repr(item) for item in items})

    @unittest.skipUnless(
        os.name == "posix",
        "no way to provoke a segfault: ctypes.string_at(0) raises a "
        "catchable OSError on Windows rather than killing the worker",
    )
    def test_faulthandler_dumps_native_traceback(self):
        """A segfaulting worker leaves a native dump in its own log.

        Proves the ``faulthandler`` armed in ``setup_process_map`` turns an
        otherwise-silent SIGSEGV into a collectable C-level traceback --
        which is what distinguishes the faulted worker from the innocents
        the executor merely terminates.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError):
                    run_pool(
                        _segfault_worker,
                        ["/lc/AAA.h5"],
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=1,
                    )

            # Worker logs live under a parent-pid subdirectory.
            logs = glob.glob(
                os.path.join(project_home, "**", "*.outerr"), recursive=True
            )
            blobs = [
                open(path, encoding="utf-8", errors="replace").read()
                for path in logs
            ]
        self.assertTrue(
            any(
                "Fatal Python error" in b or "Current thread" in b
                for b in blobs
            ),
            "no faulthandler native dump found in any worker log",
        )

    @unittest.skipUnless(
        os.name == "posix", "signal decoding is POSIX-only (see item 5)"
    )
    def test_worker_crashed_records_exit_signal(self):
        """A native crash records the decoded killing signal.

        The tell for OOM/jetsam (``SIGKILL``) vs. a native crash
        (``SIGSEGV``); here a segfault is expected to surface as
        ``SIGSEGV``. (On Windows the code is an NTSTATUS, not a signal --
        that path is covered by ``TestExitSignalDecode``.)
        """

        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _segfault_worker,
                        ["/lc/AAA.h5"],
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=1,
                    )

        entries = ctx.exception.details.get("exit_signal", [])
        self.assertIn("SIGSEGV", [entry.get("signal") for entry in entries])

    def test_worker_crashed_records_resources(self):
        """A crash records a memory snapshot + the worker count.

        ``N`` workers vs. total RAM is what makes an OOM/jetsam death easy
        to judge (paired with a SIGKILL and no native dump from items 4-5).
        """

        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _hard_exit_worker,
                        ["/lc/AAA.h5"],
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=2,
                    )

        resources = ctx.exception.details.get("resources", {})
        self.assertEqual(resources.get("num_processes"), 2)
        self.assertGreater(resources.get("ram_total", 0), 0)

    def test_worker_error_carries_config(self):
        """A worker error carries its process config (via from_config).

        The resolved config lives only at runtime, so recording it is the
        only way a report shows the settings the step actually ran with.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(FindStarsError) as ctx:
                run_pool(
                    _raise_find_stars_error,
                    ["/lc/A.h5"],
                    config=_pool_config(
                        project_home, run_id=77, step="find_stars"
                    ),
                    num_processes=1,
                )

        config = ctx.exception.details.get("config", {})
        self.assertEqual(config.get("processing_step"), "find_stars")
        self.assertEqual(config.get("code_version"), "testver")

    def test_worker_crashed_carries_scoped_config(self):
        """A crash carries the *failing step's* config, scoped by the manager.

        The parent's own context holds the base ``add_images_to_db`` config,
        so the failing step's config must come from the
        ``error_context(config=...)`` the manager scopes at dispatch (here
        simulated around ``run_pool``) -- not from the parent context.
        """

        with tempfile.TemporaryDirectory() as project_home:
            step_config = _pool_config(project_home, run_id=55, step="tfa")
            step_config["detrend_rej_level"] = 5.0  # a step-specific marker
            with error_context(step_name="tfa", config=step_config):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _hard_exit_worker,
                        ["/lc/A.h5"],
                        config=step_config,
                        num_processes=1,
                    )

        config = ctx.exception.details.get("config", {})
        self.assertEqual(config.get("detrend_rej_level"), 5.0)
        self.assertEqual(config.get("processing_step"), "tfa")

    def test_worker_error_carries_related_file(self):
        """A worker error carries the item it was processing as a file.

        The ``related_files`` classifier scopes the item as the ambient
        related file inside the worker, so an error raised for it (here a
        ``FindStarsError``) crosses back carrying that file -- the datum a
        config-vs-file-content failure most needs.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(FindStarsError) as ctx:
                run_pool(
                    _raise_find_stars_error,
                    ["/lc/AAA.h5"],
                    config=_pool_config(project_home, run_id=77),
                    num_processes=1,
                    related_files=FileKind.LIGHTCURVE,
                )

        related = ctx.exception.related_files
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].kind, FileKind.LIGHTCURVE)
        self.assertEqual(related[0].path.as_posix(), "/lc/AAA.h5")

    def test_worker_crashed_promotes_related_files(self):
        """A silent death promotes the in-flight item to a related file.

        So a ``WorkerCrashedError`` links straight to the offending file
        (rendered / FK-resolved), not only a ``details`` string.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with error_context(step_name="tfa"):
                with self.assertRaises(WorkerCrashedError) as ctx:
                    run_pool(
                        _hard_exit_worker,
                        ["/lc/AAA.h5", "/lc/BBB.h5"],
                        config=_pool_config(
                            project_home, run_id=55, step="tfa"
                        ),
                        num_processes=1,
                        related_files=FileKind.LIGHTCURVE,
                    )

        related = ctx.exception.related_files
        self.assertTrue(related)
        self.assertTrue(all(r.kind == FileKind.LIGHTCURVE for r in related))
        self.assertTrue(
            {r.path.as_posix() for r in related} <= {"/lc/AAA.h5", "/lc/BBB.h5"}
        )

    def test_worker_error_carries_reference_files(self):
        """A multi-file classifier attaches item *and* batch constants.

        Mirrors the detrending call site: the classifier returns the light
        curve plus the single photometric reference, so an error carries
        both.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(FindStarsError) as ctx:
                run_pool(
                    _raise_find_stars_error,
                    ["/lc/AAA.h5"],
                    config=_pool_config(project_home, run_id=77),
                    num_processes=1,
                    related_files=_lc_with_reference,
                )

        got = {(r.kind, r.path.as_posix()) for r in ctx.exception.related_files}
        self.assertIn((FileKind.LIGHTCURVE, "/lc/AAA.h5"), got)
        self.assertIn((FileKind.DR_FILE, "/dr/ref.h5"), got)

    def test_worker_crashed_dedups_shared_related_file(self):
        """A batch-constant file appears once across many in-flight items.

        Two workers in flight against the same reference -> the reference
        is promoted once, not once per crashed input.
        """

        inflight = {111: "/lc/A.h5", 222: "/lc/B.h5"}
        with error_context(step_name="tfa"):
            err = _worker_crashed(
                ["/lc/A.h5", "/lc/B.h5"],
                RuntimeError("pool broke"),
                inflight,
                _lc_with_reference,
            )

        pairs = [(r.kind, r.path.as_posix()) for r in err.related_files]
        self.assertIn((FileKind.LIGHTCURVE, "/lc/A.h5"), pairs)
        self.assertIn((FileKind.LIGHTCURVE, "/lc/B.h5"), pairs)
        # The shared single photref, returned for both items, is deduped:
        self.assertEqual(pairs.count((FileKind.DR_FILE, "/dr/ref.h5")), 1)


class TestProcessQueuePropagation(_ContextTestCase):
    """Process + Queue workers return stamped errors over a queue."""

    def test_queue_worker_error_round_trips(self):
        """A failed queue worker yields a stamped error the parent re-raises.

        Mirrors ``solve_astrometry``: a ``Process`` worker bootstraps its
        context, fails with a bare ``ValueError``, and puts
        ``capture_for_queue(...)`` on the queue. The object pulled off is
        the step's mapped ``AutoWISPError`` (here ``SolveAstrometryError``)
        carrying the worker ``subprocess_id``, the run snapshot, and the
        traceback in ``details``; ``reraise_from_worker`` then raises it.
        """

        with tempfile.TemporaryDirectory() as project_home:
            config = _pool_config(
                project_home, run_id=99, step="solve_astrometry"
            )
            config["parent_pid"] = os.getpid()
            result_queue = Queue()
            worker = Process(
                target=_process_queue_worker, args=(result_queue, config)
            )
            worker.start()
            error = result_queue.get(timeout=60)
            worker.join()

        # Pulled off the queue: the step's mapped, stamped AutoWISPError.
        self.assertIsInstance(error, SolveAstrometryError)
        self.assertEqual(error.step_name, "solve_astrometry")
        self.assertIsNotNone(error.subprocess_id)
        self.assertNotEqual(error.subprocess_id, os.getpid())
        self.assertIn("ValueError", error.details["original_traceback"])
        self.assertIn("no wcs in worker", error.details["original_traceback"])
        # Pipeline-run snapshot rebuilt in the worker from config:
        self.assertIsNotNone(error.pipeline_run)
        self.assertEqual(error.pipeline_run.id, 99)

        # The parent re-raises the same object off the queue.
        with self.assertRaises(SolveAstrometryError) as ctx:
            reraise_from_worker(error)
        self.assertIs(ctx.exception, error)

    def test_all_workers_dead_raises_worker_crashed(self):
        """A worker that hard-exits is detected by the parent, not a hang.

        ``manage_astrometry`` polls ``is_alive()`` while waiting on the
        result queue; when every worker has died without queuing a result
        it raises a ``WorkerCrashedError`` recording the OS exit codes
        (normalised to the same ``exit_signal`` shape as Scheme A), instead
        of blocking forever.
        """

        worker = Process(target=_instant_exit)
        worker.start()
        worker.join()
        self.assertFalse(worker.is_alive())

        with self.assertRaises(WorkerCrashedError) as ctx:
            manage_astrometry(
                {"trans_key": ["dr1.h5"]},
                Queue(),
                Queue(),
                _noop,
                _noop,
                workers=[worker],
            )

        exc = ctx.exception
        # os._exit(7) -> a plain exit code (positive), no signal decoded.
        self.assertEqual(exc.details["exit_signal"], [{"exitcode": 7}])
        self.assertEqual(exc.details["num_in_flight"], 1)


class TestExitSignalDecode(unittest.TestCase):
    """decode_exit_signals is portable and drops clean/running exits."""

    def test_drops_running_and_clean(self):
        """``None`` (running) and ``0`` (clean) contribute nothing."""

        self.assertEqual(ecmod.decode_exit_signals([None, 0]), [])

    def test_posix_signals_and_plain_codes(self):
        """POSIX: negative codes decode to their signal; positive don't."""

        with mock.patch.object(ecmod.os, "name", "posix"):
            self.assertEqual(
                ecmod.decode_exit_signals([7, -9, -11]),
                [
                    {"exitcode": 7},
                    {"exitcode": -9, "signal": "SIGKILL"},
                    {"exitcode": -11, "signal": "SIGSEGV"},
                ],
            )

    def test_windows_status_not_signal(self):
        """Windows: an abnormal code is an NTSTATUS, never a POSIX signal.

        Both the unsigned and signed spellings of an access violation map
        to the conventional hex status; a plain small code stays bare.
        """

        with mock.patch.object(ecmod.os, "name", "nt"):
            self.assertEqual(
                ecmod.decode_exit_signals([0xC0000005, -1073741819, 1]),
                [
                    {"exitcode": 0xC0000005, "status": "0xC0000005"},
                    {"exitcode": -1073741819, "status": "0xC0000005"},
                    {"exitcode": 1},
                ],
            )


class TestResourceSnapshot(unittest.TestCase):
    """collect_resource_snapshot is best-effort and never raises."""

    def test_memory_fields_present_and_sane(self):
        """psutil is a hard dependency, so the memory fields are present."""

        snap = collect_resource_snapshot()
        self.assertGreater(snap["ram_total"], 0)
        self.assertGreaterEqual(snap["ram_available"], 0)
        self.assertGreater(snap["process_rss"], 0)
        self.assertIsInstance(snap["ram_percent_used"], float)

    def test_empty_when_psutil_unavailable(self):
        """A missing psutil degrades to an empty dict, never a raise."""

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("simulated missing psutil")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(collect_resource_snapshot(), {})


class TestNestingGuard(_ContextTestCase):
    """A worker may not launch nested workers (resource control)."""

    def test_forbid_nested_workers_only_inside_worker(self):
        """The guard is a no-op in the main process, raises in a worker."""

        forbid_nested_workers()  # main process (in_worker False): no-op

        set_error_context(ErrorContext(in_worker=True))
        with self.assertRaises(PipelineError):
            forbid_nested_workers()

    def test_nested_run_pool_blocked_end_to_end(self):
        """A real worker that calls run_pool is refused, not allowed to nest.

        Goes through the full chain: the outer ``run_pool`` spawns a real
        worker via ``setup_process_map`` (which sets ``in_worker``), the
        worker calls ``run_pool`` again, and the inner guard raises a
        ``PipelineError`` that propagates back stamped with the worker PID
        -- so an accidental N^2 launch fails loudly instead of spawning.
        """

        with tempfile.TemporaryDirectory() as project_home:
            with self.assertRaises(PipelineError) as ctx:
                run_pool(
                    _nested_run_pool_worker,
                    [1],
                    config=_pool_config(project_home, run_id=33),
                    num_processes=1,
                )

        exc = ctx.exception
        self.assertIn("Nested multiprocessing", str(exc))
        self.assertIsNotNone(exc.subprocess_id)
        self.assertNotEqual(exc.subprocess_id, os.getpid())


if __name__ == "__main__":
    unittest.main()
