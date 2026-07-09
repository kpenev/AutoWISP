"""Unit tests for the ambient error context and capture layer.

The in-process tests need no pipeline fixtures or database; the
cross-process tests spin up a real ``multiprocessing.Pool`` (through
``run_pool``) and let its worker bootstrap with the real
``setup_process_map``, which initialises a throwaway project home in a
temporary directory. Each test starts from a clean ambient context (reset
in ``setUp``).
"""

import os
import pickle
import tempfile
import unittest
from multiprocessing import Process, Queue

from autowisp.multiprocessing_util import setup_process
from autowisp.error_context import (
    ErrorContext,
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
)
from autowisp.processing_steps.solve_astrometry import manage_astrometry
from autowisp.database.frozen_row import FrozenRow


def _dr_file(path="/tmp/x.h5", role="input"):
    """A small RelatedFile for related-files assertions."""

    return RelatedFile(FileKind.DR_FILE, path, role=role)


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
        it raises a ``WorkerCrashedError`` recording the OS exit codes,
        instead of blocking forever.
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
        self.assertEqual(exc.details["exitcodes"], [7])
        self.assertEqual(exc.details["num_in_flight"], 1)


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
