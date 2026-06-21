"""Unit tests for the phase-2 ambient error context and capture layer.

These need no pipeline fixtures or database, so they subclass
``unittest.TestCase`` directly. Each test starts from a clean ambient
context (reset in ``setUp``).
"""

import os
import pickle
import unittest

from autowisp.error_context import (
    ErrorContext,
    capture_errors,
    error_context,
    get_error_context,
    in_worker,
    set_error_context,
    set_pipeline_run,
    worker_entry,
)
from autowisp.exceptions import (
    Component,
    FileKind,
    FindStarsError,
    PipelineError,
    RelatedFile,
    SolveAstrometryError,
    StackToMasterError,
    StepError,
    ViewError,
    WorkerCrashedError,
)
from autowisp.database.frozen_row import FrozenRow


def _dr_file(path="/tmp/x.h5", role="input"):
    """A small RelatedFile for related-files assertions."""

    return RelatedFile(FileKind.DR_FILE, path, role=role)


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
                    [rf.path for rf in ctx.related_files], ["/a", "/b"]
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

    def test_bare_exception_wrapped_and_stamped(self):
        """A bare exception becomes a picklable WorkerCrashedError."""

        def boom():
            raise ValueError("kaboom")

        with self.assertRaises(WorkerCrashedError) as ctx:
            worker_entry(boom, Component.STEP)()

        exc = ctx.exception
        self.assertEqual(exc.subprocess_id, os.getpid())
        self.assertIn("original_traceback", exc.details)
        self.assertIsInstance(exc.__cause__, ValueError)

        restored = pickle.loads(pickle.dumps(exc))
        self.assertEqual(restored.subprocess_id, os.getpid())
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


if __name__ == "__main__":
    unittest.main()
