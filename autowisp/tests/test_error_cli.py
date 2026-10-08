"""Unit tests for CLI error reporting and the entry decorator."""

import contextlib
import io
import tempfile
import unittest

from sqlalchemy import select

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error

# pylint: enable=no-name-in-module
from autowisp.exceptions import Component, ConfigurationError, PipelineError
from autowisp.error_cli import cli_entry_point, exit_code_for, report_error
from autowisp.tests.error_fixtures import make_find_stars_error


class TestExitCodeFor(unittest.TestCase):
    """exit_code_for distinguishes the components."""

    def test_distinct_codes(self):
        self.assertEqual(exit_code_for(Component.STEP), 2)
        self.assertEqual(exit_code_for(Component.PIPELINE), 3)
        self.assertEqual(exit_code_for(Component.BUI), 4)


class _CliTestCase(unittest.TestCase):
    """Base creating one throwaway project database for the class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestReportError(_CliTestCase):
    """report_error persists, renders to the stream, returns the code."""

    def test_default_view_summary_and_pointer(self):
        """Default rendering is the summary plus a pointer line."""

        stream = io.StringIO()
        code = report_error(
            make_find_stars_error(), developer=False, stream=stream
        )

        output = stream.getvalue()
        self.assertEqual(code, 2)  # step component
        self.assertIn("[step:find_stars]", output)
        self.assertIn("no stars found", output)
        self.assertIn("wisp-crash-report", output)
        self.assertNotIn("Traceback:", output)

    def test_developer_view_includes_traceback(self):
        """developer=True renders the full technical detail."""

        stream = io.StringIO()
        report_error(make_find_stars_error(), developer=True, stream=stream)

        output = stream.getvalue()
        self.assertIn("Exception: FindStarsError", output)
        self.assertIn("Traceback:", output)

    def test_persists_a_row(self):
        """Reporting records a queryable Error row."""

        report_error(make_find_stars_error(), stream=io.StringIO())
        with start_db_session() as db_session:
            count = len(db_session.scalars(select(Error)).all())
        self.assertGreaterEqual(count, 1)


class TestCliEntryPoint(_CliTestCase):
    """cli_entry_point reports and exits non-zero on an escaping error."""

    def test_autowisp_error_exits_with_component_code(self):
        """A raised AutoWISPError -> SystemExit with the component's code."""

        @cli_entry_point(component=Component.PIPELINE)
        def main():
            raise PipelineError("orchestration broke")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 3)  # pipeline component

    def test_unknown_exception_is_wrapped_and_exits(self):
        """A bare exception is wrapped (capture_errors) then reported."""

        @cli_entry_point(component=Component.PIPELINE)
        def main():
            raise ValueError("boom")

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 3)

    def test_success_passes_through(self):
        """No error -> the wrapped return value, no exit."""

        @cli_entry_point(component=Component.STEP)
        def main():
            return 42

        self.assertEqual(main(), 42)


class TestStepEntryBoundaries(unittest.TestCase):
    """Every pipeline-step main() is wrapped as a CLI error boundary."""

    def test_all_step_mains_are_cli_entry_points(self):
        """Each step that exposes a ``main()`` carries the CLI boundary.

        Driven by :func:`autowisp.processing_steps.get_step_names` so a
        newly-added step is checked automatically; steps without a CLI
        ``main()`` (library-only steps) are skipped.
        """

        import importlib
        from autowisp.processing_steps import get_step_names

        checked = []
        for name in get_step_names():
            module = importlib.import_module(
                "autowisp.processing_steps." + name
            )
            main = getattr(module, "main", None)
            if main is None:
                continue
            checked.append(name)
            self.assertEqual(
                getattr(main, "__cli_entry_point__", None),
                Component.STEP,
                f"{name}.main() is not a Component.STEP CLI entry point",
            )
        self.assertTrue(checked, "no step main() functions discovered")


class TestStepEntryEndToEnd(_CliTestCase):
    """A real decorated step main() surfaces an escaping error."""

    def test_calibrate_main_reports_and_exits(self):
        """An error inside ``calibrate.main`` -> SystemExit with STEP code."""

        from unittest import mock
        from autowisp.processing_steps import calibrate as calibrate_step

        with (
            mock.patch.object(
                calibrate_step,
                "parse_command_line",
                return_value={"raw_images": [], "calibrate_only_if": None},
            ),
            mock.patch.object(calibrate_step, "setup_process"),
            mock.patch.object(
                calibrate_step, "find_fits_fnames", return_value=[]
            ),
            mock.patch.object(
                calibrate_step,
                "calibrate",
                side_effect=ValueError("calibration blew up"),
            ),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    calibrate_step.main()
        self.assertEqual(ctx.exception.code, 2)  # Component.STEP


class TestConfigParseErrors(unittest.TestCase):
    """Stored-config parse errors are catchable, not an uncatchable exit.

    The pipeline builds each step's config dict by feeding stored
    configuration through the step's ``ManualStepArgumentParser`` (see
    ``ProcessingManager.get_config``). A bad value there must raise a
    recordable ``ConfigurationError`` rather than argparse's default
    ``SystemExit``, which would silently end a detached run.
    """

    @staticmethod
    def _parser():
        from autowisp.processing_steps.manual_util import (
            ManualStepArgumentParser,
        )

        parser = ManualStepArgumentParser(input_type="raw", description="t")
        parser.add_argument(
            "--tool", choices=["fistar", "hatphot"], help="tool"
        )
        return parser

    def test_interactive_cli_still_exits(self):
        """Outside config-parse mode, a bad value keeps argparse exit."""

        parser = self._parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--tool", "bogus"])

    def test_config_mode_raises_configuration_error(self):
        """In config-parse mode, a bad value raises ConfigurationError."""

        from autowisp.processing_steps.manual_util import (
            raise_config_parse_errors,
        )

        parser = self._parser()
        with raise_config_parse_errors():
            with self.assertRaises(ConfigurationError):
                parser.parse_args(["--tool", "bogus"])

    def test_mode_is_restored_after_context(self):
        """Leaving the context reverts to argparse's exit behavior."""

        from autowisp.processing_steps.manual_util import (
            raise_config_parse_errors,
        )

        parser = self._parser()
        with raise_config_parse_errors():
            with self.assertRaises(ConfigurationError):
                parser.parse_args(["--tool", "bogus"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--tool", "bogus"])


if __name__ == "__main__":
    unittest.main()
