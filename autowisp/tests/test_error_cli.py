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
from autowisp.exceptions import Component, PipelineError
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


if __name__ == "__main__":
    unittest.main()
