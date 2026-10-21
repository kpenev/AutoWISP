"""Tests of the experiments' command line, wisp-experiments."""

import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

from sqlalchemy import select
from sqlalchemy.orm import Session

from autowisp.database.interface import set_project_home
from autowisp.experiments_cli import main

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import ProjectCopyCase


class TestExperimentsCli(ProjectCopyCase):
    """The command line: what it prints, asks, refuses and records."""

    @staticmethod
    def _run(*argv, answer=None):
        """
        Run wisp-experiments in this process, answering any question asked.

        Returns:
            int:    The exit status.

            str:    What it printed.

            str:    What it reported on standard error.
        """

        output = StringIO()
        report = StringIO()
        status = 0
        with (
            redirect_stdout(output),
            redirect_stderr(report),
            (
                mock.patch("builtins.input", return_value=answer)
                if answer is not None
                else nullcontext()
            ),
        ):
            try:
                main(list(argv))
            except SystemExit as exit_request:
                status = exit_request.code
        return status, output.getvalue(), report.getvalue()

    def test_commands(self):
        """Commands in turn on one project, each seeing what the last left."""

        engine, project_home = self._copy_project()

        self.assertEqual(
            self._run(project_home, "create", "exp1")[:2],
            (0, "Created experiment exp1: current.\n"),
        )

        # Refused after its run was recorded: the error is recorded against
        # it, for the browser interface to find.
        self._add_live_run(engine, ignored=False)
        status, _, report = self._run(project_home, "create", "exp2")
        self.assertNotEqual(status, 0)
        self.assertIn("Refusing while runs are going", report)
        with Session(engine) as db_session:
            self.assertEqual(
                db_session.execute(
                    select(PipelineRun.task, Error.exception_class).join(
                        PipelineRun, Error.pipeline_run_id == PipelineRun.id
                    )
                ).all(),
                [("create experiment", "ExperimentError")],
            )

        status, listed, _ = self._run(project_home, "list")
        self.assertEqual(status, 0)
        self.assertRegex(
            listed, r"^exp1: current, created [-0-9]+ [:0-9]+ UTC\n$"
        )

        self.assertEqual(
            self._run(project_home, "delete", "exp1", answer="n")[:2], (0, "")
        )
        self.assertTrue(
            os.path.isdir(os.path.join(project_home, "experiments", "exp1"))
        )
        self.assertEqual(
            self._run(project_home, "delete", "exp1", "--yes")[:2],
            (0, "Deleted experiment exp1.\n"),
        )
        self.assertEqual(
            self._run(project_home, "list")[:2], (0, "No experiments.\n")
        )

    def test_no_project(self):
        """A directory without a project is refused, and left without one."""

        home = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, home)
        # The error is recorded in whichever project is open, as a command
        # in a process of its own would find none: not the template.
        set_project_home(self._copy_project()[1])

        status, _, report = self._run(home, "list")

        self.assertNotEqual(status, 0)
        self.assertIn("holds no AutoWISP project", report)
        self.assertEqual(os.listdir(home), [])


if __name__ == "__main__":
    unittest.main()
