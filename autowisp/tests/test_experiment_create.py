"""Tests of creating an experiment."""

import os
import unittest
from contextlib import nullcontext
from unittest import mock

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import ConfigurationError, ExperimentError
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestCreate(ProjectCopyCase):
    """Creating an experiment: what refuses it, and what it makes."""

    @classmethod
    def setUpClass(cls):
        """Give the project a raw image inside it, and a file it wrote."""

        super().setUpClass()
        cls._add_processed_files()

    @staticmethod
    def _make_directory(project_home, fnames):
        """Make the directory of experiment ``exp1``, holding the files."""

        directory = os.path.join(project_home, "experiments", "exp1")
        os.makedirs(directory)
        for fname in fnames:
            with open(
                os.path.join(directory, fname), "w", encoding="ascii"
            ) as file:
                file.write(fname)

    @staticmethod
    def _delete_experiments(engine):
        """Delete the rows of every experiment, as deleting them does."""

        with start_engine_session(engine) as db_session:
            db_session.execute(delete(Experiment))

    def _try_create(self, prepare, name, change):
        """
        Create an experiment in a copy of the project, as a scenario says.

        Args:
            prepare:    Given the copy's engine and home, prepares it and
                returns the ids of the runs to ignore; or None.

            name(str):    The name to create the experiment under.

            change:    Given the original's engine, changes it once its
                database is copied; or None.

        Returns:
            tuple:    The state create returned, or the name of the class of
            the exception it raised; the state of experiment ``exp1``; what
            its directory holds, None if there is none; and the ids of the
            create runs left unfinished.
        """

        engine, project_home = self._copy_project()
        ignored_runs = (
            prepare(engine, project_home) if prepare else None
        ) or set()
        experiments = ProjectExperiments(engine, project_home)
        with (
            mock.patch(
                "autowisp.project_experiments.copy_project_database",
                self._copy_then(change),
            )
            if change
            else nullcontext()
        ):
            try:
                result = experiments.create(name, ignored_runs)
            except (ExperimentError, ConfigurationError) as error:
                result = type(error).__name__
        directory = os.path.join(project_home, "experiments", "exp1")
        with Session(engine) as db_session:
            unfinished = db_session.scalars(
                select(PipelineRun.id).where(
                    PipelineRun.task == "create experiment",
                    PipelineRun.finished.is_(None),
                )
            ).all()
        return (
            result,
            {
                experiment.name: experiment.state
                for experiment in experiments.get_experiments() or []
            }.get("exp1"),
            sorted(os.listdir(directory)) if os.path.isdir(directory) else None,
            unfinished,
        )

    def test_create(self):
        """Each create on a copy of its own: refused, undone or completed."""

        copied = sorted(os.listdir(self._template_home))
        # Each scenario: what prepares its copy of the project, given its
        # engine and home and returning the ids of the runs to ignore; the
        # name given; what changes the original once its database is copied;
        # and the result, the experiment's state, and what its directory
        # holds.
        test_scenarios = {
            "created": (None, "exp1", None, "current", "current", copied),
            "into an empty directory": (
                lambda _, project_home: self._make_directory(project_home, []),
                "exp1",
                None,
                "current",
                "current",
                copied,
            ),
            "invalid name": (
                None,
                "../exp1",
                None,
                "ExperimentError",
                None,
                None,
            ),
            "name taken": (
                lambda engine, _: self._add_experiments(
                    engine, {"exp1": "stale"}
                ),
                "exp1",
                None,
                "ExperimentError",
                "stale",
                None,
            ),
            "directory not empty": (
                lambda _, project_home: self._make_directory(
                    project_home, ["notes.txt"]
                ),
                "exp1",
                None,
                "ExperimentError",
                None,
                ["notes.txt"],
            ),
            "in an experiment": (
                lambda engine, _: Experiment.__table__.drop(engine),
                "exp1",
                None,
                "ExperimentError",
                None,
                None,
            ),
            "path outside": (
                lambda engine, _: self._configure_outside(engine),
                "exp1",
                None,
                "ConfigurationError",
                None,
                None,
            ),
            # Undone: nothing is left of it.
            "run going": (
                lambda engine, _: self._add_live_run(engine, ignored=False),
                "exp1",
                None,
                "ExperimentError",
                None,
                None,
            ),
            "run going, ignored": (
                lambda engine, _: self._add_live_run(engine, ignored=True),
                "exp1",
                None,
                "current",
                "current",
                copied,
            ),
            "migrated during the copy": (
                None,
                "exp1",
                stamp_previous_revision,
                "stale",
                "stale",
                copied,
            ),
            "deleted during the copy": (
                None,
                "exp1",
                self._delete_experiments,
                "ExperimentError",
                None,
                copied,
            ),
        }

        self.assertEqual(
            {
                scenario: self._try_create(prepare, name, change)
                for scenario, (
                    prepare,
                    name,
                    change,
                    *_,
                ) in test_scenarios.items()
            },
            {
                scenario: (result, state, contents, [])
                for scenario, (
                    *_,
                    result,
                    state,
                    contents,
                ) in test_scenarios.items()
            },
        )

    def test_created_copy(self):
        """The copy is an experiment, links the raw images, and is finished.

        The run creating it, which its database holds, is recorded there as
        finished too.
        """

        engine, project_home = self._copy_project()
        ProjectExperiments(engine, project_home).create("exp1")
        experiment_home = os.path.join(project_home, "experiments", "exp1")
        experiment_engine = create_engine(
            f"sqlite:///{os.path.join(experiment_home, 'autowisp.db')}",
            poolclass=NullPool,
        )
        self.addCleanup(experiment_engine.dispose)
        # No experiment table: the copy is an experiment.
        self.assertIsNone(
            ExperimentRecords(
                experiment_engine, experiment_home
            ).get_experiments()
        )
        self.assertTrue(
            os.path.samefile(
                os.path.join(project_home, "RAW", "frame1.fits"),
                os.path.join(experiment_home, "RAW", "frame1.fits"),
            )
        )
        with Session(experiment_engine) as db_session:
            self.assertEqual(
                [
                    pipeline_run.finished is not None
                    for pipeline_run in db_session.scalars(select(PipelineRun))
                ],
                [True],
            )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
