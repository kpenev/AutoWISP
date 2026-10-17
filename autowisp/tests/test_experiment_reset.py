"""Tests of resetting an experiment's processing."""

import os
import unittest
from contextlib import nullcontext
from unittest import mock

from sqlalchemy import create_engine, delete, func, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import ConfigurationError, ExperimentError
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    Configuration,
    Experiment,
    ImageType,
    MasterType,
    Parameter,
    PipelineRun,
)

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestReset(ProjectCopyCase):
    """Resetting an experiment: what refuses it, and what it replaces."""

    @classmethod
    def setUpClass(cls):
        """Give the project a raw image inside it, and a file it wrote."""

        super().setUpClass()
        cls._add_processed_files()

    def _create_experiment(self):
        """
        Create experiment ``exp1`` in a copy of the project.

        Returns:
            tuple:    The original's engine and home, and the experiment's.
        """

        engine, project_home = self._copy_project()
        ProjectExperiments(engine, project_home).create("exp1")
        experiment_home = os.path.join(project_home, "experiments", "exp1")
        experiment_engine = create_engine(
            f"sqlite:///{os.path.join(experiment_home, 'autowisp.db')}",
            poolclass=NullPool,
        )
        self.addCleanup(experiment_engine.dispose)
        return engine, project_home, experiment_engine, experiment_home

    @staticmethod
    def _mark_incomplete(engine):
        """Mark every experiment as still being created."""

        with start_engine_session(engine) as db_session:
            db_session.execute(update(Experiment).values(state="creating"))

    @staticmethod
    def _delete_master_conditions(experiment_engine):
        """Delete the conditions the master types use from the experiment."""

        with start_engine_session(experiment_engine) as db_session:
            db_session.execute(
                delete(Condition).where(
                    Condition.id.in_(select(MasterType.condition_id))
                )
            )

    @staticmethod
    def _read_tree(home):
        """Return the files under a home, but its database and experiments."""

        contents = {}
        for directory, subdirs, fnames in os.walk(home):
            subdirs[:] = [name for name in subdirs if name != "experiments"]
            for fname in fnames:
                path = os.path.join(directory, fname)
                if path != os.path.join(home, "autowisp.db"):
                    with open(path, "rb") as file:
                        contents[os.path.relpath(path, home)] = file.read()
        return contents

    def _try_reset(self, prepare, name, change):
        """
        Reset an experiment in a copy of the project, as a scenario says.

        Args:
            prepare:    Given the original's engine and home, and the
                experiment's, prepares them and returns the ids of the runs
                to ignore; or None.

            name(str):    The name of the experiment to reset.

            change:    Given the original's engine, changes it once its
                database is copied; or None.

        Returns:
            tuple:    The state reset returned, or the name of the class of
            the exception it raised; the state of experiment ``exp1``; the
            task of the run it refers to; and the ids of the operations' runs
            left unfinished.
        """

        engine, project_home, *experiment = self._create_experiment()
        ignored_runs = (
            prepare(engine, project_home, *experiment) if prepare else None
        ) or set()
        with (
            mock.patch(
                "autowisp.project_experiments.copy_project_database",
                self._copy_then(change),
            )
            if change
            else nullcontext()
        ):
            try:
                result = ProjectExperiments(engine, project_home).reset(
                    name, ignored_runs
                )
            except (ExperimentError, ConfigurationError) as error:
                result = type(error).__name__
        with Session(engine) as db_session:
            row = db_session.scalars(select(Experiment)).one()
            return (
                result,
                row.state,
                db_session.get(PipelineRun, row.copy_run_id).task,
                db_session.scalars(
                    select(PipelineRun.id).where(
                        PipelineRun.task != PipelineRun.processing_task,
                        PipelineRun.finished.is_(None),
                    )
                ).all(),
            )

    def test_reset(self):
        """Each reset on a copy of its own: refused, restored or completed."""

        # Each scenario: what prepares the copy, given the original's engine
        # and home and the experiment's engine and home, returning the ids
        # of the runs to ignore; the name given; what changes the original
        # once its database is copied; and the result, the experiment's
        # state, and the task of the run it refers to.
        test_scenarios = {
            "reset": (
                None,
                "exp1",
                None,
                "current",
                "current",
                "reset experiment",
            ),
            "stale": (
                lambda engine, project_home, *_: ExperimentRecords(
                    engine, project_home
                ).unlock()
                and set(),
                "exp1",
                None,
                "current",
                "current",
                "reset experiment",
            ),
            # The original's configuration is not the one reset keeps.
            "path outside, in the original": (
                lambda engine, *_: self._configure_outside(engine),
                "exp1",
                None,
                "current",
                "current",
                "reset experiment",
            ),
            "no such experiment": (
                None,
                "exp2",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "incomplete": (
                lambda engine, *_: self._mark_incomplete(engine),
                "exp1",
                None,
                "ExperimentError",
                "creating",
                "create experiment",
            ),
            "database gone": (
                lambda _, __, ___, experiment_home: os.remove(
                    os.path.join(experiment_home, "autowisp.db")
                ),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "original not migrated": (
                lambda engine, *_: stamp_previous_revision(engine),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "experiment not migrated": (
                lambda _, __, experiment_engine, ___: stamp_previous_revision(
                    experiment_engine
                ),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "path outside, in the experiment": (
                lambda _, __, experiment_engine, ___: self._configure_outside(
                    experiment_engine
                ),
                "exp1",
                None,
                "ConfigurationError",
                "current",
                "create experiment",
            ),
            "condition missing": (
                lambda _, __, experiment_engine, ___: (
                    self._delete_master_conditions(experiment_engine)
                ),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            # Restored: the experiment refers to its create again.
            "run going in the original": (
                lambda engine, *_: self._add_live_run(engine, ignored=False),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "run going in the experiment": (
                lambda _, __, experiment_engine, ___: self._add_live_run(
                    experiment_engine, ignored=False
                ),
                "exp1",
                None,
                "ExperimentError",
                "current",
                "create experiment",
            ),
            "run going, ignored": (
                lambda engine, *_: self._add_live_run(engine, ignored=True),
                "exp1",
                None,
                "current",
                "current",
                "reset experiment",
            ),
            "migrated during the copy": (
                None,
                "exp1",
                stamp_previous_revision,
                "stale",
                "stale",
                "reset experiment",
            ),
        }

        self.assertEqual(
            {
                scenario: self._try_reset(prepare, name, change)
                for scenario, (
                    prepare,
                    name,
                    change,
                    *_,
                ) in test_scenarios.items()
            },
            {
                scenario: (result, state, task, [])
                for scenario, (
                    *_,
                    result,
                    state,
                    task,
                ) in test_scenarios.items()
            },
        )

    def test_reset_copy(self):
        """The experiment's files and rows become the original's again.

        All but its configuration, which it keeps: a reset starts its
        processing over with the configuration it was trying.
        """

        engine, project_home, experiment_engine, experiment_home = (
            self._create_experiment()
        )
        experiments = ProjectExperiments(engine, project_home)
        # Processing in the experiment, with its configuration changed.
        for fname in ["frame1.h5", "frame2.h5"]:
            with open(
                os.path.join(experiment_home, "DR", fname),
                "w",
                encoding="ascii",
            ) as file:
                file.write("Processed in the experiment.")
        with start_engine_session(experiment_engine) as db_session:
            db_session.add(
                PipelineRun(
                    host="test-host",
                    process_id=1,
                    started=func.now(),
                    finished=func.now(),
                )
            )
            db_session.execute(
                update(Configuration)
                .where(
                    Configuration.parameter_id
                    == select(Parameter.id)
                    .filter_by(name="verbose")
                    .scalar_subquery()
                )
                .values(value="debug")
            )
        # The original changed meanwhile, once unlocked.
        experiments.unlock()
        with start_engine_session(engine) as db_session:
            db_session.add(
                ImageType(name="added", description="Added after the copy.")
            )

        self.assertEqual(experiments.reset("exp1"), "current")

        self.assertEqual(
            self._read_tree(experiment_home), self._read_tree(project_home)
        )
        self.assertTrue(
            os.path.samefile(
                os.path.join(project_home, "RAW", "frame1.fits"),
                os.path.join(experiment_home, "RAW", "frame1.fits"),
            )
        )
        # No experiment table: still an experiment.
        self.assertIsNone(
            ExperimentRecords(
                experiment_engine, experiment_home
            ).get_experiments()
        )
        with Session(experiment_engine) as db_session:
            self.assertEqual(
                sorted(
                    (pipeline_run.task, pipeline_run.finished is not None)
                    for pipeline_run in db_session.scalars(select(PipelineRun))
                ),
                [("create experiment", True), ("reset experiment", True)],
            )
            self.assertEqual(
                db_session.scalars(
                    select(Configuration.value).where(
                        Configuration.parameter_id
                        == select(Parameter.id)
                        .filter_by(name="verbose")
                        .scalar_subquery()
                    )
                ).all(),
                ["debug"],
            )
            self.assertIsNotNone(
                db_session.scalars(
                    select(ImageType).filter_by(name="added")
                ).one_or_none()
            )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
