"""Tests of the lock a project's experiments hold on it."""

import os
import shutil
import tempfile
import unittest
from argparse import Namespace

from sqlalchemy import select
from sqlalchemy.orm import Session

from autowisp import run_pipeline
from autowisp.error_context import get_error_context, set_error_context
from autowisp.exceptions import ExperimentError, ProjectLockedError
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import ProjectCopyCase

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestProjectLock(ProjectCopyCase):
    """A project is locked while it has experiments that are not stale."""

    @staticmethod
    def _get_locking(experiments):
        """Return the states of the experiments a refusal names, by name."""

        try:
            experiments.refuse_if_locked()
        except ProjectLockedError as error:
            return error.details["experiments"]
        return {}

    def test_lock_and_unlock(self):
        """Experiments lock the project until it is unlocked, in most states.

        Unlocking frees it of the current ones, while one being created or
        merged keeps it locked, and one stale or being deleted never locks
        it. Only the current and stale ones may be opened, each in its own
        home; a name with no experiment, or a project that is an experiment,
        is refused.
        """

        # Each project: the states of its experiments, None for a project
        # that is itself an experiment, those that must lock it, and those
        # that must still lock it once unlocked.
        projects = {
            "an experiment": (None, {}, {}),
            "without experiments": ({}, {}, {}),
            "with stale experiments": (
                {"exp1": "stale", "exp2": "stale"},
                {},
                {},
            ),
            "with experiments in every state": (
                {
                    "exp1": "current",
                    "exp2": "creating",
                    "exp3": "stale",
                    "exp4": "merging",
                    "exp5": "deleting",
                },
                {"exp1": "current", "exp2": "creating", "exp4": "merging"},
                {"exp2": "creating", "exp4": "merging"},
            ),
        }

        locking = {}
        planned = {}
        unlocked = {}
        locking_after = {}
        openable = {}
        for project, (states, _, _) in projects.items():
            engine, project_home = self._copy_project()
            if states is None:
                Experiment.__table__.drop(engine)
            else:
                self._add_experiments(engine, states)
            experiments = ProjectExperiments(engine, project_home)
            # Each experiment's home relative to the project's, if it may be
            # opened; also asked for one there is none of.
            openable[project] = {}
            for name in [*(states or {}), "missing"]:
                try:
                    openable[project][name] = os.path.relpath(
                        experiments.get_openable_home(name), project_home
                    )
                except ExperimentError:
                    pass
            locking[project] = self._get_locking(experiments)
            planned[project] = experiments.plan_unlock()
            unlocked[project] = experiments.unlock()
            locking_after[project] = self._get_locking(experiments)

        self.assertEqual(
            openable,
            {
                "an experiment": {},
                "without experiments": {},
                "with stale experiments": {
                    name: os.path.join("experiments", name)
                    for name in ["exp1", "exp2"]
                },
                "with experiments in every state": {
                    name: os.path.join("experiments", name)
                    for name in ["exp1", "exp3"]
                },
            },
        )

        self.assertEqual(
            locking,
            {project: before for project, (_, before, _) in projects.items()},
        )
        self.assertEqual(
            unlocked,
            {
                project: sorted(set(before) - set(after))
                for project, (_, before, after) in projects.items()
            },
        )
        self.assertEqual(
            locking_after,
            {project: after for project, (_, _, after) in projects.items()},
        )
        # Confirming asks first: what it says matches what unlocking did.
        self.assertEqual(
            planned,
            {
                project: (unlocked[project], locking_after[project])
                for project in projects
            },
        )

    def test_run_refused(self):
        """A run in a locked project is refused as it starts, and ended."""

        engine, project_home = self._copy_project()
        self._add_experiments(engine, {"exp1": "current"})
        image_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, image_dir)
        # The run makes itself the context of every error recorded later,
        # which a test of another project must not inherit.
        self.addCleanup(set_error_context, get_error_context())

        # No steps, as the BUI adds images: only the check at the start of
        # the run can refuse it.
        with self.assertRaises(ProjectLockedError):
            run_pipeline.main(
                Namespace(
                    project_home=project_home,
                    add_raw_images=[image_dir],
                    steps=[],
                    step_imtypes=[],
                )
            )
        # The refused run, not the one that created the experiment.
        with Session(engine) as db_session:
            self.assertEqual(
                [
                    run.finished is not None
                    for run in db_session.scalars(
                        select(PipelineRun).filter_by(
                            task=PipelineRun.processing_task
                        )
                    )
                ],
                [True],
            )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
