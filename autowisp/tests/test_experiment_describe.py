"""Tests of describing a project's experiments."""

import subprocess
import sys
import unittest
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.pipeline_runs import get_machine_id, record_run
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import get_hostname

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import ProjectCopyCase

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestDescribe(ProjectCopyCase):
    """Describing experiments: their states, and their operations' runs."""

    @staticmethod
    def _add_experiment(db_session, name, state, pipeline_run):
        """Record an experiment in the given state, referring to the run."""

        db_session.add(pipeline_run)
        db_session.flush()
        db_session.add(
            Experiment(
                name=name,
                home=f"{{PROJHOME}}/experiments/{name}",
                created=func.now(),
                fingerprint="{}",
                state=state,
                run_id=pipeline_run.id,
            )
        )

    def test_describe(self):
        """Each experiment's operation: none, going here or there, or not."""

        engine, project_home = self._copy_project()
        # A process of this machine that has ended: leaving the block waits.
        with subprocess.Popen([sys.executable, "-c", "pass"]) as gone:
            pass
        started = datetime(2026, 1, 1)
        elsewhere = {
            "host": "other-host",
            "machine_id": "other-machine",
            "process_id": 1,
        }
        with start_engine_session(engine) as db_session:
            for name, state, task, columns in [
                ("current", "current", "create", {"finished": started}),
                ("stale", "stale", "reset", {"finished": started}),
                ("reset elsewhere", "creating", "reset", elsewhere),
                (
                    "create crashed",
                    "creating",
                    "create",
                    {
                        "host": get_hostname(),
                        "machine_id": get_machine_id(),
                        "process_id": gone.pid,
                        "process_started": started.timestamp(),
                    },
                ),
                # Its run already recorded as ended, e.g. by merge after its
                # rollback failed, or when found gone after a crash: its
                # process ended, but not the merge.
                (
                    "merge interrupted",
                    "merging",
                    "merge",
                    {"finished": started},
                ),
                ("deleting elsewhere", "deleting", "delete", elsewhere),
            ]:
                self._add_experiment(
                    db_session,
                    name,
                    state,
                    PipelineRun(
                        **{
                            "host": "test-host",
                            "process_id": 1,
                            "started": started,
                            **columns,
                        },
                        task=f"{task} experiment",
                    ),
                )
            self._add_experiment(
                db_session,
                "being created",
                "creating",
                record_run(db_session, "create experiment"),
            )

        self.assertEqual(
            [
                (described["name"], described["state"], described["operation"])
                for described in ExperimentRecords(
                    engine, project_home
                ).describe_experiments()
            ],
            [
                (
                    "being created",
                    "creating",
                    {"task": "create experiment", "running": "here"},
                ),
                (
                    "create crashed",
                    "creating",
                    {"task": "create experiment", "running": None},
                ),
                ("current", "current", None),
                (
                    "deleting elsewhere",
                    "deleting",
                    {"task": "delete experiment", "running": "other-host"},
                ),
                (
                    "merge interrupted",
                    "merging",
                    {"task": "merge experiment", "running": None},
                ),
                (
                    "reset elsewhere",
                    "creating",
                    {"task": "reset experiment", "running": "other-host"},
                ),
                ("stale", "stale", None),
            ],
        )
        # Found gone, so recorded as finished.
        with Session(engine) as db_session:
            self.assertIsNotNone(
                db_session.scalar(
                    select(PipelineRun.finished)
                    .join(Experiment, Experiment.run_id == PipelineRun.id)
                    .filter(Experiment.name == "create crashed")
                )
            )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
