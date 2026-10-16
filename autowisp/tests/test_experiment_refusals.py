"""Tests of what refuses the operations on experiments."""

import os
import unittest

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.pipeline_runs import get_machine_id, get_process_started
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import (
    ConfigurationError,
    ExperimentError,
    get_hostname,
)
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    Configuration,
    Experiment,
    Parameter,
    PipelineRun,
)

# pylint: enable=no-name-in-module
from autowisp.tests.database_fixtures import ProjectCopyCase

# False positive: the declarative models are callable.
# pylint: disable=not-callable


class TestRefusals(ProjectCopyCase):
    """What refuses the operations on experiments, and runs in them."""

    def test_active_runs(self):
        """Runs going here and elsewhere refuse, unless ignored by their ids.

        Those found ended stay recorded as finished, though the check refuses.
        """

        engine = self._copy_project()[0]
        this_machine = {
            "host": get_hostname(),
            "machine_id": get_machine_id(),
            "process_id": os.getpid(),
            "process_started": get_process_started(),
        }
        other_machine = {"host": "other-host", "machine_id": "other-machine"}
        # Each run's columns besides its start; all unfinished but the first.
        runs = {
            "finished": {**this_machine, "finished": func.now()},
            "going": this_machine,
            "operation going": {**this_machine, "task": "create experiment"},
            # Its process id since taken by this test's, which started later.
            "ended": {
                **this_machine,
                "process_started": this_machine["process_started"] - 3600,
            },
            "elsewhere": {**other_machine, "process_id": 1},
            "ignored": {**other_machine, "process_id": 2},
        }
        ids = {}
        with start_engine_session(engine) as db_session:
            for name, columns in runs.items():
                pipeline_run = PipelineRun(started=func.now(), **columns)
                db_session.add(pipeline_run)
                db_session.flush()
                ids[name] = pipeline_run.id

        # pylint: disable=protected-access
        with self.assertRaises(ExperimentError) as refusal:
            ExperimentRecords._refuse_active_runs(engine, {ids["ignored"]})
        with Session(engine) as db_session:
            finished = set(
                db_session.scalars(
                    select(PipelineRun.id).where(
                        PipelineRun.finished.is_not(None)
                    )
                )
            )
        # Raises if ignoring fails for any of them.
        ExperimentRecords._refuse_active_runs(engine, set(ids.values()))
        # pylint: enable=protected-access

        self.assertEqual(
            refusal.exception.details["runs"],
            sorted(
                ids[name] for name in ("going", "operation going", "elsewhere")
            ),
        )
        self.assertEqual(finished, {ids["finished"], ids["ended"]})

    def test_external_paths(self):
        """A run in an experiment is refused for each path outside it.

        Under any condition. A run in its original, or in an experiment as
        created, is not.
        """

        outside = os.path.abspath("outside")
        # Each value configured: the parameter, whether the value is added
        # under another condition rather than replacing the parameter's, the
        # value, and whether it is refused.
        configured = [
            ("lc-fname", False, os.path.join(outside, "{:d}.h5"), True),
            (
                "master-mask",
                False,
                "[R:{PROJHOME}/mask_R.fits, "
                f"G:{os.path.join(outside, 'mask_G.fits')}]",
                True,
            ),
            ("subpixmap", False, os.path.join(outside, "subpixmap"), False),
            (
                "epd-statistics-fname",
                True,
                os.path.join(outside, "epd.txt"),
                True,
            ),
            ("tfa-statistics-fname", True, "{PROJHOME}/tfa.txt", False),
        ]

        def configure(db_session):
            condition_id = db_session.scalar(select(func.max(Condition.id)))
            for name, added, value, _ in configured:
                parameter_id = db_session.scalar(
                    select(Parameter.id).filter_by(name=name)
                )
                if added:
                    db_session.add(
                        Configuration(
                            parameter_id=parameter_id,
                            version=0,
                            condition_id=condition_id,
                            value=value,
                        )
                    )
                else:
                    db_session.execute(
                        update(Configuration)
                        .filter_by(parameter_id=parameter_id)
                        .values(value=value)
                    )

        # Each project: whether it is an experiment, whether it is configured
        # as above, and the parameters refused.
        projects = {
            "original": (False, True, []),
            "experiment": (
                True,
                True,
                sorted(name for name, _, _, refused in configured if refused),
            ),
            "experiment as created": (True, False, []),
        }

        refused = {}
        for project, (is_experiment, is_configured, _) in projects.items():
            engine, project_home = self._copy_project()
            if is_configured:
                with start_engine_session(engine) as db_session:
                    configure(db_session)
            if is_experiment:
                Experiment.__table__.drop(engine)
            try:
                ProjectExperiments(engine, project_home).refuse_run()
                refused[project] = []
            except ConfigurationError as error:
                refused[project] = error.details["parameters"]

        self.assertEqual(
            refused,
            {
                project: expected
                for project, (_, _, expected) in projects.items()
            },
        )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
