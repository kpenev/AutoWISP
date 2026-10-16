"""Tests of what makes a project's experiments stale."""

import unittest
from datetime import datetime

from sqlalchemy import DDL, delete, func, select, update

from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import start_db_session
from autowisp.database.migrate import migrate_project
from autowisp.database.sessions import start_engine_session
from autowisp.project_experiments import ProjectExperiments

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticExpression,
    Error,
    ImageType,
    Parameter,
    PipelineRun,
    Step,
)
from autowisp.database.data_model.provenance import Camera

# pylint: enable=no-name-in-module
from autowisp.database.data_model.steps_and_parameters import (
    step_param_association,
)
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    add_cameras,
    stamp_previous_revision,
)

# False positive: the declarative models are callable.
# pylint: disable=not-callable


def _drop_timestamp_trigger(db_session, model):
    """Stop the database restamping the rows of the model's table.

    Triggers have no ORM form; :mod:`autowisp.database.data_model` creates
    them as DDL too.
    """

    db_session.connection().execute(
        DDL(f"DROP TRIGGER update_{model.__tablename__}_timestamp")
    )


class TestStaleness(ProjectCopyCase):
    """Migrating marks stale the experiments a merge would lose changes of."""

    @classmethod
    def setUpClass(cls):
        """Give the project a row in each table the changes touch."""

        super().setUpClass()
        with start_db_session() as db_session:
            add_cameras(db_session)
            # Unfinished, on another host: creating an experiment ignores it,
            # as a user would a run known to have died there.
            run = PipelineRun(
                host="test-host", process_id=1, started=func.now()
            )
            db_session.add(run)
            db_session.flush()
            cls._dead_run_id = run.id
            db_session.add(
                Error(
                    pipeline_run_id=run.id,
                    component="step",
                    exception_class="TestError",
                    user_message="Failed.",
                )
            )

    def test_changes_seen(self):
        """Each change, made to its own copy, makes stale only what it must.

        Every copy holds an experiment of each state: the current one,
        created as the project stood, becomes stale if the change is one a
        merge would lose, while the others keep their states whatever it is.
        """

        def update_within_the_second(db_session):
            # Without the trigger the timestamp stays as it was, as it does
            # for an update within the same second.
            _drop_timestamp_trigger(db_session, Step)
            return db_session.execute(
                update(Step)
                .where(Step.id == select(func.min(Step.id)).scalar_subquery())
                .values(description=Step.description + " (edited)")
            )

        def delete_unkeyed_row(db_session):
            first = db_session.execute(
                select(step_param_association)
                .order_by(*step_param_association.columns)
                .limit(1)
            ).one()
            return db_session.execute(
                delete(step_param_association).where(
                    step_param_association.c.step_id == first.step_id,
                    step_param_association.c.param_id == first.param_id,
                )
            )

        def restamp(db_session):
            # Without the trigger the timestamp certainly changes, rather
            # than being set to the second it already holds.
            _drop_timestamp_trigger(db_session, Parameter)
            return db_session.execute(
                update(Parameter).values(timestamp=datetime(2000, 1, 1))
            )

        # Each change: the function making it, and whether a merge would
        # lose it, making the current experiment stale. The function returns
        # the result of its update or delete, if it has one.
        changes = {
            "value updated within the second": (update_within_the_second, True),
            "row deleted": (
                lambda db_session: db_session.execute(
                    delete(DiagnosticExpression)
                ),
                True,
            ),
            "row inserted": (
                lambda db_session: db_session.add(
                    ImageType(name="test", description="Added by the test.")
                ),
                True,
            ),
            "row deleted without a primary key": (delete_unkeyed_row, True),
            "notes edited": (
                lambda db_session: db_session.execute(
                    update(Camera).values(notes="Edited.")
                ),
                True,
            ),
            "run finished": (
                lambda db_session: db_session.execute(
                    update(PipelineRun).values(finished=func.now())
                ),
                True,
            ),
            "restamped": (restamp, False),
            "error added": (
                lambda db_session: db_session.add(
                    Error(
                        component="bui",
                        exception_class="TestError",
                        user_message="Failed.",
                    )
                ),
                False,
            ),
            "error resolved": (
                lambda db_session: db_session.execute(
                    update(Error).values(resolved=func.now())
                ),
                False,
            ),
            "experiment added, with the run creating it": (
                lambda db_session: self._add_experiments(
                    None, {"exp4": "creating"}, db_session=db_session
                ),
                False,
            ),
            "processing run added": (
                lambda db_session: db_session.add(
                    PipelineRun(
                        host="test-host", process_id=2, started=func.now()
                    )
                ),
                True,
            ),
        }
        states = {"exp1": "current", "exp2": "stale", "exp3": "creating"}

        outcome = {}
        for change, (make_change, _) in changes.items():
            engine, project_home = self._copy_project()
            ProjectExperiments(engine, project_home).create(
                "exp1", {self._dead_run_id}
            )
            self._add_experiments(
                engine, {"exp2": states["exp2"], "exp3": states["exp3"]}
            )
            stamp_previous_revision(engine)
            with start_engine_session(engine) as db_session:
                result = make_change(db_session)
                if result is not None:
                    self.assertGreater(
                        result.rowcount, 0, f"{change!r} changed nothing."
                    )

            migration = migrate_project(engine, project_home)
            outcome[change] = (
                {
                    experiment.name: experiment.state
                    for experiment in ExperimentRecords(
                        engine, project_home
                    ).get_experiments()
                    if experiment.name in states
                },
                migration["stale"],
            )

        self.assertEqual(
            outcome,
            {
                change: (
                    dict(states, exp1="stale" if seen else "current"),
                    ["exp1"] if seen else [],
                )
                for change, (_, seen) in changes.items()
            },
        )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
