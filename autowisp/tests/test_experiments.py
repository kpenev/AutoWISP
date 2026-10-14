"""Tests of a project's record of its experiments, and of its fingerprint."""

import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime

from sqlalchemy import (
    DDL,
    MetaData,
    Table,
    create_engine,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.experiments import ProjectExperiments
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticExpression,
    Error,
    Experiment,
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
from autowisp.tests.database_fixtures import add_cameras

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


class TestFingerprint(unittest.TestCase):
    """Exactly the changes a merge must not lose change the fingerprint."""

    @classmethod
    def setUpClass(cls):
        """Create a project with a row in each table the changes touch."""

        # Released via addClassCleanup, since it must outlive setUpClass.
        # pylint: disable-next=consider-using-with
        template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(template.cleanup)
        cls._template_home = template.name
        set_project_home(cls._template_home)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True)
        )
        with start_db_session() as db_session:
            add_cameras(db_session)
            run = PipelineRun(
                host="test-host", process_id=1, started=func.now()
            )
            db_session.add(run)
            db_session.flush()
            db_session.add(
                Error(
                    pipeline_run_id=run.id,
                    component="step",
                    exception_class="TestError",
                    user_message="Failed.",
                )
            )

    def _copy_project(self):
        """Return an engine on a copy of the project, and the copy's home."""

        copy_home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, copy_home)
        db_fname = os.path.join(copy_home, "autowisp.db")
        shutil.copy(os.path.join(self._template_home, "autowisp.db"), db_fname)
        # No pool, so that nothing holds the file open when it is removed.
        engine = create_engine(f"sqlite:///{db_fname}", poolclass=NullPool)
        self.addCleanup(engine.dispose)
        return engine, copy_home

    @staticmethod
    def _get_fingerprint(engine, project_home):
        """Return the fingerprint of the given project."""

        with engine.connect() as connection:
            # pylint: disable-next=protected-access
            return ProjectExperiments(engine, project_home)._get_fingerprint(
                connection
            )

    def test_changes_seen(self):
        """Each change, made to its own copy, changes only what it must."""

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

        def stamp_previous_revision(db_session):
            version = Table(
                "alembic_version",
                MetaData(),
                autoload_with=db_session.connection(),
            )
            return db_session.execute(
                update(version).values(version_num="0020_pipeline_run_identity")
            )

        def restamp(db_session):
            # Without the trigger the timestamp certainly changes, rather
            # than being set to the second it already holds.
            _drop_timestamp_trigger(db_session, Parameter)
            return db_session.execute(
                update(Parameter).values(timestamp=datetime(2000, 1, 1))
            )

        # Each change: the function making it, and what it must change. The
        # function returns the result of its update or delete, if it has one.
        changes = {
            "value updated within the second": (
                update_within_the_second,
                {"step"},
            ),
            "row deleted": (
                lambda db_session: db_session.execute(
                    delete(DiagnosticExpression)
                ),
                {"diagnostic_expression"},
            ),
            "row inserted": (
                lambda db_session: db_session.add(
                    ImageType(name="test", description="Added by the test.")
                ),
                {"image_type"},
            ),
            "row deleted without a primary key": (
                delete_unkeyed_row,
                {"step_parameters"},
            ),
            "notes edited": (
                lambda db_session: db_session.execute(
                    update(Camera).values(notes="Edited.")
                ),
                {"camera"},
            ),
            "run finished": (
                lambda db_session: db_session.execute(
                    update(PipelineRun).values(finished=func.now())
                ),
                {"pipeline_run"},
            ),
            "migrated": (stamp_previous_revision, {"revision"}),
            "restamped": (restamp, set()),
            "error added": (
                lambda db_session: db_session.add(
                    Error(
                        component="bui",
                        exception_class="TestError",
                        user_message="Failed.",
                    )
                ),
                set(),
            ),
            "error resolved": (
                lambda db_session: db_session.execute(
                    update(Error).values(resolved=func.now())
                ),
                set(),
            ),
            "experiment added": (
                lambda db_session: db_session.add(
                    Experiment(
                        name="exp1",
                        home="{PROJHOME}/experiments/exp1",
                        created=func.now(),
                        fingerprint="{}",
                        state="creating",
                    )
                ),
                set(),
            ),
        }

        original = self._get_fingerprint(*self._copy_project())
        changed = {}
        for change, (make_change, _) in changes.items():
            engine, project_home = self._copy_project()
            with Session(engine) as db_session, db_session.begin():
                result = make_change(db_session)
                if result is not None:
                    self.assertGreater(
                        result.rowcount, 0, f"{change!r} changed nothing."
                    )
            fingerprint = self._get_fingerprint(engine, project_home)
            changed[change] = {
                name
                for name in set(original["tables"]) | set(fingerprint["tables"])
                if original["tables"].get(name)
                != fingerprint["tables"].get(name)
            }
            if fingerprint["revision"] != original["revision"]:
                changed[change].add("revision")

        self.assertEqual(
            changed,
            {change: expected for change, (_, expected) in changes.items()},
        )


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
