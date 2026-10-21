"""Tests of copying a project's database, between and within backends."""

import os
import shutil
import tempfile
import unittest
import warnings
from argparse import Namespace
from datetime import datetime, timedelta, timezone

from sqlalchemy import MetaData, create_engine, func, select, update
from sqlalchemy.exc import IntegrityError, SAWarning
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from autowisp.database.data_model.base import DataModelBase

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Error,
    Experiment,
    ImageType,
    LightCurveProcessingProgress,
    LightCurveStatus,
    MasterFile,
    MasterType,
    PipelineRun,
    Step,
)
from autowisp.database.data_model.provenance import Camera

# pylint: enable=no-name-in-module
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import (
    create_project_engine,
    set_project_home,
    start_db_session,
)
from autowisp.database.project_copy import copy_project_database
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import DatabaseError
from autowisp.tests import server_test_url
from autowisp.tests.database_fixtures import (
    add_cameras,
    add_create_run,
    add_observing_session,
    stamp_previous_revision,
)
from autowisp.tests.test_database_migration import BackendMixin, on_server

# False positive: the declarative models are callable.
# pylint: disable=not-callable


def _get_contents(engine):
    """Return every row of the database, by table, but its revision.

    Read table by table here rather than through the reader the copy uses,
    so that a test of the copy does not take the reader's word for it.
    """

    metadata = MetaData()
    metadata.reflect(engine)
    with engine.connect() as connection:
        return {
            table.name: [
                tuple(row)
                for row in connection.execute(
                    select(table).order_by(
                        *(
                            list(table.primary_key.columns)
                            or list(table.columns)
                        )
                    )
                )
            ]
            for table in metadata.sorted_tables
            if table.name != "alembic_version"
        }


class TestProjectCopy(BackendMixin, unittest.TestCase):
    """A copy holds exactly what its source does, on either backend."""

    # A time with no fraction of a second, which a server does not keep.
    _known_time = datetime(2026, 1, 2, 3, 4, 5)

    def setUp(self):
        """Create the project to copy, on the backend under test.

        It holds a value of every type the schema uses: strings, doubles,
        both booleans, nulls, timestamps and datetimes, and integers at both
        ends of their ranges, 32-bit and 64-bit.
        """

        super().setUp()
        self._dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._dir)
        server_url = server_test_url()
        set_project_home(self._dir, server_url, new_project=True)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True)
        )
        with start_db_session() as db_session:
            camera_id = add_cameras(db_session, ["R"])[0]
            add_observing_session(db_session, "night1", camera_id)
            run = PipelineRun(
                host="test-host",
                process_id=2**31 - 1,
                started=self._known_time,
                process_started=1767323045.25,
            )
            db_session.add(run)
            db_session.flush()
            db_session.add(
                Error(
                    pipeline_run_id=run.id,
                    component="step",
                    exception_class="TestError",
                    user_message="Failed.",
                    subprocess_id=-(2**31),
                    created=self._known_time,
                )
            )
            master_type_id = db_session.scalar(select(func.min(MasterType.id)))
            masters = [
                MasterFile(
                    type_id=master_type_id,
                    filename=f"{{PROJHOME}}/MASTERS/{name}.fits",
                    enabled=enabled,
                )
                for name, enabled in (("enabled", True), ("disabled", False))
            ]
            db_session.add_all(masters)
            db_session.flush()
            progress = LightCurveProcessingProgress(
                run_id=run.id,
                step_id=db_session.scalar(select(func.min(Step.id))),
                single_photref_id=masters[0].id,
                configuration_version=0,
                started=self._known_time,
                final=False,
            )
            db_session.add(progress)
            db_session.flush()
            db_session.add_all(
                LightCurveStatus(
                    progress_id=progress.id, id=source_id, status=status
                )
                for source_id, status in (
                    (2**63 - 1, 2**31 - 1),
                    (-(2**63), -(2**31)),
                    (6917528997577384320, 0),
                )
            )
            db_session.add(
                Experiment(
                    name="exp1",
                    home="{PROJHOME}/experiments/exp1",
                    created=self._known_time,
                    fingerprint="{}",
                    state="current",
                    run_id=add_create_run(db_session).id,
                )
            )
        self._source = self._create_engine(server_url or "autowisp.db")

    def _create_engine(self, location):
        """Return an engine on a server URL, or on an SQLite file here."""

        engine = create_project_engine(
            location
            if "://" in location
            else f"sqlite:///{os.path.join(self._dir, location)}"
        )
        self.addCleanup(engine.dispose)
        return engine

    def test_round_trip(self):
        """Copied out to SQLite and back in, the project holds what it did.

        Back in, as merging an experiment does: replacing the source's rows,
        but not its record of its experiments, which the copy does not get,
        nor its runs of operations on them.
        """

        before = _get_contents(self._source)
        copy_engine = self._create_engine("copy.db")
        copy_project_database(self._source, copy_engine)
        copied = _get_contents(copy_engine)
        copy_project_database(
            copy_engine,
            self._source,
            replace=True,
            kept_rows={"pipeline_run": ExperimentRecords.is_operation_run},
        )

        self.assertEqual(_get_contents(self._source), before)
        self.assertEqual(
            {name: len(rows) for name, rows in copied.items()},
            {
                name: 0 if name == "experiment" else len(rows)
                for name, rows in before.items()
            },
        )
        # Stamped by the source's database as its rows were created, in UTC
        # whichever the backend, as SQLite's own stamps are.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with Session(copy_engine) as db_session:
            stamps = db_session.scalars(select(Step.timestamp)).all()
        self.assertTrue(stamps)
        for stamp in stamps:
            self.assertLess(abs(stamp - now), timedelta(minutes=10))

    def test_replace_keeps(self):
        """Replacing keeps the target's rows it is told to, with their errors.

        One copy keeps a whole table, as resetting an experiment's processing
        keeps its configuration, and the runs of operations on experiments,
        as merging one keeps the original's. After the source is copied into
        the target, each records a run of each kind, each with an error, and
        labels its own version of the rows both hold, so that which version
        survived shows.

        The target is SQLite, as a reset's always is: the table kept refers
        to rows deleted and inserted again, which only SQLite checks at the
        commit. Replacing a server's rows is the round trip's second copy.
        """

        target_engine = self._create_engine("copy.db")
        copy_project_database(self._source, target_engine)
        with Session(self._source) as db_session:
            create_run_id = db_session.scalar(select(Experiment.run_id))
            processing_run_id = db_session.scalar(
                select(PipelineRun.id).filter_by(
                    task=PipelineRun.processing_task
                )
            )

        # By database, the runs it records after the copy, by id. The ids
        # are distinct, as a merge renumbers them before copying.
        added_runs = {
            "source": {100: "processing", 101: "create experiment"},
            "target": {200: "processing", 201: "create experiment"},
        }
        for label, engine in (
            ("source", self._source),
            ("target", target_engine),
        ):
            with start_engine_session(engine) as db_session:
                for run_id, task in added_runs[label].items():
                    db_session.add(
                        PipelineRun(
                            id=run_id,
                            host=label,
                            process_id=run_id,
                            started=self._known_time,
                            task=task,
                        )
                    )
                    db_session.flush()
                    db_session.add(
                        Error(
                            pipeline_run_id=run_id,
                            component="step",
                            exception_class="TestError",
                            user_message=f"Run {run_id} failed.",
                        )
                    )
                db_session.execute(
                    update(PipelineRun)
                    .where(PipelineRun.id == create_run_id)
                    .values(host=label)
                )
                db_session.execute(
                    update(Camera).values(notes=f"Notes of the {label}.")
                )

        copy_project_database(
            self._source,
            target_engine,
            replace=True,
            kept_tables=["camera"],
            kept_rows={"pipeline_run": ExperimentRecords.is_operation_run},
        )

        with Session(target_engine) as db_session:
            outcome = {
                "runs": {
                    pipeline_run.id: pipeline_run.host
                    for pipeline_run in db_session.scalars(select(PipelineRun))
                },
                "errors": {
                    error.user_message: error.pipeline_run_id
                    for error in db_session.scalars(select(Error))
                },
                "notes": set(db_session.scalars(select(Camera.notes))),
            }
        self.assertEqual(
            outcome,
            {
                # The source's processing runs, and the target's operation
                # runs.
                "runs": {
                    processing_run_id: "test-host",
                    create_run_id: "target",
                    100: "source",
                    201: "target",
                },
                "errors": {
                    "Failed.": processing_run_id,
                    "Run 100 failed.": 100,
                    "Run 201 failed.": 201,
                },
                "notes": {"Notes of the target."},
            },
        )

    def test_ids_continue(self):
        """A row added to a copy gets an id past every row copied."""

        copy_engine = self._create_engine("copy.db")
        copy_project_database(self._source, copy_engine)
        if on_server():
            # Into a new server database too, which takes the ids given.
            self.reset_backend()
            target_engine = self._create_engine(server_test_url())
        else:
            target_engine = self._create_engine("target.db")
        copy_project_database(copy_engine, target_engine)

        added_past_copied = {}
        for name, engine in (("copy", copy_engine), ("target", target_engine)):
            with start_engine_session(engine) as db_session:
                largest = db_session.scalar(select(func.max(ImageType.id)))
                added = ImageType(name="added", description="Added after.")
                db_session.add(added)
                db_session.flush()
                added_past_copied[name] = added.id > largest
        self.assertEqual(added_past_copied, {"copy": True, "target": True})

    def test_refusals(self):
        """Each copy refused says why, and leaves its target as it was."""

        def copy_into_occupied():
            target_engine = self._create_engine("occupied.db")
            copy_project_database(self._source, target_engine)
            return self._source, target_engine

        def copy_unmigrated():
            source_engine = self._create_engine("unmigrated.db")
            copy_project_database(self._source, source_engine)
            stamp_previous_revision(source_engine)
            return source_engine, self._create_engine("from_unmigrated.db")

        def copy_broken_key():
            source_engine = self._create_engine("broken_key.db")
            copy_project_database(self._source, source_engine)
            # Without the check the project's engines make, to break one.
            unchecked = create_engine(source_engine.url, poolclass=NullPool)
            self.addCleanup(unchecked.dispose)
            with start_engine_session(unchecked) as db_session:
                db_session.add(
                    Error(
                        pipeline_run_id=999999,
                        component="step",
                        exception_class="TestError",
                        user_message="Refers to no run.",
                    )
                )
            return source_engine, self._create_engine("from_broken_key.db")

        def copy_long_string():
            # SQLite keeps a string longer than its column, a server does not.
            source_engine = self._create_engine("long_string.db")
            copy_project_database(self._source, source_engine)
            with start_engine_session(source_engine) as db_session:
                db_session.execute(
                    Camera.__table__.update().values(notes="x" * 1001)
                )
            return source_engine, self._source

        # Each case: what sets it up, returning the source and target, and
        # the error it must raise.
        cases = {
            "into a database holding a project": (
                copy_into_occupied,
                DatabaseError,
            ),
            "from a project not fully migrated": (
                copy_unmigrated,
                DatabaseError,
            ),
            "with a row breaking a foreign key": (
                copy_broken_key,
                IntegrityError,
            ),
        }
        if on_server():
            cases["a string too long for a server"] = (
                copy_long_string,
                DatabaseError,
            )

        outcome = {}
        for case, (set_up, _) in cases.items():
            source_engine, target_engine = set_up()
            before = _get_contents(target_engine)
            replace = target_engine is self._source
            try:
                copy_project_database(
                    source_engine,
                    target_engine,
                    replace=replace,
                    kept_rows=(
                        {"pipeline_run": ExperimentRecords.is_operation_run}
                        if replace
                        else None
                    ),
                )
            except (DatabaseError, IntegrityError) as error:
                raised = type(error)
            else:
                raised = None
            outcome[case] = (raised, _get_contents(target_engine) == before)

        self.assertEqual(
            outcome,
            {case: (expected, True) for case, (_, expected) in cases.items()},
        )

    def test_tables_have_an_order(self):
        """Every table can come after those it refers to: no cycles."""

        with warnings.catch_warnings():
            warnings.simplefilter("error", SAWarning)
            self.assertTrue(DataModelBase.metadata.sorted_tables)


# pylint: enable=not-callable

if __name__ == "__main__":
    unittest.main()
