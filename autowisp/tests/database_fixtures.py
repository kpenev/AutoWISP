"""Rows for tests to build project databases from, with every key satisfied.

Project databases check their foreign keys, so a row a test adds needs the
rows it refers to. Most tests care only about the sessions and their
images, not about who took them with what, and these supply the rest.
"""

import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    func,
    inspect as sa_inspect,
    select,
    update,
)
from sqlalchemy.pool import NullPool

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    Experiment,
    Image,
    ImageType,
    ObservingSession,
    Parameter,
    PipelineRun,
    Target,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraChannel,
    CameraType,
    Mount,
    MountType,
    Observatory,
    Observer,
    Telescope,
    TelescopeType,
)

# pylint: enable=no-name-in-module
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import set_project_home, start_db_session
from autowisp.database.pipeline_runs import record_run
from autowisp.database.project_copy import copy_project_database
from autowisp.database.sessions import start_engine_session

# False positive: the declarative models are callable.
# pylint: disable=not-callable


def add_cameras(db_session, channels=(), num_cameras=1):
    """
    Add a camera type with the given channels, and cameras of it.

    Args:
        db_session:    The session to add them in.

        channels(iterable of str):    The names of the type's channels, in
            the order they interleave along x.

        num_cameras(int):    How many cameras of the type to add.

    Returns:
        [int]:    The ids of the cameras.
    """

    channels = list(channels)
    camera_type = CameraType(
        make="Test",
        model=f"{len(channels)}-channel",
        version="1",
        x_resolution=16,
        y_resolution=16,
        pixel_size=1.0,
        notes="",
    )
    db_session.add(camera_type)
    db_session.flush()
    db_session.add_all(
        CameraChannel(
            camera_type_id=camera_type.id,
            name=name,
            x_offset=offset,
            y_offset=0,
            x_step=len(channels),
            y_step=1,
        )
        for offset, name in enumerate(channels)
    )
    cameras = [
        Camera(
            camera_type_id=camera_type.id,
            serial_number=f"{camera_type.id}-{index}",
            notes="",
        )
        for index in range(num_cameras)
    ]
    db_session.add_all(cameras)
    db_session.flush()
    return [camera.id for camera in cameras]


def _add_equipment(db_session):
    """Add an observer, telescope, mount, observatory and target.

    Returns:
        dict:    Their ids, keyed by the observing session's column for each.
    """

    telescope_type = TelescopeType(
        make="Test", model="lens", version="1", f_ratio=2.0, focal_length=50
    )
    mount_type = MountType(make="Test", model="mount", version="1")
    db_session.add_all([telescope_type, mount_type])
    db_session.flush()
    equipment = {
        "observer_id": Observer(name="observer"),
        "telescope_id": Telescope(
            telescope_type_id=telescope_type.id,
            serial_number="telescope",
            notes="",
        ),
        "mount_id": Mount(
            mount_type_id=mount_type.id, serial_number="mount", notes=""
        ),
        "observatory_id": Observatory(
            name="observatory", latitude=0.0, longitude=0.0, altitude=0.0
        ),
        "target_id": Target(name="field"),
    }
    db_session.add_all(equipment.values())
    db_session.flush()
    return {column: row.id for column, row in equipment.items()}


def add_observing_session(db_session, label, camera_id, *, day=0):
    """
    Add an observing session, from 20:00 to 23:00 UTC.

    Every session of a database shares one observer, telescope, mount,
    observatory and target: those of its first session, added along with
    it. The camera is the test's to choose, since cameras are what tests
    tell apart, by their channels.

    Args:
        db_session:    The session to add it in.

        label(str):    The session's label, unique in the database.

        camera_id(int):    The camera it was taken with, e.g. one from
            :func:`add_cameras`.

        day(int):    How many days after 2023-03-01 it is.

    Returns:
        int:    The id of the session.
    """

    first = db_session.scalar(
        select(ObservingSession).order_by(ObservingSession.id).limit(1)
    )
    if first is None:
        equipment = _add_equipment(db_session)
    else:
        equipment = {
            column: getattr(first, column)
            for column in (
                "observer_id",
                "telescope_id",
                "mount_id",
                "observatory_id",
                "target_id",
            )
        }

    session = ObservingSession(
        camera_id=camera_id,
        label=label,
        start_time_utc=datetime(2023, 3, 1 + day, 20, 0, 0),
        end_time_utc=datetime(2023, 3, 1 + day, 23, 0, 0),
        **equipment,
    )
    db_session.add(session)
    db_session.flush()
    return session.id


def stamp_previous_revision(engine):
    """Make the project look one revision behind, changing nothing else.

    It stands in for a project not fully migrated while there is no revision
    after the one creating the ``experiment`` table: that revision finds its
    table there and does nothing, but the project's revision does change.

    The fingerprints its experiments recorded, if it is an original, are
    stamped with it, as if the previous version had created them: the
    revision is all a fingerprint holds of it.
    """

    previous = "0020_pipeline_run_identity"
    with start_engine_session(engine) as db_session:
        version = Table(
            "alembic_version",
            MetaData(),
            autoload_with=db_session.connection(),
        )
        db_session.execute(update(version).values(version_num=previous))
        if not sa_inspect(db_session.connection()).has_table(
            Experiment.__tablename__
        ):
            return
        for experiment in db_session.scalars(select(Experiment)):
            fingerprint = json.loads(experiment.fingerprint)
            if "revision" in fingerprint:
                experiment.fingerprint = json.dumps(
                    dict(fingerprint, revision=previous), sort_keys=True
                )


def add_create_run(db_session):
    """Record the finished run of creating an experiment, and return it.

    Every experiment refers to the run that copied the original into it.
    """

    pipeline_run = PipelineRun(
        host="test-host",
        process_id=1,
        started=datetime(2026, 1, 1),
        finished=datetime(2026, 1, 1),
        task="create experiment",
    )
    db_session.add(pipeline_run)
    db_session.flush()
    return pipeline_run


class ProjectCopyCase(unittest.TestCase):
    """A new project, created once, which each test changes copies of."""

    @classmethod
    def setUpClass(cls):
        """Create the project, leaving it the current one."""

        # Released via addClassCleanup, since it must outlive setUpClass.
        # pylint: disable-next=consider-using-with
        template = tempfile.TemporaryDirectory()
        cls.addClassCleanup(template.cleanup)
        cls._template_home = template.name
        set_project_home(cls._template_home)
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True)
        )

    def _copy_project(self):
        """Return an engine on a copy of the project, and the copy's home.

        The whole home is copied, since creating an experiment copies it.
        """

        copy_home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, copy_home)
        shutil.copytree(self._template_home, copy_home, dirs_exist_ok=True)
        db_fname = os.path.join(copy_home, "autowisp.db")
        # No pool, so that nothing holds the file open when it is removed.
        engine = create_engine(f"sqlite:///{db_fname}", poolclass=NullPool)
        self.addCleanup(engine.dispose)
        return engine, copy_home

    @classmethod
    def _add_experiments(
        cls, engine, states, fingerprint="{}", db_session=None
    ):
        """Record experiments in the given states, ``{name: state}``.

        Each with the finished run that created it, as creating one leaves
        it. Within ``db_session`` if given, else in a transaction of its own
        on ``engine``.
        """

        if db_session is None:
            with start_engine_session(engine) as own_session:
                cls._add_experiments(engine, states, fingerprint, own_session)
            return
        for name, state in states.items():
            db_session.add(
                Experiment(
                    name=name,
                    home=f"{{PROJHOME}}/experiments/{name}",
                    created=func.now(),
                    fingerprint=fingerprint,
                    state=state,
                    copy_run_id=add_create_run(db_session).id,
                )
            )

    @classmethod
    def _add_processed_files(cls):
        """Give the project a raw image inside it, and a file it wrote."""

        with start_db_session() as db_session:
            db_session.add(
                Image(
                    raw_fname="{PROJHOME}/RAW/frame1.fits",
                    image_type_id=db_session.scalar(
                        select(ImageType.id).limit(1)
                    ),
                    observing_session_id=add_observing_session(
                        db_session, "night1", add_cameras(db_session)[0]
                    ),
                )
            )
        for fname in ["RAW/frame1.fits", "DR/frame1.h5"]:
            path = os.path.join(cls._template_home, *fname.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="ascii") as file:
                file.write(fname)

    @staticmethod
    def _add_live_run(engine, ignored):
        """Record a run of this process; return its id if ignored."""

        with start_engine_session(engine) as db_session:
            run_id = record_run(db_session).id
        return {run_id} if ignored else set()

    @staticmethod
    def _configure_outside(engine):
        """Configure the lightcurves outside the project."""

        with start_engine_session(engine) as db_session:
            db_session.execute(
                update(Configuration)
                .where(
                    Configuration.parameter_id
                    == select(Parameter.id)
                    .filter_by(name="lc-fname")
                    .scalar_subquery()
                )
                .values(value=os.path.abspath("{:d}.h5"))
            )

    @staticmethod
    def _copy_then(change):
        """Return the database copy, followed by changing its source."""

        def copy_and_change(source_engine, *args, **kwargs):
            copy_project_database(source_engine, *args, **kwargs)
            change(source_engine)

        return copy_and_change
