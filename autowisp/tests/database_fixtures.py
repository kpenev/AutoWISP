"""Rows for tests to build project databases from, with every key satisfied.

Project databases check their foreign keys, so a row a test adds needs the
rows it refers to. Most tests care only about the sessions and their
images, not about who took them with what, and these supply the rest.
"""

from datetime import datetime

from sqlalchemy import MetaData, Table, select, update
from sqlalchemy.orm import Session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import ObservingSession, Target
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
    """

    with Session(engine) as db_session, db_session.begin():
        version = Table(
            "alembic_version",
            MetaData(),
            autoload_with=db_session.connection(),
        )
        db_session.execute(
            update(version).values(version_num="0020_pipeline_run_identity")
        )
