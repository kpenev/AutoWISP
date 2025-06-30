#!/usr/bin/env python3

"""Manually fill whichever datbase tables and define default configuration."""

from datetime import datetime

from autowisp.database.interface import db_engine, Session
from autowisp.database.data_model.base import DataModelBase

# False positive due to unusual imports
# pylint: disable=no-name-in-module
from autowisp.database.data_model.provenance import (
    Camera,
    CameraType,
    Mount,
    MountType,
    Telescope,
    TelescopeType,
    Observatory,
    Observer,
)
from autowisp.database.data_model import ObservingSession, Target

# pylint: enable=no-name-in-module

DataModelBase.metadata.bind = db_engine

# False positive
# pylint: disable=no-member
with Session.begin() as db_session:

    camera_type = CameraType(
        make="Canon",
        model="SL1",
        version=0,
        sensor_type="CMOS",
        x_resolution=2604,
        y_resolution=1738,
        pixel_size=4.29,
        notes="PAN001 unit",
        timestamp=datetime.now(),
    )

    # False positive
    # pylint: disable=not-callable
    camera_1 = Camera(
        serial_number="012070048413",
        notes="camera_id is 14d3bd",
        timestamp=datetime.now(),
    )

    camera_2 = Camera(
        serial_number="022071246706",
        notes="camera_id is ee04d1",
        timestamp=datetime.now(),
    )

    mount_type = MountType(
        make="iOptron",
        model="iEQ30 Pro",
        version="1",
        notes="The mount used by PANOPTES project",
        timestamp=datetime.now(),
    )

    mount = Mount(
        serial_number="001", notes="PAN001 observer", timestamp=datetime.now()
    )

    telescope_type = TelescopeType(
        make="Rokinon",
        model="85M-C",
        version="",
        f_ratio=1.4,
        focal_length=85,
        notes="",
        timestamp=datetime.now(),
    )

    telescope = Telescope(
        serial_number="00000", notes="", timestamp=datetime.now()
    )

    observer = Observer(
        name="PAN001",
        email="wtylergee@gmail.com",
        phone="",
        timestamp=datetime.now(),
    )

    observatory = Observatory(
        latitude=19.54,
        longitude=-155.58,
        altitude=3400.0,
        name="Project Panoptes - Mauna Loa Observatory",
        timestamp=datetime.now(),
    )

    target = Target(
        ra=118.181,
        dec=2.5749,
        name="TESS_SEC07_CAM01",
        notes="Target here is the field, RA&DEC is RA_mnt and DEC_mnt",
        timestamp=datetime.now(),
    )

    observing_session_1 = ObservingSession(
        start_time=datetime(2018, 12, 28, 12, 4, 26),
        end_time=datetime(2019, 1, 24, 14, 29, 28),
        notes="PAN001_14d3bd_20190124T115445",
        timestamp=datetime.now(),
    )

    observing_session_2 = ObservingSession(
        start_time=datetime(2018, 12, 28, 9, 40, 19),
        end_time=datetime(2019, 1, 24, 12, 5, 21),
        notes="PAN001_ee04d1_20190124T115445",
        timestamp=datetime.now(),
    )
    # pylint: enable=not-callable

    camera_type.cameras = [camera_1, camera_2]
    mount_type.mounts = [mount]
    telescope_type.telescopes = [telescope]

    observing_session_1.observer = observer
    observing_session_1.camera = camera_1
    observing_session_1.telescope = telescope
    observing_session_1.mount = mount
    observing_session_1.observatory = observatory
    observing_session_1.target = target

    observing_session_2.observer = observer
    observing_session_2.camera = camera_2
    observing_session_2.telescope = telescope
    observing_session_2.mount = mount
    observing_session_2.observatory = observatory
    observing_session_2.target = target

    db_session.add_all(
        [
            camera_type,
            mount_type,
            telescope_type,
            observer,
            observatory,
            target,
            observing_session_1,
            observing_session_2,
        ]
    )
# pylint: enable=no-member
