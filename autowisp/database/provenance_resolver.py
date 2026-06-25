"""Resolve FITS-header provenance to ORM rows shared by the survey database.

Both the DB-backed pipeline (via
:mod:`autowisp.processing_steps.add_images_to_db`) and the manual /
individual-step CLI need to match FITS headers against the survey provenance
tables (``Camera``, ``Telescope``, ``Mount``, ``Observatory``, ``Observer``,
``Target``) and resolve the corresponding ``ObservingSession`` row. This
module centralises that logic so both processing paths use the same
header-expression flag set and the same get-or-create semantics, producing a
``/Provenance`` group that is byte-identical between manual and DB-backed
runs.

The helpers here intentionally do **not** create any ``Image`` rows -- that
remains the responsibility of ``add_images_to_db``.
"""

from datetime import timedelta
import logging

from astropy import units
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model.provenance import (
    Observer,
    Camera,
    Telescope,
    Mount,
    Observatory,
)
from autowisp.database.data_model import ObservingSession, Target

# pylint: enable=no-name-in-module

from autowisp.exceptions import PipelineError

_logger = logging.getLogger(__name__)


def get_or_create_target(
    image_type, header_eval, configuration, db_session, field_of_view
):
    """Return the target corresponding to the image (create if necessary)."""

    target_name = header_eval(configuration["target_name"])
    db_target = (
        db_session.query(Target).filter_by(name=target_name).one_or_none()
    )
    no_pointing_imtypes = ["zero", "dark", "flat"]
    if image_type in no_pointing_imtypes:
        image_target = {"ra": None, "dec": None}
    else:
        image_target = {
            "ra": header_eval(configuration["target_ra"]),
            "dec": header_eval(configuration["target_dec"]),
        }

    if db_target is None:
        # False positive
        # pylint: disable=not-callable
        db_target = Target(
            **image_target, name=header_eval(configuration["target_name"])
        )
        # pylint: enable=not-callable
        db_session.add(db_target)
    elif image_type not in no_pointing_imtypes:
        image_target = SkyCoord(
            image_target["ra"] * units.deg, image_target["dec"] * units.deg
        )
        _logger.debug(
            "Checking target %s for %s image. From DB: %s vs image: %s",
            target_name,
            repr(image_type),
            repr(db_target),
            repr(image_target),
        )
        assert (
            image_target.separation(
                SkyCoord(
                    ra=db_target.ra * units.deg, dec=db_target.dec * units.deg
                )
            )
            < configuration["target_match_tolerance"] * field_of_view
        )

    return db_target


def _match_observatory(db_observatory, image_location):
    """True iff the observatory matches the image location."""

    db_location = EarthLocation(
        lat=db_observatory.latitude * units.deg,
        lon=db_observatory.longitude * units.deg,
        height=db_observatory.altitude * units.m,
    )
    return (
        (image_location.x - db_location.x) ** 2
        + (image_location.y - db_location.y) ** 2
        + (image_location.z - db_location.z) ** 2
    ) ** 0.5 < 100 * units.km


def get_observatory(header_eval, configuration, db_session):
    """Return the observatory corresponding to the image (must exist)."""

    _logger.debug(
        "Observatory location: %s", repr(configuration["observatory_location"])
    )
    latitude, longitude, altitude = (
        header_eval(expression)
        for expression in configuration["observatory_location"]
    )
    image_location = EarthLocation(
        lat=latitude * units.deg,
        lon=longitude * units.deg,
        height=altitude * units.m,
    )

    if configuration["observatory"] is None:
        observatory = None
        for db_observatory in db_session.query(Observatory).all():
            if _match_observatory(db_observatory, image_location):
                assert observatory is None
                observatory = db_observatory
    else:
        observatory = (
            db_session.query(Observatory)
            .filter_by(name=header_eval(configuration["observatory"]))
            .one()
        )
        assert _match_observatory(observatory, image_location)

    return observatory


def get_or_create_observing_session(
    image_type, header_eval, configuration, db_session
):
    """Return the observing session the image is part of (create if needed)."""

    observer = (
        db_session.query(Observer)
        .filter_by(name=header_eval(configuration["observer"]))
        .one()
    )
    camera = (
        db_session.query(Camera)
        .filter_by(
            serial_number=header_eval(configuration["camera_serial_number"])
        )
        .one()
    )
    telescope = (
        db_session.query(Telescope)
        .filter_by(
            serial_number=header_eval(configuration["telescope_serial_number"])
        )
        .one()
    )
    mount = (
        db_session.query(Mount)
        .filter_by(
            serial_number=header_eval(configuration["mount_serial_number"])
        )
        .one()
    )
    observatory = get_observatory(header_eval, configuration, db_session)
    field_of_view = (
        max(camera.camera_type.x_resolution, camera.camera_type.y_resolution)
        * camera.camera_type.pixel_size
        * units.um
        / (telescope.telescope_type.focal_length * units.mm)
    ) * units.rad
    target = get_or_create_target(
        image_type, header_eval, configuration, db_session, field_of_view
    )
    exposure_start = None
    for time_format in ("utc", "jd"):
        if configuration[f"exposure_start_{time_format}"]:
            exposure_start = Time(
                header_eval(configuration[f"exposure_start_{time_format}"]),
                format=None if time_format == "utc" else time_format,
            )
            header_eval.symtable["JD-OBS"] = exposure_start.jd + header_eval(
                configuration["exposure_seconds"]
            ) / (2.0 * 24.0 * 3600.0)
            exposure_start = exposure_start.utc.to_value("datetime")
    assert exposure_start is not None
    exposure_end = exposure_start + timedelta(
        seconds=header_eval(configuration["exposure_seconds"])
    )

    result = (
        db_session.query(ObservingSession)
        .filter_by(label=header_eval(configuration["observing_session_label"]))
        .one_or_none()
    )
    if result is None:
        result = ObservingSession(
            observer_id=observer.id,
            camera_id=camera.id,
            telescope_id=telescope.id,
            mount_id=mount.id,
            observatory_id=observatory.id,
            target_id=target.id,
            label=header_eval(configuration["observing_session_label"]),
            start_time_utc=exposure_start,
            end_time_utc=exposure_end,
        )
        db_session.add(result)
    else:
        if any(
            [
                result.observer_id != observer.id,
                result.camera_id != camera.id,
                result.telescope_id != telescope.id,
                result.mount_id != mount.id,
                result.observatory_id != observatory.id,
                result.target_id != target.id,
            ]
        ):
            raise PipelineError(
                "Mismatch between observing session and other header "
                "information:\n\t"
                + "\n\t".join(
                    [
                        f'{what} ID: header = {getattr(result, what + "_id")} '
                        f"session = {obj.id}: {obj}"
                        for what, obj in [
                            ("observer", observer),
                            ("camera", camera),
                            ("telescope", telescope),
                            ("mount", mount),
                            ("observatory", observatory),
                            ("target", target),
                        ]
                    ]
                ),
                user_message=(
                    "The image header does not match the observing session it "
                    "was assigned to (camera/telescope/target/etc. disagree)."
                ),
            )

        result.start_time_utc = min(result.start_time_utc, exposure_start)
        result.end_time_utc = max(result.end_time_utc, exposure_end)

    return result
