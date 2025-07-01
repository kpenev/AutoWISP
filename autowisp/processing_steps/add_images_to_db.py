#!/usr/bin/env python3

"""Register new images with the database."""

from datetime import timedelta
import logging

from astropy import units
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

from autowisp.multiprocessing_util import setup_process
from autowisp import Evaluator
from autowisp.file_utilities import find_fits_fnames
from autowisp.processing_steps.manual_util import ManualStepArgumentParser
from autowisp.database.interface import Session

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model.provenance import (
    Observer,
    Camera,
    Telescope,
    Mount,
    Observatory,
)
from autowisp.database.data_model import (
    Image,
    ImageType,
    ObservingSession,
    Target,
)

# pylint: enable=no-name-in-module

_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    if args:
        inputtype = ""
    else:
        inputtype = "raw"

    parser = ManualStepArgumentParser(
        description=__doc__, input_type=inputtype, add_exposure_timing=True
    )
    parser.add_argument(
        "--observer",
        default="ORIGIN",
        help="The name of the observer who/which collected the images. Can "
        "be arbitrary expression involving header keywords. Must already have "
        "an entry in the ``observer`` table.",
    )
    parser.add_argument(
        "--camera-serial-number",
        "--cam-sn",
        default="CAMSN",
        help="The serial number of the camera which collected the images. Can "
        "be arbitrary expression involving header keywords. Must already have "
        "an entry in the ``camera`` table.",
    )
    parser.add_argument(
        "--telescope-serial-number",
        "--tel-sn",
        default="INTSN",
        help="The serial number of the telescope (lens) which collected the "
        "images (or some other unique and persistent identifier of it). Can "
        "be arbitrary expression involving header keywords. Must already have "
        "an entry in the ``telescope`` table.",
    )
    parser.add_argument(
        "--mount-serial-number",
        "--mount-sn",
        default="OBSERVER",
        help="The serial number of the mount which collected the "
        "images (or some other unique and persistent identifier of it). Can "
        "be arbitrary expression involving header keywords. "
        "Must already have an entry in the ``mount`` table.",
    )
    parser.add_argument(
        "--observatory",
        default=None,
        help="The name of the observatory from where the images were collected."
        " Can be arbitrary expression involving header keywords. Must already "
        "have an entry in the ``observatory`` table. If not specified, "
        "--observatory-location is used to determine the observatory.",
    )
    parser.add_argument(
        "--observatory-location",
        metavar=("LATITUDE", "LONGITUDE", "ALTITUDE"),
        default=["LAT_OBS", "LONG_OBS", "ALT_OBS"],
        nargs=3,
        help="The latitude and longitude of the observatory from where the "
        "images were collected. Can be arbitrary expression involving header "
        "keywords. Must already have an entry in the ``observatory`` table "
        "within approximately 100km. Only used if --observatory is not "
        "specified.",
    )
    parser.add_argument(
        "--target-ra",
        "--ra",
        default="RA_MNT",
        help="The RA targetted by the observations in degrees. Can be arbitrary"
        " expression involving header keywords. If target table already "
        "contains an entry for this target, the RA must match within 1%% of the"
        " field of view or an error is raised. It can be left unspecified if "
        "the target is already in the target table, in which case it will be "
        "identified by name.",
    )
    parser.add_argument(
        "--target-dec",
        "--dec",
        default="DEC_MNT",
        help="The Dec targetted by the observations. See --target-ra for "
        "details.",
    )
    parser.add_argument(
        "--target-name",
        "--target",
        default="FIELD",
        help="The name of the targetted area of the sky. Can be arbitrary "
        "expression involving header keywords. If not already in the target "
        "table it is automatically added.",
    )
    parser.add_argument(
        "--observing-session-label",
        "--session-label",
        "--session",
        default="SEQID",
        help="Unique label for the observing session. Can be arbitrary "
        "expression involving header keywords. If not already in the "
        "observing_session table it is automatically added. It will also be "
        "added as ``OBS-SESN`` keyword to the calibrated images.",
    )
    parser.add_argument(
        "--image-type",
        default=None,
        help="Header expression that evaluates to the image type. If it is not "
        "one of the image types listed in the database, the image is ignored. "
        "If not specified, the individual checks below are used instead.",
    )
    parser.add_argument(
        "--ignore-unknown-image-types",
        action="store_true",
        default=False,
        help="If this option is passed and an image of an unknown type is "
        "encountered it will not be added tot he database.",
    )
    # False positivie
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        for image_type in [
            record[0] for record in db_session.query(ImageType.name).all()
        ]:
            parser.add_argument(
                f"--{image_type}-check",
                default=str(image_type == "object"),
                help="Header expression that evaluates to True if the image is "
                f"a {image_type} frame.",
            )

    return parser.parse_args(*args)


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
            f"Checking target {target_name} for {image_type!r} image. "
            f"From DB: {db_target!r} vs image: {image_target!r}"
        )
        assert (
            image_target.separation(
                SkyCoord(
                    ra=db_target.ra * units.deg, dec=db_target.dec * units.deg
                )
            )
            < 0.01 * field_of_view
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
    else:
        if (
            result.observer_id != observer.id
            or result.camera_id != camera.id
            or result.telescope_id != telescope.id
            or result.mount_id != mount.id
            or result.observatory_id != observatory.id
            or result.target_id != target.id
        ):
            raise RuntimeError(
                "Mismatch between observing session and other header "
                "information:\n\t"
                + "\n\t".join(
                    [
                        f'{what} ID: header = {getattr(result, what + "_id")} '
                        f"session = {locals()[what]}"
                        for what in [
                            "observer",
                            "camera",
                            "telescope",
                            "mount",
                            "observatory",
                            "target",
                        ]
                    ]
                )
            )

        if exposure_start < result.start_time_utc:
            result.start_time_utc = exposure_start
        if exposure_end > result.end_time_utc:
            result.end_time_utc = exposure_end

    return result


def create_image(image_fname, header_eval, configuration, db_session):
    """Create the database Image entry corresponding to the given file."""

    recognized_image_types = [
        record[0] for record in db_session.query(ImageType.name).all()
    ]
    if configuration["image_type"]:
        image_type = header_eval(configuration["image_type"]).lower()
        if image_type not in recognized_image_types:
            if configuration["ignore_unknown_image_types"]:
                return None, None
            raise ValueError(
                f"Unrecognized image type {image_type!r} "
                f"(expected one of {recognized_image_types})"
            )
    else:
        image_type = None
        for test_image_type in recognized_image_types:
            if header_eval(configuration[f"{test_image_type}_check"]):
                assert image_type is None
                image_type = test_image_type
    image_type_id = (
        db_session.query(ImageType.id).filter_by(name=image_type).one()[0]
    )

    # False positive
    # pylint: disable=not-callable
    return Image(raw_fname=image_fname, image_type_id=image_type_id), image_type
    # pylint: enable=not-callable


def add_images_to_db(image_collection, configuration):
    """Add all the images in the collection to the database."""

    for image_fname in image_collection:
        logging.debug("Adding image %s to database", image_fname)
        header_eval = Evaluator(image_fname)
        header_eval.symtable["FULLPATH"] = image_fname
        _logger.debug(
            "Defining evaluator with keys: %s",
            repr(header_eval.symtable.keys()),
        )
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            image, image_type = create_image(
                image_fname, header_eval, configuration, db_session
            )
            if image is None:
                continue
            existing_image = (
                db_session.query(Image)
                .filter_by(raw_fname=image.raw_fname)
                .one_or_none()
            )
            image.observing_session = get_or_create_observing_session(
                image_type, header_eval, configuration, db_session
            )
            if existing_image is None:
                db_session.add(image)
            else:
                logging.info(
                    "Image %s already in the database with ID: %s",
                    image.raw_fname,
                    existing_image.id,
                )
                assert existing_image.image_type_id == image.image_type_id
                assert (
                    existing_image.observing_session_id
                    == image.observing_session.id
                )


if __name__ == "__main__":
    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)
    add_images_to_db(
        find_fits_fnames(cmdline_config.pop("raw_images")), cmdline_config
    )
