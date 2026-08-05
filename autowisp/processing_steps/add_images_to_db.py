#!/usr/bin/env python3

"""Register new images with the database."""

import logging

from autowisp.multiprocessing_util import setup_process
from autowisp.error_context import error_context
from autowisp.exceptions import FileKind, RelatedFile
from autowisp.evaluator import Evaluator
from autowisp.file_utilities import find_fits_fnames
from autowisp.processing_steps.manual_util import ManualStepArgumentParser
from autowisp.database.interface import start_db_session
from autowisp.database.provenance_resolver import (
    get_or_create_observing_session,
)

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import Image, ImageType

# pylint: enable=no-name-in-module

_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    if args:
        inputtype = ""
    else:
        inputtype = "raw"

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=inputtype,
        add_provenance_args=True,
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
    with start_db_session() as db_session:
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
        with error_context(
            related_files=[
                RelatedFile(FileKind.RAW_IMAGE, image_fname, role="input")
            ]
        ):
            header_eval = Evaluator(image_fname)
            header_eval.symtable["FULLPATH"] = image_fname
            _logger.debug(
                "Defining evaluator with keys: %s",
                repr(header_eval.symtable.keys()),
            )
            with start_db_session() as db_session:
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
                image.jd = header_eval.symtable.get("JD-OBS")
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
