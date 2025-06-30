#!/usr/bin/env python3

"""Perform aperture photometry on a set of frames in parallel."""

from ctypes import c_char, c_double
from functools import partial
from multiprocessing import Pool
from os import getpid
import logging

import numpy

from astrowisp import SubPixPhot, IOTree

from autowisp.fits_utilities import get_primary_header, read_image_components
from autowisp.file_utilities import find_fits_fnames
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    read_subpixmap,
    ignore_progress,
)
from autowisp.processing_steps.fit_star_shape import add_image_options

from autowisp import DataReductionFile, init_dr_process
from autowisp.data_reduction.utils import (
    fill_aperture_photometry_input_tree,
    add_aperture_photometry,
    delete_aperture_photometry,
)

input_type = "calibrated + dr"

_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("+dr" if args else input_type),
        inputs_help_extra=(
            "The corresponding DR files must alread contain a " "PSF fit."
        ),
        add_component_versions=("srcproj", "background", "shapefit", "apphot"),
        allow_parallel_processing=True,
    )
    parser.add_argument(
        "--apphot-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )

    add_image_options(parser)
    parser.add_argument(
        "--shapefit-group",
        type=int,
        default=0,
        help="If grouping was used during shape fitting, use this option to "
        "specify which of PSF map to use.",
    )
    parser.add_argument(
        "--apertures",
        nargs="+",
        type=float,
        default=[],
        help="The apretures to use for photometry.",
    )
    parser.add_argument(
        "--error-offset",
        type=float,
        default="0.0",
        help="A constant error to add to the formal error estimate from the "
        "measurement.",
    )
    return parser.parse_args(*args)


def get_photometer(configuration):
    """
    Create an instance of SubPixPhot ready to be applied.

    Args:
        configuration(dict):    The configuration specifying how to run
            photometry.

    Returns:
        SubPixPhot:
            A fully configured instance ready to carry out photometry as
            specified by the given configuration.
    """

    return SubPixPhot(
        subpixmap=read_subpixmap(configuration["subpixmap"]),
        apertures=numpy.array(configuration["apertures"]),
        gain=configuration["gain"],
        magnitude_1adu=configuration["magnitude_1adu"],
        const_error=configuration["error_offset"],
    )


def photometer_frame(frame_fname, configuration, mark_start, mark_end):
    """Perform aperture photometry on a single frame."""

    _logger.debug("Photometering %s", frame_fname)
    photometer = get_photometer(configuration)
    _logger.debug("Created photometer.")
    header = get_primary_header(frame_fname)
    header["FITGROUP"] = configuration["shapefit_group"]

    with DataReductionFile(
        configuration["data_reduction_fname"].format_map(header), "a"
    ) as dr_file:
        _logger.debug("Creating input tree for %s", frame_fname)
        io_tree = IOTree(photometer)
        _logger.debug("Filling input tree for %s", frame_fname)
        num_sources = fill_aperture_photometry_input_tree(
            dr_file,
            io_tree,
            background_version=0,
            srcproj_version=0,
            shapefit_version=0,
        )
        _logger.debug("Measuring %d sources", num_sources)
        # False positive
        # pylint: disable=unbalanced-tuple-unpacking
        pixel_values, pixel_errors, pixel_mask = read_image_components(
            frame_fname, read_header=False
        )
        # pylint: enable=unbalanced-tuple-unpacking
        photometer(
            (
                pixel_values.astype(c_double, copy=False),
                pixel_errors.astype(c_double, copy=False),
                # False positive
                # pylint: disable=no-member
                pixel_mask.astype(c_char, order="C"),
                # pylint: enable=no-member
            ),
            io_tree,
        )
        _logger.debug("Starting")
        mark_start(frame_fname)
        add_aperture_photometry(
            dr_file,
            io_tree,
            num_sources,
            len(configuration["apertures"]),
            apphot_version=configuration["apphot_version"],
        )
        mark_end(frame_fname)
        _logger.debug("Finished")


def measure_aperture_photometry(
    image_collection, start_status, configuration, mark_start, mark_end
):
    """Extract aperture photometry from the given images."""

    assert start_status is None

    photometer_one = partial(
        photometer_frame,
        configuration=configuration,
        mark_start=mark_start,
        mark_end=mark_end,
    )
    if configuration["num_parallel_processes"] == 1:
        for frame_fname in image_collection:
            photometer_one(frame_fname)
    else:
        configuration["parent_pid"] = getpid()
        _logger.debug(
            "Starting aperture photometry of %d images using %d processes",
            len(image_collection),
            min(configuration["num_parallel_processes"], len(image_collection)),
        )
        with Pool(
            processes=min(
                configuration["num_parallel_processes"], len(image_collection)
            ),
            initializer=init_dr_process,
            initargs=(configuration,),
            maxtasksperchild=1,
        ) as pool:
            pool.map(photometer_one, image_collection)


def cleanup_interrupted(interrupted, configuration):
    """Remove the aperture photometry from a frame that was interrupted."""

    for frame_fname, status in interrupted:
        assert status == 0

        header = get_primary_header(frame_fname)

        with DataReductionFile(
            configuration["data_reduction_fname"].format_map(header), "a"
        ) as dr_file:
            dr_path_substitutions = {
                version_name
                + "_version": configuration[version_name + "_version"]
                for version_name in [
                    "background",
                    "shapefit",
                    "srcproj",
                    "apphot",
                ]
            }
            delete_aperture_photometry(
                dr_file,
                len(configuration["apertures"]),
                **dr_path_substitutions,
            )

    return -1


def has_psf_model(image_fname, shapefit_version):
    """Check if the DR file contains a sky-to-frame transformation."""

    with DataReductionFile(
        header=get_primary_header(image_fname), mode="r"
    ) as dr_file:
        try:
            dr_file.check_for_dataset(
                "shapefit.map_coef",
                shapefit_version=shapefit_version,
            )
            return True
        except IOError:
            return False


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    DataReductionFile.fname_template = cmdline_config["data_reduction_fname"]
    cmdline_config["task"] = "manage"
    init_dr_process(cmdline_config)
    del cmdline_config["task"]
    measure_aperture_photometry(
        [
            image_fname
            for image_fname in find_fits_fnames(
                cmdline_config.pop("calibrated_images"),
                cmdline_config.pop("apphot_only_if"),
            )
            if has_psf_model(image_fname, cmdline_config["shapefit_version"])
        ],
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
