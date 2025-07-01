#!/usr/bin/env python3

"""Stack a collection of images to a master frame."""
from functools import partial
from os.path import exists
from os import remove
import logging

import numpy
from astropy.io import fits

from configargparse import Action

from autowisp.multiprocessing_util import setup_process
from autowisp.image_calibration.mask_utilities import mask_flags
from autowisp.image_calibration.master_maker import MasterMaker
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import get_primary_header

input_type = "calibrated"
_logger = logging.getLogger(__name__)

fail_reasons = {"discarded": -2}


# Interface define by argparse
# pylint: disable=too-few-public-methods
class ParseAverageAction(Action):
    """Parse the specification of the averaging function on the command line."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Set the callable to use for averaging."""

        assert len(values) == 1
        if ":" not in values[0]:
            result = getattr(numpy, "nan" + values[0])
        else:
            func, level = values[0].split(":")
            result = partial(getattr(numpy, "nan" + func), q=float(level))
        setattr(namespace, self.dest, result)


# pylint: enable=too-few-public-methods


def get_command_line_parser(
    *args,
    default_threshold=(5.0,),
    single_master=True,
    default_min_valid_values=5,
    default_max_iter=20,
):
    """Return a command line parser with all arguments added."""

    parser = ManualStepArgumentParser(
        description=__doc__, input_type="" if args else input_type
    )

    parser.add_argument(
        "--average-func",
        default="median",
        type=lambda f: getattr(numpy, "nan" + f),
        help="The function to use for calculating the average of pixel values "
        "accross images. For quantile and percentile, the desired level is "
        "specified by adding ``:<q>`` at the end of the function name(e.g."
        "``quantile:0.7``).",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        nargs="+",
        default=default_threshold,
        help="When averaging the values of a given pixel among the input "
        "images, values that are further than this value times the root mean "
        "square devitaion from the average are rejected and the average is "
        "recomputed iteratively until convergence or maximum number of "
        "iterations.",
    )
    parser.add_argument(
        "--min-valid-values",
        type=int,
        default=default_min_valid_values,
        help="If rejecting outlier pixels results in fewer than this many "
        "surviving pixels, the corresponding pixel gets a bad pixel mask in the"
        "master.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=default_max_iter,
        help="The maximum number of outlier rejection/averaging iterations to "
        "allow. If not converged before then, a warning is issued and the "
        "average if the last iteration is used.",
    )
    parser.add_argument(
        "--exclude-mask",
        choices=mask_flags.keys(),
        nargs="+",
        default=MasterMaker.default_exclude_mask,
        help="A list of mask flags, any of which result in the corresponding "
        "pixels being excluded from the averaging. Any mask flags not specified"
        "are ignored, treated as clean. Note that ``'BAD'`` means any kind of"
        " problem (e.g. saturated, hot/cold pixel, leaked etc.).",
    )
    parser.add_argument(
        "--compress",
        type=float,
        default=16,
        help="If zero, the final result is not compressed. Otherwise,"
        "this is the quantization level used for compressing the image (see "
        "`astropy.io.fits` documentation).",
    )
    parser.add_argument(
        "--add-averaged-keywords",
        nargs="+",
        default=["JD-OBS"],
        help="Specify any numeric-valued header keywords that should be "
        "generated for the master by averaging the corresponding values from "
        "the input frames using the same averaging as pixel values. By default "
        "the outlier rejected average JD of the input frames is added.",
    )
    if single_master:
        parser.add_argument(
            "--min-valid-frames",
            type=int,
            default=10,
            help="If there are fewer than this number of suitable frames to "
            "stack in a given master, that master is not generated.",
        )
        parser.add_argument(
            "--stacked-master-fname",
            default="MASTERS/{IMAGE_TYPE}_{CAMSN}_{CLRCHNL}_{OBS-SESN}.fits.fz",
            help="Filename for the master to generate if successful. Can "
            "involve header substitutions, but should produce the same filename"
            " for all input frames. If not, the behavior is undefined.",
        )

    return parser


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    return get_command_line_parser(*args).parse_args(*args)


def get_master_fname(
    image_fname, configuration, fname_key="stacked_master_fname"
):
    """Return the name of the master the given image should contribute to."""

    with fits.open(image_fname, "readonly") as first_image:
        substitutions = dict(get_primary_header(first_image))
    for arg in configuration:
        if arg.upper() not in substitutions:
            substitutions[arg.upper()] = configuration[arg]
    return configuration[fname_key].format_map(substitutions)


def stack_to_master(
    image_collection, start_status, configuration, mark_start, mark_end
):
    """Stack the given frames to produce a single master frame."""

    assert start_status is None

    for image_fname in image_collection:
        mark_start(image_fname)
    master_fname = get_master_fname(image_collection[0], configuration)
    success, discarded_frames = MasterMaker(
        **{
            arg: configuration[arg]
            for arg in [
                "outlier_threshold",
                "average_func",
                "min_valid_frames",
                "min_valid_values",
                "max_iter",
                "exclude_mask",
                "compress",
                "add_averaged_keywords",
            ]
        }
    )(image_collection, master_fname)
    for image_fname in image_collection:
        mark_end(
            image_fname,
            fail_reasons["discarded"] if image_fname in discarded_frames else 1,
        )

    if success:
        assert exists(master_fname)
        header = get_primary_header(master_fname)
        return {
            "filename": master_fname,
            "preference_order": f'JD_OBS - {header["JD-OBS"]}',
        }

    return None


def cleanup_interrupted(interrupted, configuration):
    """Cleanup file system after partially creating stacked image(s)."""

    master_fname = get_master_fname(interrupted[0][0], configuration)
    _logger.info(
        "Cleaning up partially created stack %s (%s)",
        repr(master_fname),
        repr(interrupted),
    )

    for image_fname, _ in interrupted:
        assert master_fname == get_master_fname(image_fname, configuration)

    if exists(master_fname):
        remove(master_fname)

    return -1


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)
    stack_to_master(
        list(find_fits_fnames(cmdline_config["calibrated_images"])),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
