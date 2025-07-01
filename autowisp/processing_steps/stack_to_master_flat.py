#!/usr/bin/env python3

"""Stack calibrated flat frames to a master flat."""

import logging
from os import remove
from os.path import exists

import numpy
from scipy import ndimage

from autowisp.multiprocessing_util import setup_process
from autowisp.image_calibration.mask_utilities import mask_flags
from autowisp.processing_steps.manual_util import ignore_progress
from autowisp.processing_steps.stack_to_master import (
    get_command_line_parser,
    get_master_fname as get_single_master_fname,
)
from autowisp.image_calibration import MasterMaker, MasterFlatMaker
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import get_primary_header
from autowisp.image_smoothing import (
    PolynomialImageSmoother,
    SplineImageSmoother,
    ChainSmoother,
    WrapFilterAsSmoother,
)


input_type = "calibrated"
_logger = logging.getLogger(__name__)
fail_reasons = {
    "stacking_failed_high": -2,
    "stacking_failed_low": -3,
    "medium": -4,
    "cloudy": -5,
    "colocated": -6,
}


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = get_command_line_parser(
        *args,
        default_threshold=(2.0, -3.0),
        single_master=False,
        default_min_valid_values=3,
        default_max_iter=3,
    )
    parser.add_argument(
        "--stamp-fraction",
        type=float,
        default=0.5,
        help="The fraction of the image cloude detection stamps should span.",
    )
    parser.add_argument(
        "--stamp-smoothing-num-x-terms",
        type=int,
        default=3,
        help="The number of x terms to include in the stamp smoothing.",
    )
    parser.add_argument(
        "--stamp-smoothing-num-y-terms",
        type=int,
        default=3,
        help="The number of y terms to include in the stamp smoothing.",
    )
    parser.add_argument(
        "--stamp-smoothing-outlier-threshold",
        type=float,
        default=3.0,
        help="Pixels deviating by more than this many standard deviations form "
        "the best fit smoothing function are discarded after each smoothnig "
        "fit iteration. One or two numbers should be specified. If two, one "
        "should be positive and the other negative specifying separate "
        "thresholds in the positive and negative directions.",
    )
    parser.add_argument(
        "--stamp-smoothing-max-iterations",
        type=int,
        default=1,
        help="The maximum number of fit, reject iterations to use when "
        "smoothing stamps for cloud detection.",
    )
    parser.add_argument(
        "--stamp-pixel-average",
        type=lambda f: getattr(numpy, "nan" + f),
        default="mean",
        help="The pixels of smoothed cloud detection stamps are averaged using "
        "this function and square difference from that mean is calculated. "
        "The process is iteratively reapeated discarding outlier pixels.",
    )
    parser.add_argument(
        "--stamp-pixel-outlier-threshold",
        type=float,
        default=3.0,
        help="The threshold in deviation around mean units to use for "
        "discarding stamp pixels during averaging of the smoothed stamps. One "
        "or two numbers should be specified. If two, one "
        "should be positive and the other negative specifying separate "
        "thresholds in the positive and negative directions.",
    )
    parser.add_argument(
        "--stamp-pixel-max-iter",
        type=int,
        default=3,
        help="The maximum number of averaging, discarding terations to use "
        "when calculating average and square deviation of the smoothed stamps.",
    )
    parser.add_argument(
        "--stamp-select-max-saturated-fraction",
        type=float,
        default=1e-4,
        help="Stamps with more that this fraction of saturated pixels are "
        "discarded.",
    )
    parser.add_argument(
        "--stamp-select-var-mean-fit-threshold",
        type=float,
        default=2.0,
        help="Variance vs mean fit for stamps is iteratively repeated after "
        "discarding from each iteration stamps that deviater from the fit by "
        "more than this number times the fit residual.",
    )
    parser.add_argument(
        "--stamp-select-var-mean-fit-iterations",
        type=int,
        default=2,
        help="Variance vs mean fit for stamps is iteratively repeated after "
        "discarding from each iteration stamps that deviater from the fit by "
        "more than this number times the fit residual.",
    )
    parser.add_argument(
        "--stamp-select-cloudy-night-threshold",
        type=float,
        default=5e3,
        help="If variance vs mean quadratic fit has residual of more than this,"
        " the entire observing session is discarded as cloudy.",
    )
    parser.add_argument(
        "--stamp-select-cloudy-frame-threshold",
        type=float,
        default=2.0,
        help="Individual frames with stamp mean and variance deviating from "
        "the variance vs mean fit by more than this number times the "
        "fit residual are discarded as cloudy.",
    )
    parser.add_argument(
        "--stamp-select-min-high-mean",
        type=float,
        default=6000.0,
        help="The minimum average value of the stamp pixels for a frame to be "
        "included in a high master flat. Default: %(default)s",
    )
    parser.add_argument(
        "--stamp-select-max-low-mean",
        type=float,
        default=3000.0,
        help="The maximum average value of the stamp pixels for a frame to be "
        "included in a low master flat. Default: %(default)s",
    )
    parser.add_argument(
        "--large-scale-smoothing-filter",
        type=lambda f: getattr(ndimage, f),
        default="median_filter",
        help="For each frame, the large scale struture is corrected by taking "
        "the ratio of the frame to the reference (median of all input frames), "
        "smoothing this ratio and then dividing by it. Two filters are "
        "consecutively applied for smoothing. This is the first and it should "
        "be one of the scipy.ndimage box filters.",
    )
    parser.add_argument(
        "--large-scale-smoothing-filter-size",
        type=int,
        default=12,
        help="The size of the box filter to apply.",
    )
    parser.add_argument(
        "--large-scale-smoothing-spline-num-x-nodes",
        type=int,
        default=4,
        help="The second smoothing filter applied fits a separable x and y "
        "spline. The x spline uses this many internal nodes.",
    )
    parser.add_argument(
        "--large-scale-smoothing-spline-num-y-nodes",
        type=int,
        default=4,
        help="The second smoothing filter applied fits a separable x and y "
        "spline. The x spline uses this many internal nodes.",
    )
    parser.add_argument(
        "--large-scale-smoothing-spline-outlier-threshold",
        type=float,
        default=5.0,
        help="Spline smoothing discards pixels deviating by more than this "
        "number times the fit residual and iterates.",
    )
    parser.add_argument(
        "--large-scale-smoothing-spline-max-iter",
        type=int,
        default=1,
        help="The maximum number of discard-refit iterations used during "
        "spline smoothing to get the large scale flat.",
    )
    parser.add_argument(
        "--large-scale-smoothing-bin-factor",
        type=int,
        default=4,
        help="Before smoothing is applied the image is binned by this factor.",
    )
    parser.add_argument(
        "--large-scale-smoothing-zoom-interp-order",
        type=int,
        default=4,
        help="After smoothing the image is zoomed back out to its original "
        "size using this order interpolation.",
    )
    parser.add_argument(
        "--cloud-check-smoothing-filter",
        type=lambda f: getattr(ndimage, f),
        default="median_filter",
        help="When stacking, images with matched large scale flat, any "
        "image which deviates too much from the large scale flat is "
        "discarded as cloudy. This argument specifies a smoothing filter to "
        "apply to the deviation from the large scale flat before checking "
        "if the image is an outlier.",
    )
    parser.add_argument(
        "--cloud-check-smoothing-filter-size",
        type=int,
        default=8,
        help="The size of the cloud check smoothing fliter to use.",
    )
    parser.add_argument(
        "--cloud-check-smoothing-bin-factor",
        type=int,
        default=4,
        help="Before smoothing is applied the deviation from large scale "
        "flat, it is binned by this factor.",
    )
    parser.add_argument(
        "--cloud-check-smoothing-zoom-interp-order",
        type=int,
        default=4,
        help="After smoothing the deviation from large scale flat is "
        "zoomed back out to its original size using this order interpolation.",
    )
    parser.add_argument(
        "--min-pointing-separation",
        type=float,
        default=150.0,
        help="Individual flat frames must be at least this many arcseconds "
        "apart (per their headers) to be combined in order to avoid stars "
        "showing up above the sky brightness.",
    )
    parser.add_argument(
        "--large-scale-deviation-threshold",
        type=float,
        default=0.05,
        help="If the smoothed difference between a frame and the combined "
        "large scale flat is more than this, the frame is discarded as "
        "potentially cloudy.",
    )

    parser.add_argument(
        "--min-high-combine",
        type=int,
        default=10,
        help="High master flat is generated only if at least this many frames "
        "of high illumination survive all checks.",
    )
    parser.add_argument(
        "--min-low-combine",
        type=int,
        default=5,
        help="Low master flat is generated only if at least this many frames "
        "of low illumination survive all checks.",
    )
    parser.add_argument(
        "--large-scale-stack-outlier-threshold",
        type=float,
        default=4.0,
        help="When determining the large scale flat outlier pixels of smoothed "
        "individual flats by more than this many sigma are discarded.",
    )
    parser.add_argument(
        "--large-scale-stack-average-func",
        type=lambda f: getattr(numpy, "nan" + f),
        default="median",
        help="The function used to average individual large scale flats.",
    )
    parser.add_argument(
        "--large-scale-stack-min-valid-values",
        type=int,
        default=3,
        help="The minimum number of surviving pixels after outlier rejection "
        "required to count a stacked large scale pixel as valid.",
    )
    parser.add_argument(
        "--large-scale-stack-max-iter",
        type=int,
        default=1,
        help="The maximum number of rejection-stacking iterations allowed when "
        "creating the large scale flat.",
    )
    parser.add_argument(
        "--large-scale-stack-exclude-mask",
        choices=mask_flags.keys(),
        nargs="+",
        default=MasterMaker.default_exclude_mask,
        help="A list of mask flags, any of which result in the corresponding "
        "pixels being excluded from the averaging. Any mask flags not specified"
        "are ignored, treated as clean. Note that ``'BAD'`` means any kind of"
        " problem (e.g. saturated, hot/cold pixel, leaked etc.).",
    )
    parser.add_argument(
        "--high-flat-master-fname",
        default="MASTERS/flat_{CAMSN}_{CLRCHNL}_{OBS-SESN}.fits.fz",
        help="Filename for the high illumination master flat to generate if "
        "successful. Can involve header substitutions, but should produce the "
        "same filename for all input frames. If not, the behavior is undefined.",
    )
    parser.add_argument(
        "--low-flat-master-fname",
        default="MASTERS/lowflat_{CAMSN}_{CLRCHNL}_{OBS-SESN}.fits.fz",
        help="Filename for the low illumination master flat to generate if "
        "successful. Can involve header substitutions, but should produce the "
        "same filename for all input frames. If not, the behavior is undefined.",
    )

    return parser.parse_args(*args)


def get_master_fnames(image_fname, configuration):
    """Return the filenames of the high and low master flats to generate."""

    return {
        illumination: get_single_master_fname(
            image_fname, configuration, illumination + "_flat_master_fname"
        )
        for illumination in ["high", "low"]
    }


# pylint: disable=too-many-locals
def stack_to_master_flat(
    image_collection, start_status, configuration, mark_start, mark_end
):
    """Stack the given frames to produce single high and/or low master flat."""

    def key_translate(k):
        return "size" if k == "filter_size" else k

    assert start_status is None

    split_config = {}
    for prefix in [
        "stamp_smoothing",
        "stamp_pixel",
        "stamp_select",
        "large_scale_smoothing_spline",
        "cloud_check_smoothing",
        "large_scale_stack",
    ]:
        split_config[prefix] = {
            key_translate(key[len(prefix) :].strip("_")): configuration.pop(key)
            for key in list(configuration.keys())
            if key.startswith(prefix)
        }

    split_config["stamp_pixel"]["fraction"] = configuration.pop(
        "stamp_fraction"
    )
    split_config["stamp_pixel"]["smoother"] = PolynomialImageSmoother(
        **split_config.pop("stamp_smoothing")
    )

    large_scale_smoother = ChainSmoother(
        WrapFilterAsSmoother(
            configuration.pop("large_scale_smoothing_filter"),
            size=configuration.pop("large_scale_smoothing_filter_size"),
        ),
        SplineImageSmoother(**split_config.pop("large_scale_smoothing_spline")),
        bin_factor=configuration.pop("large_scale_smoothing_bin_factor"),
        zoom_interp_order=(
            configuration.pop("large_scale_smoothing_zoom_interp_order")
        ),
    )
    cloud_check_smoother = WrapFilterAsSmoother(
        split_config["cloud_check_smoothing"].pop("filter"),
        **split_config.pop("cloud_check_smoothing"),
    )

    master_stack_config = {
        key: configuration.pop(key)
        for key in [
            "min_pointing_separation",
            "large_scale_deviation_threshold",
            "min_high_combine",
            "min_low_combine",
        ]
    }
    master_stack_config["large_scale_stack_options"] = split_config.pop(
        "large_scale_stack"
    )
    master_stack_config["large_scale_stack_options"][
        "add_averaged_keywords"
    ] = []
    master_stack_config["master_stack_options"] = {
        key: configuration.pop(key)
        for key in [
            "outlier_threshold",
            "average_func",
            "min_valid_values",
            "max_iter",
            "exclude_mask",
            "compress",
            "add_averaged_keywords",
        ]
    }

    create_master = MasterFlatMaker(
        stamp_statistics_config=split_config.pop("stamp_pixel"),
        stamp_select_config=split_config.pop("stamp_select"),
        large_scale_smoother=large_scale_smoother,
        cloud_check_smoother=cloud_check_smoother,
        master_stack_config=master_stack_config,
    )

    fnames = get_master_fnames(image_collection[0], configuration)

    for image_fname in image_collection:
        assert get_master_fnames(image_fname, configuration) == fnames
        mark_start(image_fname)

    success, classified_images = create_master(
        image_collection,
        high_master_fname=fnames["high"],
        low_master_fname=fnames["low"],
    )

    for classification, images in classified_images.items():
        if classification == "high":
            status = (
                2 if success["high"] else fail_reasons["stacking_failed_high"]
            )
        elif classification == "low":
            status = (
                1
                if success["high"] and success["low"]
                else fail_reasons["stacking_failed_low"]
            )
        else:
            status = fail_reasons[classification]
        for image_fname in images:
            mark_end(image_fname, status)

    result = {}
    for illumination in ["high", "low"]:
        if success[illumination]:
            assert exists(fnames[illumination])
            header = get_primary_header(fnames[illumination])
            result[illumination] = {
                "filename": fnames[illumination],
                "preference_order": f'JD_OBS - {header["JD-OBS"]}',
                "type": illumination + "flat",
            }
    return tuple(result.values())


# pylint: enable=too-many-locals


def cleanup_interrupted(interrupted, configuration):
    """Cleanup file system after partially creating stacked image(s)."""

    master_fnames = get_master_fnames(interrupted[0][0], configuration)
    _logger.info(
        "Cleaning up partially created stacks %s (%s)",
        repr(master_fnames),
        repr(interrupted),
    )

    for image_fname, _ in interrupted:
        test_master_fnames = get_master_fnames(image_fname, configuration)
        if master_fnames != test_master_fnames:
            raise RuntimeError(
                "Attempting to clean up frames with mismatched master "
                f"filenames! Example: {image_fname!r} -> {test_master_fnames!r}"
                f" vs {interrupted[0][0]!r} -> {master_fnames!r}"
            )
        for fname in master_fnames.values():
            if exists(fname):
                remove(fname)

    return -1


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)
    stack_to_master_flat(
        list(find_fits_fnames(cmdline_config["calibrated_images"])),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )

if __name__ == "__main__":
    main()
