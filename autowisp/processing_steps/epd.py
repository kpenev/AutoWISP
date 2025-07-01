#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from autowisp.multiprocessing_util import setup_process
from autowisp import EPDCorrection
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import (
    LCDetrendingArgumentParser,
)
from autowisp.processing_steps.lc_detrending import detrend_light_curves
from autowisp.processing_steps.manual_util import ignore_progress


def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode="EPD", description=__doc__, input_type=("" if args else "lc")
    ).parse_args(*args)


def epd(lc_collection, start_status, configuration, mark_progress):
    """Perform EPD on (a subset of the points in) the given lightucurves."""

    assert start_status == 0
    configuration["fit_datasets"] = configuration.pop("epd_datasets")

    detrend_light_curves(
        lc_collection,
        configuration,
        EPDCorrection(
            fit_identifier="EPD",
            used_variables=dict(configuration["variables"]),
            fit_points_filter_expression=(
                configuration["lc_points_filter_expression"]
            ),
            fit_terms_expression=configuration["epd_terms_expression"],
            fit_datasets=configuration["fit_datasets"],
            fit_weights=configuration["fit_weights"],
            error_avg=configuration["detrend_error_avg"],
            rej_level=configuration["detrend_rej_level"],
            max_rej_iter=configuration["detrend_max_rej_iter"],
            pre_reject=configuration["pre_reject_outliers"],
            mark_progress=mark_progress,
        ),
    )


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="manage", **cmdline_config)
    epd(
        find_lc_fnames(cmdline_config.pop("lc_files")),
        0,
        cmdline_config,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
