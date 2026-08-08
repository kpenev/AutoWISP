#!/usr/bin/env python3

"""Apply TFA correction to lightcurves."""

from autowisp.multiprocessing_util import setup_process
from autowisp.error_cli import cli_entry_point
from autowisp.exceptions import Component
from autowisp.light_curves.tfa_correction import TFACorrection
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import (
    LCDetrendingArgumentParser,
)
from autowisp.processing_steps.lc_detrending import (
    detrend_light_curves,
    resolve_fit_datasets,
)
from autowisp.light_curves.apply_correction import load_correction_statistics
from autowisp.processing_steps.manual_util import ignore_progress


def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode="TFA", description=__doc__, input_type=("" if args else "lc")
    ).parse_args(*args)


#: Lightcurve steps do not resume: the manager hands them every LC to
#: correct from the start.
allowed_start_status_values = (0,)


def tfa(lc_collection, start_status, configuration, mark_progress):
    """Perform TFA on (a subset of the points in) the given lightucurves."""
    # ``start_status`` is part of the signature the manager calls
    # with; the values this step accepts are declared in
    # ``allowed_start_status_values`` and checked there.
    # pylint: disable=unused-argument

    lc_collection = list(lc_collection)
    configuration["fit_datasets"] = resolve_fit_datasets(
        configuration, "tfa", lc_collection
    )
    for param in list(configuration.keys()):
        if param.startswith("tfa_"):
            configuration[param[4:]] = configuration.pop(param)

    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r"
    ) as sphotref_dr:
        sphotref_header = sphotref_dr.get_frame_header()

    configuration["fit_points_filter_expression"] = configuration.pop(
        "lc_points_filter_expression"
    )

    epd_statistics = load_correction_statistics(
        configuration["epd_statistics_fname"].format_map(sphotref_header)
    )

    if configuration["target_id"] is not None:
        epd_statistics = epd_statistics[
            epd_statistics["ID"] != int(configuration["target_id"])
        ]

    detrend_light_curves(
        lc_collection,
        configuration,
        TFACorrection(
            epd_statistics,
            configuration,
            error_avg=configuration["detrend_error_avg"],
            rej_level=configuration["detrend_rej_level"],
            max_rej_iter=configuration["detrend_max_rej_iter"],
            reject_scale_floor=configuration["detrend_reject_scale_floor"],
            fit_identifier="TFA",
            verify_template_data=True,
            mark_progress=mark_progress,
        ),
    )


@cli_entry_point(component=Component.STEP)
def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)

    tfa(
        find_lc_fnames(cmdline_config.pop("lc_files")),
        0,
        cmdline_config,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
