#!/usr/bin/env python3

"""Apply TFA correction to lightcurves."""

from autowisp.multiprocessing_util import setup_process
from autowisp.light_curves.tfa_correction import TFACorrection
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import (
    LCDetrendingArgumentParser,
)
from autowisp.processing_steps.lc_detrending import detrend_light_curves
from autowisp.light_curves.apply_correction import load_correction_statistics
from autowisp.processing_steps.manual_util import ignore_progress
from autowisp.light_curves.light_curve_file import LightCurveFile


def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode="TFA", description=__doc__, input_type=("" if args else "lc")
    ).parse_args(*args)


def tfa(lc_collection, start_status, configuration, mark_progress):
    """Perform TFA on (a subset of the points in) the given lightucurves."""

    assert start_status == 0

    configuration["fit_datasets"] = configuration.pop("tfa_datasets")
    for param in list(configuration.keys()):
        if param.startswith("tfa_"):
            print(f"Renaming {param!r} -> {param[4:]!r}")
            configuration[param[4:]] = configuration.pop(param)
        else:
            print("Not renaming " + repr(param))

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

    # Filter light curve files based on the presence of the EPD dataset
    lc_collection = filter_lc_files(lc_collection, configuration.get("ignore_missing_epd_lcs", False))

    detrend_light_curves(
        lc_collection,
        configuration,
        TFACorrection(
            epd_statistics,
            configuration,
            error_avg=configuration["detrend_error_avg"],
            rej_level=configuration["detrend_rej_level"],
            max_rej_iter=configuration["detrend_max_rej_iter"],
            fit_identifier="TFA",
            verify_template_data=True,
            mark_progress=mark_progress,
        ),
    )


def filter_lc_files(lc_fnames, ignore_missing):
    """Filter light curve files to exclude those missing the EPD dataset."""
    if not ignore_missing:
        return lc_fnames

    valid_lc_fnames = []
    for fname in lc_fnames:
        try:
            dataset_key = "apphot.epd.magnitude"
            with LightCurveFile(fname, "r") as lc:
                lc.get_dataset(dataset_key, aperture_index=0)
            valid_lc_fnames.append(fname)
        except Exception as e:
            print(f"Ignoring {fname}: {e}")
    return valid_lc_fnames


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(
        task="main", **cmdline_config
    )

    tfa(
        find_lc_fnames(cmdline_config.pop("lc_files")),
        0,
        cmdline_config,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
