#!/usr/bin/env python3

"""Detect stars within calibrated image(s)."""

from functools import partial
from multiprocessing import Pool
from os import path, getpid

from autowisp.multiprocessing_util import setup_process
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import get_primary_header
from autowisp import SourceFinder, DataReductionFile, init_dr_process

input_type = "calibrated + dr"


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("+dr" if args else input_type),
        allow_parallel_processing=True,
        add_component_versions=("srcextract",),
    )
    parser.add_argument(
        "--srcextract-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )
    parser.add_argument(
        "--srcfind-tool",
        choices=["fistar", "hatphot"],
        default="fistar",
        help="The source extractor to use.",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=1000,
        help="The minimum brightness to require of extracted sources.",
    )
    parser.add_argument(
        "--filter-sources",
        default="True",
        help="A condition involving the output columns from source extraction "
        "to impose on the list of extracted sources (sources that fail are "
        "discarded).",
    )
    parser.add_argument(
        "--srcextract-max-sources",
        type=int,
        default=4000,
        help="If more than this many sources are extracted, the list is sorted "
        "by flux and truncated to this number.",
    )
    return parser.parse_args(*args)


def find_stars_single(
    image_fname, find_stars_in_image, srcextract_version, mark_start, mark_end
):
    """Find the stars in a single image."""

    fits_header = get_primary_header(image_fname)
    with DataReductionFile(header=fits_header, mode="a") as dr_file:
        dr_file.add_frame_header(fits_header)
        extracted_sources = find_stars_in_image(image_fname)
        print(f"Extracted sources: {extracted_sources!r}")
        mark_start(image_fname)
        dr_file.add_sources(
            extracted_sources,
            "srcextract.sources",
            "srcextract_column_name",
            srcextract_version=srcextract_version,
        )
        mark_end(image_fname)


def find_stars(
    image_collection, start_status, configuration, mark_start, mark_end
):
    """Extract sources from all input images and save them to DR files."""

    assert start_status is None

    DataReductionFile.fname_template = configuration["data_reduction_fname"]
    find_stars_in_image = SourceFinder(
        tool=configuration["srcfind_tool"],
        brightness_threshold=configuration["brightness_threshold"],
        filter_sources=configuration["filter_sources"],
        max_sources=configuration["srcextract_max_sources"],
    )
    if configuration["num_parallel_processes"] == 1:
        for image_fname in image_collection:
            find_stars_single(
                image_fname,
                find_stars_in_image,
                configuration["srcextract_version"],
                mark_start,
                mark_end,
            )
    else:
        configuration["parent_pid"] = getpid()
        with Pool(
            configuration["num_parallel_processes"],
            initializer=init_dr_process,
            initargs=(configuration,),
        ) as pool:
            pool.map(
                partial(
                    find_stars_single,
                    find_stars_in_image=find_stars_in_image,
                    srcextract_version=configuration["srcextract_version"],
                    mark_start=mark_start,
                    mark_end=mark_end,
                ),
                image_collection,
            )


def cleanup_interrupted(interrupted, configuration):
    """Remove the extracted stars from the DR of the given calibrated image."""

    DataReductionFile.fname_template = configuration["data_reduction_fname"]
    for image_fname, status in interrupted:
        assert status == 0

        fits_header = get_primary_header(image_fname)
        dr_fname = DataReductionFile.get_fname_from_header(fits_header)
        if not path.exists(dr_fname):
            return -1

        with DataReductionFile(dr_fname, mode="r+") as dr_file:
            dr_file.delete_sources(
                "srcextract.sources",
                "srcextract_column_name",
                srcextract_version=configuration["srcextract_version"],
            )
    return -1


def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="manage", **cmdline_config)
    find_stars(
        find_fits_fnames(
            cmdline_config["calibrated_images"],
            cmdline_config["srcextract_only_if"],
        ),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )

if __name__ == "__main__":
    main()
