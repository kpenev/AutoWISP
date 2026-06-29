#!/usr/bin/env python3

"""Detect stars within calibrated image(s)."""

from contextlib import ExitStack
from functools import partial
from os import path
import logging

from autowisp.multiprocessing_util import setup_process
from autowisp.error_context import run_pool
from autowisp.error_cli import cli_entry_point
from autowisp.exceptions import Component
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)
from autowisp.evaluator import Evaluator
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import get_primary_header
from autowisp.source_finder import SourceFinder
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.interface import start_db_session
from autowisp.database.provenance_resolver import (
    get_or_create_observing_session,
)

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import ObservingSession

# pylint: enable=no-name-in-module

input_type = "calibrated + dr"
_logger = logging.getLogger(__name__)


def _regional_source_balance(extracted_sources):
    """Return how evenly extracted sources occupy the image halves."""

    if len(extracted_sources) == 0:
        return 0.0

    def half_balance(coordinates):
        """Return smaller/larger count for a split through coordinate range."""

        split_coord = (coordinates.min() + coordinates.max()) / 2.0
        low_count = (coordinates < split_coord).sum()
        high_count = (coordinates >= split_coord).sum()
        return (
            0.0
            if high_count == 0
            else min(low_count, high_count) / max(low_count, high_count)
        )

    source_x = extracted_sources["x"]
    source_y = extracted_sources["y"]
    return float(min(half_balance(source_x), half_balance(source_y)))


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("+dr" if args else input_type),
        allow_parallel_processing=True,
        add_component_versions=("srcextract",),
        add_provenance_args=True,
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
        default=None,
        help="The minimum brightness to require of extracted sources. If not "
        "specified, it will be automatically calculated from the image using "
        "brightness-quantile and brightness-quantile-scale parameters. It "
        "should be tuned to a value that picks out as many stars as possible, "
        "without resulting in an appreciable number of spurious detections. "
        "Two additional parameters (:option:`filter-sources` and "
        ":option:`srcextract-max-sources`) are sometimes useful to eliminate "
        "false positives.",
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
    parser.add_argument(
        "--brightness-quantile",
        default=0.999,
        type=float,
        help="The quantile to use for the brightness threshold.",
    )
    parser.add_argument(
        "--brightness-quantile-scale",
        default=1.0,
        type=float,
        help="The scale factor to use for the brightness quantile.",
    )
    return parser.parse_args(*args)


def _resolve_observing_session_ids(image_collection, configuration):
    """Pre-resolve provenance once in the parent (single-threaded).

    Returns a ``{image_fname: observing_session_id}`` map. Empty if
    ``--no-provenance`` is set. Running this in the parent avoids the
    target/session-name UNIQUE races that would otherwise hit when several
    worker processes call ``get_or_create_*`` against the same survey row.
    """

    if configuration.get("no_provenance"):
        return {}

    result = {}
    with start_db_session() as db_session:
        for image_fname in image_collection:
            header_eval = Evaluator(image_fname)
            header_eval.symtable["FULLPATH"] = image_fname
            observing_session = get_or_create_observing_session(
                "object", header_eval, configuration, db_session
            )
            db_session.flush()
            result[image_fname] = observing_session.id
    return result


def _find_stars_worker(  # pylint: disable=too-many-arguments
    image_fname,
    *,
    find_stars_in_image,
    srcextract_version,
    mark_start,
    mark_end,
    observing_session_ids,
):
    """Pool worker: dispatch to ``find_stars_single`` with the matching id."""

    find_stars_single(
        image_fname,
        find_stars_in_image,
        srcextract_version,
        mark_start,
        mark_end,
        observing_session_id=observing_session_ids.get(image_fname),
    )


def find_stars_single(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    image_fname,
    find_stars_in_image,
    srcextract_version,
    mark_start,
    mark_end,
    observing_session_id=None,
):
    """Find the stars in a single image.

    If ``observing_session_id`` is given (i.e. provenance was pre-resolved by
    the caller), the worker looks up the ``ObservingSession`` row by id and
    writes the ``/Provenance`` group to the DR file. Pre-resolving in the
    caller -- rather than per worker -- is what prevents races when multiple
    workers see the same target/session.
    """

    fits_header = get_primary_header(image_fname)
    _logger.debug("Extracting sources from %r", image_fname)
    extracted_sources = find_stars_in_image(image_fname)
    _logger.debug("Finished extracting sources: %r", extracted_sources)
    mark_start(image_fname)
    _logger.debug("Marked started: %r", extracted_sources)

    with ExitStack() as stack:
        observing_session = None
        if observing_session_id is not None:
            db_session = stack.enter_context(start_db_session())
            observing_session = db_session.get(
                ObservingSession, observing_session_id
            )

        dr_file = stack.enter_context(
            DataReductionFile(header=fits_header, mode="a")
        )
        dr_file.initialize(fits_header, observing_session=observing_session)
        _logger.debug("Added header from: %r", extracted_sources)
        dr_file.add_sources(
            extracted_sources,
            "srcextract.sources",
            "srcextract_column_name",
            srcextract_version=srcextract_version,
        )
        _logger.debug("Added sources from: %r", extracted_sources)
        mark_end(
            image_fname,
            diagnostics=[
                ("num_extracted_src", len(extracted_sources)),
                (
                    "src_count_min_half_fraction",
                    _regional_source_balance(extracted_sources),
                ),
            ],
        )
        _logger.debug("Marked end for: %r", extracted_sources)


def find_stars(
    image_collection, start_status, configuration, mark_start, mark_end
):
    """Extract sources from all input images and save them to DR files."""

    _logger.debug(
        "Start of find_stars steps for DB %s for %d images with configuration "
        "%s",
        configuration["project_home"],
        len(image_collection),
        repr(configuration),
    )
    assert start_status is None

    DataReductionFile.fname_template = configuration["data_reduction_fname"]
    find_stars_in_image = SourceFinder(
        tool=configuration["srcfind_tool"],
        brightness_threshold=configuration["brightness_threshold"],
        brightness_quantile=configuration["brightness_quantile"],
        brightness_quantile_scale=configuration["brightness_quantile_scale"],
        filter_sources=configuration["filter_sources"],
        max_sources=configuration["srcextract_max_sources"],
    )
    _logger.debug("Created source finder")
    observing_session_ids = _resolve_observing_session_ids(
        image_collection, configuration
    )
    if configuration["num_parallel_processes"] == 1:
        _logger.debug(
            "Running in serial mode for images: %s", repr(image_collection)
        )
        for image_fname in image_collection:
            _logger.debug("Extracting stars in image %s", image_fname)
            find_stars_single(
                image_fname,
                find_stars_in_image,
                configuration["srcextract_version"],
                mark_start,
                mark_end,
                observing_session_id=observing_session_ids.get(image_fname),
            )
            _logger.debug("Finished extracting stars in image %s", image_fname)

    else:
        _logger.debug(
            "Running in parallel mode with config %s and DB fname %s",
            configuration,
            configuration["project_home"],
        )

        run_pool(
            partial(
                _find_stars_worker,
                find_stars_in_image=find_stars_in_image,
                srcextract_version=configuration["srcextract_version"],
                mark_start=mark_start,
                mark_end=mark_end,
                observing_session_ids=observing_session_ids,
            ),
            image_collection,
            config=configuration,
            num_processes=configuration["num_parallel_processes"],
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


@cli_entry_point(component=Component.STEP)
def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)

    find_stars(
        list(
            find_fits_fnames(
                cmdline_config["calibrated_images"],
                cmdline_config["srcextract_only_if"],
            )
        ),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
