"""Functions for creating light curves from DR files."""

from string import Formatter
import os.path
import os
import logging

import numpy

from autowisp.hat.file_parsers import parse_fname_keywords
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.catalog import ensure_catalog, get_catalog_config
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.light_curves.lc_data_io import LCDataIO


class DecodingStringFormatter(Formatter):
    """Add one more conversion type: ``'d'`` that calls decode on the arg."""

    def convert_field(self, value, conversion):
        """If conversion is ``'d'`` -> ``value.decode()`` else pass to parent"""

        if conversion == "d":
            return value.decode()
        return super().convert_field(value, conversion)


def get_combined_sources(dr_filenames, **dr_path_substitutions):
    """Return all sources which are in at lest one of the DR files."""

    dr_sources = set()
    for dr_fname in dr_filenames:
        with DataReductionFile(dr_fname, "r") as dr_file:
            dr_sources.update(
                dr_file.get_source_ids(
                    string_source_ids=False, **dr_path_substitutions
                )
            )
    return dr_sources


# The arguments are the union of what LCDataIO.create and the catalog query
# need; bundling them would only move the same set behind one name.
# pylint: disable=too-many-arguments
# This is simple enough
# pylint: disable=too-many-locals
def _prepare_lc_collection(
    dr_filenames,
    configuration,
    mark_end,
    *,
    dr_fname_parser,
    optional_header,
    observatory,
    **path_substitutions,
):
    """Resolve everything the writing pass needs (see collect_light_curves).

    Reads the first DR file and the single photref to settle the source
    extraction configuration, the catalog and the source list, drops any
    DR file with outlier pointing, and works out where each source's
    lightcurve goes.

    Returns:
        dict:    ``data_io``, the surviving ``dr_filenames``, the
            ``(source id, lc filename)`` pairs in ``sources_lc_fnames``,
            the ``frame_chunk`` size to process at a time, and the
            ``catalog`` ``(sources, header)`` pair the caller returns.
    """

    logger = logging.getLogger(__name__)
    srcid_formatter = DecodingStringFormatter()
    with DataReductionFile(dr_filenames[0], "r") as first_dr:
        configuration["srcextract_psf_params"] = [
            param.decode()
            for param in first_dr.get_attribute(
                "srcextract.psf_map.cfg.psf_params", **path_substitutions
            )
        ]
        if not configuration["num_magfit_iterations"]:
            path_substitutions["magfit_iteration"] = (
                first_dr.get_num_magfit_iterations(**path_substitutions) - 1
            )
        with DataReductionFile(
            configuration["single_photref_dr_fname"], "r"
        ) as sphotref_dr:
            (catalog_sources, catalog_header), outliers = ensure_catalog(
                header=sphotref_dr.get_frame_header(),
                dr_files=dr_filenames,
                configuration=get_catalog_config(configuration, "lc"),
                return_metadata=True,
                skytoframe_version=configuration["skytoframe_version"],
            )[:2]
        for outlier_ind in reversed(outliers):
            outlier_dr = dr_filenames.pop(outlier_ind)
            logger.warning(
                "Data reduction file %s has outlier pointing. Discarding!",
                outlier_dr,
            )
            mark_end(outlier_dr, -2, final=True)
        dr_sources = get_combined_sources(dr_filenames, **path_substitutions)
        catalog_source_ids = set(catalog_sources.index)
        source_list = dr_sources.intersection(catalog_source_ids)
        if len(source_list) != len(dr_sources):
            skipped = dr_sources - catalog_source_ids
            logger.debug(
                "Skipping %d DR sources not present in limited LC catalog",
                len(skipped),
            )
        data_io = LCDataIO.create(
            catalog_sources=catalog_sources,
            config=configuration,
            source_list=source_list,
            source_id_parser=first_dr.parse_hat_source_id,
            dr_fname_parser=dr_fname_parser,
            optional_header=optional_header,
            observatory=observatory,
            **path_substitutions,
        )
    frame_chunk = data_io.max_dimension_size["frame"]
    logger.debug(
        "Generating LC filenames per: %s", repr(configuration["lc_fname"])
    )
    sources_lc_fnames = [
        (
            source_id,
            srcid_formatter.format(
                configuration["lc_fname"],
                *numpy.atleast_1d(source_id),
                PROJHOME=configuration["project_home"],
            ),
        )
        for source_id in data_io.source_destinations.keys()
    ]

    for dirname in {
        os.path.abspath(os.path.dirname(lc_fname))
        for _, lc_fname in sources_lc_fnames
    }:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    return {
        "data_io": data_io,
        "dr_filenames": dr_filenames,
        "sources_lc_fnames": sources_lc_fnames,
        "frame_chunk": frame_chunk,
        "catalog": (catalog_sources, catalog_header),
    }


# pylint: enable=too-many-locals
# pylint: enable=too-many-arguments


def _write_lightcurves(prepared, mark_start, mark_end):
    """Add the DR data to the lightcurves, one chunk of frames at a time.

    Args:
        prepared(dict):    The result of :func:`_prepare_lc_collection`.

        mark_start, mark_end:    Progress callbacks, invoked per DR file.

    Returns:
        None
    """

    data_io = prepared["data_io"]
    dr_filenames = prepared["dr_filenames"]
    sources_lc_fnames = prepared["sources_lc_fnames"]
    frame_chunk = prepared["frame_chunk"]

    num_processed = 0
    while num_processed < len(dr_filenames):
        stop_processing = min(len(dr_filenames), num_processed + frame_chunk)
        data_io.prepare_for_reading()
        for dr_fname in dr_filenames[num_processed:stop_processing]:
            mark_start(dr_fname)
        # No explicit related-files scoping here: ``data_io.read`` /
        # ``data_io.write`` and the confirmation pass below all open the DR
        # or lightcurve through ``with``, and HDF5 products attach
        # themselves to any error raised while they are open.
        config_skipped = list(
            map(
                data_io.read,
                enumerate(dr_filenames[num_processed:stop_processing]),
            )
        )

        data_io.prepare_for_writing([entry[0] for entry in config_skipped])
        # data_io.print_organized_configurations()

        for write_arg in sources_lc_fnames:
            data_io.write(write_arg)

        for dr_fname in dr_filenames[num_processed:stop_processing]:
            mark_end(dr_fname, status=1, final=False)

        for _, lc_fname in sources_lc_fnames:
            if os.path.exists(lc_fname):
                with LightCurveFile(lc_fname, "a") as lc_file:
                    lc_file.confirm_lc_length()

        for dr_fname in dr_filenames[num_processed:stop_processing]:
            mark_end(dr_fname, status=1, final=True)

        num_processed = stop_processing


def collect_light_curves(
    dr_filenames,
    configuration,
    mark_start,
    mark_end,
    *,
    dr_fname_parser=parse_fname_keywords,
    optional_header=None,
    observatory=None,
    **path_substitutions,
):
    """
    Add the data from a collection of DR files to LCs, creating LCs if needed.

    Args:
        dr_filenames([str]):    The filenames of the data reduction files to add
            to LCs.

        configuration:    Object with attributes configuring the LC collection
            procedure.

        path_substitutions:    Any substitutions to resolve paths within DR and
            LC files to data to read/write (e.g. versions of various
            componenents).

        dr_fname_parser:    See same name argument to LCDataIO::create().

        optional_header:    See same name argument to LCDataIO::create().

    Returns:
        [(src ID part 1, src ID part 2, ...)];
            The sources for which new lightcurves were created.
    """

    prepared = _prepare_lc_collection(
        dr_filenames,
        configuration,
        mark_end,
        dr_fname_parser=dr_fname_parser,
        optional_header=optional_header,
        observatory=observatory,
        **path_substitutions,
    )
    _write_lightcurves(prepared, mark_start, mark_end)
    return prepared["catalog"]
