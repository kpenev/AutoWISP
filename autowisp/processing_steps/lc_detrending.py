"""Functions for detrending light curves (EPD or TFA)."""

from os import path, makedirs, getpid
import logging

import numpy
from pytransit import QuadraticModel

from autowisp import DataReductionFile
from autowisp import LightCurveFile
from autowisp.catalog import read_catalog_file
from autowisp.magnitude_fitting.util import format_master_catalog
from autowisp.light_curves.apply_correction import (
    apply_parallel_correction,
    apply_reconstructive_correction_transit,
    save_correction_statistics,
    recalculate_correction_statistics,
)
from autowisp import Evaluator

_logger = logging.getLogger(__name__)


def extract_target_lc(lc_fnames, target_id):
    """Return target LC fname, & LC fname list with the target LC removed."""

    for index, fname in enumerate(lc_fnames):
        with LightCurveFile(fname, "r") as lightcurve:
            if target_id.encode("ascii") in lightcurve["Identifiers"][:, 1]:
                return lc_fnames.pop(index), lc_fnames
    raise ValueError("None of the lightcurves seems to be for the target.")


def _add_catalog_info(
    lc_fnames, catalog_sources, magnitude_expression, result=None
):
    """Fill the catalog information fields in result."""

    with DataReductionFile() as mem_dr:
        catalog = format_master_catalog(
            catalog_sources, mem_dr.parse_hat_source_id
        )

    for lc_ind, fname in enumerate(lc_fnames):
        with LightCurveFile(fname, "r") as lightcurve:
            cat_source_id = None
            for source_id in lightcurve["Identifiers"][:, 1]:
                if source_id in catalog:
                    cat_source_id = source_id
                elif source_id.decode("ascii") in catalog:
                    cat_source_id = source_id.decode("ascii")
                else:
                    try:
                        if int(source_id) in catalog:
                            cat_source_id = int(source_id)
                    except ValueError:
                        pass
            assert cat_source_id is not None

            cat_info = catalog[cat_source_id]
            if result is None:
                result = numpy.empty(
                    len(lc_fnames),
                    dtype=[
                        ("ID", numpy.dtype(type(cat_source_id))),
                        ("mag", float),
                        ("xi", float),
                        ("eta", float),
                    ],
                )
            result[lc_ind]["ID"] = cat_source_id
            result[lc_ind]["mag"] = Evaluator(cat_info)(magnitude_expression)
            result[lc_ind]["xi"] = cat_info["xi"]
            result[lc_ind]["eta"] = cat_info["eta"]
    return result


def get_transit_parameters(configuration, unwind_limb_darkening=True):
    """Return the parameters to pass to pytransit model."""

    transit_parameters = (
        [configuration["radius_ratio"]]
        + (
            list(configuration["limb_darkening"])
            if unwind_limb_darkening
            else [configuration["limb_darkening"]]
        )
        + [
            configuration["mid_transit"],
            configuration["period"],
            configuration["scaled_semimajor"],
            configuration["inclination"] * numpy.pi / 180.0,
        ]
    )
    if hasattr(configuration, "eccentricity"):
        transit_parameters.append(configuration["eccentricity"])
    if hasattr(configuration, "periastron"):
        transit_parameters.append(configuration["periastron"])
    return transit_parameters


def correct_target_lc(target_lc_fname, configuration, correct):
    """Perform reconstructive detrending on the target LC."""

    num_limbdark_coef = len(configuration["limb_darkening"])
    assert num_limbdark_coef == 2

    transit_parameters = get_transit_parameters(configuration)
    fit_parameter_flags = numpy.zeros(len(transit_parameters), dtype=bool)

    param_indices = {
        "depth": 0,
        "limbdark": list(range(1, num_limbdark_coef + 1)),
        "mid_transit": num_limbdark_coef + 1,
        "period": num_limbdark_coef + 2,
        "semimajor": num_limbdark_coef + 3,
        "inclination": num_limbdark_coef + 4,
        "eccentricity": num_limbdark_coef + 5,
        "periastron": num_limbdark_coef + 6,
    }
    for to_fit in configuration["mutable_transit_params"]:
        fit_parameter_flags[param_indices[to_fit]] = True

    return apply_reconstructive_correction_transit(
        target_lc_fname,
        correct,
        transit_model=QuadraticModel(),
        transit_parameters=numpy.array(transit_parameters),
        fit_parameter_flags=fit_parameter_flags,
        num_limbdark_coef=num_limbdark_coef,
    )


def calculate_detrending_performance(
    lc_fnames, start_status, configuration, mark_progress, detrending_mode
):
    """
    Create a statistics file after de-trending directly from LCs.

    Args:
        lc_fnames:    Iterable over the filenames of the de-trended lightcurves
            to rederive the statistics for.

        catalog_fname:     The filename of the catalog to add information to
            the statistics.

        magnitude_column:     The column from the catalog to use as brightness
            indicator in the statistics file.

        output_statistics_fname:    The filename to save the statistics under.

        recalc_arguments:    Passed directly to
            recalculate_correction_statistics()
    """

    lc_fnames = list(lc_fnames)
    assert start_status == 0

    _logger.debug(
        "Generating %s performance statistics for %d light_curves",
        detrending_mode,
        len(lc_fnames),
    )

    catalog_sources = read_catalog_file(
        configuration["detrending_catalog"], add_gnomonic_projection=True
    )

    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r"
    ) as sphotref_dr:
        sphotref_header = sphotref_dr.get_frame_header()

    output_statistics_fname = configuration[
        f"{detrending_mode}_statistics_fname"
    ].format_map(sphotref_header)

    statistics = recalculate_correction_statistics(
        lc_fnames,
        fit_datasets=configuration[f"{detrending_mode}_datasets"],
        variables=configuration["variables"],
        lc_points_filter_expression=configuration[
            "lc_points_filter_expression"
        ],
        calculate_average=getattr(
            numpy, configuration["detrend_reference_avg"]
        ),
        calculate_scatter=getattr(numpy, configuration["detrend_error_avg"]),
        outlier_threshold=configuration["detrend_rej_level"],
        max_outlier_rejections=configuration["detrend_max_rej_iter"],
    )
    _add_catalog_info(
        lc_fnames,
        catalog_sources,
        configuration.pop("magnitude_column"),
        statistics,
    )

    if not path.exists(path.dirname(output_statistics_fname)):
        makedirs(path.dirname(output_statistics_fname))
    save_correction_statistics(statistics, output_statistics_fname)
    mark_progress(lc_fnames)
    return {"filename": output_statistics_fname, "preference_order": None}


def detrend_light_curves(lc_collection, configuration, correct):
    """Detrend all lightcurves and create statistics file."""

    lc_collection = list(lc_collection)
    _logger.debug("Detrending %d light_curves", len(lc_collection))

    if configuration["target_id"] is not None:
        target_lc_fname, lc_fnames = extract_target_lc(
            lc_collection, configuration["target_id"]
        )

        _, target_result = correct_target_lc(
            target_lc_fname, configuration, correct
        )
    else:
        lc_fnames = lc_collection

    if lc_fnames:
        configuration["task"] = (
            correct.iterative_fit_config["fit_identifier"] + "_fit"
        )
        configuration["parent_pid"] = getpid()
        result = apply_parallel_correction(lc_fnames, correct, **configuration)
        if configuration["target_id"] is not None:
            result = numpy.concatenate((result, target_result))
    else:
        result = target_result

    if configuration["target_id"] is not None:
        lc_fnames.append(target_lc_fname)
