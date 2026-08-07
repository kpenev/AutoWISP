"""Functions for detrending light curves (EPD or TFA)."""

from os import path, makedirs, getpid
import logging

import numpy
from pytransit import QuadraticModel

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.error_context import error_context
from autowisp.exceptions import ConfigurationError, FileKind, RelatedFile
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.catalog import read_catalog_file
from autowisp.magnitude_fitting.util import format_master_catalog
from autowisp.light_curves.apply_correction import (
    apply_parallel_correction,
    apply_reconstructive_correction_transit,
    save_correction_statistics,
    recalculate_correction_statistics,
)
from autowisp.evaluator import Evaluator
from autowisp.processing_steps.lc_detrending_argument_parser import (
    parse_fit_datasets,
)

_logger = logging.getLogger(__name__)


def _shape_fit_varies(sphotref_dr, shapefit_version):
    """
    Whether star shape was fit rather than assumed constant accross each star.

    Shape fitting is performed on a grid (see the ``shape-grid`` option of
    ``wisp-fit-star-shape``). If only the outer boundaries of that grid are
    specified, the PSF/PRF is assumed not to vary accross the star, so the
    shape fitted magnitudes carry no information beyond aperture photometry and
    detrending them is a waste of time.

    Args:
        sphotref_dr(DataReductionFile):    The single photometric reference,
            opened for reading.

        shapefit_version(int):    The version of the shape fit to inspect.

    Returns:
        bool:
            Whether the grid the shape was fit on has any internal splits.
    """

    return any(
        len(
            sphotref_dr.get_attribute(
                f"shapefit.cfg.psf.bicubic.grid.{axis}",
                default_value=(),
                shapefit_version=shapefit_version,
            )
        )
        > 2
        for axis in ["x", "y"]
    )


def _get_default_fit_datasets(
    sphotref_dr, detrending_mode, apphot_version=0, shapefit_version=0
):
    """
    Return the datasets to detrend if the user did not specify any.

    Args:
        sphotref_dr(DataReductionFile):    The single photometric reference,
            opened for reading.

        detrending_mode(str):    Either ``'epd'`` or ``'tfa'``.

        apphot_version(int):    The version of the aperture photometry whose
            apertures to detrend.

        shapefit_version(int):    See _shape_fit_varies().

    Returns:
        [(str, dict, str)]:
            The same format the ``--<mode>-datasets`` argument parses to.
    """

    # EPD corrects the magnitude fitted magnitudes, TFA the EPD corrected ones.
    source_mode = "magfit" if detrending_mode == "epd" else "epd"

    num_apertures = sphotref_dr.get_num_apertures(
        apphot_version=apphot_version
    )
    if not num_apertures:
        raise ConfigurationError(
            f"The single photometric reference {sphotref_dr.filename!r} "
            "contains no aperture photometry, so the datasets to detrend "
            f"cannot be determined. Set the {detrending_mode}-datasets "
            "configuration explicitly."
        )

    specification = (
        f"apphot.{source_mode}.magnitude -> "
        f"apphot.{detrending_mode}.magnitude"
        f" : aperture_index in range({num_apertures})"
    )
    if _shape_fit_varies(sphotref_dr, shapefit_version):
        specification = (
            f"shapefit.{source_mode}.magnitude -> "
            f"shapefit.{detrending_mode}.magnitude; " + specification
        )

    _logger.info(
        "No %s-datasets configured; detrending %s",
        detrending_mode,
        repr(specification),
    )
    return parse_fit_datasets(specification)


def _check_fit_datasets_available(lc_fname, fit_datasets, detrending_mode):
    """
    Raise ConfigurationError if an input dataset is missing from a lightcurve.

    Without this, a mismatch between the datasets EPD produced and the ones TFA
    was configured to correct only surfaces deep inside the fitting, as a
    confusing error about the lightcurve structure.

    Args:
        lc_fname(str):    The lightcurve to check.

        fit_datasets([]):    See Correction.__init__().

        detrending_mode(str):    Either ``'epd'`` or ``'tfa'``.

    Returns:
        None
    """

    with LightCurveFile(lc_fname, "r") as lightcurve:
        for original_key, substitutions, _ in fit_datasets:
            try:
                lightcurve.check_for_dataset(original_key, **substitutions)
            except (IOError, KeyError) as error:
                raise ConfigurationError(
                    f"The dataset {original_key!r} ({substitutions!r}) that "
                    f"{detrending_mode.upper()} is configured to correct is "
                    f"not available in {lc_fname!r}. Check that the step which "
                    f"generates it ran with a matching "
                    f"{detrending_mode}-datasets configuration."
                ) from error


def resolve_fit_datasets(configuration, detrending_mode, lc_fnames):
    """
    Return the datasets to detrend, filling in a default if none were set.

    The relevant entry is removed from `configuration`.

    Args:
        configuration(dict):    The configuration of the detrending step. The
            ``<mode>_datasets`` entry is consumed.

        detrending_mode(str):    Either ``'epd'`` or ``'tfa'``.

        lc_fnames([str]):    The lightcurves about to be corrected. The first
            one is used to verify the datasets to correct actually exist.

    Returns:
        [(str, dict, str)]:
            See Correction.__init__().
    """

    fit_datasets = configuration.pop(detrending_mode + "_datasets")

    if fit_datasets is None:
        sphotref_dr_fname = configuration.get("single_photref_dr_fname")
        if sphotref_dr_fname is None:
            raise ConfigurationError(
                f"Neither {detrending_mode}-datasets nor "
                "single-photref-dr-fname is configured, so the datasets to "
                "detrend can neither be read from the configuration nor "
                "determined from the single photometric reference."
            )
        with DataReductionFile(sphotref_dr_fname, "r") as sphotref_dr:
            fit_datasets = _get_default_fit_datasets(
                sphotref_dr, detrending_mode
            )

    if lc_fnames:
        _check_fit_datasets_available(
            lc_fnames[0], fit_datasets, detrending_mode
        )

    return fit_datasets


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
            if cat_source_id is None:
                raise ConfigurationError(
                    f"None of the identifiers in {fname} appears in the "
                    "detrending catalog, so its magnitude and position "
                    "cannot be looked up!"
                )

            cat_info = catalog[cat_source_id]
            if result is None:
                result = numpy.empty(
                    len(lc_fnames),
                    dtype=[
                        # uint64 for integer IDs (a Gaia source id needs 64
                        # bits; numpy.dtype(int) is only 32-bit on Windows);
                        # string/bytes IDs keep their own dtype.
                        (
                            "ID",
                            (
                                numpy.uint64
                                if isinstance(
                                    cat_source_id, (int, numpy.integer)
                                )
                                else numpy.dtype(type(cat_source_id))
                            ),
                        ),
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
    if num_limbdark_coef != 2:
        raise ConfigurationError(
            f"{num_limbdark_coef} limb darkening coefficients were given; "
            "the quadratic law used for the transit model takes exactly two!"
        )

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
    # ``start_status`` is part of the signature the manager calls
    # with; the values this step accepts are declared in
    # ``allowed_start_status_values`` and checked there.
    # pylint: disable=unused-argument

    lc_fnames = list(lc_fnames)
    configuration["fit_datasets"] = resolve_fit_datasets(
        configuration, detrending_mode, lc_fnames
    )

    _logger.debug(
        "Generating %s performance statistics for %d light_curves",
        detrending_mode,
        len(lc_fnames),
    )

    with DataReductionFile(
        configuration["single_photref_dr_fname"], "r"
    ) as sphotref_dr:
        sphotref_header = sphotref_dr.get_frame_header()

    detrending_catalog_fname = configuration["detrending_catalog"].format_map(
        sphotref_header
    )
    output_statistics_fname = configuration[
        f"{detrending_mode}_statistics_fname"
    ].format_map(sphotref_header)

    # Both are known only after header substitution. The single photref is
    # already scoped for the whole step by the LC manager; each lightcurve
    # is scoped individually further down, inside the loops that read them.
    with error_context(
        related_files=[
            RelatedFile(
                FileKind.CATALOG, detrending_catalog_fname, role="input"
            ),
            RelatedFile(
                FileKind.OUTPUT,
                output_statistics_fname,
                role="expected_output",
            ),
        ]
    ):
        return _generate_statistics(
            lc_fnames,
            configuration,
            detrending_mode=detrending_mode,
            catalog_fname=detrending_catalog_fname,
            output_fname=output_statistics_fname,
            mark_progress=mark_progress,
        )


# The caller resolves the two filenames (header substitution) to build the
# error scope, so they arrive here rather than being re-derived.
# pylint: disable=too-many-arguments
def _generate_statistics(
    lc_fnames,
    configuration,
    *,
    detrending_mode,
    catalog_fname,
    output_fname,
    mark_progress,
):
    """Compute, augment and save the detrending statistics (see caller)."""

    catalog_sources = read_catalog_file(
        catalog_fname,
        add_gnomonic_projection=True,
    )

    statistics = recalculate_correction_statistics(
        lc_fnames,
        fit_datasets=configuration["fit_datasets"],
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

    if not path.exists(path.dirname(output_fname)):
        makedirs(path.dirname(output_fname))
    save_correction_statistics(statistics, output_fname)
    mark_progress(lc_fnames)
    return {"filename": output_fname, "preference_order": None}


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
