"""Unified interface to the detrending algorithms."""

from multiprocessing import Pool
import logging

import numpy
from scipy.optimize import minimize
import pandas

from autowisp.multiprocessing_util import setup_process
from autowisp import DataReductionFile, LightCurveFile
from autowisp.catalog import read_catalog_file
from autowisp.database.interface import db_engine
from .epd_correction import EPDCorrection
from .reconstructive_correction_transit import ReconstructiveCorrectionTransit


def save_correction_statistics(correction_statistics, filename):
    """Save the given statistics (result of apply_parallel_correction)."""

    print("Correction statistics:\n" + repr(correction_statistics))
    dframe = pandas.DataFrame(
        {
            column: correction_statistics[column]
            for column in ["ID", "mag", "xi", "eta"]
        },
    )

    num_photometries = correction_statistics["rms"][0].size

    for prefix in ["rms", "num_finite"]:
        for phot_index in range(num_photometries):
            dframe[prefix + f"_{phot_index:02d}"] = correction_statistics[
                prefix
            ][:, phot_index]

    with open(filename, "w", encoding="utf-8") as outf:
        dframe.to_string(outf, col_space=25, index=False, justify="left")


def load_correction_statistics(filename, add_catalog=False):
    """Read a previously stored statistics from a file."""

    with DataReductionFile() as mem_dr:
        dframe = pandas.read_csv(
            filename, delim_whitespace=True, index_col="ID"
        )

        num_sources, num_photometries = dframe.shape
        num_photometries = (num_photometries - 3) // 2

        result_dtype = EPDCorrection.get_result_dtype(num_photometries)
        if add_catalog:
            catalog = read_catalog_file(
                add_catalog, add_gnomonic_projection=True
            )
            result_dtype += [
                (col, catalog[col].dtype)
                for col in catalog.columns
                if col not in ["xi", "eta"]
            ]
            dframe = dframe.drop(columns=["xi", "eta"]).join(
                catalog, how="inner"
            )

        result = numpy.empty(num_sources, dtype=result_dtype)
        for column in dframe.columns:
            if column.startswith("rms_") or column.startswith("num_finite_"):
                continue
            result[column] = dframe[column]

        for prefix in ["rms", "num_finite"]:
            for phot_index in range(num_photometries):
                result[prefix][:, phot_index] = dframe[
                    prefix + f"_{phot_index:02d}"
                ]

        if "2MASSID" in dframe.columns:
            for index, source_id in enumerate(dframe["2MASSID"]):
                result["ID"][index] = mem_dr.parse_hat_source_id(source_id)
        else:
            result["ID"] = dframe.index

    return result


def calculate_iterative_rejection_scatter(
    values,
    calculate_average,
    calculate_scatter,
    outlier_threshold,
    max_outlier_rejections,
    *,
    return_average=False,
):
    """
    Calculate the scatter for a dataset, with outlier rejectio iterations.

    Args:
        values(numpy array like):     The data to calculate the scatter of.

        calculate_average(callable):    A callable that returns the average of
            the data aroung which the scatter will be calculated.

        calculate_scatter(callable):    The scatter is defined as the square
            root of whatever get_scatter calculates from the square deviations
            of the data from the average.

        outlier_threshold(float):    In units of the scatter, how far away
            should a point be from the average to be considered an outlier.

        max_outlier_rejections(int):    The maximum number of iterations between
            outlier rejection and re-calculating the scatter to perform.

        return_average(bool):    Should the average of the poinst also be
            returned?

    Returns:
        float, int:
            The scatter in values and the number of non-rejected points in the
            last scatter calculation.
    """

    include_points = numpy.ones(values.shape, dtype=bool)
    non_outliers = True
    for _ in range(max_outlier_rejections):
        include_points = numpy.logical_and(include_points, non_outliers)
        average = calculate_average(values[include_points])
        square_deviations = numpy.square(values - average)
        square_scatter = calculate_scatter(square_deviations[include_points])
        non_outliers = (
            square_deviations <= outlier_threshold**2 * square_scatter
        )
        if non_outliers[include_points].all():
            break

    if return_average:
        return numpy.sqrt(square_scatter), include_points.sum(), average
    return numpy.sqrt(square_scatter), include_points.sum()


def recalculate_correction_statistics(
    lc_fnames,
    fit_datasets,
    variables,
    lc_points_filter_expression,
    **calculate_scatter_config,
):
    """
    Extract the performance metrics for a de-trending step directly from LCs.

    Args:
        lc_fnames([str]):    The filenames of the light curves that were
            corrected.

        fit_datasets:    See Correction.__init__().

        extra_predictors:    See EPDCorrection.__init__().

        calculate__scatter_config:    Arguments passed directly to
            calculate_iterative_rejection_scatter().

    Returns:
        See apply_parallel_correction's return value.
    """

    result = numpy.empty(
        len(lc_fnames), dtype=EPDCorrection.get_result_dtype(len(fit_datasets))
    )

    for lc_index, fname in enumerate(lc_fnames):
        with LightCurveFile(fname, "r") as lightcurve:
            for fit_index, (_, substitutions, to_dset) in enumerate(
                fit_datasets
            ):
                try:
                    stat_points = lightcurve.evaluate_expression(
                        variables, lc_points_filter_expression
                    )
                    # False positive
                    # pylint: disable=unbalanced-tuple-unpacking
                    (
                        result["rms"][lc_index][fit_index],
                        result["num_finite"][lc_index][fit_index],
                    ) = calculate_iterative_rejection_scatter(
                        lightcurve.get_dataset(to_dset, **substitutions)[
                            stat_points
                        ],
                        **calculate_scatter_config,
                    )
                    # pylint: enable=unbalanced-tuple-unpacking
                except OSError:
                    result["rms"][lc_index][fit_index] = numpy.nan
                    result["num_finite"][lc_index][fit_index] = 0
    return result


def pool_init(config):
    """Setup pool process."""

    db_engine.dispose()
    setup_process(**config)


def apply_parallel_correction(
    lc_fnames, correct, num_parallel_processes, **config
):
    """
    Correct LCs running one of the detrending algorithms in parallel.

    Args:
        lc_fnames([str]):    The filenames of the light curves to correct.

        correct(Correction):    The underlying correction to apply in parallel.

        num_parallel_processes(int):    The maximum number of parallel processes
            to use.

        statistics_fname(str):    Filename to use for saving the statistics.

    Returns:
        numpy.array:
            The return values of correct.__call__() in the same order as
            lc_fnames.
    """

    logger = logging.getLogger(__name__)

    logger.info("Starting detrending %d light curves.", len(lc_fnames))

    if num_parallel_processes == 1:
        result = numpy.concatenate([correct(lcf) for lcf in lc_fnames])
    else:
        with Pool(
            num_parallel_processes, initializer=pool_init, initargs=(config,)
        ) as correction_pool:
            result = numpy.concatenate(correction_pool.map(correct, lc_fnames))

    logger.info("Finished detrending.")

    return result


def apply_reconstructive_correction_transit(
    lc_fname,
    correct,
    *,
    transit_model,
    transit_parameters,
    fit_parameter_flags,
    num_limbdark_coef,
):
    """
    Perform a reconstructive correction on a LC assuming it contains a transit.

    The corrected lightcurve, preserving the best-fit transit is saved in the
    lightcurve just like for non-reconstructive corrections.

    Args:
        transit_model:    Object which supports the transit model intefrace of
            pytransit.

        transit_parameters(scipy float array):    The full array of parameters
            required by the transit model's evaluate() method.

        fit_parameter_flags(scipy bool array):    Flags indicating parameters
            whose values should be fit for (by having a corresponding entry of
            True). Must match exactly the shape of transit_parameters.

        num_limbdark_coef(int):    How many of the transit parameters are limb
            darkening coefficinets? Those need to be passed to the model
            separately.

        correct(Correction):    Instance of one of the correction algarithms to
            make adaptive.

    Returns:
        (scipy array, scipy array):
            * The best fit transit parameters

            * The return value of ReconstructiveCorrectionTransit.__call__() for
              the best-fit transit parameters.
    """

    # This is intended to server as a callable.
    # pylint: disable=too-few-public-methods
    class MinimizeFunction:
        """Suitable callable for scipy.optimize.minimize()."""

        def __init__(self):
            """Create the underlying correction object."""

            self.correct = ReconstructiveCorrectionTransit(
                transit_model,
                correct,
                fit_amplitude=False,
            )
            self.transit_parameters = numpy.copy(transit_parameters)

        def __call__(self, fit_params):
            """
            Return the RMS residual of the corrected LC around a transit model.

            Args:
                fit_params(scipy array):    The values of the mutable model
                    parameters for the current minimization function evaluation.

            Returns:
                float:
                    RMS of the residuals after correcting around the transit
                    model with the given parameters.
            """

            self.transit_parameters[fit_parameter_flags] = fit_params
            return self.correct(
                lc_fname,
                self.transit_parameters[0],
                self.transit_parameters[1 : num_limbdark_coef + 1],
                *self.transit_parameters[num_limbdark_coef + 1 :],
                save=False,
            )["rms"]

    # pylint: enable=too-few-public-methods

    rms_function = MinimizeFunction()
    best_fit_transit = numpy.copy(transit_parameters)

    if fit_parameter_flags.any():
        minimize_result = minimize(
            rms_function, transit_parameters[fit_parameter_flags]
        )
        assert minimize_result.success
        best_fit_transit[fit_parameter_flags] = minimize_result.x

    return (
        best_fit_transit,
        rms_function.correct(
            lc_fname,
            best_fit_transit[0],
            best_fit_transit[1 : num_limbdark_coef + 1],
            *best_fit_transit[num_limbdark_coef + 1 :],
        ),
    )
