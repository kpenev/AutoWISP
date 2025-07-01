#!/usr/bin/env python3

"""Fit a smooth dependence of source extracted PSF parameters."""

import logging
from collections import namedtuple

import numpy

from autowisp.multiprocessing_util import setup_process
from autowisp.file_utilities import find_dr_fnames
from autowisp import DataReductionFile
from autowisp.fit_expression import (
    Interface as FitTermsInterface,
    iterative_fit,
)
from autowisp.evaluator import Evaluator
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)

_logger = logging.getLogger(__name__)

input_type = "dr"


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("" if args else input_type),
        inputs_help_extra="The DR files must already contain astrometry.",
        add_component_versions=("srcextract", "catalogue", "skytoframe"),
    )
    parser.add_argument(
        "--srcextract-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )
    parser.add_argument(
        "--srcextract-psf-params",
        nargs="+",
        default=None,
        help="List of the parameters describing PSF shapes of the extracted "
        "sources to fit a smooth dependence for. If left unspecified, smooth "
        "map will be fit for (`'S'`, `'D'`, `'K'`) or (`'fwhm'`, `'round'`, "
        "`'pa'`), whichever is available.",
    )
    parser.add_argument(
        "--srcextract-psfmap-terms",
        default="O3{x, y}*O1{phot_g_mean_mag}",
        help="An expression involving source extraction and/or catalogue "
        "variables for the terms to include in the smoothing fit.",
    )
    parser.add_argument(
        "--srcextract-psfmap-weights",
        default=None,
        type=str,
        help="An expression involving source extraction and/or catalogue "
        "variables for the weights to use for the smoothing fit.",
    )
    parser.add_argument(
        "--srcextract-psfmap-error-avg",
        default="median",
        help="How to average fitting residuals for outlier rejection. ",
    )
    parser.add_argument(
        "--srcextract-psfmap-rej-level",
        type=float,
        default=5.0,
        help="How far away from the fit should a point be before it is rejected"
        " in units of error_avg.",
    )
    parser.add_argument(
        "--srcextract-psfmap-max-rej-iter",
        type=int,
        default=20,
        help="The maximum number of rejection/re-fitting iterations to perform."
        "If the fit has not converged by then, the latest iteration is "
        "accepted. Default: %(default)s",
    )
    return parser.parse_args(*args)


def get_predictors_and_weights(
    matched_sources, fit_terms_expression, weights_expression
):
    """Return the matrix of predictors to use for fitting."""
    _logger.debug("Matched columns: %s", repr(matched_sources.columns))
    if weights_expression is None:
        return (FitTermsInterface(fit_terms_expression)(matched_sources), None)
    # TODO fix matched_sources to records in Evaluator not here
    return (
        FitTermsInterface(fit_terms_expression)(matched_sources),
        Evaluator(matched_sources.to_records(index=False))(weights_expression),
    )


def get_psf_param(matched_sources, psf_parameters):
    """Return a numpy structured array of the PSF parameters."""

    result = numpy.empty(
        len(matched_sources),
        dtype=[(param, numpy.float64) for param in psf_parameters],
    )
    for param in psf_parameters:
        _logger.debug("Setting PSF param %s", repr(param))
        _logger.debug("Matched columns: %s", repr(matched_sources.columns))
        result[param] = matched_sources[param]

    return result


def detect_psf_parameters(matched_sources):
    """Return the default PSF parameters to fit for he given DR file."""

    for try_psf_params in [
        ("S", "D", "K"),
        ("s", "d", "k"),
        ("fwhm", "round", "pa"),
    ]:
        found_all = True
        for param in try_psf_params:
            if param not in matched_sources.columns:
                found_all = False
        if found_all:
            return try_psf_params
    return None


def smooth_srcextract_psf(
    dr_fname, configuration, mark_start, mark_end, **path_substitutions
):
    """
    Fit PSF parameters as polynomials of srcextract and catalogue info.

    Args:
        dr_fname(str):    The name of the DR file to smooth the source extracted
            parameters for.

        configuration(NamedTuple):    Configuration on how to do the smoothing.
            Should include the following attributes:

            psf_parameters([str]):    A list of the variables from the source
                extracted datasets to smooth. If None, an attempt is made to
                auto detect the parameters.

            fit_terms_expression(str):    A fitting terms expression defining
                the terms to include in the fit.

            weights_expression(str):    An expression involving source
                extraction and/or catalogue variables for the weights to use for
                the smoothing fit.

            error_avg:    See iterative_fit().

            rej_level:    See iterative_fit().

            max_rej_iter:    See iterative_fit().



        mark_start(callable):    Invoked with the name of the DR file before the
            contents of the file are updated.

        mark_end(callable):    Invoked with the name of the DR file after
            successfully completing all updates to the DR file.

        path_substitutions:    Any substitutions required to resolve the
            path to extracted sources, catalogue sources and the
            destinationdatasets and attributes created by this method.

    Returns:
        None
    """

    with DataReductionFile(dr_fname, "r+") as dr_file:
        matched_sources = dr_file.get_matched_sources(**path_substitutions)
        predictors, weights = get_predictors_and_weights(
            matched_sources,
            configuration.fit_terms_expression,
            configuration.weights_expression,
        )
        _logger.debug(
            "Predictors (%sx%d)}: %s", *predictors.shape, repr(predictors)
        )
        if weights is not None:
            _logger.debug(
                "Weights %s: %s", format(weights.shape), repr(weights)
            )
        else:
            _logger.debug("Not using weights")

        fit_results = {"coefficients": {}, "fit_res2": {}, "num_fit_src": {}}

        psf_parameters = configuration.psf_parameters
        if psf_parameters is None:
            psf_parameters = detect_psf_parameters(matched_sources)
        if psf_parameters is None:
            raise IOError(
                f"Matched sources in {dr_file.filename} do not contain a full "
                "set of either fistar or hatphot PSF parameters."
            )

        psf_param = get_psf_param(matched_sources, psf_parameters)
        for param_name in psf_parameters:
            (
                fit_results["coefficients"][param_name],
                fit_results["fit_res2"][param_name],
                fit_results["num_fit_src"][param_name],
            ) = iterative_fit(
                predictors,
                psf_param[param_name],
                weights=weights,
                error_avg=configuration.error_avg,
                rej_level=configuration.rej_level,
                max_rej_iter=configuration.max_rej_iter,
                fit_identifier=f"Extracted sources PSF {param_name:s} map",
            )
            if fit_results["coefficients"][param_name] is None:
                mark_start(dr_fname)
                mark_end(dr_fname, -2)
                return

        mark_start(dr_fname)
        dr_file.save_source_extracted_psf_map(
            fit_results=fit_results,
            fit_configuration=configuration,
            **path_substitutions,
        )
        mark_end(dr_fname)


def get_dr_substitutions(configuration):
    """Return the path substitutions needed to resolve input datasets."""

    return {
        what + "_version": configuration[what + "_version"]
        for what in ["srcextract", "catalogue", "skytoframe"]
    }


def fit_source_extracted_psf_map(
    dr_collection, start_status, configuration, mark_start, mark_end
):
    """Fit a smooth dependence of source extraction PSF for a DR collection."""

    assert start_status is None

    # This defines a type not variable
    # pylint: disable=invalid-name
    SmoothingConfigType = namedtuple(
        "SmoothingConfigType",
        [
            "psf_parameters",
            "fit_terms_expression",
            "weights_expression",
            "error_avg",
            "rej_level",
            "max_rej_iter",
        ],
    )
    # pylint: enable=invalid-name

    smoothing_config = SmoothingConfigType(
        psf_parameters=configuration["srcextract_psf_params"],
        fit_terms_expression=configuration["srcextract_psfmap_terms"],
        weights_expression=configuration["srcextract_psfmap_weights"],
        **{
            fit_config: configuration["srcextract_psfmap_" + fit_config]
            for fit_config in ["error_avg", "rej_level", "max_rej_iter"]
        },
    )

    for dr_fname in dr_collection:
        smooth_srcextract_psf(
            dr_fname=dr_fname,
            configuration=smoothing_config,
            mark_start=mark_start,
            mark_end=mark_end,
            **get_dr_substitutions(configuration),
        )


def cleanup_interrupted(interrupted, configuration):
    """Remove the source extracted PSF map from the given DR file."""

    path_substitutions = get_dr_substitutions(configuration)

    for dr_fname, status in interrupted:
        assert status == 0

        with DataReductionFile(dr_fname, "r+") as dr_file:
            dr_file.delete_dataset("srcextract.psf_map", **path_substitutions)

    return -1


def has_astrometry(dr_fname, substitutions):
    """Check if the DR file contains a sky-to-frame transformation."""

    with DataReductionFile(dr_fname, mode="r") as dr_file:
        try:
            dr_file.check_for_dataset(
                "skytoframe.coefficients", **substitutions
            )
            return True
        except IOError:
            return False


def main():
    """Run the step from the command line."""

    configuration = parse_command_line()
    setup_process(task="main", **configuration)
    dr_substitutions = get_dr_substitutions(configuration)
    fit_source_extracted_psf_map(
        [
            dr_fname
            for dr_fname in find_dr_fnames(
                configuration.pop("dr_files"),
                configuration.pop("srcextract_only_if"),
            )
            if has_astrometry(dr_fname, dr_substitutions)
        ],
        None,
        configuration,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
