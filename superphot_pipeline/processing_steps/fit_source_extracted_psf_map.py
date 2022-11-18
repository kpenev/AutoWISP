#!/usr/bin/env python3

"""Fit a smooth dependence of source extracted PSF parameters."""

import numpy

from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline import DataReductionFile
from superphot_pipeline.fit_expression import\
    Interface as FitTermsInterface,\
    iterative_fit
from superphot_pipeline.evaluator import Evaluator
from superphot_pipeline.processing_steps.manual_util import\
    ManualStepArgumentParser

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='dr',
        inputs_help_extra='The DR files must already contain astrometry.',
        add_component_versions=('srcextract', 'catalogue', 'skytoframe')
    )
    parser.add_argument(
        '--srcextract-only-if',
        default='True',
        help='Expression involving the header of the input images that '
             'evaluates to True/False if a particular image from the specified '
             'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--srcextract-psf-params',
        nargs='+',
        default=None,
        help="List of the parameters describing PSF shapes of the extracted "
        "sources to fit a smooth dependence for. If left unspecified, smooth "
        "map will be fit for (`'S'`, `'D'`, `'K'`) or (`'fwhm'`, `'round'`, "
        "`'pa'`), whichever is available."
    )
    parser.add_argument(
        '--srcextract-psfmap-terms',
        default='O3{x, y, r, J-K}',
        help='An expression involving source extraction and/or catalogue '
        'variables for the weights to use for the smoothing fit.'
    )
    parser.add_argument(
        '--srcextract-psfmap-weights',
        default=None,
        type=str,
        help='An expression involving source extraction and/or catalogue '
        'variables for the weights to use for the smoothing fit.'
    )
    parser.add_argument(
        '--srcextract-psfmap-error-avg',
        default='median',
        help='How to average fitting residuals for outlier rejection. '
    )
    parser.add_argument(
        '--srcextract-psfmap-rej-level',
        type=float,
        default=5.0,
        help='How far away from the fit should a point be before it is rejected'
        ' in utins of error_avg.'
    )
    parser.add_argument(
        '--srcextract-psfmap-max-rej-iter',
        type=int,
        default=20,
        help='The maximum number of rejection/re-fitting iterations to perform.'
        'If the fit has not converged by then, the latest iteration is '
        'accepted. Default: %(default)s'
    )
    return parser.parse_args()


def get_predictors_and_weights(matched_sources,
                               fit_terms_expression,
                               weights_expression):
    """Return the matrix of predictors to use for fitting."""
    print('Matched columns: ' + repr(matched_sources.columns))
    if weights_expression is None:
        return (FitTermsInterface(fit_terms_expression)(matched_sources),
                None)
    #TODO fix matched_sources to records in Evaluator not here
    return (FitTermsInterface(fit_terms_expression)(matched_sources),
            Evaluator(matched_sources.to_records(index=False))(weights_expression))


def get_psf_param(matched_sources, psf_parameters):
    """Return a numpy structured array of the PSF parameters."""

    result = numpy.empty(
        len(matched_sources),
        dtype=[(param, numpy.float64) for param in psf_parameters]
    )
    for param in psf_parameters:
        print('Setting PSF param {0!r}'.format(param))
        print('Matched columns: ' + repr(matched_sources.columns))
        result[param] = matched_sources[param]

    return result


def detect_psf_parameters(matched_sources):
    """Return the default PSF parameters to fit for he given DR file."""

    for try_psf_params in [('S', 'D', 'K'), ('fwhm', 'round', 'pa')]:
        found_all = True
        for param in try_psf_params:
            if param not in matched_sources.columns:
                found_all = False
        if found_all:
            return try_psf_params
    return None


def smooth_srcextract_psf(dr_file,
                          psf_parameters,
                          fit_terms_expression,
                          weights_expression,
                          *,
                          error_avg,
                          rej_level,
                          max_rej_iter,
                          **path_substitutions):
    """
    Fit PSF parameters as polynomials of srcextract and catalogue info.

    Args:
        psf_parameters([str]):    A list of the variables from the source
            extracted datasets to smooth.

        fit_terms_expression(str):    A fitting terms expression defining
            the terms to include in the fit.

        weights_expression(str):    An expression involving source
            extraction and/or catalogue variables for the weights to use for
            the smoothing fit.

        error_avg:    See iterative_fit().

        rej_level:    See iterative_fit().

        max_rej_iter:    See iterative_fit().

        path_substitutions:    Any substitutions required to resolve the
            path to extracted sources, catalogue sources and the
            destinationdatasets and attributes created by this method.

    Returns:
        None
    """

    matched_sources = dr_file.get_matched_sources(**path_substitutions)
    predictors, weights = get_predictors_and_weights(matched_sources,
                                                     fit_terms_expression,
                                                     weights_expression)
    print('Predictors ({0:d}x{1:d}: '.format(*predictors.shape)
          +
          repr(predictors))
    print('Weights {0!r}: '.format(weights.shape) + repr(weights))

    fit_results = dict(coefficients=dict(),
                       fit_res2=dict(),
                       num_fit_src=dict())

    if psf_parameters is None:
        psf_parameters = detect_psf_parameters(matched_sources)
    if psf_parameters is None:
        raise IOError(
            'Matched sources in {0} do not contain a full set of '
            'either fistar or hatphot PSF parameters.'.format(dr_file.filename)
        )

    psf_param = get_psf_param(matched_sources, psf_parameters)
    for param_name in psf_parameters:
        (
            fit_results['coefficients'][param_name],
            fit_results['fit_res2'][param_name],
            fit_results['num_fit_src'][param_name]
        ) = iterative_fit(
            predictors,
            psf_param[param_name],
            weights=weights,
            error_avg=error_avg,
            rej_level=rej_level,
            max_rej_iter=max_rej_iter,
            fit_identifier='Extracted sources PSF %s map' % param_name
        )

    dr_file.save_source_extracted_psf_map(
        fit_results=fit_results,
        fit_terms_expression=fit_terms_expression,
        weights_expression=weights_expression,
        error_avg=error_avg,
        rej_level=rej_level,
        max_rej_iter=max_rej_iter,
        **path_substitutions
    )


def fit_srcextract_psf_map(dr_collection, configuration):
    """Fit a smooth dependence of source extraction PSF for a DR collection."""

    kwargs = {what + '_version': configuration[what + '_version']
              for what in ['srcextract', 'catalogue', 'skytoframe']}
    for fit_config in ['error_avg', 'rej_level', 'max_rej_iter']:
        kwargs[fit_config] = configuration['srcextract_psfmap_' + fit_config]


    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r+') as dr_file:
            smooth_srcextract_psf(
                dr_file=dr_file,
                psf_parameters=configuration['srcextract_psf_params'],
                fit_terms_expression=configuration['srcextract_psfmap_terms'],
                weights_expression=configuration['srcextract_psfmap_weights'],
                **kwargs
            )

if __name__ == '__main__':
    cmdline_config = parse_command_line()
    fit_srcextract_psf_map(find_dr_fnames(cmdline_config.pop('dr_files'),
                                          cmdline_config.pop('srcextract_only_if')),
                           cmdline_config)
