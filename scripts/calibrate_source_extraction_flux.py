#!/usr/bin/env python3
"""Calibrate the flux measurement of source extraction against catalogue."""

import logging
import re


from matplotlib import pyplot
import pandas
import numpy

from autowisp import DataReductionFile
from autowisp import Evaluator
from autowisp.file_utilities import find_dr_fnames
from autowisp.processing_steps.manual_util import \
    ManualStepArgumentParser
from autowisp.fit_expression import Interface as FitTermsInterface,\
    iterative_fit

_logger = logging.getLogger(__name__)

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type='dr',
        inputs_help_extra=(
            'The DR files must contain extracted sources and astrometry'
        ),
        add_component_versions=('srcextract', 'catalogue', 'skytoframe')
    )
    parser.add_argument(
        '--catalogue-brightness-expression', '--mag',
        default='phot_g_mean_mag',
        help='An expression involving catalogue variables to be used as the '
        'catalogue magnitude we are calibrating against. If empty, the '
        'brightness expression is fit for (see ``--brightness-terms`` '
        'argument).'
    )
    parser.add_argument(
        '--plot-fname',
        default=None,
        help='If specified the plot is saved under the given filename. If not, '
        'it is just displayed, but not saved.'
    )
    parser.add_argument(
        '--markersize',
        default=2.0,
        type=float,
        help='The size of the markers to use in the plot.'
    )
    parser.add_argument(
        '--use-header-vars',
        nargs='+',
        default=['AIRMASS'],
        help='The header variables to include in the fit for the brightness '
        '(ignored if fitting a single frame).'
    )
    parser.add_argument(
        '--magshift-terms',
        default=(
            'O1{'
                'phot_g_mean_mag-phot_rp_mean_mag, '
                'phot_g_mean_mag-phot_bp_mean_mag, '
                'AIRMASS'
            '} '
            '+'
            ' O2{x, y}'
        ),
        help='The terms to include in the fit for mag + 2.5log10(flux), where '
        'mag is the magnitude specified by --catalogue-brightness-expression.'
    )
    parser.add_argument(
        '--brightness-error-avg',
        default='nanmedian',
        help='How to calculate the scatter around the best fit brighgtness '
        'model.'
    )
    parser.add_argument(
        '--brightness-rej-threshold',
        default=5.0,
        type=float,
        help='Sources deviating from the best fit brightness model by more than'
        'this factor of the error average are discarded as outliers and fit is '
        'repeated.'
    )
    parser.add_argument(
        '--brightness-max-rej-iter',
        type=int,
        default=20,
        help='The maximum number of rejection/re-fitting iterations for the '
        'brightness to perform. If the fit has not converged by then, the '
        'latest iteration is accepted.'
    )
    parser.add_argument(
        '--plot-mag-range',
        nargs=2,
        type=float,
        default=None,
        help='The range to use for the x axis of the plot. If not specified, '
        'the range is determined from the data.'
    )
    parser.add_argument(
        '--plot-flux-range',
        nargs=2,
        type=float,
        default=None,
        help='The range to use for the y axis of the plot. If not specified, '
        'the range is determined from the data.'
    )


    return parser.parse_args()


def term_to_tex(expression):
    """Convert a fit terms expression to tex for plot label."""

    variable_map = {
        'phot_g_mean_mag': 'G',
        'phot_rp_mean_mag': 'R_p',
        'phot_bp_mean_mag': 'B_p',
        'AIRMASS': 'X'
    }
    var_parser = re.compile(r'(?:\W|^)(?P<var>[a-zA-Z]\w*)(?:\W|$)')
    match = var_parser.search(expression)
    if match is None:
        return expression.replace('**', '^')

    var_name = match.group('var')
    first, last = match.span()
    if expression[first] != '(' or expression[last - 1] != ')':
        if expression[first] != var_name[0]:
            first += 1
        if expression[last-1] != var_name[-1]:
            last -= 1
    return (
        expression[:first].replace('**', '^').replace('*', ' ')
        +
        variable_map.get(var_name, var_name)
        +
        term_to_tex(expression[last:])
    )


def coef_to_tex(coef):
    """Convert a numerical coefficient to tex for plot label."""

    coef_str = f'{coef:+.3g}'
    if 'e' not in coef_str:
        return coef_str
    mantissa, exponent = coef_str.split('e')
    return rf'{mantissa} \times 10^{{{int(exponent)}}}'


def get_best_fit_expression(best_fit_coef,
                            fit_terms,
                            catalogue_brightness_expression):
    """Return latex formatted expression of the best fit."""

    num_terms = len(fit_terms)
    return (
        f'${term_to_tex(catalogue_brightness_expression)}'
        +
        '$\n$'.join([
            ''.join([
                rf'{coef_to_tex(coef)} {term_to_tex(term)}'
                for coef, term in
                zip(
                    best_fit_coef[first_term:min(first_term + 6, num_terms)],
                    fit_terms[first_term:min(first_term + 6, num_terms)]
                )
            ])
            for first_term in range(0, num_terms, 6)
        ])
        +
        '$'
    )



def fit_brightness(matched, get_fit_terms, configuration):
    """Return the best fit brightness for each source and the coefficients."""

    predictors = get_fit_terms(matched)
    magnitude = Evaluator(
        matched
    )(
        configuration['catalogue_brightness_expression']
    )

    best_fit_coef, residual, num_fit_points = iterative_fit(
        predictors,
        #false positive
        #pylint: disable=invalid-unary-operand-type
        -magnitude - 2.5 * numpy.log10(matched['flux'].to_numpy(dtype=float)),
        #pylint: enable=invalid-unary-operand-type
        error_avg=configuration['brightness_error_avg'],
        rej_level=configuration['brightness_rej_threshold'],
        max_rej_iter=configuration['brightness_max_rej_iter'],
        fit_identifier='Source extracted vs catalogue brigtness'
    )
    best_fit_expression = get_best_fit_expression(
        best_fit_coef,
        get_fit_terms.get_term_str_list(),
        configuration['catalogue_brightness_expression']
    )
    _logger.info('Best fit brightness expression: %s', best_fit_expression)
    _logger.info(
        'Brightness fit residual %s based on %d/%d non-rejected sources',
        repr(residual),
        num_fit_points,
        matched['flux'].size
    )
    return magnitude + numpy.dot(best_fit_coef, predictors), best_fit_expression


def main(dr_collection, configuration):
    """Avoid polluting the global namespace."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in ['srcextract_version',
                             'catalogue_version',
                             'skytoframe_version']
    }
    get_fit_terms = FitTermsInterface(configuration['magshift_terms'])
    _logger.info(
        'Fitting for brightness using the following terms:\n\t%s',
        '\n\t'.join([
            f'{term_i:03d}: {term_str!s}'
            for term_i, term_str in enumerate(
                get_fit_terms.get_term_str_list()
            )
        ])
    )

    matched = None
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r') as dr_file:
            header = dr_file.get_frame_header()
            dr_matched = dr_file.get_matched_sources(**path_substitutions)
            dr_matched = dr_matched[dr_matched['nsatpix'] == 0]
            if len(dr_collection) > 1:
                for keyword in configuration['use_header_vars']:
                    dr_matched.insert(len(dr_matched.columns),
                                      keyword,
                                      header[keyword])
            if matched is None:
                matched = dr_matched
            else:
                matched = pandas.concat((matched, dr_matched))

    magnitude, best_fit_expression = fit_brightness(matched,
                                                    get_fit_terms,
                                                    configuration)
    pyplot.semilogy(magnitude,
                    matched['flux'],
                    ',',
                    markersize=configuration['markersize'],
                    markerfacecolor='black',
                    markeredgecolor='none')
    if configuration['plot_flux_range'] is None:
        configuration['plot_flux_range'] = pyplot.ylim()
    if configuration['plot_mag_range'] is None:
        configuration['plot_mag_range'] = pyplot.xlim()

    line_mag = numpy.linspace(*pyplot.xlim(), 1000)
    pyplot.plot(line_mag,
                numpy.power(10.0, -line_mag / 2.5),
                '-k')
    pyplot.xlim(configuration['plot_mag_range'])
    pyplot.ylim(configuration['plot_flux_range'])
    pyplot.xlabel(best_fit_expression)
    pyplot.ylabel(r'$2.5 \log_{10} (flux)$')

    if configuration['plot_fname'] is None:
        pyplot.show()
    else:
        pyplot.savefig(configuration['plot_fname'])

if __name__ == '__main__':
    cmdline_config = parse_command_line()
    main(list(find_dr_fnames(cmdline_config.pop('dr_files'))),
         cmdline_config)
