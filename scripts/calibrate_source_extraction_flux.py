#!/usr/bin/env python3
"""Calibrate the flux measurement of source extraction against catalogue."""

import logging

from matplotlib import pyplot
import pandas
import numpy

from superphot_pipeline import DataReductionFile
from superphot_pipeline import Evaluator
from superphot_pipeline.file_utilities import find_dr_fnames
from superphot_pipeline.processing_steps.manual_util import \
    ManualStepArgumentParser
from superphot_pipeline.fit_expression import Interface as FitTermsInterface,\
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
        default='V',
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
        help='The size of the markers to use in the plot.'
    )
    parser.add_argument(
        '--brightness-terms',
        default='O1{B, V, R, I}',
        help='The terms to include in the fit for the brightness expression.'
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

    return parser.parse_args()


def fit_brightness(matched, get_fit_terms, configuration):
    """Return the best fit brightness for each source and the coefficients."""

    predictors = get_fit_terms(matched)
    best_fit_coef, residual, num_fit_points = iterative_fit(
        predictors,
        -2.5 * numpy.log10(matched['flux'].to_numpy(dtype=float)),
        error_avg=configuration['brightness_error_avg'],
        rej_level=configuration['brightness_rej_threshold'],
        max_rej_iter=configuration['brightness_max_rej_iter'],
        fit_identifier='Source extracted vs catalogue brigtness'
    )
    _logger.info(
        'Best fit brightness expression: %s',
        ' + '.join([
            '{!r} * {!s}'.format(coef, term)
            for coef, term in zip(best_fit_coef,
                                  get_fit_terms.get_term_str_list())
        ])
    )
    _logger.info(
        'Brightness fit residual %s based on %d/%d non-rejected sources',
        repr(residual),
        num_fit_points,
        matched['flux'].size
    )
    return numpy.dot(best_fit_coef, predictors)


def main(dr_collection, configuration):
    """Avoid polluting the global namespace."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in ['srcextract_version',
                             'catalogue_version',
                             'skytoframe_version']
    }
    offsets = pandas.Series()
    if not configuration['catalogue_brightness_expression']:
        get_fit_terms = FitTermsInterface(configuration['brightness_terms'])
        _logger.info(
            'Fitting for brightness using the following terms:\n\t%s',
            '\n\t'.join([
                '{:03d}: {!s}'.format(term_i, term_str)
                for term_i, term_str in enumerate(
                    get_fit_terms.get_term_str_list()
                )
            ])
        )

    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r') as dr_file:
            matched = dr_file.get_matched_sources(**path_substitutions)
        if configuration['catalogue_brightness_expression']:
            magnitude = Evaluator(
                matched
            )(
                configuration['catalogue_brightness_expression']
            )
            offsets = pandas.concat(
                [
                    offsets,
                    magnitude + 2.5 * numpy.log10(matched['flux'])
                ]
            )
            zero_point = numpy.median(offsets)
        else:
            magnitude = fit_brightness(matched, get_fit_terms, configuration)
            zero_point = 0
        pyplot.semilogy(magnitude,
                        matched['flux'],
                        'o',
                        markersize=configuration['markersize'])
    print('Zero point: ' + repr(zero_point))
    line_mag = numpy.linspace(*pyplot.xlim(), 1000)
    pyplot.plot(line_mag,
                numpy.power(10.0, (zero_point - line_mag) / 2.5),
                '-k')
    if configuration['plot_fname'] is None:
        pyplot.show()
    else:
        pyplot.savefig(configuration['plot_fname'])

if __name__ == '__main__':
    cmdline_config = parse_command_line()
    main(find_dr_fnames(cmdline_config.pop('dr_files')),
         cmdline_config)
