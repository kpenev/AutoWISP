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
        processing_step='calibrate_source_extraction_flux',
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
        '--use-header-vars',
        nargs='+',
        default=['AIRMASS'],
        help='The header variables to include in the fit for the brightness '
        '(ignored if fitting a single frame).'
    )
    parser.add_argument(
        '--magshift-terms',
        default='O1{V-B, V-R, V-I, 2.5 * log10(exp(1.0)) * AIRMASS}',
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
    magnitude = Evaluator(
        matched
    )(
        configuration['catalogue_brightness_expression']
    )

    best_fit_coef, residual, num_fit_points = iterative_fit(
        predictors,
        -magnitude - 2.5 * numpy.log10(matched['flux'].to_numpy(dtype=float)),
        error_avg=configuration['brightness_error_avg'],
        rej_level=configuration['brightness_rej_threshold'],
        max_rej_iter=configuration['brightness_max_rej_iter'],
        fit_identifier='Source extracted vs catalogue brigtness'
    )
    _logger.info(
        'Best fit brightness expression: %s',
        ' + '.join([
            f'{coef!r} * {term!s}'
            for coef, term in zip(best_fit_coef,
                                  get_fit_terms.get_term_str_list())
        ]) + f' + {configuration["catalogue_brightness_expression"]}'
    )
    _logger.info(
        'Brightness fit residual %s based on %d/%d non-rejected sources',
        repr(residual),
        num_fit_points,
        matched['flux'].size
    )
    return magnitude + numpy.dot(best_fit_coef, predictors)


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
            if len(dr_collection) > 1:
                dr_matched.insert(len(dr_matched.columns),
                                  'AIRMASS',
                                  header['AIRMASS'])
            if matched is None:
                matched = dr_matched
            else:
                matched = pandas.concat((matched, dr_matched))


    magnitude = fit_brightness(matched, get_fit_terms, configuration)
    pyplot.semilogy(magnitude,
                    matched['flux'],
                    'o',
                    markersize=configuration['markersize'])
    points_xlim = pyplot.xlim()
    points_ylim = pyplot.ylim()
    line_mag = numpy.linspace(*pyplot.xlim(), 1000)
    pyplot.plot(line_mag,
                numpy.power(10.0, -line_mag / 2.5),
                '-k')
    pyplot.xlim(points_xlim)
    pyplot.ylim(points_ylim)
    if configuration['plot_fname'] is None:
        pyplot.show()
    else:
        pyplot.savefig(configuration['plot_fname'])

if __name__ == '__main__':
    cmdline_config = parse_command_line()
    main(list(find_dr_fnames(cmdline_config.pop('dr_files'))),
         cmdline_config)
