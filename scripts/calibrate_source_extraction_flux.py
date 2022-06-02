#!/usr/bin/env python3
"""Calibrate the flux measurement of source extraction against catalogue."""

from matplotlib import pyplot
import pandas
import numpy

from superphot_pipeline import DataReductionFile
from superphot_pipeline import Evaluator
from superphot_pipeline.image_utilities import find_dr_fnames
from superphot_pipeline.processing_steps.manual_util import get_cmdline_parser

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = get_cmdline_parser(
        __doc__,
        'dr',
        'The DR files must contain extracted sources and astrometry',
        ('srcextract', 'catalogue', 'skytoframe')
    )
    parser.add_argument(
        '--catalogue-brightness-expression', '--mag',
        default='V',
        help='An expression involving catalogue variables to be used as the '
        'catalogue magnitude we are calibrating against.'
    )
    parser.add_argument(
        '--plot-fname',
        default=None,
        help='If specified the plot is saved under the given filename. If not, '
        'it is just displayed, but not saved.'
    )
    return parser.parse_args()


def main(dr_collection, configuration):
    """Avoid polluting the global namespace."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in ['srcextract_version',
                             'catalogue_version',
                             'skytoframe_version']
    }
    offsets = pandas.Series()
    for dr_fname in dr_collection:
        with DataReductionFile(dr_fname, 'r') as dr_file:
            matched = dr_file.get_matched(**path_substitutions)
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
        pyplot.semilogy(magnitude,
                        matched['flux'],
                        'o')
    zero_point = numpy.median(offsets)
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
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    main(find_dr_fnames(cmdline_config.pop('dr_files')),
         cmdline_config)
