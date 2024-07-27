#!/usr/bin/env python3

#TODO:1st column- HAT-ID
#TODO:2nd column- Total points in LC
#TODO:3rd column- number of non rejected points
#TODO:4 - rms around the median
#TODO:5 - the mean deviation around the median
#TODO:Aperatures should be in order
#TODO:plot diff between magfit scatter and epd scatter
#TODO:plot one ontop of the other
#TODO: EPD stat has different setup but has labeled columns, just match to
#      catalogue, plot the catalogue matched files
#TODO: GRMatch all 3 simultaneously (grmatch cat with magfit_stat, then that
#      matched file grmatch to epd_stat)
#TODO: Have a common line argument to just do magfit not EPD
#TODO Plot Magfit ontop of EPD used matplotlib Zorder
#Executables deliberately prefixed by numbers to indicate step order.
#pylint: disable=invalid-name

#TODO: fix docstrings!!!

"""Plot scatter vs magnitude after singe/master magnitude fit."""

import subprocess
import os
from os.path import splitext
from tempfile import NamedTemporaryFile


from configargparse import ArgumentParser, DefaultsFormatter
import numpy
import pandas

from matplotlib import pyplot

from autowisp.catalog import read_catalog_file
from autowisp import Evaluator
from autowisp.diagnostics.detrending import detect_stat_columns

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=DefaultsFormatter,
        default_config_files=[]
    )
    parser.add_argument(
        'magfit_stat_fname',
        help='The magnitude fit statistics file to plot.'
    )
    parser.add_argument(
        '--catalogue-fname', '--cat',
        default='astrometry_catalogue.ucac4',
        help='The catalogue file used for magnitude fitting. Default: '
        '\'%(default)s.\''
    )
    parser.add_argument(
        '--magnitude-expression', '--mag',
        default='phot_g_mean_mag',
        help='Expression to evaluate using catalog columns indicating the '
        'brightness of stars.'
    )
    parser.add_argument(
        '--min-unrejected-fraction',
        type=float,
        default=0.5,
        help='The minimum fraction of points not declared as outliers for a '
        'star to be included in the plot.'
    )

    parser.add_argument(
        '--output', '-o',
        default='magfit_performance.eps',
        help='The filename to use for the generated plot. Default: '
        '\'%(default)s\'.'
    )
    parser.add_argument(
        '--plot-x-range',
        type=float,
        nargs=2,
        default=(6.0, 13.0),
        help='The range to impose on the x axis of the plot. Default: '
        '%(default)s'
    )
    parser.add_argument(
        '--plot-y-range',
        type=float,
        nargs=2,
        default=(0.01, 0.2),
        help='The range to impose on the x axis of the plot. Default: '
        '%(default)s'
    )
    parser.add_argument(
        '--distance-splits',
        nargs='+',
        type=float,
        default=None,
        help='Stars are split into rings of (xi, eta) at the given radii and '
        'each ring is plotted with different color. Up to six rings can be '
        'plotted before colors start to repeat.'
    )
    parser.add_argument(
        '--bottom-envelope',
        type=float,
        nargs=4,
        default=(10.0, 0.02, 7.5, 0.005),
        help='Define a line in the log-plot of scatter vs magnitude below wich '
        'points are not drawn.'
    )
    parser.add_argument(
        '--empirical-marker-size',
        type=float,
        default=15.0,
        help='The size of markers to use for plotting the empirical scatter in '
        'LCs.'
    )
    parser.add_argument(
        '--theoretical-marker-size',
        type=float,
        default=3.0,
        help='The size of markers to use for plotting the formal standard '
        'deviation (i.e. expected scatter).'
    )
    parser.add_argument(
        '--skip-first-stat',
        action='store_true',
        help='Skip the first photometry when plotting the statistics.'
    )

    return parser.parse_args()


#Context manager, no need for public methods.
#pylint: disable=too-few-public-methods
class TemporaryFileName:
    """
    A context manager that securely creates a closed temporary file.

    Attributes:
        filename:    The name of the file created.
    """

    def __init__(self, *args, **kwargs):
        """Create and close without deleteing a NamedTemporaryFile."""

        assert 'delete' not in kwargs
        #Defining a context manager!
        #pylint: disable=consider-using-with
        self.filename = NamedTemporaryFile(*args, delete=False, **kwargs).name
        #pylint: enable=consider-using-with

    def __enter__(self):
        """Return the name of the temporary file."""

        return self.filename

    def __exit__(self, *args, **kwargs):
        """Delete the newly created file."""

        assert os.path.exists(self.filename)
        os.remove(self.filename)
#pylint: enable=too-few-public-methods


def detect_catalogue_columns(catalogue_fname):
    """
        Automatically detect the relevant columns in the catalogue  file.

        Args:
            catalogue_fname:     The name of the catalogue file to process.

        Returns:
            catalogue_columns:      List of the catalogue columns used
        """
    with open(catalogue_fname, 'r', encoding='utf-8') as cat_file:
        columns = (cat_file.readline()).split()
        catalogue_columns = []
        for f in columns:
            catalogue_columns.append(list(f.split('['))[0])
        print('The catalogue_columns are:' + repr(catalogue_columns))
        print('The total catalogue columns is: ' + repr(len(catalogue_columns)))
        return catalogue_columns


def match_stat_to_catalogue(stat_fname,
                            catalogue_fname,
                            catalogue_id_column,
                            match_fname):
    """
    Match the sources in the stat file to a catalogue using grmatch.

    Args:
        stat_fname:    The name of the statistics file to add catalogue
            information to. Must have been generated by MagnitudeFitting.py.

        catalogue_fname:    The name of the catalogue file used during magnitude
            fitting which will be matched by source ID to the statistics file.

        catalogue_id_column:    The column number within catalogue (starting
            from zero) containing the source ID.

        match_fname:    The filename to save the matched file under.

    Returns:
        The name of a temporary file containing the match.
    """

    subprocess.run(
        [
            'grmatch',
            '--match-id',
            '--input', stat_fname,
            '--input-reference', catalogue_fname,
            '--col-ref-id', str(catalogue_id_column + 1),
            '--col-inp-id', '1',
            '--output-matched', match_fname
        ],
        check=True
    )


#Meant to define callable with pre-computed pieces
#pylint: disable=too-few-public-methods
class LogPlotLine:
    """Define a line in semilogy plot from a pair of points."""

    def __init__(self, x0, y0, x1, y1):
        """Create the line that goes through the two points given."""

        self.slope = numpy.log(y1 / y0) / (x1 - x0)
        self.offset = numpy.log(y1) - self.slope * x1

    def __call__(self, x):
        """Return the y value of the defined line at the given x value(s)."""

        result = numpy.exp(self.slope * x + self.offset)
        result[
            numpy.logical_and(x > 8.5, result < 6e-3)
        ] = 6e-3
        return result
#pylint: enable=too-few-public-methods


#No good way to simplify
#pylint: disable=too-many-locals
def plot_best_scatter(data,
                      *,
                      magnitude_expression,
                      num_unrejected_columns,
                      min_unrejected_fraction,
                      scatter_columns,
                      expected_scatter_columns,
                      distance_splits,
                      bottom_envelope,
                      empirical_marker_size,
                      theoretical_marker_size):
    """
    Plot the smallest scatter and error for each source vs magnitude.

    Args:
        match_fname:    The name of the file contaning the match between
            catalogue and statistics.

        magnitude_column:    The column index within `match_fname` containing
            the magnitude to use for the x-axis.

        num_unrejected_columns:    The column index within `match_fname`
            containing the number of unrejected frames for each source.

        scatter_columns:    The column index within `match_fname` containing
            the scatter around the meadn/median of the extracted magnitudes.

        expected_scatter_columns:    The column index within `match_fname`
            containing the expected scatter for the magnitude of each source.

    Returns:
        None
    """

    print(repr(data[num_unrejected_columns]))
    min_unrejected = numpy.min(data[num_unrejected_columns], 1)
    print('min_unrejected: ' + repr(min_unrejected))
    print('max(min_unrejected): ' + repr(numpy.max(min_unrejected)))
    many_unrejected = (min_unrejected
                       >
                       min_unrejected_fraction * numpy.max(min_unrejected))
    print('many_unrejected: ' + repr(many_unrejected))

    data = data[many_unrejected]
    magnitude = Evaluator(data)(magnitude_expression)

    scatter = data[scatter_columns]
    scatter[scatter == 0.0] = numpy.nan
    best_ind = numpy.nanargmin(scatter, 1)
    scatter = 10.0**(numpy.nanmin(scatter, 1) / 2.5) - 1.0

    print(f'Magnitudes:\n{magnitude!r}\nshape: {magnitude.shape}')
    print(f'Scatter:\n{scatter!r}\nshape: {scatter.shape}')
    print(f'Best index:\n{best_ind!r}\nshape: {best_ind.shape}')

    if expected_scatter_columns is not None:
        expected_scatter = data[expected_scatter_columns]
        expected_scatter[expected_scatter == 0.0] = numpy.nan
        expected_scatter = 10.0**(numpy.nanmin(expected_scatter, 1) / 2.5) - 1.0
        print('expected error: ' + repr(expected_scatter))

    distance2 = data['xi']**2 + data['eta']**2


    if distance_splits is None:
        distance_splits = [numpy.inf]
    else:
        distance_splits = list(distance_splits) + [numpy.inf]

    unplotted_sources = scatter > bottom_envelope(magnitude)

    for color_ind, max_distance in enumerate(sorted(distance_splits)):
        to_plot = numpy.logical_and(
            unplotted_sources,
            distance2 < max_distance**2
        )

        fmt = '.' + 'rgbcmy'[color_ind % 6]

        pyplot.semilogy(magnitude[to_plot],
                        scatter[to_plot],
                        fmt,
                        markersize=empirical_marker_size)

        unplotted_sources = numpy.logical_and(
            unplotted_sources,
            numpy.logical_not(to_plot)
        )

    if expected_scatter_columns is not None:
        pyplot.semilogy(magnitude,
                        expected_scatter,
                        '.k',
                        markersize=theoretical_marker_size,
                        markeredgecolor='none')

    return magnitude, best_ind
#pylint: enable=too-many-locals


def create_plot(cmdline_args):
    """Create the plot per the command line arguments."""

    data = read_catalog_file(cmdline_args.catalogue_fname,
                             add_gnomonic_projection=True)
    num_cat_columns = len(data.columns)
    data = data.join(
        pandas.read_csv(cmdline_args.magfit_stat_fname,
                        sep=r'\s+',
                        header=None,
                        index_col=0),
        how='inner'
    )
    print(f'Data:\n{data!r}')
    (
        num_unrejected_columns,
        scatter_columns,
        expected_scatter_columns
    ) = detect_stat_columns(data,
                            len(data.columns) - num_cat_columns,
                            cmdline_args.skip_first_stat)

    bottom_envelope = LogPlotLine(*cmdline_args.bottom_envelope)

    magnitude, best_ind = plot_best_scatter(
        data,
        magnitude_expression=cmdline_args.magnitude_expression,
        num_unrejected_columns=num_unrejected_columns,
        min_unrejected_fraction=cmdline_args.min_unrejected_fraction,
        scatter_columns=scatter_columns,
        expected_scatter_columns=expected_scatter_columns,
        distance_splits=cmdline_args.distance_splits,
        bottom_envelope=bottom_envelope,
        theoretical_marker_size=cmdline_args.theoretical_marker_size,
        empirical_marker_size=cmdline_args.empirical_marker_size
    )

    pyplot.xlim(cmdline_args.plot_x_range)
    pyplot.ylim(cmdline_args.plot_y_range)
    pyplot.ylabel('MAD')

    pyplot.xlabel(cmdline_args.magnitude_expression)
    pyplot.grid(True, which='both')
    pyplot.savefig(cmdline_args.output)

    pyplot.cla()
    pyplot.clf()

    pyplot.plot(magnitude, best_ind, '.k')
    pyplot.xlim(cmdline_args.plot_x_range)

    pyplot.xlabel(cmdline_args.magnitude_expression)
    pyplot.ylabel('Photometry index with smallest scatter')

    pyplot.savefig(splitext(cmdline_args.output)[0]
                   +
                   '_best_ind'
                   +
                   splitext(cmdline_args.output)[1])


if __name__ == '__main__':
    create_plot(parse_command_line())
