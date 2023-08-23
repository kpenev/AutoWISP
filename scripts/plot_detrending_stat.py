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

from matplotlib import pyplot

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
        'filter',
        help='The name of the filter whose magnitude to use as the x-axis of '
        'the plot.'
    )
    parser.add_argument(
        '--catalogue-fname', '--cat',
        default='astrometry_catalogue.ucac4',
        help='The catalogue file used for magnitude fitting. Default: '
        '\'%(default)s.\''
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


def detect_stat_columns(stat_fname):
    """
    Automatically detect the relevant columns in the statistics file.

    Args:
        stat_fname:     The name of the magnitude fitting generated statitics
            file to process.

    Returns:
        tuple(list, list):
            list:
                The indices of the columns (starting from zero) containing the
                number of unrejected frames for each extracted photometry.

            list:
                The indices of the columns (starting from zero) containing the
                scatter (e.g. median deviation around the median for magfit).

            list:
                The indices of the columns (starting from zero) containing the
                median of the formal magnitude error, or None if that is not
                tracked.
    """


    with open(stat_fname, 'r', encoding='utf-8') as stat_file:
        for first_line in stat_file:
            first_line = first_line.strip()
            if first_line[0] != '#':
                split_line = first_line.strip().split()
                num_columns = len(split_line)
                if first_line.startswith('HAT-'):
                    columns_per_set = 5
                    assert (num_columns - 1) % columns_per_set == 0
                    num_stat = (num_columns - 1) // columns_per_set
                    assert num_stat % 2 == 0
                    num_stat //= 2
                    column_mask = numpy.arange(0,
                                               num_stat * columns_per_set,
                                               columns_per_set,
                                               dtype=int)
                    return (2 + column_mask,
                            5 + column_mask,
                            columns_per_set * num_stat + 3 + column_mask)

                assert split_line[0] == '2MASSID'
                assert num_columns % 2 == 0
                result = numpy.empty((2, (num_columns - 4) // 2), dtype=int)
                scatter_ind = unrejected_ind = 0
                for column_index, column_name in enumerate(split_line):
                    if column_name.startswith('rms_'):
                        result[1, scatter_ind] = column_index
                        scatter_ind += 1
                    elif column_name.startswith('num_finite_'):
                        result[0, unrejected_ind] = column_index
                        unrejected_ind += 1
                assert scatter_ind == result.shape[1]
                assert unrejected_ind == result.shape[1]
                return result[0], result[1], None
    raise IOError(
        'Magnitude fitting statistics file contains no lines not marked as'
        ' comment!'
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
def plot_best_scatter(match_fname,
                      *,
                      magnitude_column,
                      num_unrejected_columns,
                      min_unrejected_fraction,
                      scatter_columns,
                      expected_scatter_columns,
                      xi_column,
                      eta_column,
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

    data = numpy.genfromtxt(match_fname)
    print('data: ' + repr(data))
    print(repr(data[:, num_unrejected_columns]))
    min_unrejected = numpy.min(data[:, num_unrejected_columns], 1)
    print('min_unrejected: ' + repr(min_unrejected))
    print('max(min_unrejected): ' + repr(numpy.max(min_unrejected)))
    many_unrejected = (min_unrejected
                       >
                       min_unrejected_fraction * numpy.max(min_unrejected))
    print('many_unrejected: ' + repr(many_unrejected))

    magnitude = data[:, magnitude_column][many_unrejected]

    scatter = data[:, scatter_columns][many_unrejected]
    scatter[scatter == 0.0] = numpy.nan
    best_ind = numpy.nanargmin(scatter, 1)
    scatter = 10.0**(numpy.nanmin(scatter, 1) / 2.5) - 1.0
    print(repr(scatter_columns))

    print('Magnitudes: ' + repr(magnitude))
    print('scatter: ' + repr(scatter))

    if expected_scatter_columns is not None:
        expected_scatter = data[:, expected_scatter_columns][many_unrejected]
        expected_scatter[expected_scatter == 0.0] = numpy.nan
        expected_scatter = 10.0**(numpy.nanmin(expected_scatter, 1) / 2.5) - 1.0
        print('expected error: ' + repr(expected_scatter))

    distance2 = (
        data[:, xi_column][many_unrejected]**2 +
        data[:, eta_column][many_unrejected]**2
    )


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

    catalogue_columns = detect_catalogue_columns(
        catalogue_fname=cmdline_args.catalogue_fname
    )
    (
        num_unrejected_columns,
        scatter_columns,
        expected_scatter_columns
    ) = detect_stat_columns(
        cmdline_args.magfit_stat_fname
    )

    num_catalogue_columns = len(catalogue_columns)

    scatter_columns += num_catalogue_columns
    if expected_scatter_columns is not None:
        expected_scatter_columns += num_catalogue_columns

    bottom_envelope = LogPlotLine(*cmdline_args.bottom_envelope)

    with TemporaryFileName() as match_fname:
        match_stat_to_catalogue(
            stat_fname=cmdline_args.magfit_stat_fname,
            catalogue_fname=cmdline_args.catalogue_fname,
            catalogue_id_column=catalogue_columns.index('#ID'),
            match_fname=match_fname
        )

        magnitude, best_ind = plot_best_scatter(
            match_fname,
            magnitude_column=catalogue_columns.index(cmdline_args.filter),
            num_unrejected_columns=(num_unrejected_columns
                                    +
                                    num_catalogue_columns),
            min_unrejected_fraction=cmdline_args.min_unrejected_fraction,
            scatter_columns=scatter_columns,
            expected_scatter_columns=expected_scatter_columns,
            xi_column=catalogue_columns.index('xi'),
            eta_column=catalogue_columns.index('eta'),
            distance_splits=cmdline_args.distance_splits,
            bottom_envelope=bottom_envelope,
            theoretical_marker_size=cmdline_args.theoretical_marker_size,
            empirical_marker_size=cmdline_args.empirical_marker_size
        )

        pyplot.xlim(cmdline_args.plot_x_range)
        pyplot.ylim(cmdline_args.plot_y_range)
        pyplot.ylabel('MAD')

        pyplot.xlabel(cmdline_args.filter + ' [mag]')
        pyplot.grid(True, which='both')
        pyplot.savefig(cmdline_args.output)

        pyplot.cla()
        pyplot.clf()

        pyplot.plot(magnitude, best_ind, '.k')
        pyplot.xlim(cmdline_args.plot_x_range)

        pyplot.savefig(splitext(cmdline_args.output)[0]
                       +
                       '_best_ind'
                       +
                       splitext(cmdline_args.output)[1])


if __name__ == '__main__':
    create_plot(parse_command_line())
