#!/usr/bin/env python3

"""Extract statistics directly from the LCs and plot."""

from functools import partial
from itertools import count

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from asteval import Interpreter

from superphot_pipeline import LightCurveFile
from superphot_pipeline.file_utilities import find_lc_fnames
from superphot_pipeline.light_curves.apply_correction import\
    calculate_iterative_rejection_scatter

def add_scatter_arguments(parser, multi_mode=True):
    """Add arguments to cmdline parser determining how scatter is calculated."""

    parser.add_argument(
        '--detrending-mode',
        **(dict(nargs='+') if multi_mode else dict()),
        default=('tfa',),
        choices=['magfit', 'epd', 'tfa'],
        help='Which version of the detrending to plot. If multiple modes are '
        'selected each is plotted with a different set of colors (per '
        '--distance-splits).'
    )
    parser.add_argument(
        '--magfit-iteration',
        type=int,
        default=0,
        help='Which iteration of magnitude fitting to use.'
    )
    parser.add_argument(
        '--average',
        default=numpy.nanmedian,
        type=partial(getattr, numpy),
        help='How to calculate the average of the LC around which the scatter '
        'will be measured.'
    )
    parser.add_argument(
        '--statistic',
        default=numpy.nanmedian,
        type=partial(getattr, numpy),
        help='How to get a summary statistic out of the square deviation of '
        'the LC points from the average.'
    )
    parser.add_argument(
        '--outlier-threshold',
        default=3.0,
        type=float,
        help='LC points that deviate from the mean by more than this value '
        'times the root average square are excluded and the statistic is '
        're-calculated. This process is repeated until either no points are '
        'rejected or --max-outlier-rejections is reached.'
    )
    parser.add_argument(
        '--max-outlier-rejections',
        type=int,
        default=20,
        help='The maximum number of outlier rejection iteratinos to performe.'
    )
    parser.add_argument(
        '--min-lc-length',
        type=int,
        default=200,
        help='Lightcurves should contain at least this main points, after '
        'outlier rejection, to be included in the plot.'
    )


def add_plot_config_arguments(parser):
    """Add arguments configuring the generated plot."""

    parser.add_argument(
        '--plot-x-range',
        nargs=2,
        type=float,
        default=None,
        help='The range for the x axis of the plot.'
    )
    parser.add_argument(
        '--plot-y-range',
        nargs=2,
        type=float,
        default=None,
        help='The range for the y axis of the plot. Leave unspecified to allow '
        'matplotlib to determine it automatically.'
    )
    parser.add_argument(
        '--plot-marker-size',
        type=int,
        default=2,
        help='The size of the markers to use in the plot.'
    )
    parser.add_argument(
        '--plot-fname', '-o',
        default=None,
        help='If not specified the plot is shown, if specified the plot is '
        'saved with the given filename.'
    )


def parse_command_line():
    """Return the command line arguments as attributes of an object."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['../TESS_SEC07_CAM01/lc_scatter_plots.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        'lightcurves',
        nargs='+',
        help='The filenames of the lightcurves to include in the plot.'
    )

    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )
    parser.add_argument(
        '--catalogue-fname', '--catalogue', '--cat',
        default='catalogue.ucac4',
        help='The name of the catalogue file containing all sources to '
        'include in the plot (may contain extra sources too).'
    )
    parser.add_argument(
        '--x-expression',
        default='R',
        help='An expression involving catalogue informatio to use as the x-axis'
        ' of the plot.'
    )

    add_scatter_arguments(parser)
    add_plot_config_arguments(parser)

    parser.add_argument(
        '--distance-splits',
        nargs='+',
        type=float,
        default=[],
        help='Stars are split into rings of (xi, eta) at the given radii and '
        'each ring is plotted with different color. Up to six rings can be '
        'plotted before colors start to repeat.'
    )

    return parser.parse_args()

def get_scatter_config(detrending_mode, cmdline_args):
    """Return the configuration to use for extracting scatter from LC."""

    return dict(
        detrending_mode=detrending_mode,
        magfit_iteration=cmdline_args.magfit_iteration,
        min_lc_length=cmdline_args.min_lc_length,
        calculate_average=cmdline_args.average,
        calculate_scatter=cmdline_args.statistic,
        outlier_threshold=cmdline_args.outlier_threshold,
        max_outlier_rejections=cmdline_args.max_outlier_rejections
    )

def get_minimum_scatter(lc_fname,
                        detrending_mode,
                        magfit_iteration,
                        min_lc_length,
                        **scatter_config):
    """
    Find the photometry with the smallest scatter and return that scatter.

    Args:
        lightcurve(LightCurveFile):    The lightcurve to find the smallest
            scatter for.

        detrending_mode(str):    Which version of detrending to extract the
            scatter for. Should be one of `'magfit'`, `'epd'`, or `'tfa'`.

        scatter_config:    Arguments to pass to get_scatter() configuring
            how the scatter is to be calculated.
    """

    with LightCurveFile(lc_fname, 'r') as lightcurve:
        bjd = lightcurve.get_dataset('skypos.BJD')
        try:
            best_mags = lightcurve.get_dataset(
                'shapefit.' + detrending_mode + '.magnitude',
                magfit_iteration=magfit_iteration
            )
            min_scatter, selected_lc_length = (
                calculate_iterative_rejection_scatter(
                    best_mags,
                    **scatter_config
                )
            )
        except OSError:
            selected_lc_length = 0

        if selected_lc_length < min_lc_length:
            min_scatter = numpy.inf

        try:
            for aperture_index in count():
                magnitudes = lightcurve.get_dataset(
                    'apphot.' + detrending_mode + '.magnitude',
                    aperture_index=aperture_index,
                    magfit_iteration=magfit_iteration
                )
                scatter, lc_length = calculate_iterative_rejection_scatter(
                    magnitudes,
                    **scatter_config
                )
                if lc_length > min_lc_length and scatter < min_scatter:
                    min_scatter = scatter
                    selected_lc_length = lc_length
                    best_mags = magnitudes
        except OSError:
            lightcurve.close()

    return min_scatter, selected_lc_length, bjd, best_mags

def lcfname_to_hatid(lcfname):
    """Return the HAT ID corresponding to the given LC filename."""

    with LightCurveFile(lcfname, 'r') as lightcurve:
        return dict(lightcurve['Identifiers'])[b'HAT']

def get_catalogue_data(catalogue_fname, lightcurve_fnames):
    """Return the catalogue information for the given lightcurves."""

    catalogue = numpy.genfromtxt(catalogue_fname,
                                 dtype=None,
                                 names=True,
                                 deletechars='')
    catalogue.dtype.names = [colname.split('[')[0] for colname in
                             catalogue.dtype.names]
    catalogue.sort()
    source_names = numpy.vectorize(lcfname_to_hatid)(lightcurve_fnames)
    return catalogue[
        numpy.searchsorted(catalogue['ID'], source_names)
    ]



def get_plot_x(catalogue_data, x_expression):
    """Return the x coordinate to use for each lightcurve."""

    x_evaluator = Interpreter()
    for varname in catalogue_data.dtype.names:
        x_evaluator.symtable[varname] = catalogue_data[varname]
    return x_evaluator(x_expression)

def main(cmdline_args):
    """Avoid polluting global namespace."""

    color_scheme = ['#e41a1c',
                    '#377eb8',
                    '#4daf4a',
                    '#984ea3',
                    '#ff7f00',
                    '#ffff33',
                    '#a65628',
                    '#f781bf']

    lc_fnames = list(find_lc_fnames(cmdline_args.lightcurves))

    catalogue_data = get_catalogue_data(cmdline_args.catalogue_fname,
                                        lc_fnames)
    plot_x = get_plot_x(catalogue_data, cmdline_args.x_expression)
    square_distances = catalogue_data['xi']**2 + catalogue_data['eta']**2
    distance_splits = list(cmdline_args.distance_splits) + [numpy.inf]
    color_index = 0

    for detrending_mode in cmdline_args.detrending_mode:
        scatter_data = numpy.vectorize(
            partial(
                get_minimum_scatter,
                **get_scatter_config(detrending_mode, cmdline_args)
            )
        )(
            lc_fnames
        )[0]
        unplotted_sources = numpy.ones(len(lc_fnames), dtype=bool)
        min_distance = 0
        for max_distance in sorted(distance_splits):
            #False positive
            #pylint: disable=assignment-from-no-return
            to_plot = numpy.logical_and(
                unplotted_sources,
                square_distances < max_distance**2
            )
            #pylint: enable=assignment-from-no-return
            if to_plot.any():
                plot_color = color_scheme[color_index % len(color_scheme)]
                color_index += 1
                pyplot.semilogy(
                    plot_x[to_plot],
                    scatter_data[to_plot],
                    '.',
                    markeredgecolor=plot_color,
                    markerfacecolor=plot_color,
                    markersize=cmdline_args.plot_marker_size,
                    label=(
                        r'%s $%s<\sqrt{\xi^2+\eta^2}<%s$'
                        %
                        (
                            detrending_mode.upper(),
                            min_distance,
                            max_distance
                        )
                    )
                )
                #False positive
                #pylint: disable=assignment-from-no-return
                unplotted_sources = numpy.logical_and(
                    unplotted_sources,
                    numpy.logical_not(to_plot)
                )
                #pylint: enable=assignment-from-no-return
            min_distance = max_distance

    pyplot.xlim(cmdline_args.plot_x_range)
    pyplot.ylim(cmdline_args.plot_y_range)
    pyplot.grid(True, which='both')
    pyplot.legend()
    if cmdline_args.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(cmdline_args.plot_fname)

if __name__ == '__main__':
    main(parse_command_line())
