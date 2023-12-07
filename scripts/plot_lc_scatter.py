#!/usr/bin/env python3

"""Extract statistics directly from the LCs and plot."""

from functools import partial
from itertools import count
from os import path
import logging

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from pytransit import QuadraticModel

from superphot_pipeline import LightCurveFile
from superphot_pipeline.file_utilities import find_lc_fnames
from superphot_pipeline.light_curves.apply_correction import\
    calculate_iterative_rejection_scatter
from superphot_pipeline.light_curves.reconstructive_correction_transit import\
    ReconstructiveCorrectionTransit
from superphot_pipeline.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from superphot_pipeline.processing_steps.lc_detrending import\
    get_transit_parameters
from superphot_pipeline.processing_steps.lc_detrending import add_catalog_info

def add_scatter_arguments(parser, multi_mode=True):
    """Add arguments to cmdline parser determining how scatter is calculated."""

    parser.add_argument(
        '--detrending-mode',
        **({'nargs': '+'} if multi_mode else {}),
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
        '--catalog-fname', '--catalog', '--cat',
        default='catalog.ucac4',
        help='The name of the catalog file containing all sources to '
        'include in the plot (may contain extra sources too).'
    )
    parser.add_argument(
        '--x-expression',
        default='phot_g_mean_mag',
        help='An expression involving catalog informatio to use as the x-axis'
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

    target_args = parser.add_argument_group(
        title='Followup Target',
        description='Arguments specific to processing followup '
        'observations where the target star is known to have a transit '
        'that occupies a significant fraction of the total collection '
        'of observations.'
    )
    target_args.add_argument(
        '--target-id',
        default=None,
        help='The lightcurve of the given source (any one of the '
        'catalog identifiers stored in the LC file) will be fit using'
        ' reconstructive detrending, starting the transit parameter fit'
        ' with the values supplied, and fitting for the values allowed '
        'to vary. If not specified all LCs are fit in '
        'non-reconstructive way.'
    )
    LCDetrendingArgumentParser.add_transit_parameters(
        parser,
        timing=True,
        geometry='circular',
        limb_darkening=True,
        fit_flags=False
    )
    parser.add_argument(
        '--verbose',
        default='warning',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='The type of verbosity of logger.'
    )

    result = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, result.verbose.upper()),
        format='%(levelname)s %(asctime)s %(name)s: %(message)s | '
               '%(pathname)s.%(funcName)s:%(lineno)d'
    )
    return result


def get_scatter_config(cmdline_args):
    """Return the configuration to use for extracting scatter from LC."""

    return {
        'magfit_iteration': cmdline_args.magfit_iteration,
        'min_lc_length': cmdline_args.min_lc_length,
        'calculate_average': cmdline_args.average,
        'calculate_scatter': cmdline_args.statistic,
        'outlier_threshold': cmdline_args.outlier_threshold,
        'max_outlier_rejections': cmdline_args.max_outlier_rejections
    }


def default_get_magnitudes(lightcurve, dset_key, **substitutions):
    """Return the given dataset as both entries of a 2-tuple."""

    magnitudes = lightcurve.get_dataset(dset_key, **substitutions)
    return magnitudes, magnitudes


#pylint: disable=too-many-locals
def get_minimum_scatter(lc_fname,
                        detrending_mode,
                        magfit_iteration,
                        min_lc_length,
                        *,
                        stat_only=False,
                        get_magnitudes=default_get_magnitudes,
                        **scatter_config):
    """
    Find the photometry with the smallest scatter and return that scatter.

    Args:
        lc_fname(str):    The filename of the lightcurve to find the smallest
            scatter for.

        detrending_mode(str):    Which version of detrending to extract the
            scatter for. Should be one of `'magfit'`, `'epd'`, or `'tfa'`.

        min_lc_length(int):    Only allow photometries which contain at least
            this many non-rejected points.

        stat_only(bool):    If true, returns only the scatter and number of
            non-rejected points. Otherwise also returns the magnritudes and BJD
            of the best photometry (no rejections applied).

        get_magnitudes:    Callable that returns the magnitudes from the LC
            given LightCurveFile, dataset key, and substitutions. Should either
            return a single 1-D array or two arrays, the second of which is used
            to determine the scatter, but the first is returned as the selected
            dataset. Allows using scatter around a variability model.

        scatter_config:    Arguments to pass to get_scatter() configuring
            how the scatter is to be calculated.
    """

    with LightCurveFile(lc_fname, 'r') as lightcurve:
        bjd = lightcurve.get_dataset('skypos.BJD')

        selected_lc_length = 0
        if (
            lightcurve.get_dataset('shapefit.cfg.psf.bicubic.grid.x').shape[1]
            >
            2
        ):
            try:
                best_mags, scatter_mags = get_magnitudes(
                    lightcurve,
                    'shapefit.' + detrending_mode + '.magnitude',
                    magfit_iteration=magfit_iteration
                )

                min_scatter, selected_lc_length = (
                    calculate_iterative_rejection_scatter(
                        scatter_mags,
                        **scatter_config
                    )
                )
            except OSError:
                pass

        if selected_lc_length < min_lc_length:
            min_scatter = numpy.inf

        try:
            for aperture_index in count():
                magnitudes, scatter_mags = get_magnitudes(
                    lightcurve,
                    'apphot.' + detrending_mode + '.magnitude',
                    aperture_index=aperture_index,
                    magfit_iteration=magfit_iteration
                )

                scatter, lc_length = calculate_iterative_rejection_scatter(
                    scatter_mags,
                    **scatter_config
                )
                if lc_length > min_lc_length and scatter < min_scatter:
                    min_scatter = scatter
                    selected_lc_length = lc_length
                    best_mags = magnitudes
        except OSError:
            lightcurve.close()

    if stat_only:
        return min_scatter, selected_lc_length
    return min_scatter, selected_lc_length, bjd, best_mags
#pylint: enable=too-many-locals


def lcfname_to_hatid(lcfname):
    """Return the HAT ID corresponding to the given LC filename."""

    with LightCurveFile(lcfname, 'r') as lightcurve:
        return dict(lightcurve['Identifiers'])[b'HAT']


def get_target_index(lc_fnames, target_id):
    """Return the index within the given lightcurve names of the target."""

    if target_id is None:
        return None
    for index, fname in enumerate(lc_fnames):
        with LightCurveFile(fname, 'r') as lightcurve:
            if (
                    target_id.encode('ascii')
                    in
                    lightcurve['Identifiers'][:, 1]
            ):
                return index
    raise RuntimeError('None of the lightcurves matches the target ID '
                       +
                       repr(target_id.decode()))


def get_scatter_data(lc_fnames, detrending_mode, target_index, cmdline_args):
    """Return the scatter of the given lightcurves, including the target."""

    get_scatter = partial(
        get_minimum_scatter,
        stat_only=True,
        detrending_mode=detrending_mode,
        **get_scatter_config(cmdline_args)
    )
    result = numpy.vectorize(get_scatter)(lc_fnames)[0]
    if target_index is not None:
        transit_parameters = (get_transit_parameters(vars(cmdline_args), False),
                              {})
        result[target_index] = get_scatter(
            lc_fnames[target_index],
            get_magnitudes=ReconstructiveCorrectionTransit(
                transit_model=QuadraticModel(),
                correction=None,
                fit_amplitude=False,
                transit_parameters=transit_parameters
            ).get_fit_data
        )[0]
    return result


#TODO: consider simplifying
#pylint: disable=too-many-locals
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

    target_index = get_target_index(lc_fnames, cmdline_args.target_id)

    catalog_data = add_catalog_info(lc_fnames,
                                    cmdline_args.catalog_fname,
                                    cmdline_args.x_expression)
    square_distances = catalog_data['xi']**2 + catalog_data['eta']**2
    distance_splits = list(cmdline_args.distance_splits) + [numpy.inf]
    color_index = 0

    for detrending_mode in cmdline_args.detrending_mode:
        scatter_data = get_scatter_data(lc_fnames,
                                        detrending_mode,
                                        target_index,
                                        cmdline_args)
        print(f'{"Source":25s} {"Scatter":25s}')
        for fname, scatter in zip(lc_fnames, scatter_data):
            print(f'{path.basename(fname):25s} {scatter:25.16g}')
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

                plot_config = {
                    'markeredgecolor': plot_color,
                    'markerfacecolor': plot_color,
                    'linestyle': 'none'
                }

                if target_index is not None and to_plot[target_index]:
                    to_plot[target_index] = False
                    unplotted_sources[target_index] = False

                    pyplot.semilogy(
                        catalog_data['mag'][target_index],
                        scatter_data[target_index],
                        marker='*',
                        markersize=(3 * cmdline_args.plot_marker_size),
                        label=cmdline_args.target_id,
                        **plot_config
                    )

                if min_distance == 0 and not numpy.isfinite(max_distance):
                    distance_label = ''
                else:
                    distance_label = rf' ${min_distance:s}<\sqrt{{\xi^2+\eta^2}}<{max_distance}$'

                pyplot.semilogy(
                    catalog_data['mag'][to_plot],
                    scatter_data[to_plot],
                    marker='.',
                    markersize=cmdline_args.plot_marker_size,
                    label=(
                        f'{detrending_mode.upper()}'
                        +
                        distance_label
                    ),
                    **plot_config
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
    pyplot.xlabel(cmdline_args.x_expression)
    pyplot.ylabel('MAD')


    pyplot.grid(True, which='both')
    pyplot.legend()
    if cmdline_args.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(cmdline_args.plot_fname)
#pylint: enable=too-many-locals


if __name__ == '__main__':
    main(parse_command_line())
