#!/usr/bin/env python3

"""Extract statistics directly from the LCs and plot."""

from os import path
from functools import partial
from itertools import count
import logging
import argparse

from matplotlib import pyplot, patheffects
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from pytransit import QuadraticModel
import pandas
from colormath.color_objects import sRGBColor, HSLColor
from colormath.color_conversions import convert_color
import scipy.optimize as optimize
from scipy.optimize import LinearConstraint

from autowisp import LightCurveFile, Evaluator
from autowisp.catalog import read_catalog_file
from autowisp.file_utilities import find_lc_fnames
from autowisp.light_curves.apply_correction import\
    calculate_iterative_rejection_scatter
from autowisp.light_curves.reconstructive_correction_transit import\
    ReconstructiveCorrectionTransit
from autowisp.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from autowisp.processing_steps.lc_detrending import\
    get_transit_parameters


def try_converting(value_str):
    """Try converting the given value to int or float, leave str if not."""

    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str


def add_scatter_arguments(parser, multi_mode=True):
    """Add arguments to cmdline parser determining how scatter is calculated."""

    class ParseKwargs(argparse.Action):
        """Parse arguments in the form of key=value [key=value [...]]."""

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, {})
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = try_converting(value)

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
        '--lc-substitutions',
        default={'magfit_iteration': 0},
        action=ParseKwargs,
        nargs='*',
        help='Substitutions needed to resolve the datasets in the lightcurve to'
        ' plot. For example, which iteration of magnitude fitting to use.'
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
    parser.add_argument(
        '--combine-by',
        choices=['fnum', 'imageid'],
        default='imageid',
        help='If specified, more than one lightcurve of each object is expected'
        ' and the scatter is calculated for a LC calculated by matching the LCs'
        ' on frame number (not implemented yet) or image ID and averaging.'
    )
    parser.add_argument(
        '--min-combined',
        type=int,
        default=1,
        help='At least this many lightcurves must be found for a star for it '
        'to be included in the plot.'
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
        '--add-moving-average',
        nargs=2,
        metavar=('AVGFUNC', 'WINDOW'),
        default=None,
        help='Add a moving average line for each collection of points plotted.'
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
        'common_lc_substitutions': cmdline_args.lc_substitutions,
        'min_lc_length': cmdline_args.min_lc_length,
        'calculate_average': cmdline_args.average,
        'calculate_scatter': cmdline_args.statistic,
        'outlier_threshold': cmdline_args.outlier_threshold,
        'max_outlier_rejections': cmdline_args.max_outlier_rejections,
        'match_by': cmdline_args.combine_by,
    }


def default_get_magnitudes(lightcurve, dset_key, **substitutions):
    """Return the given dataset as both entries of a 2-tuple."""

    magnitudes = lightcurve.get_dataset(dset_key, **substitutions)
    return magnitudes, magnitudes


def match_lightcurves(lightcurve_filenames, min_combined):
    """Group lightcurves by source."""

    result = {}
    for lc_fname in lightcurve_filenames:
        with LightCurveFile(lc_fname, 'r') as lightcurve:
            #False positive
            #pylint:disable=no-member
            lc_id = int(dict(lightcurve['Identifiers'].asstr())['Gaia DR3'])
            #pylint:enable=no-member
            if lc_id not in result:
                result[lc_id] = []
            result[lc_id].append(lc_fname)

    for lc_id in list(result.keys()):
        if len(result[lc_id]) < min_combined:
            del result[lc_id]

    return result


def get_specified_photometry(lightcurve_filenames,
                             *,
                             aperture_index,
                             detrending_mode,
                             lc_substitutions,
                             stat_only=False,
                             **scatter_config):
    """Same as get_minimum_scatter, except return specified aperture.
    *** WARNING: SHOULD NOT BE USED!***
    IT IS WRONG IN ITS CURRENT STATE!
    it iterates through all the lightcurves, but when iterating,
    it just keeps overwriting magnitudes!
    """

    for lc_fname in lightcurve_filenames:
        with LightCurveFile(lc_fname, 'r') as lightcurve:
            bjd = lightcurve.get_dataset('skypos.BJD')

            if aperture_index >= 0:
                magnitudes = lightcurve.get_dataset(
                    'apphot.' + detrending_mode + '.magnitude',
                    aperture_index=aperture_index,
                    **lc_substitutions
                )
            else:
                magnitudes = lightcurve.get_dataset(
                    'shapefit.' + detrending_mode + '.magnitude',
                    **lc_substitutions
                )

    min_scatter, selected_lc_length = (
        calculate_iterative_rejection_scatter(
            magnitudes,
            **scatter_config
        )
    )

    if stat_only:
        return min_scatter, selected_lc_length
    return min_scatter, selected_lc_length, bjd, magnitudes


def get_lc_minimum_scatter(lc_fname,
                           detrending_mode,
                           lc_substitutions,
                           min_lc_length,
                           *,
                           stat_only=False,
                           get_magnitudes=default_get_magnitudes,
                           match_by='imageid',
                           **scatter_config):
    """
    Find the photometry with the smallest scatter in given LC.

    Args:
        lc_fname(str):    The filename of the lightcurve for which to find the
            smallest scatter.

        detrending_mode(str):    Which version of detrending to extract the
            scatter for. Should be one of `'magfit'`, `'epd'`, or `'tfa'`.

        lc_substitutions(dict):    Arguments that should be substituted in the
                dataset paths within the lighcurve (e.g. magfit_iteration.

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

    Returns:
        float:
            The smallest scatter found in the lightcurve.

        int:
            The number of non-rejected points when calculating the scatter.

        array(float):
            The BJD of each point in the lightcurve.

        array(float):
            The magnitude of each point in the lightcurve for the aperture or
            PSF with smallest scatter.
    """

    with LightCurveFile(lc_fname, 'r') as lightcurve:
        bjd = lightcurve.get_dataset('skypos.BJD')
        if match_by == 'fnum':
            image_ids = lightcurve.get_dataset('fitsheader.' + match_by)
        else:
            image_ids = [
                img_id.rsplit(b'_', 1)[1]
                for img_id in lightcurve.get_dataset('fitsheader.' + match_by)
            ]

        selected_lc_length = 0
        if (
            lightcurve.get_dataset(
                'shapefit.cfg.psf.bicubic.grid.x'
            ).shape[1]
            >
            2
        ):
            try:
                best_mags, scatter_mags = get_magnitudes(
                    lightcurve,
                    'shapefit.' + detrending_mode + '.magnitude',
                    **lc_substitutions
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
            best_mags = None

        try:
            for aperture_index in count():
                magnitudes, scatter_mags = get_magnitudes(
                    lightcurve,
                    'apphot.' + detrending_mode + '.magnitude',
                    aperture_index=aperture_index,
                    **lc_substitutions
                )

                scatter, lc_length = calculate_iterative_rejection_scatter(
                    scatter_mags,
                    **scatter_config
                )
                if lc_length > min_lc_length and scatter < min_scatter:
                    print(f'Found better aperture {aperture_index} for '
                          f'{detrending_mode}.')
                    min_scatter = scatter
                    selected_lc_length = lc_length
                    best_mags = magnitudes
        except OSError:
            lightcurve.close()

    if stat_only:
        return min_scatter, selected_lc_length
    return min_scatter, selected_lc_length, image_ids, bjd, best_mags


#pylint: disable=too-many-locals
def get_minimum_scatter(lightcurve_filenames,
                        detrending_mode,
                        common_lc_substitutions,
                        min_lc_length,
                        *,
                        stat_only=False,
                        get_magnitudes=default_get_magnitudes,
                        match_by='imageid',
                        **scatter_config):
    """
    Get the minimum combined scatter for each star.

    Args:
        lightcurve_filenames([str]):    The filenames of all the lightcurve for
            a single object which will be combined before finding the smallest
            scatter. Each LC filename can optionally followed by
            ``:<key>=value>[:<key>=value[:...]]`` to specify substitutions for '
            'dataset paths that differ between lightcurves.

        For all other arguments see get_lc_minimum_scatter().

    Returns:
        Same as get_lc_minimum_scatter().
    """

    def get_lc_substitutions(lc_fname):
        """Return combined (common + individual) substitution for given LC."""

        if ':' not in lc_fname:
            return common_lc_substitutions

        lc_substitutions = dict(common_lc_substitutions)
        for key_value in lc_fname.split(':')[1:]:
            key, value = key_value.split('=')
            lc_substitutions[key] = try_converting(value)
        return lc_substitutions


    def get_lc_data(lc_fname, lc_index):

        return pandas.DataFrame(
            dict(
                zip(
                    ('image_id', 'bjd', f'mag_{lc_index:03d}'),
                    get_lc_minimum_scatter(
                        lc_fname.split(':')[0],
                        detrending_mode=detrending_mode,
                        lc_substitutions=get_lc_substitutions(lc_fname),
                        min_lc_length=min_lc_length,
                        stat_only=False,
                        get_magnitudes=get_magnitudes,
                        match_by=match_by,
                        **scatter_config
                    )[-3:]
                )
            )
        ).set_index('image_id')


    def get_scatter_no_fail(array):
        """Return the scatter of the given array or NaN on any failure."""

        try:
            return calculate_iterative_rejection_scatter(
                array,
                **scatter_config
            )[0]
        except Exception:
            return numpy.nan

    num_lcs = len(lightcurve_filenames)
    print(f'Finding minimum scatter for {num_lcs} lightcurves: '
          f'{lightcurve_filenames!r}')
    if num_lcs > 1:
        matched = get_lc_data(lightcurve_filenames[0], 0)

        for lc_index, lc_fname in enumerate(lightcurve_filenames[1:]):
            matched = matched.join(
                get_lc_data(lc_fname, lc_index + 1).drop('bjd', axis=1),
                how='inner'
            )

        bjd = matched['bjd']
        matched = matched.drop('bjd', axis=1)

        ### We used to have:

        # avg_mag = matched.mean(axis=1)

        # avg_scatter, lc_length = calculate_iterative_rejection_scatter(
        #     avg_mag,
        #     **scatter_config
        # )

        ### Now, we use the following function for a weighted approach:

        def optimize_weighted_avg(coeffs):

            # Normalized coeff
            coeffs = coeffs / numpy.sum(coeffs)
            weighted_avg_mag = numpy.zeros(len(matched))

            for i, coeff in enumerate(coeffs):
                weighted_avg_mag += coeff * matched.iloc[:, i]

            weighted_scatter, _ = calculate_iterative_rejection_scatter(
                weighted_avg_mag,
                **scatter_config
            )

            return weighted_scatter

        bounds = [(0, 1) for _ in range(num_lcs)]
        linear_constraint = LinearConstraint(numpy.ones(num_lcs), lb=1, ub=1)
        initial_guess = numpy.ones(num_lcs) / num_lcs
        # Optimize to find best coefficients
        coeff_result = optimize.minimize(
            optimize_weighted_avg,
            initial_guess,
            bounds=bounds,
            constraints=linear_constraint
        )

        optimal_coeffs = coeff_result.x / numpy.sum(coeff_result.x)

        # Calculate weighted average light curve with optimal coefficients
        weighted_avg_mag = numpy.zeros(len(matched))
        for i, coeff in enumerate(optimal_coeffs):
            weighted_avg_mag += coeff * matched.iloc[:, i]

        # Calculate scatter and individual light curve scatters
        avg_scatter, lc_length = calculate_iterative_rejection_scatter(
            weighted_avg_mag,
            **scatter_config
        )

        scatter = (
            [avg_scatter]
            +
            [
                get_scatter_no_fail(matched[f'mag_{lc_index:03d}'])
                for lc_index in range(num_lcs)
            ]
        )

        if stat_only:
            return scatter, lc_length
        return scatter, lc_length, None, bjd, matched

    return get_lc_minimum_scatter(
        lightcurve_filenames[0].split(':')[0],
        detrending_mode=detrending_mode,
        lc_substitutions=get_lc_substitutions(lightcurve_filenames[0]),
        min_lc_length=min_lc_length,
        stat_only=stat_only,
        get_magnitudes=get_magnitudes,
        match_by=match_by,
        **scatter_config
    )
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


def get_scatter_data(lc_fnames, detrending_mode, cmdline_args):
    """Return the scatter of the given lightcurves, including the target."""

    get_scatter = partial(
        get_minimum_scatter,
        stat_only=True,
        detrending_mode=detrending_mode,
        **get_scatter_config(cmdline_args)
    )
    max_combined = max(map(len, lc_fnames.values()))

    result = pandas.DataFrame(
        index=lc_fnames.keys(),
        columns=(
            ['combined_scatter']
            +
            (
                [f'individual_scatter_{i:03d}' for i in range(max_combined)]
                if max_combined > 1 else
                []
            )
        )
    )
    for progress, (source_id, source_lcs) in enumerate(lc_fnames.items()):
        scatter = get_scatter(source_lcs)[0]
        if max_combined > 1:
            result.loc[source_id][:len(scatter)] = scatter
        else:
            result.loc[source_id] = scatter
        if progress % 100 == 0:
            print(f'Progress: {progress}/{len(lc_fnames)}')

    if cmdline_args.target_id is not None:
        transit_parameters = (get_transit_parameters(vars(cmdline_args), False),
                              {})
        result.loc[cmdline_args.target_id] = get_scatter(
            lc_fnames[cmdline_args.target_id],
            get_magnitudes=ReconstructiveCorrectionTransit(
                transit_model=QuadraticModel(),
                correction=None,
                fit_amplitude=False,
                transit_parameters=transit_parameters
            ).get_fit_data
        )[0]
    return result


def read_catalog(catalog_fname, x_expression):
    """Return the catalog data for the given sources."""

    catalog_data = read_catalog_file(catalog_fname,
                                     add_gnomonic_projection=True)
    catalog_data['magnitude'] = Evaluator(catalog_data)(x_expression)
    catalog_data.insert(
        len(catalog_data.columns),
        'square_distance',
        catalog_data['xi']**2 + catalog_data['eta']**2
    )
    return catalog_data


def get_x_label(x_expression):
    """Return the label that should be used for the x-axis."""

    if x_expression == 'phot_g_mean_mag':
        return 'Gaia G magnitude'
    return x_expression


class MovingAverage:
    """Calculates a moving iterative rejection average of given points."""

    def __init__(self,
                 x,
                 y,
                 window_size,
                 **scatter_config):

        """
        Set up the moving average with the given configuration.

        See calculate_iterative_rejection_scatter() for a description of the
        arguments.
        """

        self._x = x
        self._y = y
        self._half_window = window_size / 2.0
        self._scatter_config = scatter_config
        self._scatter_config['return_average'] = True


    def __call__(self, x):
        """Return the moving average around the given location."""

        average_points = numpy.logical_and(
            self._x >= x - self._half_window,
            self._x <= x + self._half_window,
        )
        return calculate_iterative_rejection_scatter(
            self._y[average_points],
            **self._scatter_config
        )[-1]


def add_plot_points(scatter_data,
                    to_plot,
                    cmdline_args,
                    *,
                    detrending_mode=None,
                    min_distance=0,
                    max_distance=numpy.inf):
    """Add one collection of points to the plot."""

    color_scheme = ['#e41a1c',
                    '#377eb8',
                    '#4daf4a',
                    '#984ea3',
                    '#ff7f00',
                    '#ffff33',
                    '#a65628',
                    '#f781bf']
    #color_scheme = ['#ff0000', '#0000ff']
    if not hasattr(add_plot_points, 'color_index'):
        add_plot_points.color_index = 0
    else:
        add_plot_points.color_index += 1


    plot_color = color_scheme[add_plot_points.color_index % len(color_scheme)]

    plot_config = {
        'markeredgecolor': 'none',
        'markerfacecolor': plot_color,
        'linestyle': 'none',
        'alpha': 0.8
    }

    if cmdline_args.target_id is not None:
        target_index = scatter_data.index[
            scatter_data.index == cmdline_args.target_id
        ].index[0]

    plot_target = cmdline_args.target_id is not None and to_plot[target_index]
    if plot_target:
        plot_config['label'] = cmdline_args.target_id
        pyplot.semilogy(
            scatter_data['magnitude'][target_index],
            scatter_data['scatter'][target_index],
            marker='*',
            markersize=(3 * cmdline_args.plot_marker_size),
            **plot_config
        )

    if min_distance == 0 and not numpy.isfinite(max_distance):
        distance_label = ''
    else:
        distance_label = (
            f' ${min_distance:s}'
            r'<\sqrt{\xi^2+\eta^2}<'
            f'{max_distance}$'
        )

    plot_config['marker'] = '.'
    plot_config['markersize'] = cmdline_args.plot_marker_size

    try:
        for lc_ind in count(-1):
            if lc_ind == -1:
                plot_config['label'] = (
                    'combined'
                    +
                    (f' {detrending_mode.upper()}' if detrending_mode else '')
                    +
                    distance_label
                )
            elif lc_ind == 0:
                if len(scatter_data.columns) == 1:
                    break
                plot_config['label'] = (
                    'individual'
                    +
                    plot_config['label'][len('combined'):]
                )
            else:
                break
#                plot_config['label'] = None

            plot_xy = (
                scatter_data['magnitude'].iloc[to_plot],
                scatter_data[
                    'combined_scatter'
                    if lc_ind == -1 else
                    f'individual_scatter_{lc_ind:03d}'
                ].iloc[to_plot]
            )
            print(f'Plotting {lc_ind} {plot_config["label"]}: ' + repr(plot_xy))
            pyplot.semilogy(
                *plot_xy,
                **plot_config,
                zorder=10 * (10 - lc_ind)
            )
            if cmdline_args.add_moving_average:
                print('Adding moving average')
                moving_average = MovingAverage(
                    *plot_xy,
                    window_size=float(cmdline_args.add_moving_average[1]),
                    calculate_average=getattr(
                        numpy,
                        'nan' + cmdline_args.add_moving_average[0]
                    ),
                    calculate_scatter=cmdline_args.statistic,
                    outlier_threshold=cmdline_args.outlier_threshold,
                    max_outlier_rejections=cmdline_args.max_outlier_rejections
                )
                color = convert_color(
                    sRGBColor.new_from_rgb_hex(plot_config['markerfacecolor']),
                    HSLColor
                )
                color.hsl_l += 0.3
                color = convert_color(color, sRGBColor)
                plot_x = numpy.linspace(numpy.min(plot_xy[0]),
                                        numpy.max(plot_xy[0]),
                                        100)
                plot_y = numpy.vectorize(moving_average)(plot_x)
                print(f'Plot x: {plot_x!r}, y: {plot_y!r}')
                pyplot.plot(
                    plot_x,
                    plot_y,
                    linewidth=5,
                    color=color.get_rgb_hex(),
                    zorder=120,
                    path_effects=[
                        patheffects.Stroke(linewidth=7, foreground='black'),
                        patheffects.Normal()
                    ]
                )


            if lc_ind == -1:
                add_plot_points.color_index += 1
                plot_color = color_scheme[
                    add_plot_points.color_index % len(color_scheme)
                ]
                plot_config['markerfacecolor'] =  plot_color
    except KeyError:
        pass
    if plot_target:
        to_plot[target_index] = True


#TODO: consider simplifying
#pylint: disable=too-many-locals
def main(cmdline_args):
    """Avoid polluting global namespace."""

    lc_fnames = None
    catalog_data = read_catalog(cmdline_args.catalog_fname,
                                cmdline_args.x_expression)

    distance_splits = list(cmdline_args.distance_splits) + [numpy.inf]

    for detrending_mode in cmdline_args.detrending_mode:
        if path.exists(detrending_mode + 'scatter.pkl'):
            scatter_data = pandas.read_pickle(detrending_mode + 'scatter.pkl')
        else:
            if lc_fnames is None:
                lc_fnames = match_lightcurves(
                    find_lc_fnames(cmdline_args.lightcurves),
                    cmdline_args.min_combined
                )

            scatter_data = get_scatter_data(lc_fnames,
                                            detrending_mode,
                                            cmdline_args)
            scatter_data = catalog_data.join(scatter_data, how='inner')
            scatter_data.to_pickle(detrending_mode + 'scatter.pkl')

        print('Scatter data: ' + repr(scatter_data))

        unplotted_sources = numpy.ones(scatter_data.shape[0], dtype=bool)
        min_distance = 0
        for max_distance in sorted(distance_splits):
            #False positive
            #pylint: disable=assignment-from-no-return
            to_plot = numpy.logical_and(
                unplotted_sources,
                (scatter_data['square_distance'] < max_distance**2).values
            )
            print('To plot: ' + repr(to_plot))
            #pylint: enable=assignment-from-no-return

            if to_plot.any():
                add_plot_points(
                    scatter_data,
                    to_plot,
                    cmdline_args,
                    detrending_mode=(
                        detrending_mode if len(cmdline_args.detrending_mode) > 1
                        else None
                    ),
                    min_distance=min_distance,
                    max_distance=max_distance)
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
    pyplot.xlabel(get_x_label(cmdline_args.x_expression))
    pyplot.ylabel('Median Abssolute Deviation')


    pyplot.grid(True, which='both')
    pyplot.legend(markerscale=3, framealpha=1.0)
    if cmdline_args.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(cmdline_args.plot_fname)
#pylint: enable=too-many-locals


if __name__ == '__main__':
    main(parse_command_line())
