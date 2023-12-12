#!/usr/bin/env python3

"""Plot brightness measurements vs time for a star."""

from functools import partial

from matplotlib import pyplot
import numpy
from scipy import stats
from configargparse import ArgumentParser, DefaultsFormatter
from pytransit import QuadraticModel

from superphot_pipeline.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from superphot_pipeline.light_curves.reconstructive_correction_transit import\
    ReconstructiveCorrectionTransit
from superphot_pipeline.processing_steps.lc_detrending import\
    get_transit_parameters

from plot_lc_scatter import\
    get_minimum_scatter,\
    get_specified_photometry,\
    add_scatter_arguments,\
    add_plot_config_arguments,\
    get_scatter_config

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
        help='The filenames of the lightcurves to plot. If exactly 4 LCs are '
        'specified and only one binning mode, the assumption is that they are '
        'red, green, green, and blue LC of the same source, and hence plotted '
        'with corresponding colors (light and dark greed for the two green '
        'chanels).'
    )
    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )
    add_scatter_arguments(parser)
    add_plot_config_arguments(parser)

    parser.add_argument(
        '--aperture-index', '-a',
        type=int,
        default=None,
        help='If specified, plots the lightcurve using the given aperture (-1 '
        'for PSF fitting). If not specified, the photometry with smallest '
        'scatter is plotted.'
    )
    parser.add_argument(
        '--fold-period',
        type=float,
        default=None,
        help='If passed, the lightcurve is plotted vs phase defined by the '
        'specified period in days. If not specified, the lightcurve is not '
        'folded (plotted against days since first point).'
    )
    parser.add_argument(
        '--binning',
        default=None,
        type=float,
        help='Bin the lightcurve in bins of the given size in days. Note that '
        'for folded plots, the bin size is first converted to fraction of the '
        'folding period, and then binning is done on the folded LC.'
    )
    parser.add_argument(
        '--binning-errorbars',
        default='quantiles',
        choices=('quantiles', 'stddev'),
        help='How to determine error bars of binned points. If `"quantiles"`, '
        'the error bars span the 16-th to 84-th percentile of the points in '
        'each bin. If `"stddev`, error bars are symmetric around the median '
        'and have a length of the RMS of the points in the bin around the '
        'median, divided by sqrt(N-1).'
    )
    parser.add_argument(
        '--binned-continuous',
        action='store_true',
        help='If specified, instead of the binned LC showing independent '
        'binned points with error bars, it is used to smooth the LC.'
    )
    parser.add_argument(
        '--zero-stat',
        default=None,
        choices=('mean', 'median'),
        help='If specified, the plotted LCs are shufted by the negative of the '
        'given statistic in order to center the flux around zero.'
    )
    parser.add_argument(
        '--combined-binned-lc',
        action='store_true',
        help='If passed, instead of plotting all input LCs separately, they '
        'are assumed bo te be of the same object and binned together per '
        '`--binning`.'
    )
    parser.add_argument(
        '--variability',
        default=None,
        choices=['transit'],
        help='If specified, the star is assumed to have the given type of '
        'variability (for now only transits are supported).'
    )
    parser.add_argument(
        '--skip-gaps',
        default=None,
        type=float,
        help='If specified, the x axis (time) is broken to skip periods of '
        'time longer than the specified value (in days) when there are no '
        'observations. Ignored if multiple LCs are being plotted.'
    )
    parser.add_argument(
        '--drop-outliers',
        default=numpy.inf,
        type=float,
        help='Outliers larger than this times the scatter are not plotted.'
    )
    LCDetrendingArgumentParser.add_transit_parameters(
        parser,
        timing=True,
        geometry='circular',
        limb_darkening=True,
        fit_flags=False
    )

    return parser.parse_args()


#TODO: simplify
#pylint: disable=too-many-locals
def plot_binned(plot_x,
                magnitudes,
                bin_size,
                errorbar_mode,
                *,
                plot_axes=pyplot.gca(),
                lc_color,
                continuous=False,
                **plot_config):
    """Plot binned light curve."""

    if continuous:
        num_bins=continuous
        bin_step = (plot_x.max() - plot_x.min()) / num_bins
    else:
        bin_boundaries = numpy.arange(plot_x.min(), plot_x.max(), bin_size)
        bin_destinations = numpy.searchsorted(bin_boundaries, plot_x) - 1
        num_bins = bin_destinations.max() + 1
    binned_x = numpy.empty(num_bins, dtype=float)
    binned_magnitudes = numpy.empty(num_bins, dtype=float)
    binned_errorbars = numpy.empty(
        (2, num_bins) if errorbar_mode == 'quantiles' else (num_bins,),
        dtype=float
    )
    for bin_number in range(0, num_bins):
        if continuous:
            in_bin = numpy.abs(
                plot_x
                -
                plot_x.min()
                -
                bin_size/2
                -
                bin_number * bin_step
            ) < bin_size / 2
        else:
            in_bin = bin_destinations == bin_number
        binned_x[bin_number] = numpy.median(plot_x[in_bin])
        if errorbar_mode == 'quantiles':
            (
                binned_magnitudes[bin_number],
                binned_errorbars[0, bin_number],
                binned_errorbars[1, bin_number]
            ) = numpy.quantile(
                magnitudes[in_bin],
                [0.5, stats.norm.cdf(-1.0), stats.norm.cdf(1.0)]
            )
        else:
            binned_magnitudes[bin_number] = numpy.median(magnitudes[in_bin])
            binned_errorbars[bin_number] =  (
                numpy.sqrt(
                    numpy.mean(
                        numpy.square(
                            magnitudes[in_bin] - binned_magnitudes[bin_number]
                        )
                    )
                    /
                    (in_bin.sum() - 1)
                )
            )

    if continuous:
        if errorbar_mode == 'stddev':
            binned_errorbars = [
                binned_magnitudes - binned_errorbars,
                binned_magnitudes + binned_errorbars
            ]
        plot_config['markersize'] = 0
        plot_axes.plot(
            binned_x,
            binned_magnitudes,
            '-',
            color=lc_color,
            **plot_config
        )
        plot_config['label'] = None
        for i in range(2):
            plot_axes.plot(
                binned_x,
                binned_errorbars[i],
                '--',
                color=lc_color,
                **plot_config
            )
    else:
        if errorbar_mode == 'quantiles':
            binned_errorbars[0] = binned_magnitudes - binned_errorbars[0]
            binned_errorbars[1] -= binned_magnitudes

        plot_axes.errorbar(
            binned_x,
            binned_magnitudes,
            binned_errorbars,
            fmt='.',
            ecolor=lc_color,
            markeredgecolor='black',
            markerfacecolor=lc_color,
            **plot_config
        )
#pylint: enable=too-many-locals


def plot_transit_model(transit_flux,
                       eval_bjd,
                       oot_mag=0.0,
                       folding=None,
                       *,
                       plot_axes=pyplot.gca(),
                       **plot_config):
    """
    Add a box transit to the plot.

    Args:
        transit_flux(array):    The flux of the transit model to plot normalized
            to `1` out of transit.

        eval_bjd(array):    The BJD values at which to evaluate the transit.
            Typically something like
            ``linspace(first LC point, last LC point, 1000)``.

        oot_mag(float):    The out of transit magnitude to use.

        folding(float or None):    Period at which the LC is folded for
            plotting.

        plot_config:    Any extra arguments to pass to `pyplot.plot`.

    Returns:
        None
    """

    assert transit_flux.shape == eval_bjd.shape
    model_mag = oot_mag - 2.5 * numpy.log10(transit_flux)
    plot_axes.plot(
        (eval_bjd % folding) if folding else eval_bjd,
        model_mag,
        **plot_config
    )


def mark_transit(transit_flux,
                 eval_bjd,
                 oot_mag=0.0,
                 *,
                 plot_axes=pyplot.gca(),
                 label='',
                 **plot_config):
    """Mark in and out of transit mags + start and for non-folded LCs only."""

    def get_ages_to_mark():
        """Return a list of the ages to mark in the plot."""

        oot_flag = (transit_flux == 1.0)
        result = []
        for first, second in [(1, 0), (0, 1)]:
            select = numpy.logical_and(oot_flag[1:] == first,
                                       oot_flag[:-1] == second)
            result.append(0.5 * (eval_bjd[:-1][select]
                          +
                          eval_bjd[1:][select]))
        return result, oot_mag - 2.5 * numpy.log10(transit_flux.min())


    ages_to_mark, max_mag = get_ages_to_mark()

    color = plot_axes.axhline(y=oot_mag).get_color()
    plot_axes.axhline(y=max_mag, color=color)

    if label.strip():
        label = label.strip() + ' '
    else:
        label = ''

    style = '--'
    full_label = label + 'start'
    for mark_ages in ages_to_mark:
        for x in mark_ages:
            plot_axes.axvline(
                x=x,
                color=color,
                linestyle=style,
                label=full_label,
                **plot_config
            )
            full_label = None
        full_label = label  + 'end'
        style=':'


def get_skip_gap_axes(plot_x, min_gap, pad_frac=0.01):
    """Return list of axes skipping gaps in plot_x that exceed min_gap."""

    gap_indices = numpy.nonzero(plot_x[1:] - plot_x[:-1] > min_gap)[0]
    ax_limits = []
    min_x = plot_x[0]
    for gap_i in gap_indices:
        pad = (plot_x[gap_i] - min_x) * pad_frac * gap_indices.size
        ax_limits.append((min_x - pad, plot_x[gap_i] + pad))
        min_x = plot_x[gap_i+1]
    pad = (plot_x[-1] - min_x) * pad_frac * gap_indices.size
    ax_limits.append((min_x - pad_frac, plot_x[-1] + pad_frac))

    fig, axes = pyplot.subplots(
        1,
        gap_indices.size + 1,
        sharey=True,
        gridspec_kw=dict(width_ratios=[l[1] - l[0] for l in ax_limits])
    )

    min_x = plot_x[0]
    for gap_i, ax_part in zip(gap_indices, axes):
        ax_part.spines['right'].set_visible(False)
        ax_part.spines['left'].set_visible(False)
        ax_part.tick_params(labelleft=False)
        ax_part.tick_params(left=False)
        ax_part.tick_params(right=False)

        pad = (plot_x[gap_i] - min_x) * pad_frac * gap_indices.size
        ax_part.set_xlim(min_x - pad, plot_x[gap_i] + pad)
        min_x = plot_x[gap_i+1]

    axes[0].spines['left'].set_visible(True)
    axes[0].tick_params(labelleft=True)
    axes[0].tick_params(left=True)
#    axes[0].set_yticks([-0.02, -0.01, 0.0, 0.01, 0.02])
    axes[-1].spines['left'].set_visible(False)
    axes[-1].tick_params(left=False)

    pad = (plot_x[-1] - min_x) * pad_frac * gap_indices.size
    axes[-1].set_xlim(min_x - pad_frac, plot_x[-1] + pad_frac)

    first = True

    cutout_size=0.015
    for ax_part in axes:
        ax_limits = ax_part.get_xlim()
        tick_start = 0.1 * numpy.ceil(10.0 * ax_limits[0])
        tick_end = 0.1 * numpy.floor(10.0 * ax_limits[1])
        print('Tick range: ' + repr((tick_start, tick_end)))
        if gap_indices.size > 3:
            ax_part.set_xticks([tick_start, tick_end])
#            ax_part.tick_params(axis='x', which='both', labelrotation=45)
        else:
            ax_part.set_xticks(
                numpy.arange(tick_start, tick_end + 0.1, 0.1)
            )

        cutout_x_coords = numpy.array([
            1 + cutout_size / (ax_limits[1] - ax_limits[0]),
            1 - cutout_size / (ax_limits[1] - ax_limits[0])
        ])
        ax_part.plot(
            cutout_x_coords,
            (-cutout_size, +cutout_size),
            transform=ax_part.transAxes,
            color='k',
            clip_on=False
        )
        ax_part.plot(
            cutout_x_coords,
            (1 - cutout_size, 1 + cutout_size),
            transform=ax_part.transAxes,
            color='k',
            clip_on=False
        )

        if not first:
            ax_part.plot(
                cutout_x_coords - 1,
                (-cutout_size, +cutout_size),
                transform=ax_part.transAxes,
                color='k',
                clip_on=False
            )
            ax_part.plot(
                cutout_x_coords - 1,
                (1 - cutout_size, 1 + cutout_size),
                transform=ax_part.transAxes,
                color='k',
                clip_on=False
            )


        first = False

    return fig, axes

def plot_lc(plot_x,
            magnitudes,
            *,
            plot_axes=pyplot.gca(),
            detrending_mode,
            lc_color,
            zorder,
            configuration):
    """Plot a single LC with the given color and configuration."""

    if configuration.fold_period:
        plot_x %= configuration.fold_period

    plot_axes.plot(
        plot_x,
        magnitudes,
        '.',
        markersize=(
            configuration.plot_marker_size
            /
            (5 if configuration.binning else 1)
        ),
        zorder=(zorder + 1),
        label=detrending_mode,
        markeredgecolor=lc_color if lc_color != 'black' else 'grey',
        markerfacecolor=lc_color if lc_color != 'black' else 'grey'
    )

    if configuration.binning:
        plot_binned(
            plot_x,
            magnitudes,
            configuration.binning,
            configuration.binning_errorbars,
            continuous=(1000 if configuration.binned_continuous else False),
            markersize=configuration.plot_marker_size,
            label='binned ' + detrending_mode,
            zorder=(zorder + 1 + len(configuration.detrending_mode)),
            lc_color=lc_color
        )


#pylint: disable=too-many-arguments
def add_transit_to_plot(transit_param,
                        transit_model,
                        bjd,
                        bjd_offset,
                        oot_mag,
                        configuration,
                        plot_axes=pyplot.gca()):
    """Plot the box transit of transit marks per configuration."""

    eval_transit_bjd = numpy.linspace(bjd.min(),
                                      bjd.max(),
                                      1000)
    transit_model.set_data(eval_transit_bjd + bjd_offset)
    transit_flux = transit_model.evaluate(*transit_param[0], **transit_param[1])

    plot_config = dict(
        zorder = (2 * len(configuration.detrending_mode) + 1),
        label='T'
    )

    if configuration.fold_period:
        plot_transit_model(
            transit_flux,
            eval_transit_bjd,
            oot_mag,
            configuration.fold_period,
            linestyle='-',
            plot_axes=plot_axes,
            **plot_config
        )
    else:
        mark_transit(transit_flux,
                     eval_transit_bjd,
                     oot_mag,
                     plot_axes=plot_axes,
                     **plot_config)
#pylint: enable=too-many-arguments


def add_lc_to_plot(select_photometry, configuration):
    """
    Add the lightcurve data to the plot.

    Args:
        select_photometry(callable):    Given a lightcurve filename and
            detrending mode keyword argument, return the dataset to plot. Should
            have the same return value as plot_lc_scatter.get_minimum_scatter().

        configuration:    The parsed command line configuration.

    Returns:
        array:    The times at which lightcurve entries are available, measured
            from the last integer BJD before the first observation. These are
            the x coordinates of the points added to the plot.

        int:    The last integer BJD before the first observation (i.e. the
            offset applied to the lightcurve BJDs for plotting.

        array:    The magnitudes from the last detrending method being plotted
            (presumably the best).

        float:    The iterative rejection scatter in the last plotted
            magnitudes.
    """

    if (
            len(configuration.lightcurves) == 4
            and
            len(configuration.detrending_mode) == 1
    ):
        colors = ['red', 'green', 'lime', 'blue']
    else:
        colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

    if configuration.combined_binned_lc:
        combined_magnitudes = numpy.array([], dtype=float)
        combined_bjd = numpy.array([], dtype=float)
        if (
            len(configuration.detrending_mode) == 1
            and
            len(configuration.lightcurves) == 1
        ):
            colors = ['black']

        assert (
            len(colors)
            >=
            len(configuration.detrending_mode)
        )

    else:
        assert (
            len(colors)
            >=
            len(configuration.lightcurves) * len(configuration.detrending_mode)
        )
    colors = iter(colors)

    skip_gap_axes, figure = [pyplot.gca()], pyplot.gcf()
    bjd_offset = None
    for zorder, detrending_mode in enumerate(
            reversed(configuration.detrending_mode)
    ):
        for lc_collection in (
            [configuration.lightcurves]
            if (
                configuration.combined_binned_lc
                and
                configuration.binning is None
            ) else
            [[lc_fname] for lc_fname in configuration.lightcurves]
        ):
            #False positive
            #pylint: disable=unbalanced-tuple-unpacking
            scatter, _, _, bjd, magnitudes = select_photometry(
                lc_collection,
                detrending_mode=detrending_mode,
            )
            #pylint: enable=unbalanced-tuple-unpacking

            if configuration.zero_stat is not None:
                for colname in magnitudes.columns:
                    magnitudes[colname] -= getattr(
                        numpy, 'nan' + configuration.zero_stat
                    )(
                        magnitudes[colname]
                    )

            if len(lc_collection) == 1:
                keep_points = (numpy.abs(magnitudes)
                               <
                               configuration.drop_outliers * scatter)
                bjd = bjd.iloc[keep_points].to_numpy(dtype=float)
                magnitudes = magnitudes.iloc[keep_points]
            else:
                bjd = bjd.to_numpy(dtype=float)

            if len(lc_collection) == 1:
                magnitudes = magnitudes.to_numpy(dtype=float)

                if configuration.combined_binned_lc:
                    combined_magnitudes = numpy.concatenate((
                        combined_magnitudes,
                        magnitudes
                    ))
                    combined_bjd = numpy.concatenate((
                        combined_bjd,
                        bjd
                    ))
            bjd_offset = bjd_offset or int(bjd.min())
            bjd -= bjd_offset
            if (
                (
                    len(configuration.lightcurves) == 1
                    or
                    (
                        configuration.combined_binned_lc
                        and
                        configuration.binning is None
                    )
                )
                and
                configuration.skip_gaps
            ):
                figure, skip_gap_axes = get_skip_gap_axes(
                    bjd,
                    configuration.skip_gaps
                )
                print('Skip gap axes generated!')
            else:
                print('Skip gap axes not generated for %d LCs'
                      %
                      len(configuration.lightcurves))


            if (
                not configuration.combined_binned_lc
                or
                configuration.binning is None
            ):
                print(f'Plotting individual LCs: {magnitudes!r}')
                lc_color = next(colors)
                for plot_axes in skip_gap_axes:
                    plot_lc(bjd,
                            magnitudes,
                            detrending_mode=detrending_mode,
                            lc_color=lc_color,
                            zorder=zorder,
                            plot_axes=plot_axes,
                            configuration=configuration)

        if configuration.combined_binned_lc:
            if configuration.binning is None:
                magnitudes = magnitudes.mean(axis=1).to_numpy(dtype=float)
                combined_bjd = bjd
            else:
                magnitudes = combined_magnitudes

            bjd_offset = int(combined_bjd.min())
            bjd = combined_bjd - bjd_offset
            lc_color = next(colors)

            for plot_axes in (skip_gap_axes if configuration.skip_gaps
                              else [pyplot.gca()]):
                print('Plotting combined')
                plot_lc(bjd,
                        magnitudes,
                        detrending_mode=detrending_mode,
                        plot_axes=plot_axes,
                        lc_color=lc_color,
                        zorder=zorder,
                        configuration=configuration)

    return (bjd, bjd_offset, magnitudes, scatter, skip_gap_axes, figure)


def main(configuration):
    """Avoid polluting global namespace."""

    if configuration.fold_period and configuration.binning:
        configuration.binning /= configuration.fold_period

    scatter_config = get_scatter_config(configuration)
    print('Scatter config: %s' % repr(scatter_config))
    if configuration.aperture_index is not None:
        del scatter_config['min_lc_length']
        select_photometry = partial(
            get_specified_photometry,
            aperture_index=configuration.aperture_index,
            **scatter_config
        )
    elif configuration.variability == 'transit':
        transit_parameters = (
            get_transit_parameters(vars(configuration), False),
            {}
        )
        transit_model=QuadraticModel()
        select_photometry = partial(
            get_minimum_scatter,
            get_magnitudes=ReconstructiveCorrectionTransit(
                transit_model=transit_model,
                correction=None,
                fit_amplitude=False,
                transit_parameters=transit_parameters
            ).get_fit_data,
            **scatter_config
        )
    else:
        select_photometry = partial(get_minimum_scatter, **scatter_config)

    (
        bjd,
        bjd_offset,
        last_magnitudes,
        _scatter,
        skip_gap_axes,
        figure
    ) = add_lc_to_plot(
        select_photometry,
        configuration
    )

    if configuration.variability == 'transit':
        transit_model.set_data(bjd + bjd_offset)
        oot_flag = (
            transit_model.evaluate(*transit_parameters[0],
                                   **transit_parameters[1])
            ==
            1.0
        )
        oot_magnitude = numpy.median(last_magnitudes[oot_flag])
        for plot_axes in skip_gap_axes:
            add_transit_to_plot(transit_parameters,
                                transit_model,
                                bjd,
                                bjd_offset,
                                oot_magnitude,
                                configuration,
                                plot_axes=plot_axes)
    else:
        assert configuration.variability is None
        oot_magnitude = numpy.median(last_magnitudes)

    figure.add_subplot(111, frameon=False)
    pyplot.tick_params(labelcolor='none',
                       which='both',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)
    pyplot.ylim(configuration.plot_y_range)
    for plot_ax in skip_gap_axes:
        plot_ax.set_ylim(configuration.plot_y_range)

    if configuration.fold_period:
        pyplot.xlabel(
            'Phase (P={:.6f} d)'.format(configuration.fold_period)
        )
    else:
        pyplot.xlabel('Days')
#        pyplot.xlabel('BJD - {:d}'.format(int(bjd[0])))

#    pyplot.axhspan(ymin=oot_magnitude - scatter,
#                   ymax=oot_magnitude + scatter,
#                   color='lightgrey',
#                   zorder=0)

    skip_gap_axes[0].set_ylabel('Magnitude')
    if not configuration.skip_gaps:
        pyplot.xlim(configuration.plot_x_range)
#    pyplot.ylim()
    pyplot.legend()

    if configuration.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(configuration.plot_fname)


if __name__ == '__main__':
    main(parse_command_line())
