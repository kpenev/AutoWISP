#!/usr/bin/env python3

"""Plot brightness measurements vs time for a star."""

from matplotlib import pyplot
import numpy
from scipy import stats
from configargparse import ArgumentParser, DefaultsFormatter

from plot_lc_scatter import\
    get_minimum_scatter,\
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
        '--add-transit',
        type=float,
        nargs=4,
        action='append',
        metavar=('EPOCH', 'PERIOD', 'DURATION', 'DEPTH'),
        default=[],
        help='Add a curve showing a box transit. Time units are BJD. Can be '
        'specified multilpe times to plot multiple curves.'
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

    return parser.parse_args()


def plot_binned(plot_x,
                magnitudes,
                bin_size,
                errorbar_mode,
                *,
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
        pyplot.plot(
            binned_x,
            binned_magnitudes,
            '-',
            color=lc_color,
            **plot_config
        )
        plot_config['label'] = None
        for i in range(2):
            pyplot.plot(
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

        pyplot.errorbar(
            binned_x,
            binned_magnitudes,
            binned_errorbars,
            fmt='.',
            ecolor=lc_color,
            markeredgecolor='black',
            markerfacecolor=lc_color
            **plot_config
        )


def plot_transit_model(transit_param,
                       eval_bjd,
                       oot_mag=0.0,
                       folding=None,
                       **plot_config):
    """
    Add a box transit to the plot.

    Args:
        transit_param(4 floats):    The epoch (BJD), period (days), duration
            (days), and depth (mag) of the transit.

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

    in_transit = (
        (eval_bjd - transit_param[0] + transit_param[2]/2.0) % transit_param[1]
        <
        transit_param[2]
    )
    model_mag = numpy.full(eval_bjd.shape, oot_mag)
    model_mag[in_transit] -= transit_param[3]
    pyplot.plot(
        (eval_bjd % folding) if folding else eval_bjd,
        model_mag,
        **plot_config
    )


def mark_transit(transit_param, oot_mag=0.0, *, label='', **plot_config):
    """Mark in and out of transit mags + start and for non-folded LCs only."""

    color = pyplot.axhline(y=oot_mag).get_color()
    pyplot.axhline(y=oot_mag-transit_param[3], color=color)

    bjd_range = numpy.array(pyplot.xlim())
    start_nperiod_range = numpy.ceil(
        (bjd_range - transit_param[0] + transit_param[2] / 2.0)
        /
        transit_param[1]
    ).astype(int)
    end_nperiod_range = numpy.ceil(
        (bjd_range - transit_param[0] - transit_param[2] / 2.0)
        /
        transit_param[1]
    ).astype(int)

    if label.strip():
        label = label.strip() + ' '
    else:
        label = label.strip()

    for nperiod in range(*start_nperiod_range):
        pyplot.axvline(
            x=(transit_param[0] - transit_param[2] / 2.0
               +
               nperiod * transit_param[1]),
            color=color,
            linestyle='--',
            label=(label + 'start' if nperiod == start_nperiod_range[0]
                   else None),
            **plot_config
        )

    for nperiod in range(*end_nperiod_range):
        pyplot.axvline(
            x=(transit_param[0] + transit_param[2] / 2.0
               +
               nperiod * transit_param[1]),
            color=color,
            linestyle=':',
            label=(label + 'end' if nperiod == end_nperiod_range[0]
                   else None),
            **plot_config
        )


def plot_lc(plot_x,
            magnitudes,
            *,
            detrending_mode,
            lc_color,
            zorder,
            configuration):
    """Plot a single LC with the given color and configuration."""

    if configuration.fold_period:
        plot_x %= configuration.fold_period

    pyplot.plot(
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
            continuous=(100 if configuration.binned_continuous else 1000),
            markersize=configuration.plot_marker_size,
            label='binned ' + detrending_mode,
            zorder=(zorder + 1 + len(configuration.detrending_mode)),
            lc_color=lc_color
        )


def add_transit_to_plot(bjd, bjd_offset, oot_mag, configuration):
    """Plot the box transit of transit marks per configuration."""

    eval_transit_bjd = numpy.linspace(bjd.min(), bjd.max(), 1000)
    for transit_ind, transit_param in enumerate(configuration.add_transit):
        transit_param[0] -= bjd_offset
        plot_config = dict(
            zorder = (2 * len(configuration.detrending_mode) + 1 + transit_ind),
            label='T{:d}'.format(transit_ind)
        )

        if configuration.fold_period:
            plot_transit_model(
                transit_param,
                eval_transit_bjd,
                oot_mag,
                configuration.fold_period,
                linestyle='-',
                **plot_config
            )
        else:
            mark_transit(
                transit_param,
                oot_mag,
                **plot_config
            )


def main(configuration):
    """Avoid polluting global namespace."""

    if configuration.fold_period and configuration.binning:
        configuration.binnning /= configuration.fold_period
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
        if len(configuration.detrending_mode) == 1:
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


    for zorder, detrending_mode in enumerate(
            reversed(configuration.detrending_mode)
    ):
        for lc_fname in configuration.lightcurves:
            #False positive
            #pylint: disable=unbalanced-tuple-unpacking
            scatter, _, bjd, magnitudes = get_minimum_scatter(
                lc_fname,
                **get_scatter_config(detrending_mode, configuration)
            )
            #pylint: enable=unbalanced-tuple-unpacking

            if configuration.zero_stat is not None:
                magnitudes -= getattr(numpy, 'nan' + configuration.zero_stat)(
                    magnitudes
                )

            if configuration.combined_binned_lc:
                combined_magnitudes = numpy.concatenate((
                    combined_magnitudes,
                    magnitudes
                ))
                combined_bjd = numpy.concatenate((
                    combined_bjd,
                    bjd
                ))
            bjd_offset = int(bjd.min())
            bjd -= bjd_offset

            if not configuration.combined_binned_lc:
                plot_lc(bjd,
                        magnitudes,
                        detrending_mode=detrending_mode,
                        lc_color=next(colors),
                        zorder=zorder,
                        configuration=configuration)

        if configuration.combined_binned_lc:
            bjd_offset = int(combined_bjd.min())
            bjd = combined_bjd - bjd_offset
            magnitudes = combined_magnitudes

            plot_lc(bjd,
                    magnitudes,
                    detrending_mode=detrending_mode,
                    lc_color=next(colors),
                    zorder=zorder,
                    configuration=configuration)

    add_transit_to_plot(bjd,
                        bjd_offset,
                        numpy.median(magnitudes),
                        configuration)
    if configuration.fold_period:
        pyplot.xlabel(
            'Phase (P={:.6f} d)'.format(configuration.fold_period)
        )
    else:
        pyplot.xlabel('BJD - {:d}'.format(int(bjd[0])))

    med_mag = numpy.median(magnitudes)
    pyplot.axhspan(ymin=med_mag - scatter,
                   ymax=med_mag + scatter,
                   color='lightgrey',
                   zorder=0)

    pyplot.ylabel('Magnitude')
    pyplot.xlim(configuration.plot_x_range)
    pyplot.ylim(configuration.plot_y_range)
    pyplot.legend()

    if configuration.plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(configuration.plot_fname)


if __name__ == '__main__':
    main(parse_command_line())
