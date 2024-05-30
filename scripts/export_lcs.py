#!/usr/bin/env python3

"""Export the "best" photometry from each LC to text file."""

from os import path
from functools import partial

from astropy.table import Table
from configargparse import ArgumentParser, DefaultsFormatter
import numpy

from autowisp.file_utilities import find_lc_fnames

from plot_lc_scatter import get_lc_minimum_scatter

def parse_command_line():
    """Return command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['export_lcs.cfg'],
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
        '--min-unrejected-points', '--min-unrej',
        type=int,
        default=500,
        help='The minimum number of unrejected points required for a '
        'particular photometry version to be considered for export.'
    )
    parser.add_argument(
        '--detrending-mode',
        default='epd',
        choices=['magfit', 'epd', 'tfa'],
        help='Which version of the detrending to export.'
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
        '--output-dir', '-o',
        default='.',
        help='The directory to which the exported lightcurves will be written.'
    )

    return parser.parse_args()


def main(config):
    """Avoid global variables."""

    pick_photometry_kwargs = {
        'detrending_mode': config.detrending_mode,
        'magfit_iteration': config.magfit_iteration,
        'min_lc_length': config.min_unrejected_points,
        'calculate_average': config.average,
        'calculate_scatter': config.statistic,
        'outlier_threshold': config.outlier_threshold,
        'max_outlier_rejections': config.max_outlier_rejections,
    }
    for lc_fname in find_lc_fnames(config.lightcurves):
        bjd, best_mags = get_lc_minimum_scatter(
            lc_fname,
            **pick_photometry_kwargs
        )[-2:]
        if best_mags is not None:
            Table([bjd, best_mags], names=['bjd', 'mag']).write(
                path.join(
                    config.output_dir,
                    path.splitext(path.basename(lc_fname))[0] + '.txt'
                ),
                format='ascii'
            )


if __name__ == '__main__':
    main(parse_command_line())
