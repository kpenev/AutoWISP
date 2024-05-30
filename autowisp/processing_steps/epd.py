#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from autowisp import EPDCorrection
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from autowisp.processing_steps.lc_detrending import\
    detrend_light_curves

def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode='EPD',
        description=__doc__,
        input_type=('' if args else 'lc')
    ).parse_args(*args)

if __name__ == '__main__':
    cmdline_config = parse_command_line()

    detrend_light_curves(
        find_lc_fnames(cmdline_config.pop('lc_files')),
        cmdline_config,
        EPDCorrection(
            fit_identifier='EPD',
            used_variables=dict(cmdline_config['epd_variables']),
            fit_points_filter_expression=(
                cmdline_config['fit_points_filter_expression']
            ),
            fit_terms_expression=cmdline_config['epd_terms_expression'],
            fit_datasets=cmdline_config['fit_datasets'],
            fit_weights=cmdline_config['fit_weights'],
            error_avg=cmdline_config['detrend_error_avg'],
            rej_level=cmdline_config['detrend_rej_level'],
            max_rej_iter=cmdline_config['detrend_max_rej_iter'],
            pre_reject=cmdline_config['pre_reject_outliers']
        ),
        cmdline_config.pop('epd_statistics_fname')
    )
