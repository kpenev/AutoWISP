#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from asteval import Interpreter

from superphot_pipeline import EPDCorrection
from superphot_pipeline.file_utilities import find_lc_fnames
from superphot_pipeline.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from superphot_pipeline.processing_steps.lc_detrending import\
    detrend_light_curves

def parse_epd_variable(argument):
    """Parse epd-variables argument to the format required by EPDCorrection."""

    aeval = Interpreter()
    variable, var_tail = argument.split('=')
    dataset_id, substitution = var_tail.split(':')
    substitution = substitution.replace(';', '=')
    return variable, (dataset_id, aeval(substitution))


if __name__ == '__main__':
    cmdline_config = LCDetrendingArgumentParser(
        mode='EPD',
        description=__doc__
    ).parse_args()

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
            fit_datasets=cmdline_config['detrend_datasets'],
            fit_weights=cmdline_config['fit_weights'],
            error_avg=cmdline_config['detrend_error_avg'],
            rej_level=cmdline_config['detrend_rej_level'],
            max_rej_iter=cmdline_config['detrend_max_rej_iter'],
            pre_reject=cmdline_config['pre_reject_outliers']
        ),
        cmdline_config.pop('epd_statistics_fname')
    )
