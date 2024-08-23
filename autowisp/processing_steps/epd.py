#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from general_purpose_python_modules.multiprocessing_util import setup_process

from autowisp import EPDCorrection
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from autowisp.processing_steps.lc_detrending import\
    detrend_light_curves
from autowisp.processing_steps.manual_util import ignore_progress

def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode='EPD',
        description=__doc__,
        input_type=('' if args else 'lc')
    ).parse_args(*args)


def epd(lc_collection, start_status, configuration, mark_start, mark_end):
    """Perform EPD on (a subset of the points in) the given lightucurves."""

    assert start_status is None
    detrend_light_curves(
        lc_collection,
        configuration,
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
            pre_reject=cmdline_config['pre_reject_outliers'],
            mark_start=mark_start,
            mark_end=mark_end
        ),
        cmdline_config.pop('epd_statistics_fname'),
    )


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='manage', **cmdline_config)
    epd(find_lc_fnames(cmdline_config.pop('lc_files')),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress)
