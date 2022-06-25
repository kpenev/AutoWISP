#!/usr/bin/env python3

"""Apply TFA correction to lightcurves."""

from superphot_pipeline import TFACorrection
from superphot_pipeline.file_utilities import find_lc_fnames
from superphot_pipeline.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from superphot_pipeline.processing_steps.lc_detrending import\
    detrend_light_curves
from superphot_pipeline.light_curves.apply_correction import\
    load_correction_statistics


if __name__ == '__main__':
    cmdline_config = LCDetrendingArgumentParser(
        mode='TFA',
        description=__doc__
    ).parse_args()

    detrend_light_curves(
        find_lc_fnames(cmdline_config.pop('lc_files')),
        cmdline_config,
        TFACorrection(
            load_correction_statistics(cmdline_config['epd_statistics_fname']),
            cmdline_config,
            error_avg=cmdline_config['detrend_error_avg'],
            rej_level=cmdline_config['detrend_rej_level'],
            max_rej_iter=cmdline_config['detrend_max_rej_iter'],
            fit_identifier='TFA',
            verify_template_data=True
        ),
        cmdline_config.pop('statistics_fname')
    )
