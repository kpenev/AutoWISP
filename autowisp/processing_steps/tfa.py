#!/usr/bin/env python3

"""Apply TFA correction to lightcurves."""

from general_purpose_python_modules.multiprocessing_util import setup_process

from autowisp import TFACorrection, DataReductionFile
from autowisp.file_utilities import find_lc_fnames
from autowisp.processing_steps.lc_detrending_argument_parser import\
    LCDetrendingArgumentParser
from autowisp.processing_steps.lc_detrending import\
    detrend_light_curves
from autowisp.light_curves.apply_correction import\
    load_correction_statistics
from autowisp.processing_steps.manual_util import ignore_progress


def parse_command_line(*args):
    """Parse the commandline optinos to a dictionary."""

    return LCDetrendingArgumentParser(
        mode='TFA',
        description=__doc__,
        input_type=('' if args else 'lc')
    ).parse_args(*args)


def tfa(lc_collection, start_status, configuration, mark_progress):
    """Perform TFA on (a subset of the points in) the given lightucurves."""

    assert start_status == 0

    configuration['fit_datasets'] = configuration.pop('tfa_datasets')
    for param in list(configuration.keys()):
        if param.startswith('tfa_'):
            print(f'Renaming {param!r} -> {param[4:]!r}')
            configuration[param[4:]] = configuration.pop(param)
        else:
            print('Not renaming ' + repr(param))

    with DataReductionFile(configuration['single_photref_dr_fname'],
                           'r') as sphotref_dr:
        sphotref_header = sphotref_dr.get_frame_header()

    configuration['fit_points_filter_expression'] = configuration.pop(
        'lc_points_filter_expression'
    )

    detrend_light_curves(
        lc_collection,
        configuration,
        TFACorrection(
            load_correction_statistics(
                configuration['epd_statistics_fname'].format_map(
                    sphotref_header
                )
            ),
            configuration,
            error_avg=configuration['detrend_error_avg'],
            rej_level=configuration['detrend_rej_level'],
            max_rej_iter=configuration['detrend_max_rej_iter'],
            fit_identifier='TFA',
            verify_template_data=True,
            mark_progress=mark_progress
        ),
        configuration.pop('statistics_fname').format_map(sphotref_header)
    )


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='manage', **cmdline_config)
    tfa(find_lc_fnames(cmdline_config.pop('lc_files')),
        0,
        cmdline_config,
        ignore_progress)
