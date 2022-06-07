#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from asteval import Interpreter

from superphot_pipeline import EPDCorrection
from superphot_pipeline.file_utilities import find_lc_fnames
from superphot_pipeline.processing_steps.manual_util import\
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


def parse_command_line():
    """Return the prased command line argumets."""

    parser = LCDetrendingArgumentParser(description=__doc__)

    parser.add_argument(
        '--epd-variables',
        type=parse_epd_variable,
        action='append',
        help='Keys are variables used in `fit_points_filter_expression` and '
        '`epd_terms_expression` and the corresponding values are 2-tuples '
        'of pipeline keys corresponding to each variable and an associated '
        'dictionary of path substitutions. Each entry defines a unique '
        'independent variable to use in the fit or based on which to select '
        'points to fit. In the config file specified as a list, such as '
        '[x=srcproj.x:dict(),y=srcproj.y:dict(),'
        'S=srcextract.psf_map.eval:dict(srcextract_psf_param;"S")] '
        'which gets parsed as: '
        '[("x", ("srcproj.x", dict())), ("y", ("srcproj.y", dict())),'
        '("S", ("srcextract.psf_map.eval",dict(srcextract_psf_param="S")))]. '
        'By default, only the zenith distance is used: '
        '[z=skypos.zenith_distance:dict()]'
    )

    parser.add_argument(
        '--epd-terms-expression',
        default='O3{1/cos(z)}',
        type=str,
        help='A fitting terms expression involving only variables from '
             '`epd_variables` which expands to the various terms to use '
             'in a linear least squares EPD correction.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--fit-weights',
        default=None,
        type=str,
        help='An expression involving only variables from `epd_variables` '
        'which should evaluate to the weights to use per LC point in a linear '
        'least squares EPD correction. If left unspecified, no weighting is '
        'performed.'
    )
    parser.add_argument(
        '--skip-outlier-prerejection',
        action='store_false',
        dest='pre_reject_outliers',
        help='If passed the initial rejection of outliers before the fit begins'
        ' is not performed.'
    )

    result = parser.parse_args()
    if result.get('epd_variables') is None:
        result['epd_variables'] = [('z', ('skypos.zenith_distance', dict()))]
    return result


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
            fit_datasets=cmdline_config['detrend_datasets'],
            fit_weights=cmdline_config['fit_weights'],
            error_avg=cmdline_config['detrend_error_avg'],
            rej_level=cmdline_config['detrend_rej_level'],
            max_rej_iter=cmdline_config['detrend_max_rej_iter'],
            pre_reject=cmdline_config['pre_reject_outliers']
        ),
        cmdline_config.pop('epd_statistics_fname')
    )
