#!/usr/bin/env python3

"""Apply EPD correction to lightcurves."""

from superphot_pipeline.processing_steps.manual_util import\
    LCDetrendingArgumentParser
from superphot_pipeline.processing_steps.lc_detrending import\
    detrend_light_curves

def parse_command_line():
    """Return the prased command line argumets."""

    parser = LCDetrendingArgumentParser(description=__doc__)

    parser.add_argument(
        '--used-variables',
        type=parse_epd_variable,
        action='append',
        help='Keys are variables used in `fit_points_filter_expression` and '
        '`fit_terms_expression` and the corresponding values are 2-tuples '
        'of pipeline keys corresponding to each variable and an associated '
        'dictionary of path substitutions. Each entry defines a unique '
        'independent variable to use in the fit or based on which to select '
        'points to fit. For example: make a list such as '
        '[x=srcproj.x:dict(),y=srcproj.y:dict(),'
        'S=srcextract.psf_map.eval:dict(srcextract_psf_param;"S")] '
        'which corresponds to the interpreted list '
        '[x=("srcproj.x", dict()),y=("srcproj.y", dict()),'
        'S=("srcextract.psf_map.eval",dict(srcextract_psf_param="S"))].'
    )
    parser.add_argument(
        '--fit-terms-expression',
        default='O2{x}',
        type=str,
        help='A fitting terms expression involving only variables from '
             '`used_variables` which expands to the various terms to use '
             'in a linear least squares EPD correction.'
             'Default: %(default)s'
    )
    parser.add_argument(
        '--fit-weights',
        default=None,
        type=str,
        help='An expression involving only variables from `used_variables` '
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

    return parser.parse_args()

if __name__ == '__main__':
    cmdline_config = parse_command_line()

    detrend_light_curves(
        cmdline_args,
        EPDCorrection(
            fit_identifier='EPD',
            used_variables=dict(cmdline_args.used_variables),
            fit_points_filter_expression=(
                cmdline_args.fit_points_filter_expression
            ),
            fit_terms_expression=cmdline_args.fit_terms_expression,
            fit_datasets=cmdline_args.fit_datasets,
            fit_weights=cmdline_args.fit_weights,
            error_avg=cmdline_args.error_avg,
            rej_level=cmdline_args.rej_level,
            max_rej_iter=cmdline_args.max_rej_iter,
            pre_reject=cmdline_args.pre_reject_outliers
        ),
        cmdline_args.epd_statistics_fname
    )
