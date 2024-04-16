#!/usr/bin/env python3

"""Stack calibrated flat frames to a master flat."""

import logging

import numpy
from scipy import ndimage

from superphot_pipeline.image_calibration.mask_utilities import mask_flags
from superphot_pipeline.processing_steps.manual_util import ignore_progress
from superphot_pipeline.processing_steps.stack_to_master import \
    get_command_line_parser
from superphot_pipeline.image_calibration.master_maker import MasterMaker

input_type = 'calibrated'
_logger = logging.getLogger(__name__)


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = get_command_line_parser(*args,
                                     default_threshold=(2.0, -3.0),
                                     min_frames_arg=False,
                                     default_min_valid_values=3,
                                     default_max_iter=3)
    parser.add_argument(
        '--stamp-fraction',
        type=float,
        default=0.5,
        help='The fraction of the image cloude detection stamps should span.'
    )
    parser.add_argument(
        '--stamp-smoothing-num-x-terms',
        type=int,
        default=3,
        help='The number of x terms to include in the stamp smoothing.'
    )
    parser.add_argument(
        '--stamp-smoothing-num-y-terms',
        type=int,
        default=3,
        help='The number of y terms to include in the stamp smoothing.'
    )
    parser.add_argument(
        '--stamp-smoothing-outlier-threshold',
        type=float,
        nargs='+',
        default=[3.0],
        help='Pixels deviating by more than this many standard deviations form '
        'the best fit smoothing function are discarded after each smoothnig '
        'fit iteration. One or two numbers should be specified. If two, one '
        'should be positive and the other negative specifying separate '
        'thresholds in the positive and negative directions.'
    )
    parser.add_argument(
        '--stamp-smoothing-max-iterations',
        type=int,
        default=1,
        help='The maximum number of fit, reject iterations to use when '
        'smoothing stamps for cloud detection.'
    )
    parser.add_argument(
        '--stamp-pixel-average',
        type=lambda f: getattr(numpy, 'nan' + f),
        default='mean',
        help='The pixels of smoothed cloud detection stamps are averaged using '
        'this function and square difference from that mean is calculated. '
        'The process is iteratively reapeated discarding outlier pixels.'
    )
    parser.add_argument(
        '--stamp-pixel-outlier-threshold',
        type=float,
        nargs='+',
        default=[3.0],
        help='The threshold in deviation around mean units to use for '
        'discarding stamp pixels during averaging of the smoothed stamps. One '
        'or two numbers should be specified. If two, one '
        'should be positive and the other negative specifying separate '
        'thresholds in the positive and negative directions.'
    )
    parser.add_argument(
        '--stamp-pixel-max-iter',
        type=int,
        default=3,
        help='The maximum number of averaging, discarding terations to use '
        'when calculating average and square deviation of the smoothed stamps.'
    )
    parser.add_argument(
        '--stamp-select-max-saturated-fraction',
        type=float,
        default=1e-4,
        help='Stamps with more that this fraction of saturated pixels are '
        'discarded.'
    )
    parser.add_argument(
        '--stamp-select-var-mean-fit-threshold',
        type=float,
        default=2.0,
        help='Variance vs mean fit for stamps is iteratively repeated after '
        'discarding from each iteration stamps that deviater from the fit by '
        'more than this number times the fit residual.'
    )
    parser.add_argument(
        '--stamp-select-cloudy-night-threshold',
        type=float,
        default=5e3,
        help='If variance vs mean quadratic fit has residual of more than this,'
        ' the entire observing session is discarded as cloudy.'
    )
    parser.add_argument(
        '--stamp-select-cloudy-frame-threshold',
        type=float,
        default=2.0,
        help='Individual frames with stamp mean and variance deviating from '
        'the variance vs mean fit by more than this number times the '
        'fit residual are discarded as cloudy.'
    )
    parser.add_argument(
        '--stamp-select-min-high-mean',
        type=float,
        default=6000.0,
        help='The minimum average value of the stamp pixels for a frame to be '
        'included in a high master flat. Default: %(default)s'
    )
    parser.add_argument(
        '--stamp-select-max-low-mean',
        type=float,
        default=3000.0,
        help='The maximum average value of the stamp pixels for a frame to be '
        'included in a low master flat. Default: %(default)s'
    )
    parser.add_argument(
        '--large-scale-smoothing-filter-name',
        type=lambda f: getattr(ndimage, f),
        default='median_filter',
        help='For each frame, the large scale struture is corrected by taking '
        'the ratio of the frame to the reference (median of all input frames), '
        'smoothing this ratio and then dividing by it. Two filters are '
        'consecutively applied for smoothing. This is the first and it should '
        'be one of the scipy.ndimage box filters.'
    )
    parser.add_argument(
        '--large-scale-smoothing-filter-size',
        type=int,
        default=12,
        help='The size of the box filter to apply.'
    )
    parser.add_argument(
        '--large-scale-smoothing-spline-x-nodes',
        type=int,
        default=4,
        help='The second smoothing filter applied fits a separable x and y '
        'spline. The x spline uses this many internal nodes.'
    )
    parser.add_argument(
        '--large-scale-smoothing-spline-y-nodes',
        type=int,
        default=4,
        help='The second smoothing filter applied fits a separable x and y '
        'spline. The x spline uses this many internal nodes.'
    )
    parser.add_argument(
        '--large-scale-smoothing-spline-outlier_threshold',
        type=float,
        help='Spline smoothing discards pixels deviating by more than this '
        'number times the fit residual and iterates.'
    )
    parser.add_argument(
        '--large-scale-smoothing-spline-max-iter',
        type=int,
        default=1,
        help='The maximum number of discard-refit iterations used during '
        'spline smoothing to get the large scale flat.'
    )
    parser.add_argument(
        '--large-scale-smoothing-bin-factor',
        type=int,
        default=4,
        help='Before smoothing is applied the image is binned by this factor.'
    )
    parser.add_argument(
        '--large-scale-smoothing-zoom-interp-order',
        type=int,
        default=4,
        help='After smoothing the image is zoomed back out to its original '
        'size using this order interpolation.'
    )
    parser.add_argument(
        '--cloud-check-smoothing-filter-name',
        type=lambda f: getattr(ndimage, f),
        default='median_filter',
        help='When stacking, images with matched large scale flat, any '
        'image which deviates too much from the large scale flat is '
        'discarded as cloudy. This argument specifies a smoothing filter to '
        'apply to the deviation from the large scale flat before checking '
        'if the image is an outlier.'
    )
    parser.add_argument(
        '--cloud-check-smoothing-filter-size',
        type=int,
        default=8,
        help='The size of the cloud check smoothing fliter to use.'
    )
    parser.add_argument(
        '--cloud-check-smoothing-bin-factor',
        type=int,
        default=4,
        help='Before smoothing is applied the deviation from large scale '
        'flat, it is binned by this factor.'
    )
    parser.add_argument(
        '--cloud-check-smoothing-zoom-interp-order',
        type=int,
        default=4,
        help='After smoothing the deviation from large scale flat is '
        'zoomed back out to its original size using this order interpolation.'
    )
    parser.add_argument(
        '--min-pointing-separation',
        type=float,
        default=150.0,
        help='Individual flat frames must be at least this many arcseconds '
        'apart (per their headers) to be combined in order to avoid stars '
        'showing up above the sky brightness.'
    )
    parser.add_argument(
        '--large-scale-deviation-threshold',
        type=float,
        default=0.05,
        help='If the smoothed difference between a frame and the combined '
        'large scale flat is more than this, the frame is discarded as '
        'potentially cloudy.'
    )

    parser.add_argument(
        '--min-high-combine',
        type=int,
        default=10,
        help='High master flat is generated only if at least this many frames '
        'of high illumination survive all checks.'
    )
    parser.add_argument(
        '--min-low-combine',
        type=int,
        default=5,
        help='Low master flat is generated only if at least this many frames '
        'of low illumination survive all checks.'
    )
    parser.add_argument(
        '--large-scale-stack-outlier-threshold',
        type=float,
        default=4.0,
        help='When determining the large scale flat outlier pixels of smoothed '
        'individual flats by more than this many sigma are discarded.'
    )
    parser.add_argument(
        '--large-scale-stack-average-func',
        type=lambda f: getattr(numpy, 'nan' + f),
        default='nanmedian',
        help='The function used to average individual large scale flats.'
    )
    parser.add_argument(
        '--large-scale-stack-min-valid-values',
        type=int,
        default=3,
        help='The minimum number of surviving pixels after outlier rejection '
        'required to count a stacked large scale pixel as valid.'
    )
    parser.add_argument(
        '--large-scale-stack-max-iter',
        type=int,
        default=1,
        help='The maximum number of rejection-stacking iterations allowed when '
        'creating the large scale flat.'
    )
    parser.add_argument(
        '--large-scale-stack-exclude-mask',
        choices=mask_flags.keys(),
        nargs='+',
        default=MasterMaker.default_exclude_mask,
        help='A list of mask flags, any of which result in the corresponding '
        'pixels being excluded from the averaging. Any mask flags not specified'
        'are ignored, treated as clean. Note that ``\'BAD\'`` means any kind of'
        ' problem (e.g. saturated, hot/cold pixel, leaked etc.).'
    )
    return parser.parse_args(*args)


if __name__ == '__main__':
    cmdline_config = parse_command_line()
    setup_process(task='main', **cmdline_config)
    stack_to_master_flat(
        find_fits_fnames(cmdline_config['calibrated_images']),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress
    )

