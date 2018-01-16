"""A collection of functions useful when generating masters."""

import numpy

from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.general_purpose_stats import iterative_rejection_average
from superphot_pipeline.image_calibration.mask_utilities import mask_flags

def stack_to_master(frame_list,
                    outlier_threshold,
                    average_func=numpy.nanmedian,
                    **stacking_options):
    """
    Create a master by stacking a list of frames.

    Args:
        frame_list:    The frames to stack. Should be a list of FITS filenames.

        outlier_threshold:    See same name argument to
            `general_purpose_stats.iterative_rejection_average`

        average_func:    See same name argument to
            `general_purpose_stats.iterative_rejection_average`

        **stacking_options:    Keyword only arguments controlling how stacking
            is performed. The following are currently supported:
                - min_valid_values: The minimum number of valid values to average
                    for each pixel. If outlier rejection or masks results in
                    fewer than this, the corresponding pixel gets a bad pixel
                    mask.
                - exclude_mask: A bitwise or of mask flags, any of which result
                    in the corresponding pixels being excluded from the
                    averaging. Other mask flags in the input frames are ignored,
                    treated as clean.
                - max_iter: See same name argument to
                    `general_purpose_stats.iterative_rejection_average`

    Returns:
        master_values:    The best estimate for the values of the maseter at
            each pixel.

        master_stdev:    The best estimate of the standard deviation of the
            master pixels.

        master_mask:    The pixel quality mask for the master.
    """

    pixel_values = None
    for frame_index, frame_fname in enumerate(frame_list):
        image, mask = read_image_components(frame_fname,
                                            read_error=False,
                                            read_header=False)

        if pixel_values is None:
            pixel_values = numpy.empty((len(frame_list),) + image.shape)

        pixel_values[frame_index] = image
        if 'exclude_mask' in stacking_options:
            pixel_values[frame_index][
                numpy.bitwise_and(mask, stacking_options['exclude_mask'])
            ] = numpy.nan

    average_config = dict(outlier_threshold=outlier_threshold,
                          average_func=average_func,
                          axis=0,
                          mangle_input=True,
                          keepdims=False)
    if 'max_iter' in stacking_options:
        average_config['max_iter'] = stacking_options['max_iter']
    master_values, master_stdev, master_num_averaged = (
        iterative_rejection_average(pixel_values, **average_config)
    )

    master_mask = numpy.full(pixel_values.shape(),
                             mask_flags['CLEAR'],
                             dtype='int8')
    master_mask[
        master_num_averaged < stacking_options['min_valid_values']
    ] = mask_flags['BAD']

    return master_values, master_stdev, master_mask
