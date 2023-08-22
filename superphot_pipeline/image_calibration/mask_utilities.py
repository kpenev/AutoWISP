"""
A collection of functions for working with masks.

Attributes:

    mask_flags:    Dictionary contaning the possible bad pixel flags and the
        corresponding bitmasks.
"""

from logging import getLogger

import numpy

from superphot.hat_masks import parse_hat_mask, mask_flags

from superphot_pipeline.fits_utilities import read_image_components
from superphot_pipeline.pipeline_exceptions import ImageMismatchError

git_id = '$Id$'

_logger = getLogger(__name__)

def combine_masks(mask_filenames):
    r"""
    Create a combined mask image from the masks of all input files.

    Args:
        mask_filenames:    A list of FITS filenames from which to read mask
            images (identified by the IMAGETYP header keyword matching
            [a-z\_]*mask).

    Returns:
        numpy.array(dtype=uint8):
            A bitwise or of the mask extensions of all input FITS files.
    """

    mask = None
    for mask_index, mask_fname in enumerate(mask_filenames):
        mask_image = read_image_components(mask_fname)[2]
        if mask_image is not None:
            if mask is None:
                mask = mask_image
            else:
                if mask.shape != mask_image.shape:
                    raise ImageMismatchError(
                        (
                            'Attempting to combine masks with different'
                            f' resolutions, {mask_fname:s} '
                            f'({mask_image.shape[0]:d}x{mask_image.shape[1]:d})'
                            ' with %s, all with resolution of '
                            f'({mask.shape[0]:d}x{mask.shape[1]:d}).'
                        )
                        %
                        ', '.join(mask_filenames[:mask_index])
                    )
                mask = numpy.bitwise_or(mask, mask_image)
            break

    return mask

def get_saturation_mask(raw_image,
                        saturation_threshold,
                        leak_directions):
    """
    Create a mask indicating saturated and leaked into pixels.

    Args:
        raw_image:    The image for which to generate the saturation mask.

        saturation_threshold: The pixel value which is considered saturated.
            Generally speaking this should be where the response of the pixel
            starts to deviate from linear.

        leak_directions:    Directions in which charge overflows out of
            satuarted pixels. Should be a list of 2-tuples giving the x and
            y offset to which charge is leaked.

    Returns:
        numpy.array(dtype=uint8):
            A bitmask array flagging pixels which are above
            **saturation_threshold** or which are adjacent from a saturated
            pixel in a direction in which a charge could leak.
    """

    _logger.debug('Raw image shape: %s', repr(raw_image.shape))
    mask = numpy.full(raw_image.shape, mask_flags['CLEAR'], dtype='int8')

    mask[raw_image > saturation_threshold] = mask_flags['OVERSATURATED']
    _logger.debug('Mask shape: %s', repr(mask.shape))

    y_resolution, x_resolution = raw_image.shape
    for x_offset, y_offset in leak_directions:
        shifted_mask = mask[
            max(y_offset, 0): y_resolution + min(0, y_offset),
            max(x_offset, 0): x_resolution + min(0, x_offset)
        ]
        _logger.debug('Shifted mask shape: %s', repr(shifted_mask.shape))
        leaked_pixels = (
            mask[
                max(-y_offset, 0) : y_resolution + min(0, -y_offset),
                max(-x_offset, 0) : x_resolution + min(0, -x_offset)
            ]
            ==
            mask_flags['OVERSATURATED']
        )
        shifted_mask[leaked_pixels] = numpy.bitwise_or(
            shifted_mask[leaked_pixels],
            mask_flags['LEAKED']
        )

    return mask

if __name__ == '__main__':
    from astropy.io import fits

    with fits.open('/Users/kpenev/tmp/1-447491_4.fits.fz',
                   mode='readonly') as f:
        #pylint: disable=no-member
        #pylint false positive.
        image_mask = parse_hat_mask(f[1].header)
        #pylint: enable=no-member

        flag_name = 'OVERSATURATED'

        matched = numpy.bitwise_and(image_mask,
                                    mask_flags[flag_name]).astype(bool)

        #Print number of pixels for which the OVERSATURATED flag is raised
        _logger.debug('%s: %s', flag_name, repr(matched.sum()))

        #Output x, y, flux for the pixels flagged as OVERSATURATED
        for y, x in zip(*numpy.nonzero(matched)):
            #pylint: disable=no-member
            #pylint false positive.
            _logger.debug('%4d %4d %15d', x, y, f[1].data[y, x])
            #pylint: enable=no-member
