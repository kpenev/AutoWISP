"""A collection of functions for working with masks."""

from ctypes import cdll, c_long, c_byte, c_char_p
from ctypes.util import find_library

import numpy

from superphot_pipeline.image_utilities import read_image_components
from superphot_pipeline.pipeline_exceptions import ImageMismatchError

git_id = '$Id$'

mask_flags = dict()

def initialize_library():
    """Prepare the superphotio library for use."""

    library_fname = find_library('superphotio')
    if library_fname is None:
        raise OSError("Unable to find SuperPhot's io library.")
    library = cdll.LoadLibrary(library_fname)

    library.parse_hat_mask.argtypes = [
        c_char_p,
        c_long,
        c_long,
        numpy.ctypeslib.ndpointer(dtype=c_byte,
                                  ndim=1,
                                  flags='C_CONTIGUOUS')
    ]
    library.parse_hat_mask.restype = None

    for flag in ['OK',
                 'CLEAR',
                 'FAULT',
                 'HOT',
                 'COSMIC',
                 'OUTER',
                 'OVERSATURATED',
                 'LEAKED',
                 'SATURATED',
                 'INTERPOLATED',
                 'BAD',
                 'ALL',
                 'NAN']:
        mask_flags[flag] = c_byte.in_dll(library, 'MASK_' + flag).value

    return library

superphotio_lib = initialize_library()

def parse_hat_mask(header):
    """
    Extract the HAT-style mask contained in header.

    Args:
        header:    The header of the image whose mask to parse.

    Returns:
        mask:    A dtype=uint8 numpy array with exactly the same resolution as
            the input image containing a bit-field for each pixel indicating
            any bad-pixel flags raised per the header.

    Examples:

        >>> from astropy.io import fits

        >>> with fits.open('/Users/kpenev/tmp/1-447491_4.fits.fz',
        >>>                mode='readonly') as f:
        >>>     image_mask = parse_hat_mask(f[1].header)

        >>>     flag_name = 'OVERSATURATED'

        >>>     matched = numpy.bitwise_and(image_mask,
        >>>                                 mask_flags[flag_name]).astype(bool)

        >>>     #Print number of pixels for which the OVERSATURATED flag is raised
        >>>     print(flag_name + ': ' + repr(matched.sum()))

        >>>     #Output x, y, flux for the pixels flagged as OVERSATURATED
        >>>     for y, x in zip(*numpy.nonzero(matched)):
        >>>         print('%4d %4d %15d' % (x, y, f[1].data[y, x]))
    """

    mask_string = ''.join((c[1] + ' ') if c[0] == 'MASKINFO' else ''
                          for c in header.items()).encode('ascii')
    mask = numpy.zeros((header['NAXIS2'], header['NAXIS1']), dtype='int8')
    superphotio_lib.parse_hat_mask(mask_string,
                                   header['NAXIS1'],
                                   header['NAXIS2'],
                                   mask.ravel())
    return mask

def combine_masks(mask_filenames):
    """
    Create a combined mask image from the masks of all input files.

    Args:
        mask_filenames:    A list of FITS filenames from which to read mask
            images (identified by the IMAGETYP header keyword matching
            [a-z_]*mask).

    Returns:
        mask:    A bitwise or of the mask extensions of all input FITS files.
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
                            ' resolutions, %s (%dx%d) with %s, all with'
                            ' resolution of (%dx%d).'
                        )
                        %
                        (
                            mask_fname,
                            mask_image.shape[0],
                            mask_image.shape[1],
                            ', '.join(mask_filenames[:mask_index]),
                            mask.shape[0],
                            mask.shape[1]
                        )
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
        mask:    A 2-D numpy bitmask array flagging pixels which are above
            `saturation_threshold` or which are adjacent from a saturated pixel
            in a direction in which a charge could leak.
    """

    mask = numpy.full(raw_image.shape(), mask_flags['CLEAR'])

    mask[raw_image > saturation_threshold] = mask_flags['OVERSATURATED']

    y_resolution, x_resolution = raw_image.shape
    for x_offset, y_offset in leak_directions:
        shifted_mask = mask[y_offset:, x_offset:]
        shifted_mask[
            mask[: y_resolution - y_offset,
                 : x_resolution - x_offset] == mask_flags['OVERSATURATED'],
        ] = numpy.bitwise_or(shifted_mask, mask_flags['LEAKED'])

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
        print(flag_name + ': ' + repr(matched.sum()))

        #Output x, y, flux for the pixels flagged as OVERSATURATED
        for y, x in zip(*numpy.nonzero(matched)):
            #pylint: disable=no-member
            #pylint false positive.
            print('%4d %4d %15d' % (x, y, f[1].data[y, x]))
            #pylint: enable=no-member
