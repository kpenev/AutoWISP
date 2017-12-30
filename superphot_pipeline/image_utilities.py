"""A collection of functions for working with pipeline images."""

from astropy.io import fits

import numpy

from superphot_pipeline.pipeline_exceptions import BadImageError

git_id = '$Id$'

def read_image_components(fits_fname):
    """
    Read image, its error estimate, mask and header from pipeline FITS file.

    Args:
        fits_fname:    The filename of the FITS file to read the componets of.
            Must have been produced by the pipeline.

    Returns:
        image:   The primary image in the file. Always present.

        error:   The error estimate of image, identified by IMAGETYP=='error'.
            Set to None if none of the extensions have IMAGETYP=='error'.

        mask:    A bitmask of quality flags for each image pixel (identified
            by IMAGETYP='mask'). Set to None if none of the extensions
            have IMAGETYP='mask'.

        header:   The header of the image HDU in the file.
    """

    image = error = mask = header = None
    with fits.open(fits_fname, mode='readonly') as input_file:
        for hdu_index, hdu in enumerate(input_file):
            if image is None and hdu.data is not None:
                image = hdu.data
                header = hdu.header
            else:
                if hdu.header['IMAGETYP'] == 'error':
                    error = hdu.data
                elif hdu.header['IMAGETYP'] == 'mask':
                    mask = hdu.data
                    if mask.dtype != numpy.dtype('int8'):
                        raise BadImageError(
                            (
                                'Mask image (hdu #%d) of %s had data type %s '
                                '(not int8)'
                            )
                            %
                            (hdu_index, fits_fname, mask.dtype)
                        )

    return image, error, mask, header
