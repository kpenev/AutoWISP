"""A collection of functions for working with pipeline images."""

from astropy.io import fits

import numpy

from superphot_pipeline.pipeline_exceptions import BadImageError

git_id = '$Id$'

def read_image_components(fits_fname,
                          read_error=True,
                          read_mask=True,
                          read_header=True):
    """
    Read image, its error estimate, mask and header from pipeline FITS file.

    Args:
        fits_fname:    The filename of the FITS file to read the componets of.
            Must have been produced by the pipeline.

        read_error:    Should the error extension be searched for and read.
        read_mask:    Should the mask extension be searched for and read.
        read_header:    Should the header of the image extension be returned.

    Returns:
        image:   The primary image in the file. Always present.

        error:   The error estimate of image, identified by IMAGETYP=='error'.
            Set to None if none of the extensions have IMAGETYP=='error'. This
            is omitted from the output if `read_error == False`.

        mask:    A bitmask of quality flags for each image pixel (identified
            by IMAGETYP='mask'). Set to None if none of the extensions
            have IMAGETYP='mask'. This is omitted from the output if
            `read_mask == False`.

        header:   The header of the image HDU in the file. This is omitted from
            the output if `read_header == False`.
    """

    image = error = mask = header = None
    with fits.open(fits_fname, mode='readonly') as input_file:
        for hdu_index, hdu in enumerate(input_file):
            if hdu.data is None:
                continue
            if image is None:
                image = hdu.data
                if read_header:
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
            if (
                    image is not None
                    and
                    (error is not None or not read_error)
                    and
                    (mask is not None or not read_mask)
            ):
                break
    return (
        (image,)
        +
        ((error,) if read_error else ())
        +
        ((mask,) if read_mask else())
        +
        ((header,) if read_header else())
    )
