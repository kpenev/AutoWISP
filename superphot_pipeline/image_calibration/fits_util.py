"""Define a generic function to make 3-hdu FITS images (image, error, mask)."""

from astropy.io import fits
import numpy

def create_result(image_list,
                  header,
                  result_fname,
                  compress,
                  allow_overwrite=False):
    """
    Create a 3-extension FITS file out of 3 numpy images and header.

    All FITS files produced during calibration (calibrated frames and masters)
    contain 3 header data units:

        * the actual image and header,
        * an error estimate
        * a bad pixel mask.

    Which one is which is identified by the IMAGETYP keyword in the
    corresponding header. The image HDU can have an arbitrary IMAGETYP, while
    the mask and error HDU have 'IMAGETYP'='mask' and 'IMAGETYP'='error'
    respectively.

    Args:
        image_list:    A list with 3 entries of image data for the output
            file. Namely, the calibrated image, an estimate of the error and a
            mask image. The images are saved as extensions in this
            same order.

        header:    The header to use for the the primary (calibrated) image.

        result_fname:    The filename under which to save the craeted image.

        compress:    Should the created image be compressed? If value other than
            False or None is used, compression is enabled and this parameter
            specifies the quantization level of the compression.

        allow_overwrite:    If a file named **result_fname** already exists,
            should it be overwritten (otherwise throw an exception).

    Returns:
        None
    """


    header_list = [header, fits.Header(), fits.Header()]
    header_list[1]['IMAGETYP'] = 'error'
    header_list[2]['IMAGETYP'] = 'mask'

    header['BITPIX'] = header_list[1]['BITPIX'] = -32
    header_list[2]['BITPIX'] = 8

    for check_image in image_list:
        assert numpy.isfinite(check_image).all()

    assert (image_list[1] > 0).all()

    hdu_list = fits.HDUList([
        fits.PrimaryHDU(image_list[0], header),
        fits.ImageHDU(image_list[1], header_list[1]),
        fits.ImageHDU(image_list[2].astype('uint8'), header_list[2])
    ])
    for hdu in hdu_list:
        hdu.update_header()

    if compress is not False and compress is not None:
        hdu_list = fits.HDUList(
            [fits.PrimaryHDU()]
            +
            [
                fits.CompImageHDU(hdu.data, hdu.header, quantize_level=compress)
                for hdu in hdu_list
            ]
        )

    hdu_list.writeto(result_fname, overwrite=allow_overwrite)
