"""Define a generic function to make 3-hdu FITS images (image, error, mask)."""

from os import path, makedirs

from astropy.io import fits
import numpy

def create_result(image_list,
                  header,
                  result_fname,
                  compress,
                  *,
                  split_channels=False,
                  allow_overwrite=False,
                  **fname_substitutions):
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

        result_fname:    See Calibrator.__call__.

        compress:    Should the created image be compressed? If value other than
            False or None is used, compression is enabled and this parameter
            specifies the quantization level of the compression.

        allow_overwrite:    If a file named **result_fname** already exists,
            should it be overwritten (otherwise throw an exception).

        fname_substitutions:   Any parameters in addition to header entries
            required to generate the output filename.

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

    if not split_channels:
        split_channels = {None: slice(None)}

    for channel_name, channel_slice in split_channels.items():
        if channel_name is not None:
            header['CLRCHNL'] = channel_name
        hdu_list = fits.HDUList([
            fits.PrimaryHDU(image_list[0][channel_slice], header),
            fits.ImageHDU(image_list[1][channel_slice], header_list[1]),
            fits.ImageHDU(image_list[2][channel_slice].astype('uint8'),
                          header_list[2])
        ])
        for hdu in hdu_list:
            hdu.update_header()

        if compress is not False and compress is not None:
            hdu_list = fits.HDUList(
                [fits.PrimaryHDU()]
                +
                [
                    fits.CompImageHDU(hdu.data,
                                      hdu.header,
                                      quantize_level=compress)
                    for hdu in hdu_list
                ]
            )

        output_fname = result_fname.format(**header, **fname_substitutions)
        if not path.exists(path.dirname(output_fname)):
            makedirs(path.dirname(output_fname))
        hdu_list.writeto(output_fname, overwrite=allow_overwrite)
