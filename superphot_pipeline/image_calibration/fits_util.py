"""Define a generic function to make 3-hdu FITS images (image, error, mask)."""

from os import path, makedirs
from logging import getLogger

from astropy.io import fits
from astropy.time import Time, TimeISO
import numpy

from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline import Evaluator

class TimeISOTNoSep(TimeISO):
    """
    A class to parse ISO times without any separators.
    """
    name = "isotnosep"
    subfmts = (
        (
            "date_hms",
            "%Y%m%dT%H%M%S",
            "{year:d}{mon:02d}{day:02d}T{hour:02d}{min:02d}{sec:02d}",
        ),
        (
            "date_hm",
            "%Y%m%dT%H%M",
            "{year:d}{mon:02d}{day:02d}T{hour:02d}{min:02d}",
        ),
        ("date", "%Y-%m-%d", "{year:d}{mon:02d}{day:02d}"),
    )


    # See TimeISO for explanation
    fast_parser_pars = {
        'delims': (0, 0, 0, ord('T'), 0, 0, 0, 0),
        'starts': (0, 4, 6, 8, 11, 13, 15),
        'stops': (3, 5, 7, 10, 12, 14, -1),
        # Break allowed *before*
        #                 y  m  d  h  m  s  f
        'break_allowed': (0, 0, 0, 0, 0, 0, 0),
        'has_day_of_year': 0
    }


def add_required_keywords(header, calibration_params):
    """Add keywords required by the pipeline to the given header."""

    if calibration_params.get('utc_expression'):
        header['JD-OBS'] = Time(
            Evaluator(header)(calibration_params['utc_expression'])
        ).jd
    else:
        assert calibration_params.get('jd_expression')
        header['JD-OBS'] = Time(
            Evaluator(header)(calibration_params['jd_expression'])
        ).jd

    header['FNUM'] = Evaluator(header)(calibration_params['fnum'])


def get_raw_header(raw_image, calibration_params):
    """Return the raw header to base the calibrated frame header on."""

    result = get_primary_header(
        raw_image,
        add_filename_keywords=True
    )
    if calibration_params.get('combine_headers'):
        result.update(raw_image[calibration_params['raw_hdu']].header)
    add_required_keywords(result, calibration_params)
    return result


def add_channel_keywords(header, channel_name, channel_slice):
    """Add the extra keywords describing channel to header."""

    if channel_name is not None:
        header['CLRCHNL'] = channel_name
        #False positive
        #pylint: disable=unsubscriptable-object
        header['CHNLXOFF'] = channel_slice[1].start
        header['CHNLXSTP'] = channel_slice[1].step
        header['CHNLYOFF'] = channel_slice[0].start
        header['CHNLYSTP'] = channel_slice[0].step
        #pylint: enable=unsubscriptable-object
    else:
        header['CHNLXOFF'] = 0
        header['CHNLXSTP'] = 1
        header['CHNLYOFF'] = 0
        header['CHNLYSTP'] = 1


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

        compress:    Should the created image be compressed? If the value
            converts to True, compression is enabled and this parameter
            specifies the quantization level of the compression.

        allow_overwrite:    If a file named **result_fname** already exists,
            should it be overwritten (otherwise throw an exception).

        fname_substitutions:   Any parameters in addition to header entries
            required to generate the output filename.

    Returns:
        None
    """

    logger = getLogger(__name__)

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
        logger.debug('Slice for %s channel: %s',
                     channel_name,
                     repr(channel_slice))
        add_channel_keywords(header, channel_name, channel_slice)

        hdu_list = fits.HDUList([
            fits.PrimaryHDU(
                numpy.array(image_list[0][channel_slice]),
                header
            ),
            fits.ImageHDU(
                numpy.array(image_list[1][channel_slice]),
                header_list[1]
            ),
            fits.ImageHDU(
                numpy.array(image_list[2][channel_slice]).astype('uint8'),
                header_list[2]
            )
        ])
        for hdu in hdu_list:
            hdu.update_header()

        logger.debug('Compression level: %s', repr(compress))
        if compress:
            logger.debug('Creating compressed HDU')
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

        fname_substitutions.update(header)
        output_fname = result_fname.format_map(fname_substitutions)
        if not path.exists(path.dirname(output_fname)):
            makedirs(path.dirname(output_fname))
        hdu_list.writeto(output_fname, overwrite=allow_overwrite)
