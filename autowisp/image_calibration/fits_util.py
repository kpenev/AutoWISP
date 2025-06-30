"""Define a generic function to make 3-hdu FITS images (image, error, mask)."""

from os import path, makedirs
from logging import getLogger

from astropy.io import fits
from astropy.time import Time, TimeISO
import numpy

from autowisp.fits_utilities import read_image_components, get_primary_header
from autowisp import Evaluator
from autowisp.image_calibration.mask_utilities import combine_masks


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
        "delims": (0, 0, 0, ord("T"), 0, 0, 0, 0),
        "starts": (0, 4, 6, 8, 11, 13, 15),
        "stops": (3, 5, 7, 10, 12, 14, -1),
        # Break allowed *before*
        #                 y  m  d  h  m  s  f
        "break_allowed": (0, 0, 0, 0, 0, 0, 0),
        "has_day_of_year": 0,
    }


# pylint: disable=too-many-branches
def assemble_channels(filename, hdu, split_channels):
    """
    Assemble the channels in the pattern they have in raw frames.

    Args:
        filename(str or dict):    Either a single filename (if each channel
            is stored in a different HDU or if there is only one channel or
            if channels have not been split yet) or a dictionary indexed by
            channel name (if different channels are in different files).

        hdu(int or dict or ``'components'`` or ``'masks'``):

            * A single HDU index if channels are in separate files or there
              is only one channel or the channels have not been split yet.

            * dictionary indexed by channel (if channels are in seprate HDUs
              of the same file).

            * The string ``'components'`` if assembling a assembling files
              with image, error, and mask HDUs

        split_channels(dict):    See same name attribute of
            :class:`Calibrator`_.

    Returns:
        One or three numpy arrays constructed by staggering the individual
        channels as they were in the raw image. If hdu is ``'components'``,
        the input image is assumed already to be trimmed to the image area.
    """

    logger = getLogger(__name__)

    if (isinstance(filename, str) and isinstance(hdu, int)) or len(
        split_channels
    ) == 1:
        if isinstance(filename, dict):
            filename = next(iter(filename.values()))
        if hdu == "components":
            return read_image_components(filename, read_header=False)
        if hdu == "masks":
            return read_image_components(
                filename, read_image=False, read_error=False, read_header=False
            )[0]
        with fits.open(filename, "readonly") as image:
            return image[hdu].data.astype("float64")

    assert len(split_channels) > 1

    logger.debug(
        "Assumbling multi-channel image from %s with HDUs: %s, split_channels: "
        "%s.",
        repr(filename),
        repr(hdu),
        repr(split_channels),
    )

    shape = (0, 0)
    for channel_name, channel_slice in split_channels.items():
        with fits.open(
            (
                filename[channel_name]
                if isinstance(filename, dict)
                else filename
            ),
            "readonly",
        ) as image:
            if hdu in ["components", "masks"]:
                data_hdu = 1
            elif isinstance(hdu, dict):
                data_hdu = hdu[channel_name]
            else:
                data_hdu = hdu
            shape = tuple(
                max(
                    shape[i],
                    (channel_slice[i].start or 0)
                    + (image[data_hdu].shape[i] - 1) * channel_slice[i].step
                    + 1,
                )
                for i in range(2)
            )

    if hdu == "masks":
        assert isinstance(filename, dict)
        result = numpy.empty(shape=shape, dtype=numpy.uint8)
        for channel_name, channel_slice in split_channels:
            result[channel_slice] = combine_masks(filename[channel_name])
        return result

    if hdu == "components":
        assert isinstance(filename, dict)
        result = (
            numpy.empty(shape, dtype=numpy.float64),
            numpy.empty(shape, dtype=numpy.float64),
            numpy.empty(shape, dtype=numpy.uint8),
        )
    else:
        result = numpy.empty(shape, dtype=numpy.float64)

    for channel_name, channel_slice in split_channels.items():
        if hdu == "components":
            # False positive
            # pylint: disable=unbalanced-tuple-unpacking
            (
                result[0][channel_slice],
                result[1][channel_slice],
                result[2][channel_slice],
            ) = read_image_components(filename[channel_name], read_header=False)
            # pylint: enable=unbalanced-tuple-unpacking
        else:
            with fits.open(
                (
                    filename[channel_name]
                    if isinstance(filename, dict)
                    else filename
                ),
                "readonly",
            ) as image:
                result[channel_slice] = image[
                    hdu[channel_name] if isinstance(hdu, dict) else hdu
                ].data
    return result


# pylint: enable=too-many-branches


def add_required_keywords(header, calibration_params, for_eval=False):
    """Add keywords required by the pipeline to the given header."""

    jd_keyword = "JD_OBS" if for_eval else "JD-OBS"
    header_eval = Evaluator(header)
    if calibration_params.get("exposure_start_utc"):
        header[jd_keyword] = Time(
            header_eval(calibration_params["exposure_start_utc"])
        ).jd
    else:
        assert calibration_params.get("exposure_start_jd")
        header[jd_keyword] = Time(
            header_eval(calibration_params["exposure_start_jd"]), format="jd"
        ).jd

    header[jd_keyword] += header_eval(
        calibration_params["exposure_seconds"]
    ) / (2.0 * 24.0 * 3600.0)

    header["FNUM"] = Evaluator(header)(calibration_params["fnum"])


def get_raw_header(raw_image, calibration_params):
    """Return the raw header to base the calibrated frame header on."""

    result = get_primary_header(raw_image, add_filename_keywords=True)
    if calibration_params.get("combine_headers"):
        for raw_hdu in (
            calibration_params["raw_hdu"].values()
            if isinstance(calibration_params["raw_hdu"], dict)
            else [calibration_params["raw_hdu"]]
        ):
            result.update(raw_image[raw_hdu].header)
    add_required_keywords(result, calibration_params)
    return result


def add_channel_keywords(header, channel_name, channel_slice):
    """Add the extra keywords describing channel to header."""

    if channel_name is not None:
        header["CLRCHNL"] = channel_name
        # False positive
        # pylint: disable=unsubscriptable-object
        header["CHNLXOFF"] = channel_slice[1].start
        header["CHNLXSTP"] = channel_slice[1].step
        header["CHNLYOFF"] = channel_slice[0].start
        header["CHNLYSTP"] = channel_slice[0].step
        # pylint: enable=unsubscriptable-object
    else:
        header["CHNLXOFF"] = 0
        header["CHNLXSTP"] = 1
        header["CHNLYOFF"] = 0
        header["CHNLYSTP"] = 1


def create_result(
    image_list,
    header,
    result_fname,
    compress,
    *,
    split_channels=False,
    allow_overwrite=False,
    **fname_substitutions,
):
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

    if not split_channels:
        split_channels = {None: slice(None)}

    logger = getLogger(__name__)

    for check_image in image_list:
        assert numpy.isfinite(check_image).all()

    assert (image_list[1] >= 0).all()

    for channel_name, channel_slice in split_channels.items():
        header_list = [
            header if channel_name is None else header[channel_name],
            fits.Header(),
            fits.Header(),
        ]
        header_list[1]["IMAGETYP"] = "error"
        header_list[2]["IMAGETYP"] = "mask"

        header_list[0]["BITPIX"] = header_list[1]["BITPIX"] = -32
        header_list[2]["BITPIX"] = 8

        logger.debug(
            "Slice for %s channel: %s", channel_name, repr(channel_slice)
        )

        hdu_list = fits.HDUList(
            [
                fits.PrimaryHDU(
                    numpy.array(image_list[0][channel_slice]), header_list[0]
                ),
                fits.ImageHDU(
                    numpy.array(image_list[1][channel_slice]), header_list[1]
                ),
                fits.ImageHDU(
                    numpy.array(image_list[2][channel_slice]).astype("uint8"),
                    header_list[2],
                ),
            ]
        )
        for hdu in hdu_list:
            hdu.update_header()

        logger.debug("Compression level: %s", repr(compress))
        if compress:
            logger.debug("Creating compressed HDU")
            hdu_list = fits.HDUList(
                [fits.PrimaryHDU()]
                + [
                    fits.CompImageHDU(
                        hdu.data, hdu.header, quantize_level=compress
                    )
                    for hdu in hdu_list
                ]
            )

        fname_substitutions.update(header_list[0])
        output_fname = result_fname.format_map(fname_substitutions)
        if not path.exists(path.dirname(output_fname)):
            makedirs(path.dirname(output_fname))
        hdu_list.writeto(output_fname, overwrite=allow_overwrite)
