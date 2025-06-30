"""General use convenience functions for working with FITS images."""

from os import path
import logging

from numpy import nan
from astropy.io import fits

from autowisp.pipeline_exceptions import BadImageError, ImageMismatchError
from autowisp.data_reduction.data_reduction_file import DataReductionFile

_logger = logging.getLogger(__name__)


def read_image_components(
    fits_fname,
    *,
    read_image=True,
    read_error=True,
    read_mask=True,
    read_header=True,
):
    """
    Read image, its error estimate, mask and header from pipeline FITS file.

    Args:
        fits_fname:    The filename of the FITS file to read the componets of.
            Must have been produced by the pipeline.

        read_image:    Should the pixel values of the primary image be read.

        read_error:    Should the error extension be searched for and read.

        read_mask:    Should the mask extension be searched for and read.

        read_header:    Should the header of the image extension be returned.

    Returns:
        (tuple):
            2-D array:
                The primary image in the file. Always present.

            2-D array:
                The error estimate of image, identified by
                ``IMAGETYP=='error'``. Set to None if none of the extensions
                have ``IMAGETYP=='error'``. This is omitted from the output if
                ``read_error == False``.

            2-D array:
                A bitmask of quality flags for each image pixel (identified by
                ``IMAGETYP='mask'``). Set to None if none of the extensions have
                ``IMAGETYP='mask'``. This is omitted from the output if
                ``read_mask == False``.

            astropy.io.fits.Header:
                The header of the image HDU in the file. This is omitted from
                the output if ``read_header == False``."""

    image = error = mask = header = None
    with fits.open(fits_fname, mode="readonly") as input_file:
        for hdu_index, hdu in enumerate(input_file):
            if hdu.header["NAXIS"] == 0:
                continue
            if image is None:
                image = hdu.data if read_image else True
                if read_header:
                    header = hdu.header
            else:
                if hdu.header["IMAGETYP"] == "error":
                    error = hdu.data
                elif hdu.header["IMAGETYP"] == "mask":
                    mask = hdu.data
                    if mask.dtype.itemsize != 1:
                        raise BadImageError(
                            f"Mask image (hdu #{hdu_index:d}) of {fits_fname} "
                            f"had data type {mask.dtype!s} (not int8)."
                        )
            if (
                image is not None
                and (error is not None or not read_error)
                and (mask is not None or not read_mask)
            ):
                break
    return (
        ((image,) if read_image else ())
        + ((error,) if read_error else ())
        + ((mask,) if read_mask else ())
        + ((header,) if read_header else ())
    )


def get_primary_header(fits_image, add_filename_keywords=False):
    """
    Return the primary header of the given image (filename or opened).

    Args:
        fits_image:    Either the filename or open FITS image to get the primary
            header of.

        add_filename_keywords:    If True appends to the header keywords parsed
            from the filename.

    Returns:
        fits.Header:
            The first header in the input file with non-zero NAXIS.
    """

    if not isinstance(fits_image, fits.HDUList):
        try:
            with fits.open(fits_image, "readonly") as opened_fits:
                return get_primary_header(opened_fits, add_filename_keywords)
        except OSError:
            with DataReductionFile(fits_image, "r") as dr_file:
                return dr_file.get_frame_header()
    for hdu in fits_image:
        if hdu.header["NAXIS"] != 0 or hdu.header.get("IMAGETYP") == "mphotref":
            result = hdu.header
            if add_filename_keywords:
                result = result.copy()
                base_fname = path.basename(fits_image.fileinfo(0)["filename"])
                for ext in [".fz", ".fits"]:
                    if base_fname.endswith(ext):
                        base_fname = base_fname[: -len(ext)]

                result["RAWFNAME"] = base_fname
            return result
    raise IOError(f"No valid HDU found in {fits_image!r}!")


def update_stack_header(master_header, frame_header, filename, first_time):
    """
    Update the master header per header from one of the individual frames.

    Should be called once for each frame participating in the stack with the
    second argument being the header of that frame. The first argument
    should initially be an empty FITS header, which will get updated each
    time.

    Args:
        master_header:    The header to use for the stacked master frame,
            describing the frames being stacked. On exit contains only
            keywords shared with frame header and only if their
            corresponding values match.

        frame_header:    The header of an individual frame being added to
            the stack.

        filename:    The filename where the header was read from. Only used
            for reporting errors.

        first_time:    Is this the first time this function is called for
            the current stack? Subsequent calls remove keywords that are
            discrepant between current header and the new frame being
            stacked.

    Returns:
        None
    """

    if first_time:
        master_header.extend(
            filter(lambda c: tuple(c) != ("", "", ""), frame_header.cards)
        )
    else:
        _logger.debug("Checking master header against %s", filename)

        mismatch_keys = {
            k
            for k in master_header
            if master_header[k] != frame_header.get(k, nan)
        } | (set(master_header) - set(frame_header))

        if "IMAGEYP" in mismatch_keys:
            raise ImageMismatchError(
                "Attempting to combine images with "
                f"IMAGETYP = {master_header['IMAGETYP']} and "
                f"IMAGETYP={frame_header['IMAGETYP']} "
                "into a master!"
            )

        _logger.debug("Deleting:\n%s", "\n".join(mismatch_keys))

        _logger.debug("Starting with %d cards", len(master_header.cards))
        for key in mismatch_keys:
            master_header.remove(key, remove_all=True)
        _logger.debug("%d cards remain", len(master_header.cards))
