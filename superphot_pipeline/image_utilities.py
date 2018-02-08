"""A collection of functions for working with pipeline images."""

from os.path import exists
from astropy.io import fits
from astropy.coordinates import SkyCoord

import scipy
import scipy.interpolate

from superphot_pipeline.pipeline_exceptions import BadImageError

git_id = '$Id$'

def read_image_components(fits_fname,
                          *,
                          read_image=True,
                          read_error=True,
                          read_mask=True,
                          read_header=True):
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
            if hdu.header['NAXIS'] == 0:
                continue
            if image is None:
                image = hdu.data if read_image else True
                if read_header:
                    header = hdu.header
            else:
                if hdu.header['IMAGETYP'] == 'error':
                    error = hdu.data
                elif hdu.header['IMAGETYP'] == 'mask':
                    mask = hdu.data
                    if mask.dtype.itemsize != 1:
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
        ((image,) if read_image else ())
        +
        ((error,) if read_error else ())
        +
        ((mask,) if read_mask else())
        +
        ((header,) if read_header else())
    )

#pylint: disable=anomalous-backslash-in-string
#Triggers on doxygen commands.
def zoom_image(image, zoom, interp_order):
    """
    Increase the resolution of an image using flux conserving interpolation.

    Interpolation is performed using the following recipe:
        1.  create a cumulative image (C), i.e. C(x, y) = sum(
            image(x', y'), {x', 0, x}, {y', 0, y}). Note that C's x and y
            resolutions are both bigger than image's by one with all entries in
            the first row and the first column being zero.
        2.  Interpolate the cumulative image using a bivariate spline to get a
            continuous cumulative flux F(x, y).
        3.  Create the final image I by setting each pixel to the flux implied
            by F(x, y) from step 2, i.e. if zx is the zoom factor along x and zy
            is the zoom factor along y:

            \verbatim
                I(x, y) = F((x+1)/z, (y+1)/z)
                          - F((x+1)/z, y/z)
                          - F(x/z, (y+1)/z)
                          + F(x/z, y/z)
            \endverbatim

    Since this is a flux conserving method, zooming and then binning an image
    reproduces the original image with close to machine precision.

    Args:
        image:    The image to zoom.

        zoom:    The factor(s) by which to zoom the image. Should be either an
            integer defining a common zoom factor both dimensions or a pair of
            numbers, specifying the zoom along each axis (y first, then x).

        interp_order:    The order of the interpolation of the cumulative array.
    """

    try:
        x_zoom, y_zoom = zoom
    except TypeError:
        x_zoom = y_zoom = zoom

    if x_zoom == y_zoom == 1:
        return image

    y_res, x_res = image.shape
    cumulative_image = scipy.empty((y_res + 1, x_res + 1))
    cumulative_image[0, :] = 0
    cumulative_image[:, 0] = 0
    cumulative_image[1:, 1:] = scipy.cumsum(scipy.cumsum(image, axis=0), axis=1)

    try:
        spline_kx, spline_ky = interp_order
    except TypeError:
        spline_kx = spline_ky = interp_order

    cumulative_flux = scipy.interpolate.RectBivariateSpline(
        scipy.arange(y_res + 1),
        scipy.arange(x_res + 1),
        cumulative_image,
        kx=spline_kx,
        ky=spline_ky
    )

    cumulative_image = cumulative_flux(
        scipy.arange(y_res * y_zoom + 1) / y_zoom,
        scipy.arange(x_res * x_zoom + 1) / x_zoom,
        grid=True
    )
    return scipy.diff(scipy.diff(cumulative_image, axis=0), axis=1)
#pylint: enable=anomalous-backslash-in-string

def bin_image(image, bin_factor):
    """
    Bins the image to a lower resolution (must be exact factor of image shape).

    The output pixels are the sum of the pixels in each bin.

    Args:
        image:    The image to bin.

        bin_factor:    Either a single integer in which case this is the binning
            in both directions, or a pair of integers, specifying different
            binnin in each direction.

    Returns:
        binned_image:    The binned image with a resolution decreased by the
            binning factor for each axis, which has the same total flux as the
            input image.
    """

    try:
        x_bin_factor, y_bin_factor = bin_factor
    except TypeError:
        x_bin_factor = y_bin_factor = bin_factor

    if x_bin_factor == y_bin_factor == 1:
        return image

    y_res, x_res = image.shape

    assert x_res % x_bin_factor == 0
    assert y_res % y_bin_factor == 0

    return image.reshape((y_res // y_bin_factor,
                          y_bin_factor,
                          x_res // x_bin_factor,
                          x_bin_factor)).sum(-1).sum(1)

def get_pointing_from_header(frame):
    """
    Return the sky coordinates of this frame's pointing per its header.

    Args:
        frame:    The frame to return the pointing of. Could be in one of the
            following formats:
              * string: the filanema of a FITS frame. The pointing information
                  is extracted from the header of the first non-trivial HDU.
              * HDUList: Same as above, only this time the file is
                  already opened.
              * astropy.io.fits ImageHDU or TableHDU, containing the header to
                  extract the pointing information from.
              * asrtopy.io.fits.Header instance: the header from which to
                  extract the pointing information.

    Returns:
        pointing:    An instance of astropy.coordinates.SkyCoord containing the
            frame pointing information contained in the header.
    """

    try:
        if exists(frame):
            with fits.open(frame) as hdulist:
                return get_pointing_from_header(hdulist)
    except TypeError:
        pass

    if isinstance(frame, fits.HDUList):
        for hdu in frame:
            if hdu.data is not None:
                return get_pointing_from_header(hdu.header)
        raise BadImageError('FITS file '
                            +
                            repr(frame.filename)
                            +
                            ' contains only trivial HDUs')

    if hasattr(frame, 'header'):
        return get_pointing_from_header(frame.header)

    assert isinstance(frame, fits.Header)
    return SkyCoord(ra=frame['ra'] * 15.0, dec=frame['dec'], unit='deg')
