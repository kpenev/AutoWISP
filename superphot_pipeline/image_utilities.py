"""A collection of functions for working with pipeline images."""

import os.path
import os
from glob import glob
import logging

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
from PIL import Image
import scipy
import scipy.interpolate

from superphot_pipeline.pipeline_exceptions import BadImageError

_logger = logging.getLogger(__name__)

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
        (tuple):
            2-D array:
                The primary image in the file. Always present.

            2-D array:
                The error estimate of image, identified by ``IMAGETYP=='error'``.
                Set to None if none of the extensions have
                ``IMAGETYP=='error'``. This is omitted from the output if
                ``read_error == False``.

            2-D array:
                A bitmask of quality flags for each image pixel (identified by
                ``IMAGETYP='mask'``). Set to None if none of the extensions have
                ``IMAGETYP='mask'``. This is omitted from the output if
                ``read_mask == False``.

            astropy.io.fits.Header:
                The header of the image HDU in the file. This is omitted from
                the output if ``read_header == False``.  """

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
            is the zoom factor along y::

                I(x, y) = F((x+1)/z, (y+1)/z)
                          - F((x+1)/z, y/z)
                          - F(x/z, (y+1)/z)
                          + F(x/z, y/z)

    Since this is a flux conserving method, zooming and then binning an image
    reproduces the original image with close to machine precision.

    Args:
        image:    The image to zoom.

        zoom:    The factor(s) by which to zoom the image. Should be either an
            integer defining a common zoom factor both dimensions or a pair of
            numbers, specifying the zoom along each axis (y first, then x).

        interp_order:    The order of the interpolation of the cumulative array.

    Returns:
        2-D array:
            The zoomed image.
    """

    try:
        x_zoom, y_zoom = zoom
    except TypeError:
        x_zoom = y_zoom = zoom

    if x_zoom == y_zoom == 1:
        return image

    y_res, x_res = image.shape
    #False positive
    #pylint: disable=no-member
    cumulative_image = scipy.empty((y_res + 1, x_res + 1))
    #pylint: enable=no-member
    cumulative_image[0, :] = 0
    cumulative_image[:, 0] = 0
    #False positive
    #pylint: disable=no-member
    cumulative_image[1:, 1:] = scipy.cumsum(scipy.cumsum(image, axis=0), axis=1)
    #pylint: enable=no-member

    try:
        spline_kx, spline_ky = interp_order
    except TypeError:
        spline_kx = spline_ky = interp_order

    cumulative_flux = scipy.interpolate.RectBivariateSpline(
        #False positive
        #pylint: disable=no-member
        scipy.arange(y_res + 1),
        scipy.arange(x_res + 1),
        #pylint: enable=no-member
        cumulative_image,
        kx=spline_kx,
        ky=spline_ky
    )

    cumulative_image = cumulative_flux(
        #False positive
        #pylint: disable=no-member
        scipy.arange(y_res * y_zoom + 1) / y_zoom,
        scipy.arange(x_res * x_zoom + 1) / x_zoom,
        #pylint: enable=no-member
        grid=True
    )

    #False positive
    #pylint: disable=no-member
    return scipy.diff(scipy.diff(cumulative_image, axis=0), axis=1)
    #pylint: enable=no-member
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
        2-D array:
            The binned image with a resolution decreased by the binning factor
            for each axis, which has the same total flux as the input image.
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
        astropy.coordinates.SkyCoord:
            The frame pointing information contained in the header.
    """

    try:
        if os.path.exists(frame):
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

def create_snapshot(fits_fname,
                    snapshot_fname_pattern,
                    *,
                    image_index=0,
                    overwrite=False,
                    skip_existing=False,
                    create_directories=True):
    """
    Create a snapshot (e.g. JPEG image) from a fits file in zscale.

    Args:
        fits_fname(str):    The FITS image to create a snapshot of.

        snapshot_fname_pattern(str):    A %-substitution pattern that when
            filled using the header and the extra keyword FITS_ROOT (set to the
            filename of the FITS file with path and extension removed) expands
            to the filename to save the snapshot as.

        image_index(int):    Offset from the first non-empty HDU in the FITS
            file to make a snapshot of.

        overwrite(bool):    If a file called `snapshot_fname` already exists, an
            `OSError` is raised if this argument and `skip_existing` are both
            False (default). That file is overwritten if this argument is True
            and `skip_existing` is False.

        skip_existing(bool):    If True and a file already exists with the name
            determined for the snapshot, this function exists immediately
            without error.

        create_directories(bool):    Whether the script is allowed to create
            the directories where the output snapshot will be stored. `OSError`
            is raised if this argument is False and the destination directory
            does not exist.

    Returns:
        None
    """

    with fits.open(fits_fname, 'readonly') as fits_image:
        #False positive
        #pylint: disable=no-member
        fits_hdu = fits_image[image_index if fits_image[0].header['NAXIS']
                              else image_index + 1]
        #pylint: enable=no-member
        snapshot_fname = (
            snapshot_fname_pattern
            %
            dict(
                fits_hdu.header,
                FITS_ROOT=os.path.splitext(os.path.basename(fits_fname))[0]
            )
        )

        if os.path.exists(snapshot_fname):
            if skip_existing:
                _logger.info('Snapshot %s already exists, skipping!',
                             repr(snapshot_fname))
                return
            if overwrite:
                _logger.info('Overwriting snapshot %s',
                             repr(snapshot_fname))
                os.remove(snapshot_fname)
            else:
                raise OSError(
                    'Failed to create FITS snapshot %s. File already exists!'
                    %
                    repr(snapshot_fname)
                )

        data = fits_hdu.data
        zscale_min, zscale_max = ZScaleInterval().get_limits(data)
        #False positive
        #pylint: disable=no-member
        scaled_data = (
            255
            *
            (
                scipy.minimum(scipy.maximum(zscale_min, data), zscale_max)
                -
                zscale_min
            ) / (
                zscale_max
                -
                zscale_min
            )
        ).astype(scipy.uint8)
        #pylint: enable=no-member

        snapshot_dir = os.path.dirname(snapshot_fname)
        if snapshot_dir and not os.path.exists(snapshot_dir):
            if create_directories:
                _logger.info('Creating snaphot directory: %s',
                             repr(snapshot_dir))
                os.makedirs(snapshot_dir)
            else:
                raise OSError(
                    'Output directory %s for saving snapshot %s does not exist'
                    %
                    (
                        repr(snapshot_dir),
                        repr(snapshot_fname)
                    )
                )

        Image.fromarray(scaled_data[::-1, :], 'L').save(snapshot_fname)
        _logger.debug('Creating snapshot: %s', repr(snapshot_fname))

def fits_image_generator(image_collection):
    """
    Iterate over input images specified directly or as directories.

    Args:
        image_collection(list):    Should include either fits images and/or
        directories. In the latter case, all files with `.fits` in their
        filename in the specified directory are included (sub-directories are
        not searched).

    Yields:
        The images specified in the `image_colleciton` argument.
    """

    for entry in image_collection:
        if os.path.isdir(entry):
            for fits_fname in sorted(
                    glob(
                        os.path.join(entry, '*.fits.fz*')
                    )
            ):
                yield fits_fname
        else:
            yield entry
