"""Simple functions useful when interacing with HEALpix library."""

import numpy

from astropy.wcs import WCS
import healpy


def square_query_vertices(center, size, rotation=0.0):
    """
    Vertices defining a rectangular query to pass to healpy.query_polygon.

    Args:
        center:    The RA, Dec coordinates of the center of the query in
            degrees.

        size:    The full width/height of the rectangular region to query in
            degrees. If a single value, a square region is costructed.

        rotation:     The angle in degrees between the up direction in the query
            region and north. Going positive from zero corresponds to the up
            direction picking up a positve component along RA.

    Returns:
        array with shape (4, 3):
            The 4 unit vectors of the corners of the query on the unit sphere.
    """

    try:
        half_width = float(size / 2.0)
        half_height = half_width
    except TypeError:
        half_width, half_height = size / 2.0

    rotation_rad = rotation * numpy.pi / 180.0
    sin_rotation = numpy.sin(rotation_rad)
    cos_rotation = numpy.cos(rotation_rad)

    wcos = half_width * cos_rotation
    wsin = half_width * sin_rotation

    hcos = half_height * cos_rotation
    hsin = half_height * sin_rotation

    xi_eta_corners = numpy.empty((4, 2), dtype=float)
    xi_eta_corners[0] = wcos + hsin, -wsin + hcos
    xi_eta_corners[1] = -wcos + hsin, wsin + hcos
    xi_eta_corners[2] = -xi_eta_corners[0]
    xi_eta_corners[3] = -xi_eta_corners[1]

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.crval = center
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    ra_dec_corners = wcs.wcs_pix2world(xi_eta_corners, 1)
    print("RA, Dec of corners:\n" + repr(ra_dec_corners))
    return healpy.pixelfunc.ang2vec(*ra_dec_corners.transpose(), lonlat=True)
