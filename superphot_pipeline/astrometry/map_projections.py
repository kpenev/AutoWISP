"""Define projections from a sphere onto a plane."""

import numpy

def gnomonic_projection(sources, projected, **center):
    """
    Project the given sky position to a tangent plane (gnomonic projection).

    Args:
        sources(structured array-like):    The the sky position to project
            (should have `'RA'` and `'Dec'` keys coordinates in degrees.

        projected:    A numpy array with `'xi'` and `'eta'` fields to fill
            with the projected coordinates (in degrees).

        center(dict):    Should define the central `'RA'` and `'Dec'` around
            which to project.

    Returns:
        None
    """

    degree_to_rad = numpy.pi / 180.0
    center['RA'] *= degree_to_rad
    center['Dec'] *= degree_to_rad
    ra_diff = (sources['RA'] * degree_to_rad - center['RA'])
    cos_ra_diff = numpy.cos(ra_diff)
    cos_source_dec = numpy.cos(sources['Dec'] * degree_to_rad)
    cos_center_dec = numpy.cos(center['Dec'])
    sin_source_dec = numpy.sin(sources['Dec'] * degree_to_rad)
    sin_center_dec = numpy.sin(center['Dec'])
    denominator = (
        sin_center_dec * sin_source_dec
        +
        cos_center_dec * cos_source_dec * cos_ra_diff
    ) * degree_to_rad

    projected['xi'] = (cos_source_dec * numpy.sin(ra_diff)) / denominator

    projected['eta'] = (
        cos_center_dec * sin_source_dec
        -
        sin_center_dec * cos_source_dec * cos_ra_diff
    ) / denominator

tan_projection = gnomonic_projection
