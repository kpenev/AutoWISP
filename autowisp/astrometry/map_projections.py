"""Define projections from a sphere onto a plane."""

import numpy


def gnomonic_projection(sources, projected, **center):
    """
    Project the given sky position to a tangent plane (gnomonic projection).

    Args:
        sources(structured array-like):    The sky position to project
            (should have `'RA'` and `'Dec'` keys coordinates in degrees).

        projected:    A numpy array with `'xi'` and `'eta'` fields to fill
            with the projected coordinates (in degrees).

        center(dict):    Should define the central `'RA'` and `'Dec'` around
            which to project.

    Returns:
        None
    """

    degree_to_rad = numpy.pi / 180.0
    center["RA"] *= degree_to_rad
    center["Dec"] *= degree_to_rad
    ra_diff = sources["RA"] * degree_to_rad - center["RA"]
    cos_ra_diff = numpy.cos(ra_diff)
    cos_source_dec = numpy.cos(sources["Dec"] * degree_to_rad)
    cos_center_dec = numpy.cos(center["Dec"])
    sin_source_dec = numpy.sin(sources["Dec"] * degree_to_rad)
    sin_center_dec = numpy.sin(center["Dec"])
    denominator = (
        sin_center_dec * sin_source_dec
        + cos_center_dec * cos_source_dec * cos_ra_diff
    ) * degree_to_rad

    projected["xi"] = (cos_source_dec * numpy.sin(ra_diff)) / denominator

    projected["eta"] = (
        cos_center_dec * sin_source_dec
        - sin_center_dec * cos_source_dec * cos_ra_diff
    ) / denominator


def inverse_gnomonic_projection(sources, projected, **center):
    """
    Inverse projection from tangent plane (xi, eta) to sky position (RA, Dec)

    Args:
        sources: An empty numpy array with "RA" and "Dec" fields to fill
                 with the inverse projected of tangent plane coordinates.
                 (in degrees)

        projected: numpy array with "xi" and "eta" fields (in degrees)

        center(dict): Should define the central "RA" and "Dec" in degree

    Returns:
        None
    """

    degree_to_rad = numpy.pi / 180.0
    rad_to_degree = 180.0 / numpy.pi

    center["RA"] *= degree_to_rad
    center["Dec"] *= degree_to_rad

    projected["xi"] *= degree_to_rad
    projected["eta"] *= degree_to_rad

    rho = numpy.sqrt(projected["xi"] ** 2 + projected["eta"] ** 2)
    rho_angle = numpy.arctan(rho)
    denominator = rho * numpy.cos(center["Dec"]) * numpy.cos(
        rho_angle
    ) - projected["eta"] * numpy.sin(center["Dec"]) * numpy.sin(rho_angle)

    sources["RA"] = center["RA"] + numpy.arctan2(
        (projected["xi"] * numpy.sin(rho_angle)), denominator
    )

    sources["Dec"] = numpy.arcsin(
        numpy.cos(rho_angle) * numpy.sin(center["Dec"])
        + (projected["eta"] * numpy.sin(rho_angle) * numpy.cos(center["Dec"]))
        / rho
    )

    while (sources["RA"] < 0).any():
        sources["RA"][sources["RA"] < 0] += 2.0 * numpy.pi

    while (sources["RA"] > 2.0 * numpy.pi).any():
        sources["RA"][sources["RA"] > 2.0 * numpy.pi] -= 2.0 * numpy.pi

    sources["RA"] *= rad_to_degree
    sources["Dec"] *= rad_to_degree


tan_projection = gnomonic_projection
inverse_tan_projection = inverse_gnomonic_projection
