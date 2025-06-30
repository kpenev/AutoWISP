"""A collection of utilities for working with HAT headers."""


def get_jd(header):
    """Return the mid-exposure JD given a HAT header."""

    return 2.4e6 + header["JD"]
