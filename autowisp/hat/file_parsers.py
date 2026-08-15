"""Functions for parsing files generated with HAT tools."""

import re


def parse_fname_keywords(fits_fname):
    """
    Return the keywords defined in the given FITS or DR filename.

    Args:
        fits_fname:    The filename of a FITS frame to parse.

    Returns:
        fname_keywords:    A dictionary with the following contents:
            * STID (int): The ID of the station that acquired the image.

            * FNUM (int): The frame number.

            * CMPOS (int): The position index of the camera which acquired
                the image.

            * NIGHT (str): The night of when the image was observed. The
                format is YYYYmmdd and the date is set when observations
                start, so early morning frames get tagged with the
                previous date.
    """

    # pylint false positive
    # pylint: disable=anomalous-backslash-in-string
    frame_fname_rex = re.compile(
        "^.*/(?P<STID>[0-9]*)-(?P<NIGHT>[0-9]{8})/"
        "(?P=STID)-(?P<FNUM>[0-9]*)_(?P<CMPOS>[0-9]*)"
        "(_(?P<CHANNEL>[BGR][12]))?\.(fits(.fz)?|hdf5)?(.0)?$"
    )
    parsed_frame_fname = frame_fname_rex.match(fits_fname)
    assert parsed_frame_fname

    result = {
        keyword: (
            parsed_frame_fname.group(keyword)
            if keyword == "NIGHT"
            else int(parsed_frame_fname.group(keyword))
        )
        for keyword in ["STID", "FNUM", "CMPOS", "NIGHT"]
    }
    if parsed_frame_fname.group("CHANNEL") is not None:
        result["CHANNEL"] = parsed_frame_fname.group("CHANNEL")

    return result
