"""Functions for parsing files generated with HAT tools."""

import re

import scipy


def parse_transformation(filename):
    """
    Parse a transformation file suitable for grtrans.

    Args:
        filename(str):    The name of the file to parse.

    Returns:
        dict:
            All quantity defining the transformation, e.g. tape, order, scale,
            offset and coefficients for each transformed variable.

        dict:
            Information about the properties of the transformation contained in
            the comments.
    """

    transformation = {}
    info = {}
    with open(filename, "r", encoding="ascii") as trans_file:
        for line in trans_file:
            if line.strip()[0] == "#":
                split_line = line.strip().lstrip("#").strip().split(":", 1)
                if len(split_line) > 1:
                    quantity, value = split_line
                    if value:
                        info[quantity.strip().lower()] = value.strip()
            else:
                quantity, value = line.split("=")
                quantity = quantity.strip().lower()
                value = value.strip()
                if quantity == "order":
                    value = int(value)
                elif quantity == "scale":
                    value = float(value)
                elif quantity in ["offset", "basisshift"]:
                    value = tuple(float(v.strip()) for v in value.split(","))
                elif quantity != "type":
                    value = scipy.array(
                        [float(v.strip()) for v in value.split(",")]
                    )
                transformation[quantity] = value

    return transformation, info


def parse_anmatch_transformation(filename):
    """Parse transformation files generate by anmatch."""

    def parse_multivalue(info, value_parser, head="", tail=""):

        values, description = info.rstrip(")").split("(")
        if head:
            assert description.startswith(head)
            description = description[len(head) :]
        if tail:
            assert description.endswith(tail)
            description = description[: -len(tail)]

        keys = (k.strip() for k in description.strip().split(","))
        values = (value_parser(v) for v in values.strip().split())
        return dict(zip(keys, values))

    transformation, info = parse_transformation(filename)
    for info_key in ["residual", "unitarity"]:
        info[info_key] = float(info[info_key])

    info["points"] = parse_multivalue(info["points"], int, head="number of:")

    info["ratio"] = float(info["ratio"].split(None, 1)[0]) / 100.0

    info["timing"] = parse_multivalue(
        info["timing"], float, tail=": in seconds"
    )

    del info["all"]

    info["2mass"] = parse_multivalue(info["2mass"], float)
    for size_char in "wh":
        info["2mass"]["image" + size_char] = int(
            info["2mass"]["imag" + size_char]
        )
    return transformation, info


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
