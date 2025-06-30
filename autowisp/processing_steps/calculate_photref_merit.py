#!/usr/bin/env python3
"""Sort DR files by their single photometric reference merit function."""

import logging

import numpy
import pandas

from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from autowisp.astrometry import Transformation
from autowisp.processing_steps.fit_source_extracted_psf_map import (
    get_predictors_and_weights,
)
from autowisp import DataReductionFile
from autowisp.file_utilities import find_dr_fnames
from autowisp.fit_expression import (
    Interface as FitTermsInterface,
    iterative_fit,
)
from autowisp.processing_steps.manual_util import ManualStepArgumentParser
from autowisp import Evaluator

_logger = logging.getLogger(__name__)
input_type = "dr"


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("" if args else input_type),
        inputs_help_extra="The DR files must already contain PSF fitting and "
        "smoothed source exaction PSF map.",
        add_component_versions=(
            "srcextract",
            "catalogue",
            "skytoframe",
            "background",
            "srcproj",
        ),
    )
    parser.add_argument(
        "--observatory-location",
        metavar=("LATITUDE", "LONGITUDE", "ALTITUDE"),
        default=["LAT_OBS", "LONG_OBS", "ALT_OBS"],
        nargs=3,
        help="The latitude, longitude and altitude of the observatory from "
        "where the images were collected. Can be arbitrary expression involving"
        " header keywords.",
    )

    parser.add_argument(
        "--bg-map-fit-terms-expression",
        "--bg-map-terms",
        default="O3{x, y}",
        help="An expression involving the x and y source coordinates for the "
        "terms to include when fitting a smooth function to the background "
        "measurements.",
    )
    parser.add_argument(
        "--bg-map-error-avg",
        default="median",
        help="How to average fitting residuals for outlier rejection during "
        "background smoothing.",
    )
    parser.add_argument(
        "--bg-map-rej-level",
        type=float,
        default=5.0,
        help="How far away from the fit should a point be before it is rejected"
        " in units of error_avg.",
    )
    parser.add_argument(
        "--bg-map-max-rej-iter",
        type=int,
        default=10,
        help="The maximum number of outlier rejection/refit iterations allowed "
        "when fitting for the smooth background of an image.",
    )
    parser.add_argument(
        "--merit-function",
        default="1.0 / ((1.0 - qnt_s)**2 + qnt_bg**2)",
        help="The merit function to use. High values should indicate a better "
        "candidate to serve as photometric reference. The function may use any "
        "PSF parameter as well as the following frame properties:\n"
        "\tz: the zenith distance of the frame center\n"
        "\tbg: the background level at the center of the frame\n"
        "In addition, for each property the standard deviation over all frames "
        "can also be used as ``std_<param>`` (e.g. ``std_z``) as well as its "
        "quantile as ``qnt_<param>`` (e.g. ``qnt_bg`` for the background "
        "quantile).",
    )

    return parser.parse_args(*args)


def get_matched_sources(dr_fname, dr_path_substitutions):
    """Convenience wrapper around `DataReductionFile.get_matched_sources()`."""

    with DataReductionFile(dr_fname, "r") as dr_file:
        return dr_file.get_matched_sources(**dr_path_substitutions)


def get_typical_star(
    dr_fnames,
    source_average="median",
    frame_average="median",
    **dr_path_substitutions,
):
    """
    Return the average of the matched sources in all DR files.

    For each DR file the properties of the matched stars (i.e. everything from
    the catalog and source extraction) are averaged and then another average is
    computed over the DR files.

    Args:
        dr_fnames([str]):    The filenames of the DR files to get the typical
            star for.

        source_average('mean'|'median'):    How to avarage source properties
            within a single DR file.

        frame_average('mean'|'median'):    How the average accross DR files the
            the already averaged per DR file properties

        dr_path_substitutions(dict):    Substitutions required to uniquely
            identify the datasets within the DR file to process.

    Returns:
        pandas.Series:
            The double averaged matched star properties.
    """

    source_averaged = pandas.DataFrame(
        getattr(
            get_matched_sources(fname, dr_path_substitutions), source_average
        )()
        for fname in dr_fnames
    )
    return getattr(source_averaged, frame_average)()


def get_center_zenith_distance(header, astrometry, config):
    """Return the zenith distance of the center of the frame."""

    header_eval = Evaluator(header)
    latitude, longitude, altitude = (
        header_eval(expression) for expression in config["observatory_location"]
    )

    location = EarthLocation(
        lat=latitude * units.deg,
        lon=longitude * units.deg,
        height=altitude * units.m,
    )
    obs_time = Time(header["JD-OBS"], format="jd", location=location)
    source_coords = SkyCoord(
        ra=astrometry.pre_projection_center[0] * units.deg,
        dec=astrometry.pre_projection_center[1] * units.deg,
        frame="icrs",
    )
    altitude = source_coords.transform_to(
        AltAz(obstime=obs_time, location=location)
    ).alt.to_value(units.deg)
    return 90.0 - altitude


def get_center_background(
    dr_file,
    header,
    fit_terms_expression,
    *,
    error_avg,
    rej_level,
    max_rej_iter,
    **dr_path_substitutions,
):
    """
    Estimate the sky background at the center of the frame.

    Fit a smooth function of position to the background measurements from shape
    fitting and evaluate it at the center of the frame.
    """

    source_positions = {
        coord: dr_file.get_dataset(
            "srcproj.columns",
            srcproj_column_name=coord,
            **dr_path_substitutions,
        )
        for coord in "xy"
    }
    source_positions["x"] -= header["NAXIS1"] / 2
    source_positions["y"] -= header["NAXIS2"] / 2

    print("Source positions: " + repr(source_positions))

    fit_terms = FitTermsInterface(fit_terms_expression)(source_positions)
    measured_bg = dr_file.get_dataset("bg.value", **dr_path_substitutions)
    coef, square_residual, num_fit = iterative_fit(
        fit_terms,
        measured_bg,
        error_avg=error_avg,
        rej_level=rej_level,
        max_rej_iter=max_rej_iter,
        fit_identifier="background",
    )
    _logger.debug(
        "Background fit:\ncoefficientsn: %s\nsquare residual: %si\nnum fit: %s",
        repr(coef),
        repr(square_residual),
        repr(num_fit),
    )
    return coef[0]


def get_frame_merit_info(
    dr_fname, typical_star, bg_fit_config, config, **dr_path_substitutions
):
    """Return the properties relevant for calculating the merit of given DR."""

    with DataReductionFile(dr_fname, "r") as dr_file:
        header = dr_file.get_frame_header()
        astrometry = Transformation()
        astrometry.read_transformation(dr_file, **dr_path_substitutions)
        result = {
            "dr": dr_fname,
            "z": get_center_zenith_distance(header, astrometry, config),
        }

        typical_star["RA"] = astrometry.pre_projection_center[0]
        typical_star["Dec"] = astrometry.pre_projection_center[1]
        typical_star["x"] = header["NAXIS1"] / 2
        typical_star["y"] = header["NAXIS2"] / 2

        psf_map_predictors = get_predictors_and_weights(
            pandas.DataFrame([typical_star]),
            dr_file.get_attribute(
                "srcextract.psf_map.cfg.terms", **dr_path_substitutions
            ),
            None,
        )[0]
        psf_map_coefficients = dr_file.get_dataset(
            "srcextract.psf_map", **dr_path_substitutions
        )
        for psf_param, map_coef in zip(
            dr_file.get_attribute(
                "srcextract.psf_map.cfg.psf_params", **dr_path_substitutions
            ),
            psf_map_coefficients,
        ):
            result[psf_param.decode()] = float(
                numpy.dot(map_coef, psf_map_predictors)
            )
        result["bg"] = get_center_background(
            dr_file, header, **bg_fit_config, **dr_path_substitutions
        )

    return result


def calculate_photref_merit(dr_filenames, config):
    """Avoid pollutin global namespace."""

    dr_path_substitutions = {
        what + "_version": config[what + "_version"]
        for what in [
            "srcextract",
            "catalogue",
            "skytoframe",
            "background",
            "srcproj",
        ]
    }
    bg_fit_config = {
        argname[len("bg_map_") :]: value
        for argname, value in config.items()
        if argname.startswith("bg_map_")
    }
    typical_star = get_typical_star(dr_filenames, **dr_path_substitutions)
    _logger.debug("Typical star:\n%s", repr(typical_star))
    merit_info = pandas.DataFrame(
        get_frame_merit_info(
            dr_fname,
            typical_star,
            bg_fit_config,
            config,
            **dr_path_substitutions,
        )
        for dr_fname in dr_filenames
    )
    frame_quantities = list(merit_info.columns)
    frame_quantities.remove("dr")
    for column in frame_quantities:
        merit_info["qnt_" + column] = merit_info[column].rank(pct=True)

    eval_merit = Evaluator(merit_info)
    for column in frame_quantities:
        eval_merit.symtable["std_" + column] = merit_info[column].std()
    merit_info["merit"] = eval_merit(config["merit_function"])
    return merit_info


if __name__ == "__main__":
    cmdline_config = parse_command_line()
    _logger.info(
        "Merit info:\n%s",
        repr(
            calculate_photref_merit(
                list(find_dr_fnames(cmdline_config.pop("dr_files"))),
                cmdline_config,
            ).sort_values(by="merit", ascending=False)
        ),
    )
