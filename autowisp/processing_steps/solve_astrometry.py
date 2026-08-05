#!/usr/bin/env python3

# pylint: disable=too-many-lines

"""Fit for a transformation between sky and image coordinates."""
import logging
from contextlib import ExitStack
from multiprocessing import Queue, Process, Lock
from traceback import format_exc
import os
from os import getpid
import getpass

import numpy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units

from autowisp.multiprocessing_util import setup_process
from autowisp.error_context import (
    capture_for_queue,
    decode_exit_signals,
    error_context,
    reraise_from_worker,
    forbid_nested_workers,
)
from autowisp.error_cli import cli_entry_point
from autowisp.exceptions import (
    Component,
    FileKind,
    RelatedFile,
    WorkerCrashedError,
)
from autowisp.processing_steps.manual_util import (
    ManualStepArgumentParser,
    ignore_progress,
)
from autowisp.file_utilities import find_dr_fnames
from autowisp.astrometry import (
    estimate_transformation,
    refine_transformation,
    Transformation,
)
from autowisp.astrometry.transformation import compute_diagonal_fov
from autowisp.catalog import (
    ensure_catalog,
    check_catalog_coverage,
    get_catalog_config,
)
from sqlalchemy import select

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.evaluator import Evaluator
from autowisp.database.interface import start_db_session

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import ObservingSession, Target
from autowisp.database.data_model.provenance import Observatory

# pylint: enable=no-name-in-module

_logger = logging.getLogger(__name__)

input_type = "dr"
fail_reasons = {
    "failed to converge": -2,
    "few matched": -3,
    "high rms": -4,
    "solve-field failed": -5,
    "web solve failed": -6,
    "other": -7,
}


def add_anet_cmdline_args(parser):
    """Add to parser all command line arguments needed to run astrometry.net."""

    parser.add_argument(
        "--anet-indices",
        nargs=2,
        env_var="AUTOWISP_ANET_INDICES",
        default=(
            (
                rf"C:\Users\{getpass.getuser()}"
                rf"\AppData\Local\cygwin_ansvr"
                rf"\usr\share\astrometry\data\narrow",
                rf"C:\Users\{getpass.getuser()}"
                rf"\AppData\Local\cygwin_ansvr\usr\share\astrometry\data\wide",
            )
            if os.name == "nt"
            else ("/anet_indices/narrow", "/anet_indices/wide")
        ),
        help="Full paths to the narrow and wide astometry.net index files. If "
        "these directories are not found, the web solver is used instead.",
    )
    parser.add_argument(
        "--anet-api-key",
        default=None,
        help="The astrometry.net API key to use if web solver is used to find "
        "initial match to catalog. You can get it by signing in to the web "
        "service and selecting ``Profile``.",
    )
    parser.add_argument(
        "--frame-fov-estimate",
        nargs=2,
        type=str,
        default=("10.0 * units.deg", "15.0 * units.deg"),
        metavar=("WIDTH", "HEIGHT"),
        help="Approximate field of view of the frame in degrees. Can be an "
        "expression involving header keywords. If not specified, the field of "
        "view of the catalog divided by ``--image-scale-factor`` is used.",
    )
    parser.add_argument(
        "--tweak-order",
        type=int,
        nargs=2,
        default=(2, 5),
        help="Range of tweak arguments to solve-field to try.",
    )
    parser.add_argument(
        "--image-scale-factor",
        type=float,
        default=1.3,
        help="Astrometry.net solution is searched with scale ranging from "
        "fov/image_scale_factor to fov * image_scale_factor.",
    )


def parse_command_line(*args):
    """Return the parsed command line arguments."""

    parser = ManualStepArgumentParser(
        description=__doc__,
        input_type=("" if args else input_type),
        inputs_help_extra="The DR files must already contain extracted sources",
        add_catalog={"prefix": "astrometry"},
        add_component_versions=("srcextract", "catalogue", "skytoframe"),
        allow_parallel_processing=True,
    )
    parser.add_argument(
        "--reuse-transformation-key",
        default="RAWFNAME",
        help="Expression involving  header keywords that results in unique "
        "value for each group of frames representing separate channels of the "
        "same raw image.",
    )
    parser.add_argument(
        "--astrometry-only-if",
        default="True",
        help="Expression involving the header of the input images that "
        "evaluates to True/False if a particular image from the specified "
        "image collection should/should not be processed.",
    )
    parser.add_argument(
        "--max-srcmatch-distance",
        type=float,
        default=1.0,
        help="The maximum distance between a projected and extracted source "
        "center before we declare the two could not possibly correspond to the "
        "same star. Determining a good value to use depends on the properties "
        "of the images being processed. It should be much larger than the "
        "uncertainty with which source extraction determines potiions but much "
        "smaller than  the typical discance between stars.",
    )
    parser.add_argument(
        "--astrometry-order",
        type=int,
        default=5,
        help="The order of the transformation to fit (i.e. the maximum combined"
        " power of the cartographically projected coordinates each of the "
        "frame coordinates is allowed to depend on. The best value to use "
        "depends on the properties of the images being processed. It needs to "
        "be large enough to capture the distortion in the image away from "
        "gnomonic projection, yet small enough to have many fewer parameters "
        "than the easily detectable stars in the image. Typically, small field "
        "of view images have smaller distortions and a lower value should be "
        "used compared to wide-field images.",
    )
    parser.add_argument(
        "--min-source-safety-factor",
        "--src-safety-factor",
        type=float,
        default=5.0,
        help="When solving for transformation coefficients the number of "
        "matched stars must exceed the number of free parameters by at least "
        "this factor.",
    )
    parser.add_argument(
        "--trans-threshold",
        type=float,
        default=0.005,
        help="The threshold for the difference of two consecutive "
        "transformations",
    )
    parser.add_argument(
        "--max-astrom-iter",
        type=int,
        default=20,
        help="The maximum number of iterations the astrometry solution can "
        "pass.",
    )
    parser.add_argument(
        "--min-match-fraction",
        type=float,
        default=0.8,
        help="The minimum fraction of extracted sources that must be matched to"
        " a catalog soure for the solution to be considered valid. It should be"
        " be less than 1 only to account for possible spurious stars found "
        "during source extraction. Having a stringent (close to 1) requirement "
        "ensures that the astrometry fails for images that are not of high "
        "quality (thin clouds, poor tracking, etc.)",
    )
    parser.add_argument(
        "--max-rms-distance",
        type=float,
        default=0.5,
        help="The maximum RMS distance between projected and extracted "
        "positions for the astrometry solution to be considered valid. Should "
        "be slightly larger than the typical uncertainty with which source "
        "extraction determines the centroids of the stars.",
    )
    add_anet_cmdline_args(parser)
    result = parser.parse_args(*args)
    if result["astrometry_catalog_filter"] is not None:
        result["astrometry_catalog_filter"] = dict(
            result["astrometry_catalog_filter"]
        )
    return result


def print_file_contents(fname, label):
    """Print the entire contenst of the given file."""

    print(80 * "*")
    print(label.title() + ": ")
    print(80 * "-")
    with open(fname, "r", encoding="utf-8") as open_file:
        print(open_file.read())
    print(80 * "-")


def save_trans_to_dr(  # pylint: disable=too-many-arguments
    *,
    trans_x,
    trans_y,
    ra_cent,
    dec_cent,
    res_rms,
    configuration,
    header,
    dr_file,
    **path_substitutions,
):
    """Save the transformation to the DR file."""

    terms_expression = f'O{configuration["astrometry_order"]:d}{{xi, eta}}'

    dr_file.add_dataset(
        dataset_key="skytoframe.coefficients",
        data=numpy.stack((trans_x.flatten(), trans_y.flatten())),
        **path_substitutions,
    )
    dr_file.add_attribute(
        attribute_key="skytoframe.type",
        attribute_value="polynomial",
        **path_substitutions,
    )
    dr_file.add_attribute(
        attribute_key="skytoframe.terms",
        attribute_value=terms_expression,
        **path_substitutions,
    )
    dr_file.add_attribute(
        attribute_key="skytoframe.sky_center",
        attribute_value=numpy.array([ra_cent, dec_cent]),
        **path_substitutions,
    )
    # TODO: need to add and figure out unitarity
    # for entry in ['residual', 'unitarity']:
    #     dr_file.add_attribute(
    #         attribute_key='skytoframe.' + entry,
    #         attribute_value=res_rms,
    #         **path_substitutions
    #     )
    dr_file.add_attribute(
        attribute_key="skytoframe.residual",
        attribute_value=res_rms,
        **path_substitutions,
    )
    for component, config_attribute in [
        ("srcextract", "binning"),
        ("skytoframe", "srcextract_filter"),
        ("skytoframe", "sky_preprojection"),
        ("skytoframe", "max_match_distance"),
        ("skytoframe", "frame_center"),
        ("skytoframe", "weights_expression"),
    ]:
        if config_attribute == "max_match_distance":
            value = configuration["max_srcmatch_distance"]
        elif config_attribute == "frame_center":
            value = (header["NAXIS1"] / 2.0, header["NAXIS2"] / 2.0)
        else:
            value = configuration[config_attribute]
        dr_file.add_attribute(
            component + ".cfg." + config_attribute, value, **path_substitutions
        )


# TODO: Add catalog query configuration to DR
def save_to_dr(  # pylint: disable=too-many-arguments
    *,
    cat_extracted_corr,
    trans_x,
    trans_y,
    ra_cent,
    dec_cent,
    res_rms,
    configuration,
    header,
    dr_file,
    catalog,
):
    """Save the solved astrometry to the given DR file."""

    path_substitutions = {
        substitution: configuration[substitution]
        for substitution in [
            "srcextract_version",
            "catalogue_version",
            "skytoframe_version",
        ]
    }

    dr_file.add_sources(
        catalog,
        "catalogue.columns",
        "catalogue_column_name",
        parse_ids=False,
        **path_substitutions,
    )
    dr_file.add_dataset(
        dataset_key="skytoframe.matched",
        data=cat_extracted_corr,
        **path_substitutions,
    )
    save_trans_to_dr(
        trans_x=trans_x,
        trans_y=trans_y,
        ra_cent=ra_cent,
        dec_cent=dec_cent,
        res_rms=res_rms,
        configuration=configuration,
        header=header,
        dr_file=dr_file,
        **path_substitutions,
    )


def transformation_to_raw(trans_x, trans_y, header, in_place=False):
    """Convert the transformation coefficients to pre-channel split coords."""

    if not in_place:
        trans_x = numpy.copy(trans_x)
        trans_y = numpy.copy(trans_y)
    trans_x[0] += header["CHNLXOFF"]
    trans_x *= header["CHNLXSTP"]
    trans_y[0] += header["CHNLYOFF"]
    trans_y *= header["CHNLYSTP"]
    return trans_x, trans_y


def transformation_from_raw(trans_x, trans_y, header, in_place=False):
    """Convert the transformation coefficients from pre-channel split coords."""

    if not in_place:
        trans_x = numpy.copy(trans_x)
        trans_y = numpy.copy(trans_y)

    trans_x /= header["CHNLXSTP"]
    trans_x[0] -= header["CHNLXOFF"]

    trans_y /= header["CHNLYSTP"]
    trans_y[0] -= header["CHNLYOFF"]

    return trans_x, trans_y


def construct_transformation(transformation_info):
    """Construct a `Transformation` object from the given information."""

    transformation = Transformation()

    transformation_order = numpy.rint(
        (numpy.sqrt(1.0 + 8.0 * transformation_info["trans_x"].size) - 3.0)
        / 2.0
    ).astype(int)
    transformation.set_transformation(
        pre_projection_center=(
            transformation_info["ra_cent"],
            transformation_info["dec_cent"],
        ),
        terms_expression=f"O{transformation_order:d}{{xi, eta}}",
        coefficients=(
            transformation_info["trans_x"].flatten(),
            transformation_info["trans_y"].flatten(),
        ),
    )
    return transformation


def find_final_transformation(  # pylint: disable=too-many-locals
    header, transformation_estimate, xy_extracted, web_lock, configuration
):
    """Find the final transformation for a given image."""

    _logger.debug(
        "Using transformation estimate: %s", repr(transformation_estimate)
    )

    frame_is_covered = False
    project_to_frame = construct_transformation(transformation_estimate)
    iteration = 0
    # Each pass may resolve a *different* catalog as the transformation is
    # refined, and the coverage failure below is precisely a question of
    # which ones were tried -- so the scopes accumulate rather than
    # replacing one another, and every catalog used is named on the error.
    # Repeats (the same catalog resolved twice) are merged out when the
    # error is stamped.
    with ExitStack() as catalog_scopes:
        while not frame_is_covered:
            (catalog, catalog_header), _, catalog_fname = ensure_catalog(
                transformation=project_to_frame,
                header=header,
                configuration=get_catalog_config(configuration, "astrometry"),
                lock=web_lock,
            )
            catalog_scopes.enter_context(
                error_context(
                    related_files=[
                        RelatedFile(
                            FileKind.CATALOG, catalog_fname, role="input"
                        )
                    ]
                )
            )

            (
                transformation_estimate["trans_x"],
                transformation_estimate["trans_y"],
                cat_extracted_corr,
                res_rms,
                ratio,
                transformation_estimate["ra_cent"],
                transformation_estimate["dec_cent"],
                success,
            ) = refine_transformation(
                xy_extracted=xy_extracted,
                catalog=catalog,
                x_frame=header["NAXIS1"],
                y_frame=header["NAXIS2"],
                astrometry_order=configuration["astrometry_order"],
                max_srcmatch_distance=configuration["max_srcmatch_distance"],
                max_iterations=configuration["max_astrom_iter"],
                trans_threshold=configuration["trans_threshold"],
                min_source_safety_factor=configuration[
                    "min_source_safety_factor"
                ],
                **transformation_estimate,
            )
            project_to_frame = construct_transformation(transformation_estimate)
            frame_is_covered = check_catalog_coverage(
                header,
                project_to_frame,
                catalog_header,
                configuration["astrometry_catalog_fov_safety_margin"],
            )
            iteration += 1
            if iteration > 10:
                raise RuntimeError(
                    "Attempting to ensure catalog coverage seems to be in an "
                    "infinite loop!"
                )

    return (
        transformation_estimate,
        catalog,
        cat_extracted_corr,
        {"rms": res_rms, "ratio": ratio, "success": success},
    )


def get_xy_extracted(dr_file, srcextract_version):
    """Return array of extracted source positions."""

    sources = dr_file.get_sources(
        "srcextract.sources",
        "srcextract_column_name",
        srcextract_version=srcextract_version,
    )
    xy_extracted = numpy.zeros(
        (len(sources["x"].values)), dtype=[("x", ">f8"), ("y", ">f8")]
    )
    xy_extracted["x"] = sources["x"].values
    xy_extracted["y"] = sources["y"].values
    return xy_extracted


def _compute_zenith_distance(header, ra_cent, dec_cent):
    """Return the zenith distance of the image center (None if unavailable)."""

    try:
        with start_db_session() as db_session:
            observatory = db_session.scalar(
                select(Observatory)
                .join(ObservingSession)
                .where(ObservingSession.label == header["OBSSSNID"])
            )
        location = EarthLocation(
            lat=observatory.latitude * units.deg,
            lon=observatory.longitude * units.deg,
            height=observatory.altitude * units.m,
        )
        obs_time = Time(header["JD-OBS"], format="jd", location=location)
        source_coords = SkyCoord(
            ra=ra_cent * units.deg,
            dec=dec_cent * units.deg,
            frame="icrs",
        )
        alt = source_coords.transform_to(
            AltAz(obstime=obs_time, location=location)
        ).alt.to_value(units.deg)
        return 90.0 - alt
    except (KeyError, Exception):
        _logger.error("Cannot compute zenith distance.", exc_info=True)
        return None


def _compute_pointing_offset(header, ra_cent, dec_cent):
    """Return angular distance in degrees between target and image center."""

    try:
        with start_db_session() as db_session:
            target = db_session.scalar(
                select(Target)
                .join(ObservingSession)
                .where(ObservingSession.label == header["OBSSSNID"])
            )
        if target.ra is None or target.dec is None:
            return None
        center = SkyCoord(
            ra=ra_cent * units.deg,
            dec=dec_cent * units.deg,
            frame="icrs",
        )
        target_coords = SkyCoord(
            ra=target.ra * units.deg,
            dec=target.dec * units.deg,
            frame="icrs",
        )
        return center.separation(target_coords).to_value(units.deg)
    except (KeyError, Exception):
        _logger.error("Cannot compute pointing offset.", exc_info=True)
        return None


def _compute_mag_zeropt(catalog, cat_extracted_corr, dr_file, configuration):
    """Return the source extraction magnitude zeropoint, or None."""

    try:
        sources = dr_file.get_sources(
            "srcextract.sources",
            "srcextract_column_name",
            srcextract_version=configuration["srcextract_version"],
        )
        cat_mags = catalog["magnitude"].iloc[cat_extracted_corr[:, 0]].values
        ext_flux = sources["flux"].values[cat_extracted_corr[:, 1]]
        positive = ext_flux > 0
        if not positive.any():
            return None
        return float(
            numpy.median(
                cat_mags[positive] + 2.5 * numpy.log10(ext_flux[positive])
            )
        )
    except (KeyError, IndexError):
        _logger.error("Cannot compute magnitude zeropoint.", exc_info=True)
        return None


def _collect_astrometry_diagnostics(
    transformation_estimate, solve_diagnostics, header
):
    """Return a list of (name, value) diagnostic tuples."""

    result = [
        ("ra_center", transformation_estimate["ra_cent"]),
        ("dec_center", transformation_estimate["dec_cent"]),
        ("matched_fraction", solve_diagnostics["ratio"]),
        ("astrom_residual", solve_diagnostics["rms"]),
        (
            "diagonal_fov",
            compute_diagonal_fov(
                construct_transformation(transformation_estimate), header
            ),
        ),
    ]

    z_center = _compute_zenith_distance(
        header,
        transformation_estimate["ra_cent"],
        transformation_estimate["dec_cent"],
    )
    if z_center is not None:
        result.append(("z_center", z_center))

    pointing_offset = _compute_pointing_offset(
        header,
        transformation_estimate["ra_cent"],
        transformation_estimate["dec_cent"],
    )
    if pointing_offset is not None:
        result.append(("pointing_offset", pointing_offset))

    if "mag_zeropt" in solve_diagnostics:
        result.append(
            ("srcextract_mag_zeropt", solve_diagnostics["mag_zeropt"])
        )

    return result


def solve_image(  # pylint: disable=too-many-locals
    dr_fname,
    transformation_estimate=None,
    *,
    web_lock,
    mark_start,
    mark_end,
    **configuration,
):
    """
    Find the astrometric transformation for a single image and save to DR file.

    Args:
        dr_fname(str):    The name of the data reduction file containing the
            extracted sources from the frame and that will be updated with the
            newly solved astrometry.

        transformation_estimate(None or (matrix, matrix)):    Estimate of the
            transformations x(xi, eta) and y(xi, eta) for the raw frame (i.e.
            before channel splitting) that will be refined. If ``None``,
            ``solve_field`` from astrometry.net is used to find iniitial
            estimates.

        web_lock(multiprocessing.Lock):    A lock held while submitting a
            plate-solve request to the astrometry.net web API.

        mark_start(callable):    Called before anything is written to the DR
            file if successful transformation is found.

        mark_end(callable):    Called after a successful transformation is fully
            written to the DR file.

        configuration:    Parameters defining how astrometry is to be fit.

    Returns:
        trans_x(2D numpy array):
            the coefficients of the x(xi, eta) transformation converted to RAW
            image coordinates (i.e. before channel splitting)

        trans_y(2D numpy array):
            the coefficients of the y(xi, eta) transformation converted to RAW
            image coordinates (i.e. before channel splitting)

        ra_cent(float): the RA center around which the above transformation
            applies

        dec_cent(float): the Dec center around which the above transformation
            applies
    """

    _logger.debug(
        "Solving: %s %s transformation estimate.",
        repr(dr_fname),
        ("with" if transformation_estimate else "without"),
    )
    with DataReductionFile(dr_fname, "r+") as dr_file:

        header = dr_file.get_frame_header()
        dr_eval = Evaluator(header)

        configuration = prepare_configuration(configuration, header)

        result = {
            "dr_fname": dr_fname,
            "trans_key": dr_eval(configuration["reuse_transformation_key"]),
            "saved": False,
        }

        fov_estimate = max(*configuration["frame_fov_estimate"]).to_value("deg")

        xy_extracted = get_xy_extracted(
            dr_file, configuration["srcextract_version"]
        )
        if transformation_estimate is None:
            transformation_estimate, status = estimate_transformation(
                dr_file=dr_file,
                xy_extracted=xy_extracted,
                config={
                    "astrometry_order": configuration["astrometry_order"],
                    "tweak_order_range": configuration["tweak_order"],
                    "fov_range": (
                        fov_estimate / configuration["image_scale_factor"],
                        fov_estimate * configuration["image_scale_factor"],
                    ),
                    "anet_indices": [
                        fname.format_map(header)
                        for fname in configuration["anet_indices"]
                    ],
                    "anet_api_key": configuration["anet_api_key"],
                    "x_cent": header["NAXIS1"] / 2,
                    "y_cent": header["NAXIS2"] / 2,
                },
                header=header,
                web_lock=web_lock,
            )
            if status != "success":
                result["fail_reason"] = fail_reasons.get(
                    status, fail_reasons["other"]
                )
                return result

        else:
            (
                transformation_estimate["trans_x"],
                transformation_estimate["trans_y"],
            ) = transformation_from_raw(
                transformation_estimate["trans_x"],
                transformation_estimate["trans_y"],
                header,
                True,
            )

        try:
            (
                transformation_estimate,
                catalog,
                cat_extracted_corr,
                diagnostics,
            ) = find_final_transformation(
                header,
                transformation_estimate,
                xy_extracted,
                web_lock,
                configuration,
            )
            diagnostics["mag_zeropt"] = _compute_mag_zeropt(
                catalog, cat_extracted_corr, dr_file, configuration
            )
        # pylint: disable=bare-except
        except:
            _logger.critical(
                "Failed to find solution to DR file %s:\n%s",
                dr_fname,
                format_exc(),
            )
            result["fail_reason"] = fail_reasons["other"]
            return result
        # pylint: enable=bare-except

        try:
            _logger.debug("RMS residual: %s", repr(diagnostics["rms"]))
            _logger.debug("Ratio: %s", repr(diagnostics["ratio"]))

            if diagnostics["ratio"] < configuration["min_match_fraction"]:
                result["fail_reason"] = fail_reasons["few matched"]
            elif diagnostics["rms"] > configuration["max_rms_distance"]:
                result["fail_reason"] = fail_reasons["high rms"]
            elif not diagnostics["success"]:
                result["fail_reason"] = fail_reasons["failed to converge"]
            else:
                _logger.info(
                    "Succesful astrometry solution found for %s:", dr_fname
                )
                mark_start(dr_fname)
                save_to_dr(
                    cat_extracted_corr=cat_extracted_corr,
                    **transformation_estimate,
                    res_rms=diagnostics["rms"],
                    configuration=configuration,
                    header=header,
                    dr_file=dr_file,
                    catalog=catalog,
                )
                mark_end(
                    dr_fname,
                    diagnostics=_collect_astrometry_diagnostics(
                        transformation_estimate,
                        diagnostics,
                        header,
                    ),
                )
                result["saved"] = True

                transformation_to_raw(
                    transformation_estimate["trans_x"],
                    transformation_estimate["trans_y"],
                    header,
                    True,
                )
                result["raw_transformation"] = transformation_estimate
            return result

        # pylint: disable=bare-except
        except:
            _logger.critical(
                "Failed to save found astrometry solution to "
                "DR file %s:\n%s",
                dr_fname,
                format_exc(),
            )
            return result
        # pylint: enable=bare-except

    result["fail_reason"] = fail_reasons["solve-field failed"]
    _logger.error(
        "No Astrometry.net solution found in tweak range [%d, %d]",
        *configuration["tweak_order"],
    )
    return result


def astrometry_process(  # pylint: disable=too-many-arguments
    task_queue,
    result_queue,
    *,
    configuration,
    web_lock,
    mark_start,
    mark_end,
):
    """Run pending astrometry tasks from the queue in process."""

    setup_process(task="solve", **configuration)
    _logger.info("Starting astrometry solving process.")
    for dr_fname, transformation_estimate in iter(task_queue.get, "STOP"):
        # No explicit related-files scope: ``solve_image`` opens the DR as
        # its first action and does everything inside that block, so the
        # file attaches itself and travels out on the exception for
        # ``capture_for_queue`` to stamp.
        try:
            result = solve_image(
                dr_fname,
                transformation_estimate,
                web_lock=web_lock,
                mark_start=mark_start,
                mark_end=mark_end,
                **configuration,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(
                "Unexpected exception solving astrometry for %s:\n%s",
                dr_fname,
                format_exc(),
            )
            result_queue.put(
                {
                    "error": capture_for_queue(exc, component=Component.STEP),
                    "dr_fname": dr_fname,
                }
            )
            return
        result_queue.put(result)
    _logger.debug("Astrometry solving process finished.")


def prepare_configuration(configuration, dr_header):
    """Apply fallbacks to the configuration."""

    _logger.debug("Preparing configuration from: %s", repr(configuration))
    result = configuration.copy()

    dr_eval = Evaluator(dr_header)
    result["frame_fov_estimate"] = tuple(
        dr_eval(expr) for expr in configuration["frame_fov_estimate"]
    )

    result.update(
        binning=1,
        srcextract_filter="True",
        sky_preprojection="tan",
        weights_expression="1.0",
    )
    return result


# Could not think of good way to split
# pylint: disable=too-many-branches
def manage_astrometry(
    pending, task_queue, result_queue, mark_start, mark_end, workers=()
):
    """Manege solving all frames until they solve or fail hopelessly."""

    num_queued = 0
    for pending_set in pending.values():
        task_queue.put((pending_set.pop(), None))
        num_queued += 1

    failed = {}

    while pending or num_queued:
        _logger.debug("Pending: %s", repr(pending))
        _logger.debug("Number scheduled: %d", num_queued)
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                break
            except Exception:  # queue.Empty
                if workers and not any(p.is_alive() for p in workers):
                    # The empty-queue timeout is control flow, not the
                    # cause; the cause is the workers dying -> ``from None``.
                    # ``exit_signal`` matches the Scheme-A representation so a
                    # crash report reads it the same way (SIGKILL/OOM vs. a
                    # native crash, portably decoded).
                    raise WorkerCrashedError(
                        "All astrometry worker processes died unexpectedly "
                        f"with {num_queued} task(s) still in flight.",
                        details={
                            "exit_signal": decode_exit_signals(
                                p.exitcode for p in workers
                            ),
                            "num_in_flight": num_queued,
                        },
                    ) from None
        num_queued -= 1

        if "error" in result:
            reraise_from_worker(result["error"])

        if "raw_transformation" in result:
            if not result["saved"]:
                _logger.critical(
                    "Failed to save astrometry solution to DR file %s.",
                    result["dr_fname"],
                )
                break
            if result["trans_key"] in failed:
                if result["trans_key"] not in pending:
                    pending[result["trans_key"]] = []
                pending[result["trans_key"]].extend(
                    [f[0] for f in failed[result["trans_key"]]]
                )
                del failed[result["trans_key"]]

            for dr_fname in pending.get(result["trans_key"], []):
                task_queue.put((dr_fname, result["raw_transformation"]))
                num_queued += 1

            if result["trans_key"] in pending:
                del pending[result["trans_key"]]
        else:
            if result["trans_key"] not in failed:
                failed[result["trans_key"]] = []
            failed[result["trans_key"]].append(
                (result["dr_fname"], result["fail_reason"])
            )
            if pending.get(result["trans_key"], False):
                task_queue.put((pending[result["trans_key"]].pop(), None))
                num_queued += 1

            if not pending.get(result["trans_key"], True):
                del pending[result["trans_key"]]

    for failed_set in failed.values():
        for dr_fname, reason in failed_set:
            mark_start(dr_fname)
            mark_end(dr_fname, reason)
            _logger.error(
                "Failed astrometry for DR file %s: %s",
                dr_fname,
                [
                    fail_key
                    for fail_key, fail_reason in fail_reasons.items()
                    if fail_reason == reason
                ][0],
            )


# pylint: enable=too-many-branches


def solve_astrometry(
    dr_collection, start_status, configuration, mark_start, mark_end
):
    """Find the (RA, Dec) -> (x, y) transformation for the given DR files."""

    assert start_status is None

    _logger.debug(
        "Solving astrometry for %d DR files with configuration %s",
        len(dr_collection),
        repr(configuration),
    )
    # create_catalogs(configuration, dr_collection)
    pending = {}
    for dr_fname in dr_collection:
        reuse_key = Evaluator(dr_fname)(
            configuration["reuse_transformation_key"]
        )
        if reuse_key not in pending:
            pending[reuse_key] = [dr_fname]
        else:
            pending[reuse_key].append(dr_fname)

    forbid_nested_workers()

    task_queue = Queue()
    result_queue = Queue()

    web_lock = Lock()
    configuration["parent_pid"] = getpid()
    workers = [
        Process(
            target=astrometry_process,
            args=(task_queue, result_queue),
            kwargs={
                "configuration": configuration,
                "web_lock": web_lock,
                "mark_start": mark_start,
                "mark_end": mark_end,
            },
        )
        for _ in range(configuration["num_parallel_processes"])
    ]

    _logger.debug("Starting %d astrometry processes", len(workers))
    for process in workers:
        process.start()
    _logger.debug("Starting astrometry on %d pending frame sets", len(pending))

    try:
        manage_astrometry(
            pending, task_queue, result_queue, mark_start, mark_end, workers
        )
    finally:
        _logger.debug("Stopping astrometry solving processes.")
        for _ in workers:
            task_queue.put("STOP")
        # Drain unread results while waiting for workers to exit so the pipe
        # buffer does not fill up and prevent workers from stopping cleanly.
        for process in workers:
            while process.is_alive():
                try:
                    result_queue.get(timeout=0.05)
                except Exception:  # queue.Empty
                    pass
            process.join()


def cleanup_interrupted(interrupted, configuration):
    """Delete any astrometry datasets left over from prior interrupted run."""

    for dr_fname, status in interrupted:
        assert status == 0

        path_substitutions = {
            substitution: configuration[substitution]
            for substitution in [
                "srcextract_version",
                "catalogue_version",
                "skytoframe_version",
            ]
        }
        with DataReductionFile(dr_fname, "r+") as dr_file:
            dr_file.delete_sources(
                "catalogue.columns",
                "catalogue_column_name",
                **path_substitutions,
            )
            for dataset_key in [
                "skytoframe.matched",
                "skytoframe.coefficients",
            ]:
                dr_file.delete_dataset(dataset_key, **path_substitutions)

            for attribute_key in [
                "skytoframe.type",
                "skytoframe.terms",
                "skytoframe.sky_center",
                "skytoframe.residual",
                "srcextract.cfg.binning",
                "skytoframe.cfg.srcextract_filter",
                "skytoframe.cfg.sky_preprojection",
                "skytoframe.cfg.max_match_distance",
                "skytoframe.cfg.frame_center",
                "skytoframe.cfg.weights_expression",
            ]:
                dr_file.delete_attribute(attribute_key, **path_substitutions)
    return -1


@cli_entry_point(component=Component.STEP)
def main():
    """Run the step from the command line."""

    cmdline_config = parse_command_line()
    setup_process(task="main", **cmdline_config)

    solve_astrometry(
        list(
            find_dr_fnames(
                cmdline_config.pop("dr_files"),
                cmdline_config.pop("astrometry_only_if"),
            )
        ),
        None,
        cmdline_config,
        ignore_progress,
        ignore_progress,
    )


if __name__ == "__main__":
    main()
