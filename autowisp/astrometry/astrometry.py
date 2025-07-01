#!/usr/bin/env python3

"""Fit for a transformation between sky and image coordinates."""
import logging
from tempfile import TemporaryDirectory
import subprocess
import os
from traceback import format_exc
import time
from urllib.request import urlopen
from urllib.error import URLError
import shutil

import numpy
from numpy.lib.recfunctions import structured_to_unstructured

from scipy import linalg
from scipy import spatial
from scipy.optimize import fsolve

from astropy.io import fits

from autowisp.astrometry.map_projections import (
    gnomonic_projection,
    inverse_gnomonic_projection,
)
from autowisp.astrometry.astrometry_net_client import (
    Client as AstrometryNetClient,
)

_logger = logging.getLogger(__name__)


# pylint:disable=R0913
# pylint:disable=R0914
# pylint:disable=R0915
# pylint:disable=C0103


def transformation_matrix(astrometry_order, xi, eta):
    """
    Constructs the transformation matrix for a given astrometry order.

    Args:
        astrometry_order(int): The order of the transformation to fit

        xi(numpy.ndarray): from projected coordinates

        eta(numpy.ndarray): from projected coordinates

    Returns:
        trans_matrix(numpy.ndarray): transformation matrix

    Notes:
        Ex: for astrometry_order 2: 1, xi, eta, xi^2, xi*eta, eta^2
    """

    # TODO: alocate matrix first with right size and then fill, instead of doing
    # it the slow way below: new method seems to be not faster.
    # if it is much slower than the initial, roll back!

    # trans_matrix = numpy.ones((eta.shape[0], 1))

    # for i in range(1, astrometry_order + 1):
    #     for j in range(i + 1):
    #         trans_matrix = numpy.block(
    #             [trans_matrix, xi ** (i - j) * eta ** j]
    #         )

    num_terms = (astrometry_order + 1) * (astrometry_order + 2) // 2
    n_points = len(xi)

    trans_matrix = numpy.empty((n_points, num_terms))
    trans_matrix[:, 0] = 1

    col = 1
    for i in range(1, astrometry_order + 1):
        for j in range(i + 1):
            trans_matrix[:, col] = ((xi ** (i - j)) * (eta**j)).ravel()
            col += 1

    return trans_matrix


def find_ra_dec(xieta_guess, trans_x, trans_y, radec_cent, frame_x, frame_y):
    """
    Find the (xi, eta) that map to given coordinates in the frame.

    Args:
        xieta_guess(numpy array):    Starting point for the solver trying find
            (xi, eta) that map to the given coordinates.

        trans_x(numpy array):    transformation matrix for x

        trans_y(numpy array):    transformation matrix for y

        radec_cent(dict):    The RA and Dec of the center of the gnomonic
            projection defining (xi, eta).

        frame_x(float):    x coordinate for which to find RA, Dec.

        frame_y(float):    y coordinate for which to find RA, Dec.

    Returns:
        new_xieta_cent(numpy array): the new center function for (xi, eta)

    """

    assert trans_x.size == trans_y.size
    astrometry_order = (numpy.sqrt(1.0 + 8.0 * trans_x.size) - 3.0) / 2.0
    assert numpy.allclose(astrometry_order, numpy.round(astrometry_order), atol=1e-10)
    astrometry_order = numpy.rint(astrometry_order).astype(int)

    def equations(xieta_cent):
        """The equations that need to be solved to find the center."""

        xi = xieta_cent[0]
        eta = xieta_cent[1]

        new_xieta_cent = numpy.empty(2)

        new_xieta_cent[0] = trans_x[0, 0] - frame_x
        new_xieta_cent[1] = trans_y[0, 0] - frame_y

        k = 1
        for i in range(1, astrometry_order + 1):
            for j in range(i + 1):
                new_xieta_cent[0] = (
                    new_xieta_cent[0] + trans_x[k, 0] * xi ** (i - j) * eta**j
                )
                new_xieta_cent[1] = (
                    new_xieta_cent[1] + trans_y[k, 0] * xi ** (i - j) * eta**j
                )
                k = k + 1
        return new_xieta_cent

    xieta_cent = numpy.empty(1, dtype=[("xi", float), ("eta", float)])
    xieta_cent["xi"], xieta_cent["eta"] = fsolve(equations, xieta_guess)

    source = numpy.empty(1, dtype=[("RA", float), ("Dec", float)])
    inverse_gnomonic_projection(source, xieta_cent, **radec_cent)

    return {"RA": source["RA"][0], "Dec": source["Dec"][0]}


def estimate_transformation_from_corr(
    initial_corr, ra_cent, dec_cent, tweak_order, astrometry_order
):
    """
    Estimate the transformation from astrometry.net correspondence file.

    Args:
        initial_corr(structured numpy array):    The correspondence file
            containing field_x, field_y, index_ra, and index_dec

        ra_cent(float):    Estimate of the RA of the center of the frame

        dec_cent(float):    Estimate of the Dec of the center of the frame

        tweak_order(int):    The order of the astrometry.net transformation

        astrometry_order(int):    The order of the transformation that will be
            fit. Fills terms of higher order than `tweak_order` with zeros.

    Returns:
        matrix:
            Estimate of the transformation x(xi, eta)

        matrix:
            Estimate of the transformation y(xi, eta)



    """

    projected = numpy.empty(
        initial_corr.shape[0], dtype=[("xi", float), ("eta", float)]
    )

    radec_center = {"RA": ra_cent, "Dec": dec_cent}

    gnomonic_projection(initial_corr, projected, **radec_center)

    xi = projected["xi"][numpy.newaxis].T

    eta = projected["eta"][numpy.newaxis].T

    trans_matrix = transformation_matrix(tweak_order, xi, eta)
    num_trans_terms = ((astrometry_order + 1) * (astrometry_order + 2)) // 2
    num_tweak_terms = ((tweak_order + 1) * (tweak_order + 2)) // 2

    trans_x = numpy.zeros(num_trans_terms)
    trans_y = numpy.zeros(num_trans_terms)

    trans_x[:num_tweak_terms] = linalg.lstsq(trans_matrix, initial_corr["x"])[0]
    trans_y[:num_tweak_terms] = linalg.lstsq(trans_matrix, initial_corr["y"])[0]

    return trans_x[numpy.newaxis].T, trans_y[numpy.newaxis].T


class TempAstrometryFiles:
    """Context manager for the temporary files needed for astrometry."""

    def __init__(self, file_types=("sources", "corr", "axy", "config")):
        """Create all required temporary files."""
        self._temp_dir_obj = TemporaryDirectory()
        self.temp_dir_path = self._temp_dir_obj.name
        self._file_types = file_types

        for file_type in self._file_types:
            full_path = os.path.join(self.temp_dir_path, file_type)
            setattr(self, file_type + "_fname", full_path)

    def __enter__(self):
        """Return the filenames of the temporary files."""
        return tuple(
            getattr(self, file_type + "_fname") for file_type in self._file_types
        )

    def __exit__(self, *ignored_args, **ignored_kwargs):
        """Close and delete the temporary files."""
        self._temp_dir_obj.cleanup()


def create_sources_file(xy_extracted, sources_fname):
    """Create a FITS BinTable file with the given name containing
    the extracted sources.

    Returns: an array containing x-y extracted sources
    """

    x_extracted = fits.Column(name="x", format="D", array=xy_extracted["x"])
    y_extracted = fits.Column(name="y", format="D", array=xy_extracted["y"])
    xyls = fits.BinTableHDU.from_columns([x_extracted, y_extracted])
    xyls.writeto(sources_fname)

    return xy_extracted


def create_config_file(config_fname, fov_range, anet_indices):
    """Create configuration file set up to solve imaves FOV in given range."""

    with open(config_fname, "w", encoding="utf-8") as config_file:
        if min(fov_range) < 2.0:
            config_file.write(f"add_path {anet_indices[0]}\n")
        if max(fov_range) > 0.5:
            config_file.write(f"add_path {anet_indices[1]}\n")

        config_file.write("autoindex\n")

    with open(config_fname, "r", encoding="utf-8") as config_file:
        _logger.debug("Astrometry.net engine config:\n%s", config_file.read())


def get_initial_corr_local(
    header, xy_extracted, tweak_order_range, fov_range, anet_indices
):
    """Get inital extracted to catalog source match using ``solve-field``."""

    with TempAstrometryFiles() as (
        sources_fname,
        corr_fname,
        axy_fname,
        config_fname,
    ):
        xy_extracted = create_sources_file(xy_extracted, sources_fname)
        create_config_file(config_fname, fov_range, anet_indices)
        for tweak in range(tweak_order_range[0], tweak_order_range[1] + 1):
            solve_field_command = [
                "solve-field",
                sources_fname,
                "--backend-config",
                config_fname,
                "--corr",
                corr_fname,
                "--width",
                str(header["NAXIS1"]),
                "--height",
                str(header["NAXIS2"]),
                "--tweak-order",
                str(tweak),
                "--match",
                "none",
                "--wcs",
                "none",
                "--index-xyls",
                "none",
                "--rdls",
                "none",
                "--solved",
                "none",
                "--axy",
                axy_fname,
                "--no-plots",
                "--scale-low",
                repr(fov_range[0]),
                "--scale-high",
                repr(fov_range[1]),
                "--overwrite",
            ]
            _logger.debug(
                "Starting solve-field command:\n\t%s",
                "\n\t\t".join(solve_field_command),
            )
            try:
                subprocess.run(solve_field_command, check=True)
            except subprocess.SubprocessError:
                _logger.critical("solve-field failed with error:\n%s", format_exc())
                continue

            if not os.path.isfile(corr_fname):
                _logger.critical(
                    "Correspondence file %s not created.", repr(corr_fname)
                )
                return "solve-field failed", 0

            with fits.open(corr_fname, mode="readonly") as corr:
                result = numpy.copy(corr[1].data[:])
            if result.size > ((tweak + 1) * (tweak + 2)) // 2:
                return result, tweak

    return "solve-field failed", 0


def get_initial_corr_web(header, xy_extracted, tweak_order_range, fov_range, api_key):
    """Get initial extracted to catalog source match using web astrometry.net."""

    config = {
        "allow_commercial_use": "n",
        "allow_modifications": "n",
        "publicly_visible": "n",
        "scale_lower": fov_range[0],
        "scale_upper": fov_range[1],
        "scale_type": "ul",
        "scale_units": "degwidth",
        "image_width": header["NAXIS1"],
        "image_height": header["NAXIS2"],
        #        'center_ra',
        #        'center_dec',
        #        'radius',
        #        'downsample_factor',
        #        'positional_error',
        "x": xy_extracted["x"],
        "y": xy_extracted["y"],
    }
    client = AstrometryNetClient()
    while True:
        try:
            client.login(api_key)
            break
        except URLError as err:
            _logger.error(
                "Failed to connect to astrometry.net server: %s\nRetrying...",
                err.reason,
            )
            time.sleep(20)

    for tweak_order in range(tweak_order_range[0], tweak_order_range[1] + 1):
        config["tweak_order"] = tweak_order
        upload_result = client.upload(**config)

        if upload_result["status"] != "success":
            return upload_result["status"], 0

        assert "subid" in upload_result
        solved_job_id = None
        while solved_job_id is None:
            time.sleep(5)
            submission_status = client.sub_status(upload_result["subid"], justdict=True)
            _logger.debug("Astrometry.net submission status: %s", submission_status)
            jobs = submission_status.get("jobs", [])
            job_id = None
            if len(jobs):
                for job_id in jobs:
                    if job_id is not None:
                        break
                if job_id is not None:
                    _logger.debug("Selecting job id %s", job_id)
                    solved_job_id = job_id

        while True:
            job_status = client.job_status(solved_job_id, justdict=True).get(
                "status", ""
            )
            _logger.debug("Got job status: %s", job_status)
            if job_status in ["success", "failure"]:
                break
            time.sleep(5)

        if job_status == "failure":
            return "web solve failed", 0
        corr_url = client.apiurl.replace("/api/", f"/corr_file/{solved_job_id:d}")

        with TempAstrometryFiles(("corr",)) as (corr_fname,):
            _logger.debug("Retrieving file from '%s' to '%s'", corr_url, corr_fname)
            with urlopen(corr_url) as remote_corr, open(corr_fname, "wb") as local_corr:
                shutil.copyfileobj(remote_corr, local_corr)
            with fits.open(corr_fname, mode="readonly") as corr:
                result = numpy.copy(corr[1].data[:])
            if result.size > (tweak_order + 1) * (tweak_order + 2) // 2:
                return result, tweak_order

    return "web solve failed", 0


def estimate_transformation(
    *, dr_file, xy_extracted, config, header=None, web_lock=None
):
    """Attempt to estimate the sky-to-frame transformation for given DR file."""

    if header is None:
        header = dr_file.get_frame_header()
    initial_corr_arg = (
        header,
        xy_extracted,
        config["tweak_order_range"],
        config["fov_range"],
    )
    if (
        "anet_indices" in config
        and os.path.exists(config["anet_indices"][0])
        and os.path.exists(config["anet_indices"][1])
    ):
        field_corr, tweak_order = get_initial_corr_local(
            *initial_corr_arg, config["anet_indices"]
        )
    else:
        with web_lock:
            field_corr, tweak_order = get_initial_corr_web(
                *initial_corr_arg, config["anet_api_key"]
            )

    if tweak_order == 0:
        return None, None, field_corr

    initial_corr = numpy.zeros(
        (field_corr["field_x"].shape),
        dtype=[("x", ">f8"), ("y", ">f8"), ("RA", ">f8"), ("Dec", ">f8")],
    )

    initial_corr["x"] = field_corr["field_x"]
    initial_corr["y"] = field_corr["field_y"]
    initial_corr["RA"] = field_corr["index_ra"]
    initial_corr["Dec"] = field_corr["index_dec"]

    return estimate_transformation_from_corr(
        initial_corr=initial_corr,
        tweak_order=tweak_order,
        astrometry_order=config["astrometry_order"],
        ra_cent=config["ra_cent"],
        dec_cent=config["dec_cent"],
    ) + ("success",)


def refine_transformation(
    *,
    astrometry_order,
    max_srcmatch_distance,
    max_iterations,
    trans_threshold,
    trans_x,
    trans_y,
    ra_cent,
    dec_cent,
    x_frame,
    y_frame,
    xy_extracted,
    catalog,
    min_source_safety_factor=5.0,
):
    """
    Iterate the process until we get a transformation that
    its difference from the previous one is less than a threshold

    Args:
        astrometry_order(int):    The order of the transformation to fit

        max_srcmatch_distance(float):    The upper bound distance in the
            KD tree

        trans_threshold(float):    The threshold for the difference of two
            consecutive transformations

        trans_x(numpy array):    Initial estimate for the x transformation
            matrix

        trans_y(numpy array):    Initial estimate for the x transformation
            matrix

        ra_cent(float):    Initial estimate for the RA of the center of the
            frame

        dec_cent(float):    Initial estimate for the Dec of the center of the
            frame

        x_frame(float):    length of the frame in pixels

        y_frame(float):    width of the frame in pixels

        xy_extracted(structured numpy array):    x and y of the extracted
            sources of the frame

        catalog(pandas.DataFrame):    The catalog of sources to match to

    Returns:
        trans_x(2D numpy array):
            the coefficients of the x(xi, eta) transformation

        trans_y(2D numpy array):
            the coefficients of the y(xi, eta) transformation

        cat_extracted_corr(structured numpy array):
            the catalogs extracted correspondence indexes

        res_rms(float): the residual

        ratio(float): the ratio of matched to unmatched

        ra_cent(float): the new RA center array

        dec_cent(float): the new Dec center array
    """

    x_cent = x_frame / 2.0
    y_cent = y_frame / 2.0
    xy_extracted = structured_to_unstructured(xy_extracted)[:, 0:2]
    # xy_extracted = xy_extracted[:, 0:2]
    counter = 0
    x_transformed = numpy.inf
    y_transformed = numpy.inf
    logger = logging.getLogger(__name__)

    kdtree = spatial.KDTree(xy_extracted)

    while True:

        counter += 1
        if counter > 1:
            # TODO: fix pylint disables here
            # pylint:disable=used-before-assignment
            ra_cent = cent_new["RA"]
            dec_cent = cent_new["Dec"]
            # pylint:enable=used-before-assignment
        radec_cent = {"RA": ra_cent, "Dec": dec_cent}

        projected = numpy.empty(catalog.shape[0], dtype=[("xi", float), ("eta", float)])
        gnomonic_projection(catalog, projected, **radec_cent)

        xi = projected["xi"].reshape(-1, 1)  # Reshape to (n, 1)
        eta = projected["eta"].reshape(-1, 1)  # Reshape to (n, 1)

        trans_matrix_xy = transformation_matrix(astrometry_order, xi, eta)

        old_x_transformed = x_transformed
        old_y_transformed = y_transformed
        x_transformed = trans_matrix_xy @ trans_x
        y_transformed = trans_matrix_xy @ trans_y

        in_frame = numpy.logical_and(
            numpy.logical_and(x_transformed > 0, x_transformed < x_frame),
            numpy.logical_and(y_transformed > 0, y_transformed < y_frame),
        ).flatten()

        diff = numpy.sqrt(
            (old_x_transformed - x_transformed) ** 2
            + (old_y_transformed - y_transformed) ** 2
        ).flatten()[in_frame]

        logger.debug("diff: %s", repr(diff.max()))

        if not (diff > trans_threshold).any() or counter > max_iterations:
            # pylint:disable=used-before-assignment
            cat_extracted_corr = numpy.empty((n_matched, 2), dtype=int)
            cat_extracted_corr[:, 0] = numpy.arange(catalog.shape[0])[matched]
            cat_extracted_corr[:, 1] = ix[matched]
            # Exclude the sources that are not within the frame:

            return (
                trans_x,
                trans_y,
                cat_extracted_corr,
                res_rms,
                ratio,
                ra_cent,
                dec_cent,
                counter <= max_iterations,
            )

            # pylint:enable=used-before-assignment
        xy_transformed = numpy.block([x_transformed, y_transformed])
        d, ix = kdtree.query(xy_transformed, distance_upper_bound=max_srcmatch_distance)

        result, count = numpy.unique(ix, return_counts=True)

        for multi_match_i in result[count > 1][:-1]:
            bad_match = ix == multi_match_i
            d[bad_match] = numpy.inf
            ix[bad_match] = result[-1]

        matched = numpy.isfinite(d)
        n_matched = matched.sum()
        n_extracted = len(xy_extracted)
        # TODO: add weights to residual and to the fit eventually
        res_rms = numpy.sqrt(numpy.square(d[matched]).mean())
        ratio = n_matched / n_extracted

        logger.debug("# of matched: %d out of %d", n_matched, n_extracted)
        matched_sources = numpy.empty(
            n_matched,
            dtype=[("RA", float), ("Dec", float), ("x", float), ("y", float)],
        )

        j = 0
        k = -1

        for i in range(ix.size):
            k += 1
            if not numpy.isinf(d[i]):
                matched_sources["RA"][j] = catalog["RA"].iloc[k]
                matched_sources["Dec"][j] = catalog["Dec"].iloc[k]
                matched_sources["x"][j] = xy_extracted[ix[i], 0]
                matched_sources["y"][j] = xy_extracted[ix[i], 1]
                j += 1

        cent_new = find_ra_dec(
            numpy.array([numpy.mean(xi), numpy.mean(eta)]),
            trans_x,
            trans_y,
            radec_cent,
            x_cent,
            y_cent,
        )

        projected_new = numpy.empty(
            matched_sources.shape[0], dtype=[("xi", float), ("eta", float)]
        )

        gnomonic_projection(matched_sources, projected_new, **cent_new)

        trans_matrix = transformation_matrix(
            astrometry_order,
            projected_new["xi"].reshape(projected_new["xi"].size, 1),
            projected_new["eta"].reshape(projected_new["eta"].size, 1),
        )
        print(trans_matrix.shape)

        if trans_matrix.shape[0] <= (min_source_safety_factor * trans_matrix.shape[1]):
            raise ValueError(
                "The number of equations is "
                "insufficient to solve transformation "
                "coefficients"
            )

        trans_x = linalg.lstsq(
            trans_matrix,
            matched_sources["x"].reshape(matched_sources["x"].size, 1),
        )[0]
        trans_y = linalg.lstsq(
            trans_matrix,
            matched_sources["y"].reshape(matched_sources["y"].size, 1),
        )[0]
