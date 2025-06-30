#!/usr/bin/env python3
# pylint: disable=too-many-lines

"""Utilities for querying catalogs for astrometry."""

from os import path, makedirs
import logging
from hashlib import md5
from contextlib import nullcontext

import numpy
import pandas
from astropy import units
from astropy.io import fits
from astroquery.gaia import GaiaClass, conf

from autowisp import Evaluator, DataReductionFile
from autowisp.astrometry import Transformation
from autowisp.astrometry.map_projections import (
    gnomonic_projection,
    inverse_gnomonic_projection,
)

if __name__ == "__main__":
    import doctest
    from configargparse import ArgumentParser, DefaultsFormatter
    from matplotlib import pyplot

_logger = logging.getLogger(__name__)


class WISPGaia(GaiaClass):
    """Extend queries with condition and sorting."""

    def _get_result(self, query, add_propagated, verbose=False):
        """Get and format the result as specified by user."""

        job = self.launch_job_async(query, verbose=verbose)
        result = job.get_results()
        _logger.debug("Gaia query result: %s", repr(result))
        _logger.debug("Gaia query result columns: %s", repr(result.colnames))
        if result.colnames == ["num_obj"]:
            return result["num_obj"][0]
        result.rename_column("ra", "ra_orig")
        result.rename_column("dec", "dec_orig")
        for colname in result.colnames:
            new_colname = colname.lower()
            if colname != new_colname:
                result.rename_column(colname, colname.lower())

        if add_propagated:
            propagated = {
                coord: numpy.empty(len(result)) for coord in ["RA", "Dec"]
            }
            for i, pos in enumerate(result["propagated"]):
                for coord, value_str in zip(
                    ["RA", "Dec"], pos.strip().strip("()").split(",")
                ):
                    propagated[coord][i] = float(value_str) * 180.0 / numpy.pi

            result.remove_column("propagated")
            for coord in add_propagated:
                result.add_column(propagated[coord], name=coord)

        return result

    # pylint: disable=too-many-locals
    def query_object_filtered(
        self,
        *,
        ra,
        dec,
        width,
        height,
        order_by,
        condition=None,
        epoch=None,
        columns=None,
        order_dir="ASC",
        max_objects=None,
        verbose=False,
        count_only=False,
    ):
        """
        Get GAIA sources within a box satisfying given condition (ADQL).

        Args:
            ra:   The RA of the center of the box to query, with units.

            dec:    The declination of the center of the box to query,
                with units.

            width:    The width of the box (half goes on each side of ``ra``),
                with units.

            height:    The height of the box (half goes on each side of
                ``dec``), with units.

            order_by(str):    How should the stars be ordered.

            condition(str):    Condition the returned sources must satisfy
                (typically imposes a brightness limit)

            epoch:    The epoch to propagate the positions to. If unspecified,
                no propagation will be done.

            columns(iterable):    List of columns to select from the catalog. If
                unspecified, all columns will be returned.

            order_dir(str):    Should the order be ascending (``'ASC'``) or
                descending (``'DESC'``).

            max_objects(int):    Maximum number of objects to return.

            verbose(bool):    Use verbose mode when submitting querries to GAIA.

            count_only(bool):    If ``True``, only the number of objects is
                returned without actually fetching the data.

        Returns:
            astropy Table:
                The result of the query.
        """

        if count_only:
            columns = "COUNT(*) AS num_obj"
        elif columns is None:
            columns = "*"
        else:
            add_propagated = []
            for coord in ["RA", "Dec"]:
                try:
                    add_propagated.append(coord)
                except ValueError:
                    pass

            columns = ", ".join(map(str, columns))

        if "*" in columns and not count_only:
            add_propagated = ["RA", "Dec"]

        if epoch is not None:
            epoch = epoch.to_value(units.yr)
            columns = (
                "EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, "
                f"radial_velocity, ref_epoch, {epoch}) AS propagated, "
            ) + columns

        corners = numpy.empty(shape=(4,), dtype=[("RA", float), ("Dec", float)])
        corners_xi_eta = numpy.empty(
            shape=(4,), dtype=[("xi", float), ("eta", float)]
        )
        width = width.to_value(units.deg)
        height = height.to_value(units.deg)
        corners_xi_eta[0] = (-width / 2, -height / 2)
        corners_xi_eta[1] = (-width / 2, height / 2)
        corners_xi_eta[2] = (width / 2, height / 2)
        corners_xi_eta[3] = (width / 2, -height / 2)

        inverse_gnomonic_projection(
            corners,
            corners_xi_eta,
            RA=ra.to_value(units.deg),
            Dec=dec.to_value(units.deg),
        )

        table_name = self.MAIN_GAIA_TABLE or conf.MAIN_GAIA_TABLE

        select = "SELECT"
        if max_objects is not None:
            select += f" TOP {max_objects}"
        query_str = f"""
            {select}
            {columns}
            FROM {table_name}
            WHERE
                1 = CONTAINS(
                    POINT(
                        {self.MAIN_GAIA_TABLE_RA},
                        {self.MAIN_GAIA_TABLE_DEC}
                    ),
                    POLYGON(
                        {corners[0]['RA']},
                        {corners[0]['Dec']},
                        {corners[1]['RA']},
                        {corners[1]['Dec']},
                        {corners[2]['RA']},
                        {corners[2]['Dec']},
                        {corners[3]['RA']},
                        {corners[3]['Dec']}
                    )
                )
        """
        if condition is not None:
            query_str += f"""
                AND
                ({condition})
            """

        if not count_only:
            query_str += f"""
                ORDER BY
                    {order_by}
                    {order_dir}
            """

        return self._get_result(
            query_str,
            epoch is not None and not count_only and add_propagated,
            verbose,
        )

    # pylint: enable=too-many-locals

    def query_brightness_limited(
        self, *, magnitude_expression, magnitude_limit, **query_kwargs
    ):
        """
        Get sources within a box and a range of magnitudes.

        Args:
            magnitude_expression:    Expression for the relevant magnitude
                involving Gaia columns.

            magnitude_limit:    Either upper limit or lower and upper limit on
                the magnitude defined by ``magnitude_expression``.

            **query_kwargs:    Arguments passed directly to
                `query_object_filtered()`.

        Returns:
            astropy Table:
                The result of the query.
        """
        _logger.debug(
            "Querying Gaia for sources with magnitude: %s, "
            "limits: %s, and kwargs: %s",
            repr(magnitude_expression),
            repr(magnitude_limit),
            repr(query_kwargs),
        )

        if query_kwargs.get("columns", False):
            query_kwargs["columns"] = query_kwargs["columns"] + [
                f"({magnitude_expression}) AS magnitude"
            ]
        else:
            query_kwargs["columns"] = [
                f"({magnitude_expression}) AS magnitude",
                "*",
            ]

        if "order_by" not in query_kwargs:
            query_kwargs["order_by"] = "magnitude"
            query_kwargs["order_dir"] = "ASC"

        if magnitude_limit is not None:
            try:
                min_mag, max_mag = magnitude_limit
                condition = (
                    f"({magnitude_expression}) > {min_mag} AND "
                    f"({magnitude_expression}) < {max_mag}"
                )
            except ValueError:
                condition = f"{magnitude_expression} < {magnitude_limit[0]}"
            except TypeError:
                condition = f"{magnitude_expression} < {magnitude_limit}"

            if "condition" in query_kwargs:
                query_kwargs["condition"] = (
                    f'({query_kwargs["condition"]}) AND ({condition})'
                )
            else:
                query_kwargs["condition"] = condition

        return self.query_object_filtered(**query_kwargs)


gaia = WISPGaia()
# This comes from astroquery (not in our control)
# pylint: disable=invalid-name
gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
# pylint: enable=invalid-name


def create_catalog_file(fname, overwrite=False, **query_kwargs):
    """
    Create a catalog FITS file from a Gaia query.

    Args:
        fname(str):    Name of the catalog file to create.

        **query_kwargs:    Arguments passed directly to
            `gaia.query_brightness_limited()`.
    """

    query = gaia.query_brightness_limited(**query_kwargs)
    if query_kwargs.get("count_only", False):
        print("Number of sources: ", repr(query))
        return
    for colname in [
        "DESIGNATION",
        "phot_variable_flag",
        "datalink_url",
        "epoch_photometry_url",
        "libname_gspphot",
    ]:
        try:
            query[colname] = query[colname].astype(str)
        except KeyError:
            pass

    query.meta["CATALOG"] = "Gaia"
    query.meta["CATVER"] = gaia.MAIN_GAIA_TABLE
    for k in ["ra", "dec", "width", "height"]:
        query.meta[k.upper()] = query_kwargs[k].to_value(units.deg)

    query.meta["EPOCH"] = (
        query_kwargs["epoch"].to_value(units.yr)
        if query_kwargs["epoch"] is not None
        else None
    )
    query.meta["MAGEXPR"] = query_kwargs["magnitude_expression"]
    if query_kwargs.get("magnitude_limit") is not None:
        try:
            (query.meta["MAGMIN"], query.meta["MAGMAX"]) = query_kwargs[
                "magnitude_limit"
            ]
        except ValueError:
            query.meta["MAGMAX"] = query_kwargs["magnitude_limit"][0]
        except TypeError:
            query.meta["MAGMAX"] = query_kwargs["magnitude_limit"]

    if path.dirname(fname) and not path.exists(path.dirname(fname)):
        makedirs(path.dirname(fname))
    query.write(fname, format="fits", overwrite=overwrite)


def read_catalog_file(
    cat_fits,
    filter_expr=None,
    sort_expr=None,
    return_metadata=False,
    add_gnomonic_projection=False,
):
    """
    Read a catalog FITS file.

    Args:
        cat_fits(str, or opened FITS file):    The file to read.

    Returns:
        pandas.DataFrame:
            The catalog information as columns.
    """

    if isinstance(cat_fits, str):
        with fits.open(cat_fits) as opened_cat_fits:
            return read_catalog_file(
                opened_cat_fits,
                filter_expr,
                sort_expr,
                return_metadata,
                add_gnomonic_projection,
            )

    fixed_dtype = cat_fits[1].data.dtype.newbyteorder("=")
    result = pandas.DataFrame.from_records(
        cat_fits[1].data.astype(fixed_dtype), index="source_id"
    )
    if return_metadata or add_gnomonic_projection:
        metadata = cat_fits[1].header

    cat_eval = Evaluator(result)
    if sort_expr is not None:
        sort_val = cat_eval(sort_expr)

    if filter_expr is not None:
        print("Filter expression: " + repr(filter_expr))
        filter_val = cat_eval(filter_expr)
        print("Filter val: " + repr(filter_val))
        filter_val = filter_val.astype(bool)
        result = result.loc[filter_val]

        if sort_expr is not None:
            sort_val = sort_val[filter_val]

    if sort_expr is not None:
        result = result.iloc[numpy.argsort(sort_val)]

    if add_gnomonic_projection:
        if "xi" not in result.columns:
            assert "eta" not in result.columns
            for colname in ["xi", "eta"]:
                result.insert(len(result.columns), colname, numpy.nan)
            gnomonic_projection(
                result, result, RA=metadata["RA"], Dec=metadata["Dec"]
            )

    if return_metadata:
        return result, metadata
    return result


def parse_command_line():
    """Return configuration of catalog to create."""

    parser = ArgumentParser(
        description="Create a catalog file from a Gaia query.",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=118.0,
        help="The right ascention (deg) of the center of the field to query.",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=2.6,
        help="The declination (deg) of the center of the field to query.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=17.0,
        help="The width (deg) of the field to query (along RA direction).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=17.0,
        help="The height (deg) of the field to query (along dec direction).",
    )
    parser.add_argument(
        "--epoch",
        "-t",
        type=float,
        default=None,
        help="The epoch for proper motion corrections in years. If not "
        "specified, positions are not propagated.",
    )
    parser.add_argument(
        "--magnitude-expression",
        default="phot_g_mean_mag",
        help="The expression to use as the relevant magnitude estimate.",
    )
    parser.add_argument(
        "--magnitude-limit",
        nargs="+",
        type=float,
        default=12.0,
        help="Either maximum magnitude or minimum and maximum magnitude limits "
        "to impose.",
    )
    parser.add_argument(
        "--extra-condition",
        default=None,
        help="An extra condition to impose on the selected sources.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out information about the query being executed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing catalog file.",
    )
    parser.add_argument(
        "--catalog-fname",
        default="gaia.fits",
        help="The name of the catalog file to create.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count the number of sources in the query.",
    )
    parser.add_argument(
        "--columns",
        default=[
            "source_id",
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "phot_g_n_obs",
            "phot_g_mean_mag",
            "phot_g_mean_flux",
            "phot_g_mean_flux_error",
            "phot_bp_n_obs",
            "phot_bp_mean_mag",
            "phot_bp_mean_flux",
            "phot_bp_mean_flux_error",
            "phot_rp_n_obs",
            "phot_rp_mean_mag",
            "phot_rp_mean_flux",
            "phot_rp_mean_flux_error",
            "phot_proc_mode",
            "phot_bp_rp_excess_factor",
        ],
        help="The columns to include in the catalog file. Use '*' to include "
        "everything.",
    )
    parser.add_argument(
        "--show-stars",
        action="store_true",
        help="Show the stars in the catalog on a 3-D plot of the sky.",
    )
    parser.add_argument(
        "--run-doctests",
        "--test",
        action="store_true",
        help="Run the unit tests defined in the function docstrings of this "
        "module.",
    )
    return parser.parse_args()


def _find_best_fov(points, fov_size):
    """
    Find square of a given size that fits the most points.

    Example:
        >>> import numpy
        >>> from autowisp.catalog import _find_best_fov
        >>> points = numpy.array([(0.0, 0.0),
        ...                       (0.1, 0.0),
        ...                       (0.1, 0.1),
        ...                       (3.2, 0.0),
        ...                       (3.2, 3.1)],
        ...                      dtype=[('RAcosDec', float), ('Dec', float)])
        >>> _find_best_fov(points, 1.0)
        ((0.0, 0.0), 3)
        >>> _find_best_fov(points, 10.0)
        ((0.0, 0.0), 5)
        >>> _find_best_fov(points, 0.11)
        ((0.0, 0.0), 3)
        >>> _find_best_fov(points, 0.05)
        ((0.0, 0.0), 1)
        >>> _find_best_fov(points, 3.11)
        ((0.1, 0.0), 4)
        >>> points = numpy.array([(0.0, 0.0),
        ...                       (0.1, 0.0),
        ...                       (0.1, -0.1),
        ...                       (3.2, 0.0),
        ...                       (3.2, 3.0)],
        ...                      dtype=[('RAcosDec', float), ('Dec', float)])
        >>> _find_best_fov(points, 1.0)
        ((0.0, -0.1), 3)
        >>> _find_best_fov(points, 10.0)
        ((0.0, -0.1), 5)
        >>> _find_best_fov(points, 0.11)
        ((0.0, -0.1), 3)
        >>> _find_best_fov(points, 0.05)
        ((0.0, 0.0), 1)
        >>> _find_best_fov(points, 3.11)
        ((0.1, -0.1), 4)

    """

    max_count = 0
    best_fov = None, None

    # Sort points for faster search
    points = numpy.sort(points, order="RAcosDec")
    sorted_dec = numpy.sort(points["Dec"])

    _logger.debug(
        "Finding best FOV for %d points:\n%s", points.size, repr(points)
    )

    for x_start_index in range(points.size):
        # No need to check further if peints beyond i are fewer than max found so
        # far
        if points.size - x_start_index <= max_count:
            _logger.debug("Stopping search at x_start_index=%d", x_start_index)
            break

        x = points[x_start_index]["RAcosDec"]
        # Use a more efficient x-mask by limiting the range of points we
        # consider.
        # Find the range in the sorted array where x is within
        # [x, x + fov_size]
        x_end_index = numpy.searchsorted(
            points["RAcosDec"], x + fov_size, side="right"
        )
        if x_end_index - x_start_index <= max_count:
            continue

        for y_ind, y in enumerate(sorted_dec):
            # No need try higher y bounds
            if sorted_dec.size - y_ind <= max_count:
                break

            # Count the points within the range [y1, y1 + fov_size]
            in_range = numpy.logical_and(
                points[x_start_index:x_end_index]["Dec"] >= y,
                points[x_start_index:x_end_index]["Dec"] <= y + fov_size,
            ).sum()

            if in_range > max_count:
                max_count = in_range
                best_fov = (x, y)

            _logger.debug(
                "For x=%s, y=%s, index range is [%d, %d) with %d points surviving.",
                repr(x),
                repr(y),
                x_start_index,
                x_end_index,
                in_range,
            )

            # No need to check further y bounds
            if y + fov_size > sorted_dec[-1]:
                break

        # No need to check further once right edge of window moves past last
        # point
        if x + fov_size > points["RAcosDec"][-1]:
            _logger.debug("Stopping search at x=%s", repr(x))
            break

    _logger.debug(
        "Best FOV found: %s <= x <= %s, %s <= y <= %s with %d points.",
        repr(best_fov[0]),
        repr(best_fov[0] + fov_size),
        repr(best_fov[1]),
        repr(best_fov[1] + fov_size),
        max_count,
    )

    return best_fov, max_count


def find_outliers(center_ra_dec, max_allowed_offset):
    """
    Find frames that are outliers in pointing.


    Args:
        center_ra_dec(numpy.array):    The center RA and Dec of all the frames
            being considered. The data type should include columns named
            ``'RA'`` and ``'Dec'``.

        max_allowed_offset(float):    The maximum allowed offset in the RA and
            Dec directions in degrees. The most outlier frames is iteratively
            rejected until this criterion is satisfied.

    Returns:
        [int]:
            List of the indices within ``center_ra_dec`` that were rejected as
            outliers.
    """

    points = numpy.empty(
        center_ra_dec.shape,
        dtype=[("RAcosDec", numpy.float64), ("Dec", numpy.float64)],
    )
    points["RAcosDec"] = center_ra_dec["RA"] * numpy.cos(
        center_ra_dec["Dec"] * numpy.pi / 180.0
    )
    points["Dec"] = center_ra_dec["Dec"]

    (min_ra_cos_dec, min_dec), num_inside = _find_best_fov(
        points, max_allowed_offset
    )
    if num_inside < 2 <= center_ra_dec.size:
        raise ValueError(
            "Attempting to determine field of view to cover frames with no "
            "conssistent pointing."
        )

    outside = (
        (points["RAcosDec"] < min_ra_cos_dec)
        | (points["RAcosDec"] > min_ra_cos_dec + max_allowed_offset)
        | (points["Dec"] < min_dec)
        | (points["Dec"] > min_dec + max_allowed_offset)
    )
    return numpy.arange(points.size)[outside]


# TODO: Maybe simplify later
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
def get_max_abs_corner_xi_eta(
    header,
    *,
    transformation=None,
    dr_files=None,
    center=None,
    max_offset=10.0,
    **dr_path_substitutions,
):
    """
    Return the max absolute values of the frame corners xi and eta.

    Args:
        header (astropy.io.fits.Header): The header of the image to find corner
            xi and eta of..

        transformation: The transformation to assume applies to the image. If
            not specified, transformations are read from ``dr_files`` and max is
            taken over all of them.

        dr_files: The list of DR files to read transformations from. If
            transformation is not specified. Ignored if transformation is
            specified.

        center: The center to assume for the projection. If not specified, the
            center is calculated as average of max and min RA and Dec for all
            DR files.

        max_offset: The largest allowed offset in the RA or Dec directions
            between the centers of frames. Outliers by more than this many
            degrees are flagged and excluded from consideration.

        dr_path_substitutions: The substitutions to use when reading
            transformations from DR files.

    Returns:
        float:
            The maximum absolute value of xi over all corners of all frame.

        float:
            The maximum absolute value of eta over all corners of all frame.

        structured array:
            The center around which xi, eta are defined. Returned only if center
            was not specified and more than one DR file was specified.

        array(int):
            The indices within DR files that were dropped as outliers in the
            pointing. Returned only if center was not specified and more than
            one DR file was specified.
    """

    assert dr_files or transformation
    if transformation:
        assert header

    center_ra_dec = numpy.empty(
        1 if transformation is not None else len(dr_files),
        dtype=[("RA", float), ("Dec", float)],
    )
    change_center = transformation is None or center is not None
    corner_coords = numpy.empty(
        4 * center_ra_dec.size,
        dtype=(
            [("RA", float), ("Dec", float)]
            if change_center
            else [("xi", float), ("eta", float)]
        ),
    )

    corner_ind = 0
    for this_trans in dr_files if transformation is None else [transformation]:
        if transformation is None:
            with DataReductionFile(this_trans, "r") as dr_file:
                header = dr_file.get_frame_header()
                this_trans = Transformation()
                this_trans.read_transformation(dr_file, **dr_path_substitutions)

        center_ra_dec[corner_ind // 4] = tuple(this_trans.pre_projection_center)
        for x in [0.0, float(header["NAXIS1"])]:
            for y in [0.0, float(header["NAXIS2"])]:
                corner_coords[corner_ind] = this_trans.inverse(
                    x,
                    y,
                    result=("equatorial" if change_center else "pre_projected"),
                )
                corner_ind += 1

    if center is None and transformation is None and len(dr_files) > 1:
        center = {}
        outliers = numpy.array(sorted(find_outliers(center_ra_dec, max_offset)))
        if outliers.size:
            keep = numpy.ones(center_ra_dec.size, dtype=bool)
            keep[outliers] = False
            center_ra_dec = center_ra_dec[keep]
            keep = numpy.ones(corner_coords.size, dtype=bool)
            for i in range(4):
                keep[outliers + i] = False
            corner_coords = corner_coords[keep]

        for coord in ["Dec", "RA"]:
            min_coord = numpy.min(center_ra_dec[coord])
            max_coord = numpy.max(center_ra_dec[coord])
            center[coord] = 0.5 * (min_coord + max_coord)
        return_center = True
    else:
        return_center = False

    if change_center:
        xi_eta = numpy.empty(
            corner_coords.size, dtype=[("xi", float), ("eta", float)]
        )
        gnomonic_projection(corner_coords, xi_eta, **center)
    else:
        xi_eta = corner_coords

    result = (
        max(abs(xi_eta["xi"].min()), abs(xi_eta["xi"].max())),
        max(abs(xi_eta["eta"].min()), abs(xi_eta["eta"].max())),
    )

    if max(result) > 40.0:
        raise RuntimeError(
            "Observations with field of view exceeding 40 degrees are not "
            "supported."
        )

    return result + ((center, outliers) if return_center else ())


# pylint: enable=too-many-locals
# pylint: enable=too-many-branches


def get_catalog_info(
    *,
    dr_files=None,
    transformation=None,
    header=None,
    configuration,
    **dr_path_substitutions,
):
    """Get the configuration of the catalog needed for this frame."""

    assert dr_files or transformation

    _logger.debug("Creating catalog info from: %s", repr(configuration))

    if transformation is None and len(dr_files) == 1:
        with DataReductionFile(dr_files[0], "r") as dr_file:
            if header is None:
                header = dr_file.get_frame_header()
            transformation = Transformation()
            transformation.read_transformation(dr_file, **dr_path_substitutions)
    if dr_files is None or len(dr_files) == 1:
        max_offset = None
    else:
        max_offset = configuration["max_pointing_offset"]

    trans_fov = get_max_abs_corner_xi_eta(
        dr_files=dr_files,
        transformation=transformation,
        header=header,
        max_offset=max_offset,
        **dr_path_substitutions,
    )

    if transformation is not None:
        trans_fov += (
            dict(zip(["RA", "Dec"], transformation.pre_projection_center)),
            numpy.array([]),
        )

    _logger.debug("Catalog FOV info: %s", repr(trans_fov))

    pointing_precision = configuration["pointing_precision"] * units.deg

    catalog_info = {
        "dec_ind": int(
            numpy.round(trans_fov[2]["Dec"] * units.deg / pointing_precision)
        )
    }
    catalog_info["dec"] = pointing_precision * catalog_info["dec_ind"]

    catalog_info["ra_ind"] = int(
        numpy.round(
            (trans_fov[2]["RA"] * units.deg * numpy.cos(catalog_info["dec"]))
            / pointing_precision
        )
    )
    catalog_info["ra"] = (
        pointing_precision
        * catalog_info["ra_ind"]
        / numpy.cos(catalog_info["dec"])
    )
    catalog_info["magnitude_expression"] = configuration["magnitude_expression"]

    if configuration["max_magnitude"] is None:
        assert configuration["min_magnitude"] is None
        catalog_info["magnitude_limit"] = None
    else:
        catalog_info["magnitude_limit"] = (
            (configuration["max_magnitude"],)
            if configuration["min_magnitude"] is None
            else (
                configuration["min_magnitude"],
                configuration["max_magnitude"],
            )
        )

    _logger.debug(
        "From transformation estimate half FOV is: %s", repr(trans_fov)
    )
    frame_fov_estimate = tuple(
        numpy.round(
            (
                max(
                    2.0 * trans_fov[i] * units.deg,
                    configuration.get("frame_fov_estimate", (0, 0))[i],
                )
                + pointing_precision
            )
            / (configuration["fov_precision"] * units.deg)
        )
        * configuration["fov_precision"]
        * units.deg
        for i in range(2)
    )

    _logger.debug(
        "Determining catalog query size from: "
        "frame_fov_estimate=%s, "
        "image_scale_factor=%s",
        repr(frame_fov_estimate),
        repr(1.0 + 2.0 * configuration["fov_safety_margin"]),
    )
    catalog_info["width"], catalog_info["height"] = (
        fov_size * (1.0 + 2.0 * configuration["fov_safety_margin"])
        for fov_size in frame_fov_estimate
    )

    if header is None:
        with DataReductionFile(dr_files[0], "r") as dr_file:
            header = dr_file.get_frame_header()
        catalog_info["epoch"] = Evaluator(header)(configuration["epoch"])
        if dr_files and len(dr_files) > 1:
            for dr_fname in dr_files[1:]:
                with DataReductionFile(dr_fname, "r") as dr_file:
                    if (
                        Evaluator(dr_file.get_frame_header())(
                            configuration["epoch"]
                        )
                        != catalog_info["epoch"]
                    ):
                        raise RuntimeError(
                            "Not all data reduction files to be covered by a single"
                            " catalog have the same epoch"
                        )
    else:
        catalog_info["epoch"] = Evaluator(header)(configuration["epoch"])

    catalog_info["columns"] = configuration["columns"]

    get_checksum = md5()
    for cfg in sorted(catalog_info.items()):
        get_checksum.update(repr(cfg).encode("ascii"))

    catalog_info["fname"] = configuration["fname"].format(
        **dict(header), **catalog_info, checksum=get_checksum.hexdigest()
    )
    _logger.debug("Created catalog info: %s", repr(catalog_info))

    return catalog_info, trans_fov[2], trans_fov[3]


# No god way to simplify
# pylint: disable=too-many-branches
def ensure_catalog(
    *,
    dr_files=None,
    transformation=None,
    header=None,
    configuration,
    lock=nullcontext(),
    return_metadata=True,
    **dr_path_substitutions,
):
    """Re-use or create astrometry catalog suitable for the given frame."""

    assert dr_files or transformation

    catalog_info, query_center, outliers = get_catalog_info(
        transformation=transformation,
        dr_files=dr_files,
        header=header,
        configuration=configuration,
        **dr_path_substitutions,
    )
    with lock:
        if path.exists(catalog_info["fname"]):
            with fits.open(catalog_info["fname"]) as cat_fits:
                catalog_header = cat_fits[1].header
                # pylint: disable=too-many-boolean-expressions
                if (
                    numpy.abs(
                        catalog_header["EPOCH"]
                        - catalog_info["epoch"].to_value("yr")
                    )
                    > 0.25
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} '
                        f'has epoch {catalog_header["EPOCH"]!r}, '
                        f'but {catalog_info["epoch"]!r} is needed'
                    )

                if (
                    catalog_header["MAGEXPR"]
                    != catalog_info["magnitude_expression"]
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} has '
                        f'magnitude expression {catalog_header["MAGEXPR"]!r} '
                        f'instead of {catalog_info["magnitude_expression"]!r}'
                    )

                if (
                    catalog_header.get("MAGMIN") is not None
                    and len(catalog_info["magnitude_limit"]) == 2
                    and (
                        catalog_header["MAGMIN"]
                        > catalog_info["magnitude_limit"][0]
                    )
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} excludes '
                        f'sources brighter than {catalog_header["MAGMIN"]!r} '
                        f'but {catalog_info["magnitude_limit"][0]!r} are '
                        "required."
                    )

                if catalog_header.get("MAGMAX") is not None and (
                    catalog_header["MAGMAX"]
                    < catalog_info["magnitude_limit"][-1]
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} excludes '
                        f'sources fainter than {catalog_header["MAGMAX"]!r} but'
                        f' {catalog_info["magnitude_limit"][-1]!r} are '
                        "required."
                    )

                if catalog_header["WIDTH"] < catalog_info["width"].to_value(
                    units.deg
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} width '
                        f'{catalog_header["WIDTH"]!r} is less than the required'
                        f' {catalog_info["width"]!r}'
                    )
                if catalog_header["HEIGHT"] < catalog_info["height"].to_value(
                    units.deg
                ):
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} height '
                        f'{catalog_header["HEIGHT"]!r} is less than the '
                        f'required {catalog_info["height"]!r}'
                    )

                if (
                    catalog_header["RA"] - query_center["RA"]
                ) * units.deg * numpy.cos(
                    catalog_header["DEC"] * units.deg
                ) > configuration[
                    "pointing_precision"
                ] * units.deg:
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} center RA '
                        f'{catalog_header["RA"]!r} is too far from the '
                        f'required RA={query_center["RA"]!r}'
                    )

                if (
                    catalog_header["DEC"] - query_center["Dec"]
                ) * units.deg > configuration["pointing_precision"] * units.deg:
                    raise RuntimeError(
                        f'Catalog {catalog_info["fname"]} center Dec '
                        f'{catalog_header["DEC"]!r} is too far from the '
                        f'required Dec={query_center["Dec"]!r}'
                    )

                filter_expr = configuration["filter"]
                if filter_expr is None or header["CLRCHNL"] not in filter_expr:
                    filter_expr = []
                else:
                    filter_expr = ["(" + filter_expr[header["CLRCHNL"]] + ")"]

                if len(catalog_info["magnitude_limit"]) == 2 and (
                    catalog_header.get("MAGMIN") is None
                    or catalog_header["MAGMIN"]
                    < catalog_info["magnitude_limit"][0]
                ):
                    filter_expr.append(
                        f'(magnitude > {catalog_info["magnitude_limit"][0]!r})'
                    )
                if catalog_header.get("MAGMAX") is None or (
                    catalog_header["MAGMAX"]
                    > catalog_info["magnitude_limit"][-1]
                ):
                    filter_expr.append(
                        '(magnitude < {catalog_info["magnitude_limit"][-1]!r})'
                    )
                # pylint: enable=too-many-boolean-expressions

                return (
                    read_catalog_file(
                        cat_fits,
                        filter_expr=(
                            " and ".join(filter_expr) if filter_expr else None
                        ),
                        return_metadata=return_metadata,
                    ),
                    outliers,
                    catalog_info["fname"],
                )

        del catalog_info["ra_ind"]
        del catalog_info["dec_ind"]

        create_catalog_file(**catalog_info, verbose=True)
        return (
            read_catalog_file(
                catalog_info["fname"], return_metadata=return_metadata
            ),
            outliers,
            catalog_info["fname"],
        )


# pylint: enable=too-many-branches


def check_catalog_coverage(
    header, transformation, catalog_header, safety_margin
):
    """
    Return True iff te catalog covers the frame fully including a safety margin.

    Args:
        header(dict-like):    The header of the frame being astrometried.

        transformation(dict):    The transformation to assume applies to the
            given image.  Should contain entries for ``'trans_x'``,
            ``'trans_y'``, ``'ra_cent'`` and ``'dec_cent'``.

        catalog_header(dict-like):    The header of the catalog being used for
            astrometry.

        safety_margin(float):    The absolute values of xi and eta
            (relative to the catalog center) corresponding to the corners of the
            frame increased by this fraction must be smaller than the field of
            view of the catalogfor the frame to be considered covered.

    Returns:
        bool:
            Whether the catalog covers the frame fully including the safety
            margin.
    """

    width, height = get_max_abs_corner_xi_eta(
        header=header,
        transformation=transformation,
        center={"RA": catalog_header["RA"], "Dec": catalog_header["DEC"]},
    )
    factor = 2.0 * (1.0 + safety_margin)
    width *= factor
    height *= factor
    _logger.debug(
        "Coverage with safety requires FOV (width: %s deg, height: %s def)",
        repr(width),
        repr(height),
    )
    return width < catalog_header["WIDTH"] and height < catalog_header["HEIGHT"]


def show_stars(catalog_fname):
    """Show the stars in the catalog on a 3-D plot of the sky."""

    phi, theta = numpy.mgrid[
        0.0 : numpy.pi / 2.0 : 10j, 0.0 : 2.0 * numpy.pi : 10j
    ]
    sphere_x = numpy.sin(phi) * numpy.cos(theta)
    sphere_y = numpy.sin(phi) * numpy.sin(theta)
    sphere_z = numpy.cos(phi)

    fig = pyplot.figure()
    axes = fig.add_subplot(111, projection="3d")
    pyplot.gca().plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=1,
        cstride=1,
        color="c",
        alpha=0.6,
        linewidth=1,
    )

    stars = read_catalog_file(catalog_fname)

    stars_x = numpy.cos(numpy.radians(stars["Dec"])) * numpy.cos(
        numpy.radians(stars["RA"])
    )
    stars_y = numpy.cos(numpy.radians(stars["Dec"])) * numpy.sin(
        numpy.radians(stars["RA"])
    )
    stars_z = numpy.sin(numpy.radians(stars["Dec"]))

    axes.scatter(stars_x, stars_y, stars_z, color="k", s=20)

    pyplot.show()


def main(config):
    """Avoid polluting global namespace."""

    if config.run_doctests:
        doctest.testmod(verbose=config.verbose)
        return

    kwargs = {
        "ra": config.ra * units.deg,
        "dec": config.dec * units.deg,
        "width": config.width * units.deg,
        "height": config.height * units.deg,
        "epoch": config.epoch * units.yr if config.epoch is not None else None,
        "magnitude_expression": config.magnitude_expression,
        "magnitude_limit": config.magnitude_limit,
        "columns": config.columns,
        "verbose": config.verbose,
        "overwrite": config.overwrite,
        "count_only": config.count_only,
    }
    if config.extra_condition is not None:
        kwargs["condition"] = config.extra_condition
    create_catalog_file(config.catalog_fname, **kwargs)
    if config.show_stars and not config.count_only:
        show_stars(config.catalog_fname)


if __name__ == "__main__":

    main(parse_command_line())
