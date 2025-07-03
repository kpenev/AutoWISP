"""Convenience functions for interacting with DR files."""

from ctypes import c_uint, c_double, c_int, c_ubyte
import re
import logging

import numpy

from autowisp import fit_expression
from autowisp.data_reduction.source_extracted_psf_map import (
    SourceExtractedPSFMap,
)

key_io_tree_to_dr = {
    "projsrc.x": "srcproj.x",
    "projsrc.y": "srcproj.y",
    "bg.model": "bg.cfg.model",
    "bg.value": "bg.value",
    "bg.error": "bg.error",
    "psffit.min_bg_pix": "shapefit.cfg.src.min_bg_pix",
    "psffit.gain": "shapefit.cfg.gain",
    "psffit.magnitude_1adu": "shapefit.cfg.magnitude_1adu",
    "psffit.grid": "shapefit.cfg.psf.bicubic.grid",
    "psffit.initial_aperture": "shapefit.cfg.psf.bicubic.initial_aperture",
    "psffit.max_abs_amplitude_change": "shapefit.cfg.psf.bicubic.max_abs_amplitude_change",
    "psffit.max_rel_amplitude_change": "shapefit.cfg.psf.bicubic.max_rel_amplitude_change",
    "psffit.pixrej": "shapefit.cfg.psf.bicubic.pixrej",
    "psffit.smoothing": "shapefit.cfg.psf.bicubic.smoothing",
    "psffit.max_chi2": "shapefit.cfg.psf.max-chi2",
    "psffit.max_iterations": "shapefit.cfg.psf.max_iterations",
    "psffit.min_convergence_rate": "shapefit.cfg.psf.min_convergence_rate",
    "psffit.model": "shapefit.cfg.psf.model",
    "psffit.srcpix_cover_bicubic_grid": "shapefit.cfg.src.cover_bicubic_grid",
    "psffit.srcpix_max_aperture": "shapefit.cfg.src.max_aperture",
    # TODO: fix tree entry name to psffit.src_max_count
    "psffit.srcpix_max_count": "shapefit.cfg.src.max_count",
    "psffit.srcpix_min_pix": "shapefit.cfg.src.min_pix",
    "psffit.srcpix_max_pix": "shapefit.cfg.src.max_pix",
    "psffit.srcpix_max_sat_frac": "shapefit.cfg.src.max_sat_frac",
    "psffit.srcpix_min_signal_to_noise": "shapefit.cfg.src.min_signal_to_noise",
    "psffit.mag": "shapefit.magnitude",
    "psffit.mag_err": "shapefit.magnitude_error",
    "psffit.chi2": "shapefit.chi2",
    "psffit.sigtonoise": "shapefit.signal_to_noise",
    "psffit.npix": "shapefit.num_pixels",
    "psffit.quality": "shapefit.quality_flag",
    "psffit.psfmap": "shapefit.map_coef",
    "apphot.const_error": "apphot.cfg.error_floor",
    "apphot.aperture": "apphot.cfg.aperture",
    "apphot.gain": "apphot.cfg.gain",
    "apphot.magnitude-1adu": "apphot.cfg.magnitude_1adu",
    "apphot.mag": "apphot.magnitude",
    "apphot.mag_err": "apphot.magnitude_error",
    "apphot.quality": "apphot.quality_flag",
}

_dtype_dr_to_io_tree = {
    numpy.string_: str,
    numpy.uint: c_uint,
    numpy.uint8: c_ubyte,
    numpy.int32: c_int,
    numpy.float64: c_double,
}

_logger = logging.getLogger(__name__)


def _parse_grid_str(grid_str):
    """Parse the grid string entry from the astrowisp.IOTree."""

    result = [
        numpy.array([float(v) for v in sub_grid.split(",")])
        for sub_grid in grid_str.split(";")
    ]
    if len(result) == 1:
        return [result, result]

    assert len(result) == 2
    return result


def _add_shapefit_map(
    dr_file, fit_terms_expression, shape_fit_result_tree, **path_substitutions
):
    """
    Add the coefficients defining the PSF/PRF map to DR file.

    Args:
        shape_fit_result_tree:    See same name argument to
            add_star_shape_fit().

        path_substitutions:    See same name argument to
            _add_shapefit_sources().

    Returns:
        None
    """

    grid = _parse_grid_str(shape_fit_result_tree.get("psffit.grid", str))
    for direction, splits in zip(["x", "y"], grid):
        dr_file.add_attribute(
            "shapefit.cfg.psf.bicubic.grid." + direction,
            splits,
            if_exists="error",
            **path_substitutions,
        )
    dr_file.add_attribute(
        "shapefit.cfg.psf.terms",
        fit_terms_expression,
        if_exists="error",
        **path_substitutions,
    )

    num_terms = fit_expression.Interface(fit_terms_expression).number_terms()
    coefficients = shape_fit_result_tree.get(
        "psffit.psfmap",
        shape=(4, grid[0].size - 2, grid[1].size - 2, num_terms),
    )
    dr_file.add_dataset(
        "shapefit.map_coef",
        coefficients,
        if_exists="error",
        **path_substitutions,
    )


def _auto_add_tree_quantities(
    dr_file,
    result_tree,
    num_sources,
    skip_quantities,
    image_index=0,
    **path_substitutions,
):
    """
    Best guess for how to add tree quantities to DR file.

    Args:
        result_tree(astrowisp.IOTree):    The tree to extract quantities to
            add.

        num_sources(int):    The number of sources (assumed to be ththe
            length of all datasets).

        skip_quantities(compiled rex matcher):    Quantities matching this
            regular expression will not be added to the DR file by this
            function.

        image_index(int):    For quantities which are split by image, only
            the values associated to this image index will be added.

    Returns:
        None
    """

    indexed_rex = re.compile(r".*\.(?P<image_index_str>[0-9]+)$")
    apphot_indexed_rex = re.compile(
        r"|apphot\..*\."
        r"(?P<image_index_str>[0-9]+)\."
        r"(?P<ap_index_str>[0-9]+)$"
    )
    for quantity_name in result_tree.defined_quantity_names():
        indexed_match = apphot_indexed_rex.fullmatch(quantity_name)
        if indexed_match:
            path_substitutions["aperture_index"] = int(
                indexed_match["ap_index_str"]
            )
        else:
            path_substitutions.pop("aperture_index", 0)
            indexed_match = indexed_rex.fullmatch(quantity_name)

        if indexed_match:
            if int(indexed_match["image_index_str"]) == image_index:
                key_quantity = quantity_name[
                    : indexed_match.start("image_index_str") - 1
                ]
            else:
                continue
        else:
            key_quantity = quantity_name

        dr_key = key_io_tree_to_dr.get(key_quantity, key_quantity)

        for element_type in ["dataset", "attribute", "link"]:
            if (
                dr_key in dr_file.elements[element_type]
                and skip_quantities.match(key_quantity) is None
            ):
                _logger.debug(
                    "Getting %s (dtype = %s, tree dtype = %s) from tree",
                    repr(quantity_name),
                    repr(dr_file.get_dtype(dr_key)),
                    repr(_dtype_dr_to_io_tree[dr_file.get_dtype(dr_key)]),
                )
                value = result_tree.get(
                    quantity_name,
                    _dtype_dr_to_io_tree[dr_file.get_dtype(dr_key)],
                    shape=(num_sources if element_type == "dataset" else None),
                )
                _logger.debug(
                    "Saving %s to DR file %s",
                    repr(quantity_name),
                    dr_file.filename,
                )

                # TODO: add automatic detection for versions
                getattr(dr_file, "add_" + element_type)(
                    dr_key, value, if_exists="error", **path_substitutions
                )
                break


def _auto_delete_tree_quantities(
    dr_file, skip_quantities, **path_substitutions
):
    """Remove all elements from the DR file not matching skip_quantities rex."""

    for tree_key, dr_key in key_io_tree_to_dr.items():
        if skip_quantities.match(tree_key) is not None:
            continue
        for element_type in ["dataset", "attribute", "link"]:
            if dr_key in dr_file.elements[element_type]:
                getattr(dr_file, "delete_" + element_type)(
                    dr_key, **path_substitutions
                )


def _get_shapefit_map_grid(dr_file, **path_substitutions):
    """Return the grid used to represent star shape from this DR file."""

    return [
        dr_file.get_attribute(
            "shapefit.cfg.psf.bicubic.grid.x", **path_substitutions
        ),
        dr_file.get_attribute(
            "shapefit.cfg.psf.bicubic.grid.y", **path_substitutions
        ),
    ]


def _get_shapefit_map(dr_file, **path_substitutions):
    """
    Read the map of the shapes of point sources from this DR file.

    Args:
        path_substitutions:    See get_aperture_photometry_inputs().

    Returns:
        2-D numpy.array(dtype=numpy.float64):
            The grid used to represent source shapes.

        4-D numpy.array(dtype=numpy.float64):
            The coefficients of the shape map. See the C++ documentation for
            more details of the layout.

        str:
            The expression specifying the terms to include in the PSF/PRF
            dependence.
    """
    return (
        _get_shapefit_map_grid(dr_file, **path_substitutions),
        dr_file.get_dataset("shapefit.map_coef", **path_substitutions),
        dr_file.get_attribute("shapefit.cfg.psf.terms", **path_substitutions),
    )


def add_star_shape_fit(
    dr_file,
    *,
    fit_terms_expression,
    shape_fit_result_tree,
    num_sources,
    image_index=0,
    **path_substitutions,
):
    """
    Add the results of a star shape fit to the DR file.

    Args:
        shape_fit_result_tree(astrowisp.IOTree):    The return
            value of a successful call of astrowisp.FitStarShape.fit().

        num_sources (int):    The number of surces used in the fit (used to
            determine the expected size of datasets).

        image_index (int):    The index of the image whose DR file is being
            filled within the input list of images passed to PSF/PRF
            fitting.

        fit_variables (iterable):    The variables that were used in the
            fit in the order in which they appear in the tree.

    Returns:
        None
    """

    _add_shapefit_map(
        dr_file,
        fit_terms_expression,
        shape_fit_result_tree,
        **path_substitutions,
    )
    dr_file.add_attribute(
        key_io_tree_to_dr["psffit.srcpix_cover_bicubic_grid"],
        (
            shape_fit_result_tree.get(
                "psffit.srcpix_cover_bicubic_grid", str
            ).lower()
            == "true"
        ),
        if_exists="error",
        **path_substitutions,
    )
    # TODO: set this from command line, use it in fitting and fix here!
    dr_file.add_attribute(
        "shapefit.cfg.psf.ignore_dropped",
        False,
        if_exists="error",
        **path_substitutions,
    )

    _auto_add_tree_quantities(
        dr_file,
        result_tree=shape_fit_result_tree,
        num_sources=num_sources,
        skip_quantities=re.compile(
            "|".join(
                [
                    r"^psffit\.variables$",
                    r"^psffit\.grid$",
                    r"^psffit\.terms$",
                    r"^psffit\.psfmap$",
                    r"^psffit.srcpix_cover_bicubic_grid$",
                    r"^projsrc\.",
                    r"^apphot\.",
                ]
            )
        ),
        image_index=image_index,
        **path_substitutions,
    )


def delete_star_shape_fit(dr_file, **path_substitutions):
    """Delete all DR elements added by `add_star_shape_fit()`."""

    dr_file.delete_attribute("shapefit.cfg.psf.terms", **path_substitutions)

    dr_file.delete_attribute(
        "shapefit.cfg.psf.ignore_dropped", **path_substitutions
    )
    _auto_delete_tree_quantities(
        dr_file,
        skip_quantities=re.compile(
            "|".join(
                [
                    r"^psffit\.variables$",
                    r"^psffit\.terms$",
                    r"^projsrc\.",
                    r"^apphot\.",
                ]
            )
        ),
        **path_substitutions,
    )


def get_aperture_photometry_inputs(dr_file, **path_substitutions):
    """
    Return all required information for aperture photometry from PSF fit DR.

    Args:
        path_substitutions:    Values to substitute in the paths to the
            datasets and attributes containing shape fit informaiton
            (usually versions of various components).

    Returns:
        dict:
            All parameters required by
            astrowisp.IOTree.set_aperture_photometry_inputs() directly
            passable to that method using **.

        str:
            The expression defining which terms the PSF/PRF depends on.
    """

    _logger.debug(
        "Getting apphot inputs from %s with path substitutions: %s",
        repr(dr_file.filename),
        repr(path_substitutions),
    )
    result = {}
    result["source_data"] = dr_file.get_source_data(
        magfit_iterations=[0],
        shape_fit=True,
        apphot=False,
        shape_map_variables=True,
        string_source_ids=True,
        **path_substitutions,
    )
    result["magnitude_1adu"] = dr_file.get_attribute(
        "shapefit.cfg.magnitude_1adu", **path_substitutions
    )
    (
        result["star_shape_grid"],
        result["star_shape_map_coefficients"],
        shape_map_terms_expression,
    ) = _get_shapefit_map(dr_file, **path_substitutions)

    if not isinstance(shape_map_terms_expression, str):
        shape_map_terms_expression = shape_map_terms_expression.decode()

    result["star_shape_map_terms"] = fit_expression.Interface(
        shape_map_terms_expression
    )(result["source_data"]).T

    return result, shape_map_terms_expression


def fill_aperture_photometry_input_tree(
    dr_file, tree, shapefit_version=0, srcproj_version=0, background_version=0
):
    """
    Fill a astrowisp.IOTree with shape fit info for aperture photometry.

    Args:
        tree(astrowisp.IOTree):    The tree to fill.

        shapefit_version(int):    The version of the star shape fit results
            stored in the file to use when initializing the tree.

        srcproj_version(int):    The version of the projected sources to
            assume was used for shape fitting, and to use for aperture
            photometry.

        background_vesrion(int):    The version of the background extraction
            to assume was used for shape fitting, and to use for aperture
            photometry.

    Returns:
        int:
            The number of sources added to the tree.
    """

    aperture_photometry_inputs = get_aperture_photometry_inputs(
        dr_file,
        shapefit_version=shapefit_version,
        srcproj_version=srcproj_version,
        background_version=background_version,
    )[0]
    aperture_photometry_inputs["source_data"].rename(
        columns={
            "shapefit_" + what + "_mfit000": what
            for what in ["mag", "mag_err", "phot_flag"]
        },
        inplace=True,
    )
    aperture_photometry_inputs["source_data"] = aperture_photometry_inputs[
        "source_data"
    ].to_records()
    _logger.debug(
        "Adding aperture photometry inputs to tree: %s",
        repr(aperture_photometry_inputs),
    )
    tree.set_aperture_photometry_inputs(**aperture_photometry_inputs)
    return aperture_photometry_inputs["source_data"].size


def add_aperture_photometry(
    dr_file,
    apphot_result_tree,
    num_sources,
    num_apertures,
    **path_substitutions,
):
    """
    Add the results of aperture photometry to the DR file.

    Args:
        apphot_result_tree:(astrowisp.IOTree):    The tree which
            was passed to the :class:astrowisp.SubPixPhot instance which did
            the aperture photometry (i.e. where the results were added).

        num_sources(int):    The number of sources for which aperture
            photometry was done. The same as the number of sources the star
            shape fitting which was used by the aperture photometry was
            performed on for the photometered image.

        num_apertures(int):    The number of apertures for which photometry
            was extracted.

    Returns:
        None
    """

    for aperture_index, aperture in enumerate(
        apphot_result_tree.get(
            "apphot.aperture", c_double, shape=(num_apertures,)
        )
    ):
        dr_file.add_attribute(
            "apphot.cfg.aperture",
            aperture,
            if_exists="error",
            apphot_version=0,
            aperture_index=aperture_index,
        )

    _auto_add_tree_quantities(
        dr_file,
        result_tree=apphot_result_tree,
        num_sources=num_sources,
        skip_quantities=re.compile(r"(?!apphot\.)|^apphot.aperture$"),
        **path_substitutions,
    )


def delete_aperture_photometry(dr_file, num_apertures, **path_substitutions):
    """Delete all DR elements of an aperture photometry."""

    for aperture_index in range(num_apertures):
        _auto_delete_tree_quantities(
            dr_file,
            re.compile(r"(?!apphot\.)|^apphot.aperture$"),
            aperture_index=aperture_index,
            **path_substitutions,
        )


def get_source_extracted_psf_map(dr_file, **path_substitutions):
    """
    Return functions giving the source extraction PSF map in dr_file.

    Args:
        path_substitutions:    Substitution arguments required to resolve
            the path to the relevant datasets/attributes.

    Returns:
        SourceExtractedPSFMap:
            The PSF map derived from extracted sources stored in this DR
            file.
    """

    return SourceExtractedPSFMap(
        psf_parameters=[
            param_name.decode()
            for param_name in dr_file.get_attribute(
                "srcextract.psf_map.cfg.psf_params", **path_substitutions
            )
        ],
        terms_expression=dr_file.get_attribute(
            "srcextract.psf_map.cfg.terms", **path_substitutions
        ),
        coefficients=dr_file.get_dataset(
            "srcextract.psf_map", **path_substitutions
        ),
    )
