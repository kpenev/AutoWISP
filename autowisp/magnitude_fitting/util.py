"""Functions for formatting magnitude fitting inputs as needed."""

import os
from itertools import count

import numpy
from numpy.lib import recfunctions
from astropy.io import fits

from autowisp.project_paths import resolve_path

_PHOT_QUANTITIES = ("mag", "mag_err", "phot_flag")


def _parse_phot_column(colname):
    """Return the magfit iteration of the given phot column (-1 if not phot)."""
    is_phot = False
    for phot_quantity in _PHOT_QUANTITIES:
        is_phot = is_phot or (colname.find("_" + phot_quantity + "_") > 0)
    if not is_phot:
        return None, -1

    colname_head, colname_tail = colname.rsplit("_", 1)
    assert colname_tail.startswith("mfit")
    quantity = colname_head.split("_", 1)[1]
    assert quantity in _PHOT_QUANTITIES
    return quantity, int(colname_tail[4:])


def _init_magfit_sources(source_data):
    """Return empty array properly formatted to hold the given sources."""

    dtype = []
    if "source_id" not in source_data:
        # uint64, not the platform-dependent numpy.uint (32-bit on Windows):
        # the middle component holds the full Gaia source id (~60 bits).
        dtype.append(("source_id", numpy.uint64, 3))

    num_phot = 0
    found_magfit_iterations = set()
    for colname in source_data.columns:
        phot_quantity, magfit_iter = _parse_phot_column(colname)
        if phot_quantity is None:
            if not colname.startswith("hat_id_"):
                dtype.append(
                    (
                        colname,
                        (
                            "S"
                            if source_data[colname].dtype.kind == "O"
                            else source_data[colname].dtype
                        ),
                    )
                )
        elif phot_quantity == "mag":
            num_phot += 1
            found_magfit_iterations.add(magfit_iter)
    assert num_phot % len(found_magfit_iterations) == 0
    num_phot //= len(found_magfit_iterations)
    dtype.extend(
        [
            (
                phot_quantity,
                (numpy.uint if phot_quantity == "phot_flag" else numpy.float64),
                (len(found_magfit_iterations), num_phot),
            )
            for phot_quantity in _PHOT_QUANTITIES
        ]
    )
    return (
        numpy.empty(shape=(len(source_data),), dtype=dtype),
        num_phot,
        found_magfit_iterations,
    )


def get_magfit_sources(
    data_reduction_file, magfit_iterations="all", **path_substitutions
):
    """Return the sources in the given DR file formatted for magfit."""

    source_data = data_reduction_file.get_source_data(
        magfit_iterations=magfit_iterations,
        string_source_ids=False,
        shape_fit=data_reduction_file.has_shape_fit(
            accept_zeropsf=False, **path_substitutions
        ),
        **path_substitutions,
    )
    source_data.reset_index(inplace=True)
    result, num_phot, magfit_iterations = _init_magfit_sources(source_data)
    magfit_iterations = sorted(magfit_iterations)

    phot_index = {
        quantity: [0] * len(magfit_iterations) for quantity in _PHOT_QUANTITIES
    }
    for colname, colvalues in source_data.items():
        phot_quantity, magfit_iter = _parse_phot_column(colname)
        if phot_quantity is not None:
            magfit_iter = magfit_iterations.index(magfit_iter)
            result[phot_quantity][
                :, magfit_iter, phot_index[phot_quantity][magfit_iter]
            ] = colvalues
            phot_index[phot_quantity][magfit_iter] += 1
        elif colname == "hat_id_prefix":
            result["source_id"][:, 0][colvalues == b"HAT"] = 0
            result["source_id"][:, 0][colvalues == b"UCAC4"] = 1
            assert (
                numpy.logical_and(
                    colvalues != b"UCAC4", colvalues != b"HAT"
                ).sum()
                == 0
            )
        elif colname == "hat_id_field":
            result["source_id"][:, 1] = colvalues
        elif colname == "hat_id_source":
            result["source_id"][:, 2] = colvalues
        else:
            result[colname] = colvalues

    for phot_quantity in _PHOT_QUANTITIES:
        for phot_i in phot_index[phot_quantity]:
            assert phot_i == num_phot

    return result


def get_single_photref(data_reduction_file, **path_substitutions):
    """Create a photometric reference out of the raw photometry in a DR file."""

    source_data = get_magfit_sources(
        data_reduction_file, magfit_iterations=[0], **path_substitutions
    )
    return {
        (
            source["source_id"]
            if source["source_id"].shape == ()
            else tuple(source["source_id"])
        ): {
            "x": source["x"],
            "y": source["y"],
            "mag": source["mag"],
            "mag_err": source["mag_err"],
        }
        for source in source_data
    }


def get_master_photref(photref_fname):
    """Read a FITS photometric reference created by MasterPhotrefCollector."""

    result = {}
    with fits.open(photref_fname, "readonly") as photref_fits:
        # Masters written before the tables were named have them unnamed.
        phot_references = [
            hdu for hdu in photref_fits[1:] if hdu.name in ("MPHOTREF", "")
        ]
        num_photometries = len(phot_references)
        for phot_ind, phot_reference in enumerate(phot_references):
            if "source_id" in phot_reference.data.dtype.names:
                source_ids = phot_reference.data["source_id"]
            else:
                source_ids = zip(
                    phot_reference.data["IDprefix"].astype("int"),
                    phot_reference.data["IDfield"],
                    phot_reference.data["IDsource"],
                )
            for source_index, source_id in enumerate(source_ids):
                if source_id not in result:
                    result[source_id] = {
                        "mag": numpy.full(
                            (1, num_photometries), numpy.nan, numpy.float64
                        ),
                        "mag_err": numpy.full(
                            (1, num_photometries), numpy.nan, numpy.float64
                        ),
                    }
                result[source_id]["mag"][0, phot_ind] = phot_reference.data[
                    "magnitude"
                ][source_index]
                result[source_id]["mag_err"][0, phot_ind] = phot_reference.data[
                    "mediandev"
                ][source_index]
    return result


def format_master_catalog(cat_sources, source_id_parser=None):
    """Return the catalogue info in the given file formatted for magfitting."""

    cat_sources = cat_sources.to_records()

    try:
        cat_ids = cat_sources["source_id"]
        cat_sources = recfunctions.drop_fields(
            cat_sources, "source_id", usemask=False
        )
        return dict(zip(cat_ids, cat_sources))
    except KeyError:
        cat_ids = cat_sources["ID"]
        cat_sources = recfunctions.drop_fields(cat_sources, "ID", usemask=False)
        return {
            source_id_parser(source_id): source_data
            for source_id, source_data in zip(cat_ids, cat_sources)
        }


def get_path_substitutions(configuration, sphotref_header):
    """Return the path substitutions to find magfit datasets."""

    result = {
        what + "_version": configuration[what + "_version"]
        for what in ["shapefit", "srcproj", "apphot", "background", "magfit"]
    }
    if configuration["master_photref_fname"] is not None:
        # dict() first: a header may repeat a keyword, which ** would pass
        # to format() once per copy.
        fname_substitutions = dict(sphotref_header)
        fname_substitutions.update(result)
        for iteration in count():
            fname_substitutions["magfit_iteration"] = iteration
            if (
                configuration["master_photref_fname_format"].format_map(
                    fname_substitutions
                )
                == configuration["master_photref_fname"]
            ):
                # Master iterNNN is built after pass NNN and fit against in
                # pass NNN + 1, so a fit against it belongs at that index.
                result["magfit_iteration"] = iteration + 1
                break
            if iteration >= configuration["max_magfit_iterations"]:
                raise ValueError(
                    "Master photometric reference "
                    f"{configuration['master_photref_fname']!r} does not appear"
                    " to follow the specified filename format: "
                    f"{configuration['master_photref_fname_format']!r}!"
                )
    return result


def read_exclusions(exclusion_fname):
    """
    Return the DR files an exclusion list names, as resolved paths.

    Args:
        exclusion_fname(str or None):    The exclusion list: one DR file per
            line, possibly starting with ``{PROJHOME}``, blank lines ignored.
            None excludes nothing.

    Returns:
        set:
            The ``os.path.realpath`` of each listed DR file, so that a file
            matches however its path is spelled.
    """

    if exclusion_fname is None:
        return set()
    with open(exclusion_fname, encoding="utf-8") as exclusion_list:
        return {
            os.path.realpath(resolve_path(line.strip()))
            for line in exclusion_list
            if line.strip()
        }
