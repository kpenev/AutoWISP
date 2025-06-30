"""Functions for extracting diagnostics for the detrending steps."""

from sqlalchemy import select
from sqlalchemy.orm import aliased

import pandas
import numpy
from numpy.lib.recfunctions import append_fields

from autowisp import Evaluator
from autowisp.database.interface import Session
from autowisp.catalog import read_catalog_file
from autowisp.light_curves.apply_correction import load_correction_statistics

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import MasterType, MasterFile

# pylint: enable=no-name-in-module


def detect_magfit_stat_columns(stat, num_stat_columns, skip_first_stat=False):
    """
    Automatically detect the relevant columns in the magfit statistics file.

    Args:
        stat(pandas.DataFrame):     The statistics .

    Returns:
        tuple(list, list):
            list:
                The indices of the columns (starting from zero) containing the
                number of unrejected frames for each extracted photometry.

            list:
                The indices of the columns (starting from zero) containing the
                scatter (e.g. median deviation around the median for magfit).

            list:
                The indices of the columns (starting from zero) containing the
                median of the formal magnitude error, or None if that is not
                tracked.
    """

    columns_per_set = 5
    assert num_stat_columns % columns_per_set == 0
    num_stat = num_stat_columns // columns_per_set
    assert num_stat % 2 == 0
    num_stat //= 2

    column_mask = numpy.arange(
        0, num_stat * columns_per_set, columns_per_set, dtype=int
    )
    if skip_first_stat:
        column_mask = column_mask[1:]

    num_unrejected = column_mask + 2
    scatter = column_mask + 5
    formal_error = columns_per_set * num_stat + 3 + column_mask

    for column_selection, expected_kind in [
        (num_unrejected, "iu"),
        (scatter, "f"),
        (formal_error, "f"),
    ]:
        check_dtype = stat[column_selection].dtypes.unique()
        assert check_dtype.size == 1
        assert check_dtype[0].kind in expected_kind

    return num_unrejected, scatter, formal_error


def read_stat_data(catalog_fname, stat_fname):
    """Return the performance statistics and catalog for a pipeline step."""

    data = read_catalog_file(catalog_fname, add_gnomonic_projection=True)

    num_cat_columns = len(data.columns)
    data = data.join(
        pandas.read_csv(stat_fname, sep=r"\s+", header=None, index_col=0),
        how="inner",
    )
    return data, num_cat_columns


def find_magfit_stat_catalog(master_id):
    """Return the statistics and catalog generated during given magfit step."""

    master_file_alias = aliased(MasterFile)
    master_select = (
        select(MasterFile.filename)
        .join(
            master_file_alias,
            MasterFile.progress_id == master_file_alias.progress_id,
        )
        .join(MasterType, MasterFile.type_id == MasterType.id)
        .where(master_file_alias.id == master_id)
    )
    # False positive
    # pylint: disable=no-member
    with Session.begin() as db_session:
        # pylint: enable=no-member
        stat_fname = db_session.scalar(
            master_select.where(MasterType.name == "magfit_stat")
        )
        catalog_fname = db_session.scalar(
            master_select.where(MasterType.name == "magfit_catalog")
        )

    return catalog_fname, stat_fname


def get_detrending_performance_data(
    catalog_fname,
    stat_fname,
    detrending_mode,
    *,
    min_unrejected_fraction,
    magnitude_expression,
    skip_first_stat,
):
    """Return all data required for magnitude fitting performance plots."""

    if detrending_mode.lower() == "mfit":
        data, num_cat_columns = read_stat_data(catalog_fname, stat_fname)
        (num_unrejected_columns, scatter_columns, expected_scatter_columns) = (
            detect_magfit_stat_columns(
                data, len(data.columns) - num_cat_columns, skip_first_stat
            )
        )
    else:
        data = load_correction_statistics(stat_fname, catalog_fname)
        num_unrejected_columns = "num_finite"
        scatter_columns = "rms"
        expected_scatter_columns = None

    min_unrejected = numpy.min(data[num_unrejected_columns], 1)
    many_unrejected = min_unrejected > min_unrejected_fraction * numpy.max(
        min_unrejected
    )
    data = data[many_unrejected]

    scatter = data[scatter_columns]
    new_data = {
        "best_index": numpy.nanargmin(scatter, 1),
        "best_scatter": 10.0 ** (numpy.nanmin(scatter, 1) / 2.5) - 1.0,
        "magnitudes": Evaluator(data)(magnitude_expression),
    }
    if detrending_mode.lower() == "mfit":
        for column, values in new_data.items():
            data.insert(len(data.columns), column, values)
    else:
        data = append_fields(
            data,
            ["best_index", "best_scatter", "magnitudes"],
            [new_data[c] for c in ["best_index", "best_scatter", "magnitudes"]],
        )

    if expected_scatter_columns is not None:
        expected_scatter = data[expected_scatter_columns]
        expected_scatter[expected_scatter == 0.0] = numpy.nan
        data.insert(
            len(data.columns),
            "expected_scatter",
            10.0 ** (numpy.nanmin(expected_scatter, 1) / 2.5) - 1.0,
        )

    return data
