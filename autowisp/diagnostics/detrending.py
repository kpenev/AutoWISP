"""Functions for extracting diagnostics for the detrending steps."""

from sqlalchemy import select
from sqlalchemy.orm import aliased

import pandas
import numpy

from autowisp import Evaluator
from autowisp.database.interface import Session
from autowisp.catalog import read_catalog_file
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    MasterType,\
    MasterFile
#pylint: enable=no-name-in-module


def detect_stat_columns(stat, num_stat_columns, skip_first_stat=False):
    """
    Automatically detect the relevant columns in the statistics file.

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

    column_mask = numpy.arange(0,
                               num_stat * columns_per_set,
                               columns_per_set,
                               dtype=int)
    if skip_first_stat:
        column_mask = column_mask[1:]

    num_unrejected = column_mask + 2
    scatter = column_mask + 5
    formal_error = columns_per_set * num_stat + 3 + column_mask

    for column_selection, expected_kind in [(num_unrejected, 'iu'),
                                            (scatter, 'f'),
                                            (formal_error, 'f')]:
        print('column selection: ' + repr(column_selection))
        print('expected kind: ' + repr(expected_kind))
        check_dtype = stat[column_selection].dtypes.unique()
        print('check_dtype: ' + repr(check_dtype))
        assert check_dtype.size == 1
        assert check_dtype[0].kind in expected_kind

    return num_unrejected, scatter, formal_error


def read_magfit_stat_data(master_id):
    """Return the statistics and catalog generated during given magfit step."""

    master_file_alias = aliased(MasterFile)
    master_select = select(
        MasterFile.filename
    ).join(
        master_file_alias,
        MasterFile.progress_id == master_file_alias.progress_id
    ).join(
        MasterType,
        MasterFile.type_id == MasterType.id
    ).where(
        master_file_alias.id == master_id
    )
    #False positive
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        stat_fname = db_session.scalar(
            master_select.where(MasterType.name == 'magfit_stat')
        )
        catalog_fname = db_session.scalar(
            master_select.where(MasterType.name == 'magfit_catalog')
        )

    data = read_catalog_file(catalog_fname, add_gnomonic_projection=True)
    num_cat_columns = len(data.columns)
    data = data.join(
        pandas.read_csv(stat_fname,
                        sep=r'\s+',
                        header=None,
                        index_col=0),
        how='inner'
    )
    return data, num_cat_columns


def get_magfit_performance_data(master_id,
                                min_unrejected_fraction,
                                magnitude_expression,
                                skip_first_stat):
    """Return all data required for magnitude fitting performance plots."""

    data, num_cat_columns = read_magfit_stat_data(master_id)
    (
        num_unrejected_columns,
        scatter_columns,
        expected_scatter_columns
    ) = detect_stat_columns(data,
                            len(data.columns) - num_cat_columns,
                            skip_first_stat)

    min_unrejected = numpy.min(data[num_unrejected_columns], 1)
    many_unrejected = (min_unrejected
                       >
                       min_unrejected_fraction * numpy.max(min_unrejected))
    data = data[many_unrejected]

    scatter = data[scatter_columns]
    scatter[scatter == 0.0] = numpy.nan
    data.insert(len(data.columns),
                'best_index',
                numpy.nanargmin(scatter, 1))
    data.insert(len(data.columns),
                'best_scatter',
                10.0**(numpy.nanmin(scatter, 1) / 2.5) - 1.0)

    if expected_scatter_columns is not None:
        expected_scatter = data[expected_scatter_columns]
        expected_scatter[expected_scatter == 0.0] = numpy.nan
        data.insert(len(data.columns),
                    'expected_scatter',
                    10.0**(numpy.nanmin(expected_scatter, 1) / 2.5) - 1.0)

    data.insert(len(data.columns),
                'magnitudes',
                Evaluator(data)(magnitude_expression))

    return data
