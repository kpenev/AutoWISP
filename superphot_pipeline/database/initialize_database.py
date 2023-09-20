#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

from argparse import ArgumentParser
import re
from sqlalchemy.exc import OperationalError

from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

from superphot_pipeline.database.initialize_data_reduction_structure import\
    get_default_data_reduction_structure
from superphot_pipeline.database.initialize_light_curve_structure import\
    get_default_light_curve_structure

def parse_command_line():
    """Parse the commandline optinos to attributes of an object."""

    parser = ArgumentParser(
        description='Initialize the database for first time use of the '
        'pipeline.'
    )
    parser.add_argument(
        '--drop-all-tables',
        action='store_true',
        help='If passed all pipeline tables are deleted before new ones are '
        'created'
    )
    parser.add_argument(
        '--drop-hdf5-structure-tables', '--drop-structure',
        action='store_true',
        help='If passed, tables defining the structure of HDF5 files are '
        'dropped first and then re-created and filled. Otherwise, if tables '
        'exist, their contents is not modified.'
    )
    return parser.parse_args()

def add_default_hdf5_structures(data_reduction=True, light_curve=True):
    """
    Add a default HDF5 structure to the database.

    Args:
        data_reduction(bool):    Should the structure of data reduction files be
            initialized?

        light_curve(bool):    Should the structure of light curve files be
            initialized?
    """

    with db_session_scope() as db_session:
        if data_reduction:
            db_session.add(get_default_data_reduction_structure())
        if light_curve:
            db_session.add(get_default_light_curve_structure(db_session))

def drop_tables_matching(pattern):
    """Drop tables with names matching a pre-compiled regular expression."""

    for table in reversed(DataModelBase.metadata.sorted_tables):
        if pattern.fullmatch(table.name):
            try:
                table.drop(db_engine)
            except OperationalError:
                pass

if __name__ == '__main__':
    cmdline_args = parse_command_line()

    if cmdline_args.drop_hdf5_structure_tables:
        drop_tables_matching(re.compile('hdf5_.*'))
    if cmdline_args.drop_all_tables:
        drop_tables_matching(re.compile('.*'))
    DataModelBase.metadata.create_all(db_engine)
    add_default_hdf5_structures()
