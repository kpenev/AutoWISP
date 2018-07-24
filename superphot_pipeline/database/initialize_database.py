#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

from superphot_pipeline.database.initialize_data_reduction_structure import\
    get_default_data_reduction_structure

def add_default_hdf5_structures():
    """Add a default HDF5 structure to the database."""

    with db_session_scope() as db_session:
        db_session.add(get_default_data_reduction_structure())

def create_all_tables():
    """Create all database tables currently defined."""

    for table in DataModelBase.metadata.sorted_tables:
        if not db_engine.has_table(table.name):
            table.create(db_engine)


if __name__ == '__main__':
    create_all_tables()
    add_default_hdf5_structures()
