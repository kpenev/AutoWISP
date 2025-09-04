"""Connect to the database and provide a session scope for queries."""

from os import path
from contextlib import contextmanager

import platformdirs

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.initialize_data_reduction_structure import (
    get_default_data_reduction_structure,
)
from autowisp.database.initialize_light_curve_structure import (
    get_default_light_curve_structure,
)

_db_engine = None

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
_Session = None  # sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name

_sqlite_fname = None


def get_db_engine():
    """Return the database engine."""

    print(f"Returning engine {_db_engine!r}")
    return _db_engine


@contextmanager
def start_db_session():
    """Context manager to start a database session."""

    with _Session.begin() as db_session:  # pylint: disable=no-member
        yield db_session


def get_sqlite_fname():
    """Return the path to the sqlite database currently being used."""

    return _sqlite_fname


def initialize_cmdline_database():
    """Initialize the current database HDF5 structure tables."""

    DataModelBase.metadata.create_all(_db_engine)
    with start_db_session() as db_session:
        db_session.add(get_default_data_reduction_structure())
        db_session.add(get_default_light_curve_structure(db_session))


def set_sqlite_database(db_path):
    """
    Set the database engine and session to use the given SQLite database.

    If ``db_path`` is None, it sets the database for the one needed for command
    line processing.
    """

    global _db_engine, _Session, _sqlite_fname  # pylint: disable=global-statement

    if _db_engine is not None:
        _db_engine.dispose()

    initialize = False
    if db_path is None:
        db_path = path.join(
            platformdirs.user_data_dir("autowisp"), "autowisp.db"
        )
        if not path.exists(db_path):
            initialize = True

    _sqlite_fname = path.abspath(db_path)
    _db_engine = create_engine(
        ("sqlite:///" + _sqlite_fname + "?timeout=100&uri=true"),
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        poolclass=NullPool,
    )
    _Session = sessionmaker(_db_engine, expire_on_commit=False)

    if initialize:
        initialize_cmdline_database()
