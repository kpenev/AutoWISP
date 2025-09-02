"""Connect to the database and provide a session scope for queries."""

from os import path
from contextlib import contextmanager

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

_db_engine = None

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
_Session = None#sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name

_sqlite_fname = None

def get_db_engine():
    """Return the database engine."""

    print(f"Returning engine {_db_engine!r}")
    return _db_engine

@contextmanager
def start_db_session():
    """Context manager to start a database session."""

    with _Session.begin() as db_session: # pylint: disable=no-member
        yield db_session

def get_sqlite_fname():
    """Return the path to the sqlite database currently being used."""

    return _sqlite_fname

def set_sqlite_database(db_path):
    """Set the database engine and session to use the given SQLite database."""

    global _db_engine, _Session, _sqlite_fname  # pylint: disable=global-statement

    _sqlite_fname = path.abspath(db_path)
    _db_engine = create_engine(
        (
            "sqlite:///"
            + _sqlite_fname
            + "?timeout=100&uri=true"
        ),
        echo=True,
        pool_pre_ping=True,
        pool_recycle=3600,
        poolclass=NullPool,
    )
    _Session = sessionmaker(_db_engine, expire_on_commit=False)
