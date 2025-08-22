"""Connect to the database and provide a session scope for queries."""

from os import path
from contextlib import contextmanager

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

db_engine = None

#create_engine(
#    (
#        "sqlite:///"
#        + path.join(path.dirname(path.abspath(__file__)), "autowisp.db")
#        + "?timeout=100&uri=true"
#    ),
#    echo=False,
#    pool_pre_ping=True,
#    pool_recycle=3600,
#    poolclass=NullPool,
#)

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
Session = None#sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name

def get_db_engine():
    """Return the database engine."""

    print(f"Returning engine {db_engine!r}")
    return db_engine

@contextmanager
def start_db_session():
    """Context manager to start a database session."""

    with Session.begin() as db_session: # pylint: disable=no-member
        yield db_session

def set_sqlite_database(db_path):
    """Set the database engine and session to use the given SQLite database."""

    global db_engine, Session  # pylint: disable=global-statement

    db_engine = create_engine(
        (
            "sqlite:///"
            + path.abspath(db_path)
            + "?timeout=100&uri=true"
        ),
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        poolclass=NullPool,
    )
    Session = sessionmaker(db_engine, expire_on_commit=False)
