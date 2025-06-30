"""Connect to the database and provide a session scope for queries."""

from os import path

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

db_engine = create_engine(
    (
        "sqlite:///"
        + path.join(path.dirname(path.abspath(__file__)), "autowisp.db")
        + "?timeout=100&uri=true"
    ),
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool,
)

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
Session = sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name
