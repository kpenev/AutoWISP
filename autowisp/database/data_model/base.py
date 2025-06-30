"""Declare the base class for all table classes."""

from sqlalchemy import text, Column, Integer, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase


# Intended to be sub-classed
# pylint: disable=too-few-public-methods
class DataModelBase(DeclarativeBase):
    """The base class for all table classes."""

    id = Column(
        Integer, primary_key=True, doc="A unique identifier for each row."
    )

    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        doc="When record was last changed",
    )

    def describe_table(self):
        """Return description of the table in human readable form."""

        return f"DB name: {self.__tablename__}: " + self.__doc__


# pylint: enable=too-few-public-methods
