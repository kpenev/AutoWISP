"""Declare the base class for all table classes."""

from sqlalchemy import text, Column, Integer, TIMESTAMP
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.declarative import AbstractConcreteBase


# Intended to be sub-classed
# pylint: disable=too-few-public-methods
class DataModelSubBase(DeclarativeBase):
    """The base class for all table classes, except without 'id' column."""

    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        doc="When record was last changed",
    )

    def describe_table(self):
        """Return description of the table in human readable form."""

        return f"DB name: {self.__tablename__}: " + self.__doc__


class DataModelBase(AbstractConcreteBase, DataModelSubBase):
    """The base class for all table classes."""

    id = Column(
        Integer,
        primary_key=True,
        doc="A unique identifier for each row.",
    )


# pylint: enable=too-few-public-methods
