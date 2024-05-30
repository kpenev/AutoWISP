"""Declare the base class for all table classes."""

from sqlalchemy.orm import DeclarativeBase

#Intended to be sub-classed
#pylint: disable=too-few-public-methods
class DataModelBase(DeclarativeBase):
    """The base class for all table classes."""

    def describe_table(self):
        """Return description of the table in human readable form."""

        return f'DB name: {self.__tablename__}: ' + self.__doc__
#pylint: enable=too-few-public-methods
