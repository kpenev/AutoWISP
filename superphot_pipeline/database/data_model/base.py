"""Declare the base class for all table classes."""

from sqlalchemy.ext.declarative import declarative_base

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
DataModelBase = declarative_base()
#pylint: enable=invalid-name
