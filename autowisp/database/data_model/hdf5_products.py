"""Define the hdf5_attributes table."""

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# Pylint false positive due to quirky imports.
# pylint: disable=no-name-in-module
from autowisp.database.data_model.hdf5_structure_versions import (
    HDF5StructureVersion,
)

# pylint: enable=no-name-in-module

__all__ = ["HDF5Product"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class HDF5Product(DataModelBase):
    """The types of pipeline products stored as HDF5 files."""

    __tablename__ = "hdf5_products"

    pipeline_key = Column(
        String(100),
        index=True,
        unique=True,
        doc="How is this product referred to in the pipeline (e.g. "
        '"datareduction" or "lightcurve"',
    )
    description = Column(
        String(100), nullable=False, doc="A description of the product type."
    )

    structure_versions = relationship(
        "HDF5StructureVersion",
        order_by=HDF5StructureVersion.id,
        back_populates="product",
    )


# pylint: enable=too-few-public-methods
