"""Define the hdf5_structure_versions table."""

from sqlalchemy import Column, Integer, ForeignKey, Index
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["HDF5StructureVersion"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class HDF5StructureVersion(DataModelBase):
    """The versions of structures for the HDF5 pipeline products."""

    __tablename__ = "hdf5_structure_versions"

    hdf5_product_id = Column(
        Integer,
        ForeignKey("hdf5_products.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="The type of pipeline product this structure configuration version "
        "is for.",
    )
    version = Column(
        Integer,
        nullable=False,
        doc="An identifier for distinguishing the separate configuration "
        "versions of a single pipeline product type.",
    )

    __table_args__ = (
        Index("product_version", "hdf5_product_id", "version", unique=True),
    )

    product = relationship("HDF5Product", back_populates="structure_versions")

    attributes = relationship(
        "HDF5Attribute", back_populates="structure_version"
    )

    datasets = relationship("HDF5DataSet", back_populates="structure_version")

    links = relationship("HDF5Link", back_populates="structure_version")


# pylint: enable=too-few-public-methods
