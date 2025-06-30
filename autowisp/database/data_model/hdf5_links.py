"""Define the hdf5_links table."""

from sqlalchemy import Column, Integer, String, ForeignKey, Index
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["HDF5Link"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class HDF5Link(DataModelBase):
    """Table describing links of chosen versions of products in HDF5 files."""

    __tablename__ = "hdf5_links"

    hdf5_structure_version_id = Column(
        Integer,
        ForeignKey(
            "hdf5_structure_versions.id",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        doc="Which structure version of which pipeline product is this "
        "element configuration for.",
    )
    pipeline_key = Column(
        String(100),
        nullable=False,
        doc="How is this link referred to by the pipeline.",
    )
    abspath = Column(
        String(1000),
        nullable=False,
        doc="The full absolute path to the link in the HDF5 file.",
    )
    target = Column(
        String(1000),
        nullable=False,
        doc="The full absolute path the link should point to.",
    )

    description = Column(
        String(1000),
        nullable=False,
        doc="A brief description of what this attribute tracks.",
    )

    __table_args__ = (
        Index(
            "version_link",
            "hdf5_structure_version_id",
            "pipeline_key",
            unique=True,
        ),
    )

    structure_version = relationship(
        "HDF5StructureVersion", back_populates="links"
    )
