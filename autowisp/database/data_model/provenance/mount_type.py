"""Define the mount type dataset table for the pipeline"""

from sqlalchemy import Column, String

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["MountType"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class MountType(DataModelBase):
    """The table describing the different mount types"""

    __tablename__ = "mount_type"

    make = Column(String(100), nullable=False, doc="The make of the mount type")
    model = Column(
        String(100), nullable=False, doc="The model for each mount type"
    )
    version = Column(
        String(100), nullable=False, doc="The version of the mount type"
    )
    notes = Column(
        String(1000), nullable=True, doc="The notes provided for the mount type"
    )

    mounts = relationship("Mount", back_populates="mount_type")
