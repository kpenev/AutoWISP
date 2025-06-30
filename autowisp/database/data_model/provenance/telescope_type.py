"""Define the telescope type dataset table for the pipeline"""

from sqlalchemy import Column, String, Float

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["TelescopeType"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class TelescopeType(DataModelBase):
    """The table dscribing the different telescope types"""

    __tablename__ = "telescope_type"

    make = Column(String(100), nullable=False, doc="The make of the telescope")
    model = Column(
        String(100), nullable=False, doc="The model of the telescope"
    )
    version = Column(
        String(100), nullable=False, doc="The version of the telescope"
    )
    f_ratio = Column(
        Float, nullable=False, doc="The focal ratio of the telescope"
    )
    focal_length = Column(
        Float, nullable=False, doc="The focal length of the telescope in mm"
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="The notes provided for the telescope type",
    )

    telescopes = relationship("Telescope", back_populates="telescope_type")
