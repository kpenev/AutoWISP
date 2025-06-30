"""Define the image type table for the pipeline"""

from sqlalchemy import Column, String

from sqlalchemy.orm import relationship

# Comment for database testing
from autowisp.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

# How do I import these things properly and replace them where they need to be

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name

__all__ = ["ImageType"]
# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods


class ImageType(DataModelBase):
    """The table describing the different image types."""

    __tablename__ = "image_type"

    name = Column(String(100), nullable=False, doc="The image type name")
    description = Column(
        String(1000), nullable=True, doc="The description of the image type"
    )

    def __repr__(self):
        return f"({self.id}) {self.name} {self.description} {self.timestamp}"

    image = relationship("Image", back_populates="image_type")
