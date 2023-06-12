"""Define the image type table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    Float,\
    Date,\
    TIMESTAMP,\
    DateTime,\
    ForeignKey,\
    Index,\
    ForeignKeyConstraint

from sqlalchemy.orm import relationship

#Comment for database testing
#from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
from base import DataModelBase

#How do I import these things properly and replace them where they need to be

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name

__all__= ['ImageType']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper image terms imports

class ImageType(DataModelBase):
    """The table describing the different image types."""

    __tablename__ = 'image_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image type'
    )
    type_name = Column(
        String(100),
        nullable=False,
        doc='The image type name'
    )
    description = Column(
        String(1000),
        nullable=False,
        doc='The description of the image type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    ## ADDED 06/09/23 - Mica
    def __init__(self, id, type_name, description, timestamp):
        self.id = id
        self.type_name = type_name
        self.description = description
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.type_name} {self.description} {self.timestamp}"

    image = relationship("Image", back_populates="image_type")
