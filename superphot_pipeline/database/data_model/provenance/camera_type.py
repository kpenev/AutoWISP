"""Define the camera type dataset table for the pipeline"""

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

from superphot_pipeline.database.data_model.base import DataModelBase
#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name
__all__ = ['CameraType']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class CameraType(DataModelBase):
    """The table describing the different  camera types"""

    __tablename__ = 'camera_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each camera type'
    )
    make = Column(
        String(100),
        nullable=False,
        doc='The make of the camera'
    )
    model = Column(
        String(100),
        nullable=False,
        doc='The model of the camera'
    )
    version = Column(
        String(100),
        nullable=False,
        doc='The version of the camera'
    )
    sensor_type = Column(
        String(100),
        nullable=False,
        doc='The sensor type of the camera'
    )
    x_resolution = Column(
        Integer,
        nullable=False,
        doc='The x_resolution of the camera'
    )
    y_resolution = Column(
        Integer,
        nullable=False,
        doc='The y_resolution of the camera'
    )
    pixel_size = Column(
        Float,
        nullable=False,
        doc='The pixel size of the camera'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the camera type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )
    cameras = relationship("Camera", back_populates="camera_type")
