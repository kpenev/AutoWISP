"""Define the camera type dataset table for the pipeline"""

from sqlalchemy import Column, Integer, String, Float

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["CameraType"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class CameraType(DataModelBase):
    """The table describing the different  camera types"""

    __tablename__ = "camera_type"

    make = Column(String(100), nullable=False, doc="The make of the camera")
    model = Column(String(100), nullable=False, doc="The model of the camera")
    version = Column(
        String(100), nullable=False, doc="The version of the camera"
    )
    sensor_type = Column(
        String(100), nullable=True, doc="The sensor type of the camera"
    )
    x_resolution = Column(
        Integer, nullable=False, doc="The x_resolution of the camera"
    )
    y_resolution = Column(
        Integer, nullable=False, doc="The y_resolution of the camera"
    )
    pixel_size = Column(
        Float, nullable=False, doc="The pixel size of the camera in microns"
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="The notes provided for the camera type",
    )
    cameras = relationship("Camera", back_populates="camera_type")
    channels = relationship("CameraChannel")
