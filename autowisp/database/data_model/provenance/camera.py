"""Define the camera dataset table for the pipeline"""

from sqlalchemy import Column, Integer, String, ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["Camera"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class Camera(DataModelBase):
    """The table describing the camera specified"""

    __tablename__ = "camera"

    camera_type_id = Column(
        Integer,
        ForeignKey("camera_type.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the camera type",
    )
    serial_number = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc="The serial number of the camera",
    )
    notes = Column(
        String(1000), nullable=False, doc="The notes provided for the camera"
    )

    observing_sessions = relationship(
        "ObservingSession", back_populates="camera"
    )
    camera_type = relationship("CameraType", back_populates="cameras")
    observers = relationship(
        "Observer", secondary="camera_access", back_populates="cameras"
    )
    channels = relationship(
        "CameraChannel", secondary="camera_type", viewonly=True
    )

    def __str__(self):
        """Human readable info for the camera."""

        return (
            f"{self.camera_type.make} {self.camera_type.model} "
            f"({self.serial_number})"
        )


# pylint: enable=too-few-public-methods
