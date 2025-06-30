"""Define the observer dataset table for the pipeline"""

from sqlalchemy import Column, String

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["Observer"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class Observer(DataModelBase):
    """The table describing the observers"""

    __tablename__ = "observer"

    name = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc="The name of the observer",
    )
    email = Column(String(100), nullable=True, doc="The email of the observer")
    phone = Column(
        String(100), nullable=True, doc="The phone number of the observer"
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes describing the observer.",
    )

    cameras = relationship(
        "Camera", secondary="camera_access", back_populates="observers"
    )
    mounts = relationship(
        "Mount", secondary="mount_access", back_populates="observers"
    )
    telescopes = relationship(
        "Telescope", secondary="telescope_access", back_populates="observers"
    )
    observing_sessions = relationship(
        "ObservingSession", back_populates="observer"
    )

    def __str__(self):
        """Human readable string identifying the observer."""

        return self.name
