"""Define the observing session table for the pipeline"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# Needed to create the relationship
# pylint: disable=unused-import
from autowisp.database.data_model.provenance import observer

# pylint: enable=unused-import

__all__ = ["ObservingSession"]

# TODO replace proper provenance terms


class ObservingSession(DataModelBase):
    """The table describing the observing session"""

    __tablename__ = "observing_session"

    observer_id = Column(
        Integer,
        ForeignKey("observer.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the observer",
    )
    camera_id = Column(
        Integer,
        ForeignKey("camera.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the camera",
    )
    telescope_id = Column(
        Integer,
        ForeignKey("telescope.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the telescope",
    )
    mount_id = Column(
        Integer,
        ForeignKey("mount.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the mount",
    )
    observatory_id = Column(
        Integer,
        ForeignKey("observatory.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the observatory",
    )
    target_id = Column(
        Integer,
        ForeignKey("target.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the target",
    )
    start_time_utc = Column(
        DateTime,
        nullable=False,
        doc="The start time of the observing session in UTC",
    )
    end_time_utc = Column(
        DateTime,
        nullable=False,
        doc="The end time of the observing session in UTC",
    )
    label = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique label assigned to the observing session",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="The notes provided for the observing session",
    )

    def __repr__(self):
        return f"({self.id}) {self.notes} {self.timestamp}"

    # relationships
    observer = relationship("Observer", back_populates="observing_sessions")
    camera = relationship("Camera", back_populates="observing_sessions")
    telescope = relationship("Telescope", back_populates="observing_sessions")
    mount = relationship("Mount", back_populates="observing_sessions")
    observatory = relationship(
        "Observatory", back_populates="observing_sessions"
    )
    target = relationship("Target", back_populates="observing_sessions")
    images = relationship("Image", back_populates="observing_session")
