"""Define the mount dataset table for the pipeline"""

from sqlalchemy import Column, Integer, String, ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["Mount"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class Mount(DataModelBase):
    """The table describing the mounts specified"""

    __tablename__ = "mount"

    mount_type_id = Column(
        Integer,
        ForeignKey("mount_type.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The identifier of the mount type",
    )
    serial_number = Column(
        String(100), nullable=False, doc="The serial number for each mount"
    )
    notes = Column(
        String(1000), nullable=False, doc="The notes provided for the mount"
    )

    mount_type = relationship("MountType", back_populates="mounts")
    observers = relationship(
        "Observer", secondary="mount_access", back_populates="mounts"
    )
    observing_sessions = relationship(
        "ObservingSession", back_populates="mount"
    )

    def __str__(self):
        """Human readable info for the mount."""

        return (
            f"{self.mount_type.make} {self.mount_type.model} "
            f"({self.serial_number})"
        )

    # not sure how to use these
    # __table_args__ = (
    #     Index('description_index', 'description', unique=True),
    # )
