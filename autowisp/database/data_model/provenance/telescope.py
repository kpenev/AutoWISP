"""Define the telescope dataset table for the pipeline"""

from sqlalchemy import Column, Integer, String, ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name
__all__ = ["Telescope"]


# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods
class Telescope(DataModelBase):
    """The table describing the telescopes specified"""

    __tablename__ = "telescope"

    telescope_type_id = Column(
        Integer,
        ForeignKey(
            "telescope_type.id", onupdate="CASCADE", ondelete="RESTRICT"
        ),
        nullable=False,
        doc="The id of the telescope type",
    )
    serial_number = Column(
        String(100), nullable=False, doc="The serial number of the telescope"
    )
    notes = Column(
        String(1000), nullable=False, doc="The notes provided for the telescope"
    )

    observing_sessions = relationship(
        "ObservingSession", back_populates="telescope"
    )
    observers = relationship(
        "Observer", secondary="telescope_access", back_populates="telescopes"
    )
    telescope_type = relationship("TelescopeType", back_populates="telescopes")

    def __str__(self):
        """Human readable identifier for the telescope."""

        return (
            f"{self.telescope_type.make} {self.telescope_type.model} "
            f"({self.serial_number})"
        )
