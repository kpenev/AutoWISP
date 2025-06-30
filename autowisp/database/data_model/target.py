"""Define the target table for the pipeline"""

from sqlalchemy import Column, String, Float

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

# pylint false positive: this is actually a class name
# pylint: disable=invalid-name
# pylint: enable=invalid-name

__all__ = ["Target"]
# The standard use of SQLAlchemy ORM requires classes with no public methods.
# pylint: disable=too-few-public-methods


class Target(DataModelBase):
    """The table dsecribing the target."""

    __tablename__ = "target"

    ra = Column(Float, nullable=True, doc="The ra of the target")
    dec = Column(Float, nullable=True, doc="The dec of the target")
    name = Column(String(100), nullable=False, doc="The name of the target")
    notes = Column(
        String(1000), nullable=True, doc="The notes about the target"
    )

    observing_sessions = relationship(
        "ObservingSession", back_populates="target"
    )

    def __repr__(self):
        return (
            f"{self.name} (id: {self.id}): ra={self.ra!r}, dec={self.dec!r} "
            f"({self.notes})"
        )
