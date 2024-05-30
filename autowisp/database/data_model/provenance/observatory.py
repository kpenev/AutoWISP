"""Define the observatory dataset table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    Float,\
    TIMESTAMP

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name
__all__ = ['Observatory']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class Observatory(DataModelBase):
    """The table describing the observatory"""

    __tablename__ = 'observatory'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each observatory'
    )
    latitude = Column(
        Float,
        nullable=False,
        doc='The latitude of the observatory'
    )
    longitude = Column(
        Float,
        nullable=False,
        doc='The longitude of the observatory'
    )
    altitude = Column(
        Float,
        nullable=False,
        doc='The altitude of the observatory'
    )
    name = Column(
        String(100),
        nullable=False,
        doc='The name of the observatory'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    observing_sessions = relationship("ObservingSession",
                                      back_populates="observatory")

    def __str__(self):
        """Human readable identifier for the observatory."""

        return f'{self.name} (lat={self.latitude}, lon={self.longitude})'
