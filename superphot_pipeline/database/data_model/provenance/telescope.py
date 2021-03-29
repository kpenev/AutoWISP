"""Define the telescope dataset table for the pipeline"""

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
__all__ = ['Telescope']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class Telescope(DataModelBase):
    """The table describing the telescopes specified"""

    __tablename__ = 'telescope'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each telescope type'
    )
    telescope_type_id = Column(
        Integer,
        ForeignKey('telescope_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the telescope type'
    )
    serial_number = Column(
        String(100),
        nullable=False,
        doc='The serial number of the telescope'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the telescope'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    observing_session = relationship("ObservingSession", back_populates="telescope")
    telescope_access = relationship("TelescopeAccess", back_populates="telescope")
    telescope_type = relationship("TelescopeType", back_populates="telescope")
