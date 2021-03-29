"""Define the telescope access dataset table for the pipeline"""

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
__all__ = ['TelescopeAccess']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class TelescopeAccess(DataModelBase):
    """The table describing the telescope access"""

    __tablename__ = 'telescope_access'

    observer_id = Column(
        Integer,
        ForeignKey('observer.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='A unique identifier for the observer'
    )
    telescope_id = Column(
        Integer,
        ForeignKey('telescope.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='A unique identifier of the telescope'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    telescope = relationship("Telescope", back_populates="telescope_access")
    observer = relationship("Observer", back_populates="telescope_access")
