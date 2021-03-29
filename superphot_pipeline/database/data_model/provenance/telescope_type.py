"""Define the telescope type dataset table for the pipeline"""

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
__all__ = ['TelescopeType']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class TelescopeType(DataModelBase):
    """The table dscribing the different telescope types"""

    __tablename__ = 'telescope_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each telescope type'
    )
    make = Column(
        String(100),
        nullable=False,
        doc='The make of the telescope'
    )
    model = Column(
        String(100),
        nullable=False,
        doc='The model of the telescope'
    )
    version = Column(
        String(100),
        nullable=False,
        doc='The version of the telescope'
    )
    f_ratio = Column(
        Float,
        nullable=False,
        doc='The focal ratio of the telescope'
    )
    diameter = Column(
        Float,
        nullable=False,
        doc='The diameter of the telescope'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the telescope type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    telescope = relationship("Telescope", back_populates="telescope_type")
