"""Define the mount type dataset table for the pipeline"""

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
__all__ = ['MountType']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class MountType(DataModelBase):
    """The table describing the different mount types"""

    __tablename__ = 'mount_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each mount type'
    )
    make = Column(
        String(100),
        nullable=False,
        doc='The make of the mount type'
    )
    model = Column(
        String(100),
        nullable=False,
        doc='The model for each mount type'
    )
    version = Column(
        String(100),
        nullable=False,
        doc='The version of the mount type'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the mount type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    mount = relationship("Mount", back_populates="mount_type")