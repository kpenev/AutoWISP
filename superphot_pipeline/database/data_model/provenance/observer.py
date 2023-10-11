"""Define the observer dataset table for the pipeline"""

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
__all__ = ['Observer']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class Observer(DataModelBase):
    """The table describing the observers"""

    __tablename__ = 'observer'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each observer'
    )
    name = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc='The name of the observer'
    )
    email = Column(
        String(100),
        nullable=False,
        doc='The email of the observer'
    )
    phone = Column(
        String(100),
        nullable=False,
        doc='The phone number of the observer'
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc='Any user supplied notes describing the observer.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    camera_access = relationship("CameraAccess",
                                 back_populates="observer")
    mount_access = relationship("MountAccess",
                                back_populates="observer")
    telescope_access = relationship("TelescopeAccess",
                                    back_populates="observer")
    observing_sessions = relationship("ObservingSession",
                                      back_populates="observer")
