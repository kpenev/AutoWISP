"""Define the observing session table for the pipeline"""


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
from superphot_pipeline.database.data_model.provenance import\
    camera,\
    camera_access,\
    camera_type,\
    mount,\
    mount_type,\
    mount_access,\
    telescope,\
    telescope_type,\
    telescope_access,\
    observatory,\
    observer
#How do I import these provenance properly and replace them where they need to be

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name

__all__= ['ObservingSession']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper provenance terms

class ObservingSession(DataModelBase):
    """The table describing the observing session"""

    __tablename__ = 'observing_session'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each observing session'
    )
    observer_id = Column(
        Integer,
        ForeignKey('observer.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the observer'
    )
    camera_id = Column(
        Integer,
        ForeignKey('camera.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the camera'
    )
    telescope_id = Column(
        Integer,
        ForeignKey('telescope.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the telescope'
    )
    mount_id = Column(
        Integer,
        ForeignKey('mount.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the mount'
    )
    observatory_id = Column(
        Integer,
        ForeignKey('observatory.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the observatory'
    )
    target_id = Column(
        Integer,
        ForeignKey('target.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the target'
    )
    start_time = Column(
        DateTime,
        nullable=False,
        doc='The start time of the observing session'
    )
    end_time = Column(
        DateTime,
        nullable=False,
        doc='The end time of the observing session'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the observing session'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    observer = relationship("Observer", back_populates="observing_session")
    camera = relationship("Camera", back_populates="observing_session")
    telescope = relationship("Telescope", back_populates="observing_session")
    mount = relationship("Mount", back_populates="observing_session")
    observatory = relationship("Observatory", back_populates="observing_session")
    target = relationship("Target", back_populates="observing_session")
    images = relationship("Image", back_populates="observing_session")
