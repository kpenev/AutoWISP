"""Define the camera dataset table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey

from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__ = ['Camera']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class Camera(DataModelBase):
    """The table describing the camera specified"""

    __tablename__ = 'camera'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each camera'
    )
    camera_type_id = Column(
        Integer,
        ForeignKey('camera_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the camera type'
    )
    serial_number = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        doc='The serial number of the camera'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the camera'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    observing_session = relationship("ObservingSession",
                                     back_populates="camera")
    cameratype = relationship("CameraType",
                              back_populates="cameras")
    camera_access = relationship("CameraAccess",
                                 back_populates="camera")
#pylint: enable=too-few-public-methods
