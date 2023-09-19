"""Define the mount dataset table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey

from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name
__all__ = ['Mount']

#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods
class Mount(DataModelBase):
    """The table describing the mounts specified"""

    __tablename__ = 'mount'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each mount'
    )
    mount_type_id = Column(
        Integer,
        ForeignKey('mount_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The identifier of the mount type'
    )
    serial_number = Column(
        String(100),
        nullable=False,
        doc='The serial number for each mount'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the mount'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    mount_type = relationship("MountType", back_populates="mounts")
    mount_access = relationship("MountAccess", back_populates="mount")
    observing_session = relationship("ObservingSession", back_populates="mount")

    #not sure how to use these
    # __table_args__ = (
    #     Index('description_index', 'description', unique=True),
    # )
    #
    # def __str__(self):
    #     return '%d: %s' % (self.id, self.description)
