"""Define the image table for the pipeline"""

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

#How do I import these things properly and replace them where they need to be

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name

__all__= ['Image']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper image terms imports

class Image(DataModelBase):
    """The table describing the image specified"""

    __tablename__ = 'image'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image'
    )
    image_type_id = Column(
        Integer,
        ForeignKey('image_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the image type'
    )
    observng_session_id = Column(
        Integer,
        ForeignKey('observing_session.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the observing session'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the image'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    image_type = relationship("ImageType", back_populates="image")
    observing_session = relationship("ObservingSession", back_populates="image")
