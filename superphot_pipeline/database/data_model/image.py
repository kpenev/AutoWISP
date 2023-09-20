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

#Comment for database testing
from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

#How do I import these things properly and replace them where they need to be

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name

__all__= ['Image']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper image terms imports
#TODO  image and camera should have all the resolutions for all the channels separated
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
    observing_session_id = Column(
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

    ## ADDED 06/08/23 - Mica
    def __init__(self, id, image_type_id, observing_session_id, notes, timestamp):
        self.id = id
        self.image_type_id = image_type_id
        self.observing_session_id = observing_session_id
        self.notes = notes
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id})  {self.image_type_id} {self.observing_session_id} {self.notes} {self.timestamp}"

    image_type = relationship("ImageType", back_populates="image")
    observing_session = relationship("ObservingSession", back_populates="images")
    #img_conditions = relationship("ImageConditions", back_populates="image")
    #image_proc = relationship("ImageProcProgress", back_populates="image")
