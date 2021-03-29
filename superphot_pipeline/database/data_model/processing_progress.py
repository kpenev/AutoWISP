"""Define the processing progress table for the pipeline"""

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
from sqlalchemy.schema import Table

from superphot_pipeline.database.data_model.base import DataModelBase

#How do I import these things properly and replace them where they need to be

#pylint false positive: this is actually a class name
#pylint: disable=invalid-name
#pylint: enable=invalid-name

__all__= ['ProcessingProgress']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO remove all imports
#TODO check if the inputs relationship works

step_input = Table("step_input", DataModelBase.metadata,
    Column("consumer_id", Integer, ForeignKey("processing_progress.id"), primary_key=True),
    Column("consumed_id", Integer, ForeignKey("processing_progress.id"), primary_key=True),
    Column("timestamp", TIMESTAMP, nullable=False)
)

class ProcessingProgress(DataModelBase):
    """The table describing the processing progress"""

    __tablename__ = 'processing_progress'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the processing progress'
    )
    image_id = Column(
        Integer,
        ForeignKey('image.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the image'
    )
    step_configuration_version = Column(
        Integer,
        nullable=False,
        doc='The id of the step configuration'
    )
    processing_thread_id = Column(
        Integer,
        ForeignKey('processing_thread.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the processing thread'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )


#TODO use many to one, or one to many relationship to get association properly, have the forgeign key and relationship in the same table, processing progress should get the foreign key and th relationship
#then try to query it and check if it is properly working; relationship will have no back_populates
    #TODO experiment with a table associated with itself, mimick step_input id's to processing progress see what works
    #mess with primary and secondary joins and test what works
    image = relationship("Image")
    processing_thread = relationship("ProcessingThread")
    inputs = relationship("ProcessingProgress",
                          secondary="step_input",
                          primaryjoin=id==step_input.c.consumer_id,
                          secondaryjoin=id==step_input.c.consumed_id,
                          backref="users")