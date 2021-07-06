"""Define the processing thread table for the pipeline"""

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

__all__= ['ProcessingThread']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper import terms

class ProcessingThread(DataModelBase):
    """The table describing the processing thread"""

    __tablename__ = 'processing_thread'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the processing progress'
    )
    filename_convention_id = Column(
        Integer,
        ForeignKey('filename_convention.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the filename convention'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    filename_convention = relationship("FilenameConvention", back_populates="processing_thread")
    processing_progress = relationship("ProcessingProgress", back_populates="processing_thread")
