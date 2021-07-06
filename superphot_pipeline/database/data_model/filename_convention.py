"""Define the filename convention table for the pipeline"""

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

__all__= ['FilenameConvention']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper import terms

class FilenameConvention(DataModelBase):
    """The table describing the filename convention"""

    __tablename__ = 'filename_convention'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for the filename convention'
    )
    product_id = Column(
        Integer,
        nullable=False,
        doc='The id of the product'
    )
    filename_template = Column(
        String(100),
        nullable=False,
        doc='The filename template'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    processing_thread = relationship("ProcessingThread", back_populates="filename_convention")
