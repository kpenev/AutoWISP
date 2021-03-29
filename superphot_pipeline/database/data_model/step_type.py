"""Define the step type table for the pipeline"""

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

__all__= ['StepType']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper processing step terms imports

class StepType(DataModelBase):
    """The table describing the processing step type"""

    __tablename__ = 'step_type'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each step type'
    )
    description = Column(
        String(1000),
        nullable=False,
        doc='The description for each step type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    step_configuration = relationship("StepConfiguration", back_populates="step_type")
