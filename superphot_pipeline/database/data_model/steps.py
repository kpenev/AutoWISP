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

# Comment for database testing
from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

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
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes for each step type'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __init__(self, id, notes, timestamp):
        self.id = id
        self.notes = notes
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.notes} {self.timestamp}"

    image_proc = relationship("ImageProcProgress", back_populates="step_type")

    ## Do we still need this? 06/09/2023 - Mica
    #step_configuration = relationship("StepConfiguration", back_populates="step_type")
