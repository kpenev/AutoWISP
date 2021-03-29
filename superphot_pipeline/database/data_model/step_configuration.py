"""Define the processing step table for the pipeline"""

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

__all__= ['StepConfiguration']
#The standard use of SQLAlchemy ORM requires classes with no public methods.
#pylint: disable=too-few-public-methods

#TODO replace proper import terms

class StepConfiguration(DataModelBase):
    """The table describing the processing step"""

    __tablename__ = 'step_configuration'

    step_type_id = Column(
        Integer,
        ForeignKey('step_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        nullable=False,
        doc='The id of the processing step type'
    )
    version = Column(
        Integer,
        primary_key=True,
        nullable=False,
        doc='The id of the configuration of the step'
    )
    parameter = Column(
        String(1000),
        nullable=False,
        doc='The parameter for each step configuration'
    )
    value = Column(
        String(1000),
        nullable=False,
        doc='The value of the parameter for each step configuration'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    step_type = relationship("StepType", back_populates="step_configuration")
