"""Define the ProcessingConfiguration table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    ForeignKey,\
    TIMESTAMP
from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__ = ['ProcessingConfiguration']

class ProcessingConfiguration(DataModelBase):
    """Table recording the values of the pipeline configuration parameters."""

    __tablename__ = 'processing_configuration'

    parameter_name = Column(
        String,
        primary_key=True,
        doc='The name of the configuration parameter.'
    )
    version = Column(
        Integer,
        primary_key=True,
        doc='The version of the configuration parameter. Later versions fall '
        'back on earlier versions if an entry for the parameter is not found.'
    )
    condition_id = Column(
        Integer,
        ForeignKey('conditions.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='The id of the condition that must be met for this configuration to'
        ' apply'
    )
    value = Column(
        String,
        nullable=False,
        doc='The value of the configuration parameter for the given version '
        'for images satisfying the given conditions.'
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc='Any user supplied notes describing the configuration.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=True,
        doc='When record was last changed'
    )

    condition = relationship("Conditions")
