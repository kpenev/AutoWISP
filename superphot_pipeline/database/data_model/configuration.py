"""Define the ProcessingConfiguration table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    ForeignKey,\
    TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__ = ['Configuration']

class Configuration(DataModelBase):
    """Table recording the values of the pipeline configuration parameters."""

    __tablename__ = 'configuration'

    parameter_id = Column(
        Integer,
        ForeignKey('parameter.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
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
        primary_key=True,
        doc='The id of the condition that must be met for this configuration to'
        ' apply'
    )
    value = Column(
        String(1000),
        nullable=True,
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
        nullable=False,
        doc='When record was last changed'
    )

    conditions: Mapped[List[Condition]] = relationship(
        "Condition",
        primaryjoin='Configuration.condition_id==foreign(Condition.id)',
        order_by='Condition.id',
        uselist=True
    )
    parameter = relationship("Parameter")
