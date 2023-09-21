"""Define the step type table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import\
    Column,\
    String,\
    TIMESTAMP,\
    Table,\
    ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__= ['Step', 'Parameter']

_step_param_association = Table(
    'step_parameters',
    DataModelBase.metadata,
    Column('step_id', ForeignKey('step.id'), primary_key=True),
    Column('param_id', ForeignKey('parameter.id'), primary_key=True),
    Column('timestamp',
           TIMESTAMP,
           nullable=False,
           doc='When was this record last changed.')
)

_step_dependencies = Table(
    'step_dependencies',
    DataModelBase.metadata,
    Column('blocked_step_id', ForeignKey('step.id'), primary_key=True),
    Column('blocking_step_id', ForeignKey('step.id'), primary_key=True),
    Column('timestamp',
           TIMESTAMP,
           nullable=False,
           doc='When was this record last changed.')
)


class Step(DataModelBase):
    """The table describing the processing steps constituting the pipeline"""

    __tablename__ = 'step'

    id: Mapped[int] = mapped_column(primary_key=True)

    name = Column(
        String(100),
        nullable=False,
        doc='The name of the step within the pipeline.'
    )
    description = Column(
        String(1000),
        nullable=False,
        doc='Description of what the step does.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __str__(self):
        return f"({self.id}) {self.name}: {self.description} ({self.timestamp})"

    parameters: Mapped[List[Parameter]] = relationship(
        secondary=_step_param_association,
        back_populates='steps'
    )
    requires: Mapped[List[Step]] = relationship(
        secondary=_step_dependencies,
        primaryjoin=(id == _step_dependencies.c.blocked_step_id),
        secondaryjoin=(id == _step_dependencies.c.blocking_step_id),
        backref='required_by'
    )


class Parameter(DataModelBase):
    """Table describing the configuration parameters needed by the pipeline."""

    __tablename__ = 'parameter'

    id: Mapped[int] = mapped_column(primary_key=True)

    name = Column(
        String(100),
        nullable=False,
        doc='The name of the step within the pipeline.'
    )
    description = Column(
        String(1000),
        nullable=False,
        doc='Description of what the step does.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __str__(self):
        return f"({self.id}) {self.name}: {self.description} {self.timestamp}"

    steps: Mapped[List[Step]] = relationship(
        secondary=_step_param_association,
        back_populates='parameters'
    )
