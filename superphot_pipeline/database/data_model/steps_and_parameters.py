"""Define the step type table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    Table,\
    ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__= ['Step', 'Parameter', 'ProcessingSequence', 'StepDependencies']

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


class ProcessingSequence(DataModelBase):
    """The sequence of steps/image type to be processed by the pipeline."""

    __tablename__ = 'processing_sequence'

    id = Column(
        Integer,
        primary_key=True,
        doc='The index of this processing within the sequence to be followed '
        'by the pipeline.'
    )
    step_id = Column(
        Integer,
        ForeignKey('step.id'),
        nullable=False,
        doc='The step to be executed.'
    )
    image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        nullable=True,
        doc='The image type to be processed by the step.'
    )

    step = relationship('Step')
    image_type = relationship('ImageType')

    def __repr__(self):
        return f'{self.step.name} {self.image_type.name}'


class StepDependencies(DataModelBase):
    """The table describing the prerequisites for a step to run"""


    __tablename__ = 'step_dependencies'

    blocked_step_id = Column(
        Integer,
        ForeignKey('step.id'),
        primary_key=True,
        doc='The step for which this prerequisite applies.'
    )
    blocked_image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        primary_key=True,
        doc='The image type for which this prerequisite applies.'
    )
    blocking_step_id = Column(
        Integer,
        ForeignKey('step.id'),
        primary_key=True,
        doc='The step which must be completed before the blocked step can '
        'begin.'
    )
    blocking_image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        primary_key=True,
        doc='The image type for which the prerequisite step must be completed.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
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
        unique=True,
        doc='Description of what the step does.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __repr__(self):
        return f"({self.id}) {self.name}: {self.description} ({self.timestamp})"

    parameters: Mapped[List[Parameter]] = relationship(
        secondary=_step_param_association,
        back_populates='steps'
    )
    prerequisites: Mapped[List[StepDependencies]] = relationship(
        StepDependencies,
        primaryjoin=(id == StepDependencies.blocked_step_id),
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
