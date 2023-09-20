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

__all__= ['Steps', 'Parameters']

_step_param_association = Table(
    "step_parameters",
    DataModelBase.metadata,
    Column("step_id", ForeignKey("steps.id"), primary_key=True),
    Column("param_id", ForeignKey("parameters.id"), primary_key=True),
)

#_step_dependencies = Table(
#    "step_dependencies",
#    DataModelBase.metadata,
#    Column("blocked_step_id", ForeignKey("steps.id"), primary_key=True),
#    Column("blocking_step_id", ForeignKey("steps.id"), primary_key=True),
#)

class Steps(DataModelBase):
    """The table describing the processing steps constituting the pipeline"""

    __tablename__ = 'steps'

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

    parameters: Mapped[List[Parameters]] = relationship(
        secondary=_step_param_association,
        back_populates='steps'
    )
#    requires: Mapped[List[Steps]] = relationship(
#        secondary=_step_dependencies,
#        primaryjoin=id==_step_dependencies.c.blocked_step_id,
#        back_populates='required_by'
#    )
#    required_by: Mapped[List[Steps]] = relationship(
#        secondary=_step_dependencies,
#        primaryjoin=id==_step_dependencies.c.blocked_step_id,
#        back_populates='requires'
#    )

class Parameters(DataModelBase):
    """Table describing the configuration parameters needed by the pipeline."""

    __tablename__ = 'parameters'

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

    #used_by: Mapped[List[Steps]] = relationship(
    steps: Mapped[List[Steps]] = relationship(
        secondary=_step_param_association,
        back_populates='parameters'
    )
