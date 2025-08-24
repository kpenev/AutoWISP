"""Define the step type table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import Column, String, Table, ForeignKey, Index
from sqlalchemy.orm import Mapped, relationship

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.data_model.step_dependencies import StepDependencies

__all__ = ["Step", "Parameter", "AlternateParameterName"]

step_param_association = Table(
    "step_parameters",
    DataModelBase.metadata,
    Column("step_id", ForeignKey("step.id")),
    Column("param_id", ForeignKey("parameter.id")),
    Index("step_param_key2", "step_id", "param_id", unique=True),
)


class Step(DataModelBase):
    """The table describing the processing steps constituting the pipeline"""

    __tablename__ = "step"

    name = Column(
        String(100),
        nullable=False,
        doc="The name of the step within the pipeline.",
    )
    description = Column(
        String(1000),
        nullable=False,
        unique=True,
        doc="Description of what the step does.",
    )

    def __repr__(self):
        return f"({self.id}) {self.name}: {self.description} ({self.timestamp})"

    parameters: Mapped[List[Parameter]] = relationship(
        secondary=step_param_association, back_populates="steps"
    )
    prerequisites: Mapped[List[StepDependencies]] = relationship(
        StepDependencies,
        # False positive
        # pylint: disable=comparison-with-callable
        primaryjoin=("Step.id == StepDependencies.blocked_step_id"),
        # pylint: enable=comparison-with-callable
    )


class Parameter(DataModelBase):
    """Table describing the configuration parameters needed by the pipeline."""

    __tablename__ = "parameter"

    name = Column(
        String(100),
        nullable=False,
        doc="The name of the step within the pipeline.",
    )
    description = Column(
        String(1000), nullable=False, doc="Description of what the step does."
    )

    def __str__(self):
        return f"({self.id}) {self.name}: {self.description} {self.timestamp}"

    steps: Mapped[List[Step]] = relationship(
        secondary=step_param_association, back_populates="parameters"
    )

class AlternateParameterName(DataModelBase):
    """Table describing alternate names for parameters."""

    __tablename__ = "alternate_parameter_names"

    param_id = Column(
        ForeignKey("parameter.id"),
        doc="The ID of the parameter.",
    )
    alt_name = Column(
        String(100),
        doc="An alternate name for the parameter.",
    )

    parameter: Mapped[Parameter] = relationship("Parameter", backref="alternate_names")

    def __str__(self):
        return f"Parameter ID {self.param_id} has alternate name '{self.alt_name}'"
