"""Define the ProcessingConfiguration table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import Column, Integer, String, ForeignKey, Index
from sqlalchemy.orm import Mapped, relationship

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.data_model.condition import Condition

__all__ = ["Configuration"]


class Configuration(DataModelBase):
    """Table recording the values of the pipeline configuration parameters."""

    __tablename__ = "configuration"

    parameter_id = Column(
        Integer,
        ForeignKey("parameter.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="The name of the configuration parameter.",
    )
    version = Column(
        Integer,
        doc="The version of the configuration parameter. Later versions fall "
        "back on earlier versions if an entry for the parameter is not found.",
    )
    condition_id = Column(
        Integer,
        ForeignKey("condition.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="The id of the condition that must be met for this configuration to"
        " apply",
    )
    value = Column(
        String(1000),
        nullable=True,
        doc="The value of the configuration parameter for the given version "
        "for images satisfying the given conditions.",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes describing the configuration.",
    )

    conditions: Mapped[List[Condition]] = relationship(
        "Condition",
        primaryjoin="Configuration.condition_id==foreign(Condition.id)",
        order_by="Condition.id",
        uselist=True,
    )
    parameter = relationship("Parameter")
    condition_expressions = relationship(
        "ConditionExpression", secondary=Condition.__tablename__, viewonly=True
    )

    def __repr__(self):
        return (
            f"Config v{self.version}: {self.parameter.name}={self.value} "
            f"if {self.conditions!r}"
        )

    __table_args__ = (
        Index(
            "config_key2",
            "parameter_id",
            "version",
            "condition_id",
            unique=True,
        ),
    )
