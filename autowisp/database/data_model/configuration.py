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
        # Deliberately not a ForeignKey. A condition is a *set* of
        # expressions: `condition` holds one row per member, all sharing an
        # id, so condition.id identifies a group rather than a row and is
        # not unique. A foreign key asserts the opposite. InnoDB accepted
        # that as a non-standard extension until MySQL 8.4 began rejecting
        # it (ER_FK_NO_UNIQUE_INDEX_PARENT); the relationship below already
        # hand-annotates the join for the same reason.
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
