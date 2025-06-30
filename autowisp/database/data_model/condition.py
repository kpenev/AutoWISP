"""Define the Conditions table for the pipeline"""

from sqlalchemy import Column, Integer, String, PrimaryKeyConstraint, ForeignKey
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["Condition"]


class Condition(DataModelBase):
    """
    The table describing the Conditions for given configuration to apply.

    Each condition is a combination of condition expressions that must all be
    satisfied simultaneously for the condition to be considered satisfied.
    """

    __tablename__ = "condition"

    expression_id = Column(
        Integer,
        ForeignKey(
            "condition_expression.id", onupdate="CASCADE", ondelete="RESTRICT"
        ),
        primary_key=True,
        doc="The id of the condition expression that is part of this condition.",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes describing the condition.",
    )

    __table_args__ = (PrimaryKeyConstraint("id", "expression_id"),)

    expression = relationship("ConditionExpression")

    def __str__(self):
        return f"({self.id}) {self.expression_id} {self.notes} {self.timestamp}"
