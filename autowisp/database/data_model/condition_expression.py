"""Define the ConditionExpression table for the pipeline"""

from sqlalchemy import Column, String

# Comment for database testing
from autowisp.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

__all__ = ["ConditionExpression"]


class ConditionExpression(DataModelBase):
    """The table describing the Condition Expressions"""

    __tablename__ = "condition_expression"

    expression = Column(
        # 768, not 1000: the unique index has to fit InnoDB's 3072-byte
        # limit at 4 bytes/character. Ample for an expression over header
        # keywords.
        String(768),
        nullable=False,
        unique=True,
        index=True,
        doc="The expression to evaluate to determine if an image meets the "
        "condition.",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes describing the condition expression.",
    )

    def __str__(self):
        return f"({self.id}) {self.expression} {self.notes} {self.timestamp}"
