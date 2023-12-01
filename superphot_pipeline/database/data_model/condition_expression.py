"""Define the ConditionExpression table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP\

from sqlalchemy.orm import relationship

#Comment for database testing
from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

__all__ = ['ConditionExpression']

class ConditionExpression(DataModelBase):
    """The table describing the Condition Expressions"""

    __tablename__ = 'condition_expression'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each condition_expression.'
    )
    expression = Column(
        String(1000),
        nullable=False,
        unique=True,
        index=True,
        doc='The expression to evaluate to determine if an image meets the '
        'condition.'
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc='Any user supplied notes describing the condition expression.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc = 'When record was last changed'
    )

    def __str__(self):
        return f"({self.id}) {self.expression} {self.notes} {self.timestamp}"
