"""Define the Conditions table for the pipeline"""

from sqlalchemy import \
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey
from sqlalchemy.orm import relationship

# Comment for database testing
from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

__all__ = ['Condition']


class Condition(DataModelBase):
    """
    The table describing the Conditions for given configuration to apply.

    Each condition is a combination of condition expressions that must all be
    satisfied simultaneously for the condition to be considered satisfied.
    """

    __tablename__ = 'condition'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each condition.'
    )
    expression_id = Column(
        Integer,
        ForeignKey('condition_expression.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='The id of the condition expression that is part of this condition.'
    )
    notes = Column(
        String(1000),
        nullable=False,
        doc='Any user supplied notes describing the condition.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When record was last changed'
    )

    expression = relationship("ConditionExpression")

    def __str__(self):
        return f"({self.id}) {self.expression_id} {self.notes} {self.timestamp}"
