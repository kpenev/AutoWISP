"""Define the ConditionExpressions table for the pipeline"""

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    Float,\
    Date,\
    TIMESTAMP,\
    DateTime,\
    ForeignKey,\
    Index,\
    ForeignKeyConstraint

from sqlalchemy.orm import relationship

#Comment for database testing
#from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
from base import DataModelBase

__all__ = ['ConditionExpressions']

class ConditionExpressions(DataModelBase):
    """The table describing the Condition Expressions"""

    __tablename__ = 'condition_expressions'

    #id
    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each condition_expression'
    )
    #expression
    expression = Column(
        String(1000),
        nullable=False,
        doc='The description of the condition expression'
    )
    #notes
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the condition expression'
    )
    #timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc = 'When record was last changed'
    )

    def __init__(self, id, expression, notes, timestamp):
        self.id = id
        self.expression = expression
        self.notes = notes
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.expression} {self.notes} {self.timestamp}"

    #relationship
    condition = relationship("Conditions", back_populates="cond_expr")
