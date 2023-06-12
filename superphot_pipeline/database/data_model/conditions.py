"""Define the Conditions table for the pipeline"""

from sqlalchemy import \
    Column, \
    Integer, \
    String, \
    Float, \
    Date, \
    TIMESTAMP, \
    DateTime, \
    ForeignKey, \
    Index, \
    ForeignKeyConstraint

from sqlalchemy.orm import relationship

# Comment for database testing
# from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
from base import DataModelBase

__all__ = ['Conditions']


class Conditions(DataModelBase):
    """The table describing the Conditions"""

    __tablename__ = 'conditions'

    # id
    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each condition_expression'
    )
    # expression
    expression_id = Column(
        Integer,
        ForeignKey('condition_expressions.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the condition expression'
    )
    # notes
    notes = Column(
        String(1000),
        nullable=False,
        doc='The notes provided for the condition expression'
    )
    # timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When record was last changed'
    )

    def __init__(self, id, expression_id, notes, timestamp):
        self.id = id
        self.expression_id = expression_id
        self.notes = notes
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.expression_id} {self.notes} {self.timestamp}"

    # relationship
    cond_expr = relationship("ConditionExpressions", back_populates="condition")
    img_conditions = relationship("ImageConditions", back_populates="conditions")
    config = relationship("Configuration", back_populates="condition")
