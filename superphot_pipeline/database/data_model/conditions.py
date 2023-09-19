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
        doc='Any user supplied notes describing the condition.'
    )
    # timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable=True,
        doc='When record was last changed'
    )

    def __str__(self):
        return f"({self.id}) {self.expression_id} {self.notes} {self.timestamp}"

    # relationship
    expressions = relationship("ConditionExpressions",
                               back_populates="condition")
