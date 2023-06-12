"""Define configuration table for pipeline"""

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

# Comment for database testing
from superphot_pipeline.database.data_model.base import DataModelBase

# For database testing
# from base import DataModelBase

__all__ = ['Configuration']

class Configuration(DataModelBase):
    """The table describing the Image Processing Progress"""

    __tablename__ = 'configuration'
    #id
    id = Column(
        Integer,
        primary_key=True,
        doc="unique identifier"
    )
    # version
    version = Column(
        Integer,
        nullable=False,
        doc="version of the configuration"
    )
    # condition_id
    condition_id = Column(
        Integer,
        ForeignKey("conditions.id",
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc="the condition_id"
    )
    # parameter
    parameter = Column(
        String(1000),
        nullable=False,
        doc="the parameter of the configuration"
    )
    # value
    value = Column(
        String(1000),
        nullable=False,
        doc = "the value of the configuration"
    )
    # notes
    notes = Column(
        String(1000),
        nullable=False,
        doc = "the notes of the configuration"
    )
    # timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable= False,
        doc = 'When record was last changed'
    )

    def __init__(self, id, version, condition_id, parameter, value, notes, timestamp):
        self.id = id
        self.version = version
        self.condition_id = condition_id
        self.parameter = parameter
        self.value = value
        self.notes = notes
        self.timestamp = timestamp
    def __repr__(self):
        return f"({self.id}) {self.condition_id} {self.timestamp}"

    #relationship
    condition = relationship("Conditions", back_populates="config")
    imgprocprog = relationship("ImageProcProgress", back_populates="config")
