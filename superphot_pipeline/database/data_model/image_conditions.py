"""Define the Image Conditions table for the pipeline"""

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

__all__ = ['ImageConditions']


class ImageConditions(DataModelBase):
    """The table describing the Conditions"""

    __tablename__ = 'image_conditions'

    # id
    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each condition_expression'
    )
    # image_id
    image_id = Column(
        Integer,
        ForeignKey('image.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the image'
    )
    # condition_id
    condition_id = Column(
        Integer,
        ForeignKey('conditions.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the condition expression'
    )
    # timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When record was last changed'
    )

    def __init__(self, id, image_id, condition_id, timestamp):
        self.id = id
        self.image_id = image_id
        self.condition_id = condition_id
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.image_id} {self.condition_id} {self.timestamp}"

    # relationship
    conditions = relationship("Conditions", back_populates="img_conditions")
    image = relationship("Image", back_populates="img_conditions")
