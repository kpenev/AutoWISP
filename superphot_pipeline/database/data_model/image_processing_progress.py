"""Define image processing progress table for pipeline"""

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

__all__ = ['ImageProcessingProgress']

class ImageProcessingProgress(DataModelBase):
    """The table describing the Image Processing Progress"""

    __tablename__ = 'image_processing_progress'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image_proc_processing'
    )
    step = Column(
        Integer,
        ForeignKey('steps.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc = 'Id of the steps that was applied'
    )
    configuration_version = Column(
        Integer,
        ForeignKey('configuration.version',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable= False,
        doc = 'config version of image'
    )
    # timestamp
    timestamp = Column(
        TIMESTAMP,
        nullable= False,
        doc = 'When record was last changed'
    )

    def __init__(self, id, image_id, step_type_id, config_version, timestamp):
        self.id = id
        self.image_id = image_id
        self.step_type_id = step_type_id
        self.config_version = config_version
        self.timestamp = timestamp

    def __repr__(self):
        return f"({self.id}) {self.image_id} {self.step_type_id} {self.config_version} {self.timestamp}"

    #relationship
    #image = relationship("Image", back_populates="image_proc")
    #step_type = relationship("StepType", back_populates="image_proc")
    #config = relationship("Configuration", back_populates="imgprocprog")
