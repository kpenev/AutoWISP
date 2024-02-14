"""Define class specifying what masters are needed for what step/image type."""

from __future__ import annotations

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey
from sqlalchemy.orm import relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__ = ['RequiredMasterTypes']

class RequiredMasterTypes(DataModelBase):
    """The table describing the prerequisites for a step to run"""

    __tablename__ = 'required_master_types'

    step_id = Column(
        Integer,
        ForeignKey('step.id'),
        primary_key=True,
        doc='The step for which this prerequisite applies.'
    )
    image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        primary_key=True,
        doc='The image type for which this prerequisite applies.'
    )
    master_type_id = Column(
        Integer,
        ForeignKey('master_type.id'),
        primary_key=True,
        doc='The master type required for the step to run.'
    )
    config_name = Column(
        String(50),
        nullable=False,
        doc='The name of the configuration parameter that specifies the '
        'master when running the specified step.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    step = relationship('Step')
    image_type = relationship('ImageType')
    master_type = relationship('MasterType')

    def __repr__(self):
        return (f'{self.step} {self.image_type} needs master '
                f'{self.master_type.name}')
