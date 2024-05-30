"""Define the class that sets the processing order of step/image type."""

from sqlalchemy import\
    Column,\
    Integer,\
    TIMESTAMP,\
    ForeignKey

from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ['ProcessingSequence']

class ProcessingSequence(DataModelBase):
    """The sequence of steps/image type to be processed by the pipeline."""

    __tablename__ = 'processing_sequence'

    id = Column(
        Integer,
        primary_key=True,
        doc='The index of this processing within the sequence to be followed '
        'by the pipeline.'
    )
    step_id = Column(
        Integer,
        ForeignKey('step.id'),
        nullable=False,
        doc='The step to be executed.'
    )
    image_type_id = Column(
        Integer,
        ForeignKey('image_type.id'),
        nullable=True,
        doc='The image type to be processed by the step.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    step = relationship('Step')
    image_type = relationship('ImageType')

    def __repr__(self):
        return f'{self.step.name} {self.image_type.name}'
