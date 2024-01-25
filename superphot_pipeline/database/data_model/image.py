"""Define the image table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import\
    Column,\
    Integer,\
    Boolean,\
    String,\
    TIMESTAMP,\
    ForeignKey,\
    Table

from sqlalchemy.orm import Mapped, mapped_column, relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__= ['Image', 'ImageProcessingProgress', 'ProcessedImages']

_processing_input = Table(
    'processing_input',
    DataModelBase.metadata,
    Column('completed_progress_id',
           ForeignKey('image_processing_progress.id'),
           primary_key=True),
    Column('input_progress_id',
           ForeignKey('image_processing_progress.id'),
           primary_key=True),
    Column('timestamp',
           TIMESTAMP,
           nullable=False,
           doc='When was this record last changed.')
)

class ProcessedImages(DataModelBase):
    """The table describing the processed images/channels by each step."""

    __tablename__ = 'processed_images'

    image_id = Column(
        Integer,
        ForeignKey('image.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='The image that was processed.'
    )
    channel = Column(
        String(3),
        primary_key=True,
        doc='The channel of the image that was processed.'
    )
    progress_id = Column(
        Integer,
        ForeignKey('image_processing_progress.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        primary_key=True,
        doc='The id of the processing progress'
    )
    status = Column(
        Integer,
        nullable=False,
        doc='The status of the processing (0 = started, >0 = successfully saved'
        ' progress, negative values indicate various reasons for failure). The '
        'meaning of negative values is step dependent. For most steps 1 is '
        'the final status, but for magnitude fitting the value indicates the '
        'iteration.'
    )
    final = Column(
        Boolean,
        nullable=False,
        doc='Is this the final processing status? The only case where '
        '`status`=1 is not final is for magnitude fitting, where there may be '
        'additional iterations needed.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __repr__(self):
        return (
            f'({self.image_id}) {self.channel} {self.progress_id} '
            f'{self.timestamp}'
        )

    image = relationship("Image",
                         back_populates="processing")
    processing = relationship("ImageProcessingProgress",
                              back_populates="applied_to")


class Image(DataModelBase):
    """The table describing the image specified"""

    __tablename__ = 'image'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image'
    )
    raw_fname = Column(
        String(1000),
        nullable=False,
        unique=True,
        doc='The filename of the raw image'
    )
    image_type_id = Column(
        Integer,
        ForeignKey('image_type.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the image type'
    )
    observing_session_id = Column(
        Integer,
        ForeignKey('observing_session.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc='The id of the observing session'
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc='The notes provided for the image'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __repr__(self):
        return (
            f'({self.id}) {self.raw_fname}: {self.image_type_id} '
            f'{self.observing_session_id} {self.notes} {self.timestamp}'
        )

    image_type = relationship("ImageType",
                              back_populates="image")
    observing_session = relationship("ObservingSession",
                                     back_populates="images")
    processing: Mapped[List[ProcessedImages]] = relationship(
        back_populates='image'
    )

class ImageProcessingProgress(DataModelBase):
    """The table describing the Image Processing Progress"""

    __tablename__ = 'image_processing_progress'

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image_proc_processing'
    )

    step_id = Column(
        Integer,
        ForeignKey('step.id',
                   onupdate='CASCADE',
                   ondelete='RESTRICT'),
        nullable=False,
        doc = 'Id of the step that was applied'
    )
    configuration_version = Column(
        Integer,
        nullable=False,
        doc='config version of image'
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc='Any user supplied notes about the processing.'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable= False,
        doc = 'When record was last changed'
    )

    def __str__(self):
        return (
            f'({self.id}) {self.step} v{self.configuration_version}'
            f'{self.timestamp}: {self.notes}'
        )

    step = relationship('Step')

    inputs: Mapped[List[ImageProcessingProgress]] = relationship(
        secondary=_processing_input,
        primaryjoin=(id == _processing_input.c.completed_progress_id),
        secondaryjoin=(id == _processing_input.c.input_progress_id),
        backref='consumers'
    )

    applied_to: Mapped[List[ProcessedImages]] = relationship(
        back_populates='processing'
    )
