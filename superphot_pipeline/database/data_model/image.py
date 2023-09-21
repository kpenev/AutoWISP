"""Define the image table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import\
    Column,\
    Integer,\
    String,\
    TIMESTAMP,\
    ForeignKey,\
    Table

from sqlalchemy.orm import Mapped, mapped_column, relationship

from superphot_pipeline.database.data_model.base import DataModelBase

__all__= ['Image', 'ImageProcessingProgress']

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

_processed_images = Table(
    'processed_images',
    DataModelBase.metadata,
    Column('image_id',
           ForeignKey('image.id'),
           primary_key=True),
    Column('progress_id',
           ForeignKey('image_processing_progress.id'),
           primary_key=True),
    Column('timestamp',
           TIMESTAMP,
           nullable=False,
           doc='When was this record last changed.')
)

class Image(DataModelBase):
    """The table describing the image specified"""

    __tablename__ = 'image'

    id = Column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image'
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
        nullable=False,
        doc='The notes provided for the image'
    )
    timestamp = Column(
        TIMESTAMP,
        nullable=False,
        doc='When was this record last changed.'
    )

    def __repr__(self):
        return (
            f'({self.id})  {self.image_type_id} {self.observing_session_id} '
            f'{self.notes} {self.timestamp}'
        )

    image_type = relationship("ImageType",
                              back_populates="image")
    observing_session = relationship("ObservingSession",
                                     back_populates="images")
    processing: Mapped[List[ImageProcessingProgress]] = relationship(
        secondary=_processed_images,
        back_populates='images'
    )

class ImageProcessingProgress(DataModelBase):
    """The table describing the Image Processing Progress"""

    __tablename__ = 'image_processing_progress'

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        doc='A unique identifier for each image_proc_processing'
    )

    step = Column(
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

    inputs: Mapped[List[ImageProcessingProgress]] = relationship(
        secondary=_processing_input,
        primaryjoin=(id == _processing_input.c.completed_progress_id),
        secondaryjoin=(id == _processing_input.c.input_progress_id),
        backref='consumers'
    )

    images: Mapped[List[Image]] = relationship(
        secondary=_processed_images,
        back_populates='processing'
    )
