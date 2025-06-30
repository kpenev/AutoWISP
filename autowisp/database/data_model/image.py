"""Define the image table for the pipeline"""

from __future__ import annotations
from typing import List

from sqlalchemy import (
    Column,
    Integer,
    Boolean,
    String,
    Index,
    TIMESTAMP,
    ForeignKey,
)

from sqlalchemy.orm import Mapped, relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["Image", "ImageProcessingProgress", "ProcessedImages"]


class ProcessedImages(DataModelBase):
    """The table describing the processed images/channels by each step."""

    __tablename__ = "processed_images"

    image_id = Column(
        Integer,
        ForeignKey("image.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="The image that was processed.",
    )
    channel = Column(
        String(3), doc="The channel of the image that was processed."
    )
    progress_id = Column(
        Integer,
        ForeignKey(
            "image_processing_progress.id",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        doc="The id of the processing progress",
    )
    status = Column(
        Integer,
        nullable=False,
        doc="The status of the processing (0 = started, >0 = successfully saved"
        " progress, negative values indicate various reasons for failure). The "
        "meaning of negative values is step dependent. For most steps 1 is "
        "the final status, but for magnitude fitting the value indicates the "
        "iteration.",
    )
    final = Column(
        Boolean,
        nullable=False,
        doc="Is this the final processing status? The only case where "
        "`status`=1 is not final is for magnitude fitting, where there may be "
        "additional iterations needed.",
    )

    __table_args__ = (
        Index("processed_images_key2", "image_id", "channel", "progress_id"),
    )

    def __str__(self):
        return (
            f"({self.image_id}) {self.channel} {self.progress_id} "
            f"{self.timestamp}"
        )

    image = relationship("Image", back_populates="processing")
    processing = relationship(
        "ImageProcessingProgress", back_populates="applied_to"
    )


class Image(DataModelBase):
    """The table describing the image specified"""

    __tablename__ = "image"

    raw_fname = Column(
        String(1000),
        nullable=False,
        unique=True,
        doc="The full path of the raw image",
    )
    image_type_id = Column(
        Integer,
        ForeignKey("image_type.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the image type",
    )
    observing_session_id = Column(
        Integer,
        ForeignKey(
            "observing_session.id", onupdate="CASCADE", ondelete="RESTRICT"
        ),
        nullable=False,
        doc="The id of the observing session",
    )
    notes = Column(
        String(1000), nullable=True, doc="The notes provided for the image"
    )

    def __repr__(self):
        return (
            f"({self.id}) {self.raw_fname}: {self.image_type_id} "
            f"{self.observing_session_id} {self.notes} {self.timestamp}"
        )

    image_type = relationship("ImageType", back_populates="image")
    observing_session = relationship(
        "ObservingSession", back_populates="images"
    )
    processing: Mapped[List[ProcessedImages]] = relationship(
        back_populates="image"
    )


class ImageProcessingProgress(DataModelBase):
    """The table describing the Image Processing Progress"""

    __tablename__ = "image_processing_progress"

    step_id = Column(
        Integer,
        ForeignKey("step.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="Id of the step that was applied",
    )
    image_type_id = Column(
        Integer,
        ForeignKey("image_type.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="The id of the image type being processed",
    )
    configuration_version = Column(
        Integer, nullable=False, doc="config version of image"
    )
    host = Column(
        String(1000),
        nullable=False,
        doc="Hostname or other identifier of the computer where processing "
        "is/was done",
    )
    process_id = Column(
        Integer,
        nullable=False,
        doc="Identifier of the process performing this calibration step",
    )
    started = Column(
        TIMESTAMP, nullable=True, doc="The time processing started"
    )
    finished = Column(
        TIMESTAMP,
        nullable=True,
        doc="The time processing is known to have ended (NULL if possibly still"
        " on-going)",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes about the processing.",
    )

    def __str__(self):
        return (
            f"({self.id}) {self.step} v{self.configuration_version} for "
            f"{self.image_type.name} images started "
            f"{self.started} on {self.host} "
            + (
                "in progress"
                if self.finished is None
                else f"finished {self.finished}"
            )
            + f" timestamp: {self.timestamp}: {self.notes}"
        )

    step = relationship("Step")
    image_type = relationship("ImageType")

    applied_to: Mapped[List[ProcessedImages]] = relationship(
        back_populates="processing"
    )
