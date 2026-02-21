"""Define the image diagnostics tables for the pipeline."""

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["DiagnosticType", "ImageDiagnostics"]


class DiagnosticType(DataModelBase):
    """The table listing available diagnostic quantities."""

    __tablename__ = "diagnostic_type"

    name = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="The name of the diagnostic.",
    )
    description = Column(
        String(1000),
        nullable=False,
        doc="A description of the diagnostic.",
    )


class ImageDiagnostics(DataModelBase):
    """Floating point diagnostics associated with a calibrated image."""

    __tablename__ = "image_diagnostics"

    image_id = Column(
        Integer,
        ForeignKey("image.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        doc="The image these diagnostics belong to.",
    )
    channel = Column(
        String(10),
        nullable=False,
        doc="The color channel this diagnostic value corresponds to.",
    )
    diagnostic_id = Column(
        Integer,
        ForeignKey(
            "diagnostic_type.id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
        doc="The diagnostic this value corresponds to.",
    )
    value = Column(
        Float,
        nullable=False,
        doc="The value of the diagnostic for this image and channel.",
    )

    __table_args__ = (
        Index(
            "image_channel_diagnostic",
            "image_id",
            "channel",
            "diagnostic_id",
            unique=True,
        ),
        Index("daignostic_value", "diagnostic_id", "value"),
    )

    image = relationship("Image", back_populates="diagnostics")
    type_ = relationship("DiagnosticType")
