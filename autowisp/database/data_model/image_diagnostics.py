"""Define the image diagnostics tables for the pipeline."""

from sqlalchemy import (
    Column,
    Integer,
    Double,
    String,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["DiagnosticType", "ImageDiagnostics", "PhotometryDiagnostics"]


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


class ImageDiagnosticBase(DataModelBase):
    """Base class for all per-image diagnostics."""

    __abstract__ = True

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
        Double,
        nullable=False,
        doc="The value of the diagnostic for this image and channel.",
    )


class ImageDiagnostics(ImageDiagnosticBase):
    """Floating point diagnostics associated with a calibrated image."""

    __tablename__ = "image_diagnostics"

    image = relationship("Image", back_populates="diagnostics")

    type_ = relationship("DiagnosticType")

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


class PhotometryDiagnostics(ImageDiagnosticBase):
    """Floating point diagnostics for each photometry in each image."""

    __tablename__ = "photometry_diagnostics"

    photometry_id = Column(
        Integer,
        nullable=False,
        doc="The photometry these diagnostics belong to.",
    )

    image = relationship("Image", back_populates="photometry_diagnostics")

    type_ = relationship("DiagnosticType")

    __table_args__ = (
        Index(
            "img_chnl_phot_diagnostic",
            "image_id",
            "channel",
            "photometry_id",
            "diagnostic_id",
            unique=True,
        ),
        Index("diagnostic_value", "diagnostic_id", "value"),
    )
