"""Define the image diagnostics tables for the pipeline."""

from sqlalchemy import Column, Integer, Float, String, ForeignKey, Index
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["DiagnosticTypes", "ImageDiagnostics"]


class DiagnosticTypes(DataModelBase):
    """The table listing available diagnostic quantities."""

    __tablename__ = "diagnostic_names"

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
    diagnostic_id = Column(
        Integer,
        ForeignKey("diagnostic.id", onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
        doc="The diagnostic this value corresponds to.",
    )
    value = Column(
        Float,
        nullable=False,
        doc="The value of the diagnostic for this image.",
    )

    __table_args__ = (
        Index("diagnostic_id"),
        Index("diagnostic_id", "value"),
    )



    image = relationship("Image", back_populates="diagnostics")
    type_ = relationship("DiagnosticTypes")
