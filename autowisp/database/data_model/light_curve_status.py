"""Define table tracking the state of interrupted lightcurve processing."""

from __future__ import annotations

from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["LightCurveStatus"]


class LightCurveStatus(DataModelBase):
    """Table tracking the status of lightcurves for interrupted steps."""

    __tablename__ = "light_curve_status"

    progress_id = Column(
        Integer,
        ForeignKey(
            "light_curve_processing_progress.id",
            onupdate="CASCADE",
            ondelete="RESTRICT",
        ),
        primary_key=True,
        doc="The ID of the LC processing progress which was interrupted",
    )
    status = Column(
        Integer,
        nullable=False,
        doc="The status of the processing (0 = started, >0 = successfully "
        "saved progress, negative values indicate various reasons for failure).",
    )

    def __str__(self):
        return f"Star {self.id} interrupted processing: {self.processing}"

    processing = relationship("LightCurveProcessingProgress")
