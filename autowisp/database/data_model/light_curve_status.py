"""Define table tracking the state of interrupted lightcurve processing."""

from __future__ import annotations

from sqlalchemy import Column, Integer, BigInteger, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import postgresql, mysql, sqlite

from autowisp.database.data_model.base import DataModelSubBase


__all__ = ["LightCurveStatus"]

GaiaIDType = BigInteger()
GaiaIDType = GaiaIDType.with_variant(postgresql.BIGINT(), "postgresql")
GaiaIDType = GaiaIDType.with_variant(mysql.BIGINT(), "mysql")
GaiaIDType = GaiaIDType.with_variant(sqlite.INTEGER(), "sqlite")


class LightCurveStatus(DataModelSubBase):
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
        doc="The status of the processing (0 = started, "
        ">0 = successfully saved progress, "
        "negative values indicate various reasons "
        "for failure).",
    )

    id = Column(
        Integer,
        primary_key=True,
        doc="A unique identifier for each row.",
    )

    def __str__(self):
        return f"Star {self.id} interrupted processing: {self.processing}"

    processing = relationship("LightCurveProcessingProgress")
