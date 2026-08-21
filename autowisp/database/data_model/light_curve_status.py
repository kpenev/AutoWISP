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
        # The Gaia source id of the star, not a row counter --
        # LightCurveProcessingManager stores src_id here. Those run to ~8e17,
        # which overflows the 32-bit INT that ``Integer`` becomes on MySQL and
        # MariaDB ("Out of range value for column 'id'"). SQLite hid it: its
        # INTEGER is 64-bit and it does not enforce declared types anyway.
        GaiaIDType,
        primary_key=True,
        doc="The Gaia source id of the star whose lightcurve this row tracks.",
    )

    def __str__(self):
        return f"Star {self.id} interrupted processing: {self.processing}"

    processing = relationship("LightCurveProcessingProgress")
