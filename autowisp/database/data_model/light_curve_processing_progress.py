"""Define table tracknig what processing has been applied to which LCs."""

from __future__ import annotations

from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from autowisp.database.data_model.base import DataModelBase

__all__ = ["LightCurveProcessingProgress"]


class LightCurveProcessingProgress(DataModelBase):
    """The table describing the light curve processing progress"""

    __tablename__ = "light_curve_processing_progress"

    step_id = Column(
        Integer,
        ForeignKey("step.id", onupdate="CASCADE", ondelete="RESTRICT"),
        nullable=False,
        doc="Id of the step that was applied",
    )
    single_photref_id = Column(
        Integer,
        ForeignKey("master_file.id", onupdate="CASCADE", ondelete="RESTRICT"),
        doc="The ID of the single photometric reference for which LC points "
        "were processed.",
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
    final = Column(
        Boolean,
        nullable=False,
        default=False,
        doc="Is this the final processing status? The only case where "
        "`status`=1 is not final is for magnitude fitting, where there may be "
        "additional iterations needed.",
    )
    notes = Column(
        String(1000),
        nullable=True,
        doc="Any user supplied notes about the processing.",
    )

    def __str__(self):
        return (
            f"({self.id}) {self.step} v{self.configuration_version} on LCs "
            f"fit against {self.sphotref.filename} started "
            f"{self.started} on {self.host} "
            + (
                "in progress"
                if self.finished is None
                else f"finished {self.finished}"
            )
            + f" timestamp: {self.timestamp}: {self.notes}"
        )

    step = relationship("Step")
    sphotref = relationship("MasterFile")
