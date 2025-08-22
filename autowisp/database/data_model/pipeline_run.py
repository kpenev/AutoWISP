"""Define the :class:`PipelineRun` table."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    TIMESTAMP,
)

from autowisp.database.data_model.base import DataModelBase

__all__ = ["PipelineRun"]


class PipelineRun(DataModelBase):
    """The table tracking runs of the pipeline."""

    __tablename__ = "pipeline_run"

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
        TIMESTAMP, nullable=False, doc="The start time of the pipeline run."
    )
    finished = Column(
        TIMESTAMP, nullable=True, doc="The end time of the pipeline run."
    )

    def __repr__(self):
        return (
            f"PipelineRun(id={self.id}, start_time={self.started}, "
            f"end_time={self.finished})"
        )
