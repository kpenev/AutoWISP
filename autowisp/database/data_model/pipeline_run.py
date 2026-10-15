"""Define the :class:`PipelineRun` table."""

from sqlalchemy import (
    Column,
    Double,
    Integer,
    String,
    TIMESTAMP,
)

from autowisp.database.data_model.base import DataModelBase

__all__ = ["PipelineRun"]


class PipelineRun(DataModelBase):
    """The table tracking runs of the pipeline."""

    __tablename__ = "pipeline_run"

    # The task of a run processing images or lightcurves, rather than e.g.
    # creating an experiment of the project.
    processing_task = "processing"

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
    code_version = Column(
        String(1000),
        nullable=True,
        doc="Identifier of the code version used for the run, as returned "
        "by get_code_version_str() (a git hash, with a ':dirty' suffix "
        "when the working tree had uncommitted changes).",
    )
    machine_id = Column(
        String(64),
        nullable=True,
        doc="Hashed identifier of the machine running the pipeline, which "
        "survives renaming it (see get_machine_id()). Null for runs recorded "
        "before it was, or on a machine with no identifier to find.",
    )
    process_started = Column(
        Double,
        nullable=True,
        doc="When the process running the pipeline started, in seconds since "
        "the epoch, so that a process reusing its ID after it ended is not "
        "taken for it. Null for runs recorded before it was.",
    )
    task = Column(
        String(50),
        nullable=False,
        default=processing_task,
        server_default=processing_task,
        doc="What the run does: 'processing' for a run of the pipeline's "
        "steps, or an operation on the project's experiments, e.g. 'create "
        "experiment' (see :mod:`autowisp.database.experiments`). Only "
        "processing runs count in an experiment's fingerprint.",
    )

    def __repr__(self):
        return (
            f"PipelineRun(id={self.id}, task={self.task!r}, "
            f"start_time={self.started}, end_time={self.finished})"
        )
