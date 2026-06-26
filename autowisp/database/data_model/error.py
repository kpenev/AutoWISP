"""Define the :class:`Error` table: the queryable record of a failure."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    TIMESTAMP,
    ForeignKey,
)

from autowisp.database.data_model.base import DataModelBase

__all__ = ["Error"]


class Error(DataModelBase):
    """A persisted pipeline/step/BUI error.

    Holds the *queryable subset* of an :class:`~autowisp.exceptions.
    AutoWISPError` -- the fields list views and aggregate queries need, so
    they never have to open the sidecar. The heavy remainder (full
    message, complete related-file list, ``details``, traceback) lives in
    a per-error JSON sidecar referenced by :attr:`sidecar_path`; the two
    are disjoint, so nothing has to be kept in sync.

    Artifact links are foreign keys only for artifacts that are genuine
    database rows -- :attr:`image_id` and :attr:`master_file_id`. DR files
    and lightcurves are HDF5 files on disk with no row to reference, so
    they appear (by path) only in the sidecar's related-file list.

    All links use ``ON DELETE SET NULL`` so an error row stays valid as a
    historical record even if the artifact or run it referred to is later
    removed.
    """

    __tablename__ = "error"

    pipeline_run_id = Column(
        Integer,
        ForeignKey("pipeline_run.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        doc="The pipeline run during which the error occurred; NULL for "
        "standalone CLI/BUI errors with no run.",
    )
    component = Column(
        String(20),
        nullable=False,
        doc="Which component raised it: 'step', 'pipeline', or 'bui'.",
    )
    step_name = Column(
        String(255),
        nullable=True,
        doc="The processing step that failed; set for step errors.",
    )
    exception_class = Column(
        String(255),
        nullable=False,
        doc="Concrete exception class name, e.g. 'SolveAstrometryError', "
        "for filtering all occurrences of a kind of failure.",
    )
    image_id = Column(
        Integer,
        ForeignKey("image.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        doc="The image the error is about, when a related file maps to a "
        "known image row.",
    )
    master_file_id = Column(
        Integer,
        ForeignKey("master_file.id", onupdate="CASCADE", ondelete="SET NULL"),
        nullable=True,
        doc="The master file the error is about, when a related file maps "
        "to a known master row.",
    )
    subprocess_id = Column(
        Integer,
        nullable=True,
        doc="PID of the multiprocessing worker that raised, when the error "
        "travelled out of a worker; NULL for main-process errors.",
    )
    user_message = Column(
        String(2000),
        nullable=False,
        doc="Short, jargon-free description for the BUI/CLI. The full "
        "technical message lives in the sidecar.",
    )
    created = Column(
        TIMESTAMP,
        nullable=True,
        doc="When the failure surfaced (the exception's 'crashed' time).",
    )
    sidecar_path = Column(
        String(1000),
        nullable=True,
        doc="Path to the JSON detail sidecar, relative to the project "
        "home; NULL if the sidecar write failed (row stays valid).",
    )
    resolved = Column(
        TIMESTAMP,
        nullable=True,
        doc="When a user marked this error resolved; NULL while open. "
        "Open errors drive the error badge, progress-grid markers, and "
        "the start-processing gate; resolved errors are kept as history.",
    )

    def __repr__(self):
        return (
            f"Error(id={self.id}, component={self.component!r}, "
            f"exception_class={self.exception_class!r}, "
            f"step_name={self.step_name!r})"
        )
