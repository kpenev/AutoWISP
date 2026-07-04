"""The AutoWISP exception hierarchy.

Every exception AutoWISP raises on purpose derives from
:class:`AutoWISPError`, which carries enough context (the affected
artifact(s), the pipeline run, the worker that raised it, a short
user-facing message, and an arbitrary ``details`` dict) to power CLI
messages, BUI surfacing, and post-mortem debugging from a single source
of truth. See ``error_handling_plan.md`` for the full design.

This module imports only :class:`FrozenRow` from the database package
(itself dependency-free), so the hierarchy stays free of a hard
SQLAlchemy dependency.
"""

import os
import traceback as traceback_module
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy

from autowisp.database.frozen_row import FrozenRow

git_id = "$Id$"


# A type-dispatch sanitizer: one return per handled kind reads clearer
# than nesting.
# pylint: disable=too-many-return-statements
def sanitize_for_json(obj, max_inline_array_size=64):
    """Coerce an otherwise-unserializable object to a JSON-friendly form.

    Intended as the ``default=`` argument to :func:`json.dump`/`json.dumps`
    when writing the error sidecar: ``details`` may carry numpy
    scalars/arrays, :class:`~pathlib.Path`, :class:`~datetime.datetime`,
    sets, and arbitrary objects. This is *total* -- it never raises, so a
    pathological value can never turn recording an error into a second
    error; the last resort is ``repr()``.

    Use directly (``default=sanitize_for_json``) for the default
    threshold, or ``default=functools.partial(sanitize_for_json,
    max_inline_array_size=N)`` to override it.

    Args:
        obj:    The value ``json`` could not serialize natively.

        max_inline_array_size(int):    ndarrays with at most this many
            elements are dumped in full; larger ones are summarized so a
            stray full-frame array cannot write hundreds of MB.

    Returns:
        A JSON-serializable stand-in for ``obj``.
    """

    if isinstance(obj, numpy.ndarray):
        if obj.size <= max_inline_array_size:
            return obj.tolist()
        return {
            "__ndarray__": {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "head": obj.ravel()[:max_inline_array_size].tolist(),
            }
        }
    if isinstance(obj, numpy.generic):
        return obj.item()
    if isinstance(obj, Path):
        # as_posix() so the serialized path is stable across platforms
        # (str() would emit backslashes on Windows).
        return obj.as_posix()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    try:
        return repr(obj)
    except Exception:  # pylint: disable=broad-except
        return "<unrepresentable>"


class Component(str, Enum):
    """Which broad part of AutoWISP an error belongs to."""

    STEP = "step"
    PIPELINE = "pipeline"
    BUI = "bui"


class FileKind(str, Enum):
    """The sort of file a :class:`RelatedFile` points at."""

    RAW_IMAGE = "raw_image"
    CALIBRATED_IMAGE = "calibrated_image"
    MASTER_BIAS = "master_bias"
    MASTER_DARK = "master_dark"
    MASTER_FLAT = "master_flat"
    MASTER_PHOTREF = "master_photref"
    DR_FILE = "dr_file"
    LIGHTCURVE = "lightcurve"
    CATALOG = "catalog"
    CONFIG = "config"
    OUTPUT = "output"
    OTHER = "other"


@dataclass(frozen=True)
class RelatedFile:
    """A file the error is about (input, output, or intermediate).

    Attributes:
        kind(FileKind):    What sort of file this is.

        path(Path):    Location of the file.

        role(str):    How the file relates to the failure, e.g.
            ``"input"`` / ``"intermediate"`` / ``"expected_output"``.
    """

    kind: FileKind
    path: Path
    role: str = ""


def _rebuild_autowisp_error(cls, args, state):
    """Reconstruct an :class:`AutoWISPError` subclass for unpickling.

    Bypasses ``__init__`` (so keyword-only arguments on subclasses do not
    block unpickling) and restores both ``BaseException.args`` (where the
    message lives, in C-level storage outside ``__dict__``) and the
    instance ``__dict__`` directly. See :meth:`AutoWISPError.__reduce__`.

    Args:
        cls(type):    The concrete exception class to rebuild.

        args(tuple):    The original ``self.args`` (the message).

        state(dict):    The instance ``__dict__`` captured at pickle
            time.

    Returns:
        AutoWISPError:    The reconstructed exception.
    """

    obj = cls.__new__(cls)
    obj.args = args
    obj.__dict__.update(state)
    return obj


class AutoWISPError(Exception):
    """Base class for every AutoWISP-raised exception.

    Every concrete subclass selects a :class:`Component`.

    Attributes:
        component(Component):    Set on each subclass; verified by tests.

        related_files(tuple):    The :class:`RelatedFile` entries this
            error is about.

        pipeline_run(FrozenRow or None):    Snapshot of the
            ``PipelineRun`` row (see :class:`FrozenRow`), set by the
            pipeline driver when it wraps a step's exception, or by
            ``wisp-*`` entry points. ``None`` for runs with no DB row.

        crashed(datetime or None):    When the failure surfaced, filled
            in by the top-level handler.

        subprocess_id(int or None):    PID of the multiprocessing worker
            that raised, when the exception travelled out of a Pool;
            ``None`` for errors raised in the main process.

        user_message(str):    Short, free of jargon, suitable for the
            BUI.

        details(dict):    Arbitrary key/value pairs giving extra context
            about the failure (e.g. shape mismatches, expected/actual
            values, parsed config). Useful to both users and developers.
    """

    component: Component  # set on each subclass; verified by tests

    # The many keyword-only arguments are the whole point: this is the
    # single context-carrying constructor for every AutoWISP error.
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        message: str,
        *,
        related_files: Sequence[RelatedFile] = (),
        pipeline_run: Optional[FrozenRow] = None,
        crashed: Optional[datetime] = None,
        subprocess_id: Optional[int] = None,
        user_message: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """Store the context attributes (see class ``Attributes``)."""

        super().__init__(message)
        self.related_files = tuple(related_files)
        self.pipeline_run = pipeline_run
        self.crashed = crashed
        self.subprocess_id = subprocess_id
        self.user_message = user_message or message
        self.details = dict(details or {})

    def __reduce__(self):
        """Pickle by restoring ``__dict__`` rather than re-running
        ``__init__``.

        Subclasses (e.g. :class:`StepError`) accept keyword-only
        arguments and carry context attributes that are not part of
        ``self.args``, so the default exception unpickler -- which calls
        ``cls(*self.args)`` -- would both drop those fields and (for
        required kwargs) raise ``TypeError``. Reconstructing through
        ``__new__`` + ``__dict__`` keeps every field intact and lets the
        exception travel back out of a multiprocessing worker faithfully.

        Returns:
            tuple:    ``(callable, args)`` per the pickle protocol.
        """

        return (
            _rebuild_autowisp_error,
            (type(self), self.args, self.__dict__.copy()),
        )

    def stamp_subprocess(self) -> None:
        """Record the current PID as the raising sub-process.

        Called by the worker before re-raising out of a multiprocessing
        Pool. Idempotent: a value already set in a deeper worker wins.

        Returns:
            None
        """

        if self.subprocess_id is None:
            self.subprocess_id = os.getpid()

    def with_pipeline_run(
        self, run: Optional[FrozenRow], *, crashed: Optional[datetime] = None
    ) -> "AutoWISPError":
        """Attach a :class:`FrozenRow` snapshot of the ``PipelineRun``.

        Args:
            run(FrozenRow or None):    Row snapshot to attach. Built by
                ``snapshot_row`` (parent) or from config primitives
                (worker), so host/started are already populated; nothing
                is reconstructed here.

            crashed(datetime or None):    Failure time; defaults to now
                if not already set on the exception.

        Returns:
            AutoWISPError:    ``self``, so it can be used inline before
                re-raising.
        """

        self.pipeline_run = run
        self.crashed = self.crashed or crashed or datetime.now(timezone.utc)
        return self

    def to_detail_dict(self) -> dict:
        """Return the heavy, non-column fields for the error sidecar.

        Complements the queryable columns of the ``Error`` row:
        everything here is what does *not* live inline on the row -- the
        full technical message, the complete related-file list (a superset
        of the artifact FKs), the arbitrary ``details`` dict, and the
        formatted traceback (``__cause__`` chain included). The result may
        still contain numpy/`Path`/etc. inside ``details``; serialize it
        with ``json.dump(..., default=sanitize_for_json)``.

        Returns:
            dict:    The sidecar payload (see ``error_handling_plan.md``).
        """

        return {
            "schema_version": 1,
            "message": str(self),
            "related_files": [
                {
                    "kind": related.kind.value,
                    "path": related.path.as_posix(),
                    "role": related.role,
                }
                for related in self.related_files
            ],
            "details": self.details,
            "traceback": "".join(
                traceback_module.format_exception(
                    type(self), self, self.__traceback__
                )
            ),
        }


class StepError(AutoWISPError):
    """Failure inside a processing step.

    Attributes:
        step_name(str or None):    Name of the step that failed. May be
            ``None`` at raise time and filled in from the ambient
            context by the capture layer.
    """

    component = Component.STEP

    def __init__(
        self, message: str, *, step_name: Optional[str] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.step_name = step_name


class PipelineError(AutoWISPError):
    """Failure in the orchestration layer (not "the algorithm")."""

    component = Component.PIPELINE


class BUIError(AutoWISPError):
    """Failure in the Django views/forms/templates of the BUI."""

    component = Component.BUI


# --- Concrete step-level exceptions, one per pipeline stage. ----------


class CalibrationError(StepError):
    """Failure in the calibrate step."""


class OutsideImageError(CalibrationError):
    """Attempt to access image data outside the bounds of the image.

    Raised only in the calibration path, so it specializes
    :class:`CalibrationError` rather than the generic :class:`StepError`.
    """


class StackToMasterError(StepError):
    """Failure stacking calibration frames into a master."""


class FindStarsError(StepError):
    """Failure in the find_stars step."""


class SolveAstrometryError(StepError):
    """Failure in the solve_astrometry step."""


class FitStarShapeError(StepError):
    """Failure in the fit_star_shape step."""


class MeasurePhotometryError(StepError):
    """Failure in the measure_aperture_photometry step."""


class FitPSFMapError(StepError):
    """Failure in the fit_source_extracted_psf_map step."""


class FitMagnitudesError(StepError):
    """Failure in the fit_magnitudes step."""


class CreateLightCurvesError(StepError):
    """Failure in the create_lightcurves step."""


class EPDError(StepError):
    """Failure in the EPD detrending step."""


class TFAError(StepError):
    """Failure in the TFA detrending step."""


class DetrendingStatError(StepError):
    """Failure computing detrending statistics."""


# --- Cross-cutting step "reason" exceptions. --------------------------
#
# These name *why* a step failed (a bad image, incompatible images, a
# non-converging iteration) rather than *which* step. They are raised
# from shared low-level modules (fits_utilities, image_utilities,
# iterative_rejection_util, ...) reached by many steps, so they specialize
# the generic StepError; the step identity is supplied separately by the
# ``step_name`` the capture layer stamps from the ambient context.


class ImageMismatchError(StepError):
    """Attempt to combine incompatible images in some way."""


class BadImageError(StepError):
    """An image does not look like it is expected to."""


class ConvergenceError(StepError):
    """Some iterative procedure failed to converge."""


# --- Pipeline-level exceptions. ---------------------------------------


class ConfigurationError(PipelineError):
    """Invalid or inconsistent pipeline configuration."""


class DatabaseError(PipelineError):
    """Failure interacting with the pipeline database."""


class HDF5LayoutError(PipelineError):
    """Error caused by invalid specification of HDF5 layout."""


class ResourceError(PipelineError):
    """Insufficient disk / memory / CPU to continue."""


class MasterSelectionError(PipelineError):
    """Failure selecting the master frame(s) for an image."""


class PhotrefBindingError(PipelineError):
    """Failure binding a photometric reference."""


class DependencyResolutionError(PipelineError):
    """Failure resolving processing-step dependencies."""


class WorkerCrashedError(PipelineError):
    """A multiprocessing worker died without preserving its exception.

    Re-raise wrapper used when a worker dies in a way that does not
    propagate the original exception (segfault, OOM-killer, ``os._exit``).
    """


# --- BUI-level exceptions. --------------------------------------------


class ViewError(BUIError):
    """Failure rendering or handling a BUI view."""


class FormValidationError(BUIError):
    """A BUI form failed validation in a way worth recording."""


class ProjectStateError(BUIError):
    """The BUI project is in a state that blocks the requested action."""
