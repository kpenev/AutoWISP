"""Collection of non-standard exceptions raised by the pipeline.

These names are kept for backwards compatibility with existing call
sites. Each is now re-rooted into the :mod:`autowisp.exceptions`
hierarchy (gaining a :class:`~autowisp.exceptions.Component` and the
shared error context), while *also* keeping the stdlib base class it
used to derive from so that existing ``except ValueError`` /
``except RuntimeError`` / ``except IndexError`` handlers keep catching
them.
"""

from autowisp.exceptions import (
    CalibrationError,
    PipelineError,
    StepError,
)

git_id = "$Id$"


class OutsideImageError(CalibrationError, IndexError):
    """Attempt to access image data outside the bounds of the image."""


class ImageMismatchError(StepError, ValueError):
    """Attempt to combine incompatible images in some way."""


class BadImageError(StepError, ValueError):
    """An image does not look like it is expected to."""


class ConvergenceError(StepError, RuntimeError):
    """Some iterative procedure failed to converge."""


class HDF5LayoutError(PipelineError, RuntimeError):
    """Error caused by invalid specification of HDF5 layout."""
