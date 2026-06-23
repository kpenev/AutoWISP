"""Shared fixtures for the error-handling unit tests.

Keeps the canonical "realistic" sample errors in one place so the
hierarchy, persistence, and rendering tests all exercise the same shape.
"""

from pathlib import Path

from autowisp.exceptions import FileKind, FindStarsError, RelatedFile


def find_stars_related_files():
    """The related files of a ``find_stars`` failure (real I/O shape).

    The step's input is a calibrated FITS image and its (expected) output
    is the image's DR file.
    """

    return [
        RelatedFile(
            FileKind.CALIBRATED_IMAGE,
            Path("/data/cal/img001.fits"),
            "input",
        ),
        RelatedFile(
            FileKind.DR_FILE,
            Path("/data/dr/img001.h5"),
            "expected_output",
        ),
    ]


def make_find_stars_error(*, related_files=None, details=None):
    """Build a representative ``FindStarsError`` for the tests.

    Args:
        related_files(Sequence[RelatedFile] or None):    Override the
            default calibrated-in / DR-out related files.

        details(dict or None):    Override the default ``details`` dict.

    Returns:
        FindStarsError:    The sample error (no cause/traceback attached;
            callers that need one raise inside an ``except`` and set it).
    """

    return FindStarsError(
        "no stars found",
        step_name="find_stars",
        related_files=(
            find_stars_related_files()
            if related_files is None
            else related_files
        ),
        details=(
            {"brightness_quantile": 0.999} if details is None else details
        ),
    )
