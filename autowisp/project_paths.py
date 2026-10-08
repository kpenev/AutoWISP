"""Paths inside a project, kept as ``{PROJHOME}/...`` until a file is opened.

A project stores every path that lies inside it with its home replaced by the
``{PROJHOME}`` marker -- in the database, in DR and lightcurve files and in
FITS headers -- so that moving or copying the project directory is all it
takes to relocate it. Paths outside the project stay absolute.

Code carries paths in that stored form, and resolves them only where a file
is opened or created. This module holds the project home they are resolved
against: :func:`autowisp.database.interface.set_project_home` sets it, and it
lives here rather than there so that the file classes can resolve paths
without importing the database.
"""

import os.path
from pathlib import PurePath

from autowisp.exceptions import ConfigurationError

PROJHOME_MARKER = "{PROJHOME}"

_project_home = None


def get_project_home():
    """Return the project home directory currently being used."""

    return _project_home


def set_project_home_path(project_home):
    """
    Set the directory ``{PROJHOME}`` resolves to, without touching the DB.

    Only :func:`autowisp.database.interface.set_project_home` should call
    this; everything else sets the project home through it.

    Args:
        project_home(str):    The absolute path of the project home.

    Returns:
        None
    """

    global _project_home  # pylint: disable=global-statement
    _project_home = project_home


def _require_project_home():
    """Return the project home, raising if none has been set."""

    if _project_home is None:
        raise ConfigurationError(
            f"No project home is set, so {PROJHOME_MARKER} cannot be resolved."
        )
    return _project_home


class _ProjhomeSubstitution:
    """Substitute ``PROJHOME`` in a path, keeping or resolving it."""

    def __init__(self, substitutions=None):
        """Resolve ``PROJHOME`` if ``substitutions`` is None, else keep it."""

        self._substitutions = substitutions

    def __getitem__(self, key):
        if key == "PROJHOME":
            if self._substitutions is None:
                return _require_project_home()
            return PROJHOME_MARKER
        if self._substitutions is None:
            raise KeyError(key)
        return self._substitutions[key]


def fill_path_template(template, substitutions):
    """
    Fill a path template with everything except the project home.

    A ``PROJHOME`` entry in ``substitutions`` (e.g. a header recorded by an
    older version) is never used.

    Args:
        template(str):    The template, e.g.
            ``"{PROJHOME}/DR/{RAWFNAME}_{CLRCHNL}.h5"``.

        substitutions:    Mapping (e.g. a FITS header) supplying the other
            replacement fields.

    Returns:
        str:    The path in stored form, still starting with ``{PROJHOME}``
        if the template does.
    """

    return template.format_map(_ProjhomeSubstitution(substitutions))


def resolve_path(stored_path):
    """
    Return the path to open for a path in stored form.

    The project home is needed only if ``stored_path`` refers to it, so paths
    outside the project resolve even when none is set.

    Args:
        stored_path(str):    A path in stored form.

    Returns:
        str:    The path with ``{PROJHOME}`` replaced by the project home.
    """

    return os.fspath(stored_path).format_map(_ProjhomeSubstitution())


def to_stored_path(fname):
    """
    Return the stored form of a path entering from outside the code.

    Args:
        fname(str):    A path, e.g. from the command line. Relative paths are
            taken relative to the working directory.

    Returns:
        str:    ``{PROJHOME}/<relative>`` if ``fname`` lies under the project
        home, its absolute path otherwise.
    """

    fname = os.fspath(fname)
    if fname.startswith(PROJHOME_MARKER):
        return fname
    absolute = os.path.abspath(fname)
    project_home = _require_project_home()
    try:
        relative = PurePath(os.path.relpath(absolute, project_home))
    except ValueError:
        # On Windows, paths on different drives have no relative path.
        return absolute
    if relative.parts and relative.parts[0] == os.path.pardir:
        return absolute
    return PROJHOME_MARKER + "/" + relative.as_posix()
