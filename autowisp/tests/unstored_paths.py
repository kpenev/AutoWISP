"""Find the paths a project holds other than in stored form.

A project holds every path inside it as ``{PROJHOME}/...``, so that moving or
copying its directory is all it takes to relocate it. Two kinds of value
break that:

* the project's home by absolute path, in any spelling a process could have
  written;
* a relative path, which would be read relative to whatever directory the
  process runs in.

Strings cannot be told apart from paths by their look (``'f/0.0'`` is a lens
focal ratio), so a value counts as a relative path only if, joined to the
home, it names something that exists. That misses templates and the names
of deleted files; the rules on configured paths and
:func:`autowisp.project_paths.to_stored_path` cover those.

Everything in the project is scanned except ``LOGS/``, which holds only
records of runs that nothing reads back.
"""

import os
import re
from collections import namedtuple
from functools import partial
from pathlib import PurePath

import h5py
import numpy
from astropy.io import fits
from sqlalchemy import MetaData, String, select

from autowisp.project_paths import PROJHOME_MARKER

UnstoredPath = namedtuple(
    "UnstoredPath", ["source", "location", "value", "kind"]
)
"""A value not in stored form: the file or database, where in it, which kind."""

ABSOLUTE_HOME = "absolute home"  # pylint: disable=invalid-name
RELATIVE_PATH = "relative path"  # pylint: disable=invalid-name


class _PathFinder:
    """Collect the values of one project that are not in stored form."""

    def __init__(self, project_home):
        """Look for paths into ``project_home``."""

        self._project_home = PurePath(os.path.abspath(project_home))
        spellings = {
            self._normalize(self._strip_drive_letter(spelling))
            for spelling in (
                os.path.abspath(project_home),
                os.path.realpath(project_home),
            )
        }
        # A directory whose name merely starts with the home's is not it.
        self._home_pattern = re.compile(
            "(?:" + "|".join(map(re.escape, spellings)) + r")(?![\w.-])",
            re.IGNORECASE if os.name == "nt" else 0,
        )
        self._kinds = {}
        self.found = []

    @staticmethod
    def _strip_drive_letter(path):
        """Return ``path`` without a drive letter, which may get lost.

        ``\\some\\path`` still names ``C:\\some\\path``'s home, and contains
        it as written with the drive. A share (``\\\\server\\share``) is kept:
        without it, too little of the path is left to recognise.
        """

        drive, rest = os.path.splitdrive(path)
        return rest if drive.endswith(":") else path

    @staticmethod
    def _normalize(value):
        """Return ``value`` with the system's separators replaced by ``/``."""

        return value.replace(os.sep, "/")

    def _names_existing_file(self, value):
        """Is ``value`` a relative path naming something that exists?

        Relative to the project home, which is where 0018 takes it to be,
        wherever it then leads. On Windows, a path without a drive (``\\x``)
        is relative too, to the current drive.
        """

        value = value.strip()
        if value.startswith(PROJHOME_MARKER) or PurePath(value).is_absolute():
            return False
        # Nothing but ``.`` and ``..`` exists trivially, and as a value is
        # more often not a path at all.
        if set(PurePath(value).parts) <= {".", ".."}:
            return False
        # A bare name is more often a word than a path ("DR", "Gaia"), and
        # every product path has a directory part.
        if len(PurePath(value).parts) < 2:
            return False
        return os.path.exists(self._project_home / value)

    def _get_kind(self, value):
        """Return which kind of unstored path ``value`` is, or None."""

        if value not in self._kinds:
            if self._home_pattern.search(self._normalize(value)):
                self._kinds[value] = ABSOLUTE_HOME
            elif self._names_existing_file(value):
                self._kinds[value] = RELATIVE_PATH
            else:
                self._kinds[value] = None
        return self._kinds[value]

    def check(self, source, location, value):
        """Record ``value`` if it is a path not in stored form."""

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                return
        if not isinstance(value, str):
            return
        kind = self._get_kind(value)
        if kind is not None:
            self.found.append(UnstoredPath(source, location, value, kind))

    def check_array(self, source, location, values):
        """Check every string in ``values``, of any shape or structure."""

        values = numpy.asarray(values)
        if values.dtype.names:
            for field in values.dtype.names:
                self.check_array(source, f"{location}[{field}]", values[field])
        elif values.dtype.kind in "SUO":
            for index, value in numpy.ndenumerate(values):
                self.check(
                    source,
                    location + (str(list(index)) if index else ""),
                    value,
                )

    def check_text(self, source, location, text):
        """Check free text, line by line and word by word."""

        for line_number, line in enumerate(text.splitlines(), 1):
            line_location = f"{location} line {line_number}".strip()
            line_kind = self._get_kind(line)
            if line_kind is not None:
                self.found.append(
                    UnstoredPath(source, line_location, line, line_kind)
                )
            # A line that is a path as a whole may contain spaces.
            if line_kind == RELATIVE_PATH:
                continue
            for word in line.split():
                word = word.strip("'\",;()[]")
                if self._get_kind(word) == RELATIVE_PATH:
                    self.found.append(
                        UnstoredPath(source, line_location, word, RELATIVE_PATH)
                    )

    def check_header(self, source, location, header):
        """Check the value and comment of every card of a FITS header."""

        for card in header.cards:
            self.check(source, f"{location} {card.keyword}", card.value)
            self.check(
                source, f"{location} {card.keyword} comment", card.comment
            )

    def scan_database(self, db_engine):
        """Check every string column of every table of the database."""

        metadata = MetaData()
        metadata.reflect(db_engine)
        with db_engine.connect() as connection:
            for table in metadata.sorted_tables:
                columns = [
                    column
                    for column in table.columns
                    if isinstance(column.type, String)
                ]
                if not columns:
                    continue
                for row in connection.execute(select(table)):
                    row = row._mapping  # pylint: disable=protected-access
                    key = {
                        column.name: row[column]
                        for column in table.primary_key.columns
                    }
                    for column in columns:
                        self.check(
                            "database",
                            f"{table.name}.{column.name} {key}",
                            row[column],
                        )

    def scan_fits(self, source, fname):
        """Check every header and every table column of a FITS file."""

        with fits.open(fname) as hdu_list:
            for hdu_index, hdu in enumerate(hdu_list):
                self.check_header(source, f"HDU {hdu_index}", hdu.header)
                # Before astropy 7, a compressed image is a binary table.
                if (
                    isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
                    and not isinstance(hdu, fits.CompImageHDU)
                    and hdu.data is not None
                ):
                    for column in hdu.columns.names:
                        self.check_array(
                            source,
                            f"HDU {hdu_index} column {column}",
                            hdu.data[column],
                        )

    def _scan_hdf5_object(self, source, name, h5_object):
        """Check the attributes of a group or dataset, and a dataset's text."""

        for attribute, value in h5_object.attrs.items():
            self.check_array(source, f"{name} attribute {attribute}", value)
        if not isinstance(h5_object, h5py.Dataset):
            return
        if h5_object.dtype == numpy.int8:
            # Text, such as the frame header, is stored as its bytes.
            try:
                text = h5_object[()].tobytes().decode("ascii")
            except UnicodeDecodeError:
                return
            # A FITS header is 80 character cards, the first holding a value
            # (SIMPLE or XTENSION).
            if len(text) % 80 == 0 and text[8:10] == "= ":
                self.check_header(source, name, fits.Header.fromstring(text))
            else:
                self.check_text(source, name, text)
        elif h5_object.dtype.kind in "SUO" or h5_object.dtype.names:
            self.check_array(source, name, h5_object[()])

    def scan_hdf5(self, source, fname):
        """Check every attribute and every string dataset of an HDF5 file."""

        with h5py.File(fname, "r") as h5_file:
            # visititems() does not visit the root itself.
            self._scan_hdf5_object(source, "/", h5_file)
            h5_file.visititems(partial(self._scan_hdf5_object, source))

    def scan_file(self, source, fname):
        """Check a file of the project, by its type."""

        if os.path.basename(fname).startswith("autowisp.db"):
            # The database, scanned through its engine, and its journal.
            return
        suffix = PurePath(fname).suffix
        if suffix in (".fits", ".fz"):
            self.scan_fits(source, fname)
        elif suffix in (".h5", ".hdf5"):
            self.scan_hdf5(source, fname)
        else:
            with open(fname, "rb") as opened:
                content = opened.read()
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{fname!r} is neither text nor a file type the scan "
                    "can read: teach it this one."
                ) from error
            self.check_text(source, "", text)


def find_unstored_paths(project_home, db_engine):
    """
    Return every path the project holds other than in stored form.

    Args:
        project_home(str):    The project's home directory.

        db_engine:    The engine of the project's database.

    Returns:
        [UnstoredPath]:    Each value found, with the file it is in (relative
        to the project home, ``/``-separated) or ``"database"``, and where in
        that.

    Raises:
        ValueError:    If the project holds a binary file of a type the scan
            cannot read, which would otherwise go unchecked.
    """

    finder = _PathFinder(project_home)
    finder.scan_database(db_engine)
    project_home = os.path.abspath(project_home)
    for directory, subdirectories, fnames in os.walk(project_home):
        if directory == project_home and "LOGS" in subdirectories:
            subdirectories.remove("LOGS")
        subdirectories.sort()
        for fname in sorted(fnames):
            fname = os.path.join(directory, fname)
            finder.scan_file(
                PurePath(os.path.relpath(fname, project_home)).as_posix(), fname
            )
    return finder.found
