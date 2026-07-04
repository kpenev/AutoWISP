"""Persist AutoWISP errors as a queryable row plus a JSON sidecar.

Each persisted error becomes a small, queryable
:class:`~autowisp.database.data_model.error.Error` row (the fields list
views and aggregate queries need) plus a per-error JSON sidecar holding
the heavy remainder (full message, complete related-file list,
``details``, traceback). The split keeps the SQLite file small while
still capturing rich context.

Two hard rules: persisting an error **never raises** (recording a
failure must not cause a second one), and **only the main process**
writes -- a worker's exception is already marshalled back to the main
process before the top-level handler calls :func:`persist_error`.
"""

import argparse
import gzip
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update

from autowisp.database.interface import (
    start_db_session,
    get_project_home,
    set_project_home,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error, Image, MasterFile

# pylint: enable=no-name-in-module
from autowisp.exceptions import Component, sanitize_for_json

git_id = "$Id$"

_logger = logging.getLogger(__name__)


def _resolve_artifact_fks(related_files, db_session):
    """Map related files to known artifact rows (best-effort).

    Resolves only artifacts that are genuine database rows with a stored
    path: a related file whose path matches ``Image.raw_fname`` gives the
    image, one matching ``MasterFile.filename`` gives the master. DR
    files, calibrated images, and lightcurves are HDF5 files with no row,
    so they are not linked here -- they remain in the sidecar's
    related-file list by path.

    Args:
        related_files(Sequence[RelatedFile]):    The error's related
            files.

        db_session:    Active database session.

    Returns:
        tuple:    ``(image_id, master_file_id)``, each ``None`` if no
            related file maps to such a row.
    """

    # as_posix() so matching against the DB's forward-slash paths works on
    # Windows too (str() would emit backslashes).
    paths = [Path(related.path).as_posix() for related in related_files]
    if not paths:
        return None, None
    # pylint: disable=no-member
    image_id = db_session.scalar(
        select(Image.id).where(Image.raw_fname.in_(paths))
    )
    master_file_id = db_session.scalar(
        select(MasterFile.id).where(MasterFile.filename.in_(paths))
    )
    # pylint: enable=no-member
    return image_id, master_file_id


def _error_bucket(exc):
    """Return the sidecar sub-directory name for ``exc``.

    Bucketing by run keeps directories small and makes "drop everything
    from run 88" a single ``rmtree``. Errors with no run go to ``bui`` or
    ``cli`` by component.
    """

    if exc.pipeline_run is not None:
        return str(exc.pipeline_run.id)
    if exc.component is Component.BUI:
        return "bui"
    return "cli"


def _build_error_row(exc, db_session):
    """Build the inline ``Error`` row (without ``sidecar_path``)."""

    image_id, master_file_id = _resolve_artifact_fks(
        exc.related_files, db_session
    )
    return Error(  # pylint: disable=not-callable
        pipeline_run_id=(
            exc.pipeline_run.id if exc.pipeline_run is not None else None
        ),
        component=exc.component.value,
        step_name=getattr(exc, "step_name", None),
        exception_class=type(exc).__name__,
        image_id=image_id,
        master_file_id=master_file_id,
        subprocess_id=exc.subprocess_id,
        user_message=exc.user_message,
        created=exc.crashed,
    )


def _write_sidecar(exc, error_id, bucket, *, gzip_threshold=64 * 1024):
    """Atomically write the sidecar JSON and return its relative path.

    The payload is written to a ``.tmp`` file and then ``os.replace``d
    into place, so a reader never sees a half-written file. Payloads above
    ``gzip_threshold`` bytes are gzipped (the stored filename records
    which).

    Args:
        exc(AutoWISPError):    The error to serialize.

        error_id(int):    The row id; names the file.

        bucket(str):    The sub-directory (see :func:`_error_bucket`).

        gzip_threshold(int):    Byte size above which the payload is
            gzipped.

    Returns:
        str:    The sidecar path relative to the project home.
    """

    project_home = get_project_home()
    relative_dir = os.path.join("errors", bucket)
    os.makedirs(os.path.join(project_home, relative_dir), exist_ok=True)

    payload = json.dumps(
        exc.to_detail_dict(), default=sanitize_for_json, indent=2
    ).encode("utf-8")

    suffix = ".json.gz" if len(payload) > gzip_threshold else ".json"
    relative_path = os.path.join(relative_dir, f"{error_id}{suffix}")
    absolute_path = os.path.join(project_home, relative_path)
    tmp_path = absolute_path + ".tmp"

    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(tmp_path, "wb") as sidecar:
        sidecar.write(payload)
    os.replace(tmp_path, absolute_path)
    return relative_path


def persist_error(exc, *, sidecar_gzip_threshold=64 * 1024):
    """Persist ``exc`` as an ``Error`` row plus a JSON sidecar.

    Best-effort and never raises. The row is committed first, so it
    survives even if the sidecar write later fails (its ``sidecar_path``
    then stays NULL and readers treat it as "inline fields only"). Only
    the parent process should call this.

    Args:
        exc(AutoWISPError):    The (already-stamped) error to record.

        sidecar_gzip_threshold(int):    Byte size above which the sidecar
            payload is gzipped.

    Returns:
        int or None:    The new ``Error.id``, or ``None`` if even the row
            insert failed.
    """

    error_id = None
    try:
        with start_db_session() as db_session:
            error_row = _build_error_row(exc, db_session)
            db_session.add(error_row)
            db_session.flush()
            error_id = error_row.id
    except Exception:  # pylint: disable=broad-except
        _logger.exception("Failed to record error row for %r", exc)
        return None

    try:
        relative_path = _write_sidecar(
            exc,
            error_id,
            _error_bucket(exc),
            gzip_threshold=sidecar_gzip_threshold,
        )
        with start_db_session() as db_session:
            db_session.execute(
                update(Error)
                .where(Error.id == error_id)  # pylint: disable=no-member
                .values(sidecar_path=relative_path)
            )
    except Exception:  # pylint: disable=broad-except
        _logger.exception(
            "Failed to write error sidecar for error %s", error_id
        )

    return error_id


def load_sidecar(error_row):
    """Return the parsed sidecar detail for an ``Error`` row, or ``None``.

    The lazy read path: list views use only the inline columns; this is
    called only when drilling into one error. A missing or unreadable
    sidecar degrades to ``None`` ("detail unavailable"), never raises.

    Args:
        error_row(Error):    The row whose sidecar to load.

    Returns:
        dict or None:    The parsed sidecar payload, or ``None``.
    """

    if not error_row.sidecar_path:
        return None
    absolute_path = os.path.join(get_project_home(), error_row.sidecar_path)
    try:
        opener = gzip.open if absolute_path.endswith(".gz") else open
        with opener(absolute_path, "rt", encoding="utf-8") as sidecar:
            return json.load(sidecar)
    except (OSError, ValueError):
        return None


def delete_error(error_id, db_session=None):
    """Delete an error record entirely: its row and its sidecar file.

    A no-op if the row does not exist. Safe to call from a user action.

    Args:
        error_id(int):    The id of the error to delete.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        bool:    True if a row was deleted, False if none was found.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return delete_error(error_id, own_session)

    row = db_session.get(Error, error_id)
    if row is None:
        return False
    if row.sidecar_path:
        _safe_unlink(os.path.join(get_project_home(), row.sidecar_path))
    db_session.delete(row)
    return True


def delete_all_error_sidecars(db_session=None):
    """Delete the sidecar file of every recorded error.

    Used when a project is deleted: removes exactly the files error
    persistence wrote (one per ``Error`` row), leaving any unrelated files
    under the ``errors`` directory untouched. The emptied directories are
    cleaned up by the caller's directory pruning.

    Args:
        db_session:    Optional active session; one is opened if omitted.

    Returns:
        None
    """

    if db_session is None:
        with start_db_session() as own_session:
            return delete_all_error_sidecars(own_session)

    project_home = get_project_home()
    for row in db_session.scalars(
        select(Error).where(
            Error.sidecar_path.isnot(None)  # pylint: disable=no-member
        )
    ).all():
        _safe_unlink(os.path.join(project_home, row.sidecar_path))
    return None


# --- Retention & cleanup. ---------------------------------------------


def parse_duration(text):
    """Parse a compact duration like ``30d`` / ``12h`` / ``2w`` to timedelta.

    Args:
        text(str):    An integer followed by a unit (``s``/``m``/``h``/
            ``d``/``w``).

    Returns:
        timedelta:    The parsed duration.

    Raises:
        ValueError:    If ``text`` is not a recognized duration.
    """

    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    match = re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", text)
    if not match:
        raise ValueError(
            f"Invalid duration {text!r}; expected e.g. '30d', '12h', '2w'."
        )
    return timedelta(seconds=int(match.group(1)) * unit_seconds[match.group(2)])


def _safe_unlink(path):
    """Remove ``path`` if present, swallowing OS errors. Returns success."""

    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        _logger.warning("Could not remove %r during cleanup.", path)
        return False


def _row_age(row):
    """The time an error row is dated by: its crash time, else write time."""

    when = row.created or row.timestamp
    if when is not None and when.tzinfo is not None:
        when = when.replace(tzinfo=None)
    return when


def _iter_sidecar_files(errors_dir):
    """Yield ``(absolute_path, basename)`` for every file under errors_dir."""

    for bucket in os.listdir(errors_dir):
        bucket_path = os.path.join(errors_dir, bucket)
        if not os.path.isdir(bucket_path):
            continue
        for name in os.listdir(bucket_path):
            yield os.path.join(bucket_path, name), name


def cleanup_errors(*, older_than=None):
    """Prune persisted errors: aged rows, orphan files, dangling rows.

    Three passes, all best-effort:

    1. **Aged rows** -- when ``older_than`` is given, delete every
       ``Error`` row dated (crash time, else row write time) before the
       cutoff, along with its sidecar.
    2. **Dangling rows** -- a surviving row whose ``sidecar_path`` points
       at a missing file has the path cleared (the row stays valid as
       inline-only).
    3. **Orphan files** -- any file under ``<project_home>/errors`` that
       is not the sidecar of a surviving row (leftovers from write-path
       crashes, including ``.tmp`` files) is removed.

    Args:
        older_than(timedelta or None):    Retention cutoff; ``None`` skips
            the aged-row pass and only sweeps orphans/dangling rows.

    Returns:
        dict:    Counts ``{"removed_rows", "removed_files",
            "cleared_dangling"}``.
    """

    project_home = get_project_home()
    errors_dir = os.path.join(project_home, "errors")
    removed_rows = 0
    removed_files = 0
    cleared_dangling = 0

    if older_than is not None:
        # Naive UTC to match the (tz-stripped) stored timestamps; see
        # _row_age. now(utc) avoids the deprecated utcnow().
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - older_than
        with start_db_session() as db_session:
            for row in db_session.scalars(select(Error)).all():
                age = _row_age(row)
                if age is None or age >= cutoff:
                    continue
                if row.sidecar_path:
                    _safe_unlink(os.path.join(project_home, row.sidecar_path))
                db_session.delete(row)
                removed_rows += 1

    # Survivors: clear dangling sidecar paths and collect the valid files.
    valid_files = set()
    with start_db_session() as db_session:
        for row in db_session.scalars(
            select(Error).where(
                Error.sidecar_path.isnot(None)  # pylint: disable=no-member
            )
        ).all():
            absolute_path = os.path.join(project_home, row.sidecar_path)
            if os.path.exists(absolute_path):
                valid_files.add(os.path.abspath(absolute_path))
            else:
                row.sidecar_path = None
                cleared_dangling += 1

    if os.path.isdir(errors_dir):
        for absolute_path, _name in _iter_sidecar_files(errors_dir):
            if os.path.abspath(absolute_path) not in valid_files:
                if _safe_unlink(absolute_path):
                    removed_files += 1

    return {
        "removed_rows": removed_rows,
        "removed_files": removed_files,
        "cleared_dangling": cleared_dangling,
    }


def cleanup_main():
    """CLI entry point for ``wisp-cleanup-errors``."""

    parser = argparse.ArgumentParser(
        prog="wisp-cleanup-errors",
        description=(
            "Prune persisted pipeline errors: delete error records older "
            "than a cutoff and clean up orphaned sidecar files."
        ),
    )
    parser.add_argument(
        "project_home", help="Path to the project home directory."
    )
    parser.add_argument(
        "--older-than",
        type=parse_duration,
        default=None,
        help="Delete errors older than this (e.g. '30d', '12h', '2w'). "
        "Omit to only sweep orphan files and dangling rows.",
    )
    args = parser.parse_args()

    set_project_home(args.project_home)
    summary = cleanup_errors(older_than=args.older_than)
    print(
        "Removed {removed_rows} error row(s), {removed_files} orphan "
        "file(s); cleared {cleared_dangling} dangling sidecar "
        "reference(s).".format(**summary)
    )
