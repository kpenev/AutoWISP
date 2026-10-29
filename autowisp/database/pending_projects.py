"""Keep the record of a project being added by cloning its database.

A server project's home copied by hand is added by cloning the database it
names (:func:`.clone_server_database`), which can take a while, so it is done
in the background. The project is registered only once the clone is
complete: until then its home still names the database it was copied with.
Meanwhile the addition is recorded in a file of its own under the user data
directory's ``pending_projects/``, which is how the interface follows it:

* **Written first by the interface**, before it launches the clone
  (:func:`add_pending`): ``name``, ``description``, ``home``, ``target_url``
  (None to clone into SQLite) and ``launched``, a timestamp.
* **Then only by the clone** (:func:`run_clone`), which adds its
  ``process_id`` and ``process_started`` as its first act, and its
  ``outcome`` as its last: ``"done"`` with the experiments it made ``stale``,
  or ``"failed"`` with a ``message``.

Each write replaces the whole file at once, so it is never read half
written. Free of Django: the interface registers the project itself, once it
finds the clone done (:func:`list_pending`).
"""

import json
import logging
import os
import re
import time
from uuid import uuid4

import platformdirs

from autowisp.database.existing_project import clone_server_database
from autowisp.database.pipeline_runs import (
    get_process_started,
    is_process_alive,
)

_logger = logging.getLogger(__name__)

# How long, in seconds, a clone is given to record its process before it is
# taken never to have started, e.g. for a broken install.
_START_TIMEOUT = 60.0  # pylint: disable=invalid-name


def get_pending_dir():
    """Return the directory holding the records of pending additions."""

    return os.path.join(
        platformdirs.user_data_dir("autowisp"), "pending_projects"
    )


def _is_pending_id(pending_id):
    """Is this an id :func:`add_pending` gives, naming no file elsewhere?"""

    return re.fullmatch("[0-9a-f]{32}", pending_id) is not None


def get_pending_fname(pending_id, pending_dir=None):
    """
    Return the file recording a pending addition.

    Args:
        pending_id(str):    The addition's id, as :func:`add_pending` gave it.

        pending_dir(str):    The directory of the records, None for
            :func:`get_pending_dir`.

    Returns:
        str:    The file's name.

    Raises:
        ValueError:    If the id is not one :func:`add_pending` gives, which
            could name a file elsewhere.
    """

    if not _is_pending_id(pending_id):
        raise ValueError(f"{pending_id!r} is not the id of a pending addition.")
    return os.path.join(pending_dir or get_pending_dir(), pending_id + ".json")


def _write_record(fname, record):
    """Replace the record in the file at once, never leaving it half written."""

    with open(fname + ".new", "w", encoding="utf-8") as record_file:
        json.dump(record, record_file)
    for _ in range(50):
        try:
            os.replace(fname + ".new", fname)
            return
        except PermissionError:
            # Windows refuses to replace a file another process has open, as
            # the interface does for a moment while it reads the record.
            time.sleep(0.1)
    os.replace(fname + ".new", fname)


def _read_record(fname):
    """Return the record in the file."""

    with open(fname, encoding="utf-8") as record_file:
        return json.load(record_file)


def add_pending(name, description, home, target_url, pending_dir=None):
    """
    Record an addition about to be launched, and return its id.

    Args:
        name(str):    The name to register the project under.

        description(str):    Its description.

        home(str):    The project's home.

        target_url(str or None):    The URL of the empty server database to
            clone into, None to clone into SQLite.

        pending_dir(str):    The directory of the records, None for
            :func:`get_pending_dir`.

    Returns:
        str:    The addition's id, for :func:`get_pending_fname`.
    """

    pending_dir = pending_dir or get_pending_dir()
    os.makedirs(pending_dir, exist_ok=True)
    pending_id = uuid4().hex
    _write_record(
        get_pending_fname(pending_id, pending_dir),
        {
            "name": name,
            "description": description,
            "home": home,
            "target_url": target_url,
            "launched": time.time(),
        },
    )
    return pending_id


def run_clone(fname):
    """
    Clone the database of a pending addition, recording how it went.

    Whatever ends the clone with an exception is recorded as its failure,
    and logged, rather than raised: the record is all the interface reads.

    Args:
        fname(str):    The file recording the addition.

    Returns:
        None
    """

    record = _read_record(fname)
    record.update(process_id=os.getpid(), process_started=get_process_started())
    _write_record(fname, record)
    try:
        stale = clone_server_database(record["home"], record["target_url"])
    # Anything ending the clone is its outcome, to be reported.
    except Exception as error:  # pylint: disable=broad-exception-caught
        _logger.exception(
            "Cloning the database of the project in %s failed.", record["home"]
        )
        record.update(
            outcome="failed",
            message=getattr(error, "user_message", None)
            or str(error)
            or type(error).__name__,
        )
    else:
        record.update(outcome="done", stale=stale)
    _write_record(fname, record)


def _judge_record(record):
    """Return how the addition recorded stands, read from the record alone."""

    if "outcome" in record:
        return record["outcome"]
    if "process_id" in record:
        if is_process_alive(record["process_id"], record["process_started"]):
            return "cloning"
        return "interrupted"
    if time.time() - record["launched"] < _START_TIMEOUT:
        return "cloning"
    return "not started"


def list_pending(pending_dir=None):
    """
    Return each pending addition, with how it stands, oldest first.

    Nothing is changed: the interface registers the done ones and removes
    their records (:func:`remove_pending`), as it does those dismissed.

    Args:
        pending_dir(str):    The directory of the records, None for
            :func:`get_pending_dir`.

    Returns:
        [dict]:    Each addition's record, with its ``id`` and its ``state``:
        ``"cloning"``, also while the clone has not yet recorded its process;
        ``"done"`` or ``"failed"``, its outcome; ``"interrupted"``, its
        process gone without recording one; or ``"not started"``, no process
        recorded once it was given long enough to.
    """

    pending_dir = pending_dir or get_pending_dir()
    if not os.path.isdir(pending_dir):
        return []
    result = []
    for entry in os.listdir(pending_dir):
        pending_id, extension = os.path.splitext(entry)
        # Anything else in the directory is not a record add_pending wrote.
        if extension != ".json" or not _is_pending_id(pending_id):
            continue
        fname = os.path.join(pending_dir, entry)
        record = _read_record(fname)
        state = _judge_record(record)
        if state in ("interrupted", "not started"):
            # The clone may have recorded more since: its outcome, written
            # before its process ends, or its process, once it has started.
            record = _read_record(fname)
            state = _judge_record(record)
        result.append(dict(record, id=pending_id, state=state))
    return sorted(result, key=lambda pending: pending["launched"])


def remove_pending(pending_id, pending_dir=None):
    """Remove the record of a pending addition, e.g. once it is registered."""

    os.remove(get_pending_fname(pending_id, pending_dir))
