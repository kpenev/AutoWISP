"""Tell which pipeline runs are still going, and on which machine.

A run records the machine running it, by a hashed identifier, and when its
process started, as well as the host name and process id
(:class:`PipelineRun`). The machine and the process are what identify it: a
host name can change while the run goes on (macOS renames itself with the
network), and a process id can be reused once the run has died.

Machines cloned from one image (virtual machines, cluster nodes, containers)
share an identifier unless it is regenerated (``systemd-machine-id-setup``),
and then take each other's runs for their own. Where several machines share a
project's server database, each needs an identifier of its own.
"""

import logging
import os
from functools import cache

import machineid
import psutil
from sqlalchemy import select, sql

# pylint: disable=no-name-in-module
from autowisp.database.data_model import PipelineRun

# pylint: enable=no-name-in-module
from autowisp.exceptions import get_hostname
from autowisp.miscellaneous import get_code_version_str

_logger = logging.getLogger(__name__)

# How far, in seconds, a process's start time may stray from the one
# recorded and still be the same process. The system derives it from its boot
# time, which it may estimate again after the clock is adjusted; a process
# reusing the id starts far later than that.
_START_TIME_TOLERANCE = 1.0  # pylint: disable=invalid-name


@cache
def get_machine_id():
    """
    Return this machine's identifier, or None if it has none to find.

    Hashed, since the raw identifier is meant to stay private, and a project
    database or crash report may be shared. Found once per process: on macOS
    each lookup runs ``ioreg``.
    """

    try:
        return machineid.hashed_id("autowisp")
    except machineid.MachineIdNotFound:
        return None


def get_process_started(process_id=None):
    """Return when a process (this one by default) started, as a timestamp."""

    return psutil.Process(process_id).create_time()


def record_run(db_session, task=PipelineRun.processing_task):
    """
    Record a run by this process, and return it with its id assigned.

    Flushed, not committed: the caller's transaction decides when it becomes
    visible, so that it can record something else along with it.

    Args:
        db_session:    The session to record the run in.

        task(str):    What the run does (see :attr:`PipelineRun.task`).

    Returns:
        PipelineRun:    The run.
    """

    pipeline_run = PipelineRun(
        host=get_hostname(),
        process_id=os.getpid(),
        started=sql.func.now(),  # pylint: disable=not-callable
        code_version=get_code_version_str(),
        machine_id=get_machine_id(),
        process_started=get_process_started(),
        task=task,
    )
    db_session.add(pipeline_run)
    db_session.flush()
    return pipeline_run


def is_this_machine(pipeline_run):
    """
    Was the run started on this machine?

    By the machine's identifier, which survives renaming it. Runs recorded
    before identifiers were, and machines with none to find, are compared by
    host name instead.
    """

    machine_id = get_machine_id()
    if pipeline_run.machine_id is not None and machine_id is not None:
        return pipeline_run.machine_id == machine_id
    return pipeline_run.host == get_hostname()


def _is_process_alive(pipeline_run):
    """Is the process of a run of this machine still running it?"""

    try:
        process = psutil.Process(pipeline_run.process_id)
        if process.status() == psutil.STATUS_ZOMBIE:
            # Ended, and only waiting for its parent to collect it.
            return False
        started = process.create_time()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Another user's process: it exists, which is all there is to know.
        return True
    if pipeline_run.process_started is None:
        # Recorded before start times were: the process id is all there is.
        return True
    return abs(started - pipeline_run.process_started) <= _START_TIME_TOLERANCE


def find_active_runs(db_session):
    """
    Return the runs still going, on this machine and on others.

    A run of this machine whose process is gone ended without saying so, e.g.
    killed, and is recorded as finished. Another machine's run is never: there
    is no telling from here whether it is still going.

    Args:
        db_session:    The session to read the runs in, and to record as
            finished the runs of this machine that ended.

    Returns:
        [PipelineRun]:    The runs going on this machine.

        [PipelineRun]:    The unfinished runs of other machines.
    """

    here = []
    elsewhere = []
    for pipeline_run in db_session.scalars(
        select(PipelineRun).filter_by(finished=None)
    ).all():
        if not is_this_machine(pipeline_run):
            elsewhere.append(pipeline_run)
        elif _is_process_alive(pipeline_run):
            here.append(pipeline_run)
        else:
            _logger.info(
                "Marking %s as finished: its process is gone.", pipeline_run
            )
            # pylint: disable-next=not-callable
            pipeline_run.finished = sql.func.now()
    return here, elsewhere


def find_earlier_run(db_session, pipeline_run_id):
    """
    Return the earliest run of this machine still going, if before this one.

    A run records itself, then calls this, and gives up if it returns a run:
    the tie between runs starting together is broken by their ids, which
    the database hands out in the order the runs recorded themselves, so
    exactly one of them goes on, whichever order they check in. A run of
    another machine may have ended without anyone here being able to tell,
    and is warned about rather than waited for.

    Args:
        db_session:    The session to read the runs in, as for
            find_active_runs().

        pipeline_run_id(int):    The id of the run asking, already recorded.

    Returns:
        PipelineRun or None:    The earliest run going on this machine, if it
        was recorded before the one asking.
    """

    here, elsewhere = find_active_runs(db_session)
    for pipeline_run in elsewhere:
        _logger.warning(
            "Run %d, on %s, has not finished; it may still be processing "
            "this project.",
            pipeline_run.id,
            pipeline_run.host,
        )
    return min(
        (
            pipeline_run
            for pipeline_run in here
            if pipeline_run.id < pipeline_run_id
        ),
        key=lambda pipeline_run: pipeline_run.id,
        default=None,
    )
