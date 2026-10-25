"""Start commands that outlive the request starting them."""

import os
import subprocess
import sys
import threading

import platformdirs
from sqlalchemy import func, select

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import PipelineRun

# pylint: enable=no-name-in-module
from autowisp.database.interface import create_project_engine, get_project_url
from autowisp.database.sessions import start_engine_session

# The session key under which the newest run of the project is kept when a
# command is launched, which tells the progress page a run started since.
_launched_after_key = "launched_after_run"


def launch_detached(cmd, log_name, session):
    """
    Start a command in the background, its output going to a log.

    The command records what it does itself (a run, its errors), in the
    selected project, which is how the interface follows it; the log is for
    what goes wrong before it can, e.g. a broken install. The project's
    newest run is noted first (:func:`_note_newest_run`).

    Args:
        cmd([str]):    The command and its arguments.

        log_name(str):    The log's file name, in the user data directory.
            Overwritten by each launch with that name.

        session:    The request's session, whose selected project, if any,
            the command records its run in.

    Returns:
        None
    """

    _note_newest_run(session)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(
        os.path.join(platformdirs.user_data_dir("autowisp"), log_name),
        "w",
        encoding="utf-8",
        buffering=1,
    ) as outf:
        # Not a with block, which would wait for the command: it is
        # detached, and outlives the request.
        # pylint: disable-next=consider-using-with
        process = subprocess.Popen(
            cmd,
            start_new_session=(os.name == "posix"),
            stdout=outf,
            stderr=outf,
        )
    if os.name == "posix":
        # Reap the child when it exits so it does not become a zombie. A
        # command that double-forks (run_pipeline.py on Linux) orphans its
        # grandchild to init, which reaps it, but the child still needs to
        # be waited on here.
        threading.Thread(target=process.wait, daemon=True).start()


def _get_newest_run_id(db_session):
    """Return the id of the project's newest run, 0 if it has none."""

    # pylint: disable-next=not-callable
    return db_session.scalar(select(func.max(PipelineRun.id))) or 0


def _note_newest_run(session):
    """
    Remember the selected project's newest run, before launching a command.

    A command records its run only once it has started, and one that ends
    quickly may be over before the progress page next looks: unfinished runs
    alone cannot tell "not started yet" from "already finished". A run newer
    than this one can (see :func:`has_run_since_launch`).

    Read through an engine of its own: the request's may be another
    project's, e.g. the experiment's when an operation on it has just made
    its original the selected project. Nothing is noted with no project
    selected, e.g. from the project list, where nothing waits for a run.

    Args:
        session:    The request's session.
    """

    project_home = session.get("project_home")
    if project_home is None:
        session.pop(_launched_after_key, None)
        return
    engine = create_project_engine(get_project_url(project_home))
    try:
        with start_engine_session(engine) as db_session:
            session[_launched_after_key] = _get_newest_run_id(db_session)
    finally:
        engine.dispose()


def has_run_since_launch(session, db_session):
    """
    Has a run been recorded since the last command launched?

    Args:
        session:    The request's session.

        db_session:    A session on the database the command records its run
            in.

    Returns:
        bool:    Whether a run newer than the one noted at launch
        (:func:`note_newest_run`) exists, going or finished. False if no
        launch was noted.
    """

    launched_after = session.get(_launched_after_key)
    return (
        launched_after is not None
        and _get_newest_run_id(db_session) > launched_after
    )
