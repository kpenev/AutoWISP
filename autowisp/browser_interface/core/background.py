"""Start commands that outlive the request starting them."""

import os
import subprocess
import sys
import threading

import platformdirs
from sqlalchemy import func, select

# False positive
# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.database.interface import create_project_engine, get_project_url
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import Component

# The session key under which the last launch is kept (see launch_detached),
# for the progress page to tell what it recorded since.
_last_launch_key = "last_launch"


def launch_detached(cmd, log_name, session):
    """
    Start a command in the background, its output going to a log.

    The command records what it does itself (a run, its errors), in the
    selected project, which is how the interface follows it; the log is for
    what goes wrong before it can, e.g. a broken install.

    A command records its run only once it has started, and one that ends
    quickly may be over before the progress page next looks: unfinished runs
    alone cannot tell "not started yet" from "already finished". So the
    project's newest run and error are noted first, and what the command
    recorded since can (see :func:`find_since_launch`), a refusal before its
    run included, which records only an error. They are read through an
    engine of their own: the request's may be another project's, e.g. the
    experiment's when an operation on it has just made its original the
    selected project. Nothing is noted with no project selected, e.g. from
    the project list, where nothing waits for a run.

    Args:
        cmd([str]):    The command and its arguments.

        log_name(str):    The log's file name, in the user data directory.
            Overwritten by each launch with that name.

        session:    The request's session, whose selected project, if any,
            the command records in. The launch returned is kept in it too,
            as the last launch (:func:`get_last_launch`).

    Returns:
        dict or None:    The launch, for :func:`find_since_launch`: the ids of
        the project's newest ``pipeline_run`` and ``error`` before it, and
        the ``process`` started. Whether that process still runs says whether
        the command's work is over only for a command doing the work itself:
        the pipeline's hands it to another process on Linux. None with no
        project selected.
    """

    launch = None
    project_home = session.get("project_home")
    if project_home is not None:
        engine = create_project_engine(get_project_url(project_home))
        try:
            with start_engine_session(engine) as db_session:
                launch = {
                    table.__tablename__: db_session.scalar(
                        # pylint: disable-next=not-callable
                        select(func.max(table.id))
                    )
                    or 0
                    for table in (PipelineRun, Error)
                }
        finally:
            engine.dispose()

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

    if launch is not None:
        launch["process"] = process.pid
        session[_last_launch_key] = launch
    return launch


def get_last_launch(session):
    """Return the last launch from the session (see launch_detached)."""

    return session.get(_last_launch_key)


def find_since_launch(launch, db_session):
    """
    Return what a command launched has recorded since: runs and errors.

    Args:
        launch(dict or None):    The launch, as :func:`launch_detached`
            returned it.

        db_session:    A session on the database the command records in.

    Returns:
        [PipelineRun]:    The runs newer than those noted at launch, going or
        finished, oldest first.

        [Error]:    The errors newer than those noted at launch, but the
        interface's own, oldest first.

        Both are empty if the launch is None.
    """

    if launch is None:
        return [], []
    return (
        list(
            db_session.scalars(
                select(PipelineRun)
                .where(PipelineRun.id > launch["pipeline_run"])
                .order_by(PipelineRun.id)
            )
        ),
        list(
            db_session.scalars(
                select(Error)
                .where(Error.id > launch["error"])
                .where(Error.component != Component.BUI.value)
                .order_by(Error.id)
            )
        ),
    )
