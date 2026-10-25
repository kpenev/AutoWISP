"""Start commands that outlive the request starting them."""

import os
import subprocess
import sys
import threading

import platformdirs


def launch_detached(cmd, log_name):
    """
    Start a command in the background, its output going to a log.

    The command records what it does itself (a run, its errors), which is
    how the interface follows it; the log is for what goes wrong before it
    can, e.g. a broken install.

    Args:
        cmd([str]):    The command and its arguments.

        log_name(str):    The log's file name, in the user data directory.
            Overwritten by each launch with that name.

    Returns:
        None
    """

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
