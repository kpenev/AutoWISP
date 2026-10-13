"""Tests of telling which pipeline runs are still going, and where."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import psutil
from sqlalchemy import select, sql

from autowisp.database import pipeline_runs
from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import PipelineRun

# pylint: enable=no-name-in-module
from autowisp.exceptions import get_hostname

# The identifier the machine running the tests is given, rather than its
# own, which it may not have.
_THIS_MACHINE = "this-machine"  # pylint: disable=invalid-name


def _start_process(*, wait):
    """Start a Python process that sleeps, or one that has already ended."""

    process = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            "-c",
            f"import time; time.sleep({0 if wait else 600})",
        ]
    )
    if wait:
        process.wait()
    return process


class TestFindActiveRuns(unittest.TestCase):
    """Runs are told apart by machine and process, not by host name."""

    def setUp(self):
        """Create a project, and the processes its runs refer to."""

        # Released via addCleanup, since it must outlive setUp.
        # pylint: disable-next=consider-using-with
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        set_project_home(tmp.name)

        self._live = _start_process(wait=False)
        self.addCleanup(self._live.wait)
        self.addCleanup(self._live.kill)
        self._live_started = psutil.Process(self._live.pid).create_time()

        dead = _start_process(wait=True)
        self._dead_pid = dead.pid

        patcher = mock.patch.object(
            pipeline_runs, "get_machine_id", return_value=_THIS_MACHINE
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _start_zombie(self):
        """Return the id of a process that ended and was not collected."""

        zombie = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, "-c", "pass"]
        )
        self.addCleanup(zombie.wait)
        deadline = time.monotonic() + 30
        while psutil.Process(zombie.pid).status() != psutil.STATUS_ZOMBIE:
            self.assertLess(time.monotonic(), deadline, "No zombie appeared.")
            time.sleep(0.05)
        return zombie.pid

    def _add_runs(self, runs):
        """Add unfinished runs, ``{name: column values}``; return their ids."""

        ids = {}
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            for name, columns in runs.items():
                run = PipelineRun(
                    **{
                        "host": get_hostname(),
                        "started": sql.func.now(),
                        **columns,
                    }
                )
                db_session.add(run)
                db_session.flush()
                ids[name] = run.id
        # pylint: enable=not-callable
        return ids

    def _find(self):
        """Return the ids of the runs found going here, and elsewhere."""

        with start_db_session() as db_session:
            here, elsewhere = pipeline_runs.find_active_runs(db_session)
            return {run.id for run in here}, {run.id for run in elsewhere}

    def _get_finished(self):
        """Return the ids of the runs recorded as finished."""

        with start_db_session() as db_session:
            return set(
                db_session.scalars(
                    select(PipelineRun.id).where(
                        PipelineRun.finished.is_not(None)
                    )
                )
            )

    def test_identifies_runs_by_machine_and_process(self):
        """Each kind of unfinished run, side by side."""

        this_machine = {"machine_id": _THIS_MACHINE}
        live = {
            "process_id": self._live.pid,
            "process_started": self._live_started,
        }
        runs = {
            "live": {**this_machine, **live},
            # SUP-491: the machine renamed itself while the run went on.
            "renamed": {**this_machine, **live, "host": "old-name"},
            "reused_id": {
                **this_machine,
                **live,
                "process_started": self._live_started - 3600,
            },
            "dead": {
                **this_machine,
                "process_id": self._dead_pid,
                "process_started": self._live_started,
            },
            "own_process": {
                **this_machine,
                "process_id": os.getpid(),
                "process_started": pipeline_runs.get_process_started(),
            },
            "other_machine": {"machine_id": "other-machine", **live},
            "legacy": {"process_id": self._live.pid},
            "legacy_other_host": {
                "process_id": self._dead_pid,
                "host": "elsewhere",
            },
        }
        if os.name == "posix":
            runs["zombie"] = {
                **this_machine,
                "process_id": self._start_zombie(),
                "process_started": None,
            }
        ids = self._add_runs(runs)

        here, elsewhere = self._find()

        self.assertEqual(here, {ids["live"], ids["renamed"], ids["legacy"]})
        self.assertEqual(
            elsewhere, {ids["other_machine"], ids["legacy_other_host"]}
        )
        self.assertEqual(
            self._get_finished(),
            {
                ids[name]
                for name in ("reused_id", "dead", "own_process", "zombie")
                if name in ids
            },
        )

    def test_compares_host_names_without_a_machine_id(self):
        """A machine with no identifier to find falls back to its name."""

        live = {
            "machine_id": _THIS_MACHINE,
            "process_id": self._live.pid,
            "process_started": self._live_started,
        }
        ids = self._add_runs(
            {"same_name": live, "other_name": {**live, "host": "elsewhere"}}
        )
        with mock.patch.object(
            pipeline_runs, "get_machine_id", return_value=None
        ):
            here, elsewhere = self._find()

        self.assertEqual(here, {ids["same_name"]})
        self.assertEqual(elsewhere, {ids["other_name"]})
        self.assertEqual(self._get_finished(), set())


if __name__ == "__main__":
    unittest.main()
