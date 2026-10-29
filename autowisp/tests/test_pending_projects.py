"""Tests of following a project being added by cloning, through its record."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from autowisp.database.existing_project import clone_fname
from autowisp.database.pending_projects import (
    add_pending,
    dismiss_pending,
    get_log_fname,
    get_pending_fname,
    list_pending,
    remove_pending,
)
from autowisp.database.pipeline_runs import get_process_started
from autowisp.exceptions import ProjectStateError


class TestPendingProjects(unittest.TestCase):
    """A pending addition's state is read from its record and its process."""

    def setUp(self):
        # Released by addCleanup, since it must outlive setUp.
        # pylint: disable-next=consider-using-with
        pending_dir = tempfile.TemporaryDirectory()
        self.addCleanup(pending_dir.cleanup)
        self._pending_dir = pending_dir.name

    def _add_record(self, name, fields, home="/nowhere", target_url=None):
        """Record an addition, then add to it what the clone would write."""

        pending_id = add_pending(
            name, "", home, target_url, pending_dir=self._pending_dir
        )
        fname = get_pending_fname(pending_id, self._pending_dir)
        with open(fname, encoding="utf-8") as record_file:
            record = json.load(record_file)
        record.update(fields)
        with open(fname, "w", encoding="utf-8") as record_file:
            json.dump(record, record_file)
        return pending_id

    def _make_home(self, *fnames):
        """Return a new directory holding empty files of the given names."""

        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        for fname in fnames:
            with open(os.path.join(home, fname), "wb"):
                pass
        return home

    @staticmethod
    def _get_ended_process():
        """Return the id and start time of a process that has ended."""

        with subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
        ) as process:
            started = get_process_started(process.pid)
            process.communicate()
        return process.pid, started

    def test_states(self):
        """Each record is judged by its outcome, its process, or its age."""

        ended_id, ended_started = self._get_ended_process()
        # By name: the fields the clone would have written, and the state
        # each must be found in.
        records = {
            "starting": ({}, "cloning"),
            "never started": (
                {"launched": time.time() - 120},
                "not started",
            ),
            "cloning": (
                {
                    "process_id": os.getpid(),
                    "process_started": get_process_started(),
                },
                "cloning",
            ),
            "process id reused": (
                {
                    "process_id": os.getpid(),
                    "process_started": get_process_started() - 3600,
                },
                "interrupted",
            ),
            "process ended": (
                {"process_id": ended_id, "process_started": ended_started},
                "interrupted",
            ),
            "done": (
                {
                    "process_id": ended_id,
                    "process_started": ended_started,
                    "outcome": "done",
                    "stale": ["exp1"],
                },
                "done",
            ),
            "failed": (
                {
                    "process_id": ended_id,
                    "process_started": ended_started,
                    "outcome": "failed",
                    "message": "refused",
                },
                "failed",
            ),
        }
        ids = {
            name: self._add_record(name, fields)
            for name, (fields, _) in records.items()
        }
        # Not records add_pending wrote: ignored.
        for stray in ("notes.json", ids["done"] + ".json.new"):
            with open(
                os.path.join(self._pending_dir, stray), "w", encoding="utf-8"
            ) as stray_file:
                stray_file.write("not a record")

        pending = list_pending(self._pending_dir)
        self.assertEqual(
            {entry["name"]: entry["state"] for entry in pending},
            {name: state for name, (_, state) in records.items()},
        )
        self.assertEqual({entry["name"]: entry["id"] for entry in pending}, ids)
        self.assertEqual(
            [entry["name"] for entry in pending][0], "never started"
        )

        log_fname = get_log_fname(ids["failed"], self._pending_dir)
        with open(log_fname, "w", encoding="utf-8") as log_file:
            log_file.write("the clone's log")
        remove_pending(ids["failed"], self._pending_dir)
        self.assertNotIn(
            "failed",
            [entry["name"] for entry in list_pending(self._pending_dir)],
        )
        self.assertFalse(os.path.exists(log_fname))
        with self.assertRaises(ValueError):
            remove_pending(os.path.join("..", "notes"), self._pending_dir)

    def test_dismiss(self):
        """Only an unfinished addition is dismissed, with its SQLite clone."""

        ended_id, ended_started = self._get_ended_process()
        ended = {"process_id": ended_id, "process_started": ended_started}
        # By name: the target, the fields the clone would have written, and
        # whether dismissing is refused.
        additions = {
            "failed": (
                None,
                dict(ended, outcome="failed", message="refused"),
                False,
            ),
            "interrupted": (None, ended, False),
            "never started": (None, {"launched": time.time() - 120}, False),
            # The SQLite clone in its home is not this clone's.
            "interrupted into a server": ("mysql://nowhere/x", ended, False),
            "cloning": (
                None,
                {
                    "process_id": os.getpid(),
                    "process_started": get_process_started(),
                },
                True,
            ),
            "done": (None, dict(ended, outcome="done", stale=[]), True),
        }
        homes = {}
        logs = {}
        for name, (target_url, fields, refused) in additions.items():
            homes[name] = self._make_home("kept", clone_fname)
            pending_id = self._add_record(name, fields, homes[name], target_url)
            logs[name] = get_log_fname(pending_id, self._pending_dir)
            with open(logs[name], "w", encoding="utf-8") as log_file:
                log_file.write("the clone's log")
            if refused:
                with self.assertRaises(ProjectStateError):
                    dismiss_pending(pending_id, self._pending_dir)
            else:
                dismiss_pending(pending_id, self._pending_dir)

        self.assertEqual(
            {entry["name"] for entry in list_pending(self._pending_dir)},
            {"cloning", "done"},
        )
        for name, (target_url, _, refused) in additions.items():
            with self.subTest(name):
                self.assertEqual(
                    set(os.listdir(homes[name])),
                    (
                        {"kept", clone_fname}
                        if refused or target_url is not None
                        else {"kept"}
                    ),
                )
                self.assertEqual(os.path.exists(logs[name]), refused)


if __name__ == "__main__":
    unittest.main()
