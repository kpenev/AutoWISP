"""Tests of following a project being added by cloning, through its record."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from autowisp.database.pending_projects import (
    add_pending,
    get_pending_fname,
    list_pending,
    remove_pending,
)
from autowisp.database.pipeline_runs import get_process_started


class TestPendingProjects(unittest.TestCase):
    """A pending addition's state is read from its record and its process."""

    def setUp(self):
        # Released by addCleanup, since it must outlive setUp.
        # pylint: disable-next=consider-using-with
        pending_dir = tempfile.TemporaryDirectory()
        self.addCleanup(pending_dir.cleanup)
        self._pending_dir = pending_dir.name

    def _add_record(self, name, fields):
        """Record an addition, then add to it what the clone would write."""

        pending_id = add_pending(
            name, "", "/nowhere", None, pending_dir=self._pending_dir
        )
        fname = get_pending_fname(pending_id, self._pending_dir)
        with open(fname, encoding="utf-8") as record_file:
            record = json.load(record_file)
        record.update(fields)
        with open(fname, "w", encoding="utf-8") as record_file:
            json.dump(record, record_file)
        return pending_id

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

        remove_pending(ids["failed"], self._pending_dir)
        self.assertNotIn(
            "failed",
            [entry["name"] for entry in list_pending(self._pending_dir)],
        )
        with self.assertRaises(ValueError):
            remove_pending(os.path.join("..", "notes"), self._pending_dir)


if __name__ == "__main__":
    unittest.main()
