"""Tests for finding the logs of a step's run under a project's log names.

The BUI's log review and crash reports find a run's logs by expanding the
project's stored log names with the run's step, image type and process, and
globbing the rest. One pipeline process runs a step for each image type in
turn, e.g. stacking the zero frames and then the dark ones, so the names must
tell those apart, and where they do not, the lookup refuses rather than
guessing.
"""

import logging
import os
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime

from sqlalchemy import select

from autowisp import processing_steps
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.initialize_database import initialize_database
from autowisp.database.interface import (
    get_project_home,
    set_project_home,
    start_db_session,
)
from autowisp.exceptions import LogLookupError
from autowisp.multiprocessing_util import get_log_outerr_filenames

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    ImageProcessingProgress,
    ImageType,
    Parameter,
    PipelineRun,
    Step,
)

# pylint: enable=no-name-in-module

_log_parameters = ("logging-fname", "std-out-err-fname")


class TestLogLookup(unittest.TestCase):
    """A run's logs are found under the names its project stores."""

    #: The stackings one pipeline process runs, each by step and image type,
    #: and the time its logs are named by.
    _stackings = (
        ("stack_to_master", "zero", "20260101000100"),
        ("stack_to_master", "dark", "20260101000200"),
        ("stack_to_master_flat", "flat", "20260101000300"),
    )

    _process_id = 3126850

    #: A later process stacking the zero frames, whose logs are gone.
    _logless_process_id = 3126851

    @classmethod
    def setUpClass(cls):
        """Create a project under each of the new and the old log names.

        The old names are the ones a step run by hand still uses, which is
        what a project stored before it was given names of its own.
        """

        hand_run_defaults = processing_steps.stack_to_master.parse_command_line(
            []
        )["argument_defaults"]
        cls._homes = {}
        cls._progress_ids = {}
        cls._written = {}
        for names, overwrites in [
            ("new", None),
            (
                "old",
                {
                    parameter: [(None, hand_run_defaults[parameter])]
                    for parameter in _log_parameters
                },
            ),
        ]:
            # Released via addClassCleanup, since it must outlive setUpClass.
            # pylint: disable-next=consider-using-with
            home = tempfile.TemporaryDirectory()
            cls.addClassCleanup(home.cleanup)
            set_project_home(home.name)
            initialize_database(
                Namespace(
                    drop_hdf5_structure_tables=False, drop_all_tables=True
                ),
                overwrite_default_config=overwrites,
            )
            cls._homes[names] = home.name
            cls._progress_ids[names], cls._written[names] = cls._fill_project()

    @classmethod
    def _fill_project(cls):
        """Record the runs in the current project and write their logs.

        Returns:
            dict:    The progress id of each stacking, keyed by its step and
                image type, and under ``None`` that of the logless one.

            dict:    The log and output file written for each stacking.
        """

        progress_ids = {}
        written = {}
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            stored_names = dict(
                db_session.execute(
                    # pylint: disable=no-member
                    select(Parameter.name, Configuration.value)
                    .join(Configuration)
                    .where(Parameter.name.in_(_log_parameters))
                ).all()
            )
            runs = {
                process_id: PipelineRun(
                    host="test-host",
                    process_id=process_id,
                    started=datetime(2026, 1, 1),
                )
                for process_id in (cls._process_id, cls._logless_process_id)
            }
            db_session.add_all(runs.values())
            db_session.flush()
            for step_name, image_type, now in cls._stackings:
                progress_ids[step_name, image_type] = cls._add_progress(
                    db_session, runs[cls._process_id], step_name, image_type
                )
                written[step_name, image_type] = tuple(
                    cls._write_file(
                        stored_names[parameter].format_map(
                            {
                                "PROJHOME": get_project_home(),
                                "processing_step": step_name,
                                "image_type": image_type,
                                "task": "main",
                                "now": now,
                                "pid": cls._process_id,
                            }
                        )
                    )
                    for parameter in _log_parameters
                )
            progress_ids[None] = cls._add_progress(
                db_session,
                runs[cls._logless_process_id],
                "stack_to_master",
                "zero",
            )
        # pylint: enable=not-callable
        return progress_ids, written

    @staticmethod
    def _add_progress(db_session, run, step_name, image_type):
        """Record that ``run`` processed images of a type, returning its id."""

        # pylint: disable=not-callable
        progress = ImageProcessingProgress(
            run_id=run.id,
            step_id=db_session.scalar(
                select(Step.id).where(Step.name == step_name)
            ),
            image_type_id=db_session.scalar(
                select(ImageType.id).where(ImageType.name == image_type)
            ),
            configuration_version=0,
        )
        # pylint: enable=not-callable
        db_session.add(progress)
        db_session.flush()
        return progress.id

    @staticmethod
    def _write_file(fname):
        """Create the given file, returning its name."""

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding="utf-8") as log_file:
            log_file.write("INFO 2026-01-01 test: started\n")
        return fname

    def setUp(self):
        # A manager without a pipeline run switches logging off entirely.
        self.addCleanup(logging.disable, logging.root.manager.disable)

    def _find_logs(self, names, progress_key):
        """Return the main process's log and output found for a stacking."""

        set_project_home(self._homes[names])
        return tuple(
            os.path.realpath(fname)
            for fname in ImageProcessingManager(
                pipeline_run_id=None
            ).find_processing_outputs(self._progress_ids[names][progress_key])[
                0
            ]
        )

    def _assert_finds_own_logs(self, names, stacking):
        """Assert the lookup for a stacking finds exactly what it wrote."""

        self.assertEqual(
            self._find_logs(names, stacking),
            tuple(
                os.path.realpath(fname)
                for fname in self._written[names][stacking]
            ),
        )

    def test_new_names_tell_stackings_apart(self):
        """Each stacking finds its own logs; one without logs is refused."""

        for stacking in self._written["new"]:
            with self.subTest(stacking=stacking):
                self._assert_finds_own_logs("new", stacking)
        with self.assertRaisesRegex(LogLookupError, "No logs"):
            self._find_logs("new", None)

    def test_old_names_refuse_what_they_cannot_tell_apart(self):
        """Old names refuse the stackings sharing a name, find the rest."""

        for stacking in [
            ("stack_to_master", "zero"),
            ("stack_to_master", "dark"),
        ]:
            with self.subTest(stacking=stacking):
                with self.assertRaisesRegex(
                    LogLookupError, "cannot be told apart"
                ):
                    self._find_logs("old", stacking)
        self._assert_finds_own_logs("old", ("stack_to_master_flat", "flat"))
        with self.assertRaisesRegex(LogLookupError, "No logs"):
            self._find_logs("old", None)

    def test_hand_run_names_its_logs(self):
        """A step run by hand names its logs without an image type."""

        configuration = processing_steps.stack_to_master.parse_command_line([])
        configuration["project_home"] = self._homes["new"]
        # Raises if the names need anything a run by hand does not have.
        get_log_outerr_filenames(task="main", parent_pid="", **configuration)


if __name__ == "__main__":
    unittest.main()
