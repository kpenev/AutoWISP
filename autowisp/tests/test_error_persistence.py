"""Unit tests for error persistence (row + JSON sidecar).

These need a real (throwaway) project database, created once per class in
a temporary directory.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import select, sql, update

from autowisp import run_pipeline
from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error, Image, MasterFile, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.exceptions import FileKind, FindStarsError, RelatedFile
from autowisp.error_context import _worker_crashed, error_context
from autowisp.error_persistence import (
    cleanup_errors,
    delete_all_error_sidecars,
    delete_error,
    load_sidecar,
    parse_duration,
    persist_error,
)
from autowisp.tests.error_fixtures import make_find_stars_error


class _PersistenceTestCase(unittest.TestCase):
    """Base creating one throwaway project database for the class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _get_error(self, error_id):
        with start_db_session() as db_session:
            return db_session.get(Error, error_id)


class TestPersistError(_PersistenceTestCase):
    """persist_error writes the queryable row and the detail sidecar."""

    def test_row_and_sidecar_are_complementary(self):
        """Inline fields on the row; the heavy remainder in the sidecar."""

        exc = make_find_stars_error()
        error_id = persist_error(exc)
        self.assertIsNotNone(error_id)

        row = self._get_error(error_id)
        # Inline, queryable fields:
        self.assertEqual(row.component, "step")
        self.assertEqual(row.step_name, "find_stars")
        self.assertEqual(row.exception_class, "FindStarsError")
        self.assertEqual(row.user_message, "no stars found")
        self.assertIsNotNone(row.sidecar_path)

        sidecar = load_sidecar(row)
        # Heavy remainder lives in the sidecar...
        self.assertEqual(sidecar["message"], "no stars found")
        self.assertEqual(sidecar["details"], {"brightness_quantile": 0.999})
        self.assertEqual(len(sidecar["related_files"]), 2)
        self.assertIn("traceback", sidecar)
        # ... including the crash-time environment (captured here, in the
        # process that recorded the error -- immune to later upgrades).
        self.assertIn("environment", sidecar)
        self.assertIn("numpy", sidecar["environment"]["packages"])
        self.assertIn("platform", sidecar["environment"])
        # ... and the inline fields are NOT duplicated into it.
        for inline in (
            "component",
            "step_name",
            "user_message",
            "subprocess_id",
        ):
            self.assertNotIn(inline, sidecar)

    def test_worker_crashed_persists_step_name(self):
        """A synthesised worker-death error lands its step on the row.

        End-to-end regression for the crash-report "no matching logs
        found" gap: a ``WorkerCrashedError`` is synthesised by the parent
        (so it is not a per-stage ``StepError`` subclass), but it must
        still persist ``step_name`` into the queryable column -- that is
        what crash-report log-collection resolves the run/step logs from.
        It is a ``step``-component error because the failure is in the
        algorithm running inside the step.
        """

        with error_context(step_name="tfa"):
            exc = _worker_crashed(
                ["/lc/a.h5", "/lc/b.h5"],
                RuntimeError("A process in the process pool was terminated"),
            )

        row = self._get_error(persist_error(exc))
        self.assertEqual(row.step_name, "tfa")
        self.assertEqual(row.component, "step")
        self.assertEqual(row.exception_class, "WorkerCrashedError")

    def test_artifact_fks_resolved_from_related_files(self):
        """A related file matching a known image/master sets the FK."""

        # pylint: disable=not-callable
        with start_db_session() as db_session:
            run = PipelineRun(
                host="h",
                process_id=1,
                started=sql.func.now(),
            )
            image = Image(
                raw_fname="/data/raw/img042.fits",
                image_type_id=1,
                observing_session_id=1,
            )
            master = MasterFile(type_id=1, filename="/data/masters/mbias.fits")
            # pylint: enable=not-callable
            db_session.add_all([run, image, master])
            db_session.flush()
            image_id, master_id = image.id, master.id

        exc = make_find_stars_error(
            related_files=[
                RelatedFile(
                    FileKind.RAW_IMAGE, Path("/data/raw/img042.fits"), "input"
                ),
                RelatedFile(
                    FileKind.MASTER_BIAS,
                    Path("/data/masters/mbias.fits"),
                    "input",
                ),
            ]
        )
        row = self._get_error(persist_error(exc))
        self.assertEqual(row.image_id, image_id)
        self.assertEqual(row.master_file_id, master_id)

    def test_large_payload_is_gzipped(self):
        """A payload over the threshold is written as .json.gz and reloads."""

        exc = make_find_stars_error(details={"dump": "x" * 200000})
        row = self._get_error(persist_error(exc))

        self.assertTrue(row.sidecar_path.endswith(".json.gz"))
        sidecar = load_sidecar(row)
        self.assertEqual(sidecar["details"]["dump"], "x" * 200000)

    def test_sidecar_failure_leaves_valid_row(self):
        """A sidecar-write failure still leaves a row, sidecar_path NULL."""

        exc = make_find_stars_error()
        with mock.patch(
            "autowisp.error_persistence._write_sidecar",
            side_effect=OSError("disk full"),
        ):
            # The failure is logged (and swallowed), not raised.
            with self.assertLogs("autowisp.error_persistence", "ERROR"):
                error_id = persist_error(exc)

        self.assertIsNotNone(error_id)
        row = self._get_error(error_id)
        self.assertEqual(row.exception_class, "FindStarsError")
        self.assertIsNone(row.sidecar_path)

    def test_missing_sidecar_degrades_to_none(self):
        """Reading an error whose sidecar is gone returns None, no crash."""

        row = self._get_error(persist_error(make_find_stars_error()))
        # Remove the sidecar behind the row.
        os.remove(os.path.join(self._tmp.name, row.sidecar_path))
        self.assertIsNone(load_sidecar(row))

    def test_delete_removes_row_and_sidecar(self):
        """delete_error removes both the row and its sidecar file."""

        error_id = persist_error(make_find_stars_error())
        row = self._get_error(error_id)
        sidecar = os.path.join(self._tmp.name, row.sidecar_path)
        self.assertTrue(os.path.exists(sidecar))

        self.assertTrue(delete_error(error_id))

        self.assertIsNone(self._get_error(error_id))
        self.assertFalse(os.path.exists(sidecar))

    def test_delete_missing_error_is_noop(self):
        """Deleting a non-existent error returns False, does not raise."""

        self.assertFalse(delete_error(999999))


class TestRunPipelineHandler(_PersistenceTestCase):
    """run_pipeline.main records an escaping error and re-raises it."""

    def test_handler_persists_and_reraises(self):
        """An AutoWISPError out of the run is persisted, then re-raised."""

        config = SimpleNamespace(project_home=self._tmp.name)
        with mock.patch.object(
            run_pipeline,
            "_run_pipeline",
            side_effect=make_find_stars_error(),
        ):
            with self.assertRaises(FindStarsError):
                run_pipeline.main(config)

        with start_db_session() as db_session:
            rows = db_session.scalars(
                select(Error).where(
                    Error.exception_class  # pylint: disable=no-member
                    == "FindStarsError"
                )
            ).all()
        self.assertTrue(
            any(row.step_name == "find_stars" for row in rows),
            "the escaping error was not persisted",
        )

    def test_handler_records_plain_exception(self):
        """A non-AutoWISP exception is recorded; the original re-raises.

        Mirrors a real failure: a bad config expression raises a plain
        NameError in the orchestration layer, outside any step's capture
        boundary. The handler records it (wrapped as a PipelineError) but
        re-raises the original so its true traceback reaches the log.
        """

        config = SimpleNamespace(project_home=self._tmp.name)
        with mock.patch.object(
            run_pipeline,
            "_run_pipeline",
            side_effect=NameError("name 'OBS_SESN' is not defined"),
        ):
            with self.assertRaises(NameError):
                run_pipeline.main(config)

        with start_db_session() as db_session:
            rows = db_session.scalars(
                select(Error).where(
                    Error.component == "pipeline"  # pylint: disable=no-member
                )
            ).all()
        recorded = [
            row for row in rows if "OBS_SESN" in (row.user_message or "")
        ]
        self.assertTrue(
            recorded, "the plain orchestration exception was not persisted"
        )
        self.assertEqual(recorded[0].exception_class, "PipelineError")


class TestCleanupErrors(unittest.TestCase):
    """wisp-cleanup-errors prunes aged rows, orphan files, dangling rows.

    Uses a fresh project per test, since cleanup mutates global DB state.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        set_project_home(self._tmp.name)

    def _abs(self, relative):
        return os.path.join(self._tmp.name, relative)

    def _age_error(self, error_id, *, days):
        """Backdate an error's ``created`` so it counts as old."""

        old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=days
        )
        with start_db_session() as db_session:
            db_session.execute(
                update(Error)
                .where(Error.id == error_id)  # pylint: disable=no-member
                .values(created=old)
            )

    def test_aged_rows_and_sidecars_removed(self):
        """An old error and its sidecar are deleted; a recent one is kept."""

        old_id = persist_error(make_find_stars_error())
        recent_id = persist_error(make_find_stars_error())
        self._age_error(old_id, days=40)

        with start_db_session() as db_session:
            old_sidecar = self._abs(db_session.get(Error, old_id).sidecar_path)
        self.assertTrue(os.path.exists(old_sidecar))

        summary = cleanup_errors(older_than=parse_duration("30d"))

        self.assertEqual(summary["removed_rows"], 1)
        self.assertFalse(os.path.exists(old_sidecar))
        with start_db_session() as db_session:
            self.assertIsNone(db_session.get(Error, old_id))
            self.assertIsNotNone(db_session.get(Error, recent_id))

    def test_orphan_files_removed(self):
        """A stray file with no owning row is swept away."""

        persist_error(make_find_stars_error())
        orphan = self._abs(os.path.join("errors", "cli", "9999.json"))
        with open(orphan, "w", encoding="utf-8") as orphan_file:
            orphan_file.write("{}")
        # A leftover temp file from a crashed write is junk too.
        tmp_leftover = self._abs(os.path.join("errors", "cli", "5.json.tmp"))
        with open(tmp_leftover, "w", encoding="utf-8") as tmp_file:
            tmp_file.write("{}")

        summary = cleanup_errors()

        self.assertEqual(summary["removed_files"], 2)
        self.assertFalse(os.path.exists(orphan))
        self.assertFalse(os.path.exists(tmp_leftover))

    def test_dangling_row_reference_cleared(self):
        """A row whose sidecar vanished keeps the row but clears the path."""

        error_id = persist_error(make_find_stars_error())
        with start_db_session() as db_session:
            os.remove(self._abs(db_session.get(Error, error_id).sidecar_path))

        summary = cleanup_errors()

        self.assertEqual(summary["cleared_dangling"], 1)
        with start_db_session() as db_session:
            row = db_session.get(Error, error_id)
            self.assertIsNotNone(row)
            self.assertIsNone(row.sidecar_path)

    def test_sweep_keeps_valid_sidecar(self):
        """A healthy error's sidecar is not touched by a plain sweep."""

        error_id = persist_error(make_find_stars_error())
        with start_db_session() as db_session:
            sidecar = self._abs(db_session.get(Error, error_id).sidecar_path)

        summary = cleanup_errors()

        self.assertEqual(
            summary,
            {
                "removed_rows": 0,
                "removed_files": 0,
                "cleared_dangling": 0,
            },
        )
        self.assertTrue(os.path.exists(sidecar))

    def test_delete_all_sidecars_keeps_foreign_files(self):
        """Project-deletion sidecar purge removes tracked files only.

        Every recorded error's sidecar is removed, but an unrelated file
        placed under the errors directory survives.
        """

        ids = [
            persist_error(make_find_stars_error()),
            persist_error(make_find_stars_error()),
        ]
        with start_db_session() as db_session:
            sidecars = [
                self._abs(db_session.get(Error, eid).sidecar_path)
                for eid in ids
            ]
        # Drop an unrelated file into the errors tree.
        foreign = self._abs(os.path.join("errors", "keep_me.txt"))
        with open(foreign, "w", encoding="utf-8") as foreign_file:
            foreign_file.write("not an error sidecar")

        delete_all_error_sidecars()

        for sidecar in sidecars:
            self.assertFalse(os.path.exists(sidecar))
        self.assertTrue(os.path.exists(foreign))


class TestParseDuration(unittest.TestCase):
    """parse_duration accepts compact durations and rejects junk."""

    def test_units(self):
        self.assertEqual(parse_duration("45s"), timedelta(seconds=45))
        self.assertEqual(parse_duration("12h"), timedelta(hours=12))
        self.assertEqual(parse_duration("30d"), timedelta(days=30))
        self.assertEqual(parse_duration("2w"), timedelta(weeks=2))

    def test_invalid_raises(self):
        for bad in ("", "30", "5y", "abc", "-3d"):
            with self.assertRaises(ValueError):
                parse_duration(bad)


if __name__ == "__main__":
    unittest.main()
