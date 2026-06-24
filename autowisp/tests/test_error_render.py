"""Unit tests for error rendering (the shared formatter).

Renders persisted ``Error`` rows (written via ``persist_error``) through
``error_render``; uses a throwaway project database.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error
from autowisp.database.data_model import Image

# pylint: enable=no-name-in-module
from autowisp.exceptions import FileKind, RelatedFile
from autowisp.error_persistence import persist_error
from autowisp.error_render import (
    error_count,
    error_counts_by_step,
    error_detail,
    error_list_rows,
    error_summary,
    format_detail_text,
    open_error_count_for_steps,
)
from autowisp.exceptions import PipelineError, StackToMasterError, ViewError
from autowisp.tests.error_fixtures import make_find_stars_error


class _RenderTestCase(unittest.TestCase):
    """Base creating one throwaway project database for the class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _persist_and_get(self, exc):
        error_id = persist_error(exc)
        with start_db_session() as db_session:
            return db_session.get(Error, error_id)


class TestErrorSummary(_RenderTestCase):
    """error_summary: one-line component/step/artifact/message."""

    def test_summary_without_artifact(self):
        """No linked artifact -> component:step + message only."""

        row = self._persist_and_get(make_find_stars_error())
        summary = error_summary(row)
        self.assertIn("[step:find_stars]", summary)
        self.assertIn("no stars found", summary)

    def test_summary_with_artifact(self):
        """A resolved image FK appears in the summary."""

        # pylint: disable=not-callable
        with start_db_session() as db_session:
            image = Image(
                raw_fname="/data/raw/img042.fits",
                image_type_id=1,
                observing_session_id=1,
            )
            db_session.add(image)
            db_session.flush()
        # pylint: enable=not-callable

        exc = make_find_stars_error(
            related_files=[
                RelatedFile(
                    FileKind.RAW_IMAGE,
                    "/data/raw/img042.fits",
                    "input",
                )
            ]
        )
        summary = error_summary(self._persist_and_get(exc))
        self.assertIn("/data/raw/img042.fits", summary)


class TestErrorDetail(_RenderTestCase):
    """error_detail: user vs developer views, remediation, degradation."""

    def test_user_view_omits_technical_fields(self):
        """developer=False shows user fields, hides traceback/details."""

        row = self._persist_and_get(make_find_stars_error())
        detail = error_detail(row, developer=False)

        self.assertEqual(detail["user_message"], "no stars found")
        self.assertEqual(detail["component"], "step")
        self.assertNotIn("traceback", detail)
        self.assertNotIn("details", detail)
        self.assertNotIn("exception_class", detail)

    def test_developer_view_adds_sidecar_backed_fields(self):
        """developer=True adds message/traceback/details and provenance."""

        row = self._persist_and_get(make_find_stars_error())
        detail = error_detail(row, developer=True)

        self.assertEqual(detail["exception_class"], "FindStarsError")
        self.assertIn("traceback", detail)
        self.assertEqual(detail["details"], {"brightness_quantile": 0.999})
        self.assertEqual(len(detail["related_files"]), 2)
        self.assertTrue(detail["sidecar_available"])

    def test_remediation_only_when_present(self):
        """Remediation appears iff the error provided one in details."""

        without = error_detail(
            self._persist_and_get(make_find_stars_error()), developer=False
        )
        self.assertNotIn("remediation", without)

        exc = make_find_stars_error(
            details={"remediation": "Lower the brightness threshold."}
        )
        with_rem = error_detail(self._persist_and_get(exc), developer=False)
        self.assertEqual(
            with_rem["remediation"], "Lower the brightness threshold."
        )

    def test_missing_sidecar_degrades(self):
        """With no sidecar, developer fields are present but empty/flagged."""

        error_id = persist_error(make_find_stars_error())
        with start_db_session() as db_session:
            row = db_session.get(Error, error_id)
            # Drop the sidecar reference so load_sidecar returns None.
            row.sidecar_path = None
            db_session.flush()
            detail = error_detail(row, db_session, developer=True)

        self.assertFalse(detail["sidecar_available"])
        self.assertIsNone(detail["traceback"])
        self.assertEqual(detail["details"], {})
        # Falls back to the inline user_message for the technical message.
        self.assertEqual(detail["message"], "no stars found")


class TestFormatDetailText(_RenderTestCase):
    """format_detail_text: terminal rendering of the detail dict."""

    def test_user_text_is_summary_plus_remediation(self):
        """The user view text shows the summary and (any) remediation."""

        exc = make_find_stars_error(details={"remediation": "Do X."})
        detail = error_detail(self._persist_and_get(exc), developer=False)
        text = format_detail_text(detail)

        self.assertIn("no stars found", text)
        self.assertIn("Remediation: Do X.", text)
        self.assertNotIn("Traceback:", text)

    def test_developer_text_includes_traceback(self):
        """The developer view text includes the traceback block."""

        row = self._persist_and_get(make_find_stars_error())
        text = format_detail_text(error_detail(row, developer=True))

        self.assertIn("Exception: FindStarsError", text)
        self.assertIn("Traceback:", text)


class TestErrorListRows(_RenderTestCase):
    """error_list_rows: the cheap, inline-only list-view projection."""

    def test_rows_newest_first_with_summary_fields(self):
        """Returns one dict per error, newest first, with summary fields."""

        first_id = persist_error(make_find_stars_error())
        second_id = persist_error(make_find_stars_error())

        rows = error_list_rows()

        ids = [row["id"] for row in rows]
        # Both present; the later-created one comes first.
        self.assertIn(first_id, ids)
        self.assertIn(second_id, ids)
        self.assertLess(ids.index(second_id), ids.index(first_id))

        sample = rows[0]
        self.assertEqual(sample["component"], "step")
        self.assertEqual(sample["step_name"], "find_stars")
        self.assertIn("no stars found", sample["summary"])

    def test_does_not_read_sidecars(self):
        """The list projection never opens a sidecar file."""

        persist_error(make_find_stars_error())
        with mock.patch(
            "autowisp.error_render.load_sidecar"
        ) as load_sidecar_mock:
            error_list_rows()
        load_sidecar_mock.assert_not_called()


class TestErrorCounts(unittest.TestCase):
    """error_count / error_counts_by_step / the step filter.

    Fresh project per test, since these assert exact counts.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        set_project_home(self._tmp.name)

    def test_total_count(self):
        """error_count totals every recorded error, any component."""

        persist_error(make_find_stars_error())
        persist_error(make_find_stars_error())
        persist_error(PipelineError("orchestration broke"))
        self.assertEqual(error_count(), 3)

    def test_counts_by_step_excludes_stepless(self):
        """Per-step counts group step errors and skip pipeline/BUI ones."""

        persist_error(make_find_stars_error())
        persist_error(make_find_stars_error())
        persist_error(
            StackToMasterError("cannot stack", step_name="stack_to_master")
        )
        persist_error(PipelineError("no step here"))

        by_step = error_counts_by_step()
        self.assertEqual(by_step, {"find_stars": 2, "stack_to_master": 1})

    def test_step_filter_on_list_rows(self):
        """error_list_rows(step_name=...) restricts to that step."""

        persist_error(make_find_stars_error())
        persist_error(
            StackToMasterError("cannot stack", step_name="stack_to_master")
        )

        rows = error_list_rows(step_name="find_stars")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step_name"], "find_stars")

    def _resolve(self, error_id):
        with start_db_session() as db_session:
            db_session.get(Error, error_id).resolved = datetime.now(
                timezone.utc
            )

    def test_counts_exclude_resolved(self):
        """Resolving an error drops it from the badge and step markers."""

        open_id = persist_error(make_find_stars_error())
        resolved_id = persist_error(make_find_stars_error())
        self._resolve(resolved_id)

        self.assertEqual(error_count(), 1)
        self.assertEqual(error_counts_by_step(), {"find_stars": 1})
        self.assertIsNotNone(open_id)

    def test_list_shows_resolved_open_first(self):
        """The list keeps resolved errors (annotated), open ones first."""

        resolved_id = persist_error(make_find_stars_error())
        self._resolve(resolved_id)
        open_id = persist_error(make_find_stars_error())

        rows = error_list_rows()
        self.assertEqual(len(rows), 2)
        # Open first, resolved (dimmed) after.
        self.assertEqual(rows[0]["id"], open_id)
        self.assertIsNone(rows[0]["resolved"])
        self.assertEqual(rows[1]["id"], resolved_id)
        self.assertIsNotNone(rows[1]["resolved"])

    def test_gate_scopes_step_errors_to_selected_steps(self):
        """Step errors gate only their own step."""

        persist_error(make_find_stars_error())
        persist_error(make_find_stars_error())
        persist_error(
            StackToMasterError("cannot stack", step_name="stack_to_master")
        )

        self.assertEqual(open_error_count_for_steps(["find_stars"]), 2)
        self.assertEqual(
            open_error_count_for_steps(["find_stars", "stack_to_master"]), 3
        )
        self.assertEqual(open_error_count_for_steps(["calibrate"]), 0)

    def test_gate_always_includes_pipeline_errors(self):
        """A pipeline error gates any launch, even unrelated steps."""

        persist_error(PipelineError("orchestration broke"))

        # The stepless pipeline error gates regardless of the selection.
        self.assertEqual(open_error_count_for_steps(["calibrate"]), 1)
        self.assertEqual(open_error_count_for_steps(["find_stars"]), 1)
        self.assertEqual(open_error_count_for_steps([]), 1)

    def test_gate_excludes_bui_errors(self):
        """BUI errors do not gate processing."""

        persist_error(ViewError("a form blew up"))
        self.assertEqual(open_error_count_for_steps([]), 0)
        self.assertEqual(open_error_count_for_steps(["calibrate"]), 0)

    def test_gate_excludes_resolved(self):
        """Resolving an error removes it from the gate."""

        pipeline_id = persist_error(PipelineError("orchestration broke"))
        self.assertEqual(open_error_count_for_steps([]), 1)
        self._resolve(pipeline_id)
        self.assertEqual(open_error_count_for_steps([]), 0)


if __name__ == "__main__":
    unittest.main()
