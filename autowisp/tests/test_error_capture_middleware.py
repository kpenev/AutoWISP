"""Unit tests for the BUI error-capture middleware.

The middleware is exercised directly with a mock request (no running
Django server), against a throwaway project database.
"""

import tempfile
import unittest
from types import SimpleNamespace

from sqlalchemy import delete, select

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error

# pylint: enable=no-name-in-module
from autowisp.exceptions import ViewError
from autowisp.error_persistence import load_sidecar
from autowisp.browser_interface.error_capture_middleware import (
    ErrorCaptureMiddleware,
)


def _mock_request():
    """A stand-in Django request exposing the attributes we snapshot."""

    return SimpleNamespace(
        path="/processing/progress/",
        method="POST",
        resolver_match=SimpleNamespace(view_name="processing:progress"),
        GET={},
        # Values present but should NOT be captured -- only keys.
        POST={"password": "hunter2", "steps": "calibrate"},
        session=SimpleNamespace(session_key="abc123"),
    )


class TestErrorCaptureMiddleware(unittest.TestCase):
    """process_exception records the error with a keys-only request snapshot."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        # get_response is unused by process_exception; a no-op suffices.
        self.middleware = ErrorCaptureMiddleware(lambda request: None)
        # Isolate each test: start from an empty error table.
        with start_db_session() as db_session:
            db_session.execute(delete(Error))

    def _only_error(self):
        with start_db_session() as db_session:
            rows = db_session.scalars(select(Error)).all()
            self.assertEqual(len(rows), 1)
            return rows[0], load_sidecar(rows[0])

    def test_unknown_exception_wrapped_and_recorded(self):
        """A bare view exception is recorded as a BUI ViewError."""

        # Returns None so Django still renders its 500.
        self.assertIsNone(
            self.middleware.process_exception(
                _mock_request(), KeyError("missing")
            )
        )

        row, _sidecar = self._only_error()
        self.assertEqual(row.component, "bui")
        self.assertEqual(row.exception_class, "ViewError")
        self.assertIsNotNone(row.created)

    def test_request_snapshot_keys_only(self):
        """The snapshot captures parameter keys, never their values."""

        self.middleware.process_exception(
            _mock_request(), ViewError("bad form")
        )
        _row, sidecar = self._only_error()

        snapshot = sidecar["details"]["bui_request"]
        self.assertEqual(snapshot["path"], "/processing/progress/")
        self.assertEqual(snapshot["view_name"], "processing:progress")
        self.assertIn("password", snapshot["post_keys"])
        self.assertIn("steps", snapshot["post_keys"])
        # The secret value must not appear anywhere in the snapshot.
        self.assertNotIn("hunter2", repr(snapshot))

    def test_recording_failure_is_swallowed(self):
        """A failure while recording does not propagate from the hook."""

        bad_request = SimpleNamespace()  # missing everything
        with self.assertLogs(
            "autowisp.browser_interface.error_capture_middleware", "ERROR"
        ):
            # A non-exception object makes attribute access inside fail,
            # which must be swallowed (and logged), not propagated.
            self.assertIsNone(
                self.middleware.process_exception(bad_request, object())
            )


if __name__ == "__main__":
    unittest.main()
