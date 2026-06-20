"""Unit tests for the phase-1 exception hierarchy and FrozenRow.

These tests need no pipeline fixtures or database, so they subclass
``unittest.TestCase`` directly rather than ``AutoWISPTestCase``.
"""

import inspect
import os
import pickle
import unittest
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

from autowisp import exceptions
from autowisp.exceptions import (
    AutoWISPError,
    Component,
    FileKind,
    RelatedFile,
    StepError,
)
from autowisp.pipeline_exceptions import ConvergenceError, HDF5LayoutError
from autowisp.database.frozen_row import FrozenRow
from autowisp.database.interface import snapshot_row


def _concrete_exception_classes():
    """All AutoWISPError subclasses defined in ``autowisp.exceptions``.

    Excludes ``AutoWISPError`` itself, which deliberately does not set a
    ``component`` (that is the contract every subclass must satisfy).
    """

    return [
        obj
        for _, obj in inspect.getmembers(exceptions, inspect.isclass)
        if issubclass(obj, AutoWISPError) and obj is not AutoWISPError
    ]


def _full_payload():
    """A populated kwargs dict exercising every context field."""

    return {
        "related_files": [
            RelatedFile(FileKind.DR_FILE, Path("/tmp/x.h5"), role="input")
        ],
        "pipeline_run": FrozenRow(
            "pipeline_run", {"id": 7, "host": "node1", "started": None}
        ),
        "crashed": datetime(2026, 6, 20, 12, 0, 0),
        "subprocess_id": 4321,
        "user_message": "something went wrong",
        "details": {"expected": 3, "actual": 5, "path": Path("/tmp/y")},
    }


class TestExceptionHierarchy(unittest.TestCase):
    """Contracts every concrete AutoWISP exception must satisfy."""

    def test_every_concrete_class_sets_component(self):
        """Each concrete subclass has a ``Component`` (never ``None``)."""

        classes = _concrete_exception_classes()
        self.assertTrue(classes, "no exception classes discovered")
        for cls in classes:
            exc = cls("test message")
            self.assertIsInstance(
                exc.component,
                Component,
                f"{cls.__name__}.component is not a Component",
            )

    def test_minimal_construction_defaults(self):
        """A message-only construction leaves context fields empty."""

        exc = StepError("boom")
        self.assertEqual(str(exc), "boom")
        self.assertEqual(exc.user_message, "boom")
        self.assertEqual(exc.related_files, ())
        self.assertIsNone(exc.pipeline_run)
        self.assertIsNone(exc.crashed)
        self.assertIsNone(exc.subprocess_id)
        self.assertEqual(exc.details, {})
        self.assertIsNone(exc.step_name)

    def test_full_payload_pickle_round_trip(self):
        """All context fields survive a pickle round-trip (Pool transport)."""

        exc = StepError("boom", step_name="solve_astrometry", **_full_payload())
        restored = pickle.loads(pickle.dumps(exc))

        self.assertIs(type(restored), StepError)
        self.assertEqual(str(restored), "boom")
        self.assertEqual(restored.step_name, "solve_astrometry")
        self.assertEqual(restored.component, Component.STEP)
        self.assertEqual(restored.user_message, "something went wrong")
        self.assertEqual(restored.subprocess_id, 4321)
        self.assertEqual(restored.crashed, exc.crashed)
        self.assertEqual(restored.details, exc.details)
        self.assertEqual(restored.related_files, exc.related_files)
        self.assertEqual(restored.pipeline_run.id, 7)
        self.assertEqual(restored.pipeline_run.host, "node1")

    def test_all_classes_pickle_round_trip(self):
        """Every concrete class round-trips with a full payload."""

        for cls in _concrete_exception_classes():
            exc = cls("msg", **_full_payload())
            restored = pickle.loads(pickle.dumps(exc))
            self.assertIs(type(restored), cls)
            self.assertEqual(restored.component, exc.component)
            self.assertEqual(restored.subprocess_id, 4321)
            self.assertEqual(restored.pipeline_run.host, "node1")

    def test_stamp_subprocess_sets_and_is_idempotent(self):
        """First stamp records this PID; a later stamp does not overwrite."""

        exc = StepError("boom")
        self.assertIsNone(exc.subprocess_id)
        exc.stamp_subprocess()
        self.assertEqual(exc.subprocess_id, os.getpid())

        exc.subprocess_id = 999
        exc.stamp_subprocess()
        self.assertEqual(exc.subprocess_id, 999)

    def test_with_pipeline_run_attaches_and_sets_crashed(self):
        """``with_pipeline_run`` stores the row and fills ``crashed``."""

        run = FrozenRow("pipeline_run", {"id": 12, "host": "h"})
        exc = StepError("boom")
        result = exc.with_pipeline_run(run)

        self.assertIs(result, exc)
        self.assertIs(exc.pipeline_run, run)
        self.assertIsInstance(exc.crashed, datetime)

    def test_with_pipeline_run_keeps_existing_crashed(self):
        """An already-set ``crashed`` is not overwritten."""

        when = datetime(2020, 1, 1)
        exc = StepError("boom", crashed=when)
        exc.with_pipeline_run(FrozenRow("pipeline_run", {"id": 1}))
        self.assertEqual(exc.crashed, when)


class TestMigratedExceptions(unittest.TestCase):
    """Re-rooted legacy classes keep both lineages."""

    def test_convergence_error_dual_lineage(self):
        """``ConvergenceError`` is both AutoWISPError and RuntimeError."""

        exc = ConvergenceError("did not converge")
        self.assertIsInstance(exc, AutoWISPError)
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.component, Component.STEP)
        self.assertEqual(str(exc), "did not converge")

    def test_caught_as_runtime_error(self):
        """Legacy ``except RuntimeError`` still catches it."""

        with self.assertRaises(RuntimeError):
            raise ConvergenceError("nope")

    def test_hdf5_layout_error_is_pipeline(self):
        """``HDF5LayoutError`` is a pipeline-component RuntimeError."""

        exc = HDF5LayoutError("bad layout")
        self.assertIsInstance(exc, AutoWISPError)
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.component, Component.PIPELINE)


class TestFrozenRow(unittest.TestCase):
    """FrozenRow attribute access, immutability, and pickling."""

    def test_attribute_access(self):
        """Columns are reachable as attributes and via ``columns``."""

        row = FrozenRow("pipeline_run", {"id": 3, "host": "node"})
        self.assertEqual(row.id, 3)
        self.assertEqual(row.host, "node")
        self.assertEqual(row.columns["host"], "node")
        self.assertEqual(row.table, "pipeline_run")

    def test_missing_attribute_raises(self):
        """An unknown column raises ``AttributeError``."""

        row = FrozenRow("pipeline_run", {"id": 3})
        with self.assertRaises(AttributeError):
            _ = row.does_not_exist

    def test_pickle_round_trip(self):
        """A FrozenRow survives pickling with columns intact."""

        row = FrozenRow("pipeline_run", {"id": 3, "host": "node"})
        restored = pickle.loads(pickle.dumps(row))
        self.assertEqual(restored.table, "pipeline_run")
        self.assertEqual(restored.id, 3)
        self.assertEqual(restored.host, "node")


_Base = declarative_base()


class _ToyRow(_Base):
    """Throwaway ORM model for exercising ``snapshot_row``."""

    __tablename__ = "toy_row"
    id = Column(Integer, primary_key=True)
    host = Column(String)
    secret = Column(String)


class TestSnapshotRow(unittest.TestCase):
    """``snapshot_row`` freezes a live ORM instance into a FrozenRow."""

    def test_snapshot_captures_columns(self):
        """All mapped columns are captured and reachable by attribute."""

        obj = _ToyRow(id=5, host="node2", secret="x")
        snap = snapshot_row(obj)

        self.assertIsInstance(snap, FrozenRow)
        self.assertEqual(snap.table, "toy_row")
        self.assertEqual(snap.id, 5)
        self.assertEqual(snap.host, "node2")
        self.assertEqual(set(snap.columns), {"id", "host", "secret"})

    def test_exclude_omits_columns(self):
        """Excluded column keys do not appear in the snapshot."""

        obj = _ToyRow(id=5, host="node2", secret="x")
        snap = snapshot_row(obj, exclude=("secret",))
        self.assertNotIn("secret", snap.columns)
        self.assertIn("host", snap.columns)

    def test_snapshot_pickles(self):
        """A snapshot of a live row pickles cleanly (no session attached)."""

        snap = snapshot_row(_ToyRow(id=5, host="node2", secret="x"))
        restored = pickle.loads(pickle.dumps(snap))
        self.assertEqual(restored.host, "node2")


if __name__ == "__main__":
    unittest.main()
