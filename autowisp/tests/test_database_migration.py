"""Unit tests for the lightweight additive-column migration.

These need only a throwaway SQLite database, not a full project, so they
subclass ``unittest.TestCase`` directly.
"""

import os
import tempfile
import unittest

from sqlalchemy import create_engine, inspect, text

from autowisp.database.interface import apply_additive_migrations


class TestAdditiveMigrations(unittest.TestCase):
    """``apply_additive_migrations`` adds missing nullable columns."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            f"sqlite:///{os.path.join(self._tmp.name, 'm.db')}"
        )
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.engine.dispose)

    def _columns(self, table):
        return {col["name"] for col in inspect(self.engine).get_columns(table)}

    def _make_old_project(self):
        """Simulate an existing project: a ``pipeline_run`` table from
        before ``code_version`` and the ``error`` table existed, holding a
        row."""

        with self.engine.begin() as conn:
            conn.execute(
                text("CREATE TABLE pipeline_run (id INTEGER PRIMARY KEY)")
            )
            conn.execute(text("INSERT INTO pipeline_run (id) VALUES (1)"))

    def test_adds_missing_code_version(self):
        """An old pipeline_run (no code_version) gains the column."""

        self._make_old_project()
        self.assertNotIn("code_version", self._columns("pipeline_run"))

        apply_additive_migrations(self.engine)

        self.assertIn("code_version", self._columns("pipeline_run"))

    def test_creates_missing_error_table(self):
        """An existing project gains the new ``error`` table."""

        self._make_old_project()
        self.assertNotIn("error", inspect(self.engine).get_table_names())

        apply_additive_migrations(self.engine)

        self.assertIn("error", inspect(self.engine).get_table_names())

    def test_preexisting_rows_and_data_preserved(self):
        """Existing rows survive; the new column is NULL for them."""

        self._make_old_project()

        apply_additive_migrations(self.engine)

        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT id, code_version FROM pipeline_run WHERE id = 1")
            ).one()
        self.assertEqual(row.id, 1)
        self.assertIsNone(row.code_version)

    def test_idempotent(self):
        """Re-running is a no-op and does not raise."""

        self._make_old_project()

        apply_additive_migrations(self.engine)
        apply_additive_migrations(self.engine)

        self.assertIn("code_version", self._columns("pipeline_run"))
        self.assertIn("error", inspect(self.engine).get_table_names())


if __name__ == "__main__":
    unittest.main()
