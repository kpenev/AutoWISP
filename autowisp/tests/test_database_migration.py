"""Unit tests for project database migration.

These need only a throwaway SQLite database, not a full project, so they
subclass ``unittest.TestCase`` directly.
"""

import os
import tempfile
import unittest

from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.migrate import (
    BASELINE_REVISION,
    _alembic_config,
    _apply_additive_migrations as apply_additive_migrations,
    check_project_schema,
    create_project_schema,
    get_head_revision,
    get_project_revision,
    migrate_project,
)
from autowisp.exceptions import DatabaseError

# Resources here are released via addCleanup rather than `with`, since they
# are created in setUp and must outlive it.
# pylint: disable=consider-using-with


class TestAdditiveMigrations(unittest.TestCase):
    """The pre-Alembic helper still adds missing nullable columns.

    It is private now and has exactly one caller -- the baseline path of
    :func:`migrate_project` -- but it is what every legacy project passes
    through, so it keeps its own tests.
    """

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

    def test_adds_missing_resolved_to_old_error_table(self):
        """An older ``error`` table (no ``resolved``) gains the column."""

        with self.engine.begin() as conn:
            conn.execute(text("CREATE TABLE error (id INTEGER PRIMARY KEY)"))
        self.assertNotIn("resolved", self._columns("error"))

        apply_additive_migrations(self.engine)

        self.assertIn("resolved", self._columns("error"))

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


class TestRevisionChain(unittest.TestCase):
    """The revision chain is linear, ordered, and consistently named.

    A fork is what actually breaks ``upgrade head``: two branches each
    adding a revision both point at the same parent, and Alembic then
    refuses to upgrade because the head is ambiguous. Catching that here
    means catching it in CI rather than at a user's next project open.
    """

    def setUp(self):
        self.script = ScriptDirectory.from_config(_alembic_config())

    def test_single_head(self):
        """Exactly one head: no fork left unresolved by a merge or rebase."""

        heads = self.script.get_heads()
        self.assertEqual(
            len(heads),
            1,
            f"Expected a linear chain but found {len(heads)} heads: {heads}. "
            "Re-point the newer revision's down_revision at the other.",
        )

    def test_revision_ids_are_numbered_slugs(self):
        """Every revision id matches the ``NNNN_slug`` convention."""

        for revision in self.script.walk_revisions():
            self.assertRegex(revision.revision, r"^\d{4}_[a-z0-9_]+$")

    def test_numbers_increase_along_the_chain(self):
        """Numeric prefixes strictly increase from base to head.

        Catches a duplicate number and a revision merged out of order.
        """

        numbers = [
            int(revision.revision[:4])
            for revision in self.script.walk_revisions()
        ]
        self.assertEqual(
            numbers,
            sorted(set(numbers), reverse=True),
            "Revision numbers must be unique and increase from base to head.",
        )

    def test_no_revision_has_two_parents(self):
        """No merge revisions: the history stays linear."""

        for revision in self.script.walk_revisions():
            parents = revision.down_revision
            if isinstance(parents, tuple):
                self.fail(
                    f"{revision.revision} has {len(parents)} parents; resolve "
                    "forks by re-pointing down_revision, not alembic merge."
                )

    def test_baseline_is_the_root(self):
        """``BASELINE_REVISION`` names the one revision with no parent."""

        roots = [
            revision.revision
            for revision in self.script.walk_revisions()
            if revision.down_revision is None
        ]
        self.assertEqual(roots, [BASELINE_REVISION])


class TestMigrateProject(unittest.TestCase):
    """The three database states :func:`migrate_project` has to handle."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _engine(self, name):
        engine = create_engine(
            f"sqlite:///{os.path.join(self._tmp.name, name)}"
        )
        self.addCleanup(engine.dispose)
        return engine

    def _legacy_engine(self):
        """A 1.8.1-era database: full schema, unstamped, missing the index."""

        engine = self._engine("legacy.db")
        DataModelBase.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS image_observing_session"))
        return engine

    @staticmethod
    def _has_index(engine, table, name):
        return any(
            index["name"] == name
            for index in inspect(engine).get_indexes(table)
        )

    def test_fresh_database_is_stamped_without_running_revisions(self):
        """``create_all`` builds the current schema, so nothing is applied."""

        engine = self._engine("fresh.db")
        create_project_schema(engine)

        self.assertEqual(get_project_revision(engine), get_head_revision())
        check_project_schema(engine)

    def test_legacy_database_is_baselined_then_upgraded(self):
        """An unstamped project reaches head in one pass, no user action."""

        engine = self._legacy_engine()
        self.assertIsNone(get_project_revision(engine))

        result = migrate_project(engine)

        self.assertEqual(result["from"], BASELINE_REVISION)
        self.assertEqual(result["to"], get_head_revision())
        self.assertTrue(
            self._has_index(engine, "image", "image_observing_session")
        )
        check_project_schema(engine)

    def test_older_database_reaches_the_same_state(self):
        """One missing a pre-baseline column still lands at head."""

        engine = self._legacy_engine()
        # Simulate predating `error.resolved` by dropping the whole table;
        # the baseline step recreates it via create_all.
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS error"))

        migrate_project(engine)

        self.assertIn("error", inspect(engine).get_table_names())
        self.assertIn(
            "resolved",
            {col["name"] for col in inspect(engine).get_columns("error")},
        )
        check_project_schema(engine)

    def test_migration_is_idempotent(self):
        """A second run applies nothing and reports no change."""

        engine = self._legacy_engine()
        migrate_project(engine)

        result = migrate_project(engine)

        self.assertEqual(result["from"], result["to"])
        self.assertIsNone(result["backup"])

    def test_sqlite_is_backed_up_before_migrating(self):
        """The copy is named for the revision it is a snapshot of."""

        engine = self._legacy_engine()

        backup = migrate_project(engine)["backup"]

        self.assertIsNotNone(backup)
        self.assertTrue(os.path.exists(backup))
        self.assertTrue(backup.endswith(".pre-baseline"))

    def test_project_creation_leaves_no_backup(self):
        """There is nothing to preserve when the database is empty."""

        engine = self._engine("new.db")

        self.assertIsNone(migrate_project(engine)["backup"])

    def test_crash_window_between_ddl_and_version_update(self):
        """DDL applied but the stamp lost -- the state MySQL can crash into.

        MySQL commits DDL implicitly, so a failure between the schema change
        and the ``alembic_version`` update leaves exactly this. Re-running
        must reach head rather than failing on an index that already exists.
        """

        engine = self._legacy_engine()
        migrate_project(engine)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM alembic_version"))

        migrate_project(engine)

        check_project_schema(engine)
        self.assertTrue(
            self._has_index(engine, "image", "image_observing_session")
        )


class TestCheckProjectSchema(unittest.TestCase):
    """The read-only gate every project open -- and every worker -- runs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.engine = create_engine(
            f"sqlite:///{os.path.join(self._tmp.name, 'c.db')}"
        )
        self.addCleanup(self.engine.dispose)

    def test_passes_on_a_current_database(self):
        """No exception, and nothing written."""

        create_project_schema(self.engine)
        check_project_schema(self.engine)

    def test_raises_on_an_unbaselined_database(self):
        """A legacy project is refused, pointing at wisp-migrate."""

        DataModelBase.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS image_observing_session"))

        with self.assertRaises(DatabaseError) as caught:
            check_project_schema(self.engine)

        self.assertIn("wisp-migrate", str(caught.exception))

    def test_raises_on_a_database_from_the_future(self):
        """A newer AutoWISP migrated it: advise upgrading, not migrating.

        Reachable whenever a centralised database is shared -- one user
        upgrades and migrates, another opens it on older code.
        """

        create_project_schema(self.engine)
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE alembic_version SET version_num = '9999_later'")
            )

        with self.assertRaises(DatabaseError) as caught:
            check_project_schema(self.engine)

        message = str(caught.exception)
        self.assertIn("does not know about", message)
        self.assertNotIn("wisp-migrate", message)

    def test_does_not_mutate_a_stale_database(self):
        """The gate must never issue DDL: workers run it concurrently."""

        DataModelBase.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS image_observing_session"))

        with self.assertRaises(DatabaseError):
            check_project_schema(self.engine)

        self.assertIsNone(get_project_revision(self.engine))
        self.assertFalse(
            any(
                index["name"] == "image_observing_session"
                for index in inspect(self.engine).get_indexes("image")
            )
        )


if __name__ == "__main__":
    unittest.main()
