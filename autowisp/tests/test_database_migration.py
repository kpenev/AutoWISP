"""Unit tests for project database migration.

These need only a throwaway SQLite database, not a full project, so they
subclass ``unittest.TestCase`` directly.
"""

import os
import re
import tempfile
import threading
import unittest

from alembic.script import ScriptDirectory
from sqlalchemy import (
    Index,
    MetaData,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.migrate import (
    BASELINE_REVISION,
    _alembic_config,
    _sqlite_immediate,
    _apply_additive_migrations as apply_additive_migrations,
    check_project_schema,
    create_project_schema,
    get_head_revision,
    get_project_revision,
    get_schema_drift,
    migrate_project,
)
from autowisp.exceptions import DatabaseError
from autowisp.tests import SERVER_URL_ENV, empty_server_database

# Resources here are released via addCleanup rather than `with`, since they
# are created in setUp and must outlive it.
# pylint: disable=consider-using-with


def on_server():
    """Whether this run is pointed at a MySQL/MariaDB server.

    The same switch the rest of the suite uses (see
    :mod:`autowisp.tests`), so one variable turns everything onto a server
    rather than each part having its own idea of where to look. Setting it
    runs *these* scenarios against one too, covering the backend-specific
    paths -- GET_LOCK, implicitly committed DDL, type comparison in the
    drift check -- with the tests that already exist rather than a parallel
    copy that would drift out of step.
    """

    return bool(os.environ.get(SERVER_URL_ENV))


class BackendMixin:
    """Supplies clean project databases on whichever backend is under test.

    SQLite gets a fresh file per engine; a server has only the one database,
    so it is emptied before each test instead.
    """

    # Mixed into unittest.TestCase subclasses, so setUp is an override
    # despite this class not deriving from TestCase itself.
    # pylint: disable=invalid-name
    def setUp(self):
        """Leave a clean project database ready for the test."""

        super().setUp()
        if on_server():
            empty_server_database(os.environ[SERVER_URL_ENV])
        else:
            self._tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self._tmp.cleanup)

    # pylint: enable=invalid-name

    def reset_backend(self):
        """Return the backend to empty part way through a test.

        The asymmetry ``setUp`` lives with, reached one level in: a test
        that builds several schemas in turn -- one per released version,
        say -- gets a fresh file per engine on SQLite and the *same*
        database every time on a server, where the second iteration would
        otherwise start from whatever the first left behind.
        """

        if on_server():
            empty_server_database(os.environ[SERVER_URL_ENV])

    def make_engine(self, name="project.db"):
        """An engine for a clean project database on the current backend."""

        if on_server():
            engine = create_engine(
                os.environ[SERVER_URL_ENV], poolclass=NullPool
            )
        else:
            engine = create_engine(
                f"sqlite:///{os.path.join(self._tmp.name, name)}",
                connect_args={"timeout": 30.0},
                poolclass=NullPool,
            )
        self.addCleanup(engine.dispose)
        return engine

    def migrate(self, engine):
        """Migrate, confirming the backup where the backend demands one."""

        return migrate_project(engine, assume_backed_up=on_server())

    @staticmethod
    def create_legacy_schema(engine):
        """Build the pre-migration schema: everything but the new index.

        Not create_all-then-drop. InnoDB refuses to drop
        ``image_observing_session`` because it is the index backing image's
        foreign key on ``observing_session_id`` -- MySQL indexes a foreign
        key column whether or not anyone asked. Leaving the index out of
        the metadata is both portable and a truer picture of a 1.8.1
        database, which never had it.
        """

        image = DataModelBase.metadata.tables["image"]
        held_back = {
            index
            for index in image.indexes
            if index.name == "image_observing_session"
        }
        image.indexes -= held_back
        try:
            DataModelBase.metadata.create_all(engine)
        finally:
            image.indexes |= held_back

    @staticmethod
    def add_stray_index(engine, name):
        """Add an index the models do not declare, to produce drift.

        Drift is provoked by adding something rather than removing it: the
        index this branch introduces cannot be dropped on MySQL (see
        :meth:`create_legacy_schema`).
        """

        with engine.begin() as connection:
            table = Table("image", MetaData(), autoload_with=connection)
            # jd, not a text column: indexing VARCHAR(1000) would exceed
            # InnoDB's 3072-byte key limit and fail for the wrong reason.
            Index(name, table.c.jd).create(connection)

    @staticmethod
    def has_index(engine, table, name):
        """Whether *table* carries an index called *name*."""

        return any(
            index["name"] == name
            for index in inspect(engine).get_indexes(table)
        )

    @staticmethod
    def list_triggers(engine):
        """The names of every trigger defined in the project database."""

        with engine.connect() as connection:
            if connection.dialect.name == "sqlite":
                query = "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            else:
                query = (
                    "SELECT trigger_name FROM information_schema.triggers "
                    "WHERE trigger_schema = DATABASE()"
                )
            return {row[0] for row in connection.exec_driver_sql(query)}


def expected_timestamp_triggers():
    """The triggers the models install when they create the schema.

    Derived from the metadata rather than listed, so a table added later
    is covered without anyone remembering to extend a literal here.
    """

    return {
        f"update_{table}_timestamp"
        for table in DataModelBase.metadata.tables.values()
        if "timestamp" in table.columns and table.primary_key.columns
    }


class TestAdditiveMigrations(BackendMixin, unittest.TestCase):
    """The pre-Alembic helper still adds missing nullable columns.

    It is private now and has exactly one caller -- the baseline path of
    :func:`migrate_project` -- but it is what every legacy project passes
    through, so it keeps its own tests.
    """

    def setUp(self):
        super().setUp()
        self.engine = self.make_engine("m.db")

    def _columns(self, table):
        return {col["name"] for col in inspect(self.engine).get_columns(table)}

    def _apply(self):
        """Run the helper the way migrate_project does: on one connection."""

        with self.engine.begin() as conn:
            apply_additive_migrations(conn)

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

        self._apply()

        self.assertIn("code_version", self._columns("pipeline_run"))

    def test_creates_missing_error_table(self):
        """An existing project gains the new ``error`` table."""

        self._make_old_project()
        self.assertNotIn("error", inspect(self.engine).get_table_names())

        self._apply()

        self.assertIn("error", inspect(self.engine).get_table_names())

    def test_adds_missing_resolved_to_old_error_table(self):
        """An older ``error`` table (no ``resolved``) gains the column."""

        with self.engine.begin() as conn:
            conn.execute(text("CREATE TABLE error (id INTEGER PRIMARY KEY)"))
        self.assertNotIn("resolved", self._columns("error"))

        self._apply()

        self.assertIn("resolved", self._columns("error"))

    def test_preexisting_rows_and_data_preserved(self):
        """Existing rows survive; the new column is NULL for them."""

        self._make_old_project()

        self._apply()

        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT id, code_version FROM pipeline_run WHERE id = 1")
            ).one()
        self.assertEqual(row.id, 1)
        self.assertIsNone(row.code_version)

    def test_idempotent(self):
        """Re-running is a no-op and does not raise."""

        self._make_old_project()

        self._apply()
        self._apply()

        self.assertIn("code_version", self._columns("pipeline_run"))
        self.assertIn("error", inspect(self.engine).get_table_names())


class TestTimestampTriggers(BackendMixin, unittest.TestCase):
    """Every table carrying ``timestamp`` keeps a trigger maintaining it.

    Nothing else checks this. ``get_schema_drift`` is alembic's comparison
    of tables, columns, indexes and constraints, and a trigger is none of
    those -- so the schema checks elsewhere in this file pass unchanged
    with every trigger in the database dropped.
    """

    def test_creating_the_schema_installs_all_of_them(self):
        """Creation covers the provenance tables, not just data_model's.

        The triggers used to be attached to each class
        ``import_table_definitions`` discovered, and that discovery is a
        glob which does not descend into ``data_model/provenance`` -- so
        twelve tables carried a ``timestamp`` column that nothing ever
        updated. Comparing against the metadata catches a repeat.
        """

        engine = self.make_engine("created.db")
        DataModelBase.metadata.create_all(engine)

        self.assertEqual(
            self.list_triggers(engine), expected_timestamp_triggers()
        )

    def test_migrating_reinstates_a_dropped_trigger(self):
        """Covers the path taken when the revisions have nothing to do.

        A database already at head skips the upgrade entirely, so the
        repair cannot ride along with a revision; it has to be a check
        made on the way past.
        """

        engine = self.make_engine("dropped.db")
        self.create_legacy_schema(engine)
        self.migrate(engine)

        casualty = "update_image_timestamp"
        self.assertIn(casualty, self.list_triggers(engine))
        with engine.begin() as connection:
            connection.exec_driver_sql(f"DROP TRIGGER {casualty}")
        self.assertNotIn(casualty, self.list_triggers(engine))

        self.migrate(engine)
        self.assertEqual(
            self.list_triggers(engine), expected_timestamp_triggers()
        )


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

    def test_revision_ids_fit_the_version_table(self):
        """Ids stay within Alembic's ``VARCHAR(32)`` version column.

        SQLite ignores a declared length, so an over-long id passes there
        and only fails on MySQL, mid-upgrade, with "Data too long for
        column 'version_num'" -- after some revisions have already been
        applied. Cheaper to catch here.
        """

        for revision in self.script.walk_revisions():
            self.assertLessEqual(
                len(revision.revision),
                32,
                f"{revision.revision!r} is {len(revision.revision)} "
                "characters; alembic_version.version_num holds 32",
            )

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

    def test_the_build_installs_every_revision(self):
        """The ``meson.build`` beside the revisions lists each one of them.

        The build installs only the files it lists. A revision left out
        works from a checkout, where the file is, and is absent from an
        installed package, which then stops one revision short and stamps
        projects at the wrong head. Nothing can tell that from the
        installed package: a chain that ends early is still a chain. So
        this compares the list with the directory, where a checkout has
        both.
        """

        build_fname = os.path.join(self.script.versions, "meson.build")
        if not os.path.exists(build_fname):
            self.skipTest("an installed package has no meson.build to check")
        with open(build_fname, encoding="utf-8") as build_file:
            listed = set(re.findall(r"'([^']+\.py)'", build_file.read()))

        self.assertEqual(
            listed,
            {
                fname
                for fname in os.listdir(self.script.versions)
                if fname.endswith(".py")
            },
        )


class TestMigrateProject(BackendMixin, unittest.TestCase):
    """The three database states :func:`migrate_project` has to handle."""

    def _legacy_engine(self):
        """A 1.8.1-era database: full schema, unstamped, missing the index."""

        engine = self.make_engine("legacy.db")
        self.create_legacy_schema(engine)
        return engine

    def test_fresh_database_is_stamped_without_running_revisions(self):
        """``create_all`` builds the current schema, so nothing is applied."""

        engine = self.make_engine("fresh.db")
        create_project_schema(engine)

        self.assertEqual(get_project_revision(engine), get_head_revision())
        check_project_schema(engine)

    def test_legacy_database_is_baselined_then_upgraded(self):
        """An unstamped project reaches head in one pass, no user action."""

        engine = self._legacy_engine()
        self.assertIsNone(get_project_revision(engine))

        result = self.migrate(engine)

        self.assertEqual(result["from"], BASELINE_REVISION)
        self.assertEqual(result["to"], get_head_revision())
        self.assertTrue(
            self.has_index(engine, "image", "image_observing_session")
        )
        check_project_schema(engine)

    def test_older_database_reaches_the_same_state(self):
        """One missing a pre-baseline column still lands at head."""

        engine = self._legacy_engine()
        # Simulate predating `error.resolved` by dropping the whole table;
        # the baseline step recreates it via create_all.
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS error"))

        self.migrate(engine)

        self.assertIn("error", inspect(engine).get_table_names())
        self.assertIn(
            "resolved",
            {col["name"] for col in inspect(engine).get_columns("error")},
        )
        check_project_schema(engine)

    def test_migration_is_idempotent(self):
        """A second run applies nothing and reports no change."""

        engine = self._legacy_engine()
        self.migrate(engine)

        result = self.migrate(engine)

        self.assertEqual(result["from"], result["to"])
        self.assertIsNone(result["backup"])

    def test_backup_is_taken_before_migrating(self):
        """SQLite is copied aside; a server is the administrator's job."""

        engine = self._legacy_engine()

        backup = self.migrate(engine)["backup"]

        if on_server():
            self.assertIsNone(backup)
        else:
            self.assertTrue(os.path.exists(backup))
            self.assertTrue(backup.endswith(".pre-baseline"))

    def test_project_creation_leaves_no_backup(self):
        """There is nothing to preserve when the database is empty."""

        engine = self.make_engine("new.db")

        self.assertIsNone(self.migrate(engine)["backup"])

    def test_server_refuses_without_a_confirmed_backup(self):
        """A shared database is not migrated as a side effect of anything.

        The SQLite path has no equivalent: it copies the file aside itself.
        """

        if not on_server():
            self.skipTest("backup confirmation only applies to servers")

        engine = self._legacy_engine()

        with self.assertRaises(DatabaseError) as caught:
            migrate_project(engine)

        self.assertIn("--assume-backed-up", str(caught.exception))
        self.assertIsNone(get_project_revision(engine))

    def test_crash_window_between_ddl_and_version_update(self):
        """DDL applied but the stamp lost -- the state MySQL can crash into.

        MySQL commits DDL implicitly, so a failure between the schema change
        and the ``alembic_version`` update leaves exactly this. Re-running
        must reach head rather than failing on an index that already exists.
        """

        engine = self._legacy_engine()
        self.migrate(engine)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM alembic_version"))

        self.migrate(engine)

        check_project_schema(engine)
        self.assertTrue(
            self.has_index(engine, "image", "image_observing_session")
        )


class TestCheckProjectSchema(BackendMixin, unittest.TestCase):
    """The read-only gate every project open -- and every worker -- runs."""

    def setUp(self):
        super().setUp()
        self.engine = self.make_engine("c.db")

    def test_passes_on_a_current_database(self):
        """No exception, and nothing written."""

        create_project_schema(self.engine)
        check_project_schema(self.engine)

    def test_raises_on_an_unbaselined_database(self):
        """A legacy project is refused, pointing at wisp-migrate."""

        self.create_legacy_schema(self.engine)

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

        self.create_legacy_schema(self.engine)

        with self.assertRaises(DatabaseError):
            check_project_schema(self.engine)

        self.assertIsNone(get_project_revision(self.engine))
        self.assertFalse(
            any(
                index["name"] == "image_observing_session"
                for index in inspect(self.engine).get_indexes("image")
            )
        )


@unittest.skipIf(on_server(), "exercises SQLite's own locking")
class TestSqliteMigrationLock(unittest.TestCase):
    """The SQLite half of the migration lock really does exclude a second
    writer.

    Alembic does no locking of its own, so without this two migrators can
    both read the same revision and both try to apply it. On SQLite the
    protection is ``BEGIN IMMEDIATE``, which pysqlite does not issue by
    default -- it defers ``BEGIN`` until the first write, leaving a window in
    which both have already read. The server equivalent is ``GET_LOCK``,
    covered by :class:`TestConcurrentMigration` when pointed at one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "lock.db")
        with self._engine().begin() as connection:
            connection.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY)"))

    def _engine(self, timeout=30.0):
        """An engine with a busy timeout, as interface.py builds them."""

        engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"timeout": timeout},
            poolclass=NullPool,
        )
        self.addCleanup(engine.dispose)
        return engine

    def _write_from_elsewhere(self):
        """Write from an independent connection, failing fast if locked."""

        with self._engine(timeout=0.5).begin() as connection:
            connection.execute(text("INSERT INTO t (id) VALUES (1)"))

    def test_transaction_alone_does_not_hold_the_write_lock(self):
        """Baseline: the window this guard closes is real.

        Without the guard an open transaction that has not yet written lets
        another connection write, which is what allows two migrators to both
        read a stale revision.
        """

        engine = self._engine()
        with engine.begin():
            self._write_from_elsewhere()

    def test_begin_immediate_excludes_a_second_writer(self):
        """With the guard, the write lock is held from the start."""

        engine = self._engine()
        with _sqlite_immediate(engine):
            with engine.begin():
                with self.assertRaises(OperationalError) as caught:
                    self._write_from_elsewhere()
        self.assertIn("locked", str(caught.exception).lower())

    def test_guard_is_removed_afterwards(self):
        """The listeners are per-call and must not leak onto the engine."""

        engine = self._engine()
        with _sqlite_immediate(engine):
            pass
        with engine.begin():
            self._write_from_elsewhere()


class TestConcurrentMigration(BackendMixin, unittest.TestCase):
    """Two migrators racing on one database must not corrupt or crash it.

    Exercises whichever lock the backend uses: ``BEGIN IMMEDIATE`` on
    SQLite, ``GET_LOCK`` on a server. Alembic supplies neither.
    """

    def setUp(self):
        super().setUp()
        engine = self.make_engine("race.db")
        self.create_legacy_schema(engine)

    def test_two_migrators_reach_head_without_error(self):
        """Both calls return, and the database ends up correctly migrated."""

        barrier = threading.Barrier(2)
        errors = []

        def migrate():
            engine = self.make_engine("race.db")
            try:
                barrier.wait(timeout=30)
                self.migrate(engine)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)

        threads = [threading.Thread(target=migrate) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
            self.assertFalse(thread.is_alive(), "migration deadlocked")

        self.assertEqual([str(error) for error in errors], [])

        engine = self.make_engine("race.db")
        self.assertEqual(get_project_revision(engine), get_head_revision())
        check_project_schema(engine)
        indexes = [
            index["name"]
            for index in inspect(engine).get_indexes("image")
            if index["name"] == "image_observing_session"
        ]
        self.assertEqual(len(indexes), 1)


class TestSchemaDrift(BackendMixin, unittest.TestCase):
    """The revision chain and the ORM models describe the same schema.

    A revision may not import the models -- it has to keep meaning the same
    schema forever, while the models move -- so anything a revision creates
    is declared twice, once in ``data_model`` and once in the revision.
    Nothing keeps the two in step except this check, which is why the plan
    calls for it rather than for sharing the definitions.

    Worth running per backend: the comparison is over reflected types, and
    what MySQL reports for a column is not what SQLite does.
    """

    def test_migrated_database_matches_the_models(self):
        """A database the revisions built agrees with the models.

        Note this cannot catch a model changed with no revision to match:
        the "before" state here comes from today's metadata too, so both
        sides move together. :class:`TestUpgradeFromRelease` is the test
        that catches that, by building the "before" state from released
        code.
        """

        engine = self.make_engine("migrated.db")
        self.create_legacy_schema(engine)
        self.migrate(engine)

        self.assertEqual(get_schema_drift(engine), [])

    def test_created_database_matches_the_models(self):
        """Control: create_all builds from the models, so it must agree."""

        engine = self.make_engine("created.db")
        create_project_schema(engine)

        self.assertEqual(get_schema_drift(engine), [])

    def test_drift_is_actually_detected(self):
        """The check discriminates -- an empty result means agreement.

        Without this the two tests above would pass just as happily if
        get_schema_drift() always returned nothing.
        """

        engine = self.make_engine("drifted.db")
        create_project_schema(engine)
        self.add_stray_index(engine, "not_in_the_models")

        drift = get_schema_drift(engine)

        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0][0], "remove_index")
        self.assertEqual(drift[0][1].name, "not_in_the_models")


if __name__ == "__main__":
    unittest.main()
