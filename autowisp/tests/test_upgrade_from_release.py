"""Tests that projects created by released versions upgrade to today's.

Unlike the rest of the migration tests, these start from what a release
itself creates, exported from its tag and run in a subprocess, so they need
the repository's history and are skipped outside a checkout.
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from collections import Counter

from alembic import command as alembic_command
from alembic.runtime.migration import MigrationContext
from sqlalchemy import MetaData, Table, select

from autowisp.database.migrate import (
    BASELINE_REVISION,
    _alembic_config,
    _locked_connection,
    _apply_additive_migrations as apply_additive_migrations,
    get_head_revision,
    get_project_revision,
    get_schema_drift,
)
from autowisp.exceptions import DatabaseError
from autowisp.tests.test_database_migration import (
    BackendMixin,
    expected_timestamp_triggers,
    on_server,
)


def _repo_root():
    """Return the repository's top level, or None outside a checkout."""

    try:
        return subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(__file__),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git(*args, binary=False):
    """Run git at the repository root; return output, or None if it fails.

    The root, not this file's directory: ``git archive`` refuses a pathspec
    reaching outside the current directory, so it has to be invoked from
    the top level.
    """

    root = _repo_root()
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=not binary,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout if binary else result.stdout.strip()


def get_project_contents(engine, own_values=()):
    """
    Return every row of a project database, comparable between projects.

    Two projects holding the same definitions number their rows differently:
    a project that has been migrated keeps the ids it was created with, and
    the revisions append what a new project creates in place. So each row is
    given without its own id and timestamp, and each foreign key is replaced
    by the row it points to, which makes "this step depends on that one"
    read the same whatever the two are numbered.

    Every table the database has is included, so one added later is covered
    without being listed here.

    Args:
        engine:    The SQLAlchemy engine for the project database.

        own_values([str]):    The names of the parameters whose configured
            value belongs to the project, and so is not to be compared. It is
            replaced by a placeholder.

    Returns:
        dict:
            The rows of each table by its name, each a tuple of (column,
            value) pairs. Sorted, except those of ``processing_sequence``,
            which stay in the order of their ids: that order is what the
            table records.
    """

    metadata = MetaData()
    metadata.reflect(engine)
    tables = [
        table
        for table in metadata.tables.values()
        if table.name != "alembic_version"
    ]
    with engine.connect() as connection:
        rows = {
            table.name: connection.execute(
                select(table).order_by(*table.primary_key.columns)
            )
            .mappings()
            .all()
            for table in tables
        }
    own_value_ids = {
        row["id"] for row in rows["parameter"] if row["name"] in own_values
    }

    def resolve(table, row):
        """Return the row with each reference replaced by what it names."""

        resolved = []
        for column in table.columns:
            # A condition's id is not a row id: it names the group of
            # expressions the condition consists of, and is how the
            # configuration refers to it.
            if column.name == "timestamp" or (
                column.primary_key and table.name != "condition"
            ):
                continue
            value = row[column.name]
            for foreign_key in column.foreign_keys:
                if value is not None:
                    referred = foreign_key.column
                    value = resolve(
                        referred.table,
                        next(
                            candidate
                            for candidate in rows[referred.table.name]
                            if candidate[referred.name] == value
                        ),
                    )
            if (
                table.name == "configuration"
                and column.name == "value"
                and row["parameter_id"] in own_value_ids
            ):
                value = "<the project's own>"
            resolved.append((column.name, value))
        return tuple(resolved)

    result = {}
    for table in tables:
        resolved_rows = [resolve(table, row) for row in rows[table.name]]
        result[table.name] = (
            resolved_rows
            if table.name == "processing_sequence"
            else sorted(resolved_rows, key=repr)
        )
    return result


class TestUpgradeFromRelease(BackendMixin, unittest.TestCase):
    """A project created by a *released* AutoWISP becomes one of today's.

    In its schema, and in the rows creating a project fills it with: the
    steps, their parameters, the layout of its products.

    This is the test that catches a model, or what a new project is given,
    changed without a revision to match. The other migration tests build
    their "before" state from today's metadata, so a missing revision moves
    both sides together and goes unnoticed. Here the starting point is
    built by the released code itself, checked out from its tag, so the
    revision chain is the only thing that can close the gap.

    That released package is loaded in a **subprocess**: it defines the
    same module names as the code under test, so importing both into one
    interpreter would have whichever came first shadow the other.
    """

    release_baselines = ("1.8.1", "2.0.0")
    """Released versions a project database may be upgraded from.

    Add each new release tag as it ships; every entry gets its own
    upgrade-to-current check.
    """

    stored_values_kept = (
        # Existing projects go on identifying observations as they did.
        "tfa-observation-id",
        # The default became a list; the single number means the same.
        "stamp-pixel-outlier-threshold",
        "stamp-smoothing-outlier-threshold",
        # The iteration of an existing master is recovered by matching its
        # file name against the format, so a project's masters stay
        # parseable only under the format they were named by.
        "master-photref-fname-format",
        "magfit-stat-fname-format",
    )
    """Parameters whose default has changed since one of the releases.

    An upgraded project keeps the value it stores, so it is the one thing
    such a project may hold that a new one does not. Each entry is a
    decision that the stored value is to be kept rather than migrated: a
    default changed without one fails the comparison with a new project.
    """

    last_schema_revision = "0011_diagnostic_expression"
    """The revision the ones that change a project's rows come after."""

    def setUp(self):
        super().setUp()
        if _git("rev-parse", "--git-dir") is None:
            self.skipTest("not a git checkout, so releases cannot be exported")

    def _export_release(self, ref):
        """Extract the ``autowisp`` package as of *ref* into a temp dir."""

        if _git("rev-parse", "--verify", f"{ref}^{{commit}}") is None:
            self.skipTest(
                f"tag {ref} unavailable -- CI needs fetch-depth: 0 for tags"
            )
        archive = _git("archive", ref, "autowisp", binary=True)
        self.assertIsNotNone(archive, f"could not export {ref}")

        target = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target, True)
        # tarfile rather than the tar binary: no external command, and no
        # assumption about which tar the platform ships.
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
        return target

    def _build_release_schema(self, source, engine):
        """Create the release's schema, running that release's own code."""

        # hide_password=False: str(URL) masks the password, which would
        # make the subprocess fail to connect to a server.
        url = engine.url.render_as_string(hide_password=False)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {source!r})\n"
                "from sqlalchemy import create_engine\n"
                "from autowisp.database.data_model.base import DataModelBase\n"
                "import autowisp.database.data_model\n"
                f"DataModelBase.metadata.create_all(create_engine({url!r}))\n",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            # The release cannot create its own schema here, so there is no
            # upgrade to check -- not a failure of the revisions. Reachable:
            # 1.8.1 cannot be created on MySQL 8.4 under utf8mb4 at all,
            # because its VARCHAR(1000) unique keys exceed InnoDB's index
            # limit. That is precisely what these revisions fix, and why
            # real deployments run a narrower charset.
            self.skipTest(
                "the release cannot build its schema on this backend, so "
                f"there is no upgrade path to check:\n{result.stderr[-300:]}"
            )

    def _initialize_project(self, source, label, *, released=True):
        """
        Create a project and fill it, running the code found in *source*.

        Filled the way creating a project fills it: the steps, their
        parameters and default configuration, the processing sequence, the
        master types, the layout of the HDF5 products, the diagnostic types
        and the expression library.

        Args:
            source(str):    The directory holding the ``autowisp`` package
                to create the project with.

            label(str):    Tells the project apart from the others a test
                creates.

            released(bool):    Is *source* a release rather than the code
                under test? A release that cannot create its project on this
                backend leaves nothing to check; the code under test must.

        Returns:
            The SQLAlchemy engine for the project's database.
        """

        if on_server():
            project_home = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, project_home, True)
        else:
            project_home = os.path.join(self._tmp.name, label)
            os.makedirs(project_home)
        engine = self.make_engine(os.path.join(label, "autowisp.db"))
        # hide_password=False: see _build_release_schema().
        url = (
            ", db_url=" + repr(engine.url.render_as_string(hide_password=False))
            if on_server()
            else ""
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {source!r})\n"
                "from argparse import Namespace\n"
                "from autowisp.database.interface import set_project_home\n"
                "from autowisp.database.initialize_database import (\n"
                "    initialize_database\n"
                ")\n"
                f"set_project_home({project_home!r}{url})\n"
                # As the browser interface creates a project.
                "initialize_database(\n"
                "    Namespace(\n"
                "        drop_hdf5_structure_tables=False,\n"
                "        drop_all_tables=True,\n"
                "    )\n"
                ")\n",
            ],
            cwd=project_home,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode and released:
            self.skipTest(
                "the release cannot create its project on this backend, so "
                f"there is no upgrade path to check:\n{result.stderr[-300:]}"
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return engine

    @staticmethod
    def _run_alembic(engine, command, revision):
        """
        Run one Alembic command on a project, as ``migrate_project`` would.

        ``migrate_project`` only ever goes to the head, so stopping at an
        earlier revision, downgrading or re-stamping go through here.

        Args:
            engine:    The SQLAlchemy engine for the project database.

            command(callable):    The function of ``alembic.command`` to run,
                e.g. ``alembic.command.downgrade``.

            revision(str):    The revision to run it to.
        """

        config = _alembic_config()
        with _locked_connection(engine) as connection:
            config.attributes["connection"] = connection
            if (
                MigrationContext.configure(connection).get_current_revision()
                is None
            ):
                # Predates Alembic: see migrate_project().
                apply_additive_migrations(connection)
                alembic_command.stamp(config, BASELINE_REVISION)
            command(config, revision)

    def _get_stored_values(self, engine):
        """Return (parameter, value) for each of ``stored_values_kept``."""

        with engine.connect() as connection:
            parameter = Table("parameter", MetaData(), autoload_with=connection)
            configuration = Table(
                "configuration", MetaData(), autoload_with=connection
            )
            return sorted(
                tuple(row)
                for row in connection.execute(
                    select(parameter.c.name, configuration.c.value)
                    .join(
                        configuration,
                        configuration.c.parameter_id == parameter.c.id,
                    )
                    .where(parameter.c.name.in_(self.stored_values_kept))
                )
            )

    @staticmethod
    def _add_user_rule(engine):
        """Set an exclusion rule under a condition, as a user would."""

        with engine.begin() as connection:
            parameter = Table("parameter", MetaData(), autoload_with=connection)
            configuration = Table(
                "configuration", MetaData(), autoload_with=connection
            )
            connection.execute(
                configuration.insert().values(
                    parameter_id=connection.scalar(
                        select(parameter.c.id).where(
                            parameter.c.name == "epd-exclusion-rule"
                        )
                    ),
                    version=0,
                    condition_id=2,
                    value="cloud[0] > 0.3",
                )
            )

    def test_every_release_upgrades_to_the_current_schema(self):
        """Each released schema, once migrated, agrees with today's models."""

        for ref in self.release_baselines:
            with self.subTest(release=ref):
                self.reset_backend()
                engine = self.make_engine(f"from_{ref}.db")
                self._build_release_schema(self._export_release(ref), engine)

                # Predates Alembic, so this covers the whole path: reach the
                # baseline, stamp it, then apply every revision.
                self.assertIsNone(get_project_revision(engine))
                self.migrate(engine)

                self.assertEqual(
                    get_project_revision(engine), get_head_revision()
                )
                self.assertEqual(
                    get_schema_drift(engine),
                    [],
                    f"a database from {ref} does not reach the current "
                    "schema; the differences above each need a revision",
                )

    def test_every_release_keeps_its_timestamp_triggers(self):
        """Upgrading does not cost the database its triggers.

        SQLite cannot alter a column in place, so ``batch_alter_table``
        rebuilds the table and the drop takes its triggers with it. The
        rebuilt table is created by the revision rather than from the
        models, so nothing puts them back -- a 1.8.1 database used to lose
        seven this way, and the check above could not see it.
        """

        expected = expected_timestamp_triggers()
        for ref in self.release_baselines:
            with self.subTest(release=ref):
                self.reset_backend()
                engine = self.make_engine(f"triggers_{ref}.db")
                self._build_release_schema(self._export_release(ref), engine)
                self.migrate(engine)

                self.assertEqual(self.list_triggers(engine), expected)

    def test_a_value_too_long_to_keep_stops_the_migration(self):
        """Narrowing a column refuses rather than truncating.

        Refusing is the point: on a server not running in strict mode the
        ALTER would truncate the value silently.

        Uses ``condition_expression.expression`` (1000 -> 768 in ``0005``)
        rather than ``image.raw_fname``, which narrows identically in
        ``0004``. The guard lives in the shared ``resize_varchar_column``,
        so either exercises it -- but condition_expression has no foreign
        keys, whereas an image row needs an image_type and an observing
        session, and that in turn needs an observer, camera, telescope,
        mount, observatory and target. A server enforces every one of
        those, so the alternative was either a dozen rows of fixture or
        switching the checks off, and neither has anything to do with
        column widths.
        """

        ref = self.release_baselines[0]
        engine = self.make_engine(f"toolong_{ref}.db")
        self._build_release_schema(self._export_release(ref), engine)

        long_expression = "x" * 800
        with engine.begin() as connection:
            table = Table(
                "condition_expression", MetaData(), autoload_with=connection
            )
            connection.execute(
                table.insert().values(expression=long_expression)
            )

        with self.assertRaises(DatabaseError) as caught:
            self.migrate(engine)

        message = str(caught.exception)
        self.assertIn("expression", message)
        self.assertIn(str(len(long_expression)), message)

        # The value is still intact, and the schema was left alone.
        with engine.begin() as connection:
            table = Table(
                "condition_expression", MetaData(), autoload_with=connection
            )
            kept = connection.execute(select(table.c.expression)).scalar()
        self.assertEqual(kept, long_expression)

    def _assert_holds(self, engine, expected, *, own_values=(), reason):
        """
        Check that a project holds exactly the given contents.

        Args:
            engine:    The SQLAlchemy engine for the project database.

            expected(dict):    What it should hold, as returned by
                get_project_contents().

            own_values([str]):    See get_project_contents().

            reason(str):    Why it should, to report with the rows that
                differ.
        """

        contents = get_project_contents(engine, own_values)
        self.assertEqual(set(contents), set(expected))
        for table, rows in expected.items():
            # Only the rows on one side: the tables are too long to read a
            # difference of the whole of them.
            unexpected = Counter(contents[table]) - Counter(rows)
            missing = Counter(rows) - Counter(contents[table])
            self.assertFalse(
                unexpected or missing,
                f"{reason}, but its {table} rows differ.\nIt has:\n\t"
                + "\n\t".join(repr(row) for row in unexpected.elements())
                + "\nIt lacks:\n\t"
                + "\n\t".join(repr(row) for row in missing.elements()),
            )
            self.assertEqual(
                contents[table],
                rows,
                f"{reason}, but its {table} rows are in another order",
            )

    def test_every_release_ends_up_holding_what_a_new_project_does(self):
        """Each released project, once migrated, holds a new one's rows.

        The schema reaching today's is not enough: creating a project also
        fills it with the steps, their parameters, the processing sequence
        and the layout of its products, all of which change between
        releases. A project created by a release must end up with the same
        rows as one created today, whichever table they are in, except for
        the values it stores for ``stored_values_kept``, which must be
        exactly the ones it had.

        The revisions that change rows are then put through what else can
        happen to them, on the same project, since creating one is what
        takes the time:

        * run a second time, as after an upgrade interrupted before it was
          recorded, they add nothing;

        * downgraded, they leave what the project held before them, even
          of an exclusion rule a user has since set under a condition, which
          goes with the parameter it was a value of.
        """

        new_project = get_project_contents(
            self._initialize_project(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "new",
                released=False,
            ),
            self.stored_values_kept,
        )
        for ref in self.release_baselines:
            with self.subTest(release=ref):
                self.reset_backend()
                engine = self._initialize_project(
                    self._export_release(ref), f"from_{ref}"
                )
                stored_values = self._get_stored_values(engine)
                self._run_alembic(
                    engine, alembic_command.upgrade, self.last_schema_revision
                )
                before_data_revisions = get_project_contents(engine)

                self.migrate(engine)
                self._assert_holds(
                    engine,
                    new_project,
                    own_values=self.stored_values_kept,
                    reason=f"A migrated project from {ref} should hold what "
                    "a new one does, by a revision for each difference or, "
                    "for a stored value to keep, an entry in "
                    "stored_values_kept",
                )
                self.assertEqual(self._get_stored_values(engine), stored_values)

                self._run_alembic(
                    engine, alembic_command.stamp, self.last_schema_revision
                )
                self.migrate(engine)
                self._assert_holds(
                    engine,
                    new_project,
                    own_values=self.stored_values_kept,
                    reason="Repeating the revisions that change rows should "
                    "change nothing",
                )

                self._add_user_rule(engine)
                self._run_alembic(
                    engine, alembic_command.downgrade, self.last_schema_revision
                )
                self._assert_holds(
                    engine,
                    before_data_revisions,
                    reason="Downgrading the revisions that change rows "
                    "should leave what the project held before them",
                )


if __name__ == "__main__":
    unittest.main()
