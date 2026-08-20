"""Apply and check Alembic migrations for a project database.

Alembic supplies the revision graph, the ``alembic_version`` table, stamping
and SQLite batch mode. What it does not supply, and what this module is
mostly about, is *when and where* migrations run:

- :func:`check_project_schema` is read-only and is what
  :func:`autowisp.database.interface.set_project_home` calls. Every process
  opening a project runs it, **including every pipeline worker**, so it must
  never issue DDL -- dozens of workers racing to ``CREATE INDEX`` is the
  failure this split exists to prevent. A database behind head fails loudly
  here instead of misbehaving later.

- :func:`migrate_project` is the only thing that mutates, and is called from
  the browser interface when a project is selected, from ``wisp-migrate``,
  and once from the main process of a pipeline run.

The revision chain is kept strictly linear (a test asserts it), so the
singular Alembic APIs are used throughout: ``get_current_revision`` and
``get_current_head`` rather than their plural counterparts. Both raise if a
fork ever reaches them, which is the behaviour wanted -- a fork is a mistake
to surface, not a state to support.

See ``project_db_migrations_plan.md`` for the reasoning.
"""

import os
import shutil
from contextlib import contextmanager
from functools import lru_cache

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
from sqlalchemy import event, inspect as sa_inspect, text

from autowisp.database.data_model.base import DataModelBase
from autowisp.exceptions import DatabaseError

BASELINE_REVISION = "0001_baseline"  # pylint: disable=invalid-name
"""Revision marking the schema :func:`_apply_additive_migrations` produces."""

_LOCK_NAME = "autowisp_migrate"  # pylint: disable=invalid-name
_LOCK_TIMEOUT = 600  # pylint: disable=invalid-name


def _apply_additive_migrations(connection):
    """Bring a project database that predates Alembic up to the 1.8.1 schema.

    .. note::

        **This is not where new schema changes go.** It is frozen at its
        1.8.1 contents and is called from exactly one place --
        :func:`migrate_project`, on a database carrying no
        ``alembic_version`` table -- to reach the state that
        ``0001_baseline`` stamps. Everything after 1.8.1 is an Alembic
        revision under ``migrations/versions``.

    It survives rather than being converted into revisions because its
    ``create_all`` half cannot be: that is a catch-all for any table added at
    any point since a project was initialised, and the history of when each
    table appeared is not recorded anywhere. Expressing it as revisions would
    mean either reconstructing that history by git archaeology, or calling
    ``create_all`` from inside a revision -- which would create *today's*
    tables rather than the ones contemporary with the revision, and so would
    silently change meaning every time a model is added.

    Idempotent, so running it against an already-current database does
    nothing.

    Takes a connection rather than an engine so it runs inside the caller's
    migration lock. Opening a second connection here would block against
    that lock instead of cooperating with it.

    Args:
        connection:    An open connection to the project database, inside a
            transaction held by :func:`_locked_connection`.

    Returns:
        None
    """

    # Create any tables added since the project was initialized (idempotent;
    # existing tables and data are left as-is).
    DataModelBase.metadata.create_all(connection)

    # Add any nullable columns added to tables that already existed.
    additive_columns = [
        ("pipeline_run", "code_version", "VARCHAR(1000)"),
        ("error", "resolved", "TIMESTAMP"),
    ]
    inspector = sa_inspect(connection)
    present_tables = set(inspector.get_table_names())
    for table, column, sql_type in additive_columns:
        if table not in present_tables:
            continue
        columns = {col["name"] for col in inspector.get_columns(table)}
        if column in columns:
            continue
        # table/column/type come from the trusted list above, not from user
        # input, so the f-string is safe here.
        connection.execute(
            text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        )


def _alembic_config():
    """Return the shipped Alembic configuration."""

    return Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))


@lru_cache(maxsize=1)
def get_head_revision():
    """Return the newest revision shipped with the installed code.

    Cached: this is a property of the code, not of any database, and
    :func:`check_project_schema` runs in every worker process on every
    project open. Without the cache each of those would re-walk and re-import
    ``versions/``.
    """

    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def get_project_revision(engine):
    """Return the revision a project database is stamped at, or None.

    None means the database has no ``alembic_version`` table -- it predates
    Alembic and needs baselining.
    """

    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def _is_known_revision(revision):
    """Whether *revision* is one the installed code ships.

    False means the database was migrated by a newer AutoWISP than this one,
    which needs different advice: upgrading the code, not running a
    migration this install does not have.
    """

    try:
        ScriptDirectory.from_config(_alembic_config()).get_revision(revision)
    except CommandError:
        return False
    return True


def check_project_schema(engine):
    """Raise unless the project database is at the current head revision.

    Read-only by design -- see the module docstring. This is what
    :func:`autowisp.database.interface.set_project_home` calls, so it runs in
    every pipeline worker and must never attempt DDL.

    Args:
        engine:    The SQLAlchemy engine for the project database.

    Raises:
        DatabaseError:    If the database is behind, ahead of, or otherwise
            disagrees with the installed migration scripts.
    """

    current = get_project_revision(engine)
    head = get_head_revision()
    if current == head:
        return

    if current is not None and not _is_known_revision(current):
        raise DatabaseError(
            f"The project database is at revision {current}, which this "
            "version of AutoWISP does not know about: it was migrated by a "
            "newer version. Upgrade AutoWISP -- migrating will not help, and "
            "this install cannot safely write to that schema."
        )

    if current is None:
        state = "it predates the migration system and has not been baselined"
    else:
        state = f"it is at {current}, but this version expects {head}"

    raise DatabaseError(
        f"The project database needs migrating: {state}. Run "
        "'wisp-migrate <project_home>', or open the project in the browser "
        "interface, which migrates on selection."
    )


@contextmanager
def _sqlite_immediate(engine):
    """Make this engine's transactions take SQLite's write lock up front.

    pysqlite defers ``BEGIN`` until a DML statement, which would let two
    migrators both read the current revision before either writes. ``BEGIN
    IMMEDIATE`` takes the write lock when the transaction opens instead, so
    the second blocks (on the engine's busy timeout) rather than re-running a
    migration that is already being applied.

    Turning off the driver's implicit transaction handling is the documented
    SQLAlchemy recipe for controlling SQLite's ``BEGIN``.
    """

    def disable_implicit_begin(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    def begin_immediate(connection):
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    event.listen(engine, "connect", disable_implicit_begin)
    event.listen(engine, "begin", begin_immediate)
    try:
        yield
    finally:
        event.remove(engine, "begin", begin_immediate)
        event.remove(engine, "connect", disable_implicit_begin)


@contextmanager
def _locked_connection(engine):
    """Yield a connection in a transaction, with the migration lock held.

    Alembic has no locking of its own -- two concurrent ``upgrade`` calls
    both read the same revision and both run it -- and concurrency is
    reachable here: a centralised MySQL project database serves several users
    at once, and starting a pipeline run while the browser interface is open
    is ordinary.
    """

    if engine.dialect.name == "sqlite":
        with _sqlite_immediate(engine), engine.begin() as connection:
            yield connection
        return

    # A session-scoped lock, not a transaction-scoped one such as
    # SELECT ... FOR UPDATE: DDL on MySQL commits implicitly, which would drop
    # a transaction-scoped lock partway through the first revision.
    with engine.begin() as connection:
        acquired = connection.exec_driver_sql(
            f"SELECT GET_LOCK('{_LOCK_NAME}', {_LOCK_TIMEOUT})"
        ).scalar()
        if acquired != 1:
            raise DatabaseError(
                f"Timed out after {_LOCK_TIMEOUT}s waiting for another "
                "process to finish migrating this database."
            )
        try:
            yield connection
        finally:
            connection.exec_driver_sql(f"SELECT RELEASE_LOCK('{_LOCK_NAME}')")


def _backup_sqlite(engine, current):
    """Copy an SQLite project database aside before migrating it.

    The copy is named for the revision it is a snapshot *of*, so
    ``autowisp.db.pre-0001_baseline`` is the database as it stood at
    ``0001_baseline``.

    Returns the backup path, or None for a non-SQLite database, where backups
    are the administrator's job (see :func:`migrate_project`).
    """

    if engine.dialect.name != "sqlite":
        return None
    db_path = engine.url.database
    if not db_path or not os.path.exists(db_path):
        return None
    backup_path = f"{db_path}.pre-{current or 'baseline'}"
    shutil.copy2(db_path, backup_path)
    return backup_path


def _stamp(connection, revision):
    """Record *revision* as applied, on the caller's connection."""

    config = _alembic_config()
    config.attributes["connection"] = connection
    command.stamp(config, revision)


def create_project_schema(engine):
    """Build the current schema and record it as being at head.

    The two halves belong together: ``create_all`` produces today's tables
    directly, so the revisions describing how to get there must be marked
    applied rather than run. Creating the schema without stamping would
    leave the database looking like an un-baselined legacy project, and
    :func:`migrate_project` would then try to apply revisions it already
    satisfies.

    Pairing them in one function is what stops those two steps drifting
    apart across the several places a project database gets created.
    """

    DataModelBase.metadata.create_all(engine)
    with engine.begin() as connection:
        _stamp(connection, "head")


def migrate_project(engine, *, assume_backed_up=False):
    """Bring a project database up to the current head revision.

    The only function here that mutates. Handles three cases:

    - **stamped** -- upgrade to head;
    - **AutoWISP tables but no** ``alembic_version`` -- predates Alembic, so
      reach a known state with :func:`_apply_additive_migrations`, stamp
      ``0001_baseline``, then upgrade;
    - **empty** -- create the schema and stamp head.

    Args:
        engine:    The SQLAlchemy engine for the project database.

        assume_backed_up(bool):    Required to migrate a server database.
            SQLite databases are copied aside automatically, but a
            MySQL/MariaDB one cannot be, and MySQL cannot roll DDL back, so
            proceeding without a backup has to be a deliberate choice.

    Returns:
        dict:    ``{"from": ..., "to": ..., "backup": ...}``. ``from`` equals
        ``to`` if the database was already current. Which revisions ran in
        between is Alembic's business, not something worth recomputing here.

    Raises:
        DatabaseError:    If a server database is migrated without
            ``assume_backed_up``, or the migration lock cannot be taken.
    """

    current = get_project_revision(engine)
    head = get_head_revision()
    result = {"from": current, "to": head, "backup": None}
    if current == head:
        return result

    if engine.dialect.name != "sqlite" and not assume_backed_up:
        raise DatabaseError(
            "Refusing to migrate a centralised database without confirmation "
            "that it has been backed up: it cannot be copied aside "
            "automatically, and MySQL commits DDL implicitly, so a failed "
            "migration cannot be rolled back. Migrating a shared database is "
            "also not something to do as a side effect of another command. "
            "Back it up, then run "
            "'wisp-migrate <project_home> --assume-backed-up'."
        )

    tables = set(sa_inspect(engine).get_table_names())
    if not set(DataModelBase.metadata.tables) & tables:
        # Nothing to preserve, so no backup: this is project creation.
        create_project_schema(engine)
        return result

    config = _alembic_config()
    with _locked_connection(engine) as connection:
        # Re-read now that the lock is held. The check above was an unlocked
        # fast path, and another migrator may have finished while this one
        # waited -- without this it would redo work already done, and stamp
        # the baseline over a newer revision on the way.
        current = MigrationContext.configure(connection).get_current_revision()
        result["from"] = current
        if current == head:
            return result

        # Inside the lock, so the copy cannot catch a half-applied migration.
        result["backup"] = _backup_sqlite(engine, current)

        if current is None:
            # Predates Alembic: reach a known schema, then record that it is
            # the one 0001_baseline describes. Both run on this connection;
            # opening another would block against the lock we are holding.
            _apply_additive_migrations(connection)
            _stamp(connection, BASELINE_REVISION)
            result["from"] = BASELINE_REVISION

        config.attributes["connection"] = connection
        command.upgrade(config, "head")

    return result
