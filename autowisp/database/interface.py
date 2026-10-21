"""Connect to the database and provide a session scope for queries."""

from os import path, makedirs
from contextlib import contextmanager

import platformdirs

from sqlalchemy import create_engine, event, inspect as sa_inspect
from sqlalchemy.pool import NullPool

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.migrate import (
    check_project_schema,
    create_project_schema,
    migrate_project,
)
from autowisp.database.initialize_data_reduction_structure import (
    get_default_data_reduction_structure,
)
from autowisp.database.initialize_light_curve_structure import (
    get_default_light_curve_structure,
)
from autowisp.database.sessions import start_engine_session

# get_project_home is re-exported: the project home is kept in project_paths
# so that the file classes can resolve paths without importing the database.
from autowisp.project_paths import (  # pylint: disable=unused-import
    get_project_home,
    set_project_home_path,
)
from autowisp.exceptions import DatabaseError

_db_engine = None

DB_URL_FNAME = "autowisp_db.url"  # pylint: disable=invalid-name
"""
Filename (relative to project home) where a non-SQLite connection URL is stored.

When a project is initialised with a centralised database (MySQL, MariaDB,
etc.) the connection URL is written to this file so that subsequent calls to
:func:`set_project_home` with only the directory path can reconnect without
requiring the caller to supply the URL again.
"""


def get_db_engine():
    """Return the database engine."""

    return _db_engine


@contextmanager
def start_db_session():
    """
    Yield a session on the current project's database, in a transaction.

    See :func:`~autowisp.database.sessions.start_engine_session`. Objects
    loaded in it stay readable after the block, as its callers have always
    been able to rely on.
    """

    with start_engine_session(_db_engine, expire_on_commit=False) as db_session:
        yield db_session


def initialize_cmdline_database():
    """Initialize the current database HDF5 structure tables."""

    create_project_schema(_db_engine)
    with start_db_session() as db_session:
        db_session.add(get_default_data_reduction_structure())
        db_session.add(get_default_light_curve_structure(db_session))


def enforce_sqlite_foreign_keys(engine):
    """Make SQLite check the foreign keys on every connection *engine* opens.

    SQLite checks them only on a connection that asks with ``PRAGMA
    foreign_keys = ON``. The setting is not stored in the database, and a
    project engine opens a new connection for every session (``NullPool``),
    so it has to be issued as each connection is made. A server
    (MySQL/MariaDB) always checks them, so other engines are left alone.

    Migrating turns it back off for its own connection: see
    :func:`autowisp.database.migrate.migrate_project`.

    Args:
        engine:    The SQLAlchemy engine for a project database, before
            anything has connected through it.

    Returns:
        None
    """

    if engine.dialect.name != "sqlite":
        return

    def enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    event.listen(engine, "connect", enable_foreign_keys)


def use_utc_sessions(engine):
    """Make a server present and take timestamps in UTC, as SQLite does.

    A server stores ``TIMESTAMP`` values in UTC, but converts them to and
    from the session's time zone, which is the server's own unless set.
    SQLite's ``CURRENT_TIMESTAMP`` is UTC, so without this a project copied
    between the two would mix local and UTC times. Like the foreign keys on
    SQLite, it is a setting of the session, issued as each connection is
    made.

    Args:
        engine:    The SQLAlchemy engine for a project database, before
            anything has connected through it.

    Returns:
        None
    """

    if engine.dialect.name not in ("mysql", "mariadb"):
        return

    def set_utc(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET time_zone = '+00:00'")
        cursor.close()

    event.listen(engine, "connect", set_utc)


def get_sqlite_url(project_home):
    """Return the URL of the SQLite database of the project at the home."""

    db_path = path.abspath(path.join(project_home, "autowisp.db"))
    return f"sqlite:///{db_path}?timeout=600&uri=true"


def create_project_engine(db_url):
    """
    Return an engine for a project database, set up as every one must be.

    The one place a project engine is made, so that every connection to a
    project, whichever code opens it, checks foreign keys on SQLite and uses
    UTC on a server.

    Args:
        db_url(str):    SQLAlchemy URL of the project database.

    Returns:
        sqlalchemy.engine.Engine:    The engine, not yet connected.
    """

    engine_kwargs = {
        "echo": False,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }
    if db_url.startswith("sqlite"):
        engine_kwargs["poolclass"] = NullPool
        engine_kwargs["connect_args"] = {"timeout": 600}

    engine = create_engine(db_url, **engine_kwargs)
    enforce_sqlite_foreign_keys(engine)
    use_utc_sessions(engine)
    return engine


def set_project_home(
    project_home,
    db_url=None,
    new_project=False,
    *,
    migrate=False,
    assume_backed_up=False,
):
    """
    Set the database engine and session for the given project home.

    On first use with a non-SQLite ``db_url`` the URL is persisted to
    ``<project_home>/autowisp_db.url`` so that subsequent calls with only
    ``project_home`` reconnect to the same database automatically.

    Args:
        project_home:
            Directory used as the project home. For SQLite (the default), the
            database file ``autowisp.db`` is created here. For centralised
            databases the directory is still used for other project files (HDF5
            products, etc.). Pass ``None`` to use the platform-appropriate user
            data directory.

        db_url:
            SQLAlchemy connection URL. When omitted (or ``None``) the function
            first checks for a previously saved URL in
            ``<project_home>/autowisp_db.url``; if none is found it falls back
            to an SQLite database in ``project_home``:
            ``sqlite:///<project_home>/autowisp.db?timeout=100&uri=true``.
            To connect to a centralised server pass the full URL, e.g.:
            ``"mysql+pymysql://user:password@host:3306/dbname"``
            ``"mariadb+pymysql://user:password@host:3306/dbname"``
            Passing an explicit URL raises an error if a saved URL is found.

        new_project:
            Pass ``True`` when this call is creating a project rather than
            opening an existing one. The target database must then contain
            none of the AutoWISP tables, since project creation goes on to
            drop and recreate them (see ``initialize_database``) -- which
            on a centralised server would silently destroy whichever
            project already lives in that database. Independent of
            backend: it equally catches an SQLite project home that
            already holds a database.

        migrate:
            Bring the schema up to date rather than merely requiring that it
            already is. Off by default because every process opening a
            project calls this, including every pipeline worker, and
            concurrent schema changes from dozens of workers is exactly what
            must not happen. Set it only where a single process is known to
            be in charge: the ``wisp-migrate`` command, the browser
            interface selecting a project, and the main process of a
            pipeline run.

        assume_backed_up:
            Passed through to :func:`migrate_project`; only meaningful with
            ``migrate``. Required to migrate a server database, which cannot
            be copied aside automatically.

    Returns:
        dict or None:    The result of :func:`migrate_project` when
            ``migrate`` is set and the project already existed, otherwise
            None.

    Raises:
        DatabaseError:  If ``new_project`` is set and the target database
            already contains AutoWISP tables. Nothing is written or
            dropped in that case: the URL file is persisted only after
            this check passes.
    """

    global _db_engine  # pylint: disable=global-statement
    # print(f"Setting project home to {project_home!r}")
    if _db_engine is not None:
        _db_engine.dispose()

    if project_home is None:
        project_home = platformdirs.user_data_dir("autowisp")
    else:
        assert path.isdir(
            project_home
        ), f"Project home {project_home!r} is not a directory."

    # Ensure directory exists
    makedirs(project_home, exist_ok=True)
    project_home = path.abspath(project_home)

    url_file = path.join(project_home, DB_URL_FNAME)

    persist_db_url = None
    if db_url is not None:
        assert not path.exists(url_file), (
            f"Attempting to set a new db_url in {project_home!r} which already"
            f" contains {url_file!r}"
        )
        # Persisted (further down) only once the target database has been
        # accepted, so a rejected one leaves no half-created project whose
        # URL file points at somebody else's data.
        persist_db_url = db_url
    elif path.exists(url_file):
        with open(url_file, encoding="utf-8") as fobj:
            db_url = fobj.read().strip()

    if db_url is None:
        db_url = get_sqlite_url(project_home)

    # Built into a local first: an engine rejected below must not become
    # the module-wide one, or the caller would be left connected to the
    # database it was just refused.
    engine = create_project_engine(db_url)
    existing_tables = set(sa_inspect(engine).get_table_names())
    already_present = set(DataModelBase.metadata.tables) & existing_tables
    if new_project and already_present:
        # The URL can carry a password, so report the sanitised form.
        target = engine.url.render_as_string(hide_password=True)
        engine.dispose()
        raise DatabaseError(
            f"Refusing to create a new project in {project_home!r}: its "
            f"database ({target}) already contains "
            f"{len(already_present)} AutoWISP table(s), including "
            f"{', '.join(sorted(already_present)[:5])}. Creating the "
            "project would drop them, destroying whatever project they "
            "belong to. Point the new project at an empty database, or "
            "delete the existing project first."
        )

    _db_engine = engine
    set_project_home_path(project_home)

    if persist_db_url is not None:
        # Lets future calls with only project_home reconnect here.
        with open(url_file, "w", encoding="utf-8") as fobj:
            fobj.write(persist_db_url)

    if not already_present:
        initialize_cmdline_database()
        return None

    if migrate:
        return migrate_project(
            _db_engine, project_home, assume_backed_up=assume_backed_up
        )

    # Read-only on purpose: every process opening a project runs this,
    # including every pipeline worker, so it must never issue DDL. Migrating
    # is migrate_project()'s job -- see autowisp.database.migrate.
    check_project_schema(_db_engine)
    return None
