"""Connect to the database and provide a session scope for queries."""

from os import path, makedirs
from contextlib import contextmanager

import platformdirs

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.pool import NullPool

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.frozen_row import FrozenRow
from autowisp.database.initialize_data_reduction_structure import (
    get_default_data_reduction_structure,
)
from autowisp.database.initialize_light_curve_structure import (
    get_default_light_curve_structure,
)

_db_engine = None

# pylint false positive - Session is actually a class name.
# pylint: disable=invalid-name
_Session = None  # sessionmaker(db_engine, expire_on_commit=False)
# pylint: enable=invalid-name

_project_home = None

DB_URL_FNAME = "autowisp_db.url"  # pylint: disable=invalid-name
"""
Filename (relative to project home) where a non-SQLite connection URL is stored.

When a project is initialised with a centralised database (MySQL, MariaDB,
etc.) the connection URL is written to this file so that subsequent calls to
:func:`set_project_home` with only the directory path can reconnect without
requiring the caller to supply the URL again.
"""


def apply_additive_migrations(engine):
    """Bring an existing project database up to the current schema.

    A minimal, idempotent stand-in for a migration framework (the project
    has none), covering the additive changes that are safe to apply
    automatically on connect:

    - **New tables** (e.g. ``error``) are created via ``create_all``,
      which leaves existing tables and their data untouched. This is how a
      table added after a project was initialized reaches that project.
    - **New nullable columns** on existing tables are added with
      ``ALTER TABLE ... ADD COLUMN``. Rows that predate a column keep
      NULL -- e.g. ``pipeline_run`` rows from before ``code_version``
      existed have no recorded code version, which is correct (it is
      genuinely unknown).

    Freshly initialized databases get everything from ``create_all``
    already, so this is effectively a no-op for them. When the pipeline
    runs, the main process opens the project first, so by the time workers
    connect every table exists and their ``create_all`` does no DDL.

    Args:
        engine:    The SQLAlchemy engine for the project database.

    Returns:
        None
    """

    # Create any tables added since the project was initialized (idempotent;
    # existing tables and data are left as-is).
    DataModelBase.metadata.create_all(engine)

    # Add any nullable columns added to tables that already existed.
    additive_columns = [
        ("pipeline_run", "code_version", "VARCHAR(1000)"),
        ("error", "resolved", "TIMESTAMP"),
    ]
    inspector = sa_inspect(engine)
    present_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table, column, sql_type in additive_columns:
            if table not in present_tables:
                continue
            columns = {col["name"] for col in inspector.get_columns(table)}
            if column in columns:
                continue
            # table/column/type come from the trusted list above, not from
            # user input, so the f-string is safe here.
            connection.execute(
                text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            )


def get_db_engine():
    """Return the database engine."""

    print(f"Returning engine {_db_engine!r}")
    return _db_engine


@contextmanager
def start_db_session():
    """Context manager to start a database session."""

    with _Session.begin() as db_session:  # pylint: disable=no-member
        yield db_session


def get_project_home():
    """Return the project home directory currently being used."""

    return _project_home


def snapshot_row(orm_obj, *, exclude=()):
    """Freeze all mapped columns of a live ORM instance into a FrozenRow.

    Must be called while ``orm_obj`` is still attached/loaded (i.e.
    inside the ``start_db_session()`` block that produced it), otherwise
    touching an expired column would raise ``DetachedInstanceError``.

    Args:
        orm_obj:    A SQLAlchemy ORM instance.

        exclude(Iterable[str]):    Column keys to omit (e.g. large or
            sensitive columns).

    Returns:
        FrozenRow:    Snapshot of the instance's column values, detached
            from the session and safe to pickle.
    """

    mapper = sa_inspect(orm_obj).mapper
    exclude = set(exclude)
    return FrozenRow(
        table=mapper.local_table.name,
        columns={
            attr.key: getattr(orm_obj, attr.key)
            for attr in mapper.column_attrs
            if attr.key not in exclude
        },
    )


def initialize_cmdline_database():
    """Initialize the current database HDF5 structure tables."""

    DataModelBase.metadata.create_all(_db_engine)
    with start_db_session() as db_session:
        db_session.add(get_default_data_reduction_structure())
        db_session.add(get_default_light_curve_structure(db_session))


def set_project_home(project_home, db_url=None):
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
    """

    global _db_engine, _Session, _project_home  # pylint: disable=global-statement
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
    _project_home = path.abspath(project_home)

    url_file = path.join(_project_home, DB_URL_FNAME)

    if db_url is not None:
        assert not path.exists(url_file), (
            f"Attempting to set a new db_url in {_project_home!r} which already"
            f" contains {url_file!r}"
        )
        # Persist the URL so future calls without db_url reconnect correctly.
        with open(url_file, "w", encoding="utf-8") as fobj:
            fobj.write(db_url)
    elif path.exists(url_file):
        with open(url_file, encoding="utf-8") as fobj:
            db_url = fobj.read().strip()

    engine_kwargs = {
        "echo": False,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }
    url_file = path.join(_project_home, DB_URL_FNAME)

    if db_url is None:
        db_path = path.join(_project_home, "autowisp.db")
        db_url = f"sqlite:///{path.abspath(db_path)}?timeout=600&uri=true"
    if db_url.startswith("sqlite"):
        engine_kwargs["poolclass"] = NullPool
        engine_kwargs["connect_args"] = {"timeout": 600}

    _db_engine = create_engine(db_url, **engine_kwargs)
    _Session = sessionmaker(_db_engine, expire_on_commit=False)

    existing_tables = set(sa_inspect(_db_engine).get_table_names())
    if not set(DataModelBase.metadata.tables).intersection(existing_tables):
        initialize_cmdline_database()
    else:
        apply_additive_migrations(_db_engine)
