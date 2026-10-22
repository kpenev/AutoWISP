"""Add a project moved or copied by hand: check it, and clone its database.

Nothing in a project's home ties it to where it lies, its paths being stored
relative to the home, so a project moved or copied by hand is the project it
was. What the browser interface needs to register one, besides the record
itself, is here, free of Django: the checks, and cloning the server database
a copied home still names, copying the home having copied only its URL.
"""

import os

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from autowisp.database.data_model.base import DataModelBase
from autowisp.database.interface import (
    DB_URL_FNAME,
    create_project_engine,
    get_sqlite_url,
    read_db_url,
)
from autowisp.database.migrate import check_known_revision
from autowisp.exceptions import DatabaseError, ProjectStateError
from autowisp.project_experiments import ProjectExperiments


def _get_database_key(db_url):
    """
    Return what tells a server database apart: backend, host, port and name.

    The user and the driver are left out, being other ways into the same
    database. Other spellings of one server, e.g. its name and its address,
    still tell apart.
    """

    url = make_url(db_url)
    return (url.get_backend_name(), url.host, url.port, url.database)


def check_existing_project(path, registered_homes=()):
    """
    Check that a directory holds a project that can be added; describe it.

    It must hold ``autowisp.db`` or ``autowisp_db.url``, naming a database
    that holds an AutoWISP project at a revision this version can migrate
    to: adding does not migrate, but selecting the project does. It must not
    be registered already, under any path leading to it.

    Args:
        path(str):    The directory.

        registered_homes:    The homes of the projects registered already.

    Returns:
        dict:    The project's home, resolved, under ``"home"``; its database
        URL with the password hidden, under ``"db_url"``, None for SQLite;
        and under ``"sharing_homes"``, those of ``registered_homes`` whose
        project names the same server database, as given. Adding the project
        on that database would have two projects write it.

    Raises:
        DatabaseError:    If the directory holds no project database, it
            cannot be opened, holds no AutoWISP tables, or is at a revision
            a newer AutoWISP migrated it to.

        ProjectStateError:    If the project is registered already.
    """

    home = os.path.realpath(path)
    if not any(
        os.path.isfile(os.path.join(home, fname))
        for fname in ("autowisp.db", DB_URL_FNAME)
    ):
        raise DatabaseError(
            f"{path!r} holds no AutoWISP project: it has neither autowisp.db "
            f"nor {DB_URL_FNAME}."
        )
    for registered in registered_homes:
        if os.path.normcase(os.path.realpath(registered)) == os.path.normcase(
            home
        ):
            raise ProjectStateError(
                f"The project in {path!r} is registered already, as "
                f"{registered!r}."
            )

    db_url = read_db_url(home)
    engine = create_project_engine(db_url or get_sqlite_url(home))
    shown_url = engine.url.render_as_string(hide_password=True)
    try:
        try:
            tables = set(sa_inspect(engine).get_table_names())
        except SQLAlchemyError as error:
            raise DatabaseError(
                f"Cannot open the database of the project in {path!r}, "
                f"{shown_url}: {error}"
            ) from error
        if not set(DataModelBase.metadata.tables) & tables:
            raise DatabaseError(
                f"The database of the project in {path!r}, {shown_url}, "
                "holds no AutoWISP project."
            )
        check_known_revision(engine)
    finally:
        engine.dispose()

    sharing_homes = []
    if db_url is not None:
        database_key = _get_database_key(db_url)
        sharing_homes = [
            registered
            for registered in registered_homes
            if (registered_url := read_db_url(registered)) is not None
            and _get_database_key(registered_url) == database_key
        ]
    return {
        "home": home,
        "db_url": None if db_url is None else shown_url,
        "sharing_homes": sharing_homes,
    }


def _write_db_url(project_home, db_url):
    """Name the server database in the home, replacing the file at once."""

    url_fname = os.path.join(project_home, DB_URL_FNAME)
    with open(url_fname + ".new", "w", encoding="utf-8") as url_file:
        url_file.write(db_url)
    os.replace(url_fname + ".new", url_fname)


def clone_server_database(project_home, target_url=None, ignored_runs=()):
    """
    Give a project a database of its own, cloned from the one it names.

    Copying a server project's home copies the URL of its database, not the
    database, leaving two projects on one database. This clones it into
    SQLite in the home, removing ``autowisp_db.url`` as a project without a
    server has none, or into the empty server database at ``target_url``,
    naming that in ``autowisp_db.url``. Keeping the database as it is needs
    nothing done.

    The clone is the whole project, its record of its experiments included,
    since the copied home has their files. Those still current stay
    mergeable (see :meth:`.ProjectExperiments.clone_original`).

    The home names the clone only once it is complete: an SQLite clone is
    built under another name, which the next clone overwrites, and moved into
    place; a server clone is named once filled. Until then the home names the
    database it did.

    Args:
        project_home(str):    The project's home.

        target_url(str):    The URL of the empty server database to clone
            into, None to clone into SQLite.

        ignored_runs:    The ids of the runs in the database known to have
            ended without recording it (see
            :meth:`.ExperimentRecords.refuse_active_runs`).

    Returns:
        [str]:    The names of the experiments marked stale, the original
        having changed since they were copied.

    Raises:
        DatabaseError:    If the home names no server database, an SQLite
            clone would replace an ``autowisp.db`` already there, the target
            server database is not empty, or the database is not at this
            AutoWISP's revision, at which the clone is created.

        ExperimentError:    If a run is going in the database: the home was
            copied earlier, and a run going since would leave its files and
            the clone out of step.
    """

    source_url = read_db_url(project_home)
    if source_url is None:
        raise DatabaseError(
            f"The project in {project_home!r} names no server database: its "
            "database is its own already."
        )
    if target_url is None:
        db_fname = os.path.join(project_home, "autowisp.db")
        if os.path.exists(db_fname):
            raise DatabaseError(
                f"Refusing to clone the database of the project in "
                f"{project_home!r} into {db_fname!r}, which exists already. "
                "Remove it, or clone into a server database."
            )
        clone_fname = os.path.join(project_home, "autowisp.db-clone")
        if os.path.exists(clone_fname):
            # Left by a clone that failed or was interrupted.
            os.remove(clone_fname)

    source_engine = create_project_engine(source_url)
    target_engine = create_project_engine(
        target_url or get_sqlite_url(project_home, "autowisp.db-clone")
    )
    try:
        experiments = ProjectExperiments(source_engine, project_home)
        experiments.refuse_active_runs(source_engine, ignored_runs)
        stale = experiments.clone_original(target_engine)
    finally:
        source_engine.dispose()
        # Before the file is moved, which Windows refuses while it is open.
        target_engine.dispose()

    if target_url is None:
        os.replace(clone_fname, db_fname)
        os.remove(os.path.join(project_home, DB_URL_FNAME))
    else:
        _write_db_url(project_home, target_url)
    return stale
