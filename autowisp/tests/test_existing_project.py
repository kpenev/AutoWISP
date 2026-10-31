"""Tests of adding a project moved or copied by hand."""

import contextlib
import os

from sqlalchemy import text
from sqlalchemy.engine import make_url

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Target

# pylint: enable=no-name-in-module
from autowisp.database.existing_project import (
    check_existing_project,
    clone_server_database,
)
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.pending_projects import (
    add_pending,
    get_pending_fname,
    list_pending,
    run_clone,
)
from autowisp.database.interface import (
    DB_URL_FNAME,
    create_project_engine,
    get_sqlite_url,
    read_db_url,
)
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import (
    DatabaseError,
    ExperimentError,
    ProjectStateError,
)
from autowisp.project_experiments import ProjectExperiments
from autowisp.tests import empty_server_database, server_test_url
from autowisp.tests.database_fixtures import (
    ProjectCopyCase,
    stamp_previous_revision,
)


class TestExistingProject(ProjectCopyCase):
    """Checking a project before adding it, and cloning its server database."""

    # A copy keeps its database in its home, as most projects do: the check
    # tells those from a project naming its database by URL, which the tests
    # make by hand (see _make_server_project).
    _original_on_server = False

    def _name_database(self, db_url, home=None):
        """Make a home name the database at the URL, as a server's does.

        Args:
            db_url(str):    The database's URL.

            home(str):    The home, None for a new empty directory.

        Returns:
            str:    The home.
        """

        if home is None:
            home = self._make_dir()
        with open(
            os.path.join(home, DB_URL_FNAME), "w", encoding="utf-8"
        ) as url_file:
            url_file.write(db_url)
        return home

    def _move_database_out(self, home):
        """Move the home's database out, naming it by URL, and return that.

        An SQLite file outside the home stands in for a server's database,
        when there is no server under test.
        """

        db_url = get_sqlite_url(self._make_dir())
        os.replace(os.path.join(home, "autowisp.db"), make_url(db_url).database)
        self._name_database(db_url, home)
        return db_url

    @staticmethod
    def _get_server_url(suffix):
        """
        Return the URL of a database on the server under test, None if none.

        Args:
            suffix(str):    Added to the name of the database under test, to
                name the one wanted.
        """

        server_url = server_test_url()
        if server_url is None:
            return None
        server_url = make_url(server_url)
        return server_url.set(
            database=server_url.database + suffix
        ).render_as_string(hide_password=False)

    def _make_other_database(self):
        """
        Return the URL of an empty database, not the one under test.

        On the server under test, if any, it is the database named after
        that with ``_other`` added, created if missing, which the account
        testing must be allowed to do or to use it. Else an SQLite file
        outside any home.
        """

        other_url = self._get_server_url("_other")
        if other_url is None:
            return get_sqlite_url(self._make_dir())
        engine = create_project_engine(server_test_url())
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE DATABASE IF NOT EXISTS "
                        f"`{make_url(other_url).database}`"
                    )
                )
        finally:
            engine.dispose()
        empty_server_database(other_url)
        return other_url

    def _make_server_project(self, with_experiments):
        """
        Return the home of a project naming its database by URL, and an engine.

        The database is on the server under test, else in an SQLite file
        outside the home.

        Args:
            with_experiments(bool):    Whether to give the project three
                experiments: ``current``, copied from the project as it is;
                ``changed``, still current although the project changed
                after it was copied, despite the lock; and ``stale``.
        """

        home = self._copy_project()[1]
        db_url = server_test_url()
        if db_url is None:
            db_url = self._move_database_out(home)
        else:
            self._move_database_to_server(home, db_url)
        source_engine = create_project_engine(db_url)
        self.addCleanup(source_engine.dispose)
        if with_experiments:
            experiments = ProjectExperiments(source_engine, home)
            experiments.create("stale")
            experiments.unlock()
            experiments.create("changed")
            with start_engine_session(source_engine) as db_session:
                db_session.add(Target(name="moved"))
            experiments.create("current")
        return home, source_engine

    def test_check_existing_project(self):
        """Each directory is accepted, or refused for its own reason."""

        registered_home = self._copy_project()[1]
        shared_home = self._make_server_project(False)[0]
        shared_url = read_db_url(shared_home)
        registered_homes = [registered_home, shared_home]

        older_engine, older_home = self._copy_project()
        stamp_previous_revision(older_engine)
        newer_engine, newer_home = self._copy_project()
        with newer_engine.begin() as connection:
            connection.execute(
                text("UPDATE alembic_version SET version_num = '9999_later'")
            )

        # Each accepted with its description, but its home: the path given,
        # resolved.
        candidates = {
            "current": (
                self._copy_project()[1],
                {"db_url": None, "sharing_homes": []},
            ),
            # Migrated once selected.
            "older": (older_home, {"db_url": None, "sharing_homes": []}),
            "on a registered project's database": (
                self._name_database(shared_url),
                {
                    "db_url": make_url(shared_url).render_as_string(
                        hide_password=True
                    ),
                    "sharing_homes": [shared_home],
                },
            ),
            "without a database": (self._make_dir(), DatabaseError),
            "newer": (newer_home, DatabaseError),
            "on an empty database": (
                self._name_database(self._make_other_database()),
                DatabaseError,
            ),
            "on a database that cannot be opened": (
                self._name_database(
                    self._get_server_url("_missing")
                    or get_sqlite_url(os.path.join(self._make_dir(), "missing"))
                ),
                DatabaseError,
            ),
            "registered, by another path": (
                os.path.join(registered_home, "sub", os.pardir),
                ProjectStateError,
            ),
        }
        if os.name == "nt":
            candidates["registered, in another case"] = (
                registered_home.swapcase(),
                ProjectStateError,
            )

        for name, (path, outcome) in candidates.items():
            with self.subTest(name):
                if isinstance(outcome, dict):
                    self.assertEqual(
                        check_existing_project(path, registered_homes),
                        {"home": os.path.realpath(path), **outcome},
                    )
                else:
                    with self.assertRaises(outcome):
                        check_existing_project(path, registered_homes)

    def test_clone_server_database(self):
        """A clone takes the database's place, or the home is left alone.

        The clone keeps the experiments, and those current stay mergeable,
        unless the project changed after they were copied.
        """

        scenarios = {
            # Over a clone an earlier one left, ignoring a run known to have
            # ended.
            "into SQLite": {"leftover": True, "run": "ignored"},
            "into another database": {"target": "new"},
            "while a run is going": {
                "run": "going",
                "outcome": ExperimentError,
            },
            "over a database in the home": {
                "in_home": True,
                "outcome": DatabaseError,
            },
            "into a database in use": {
                "target": "source",
                "outcome": DatabaseError,
            },
        }

        for name, scenario in scenarios.items():
            with self.subTest(name):
                home, source_engine = self._make_server_project(
                    "outcome" not in scenario
                )
                for fname, wanted in (
                    ("autowisp.db-clone", "leftover"),
                    ("autowisp.db", "in_home"),
                ):
                    if scenario.get(wanted):
                        with open(os.path.join(home, fname), "wb"):
                            pass
                target_url = None
                if scenario.get("target") == "new":
                    target_url = self._make_other_database()
                elif scenario.get("target") == "source":
                    target_url = read_db_url(home)
                # Of an operation on experiments, the only runs an original
                # they lock can have going, which change nothing a merge
                # would lose.
                ignored_runs = (
                    self._add_live_run(
                        source_engine,
                        scenario["run"] == "ignored",
                        "delete experiment",
                    )
                    if "run" in scenario
                    else set()
                )
                fnames = set(os.listdir(home))
                contents = self._read_tree(home)

                if "outcome" in scenario:
                    with self.assertRaises(scenario["outcome"]):
                        clone_server_database(home, target_url, ignored_runs)
                    self.assertEqual(set(os.listdir(home)), fnames)
                    self.assertEqual(self._read_tree(home), contents)
                    continue

                stale = clone_server_database(home, target_url, ignored_runs)

                self.assertEqual(stale, ["changed"])
                self.assertEqual(read_db_url(home), target_url)
                fnames -= {DB_URL_FNAME, "autowisp.db-clone"}
                fnames.add(
                    "autowisp.db" if target_url is None else DB_URL_FNAME
                )
                self.assertEqual(set(os.listdir(home)), fnames)
                self.assertEqual(self._read_tree(home), contents)
                clone_engine = create_project_engine(
                    target_url or get_sqlite_url(home)
                )
                self.addCleanup(clone_engine.dispose)
                clone_experiments = ExperimentRecords(clone_engine, home)
                self.assertEqual(
                    {
                        experiment.name: experiment.state
                        for experiment in clone_experiments.get_experiments()
                    },
                    {
                        "changed": "stale",
                        "current": "current",
                        "stale": "stale",
                    },
                )
                self.assertEqual(clone_experiments.mark_changed_stale(), [])

        with self.assertRaises(DatabaseError):
            clone_server_database(self._copy_project()[1])

    def test_clone_from_record(self):
        """A clone run from its record writes there how it ended."""

        pending_dir = self._make_dir()
        homes = {}
        # One at a time: on a server under test, each project made there
        # replaces the one database, so each is cloned before the next.
        for name in ["cloned", "refused"]:
            home = homes[name] = self._make_server_project(name == "cloned")[0]
            if name == "refused":
                # An SQLite clone would replace a database there already.
                with open(os.path.join(home, "autowisp.db"), "wb"):
                    pass
            fname = get_pending_fname(
                add_pending(name, "", home, None, pending_dir), pending_dir
            )
            with (
                self.assertLogs("autowisp.database.pending_projects", "ERROR")
                if name == "refused"
                else contextlib.nullcontext()
            ):
                run_clone(fname)

        outcomes = {
            entry["name"]: (
                entry["state"],
                entry.get("stale"),
                bool(entry.get("message")),
            )
            for entry in list_pending(pending_dir)
        }
        self.assertEqual(
            outcomes,
            {
                "cloned": ("done", ["changed"], False),
                "refused": ("failed", None, True),
            },
        )
        self.assertIsNone(read_db_url(homes["cloned"]))
