"""Manage a project's experiments: copies processed on their own.

An experiment is a copy of a project, living at ``<home>/experiments/<name>``,
processed on its own and possibly merged back. The original records its
experiments in the ``experiment`` table (:class:`Experiment`); an
experiment's own database has no such table, and that absence is what makes
it an experiment.
"""

import hashlib
import json
from contextlib import contextmanager

from alembic.runtime.migration import MigrationContext
from sqlalchemy import MetaData, inspect as sa_inspect, select, update
from sqlalchemy.orm import Session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment

# pylint: enable=no-name-in-module
from autowisp.exceptions import ProjectLockedError


class ProjectExperiments:
    """
    The experiments of one project, and every operation on them.

    Every operation runs from the original, whose ``experiment`` table is the
    only record of its experiments. Each method opens its own connections, so
    that it decides when what it writes is committed.

    The original's fingerprint (:meth:`_get_fingerprint`) tells whether it
    has changed since an experiment was copied from it. It covers what a
    merge must not lose: a merge replaces the original with the experiment,
    so a change made to the original meanwhile counts only if losing it would
    matter. Everything is included except :attr:`_unhashed_tables` and
    :attr:`_unhashed_columns`, each with its reason, so that a table or
    column added later is covered by default. Kept deliberately:
    ``pipeline_run``, which the active-run checks read (a run recorded as
    finished in the original would come back unfinished, and block every
    later create and merge), and the ``notes`` columns, the user's
    annotations.
    """

    # The directory under the original's home that holds its experiments.
    _dirname = "experiments"

    # Until the copy is complete. Counts as current: it locks the original.
    _creating_state = "creating"
    # The original has not changed since the experiment was copied from it.
    _current_state = "current"
    # The original has changed since the experiment was copied from it.
    _stale_state = "stale"

    _unhashed_tables = {
        # Recorded as the fingerprint's revision instead.
        "alembic_version",
        # An error resolved or deleted in the original comes back after a
        # merge, to be resolved again; processing does not depend on it.
        "error",
        # The original's record of its experiments, which a merge does not
        # copy either.
        "experiment",
    }

    _unhashed_columns = {
        # In every table. Nothing decides anything by it (it is shown, and is
        # the fallback date of an error report), and a row restamped with the
        # same values, by saving unchanged values or by a migration, changes
        # nothing a merge would lose.
        "timestamp",
    }

    class _TableHasher:
        """
        Hash the rows of one table, fed in the order of ``_select_hashed_rows``.

        Each row is encoded as a compact JSON list of the values of its
        hashed columns, in the table's column order, one line per row. Values
        JSON has no form for (dates, decimals, bytes) are written as their
        ``str``. The encoding need only agree with itself: the hashes
        compared are always of the same original, on its own backend and
        driver.

        Separate from ``_get_fingerprint`` so that copying the project, which
        reads every row anyway, can hash them as it goes.
        """

        def __init__(self, table, unhashed_columns):
            """
            Start hashing the rows of the given table.

            Args:
                table(sqlalchemy.Table):    The table, reflected from the
                    database. The rows fed must have its columns, in its
                    order.

                unhashed_columns(set):    The names of the columns left out
                    of the hash.
            """

            self._positions = [
                position
                for position, column in enumerate(table.columns)
                if column.name not in unhashed_columns
            ]
            self._hash = hashlib.sha256()

        def update(self, row):
            """Add the next row to the hash."""

            self._hash.update(
                json.dumps(
                    [row[position] for position in self._positions],
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
                + b"\n"
            )

        def hexdigest(self):
            """Return the hash of the rows fed so far, as hexadecimal."""

            return self._hash.hexdigest()

    def __init__(self, engine, project_home):
        """
        Manage the experiments of the given project.

        Args:
            engine:    The SQLAlchemy engine for the original's database.

            project_home(str):    The original's home directory.
        """

        self._engine = engine
        self._home = project_home

    @contextmanager
    def _begin_session(self):
        """
        Yield a session on the original's database, within a transaction.

        The transaction is committed on leaving the block, and rolled back if
        it raises. Objects loaded in it stay readable afterwards.
        """

        with (
            Session(self._engine, expire_on_commit=False) as db_session,
            db_session.begin(),
        ):
            yield db_session

    def get_experiments(self):
        """
        Return the experiments of the project, or None if it is an experiment.

        Every read of the ``experiment`` table goes through here: its absence
        is what marks a project as an experiment, and only here is that
        decided.

        Returns:
            [Experiment] or None:    The project's experiments, by name,
                detached from the session that read them; None if the project
                has no ``experiment`` table, i.e. is an experiment itself.
        """

        with self._begin_session() as db_session:
            if not sa_inspect(db_session.connection()).has_table(
                Experiment.__tablename__
            ):
                return None
            return list(
                db_session.scalars(select(Experiment).order_by(Experiment.name))
            )

    def refuse_if_locked(self):
        """
        Raise if the project is locked: it has experiments not yet stale.

        Any change to the original makes its experiments unmergeable, so it
        is refused while one could still be merged. One still being created
        counts: it is locked from the moment its row is inserted. An
        experiment, having no ``experiment`` table, is never locked.

        Raises:
            ProjectLockedError:    Naming the experiments holding the lock.
        """

        experiments = self.get_experiments()
        if experiments is None:
            return
        locking = {
            experiment.name: experiment.state
            for experiment in experiments
            if experiment.state != self._stale_state
        }
        if not locking:
            return
        raise ProjectLockedError(
            "The project cannot be changed while it has experiments that may "
            "still be merged: "
            + ", ".join(f"{name} ({state})" for name, state in locking.items())
            + ". Merge or delete them, or unlock the project "
            "('wisp-experiments unlock'), which marks them stale.",
            details={"experiments": locking},
        )

    def unlock(self):
        """
        Release the lock by marking every experiment that is not stale.

        Their configurations are kept: resetting the processing of one brings
        it up to date with the original, and so locks the original again.

        Returns:
            [str]:    The names of the experiments marked stale.
        """

        experiments = self.get_experiments()
        if experiments is None:
            return []
        unlocked = [
            experiment.name
            for experiment in experiments
            if experiment.state != self._stale_state
        ]
        if unlocked:
            # By name, so that an experiment created since the read above is
            # not marked stale without being reported.
            with self._begin_session() as db_session:
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name.in_(unlocked))
                    .values(state=self._stale_state)
                )
        return unlocked

    def _select_hashed_rows(self, table):
        """
        Return the statement reading a table's rows in the order they hash.

        That is primary key order, or, for a table without a primary key, the
        order of the columns that are hashed, so that the rows come in the
        same order however their timestamps change.

        Args:
            table(sqlalchemy.Table):    The table, reflected from the
                database.

        Returns:
            sqlalchemy.Select:    Every column of every row, in the table's
                column order.
        """

        order = list(table.primary_key.columns) or [
            column
            for column in table.columns
            if column.name not in self._unhashed_columns
        ]
        return select(table).order_by(*order)

    def _get_fingerprint(self, connection):
        """
        Return the original's fingerprint: its revision, and a hash per table.

        Exact: deletions, updates within the same second and updates after
        the clock is set back all change it. Tables are reflected from the
        database rather than taken from the models, so that it covers
        whatever the database holds.

        Every table is read within one transaction, so that the fingerprint
        is of one state of the project. SQLite does not start one for a read,
        so it is started here unless the connection is already in one; it
        ends when the caller's does.

        Args:
            connection:    A connection to the original's database.

        Returns:
            dict:    ``{"revision": <Alembic revision>, "tables": {<name>:
                <hex SHA-256>}}``, which is stored as JSON.
        """

        if (
            connection.dialect.name == "sqlite"
            and not connection.connection.dbapi_connection.in_transaction
        ):
            connection.exec_driver_sql("BEGIN")

        revision = MigrationContext.configure(connection).get_current_revision()
        metadata = MetaData()
        metadata.reflect(connection)
        table_hashes = {}
        for name in sorted(set(metadata.tables) - self._unhashed_tables):
            table = metadata.tables[name]
            hasher = self._TableHasher(table, self._unhashed_columns)
            for row in connection.execute(self._select_hashed_rows(table)):
                hasher.update(row)
            table_hashes[name] = hasher.hexdigest()

        return {"revision": revision, "tables": table_hashes}
