"""The records of a project's experiments, kept in its database.

An experiment is a copy of a project, living at ``<home>/experiments/<name>``,
processed on its own and possibly merged back. The original records its
experiments in the ``experiment`` table (:class:`Experiment`); an
experiment's own database has no such table, and that absence is what makes
it an experiment.

Only what the database layer itself needs is here: the lock, the fingerprint,
and keeping experiments mergeable across a migration. :mod:`migrate
<autowisp.database.migrate>` imports it, so it cannot import ``migrate``, nor
anything that does, such as the copy of a project's database. The operations
on experiments, which are built on both, are in
:mod:`autowisp.project_experiments`, database work and file work alike.
"""

import hashlib
import json
from contextlib import contextmanager

from sqlalchemy import inspect as sa_inspect, select, update
from sqlalchemy.orm import Session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Experiment, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.database.pipeline_runs import find_active_runs
from autowisp.database.project_rows import read_project
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import ExperimentError, ProjectLockedError


class ExperimentRecords:
    """
    The records of one project's experiments, in its database.

    Everything runs from the original, whose ``experiment`` table is the only
    record of its experiments. Each method opens its own connections, so that
    it decides when what it writes is committed. The operations on
    experiments, which need what this module cannot import, are added by its
    subclass, :class:`~autowisp.project_experiments.ProjectExperiments`.

    The original's fingerprint (:meth:`_get_fingerprint`) tells whether it
    has changed since an experiment was copied from it. It covers what a
    merge must not lose: a merge replaces the original with the experiment,
    so a change made to the original meanwhile counts only if losing it would
    matter. Everything is included except :attr:`_unhashed_tables`,
    :attr:`_unhashed_columns` and :attr:`_unhashed_rows`, each with its
    reason, so that a table or column added later is covered by default.
    Kept deliberately: the processing runs in ``pipeline_run``, which the
    active-run checks read (a run recorded as finished in the original would
    come back unfinished, and block every later create and merge), and the
    ``notes`` columns, the user's annotations.
    """

    # The directory under the original's home that holds its experiments.
    _dirname = "experiments"

    # Until the copy is complete, by a create or a reset.
    _creating_state = "creating"
    # The original has not changed since the experiment was copied from it.
    _current_state = "current"
    # The original has changed since the experiment was copied from it.
    _stale_state = "stale"
    # While the experiment is merged into the original.
    _merging_state = "merging"
    # While the experiment is deleted.
    _deleting_state = "deleting"
    # The states of experiments that do not lock the original, which no
    # merge would lose a change to: every other state does, a state added
    # later included, until it is added here.
    _unlocked_states = {_stale_state, _deleting_state}
    # The states of experiments that may be opened as projects. In any
    # other, an operation is working on it, or was interrupted doing so.
    _openable_states = {_current_state, _stale_state}

    # By operation, the task of the original's runs performing it. Only these
    # are operations on experiments: a task added later counts as
    # processing, hashed and replaced by a merge, until added here.
    _operation_tasks = {
        operation: f"{operation} experiment"
        for operation in ("create", "reset", "merge", "restore", "delete")
    }

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

    # By table, a column and the values of it whose rows are left out of the
    # table's hash.
    _unhashed_rows = {
        # The runs of operations on experiments: processing depends on none
        # of them, and each, e.g. creating another experiment, would
        # otherwise make every current experiment stale. A merge keeps the
        # original's.
        "pipeline_run": ("task", set(_operation_tasks.values())),
    }

    class _ProjectHasher:
        """
        Hash each table a fingerprint covers, as :func:`read_project` reads it.

        Each row is encoded as a compact JSON list of the values of its
        hashed columns, in the table's column order, one line per row. Values
        JSON has no form for (dates, decimals, bytes) are written as their
        ``str``. The encoding need only agree with itself: the hashes
        compared are always of the same database, read by the same code.

        It does not agree across databases, since the rows are read as the
        tables reflect: a server returns its doubles as ``Decimal`` and its
        booleans as integers, and may order the columns differently. An
        original moved to another database by a copy therefore has its
        experiments' fingerprints recorded afresh (see
        :meth:`.ProjectExperiments.clone_original`).

        A consumer of the reader rather than a reader of its own, so that
        copying the original can hash it in the same pass.
        """

        def __init__(self, unhashed_tables, unhashed_columns, unhashed_rows):
            """
            Prepare to hash the tables, columns and rows not excluded.

            Args:
                unhashed_tables(set):    The names of the tables left out.

                unhashed_columns(set):    The names of the columns left out
                    of every table.

                unhashed_rows(dict):    By table name, a column and a set of
                    values: the table's rows holding one of them in that
                    column are left out.
            """

            self._unhashed_tables = unhashed_tables
            self._unhashed_columns = unhashed_columns
            self._unhashed_rows = unhashed_rows
            self._hashes = {}
            # The project's Alembic revision, once read; None if it has none.
            self._revision = None

        def __call__(self, table):
            """Start hashing the table; return what to give its rows to."""

            if table.name == "alembic_version":

                def record_revision(batch):
                    for row in batch:
                        self._revision = row["version_num"]

                return record_revision
            if table.name in self._unhashed_tables:
                return None
            hashed_columns = [
                column.name
                for column in table.columns
                if column.name not in self._unhashed_columns
            ]
            rows_hash = hashlib.sha256()
            self._hashes[table.name] = rows_hash
            unhashed_column, unhashed_values = self._unhashed_rows.get(
                table.name, (None, ())
            )

            def hash_rows(batch):
                for row in batch:
                    if (
                        unhashed_column is not None
                        and row[unhashed_column] in unhashed_values
                    ):
                        continue
                    rows_hash.update(
                        json.dumps(
                            [row[column] for column in hashed_columns],
                            separators=(",", ":"),
                            default=str,
                        ).encode("utf-8")
                        + b"\n"
                    )

            return hash_rows

        def get_fingerprint(self):
            """Return the revision read, and each table's hash, by name."""

            return {
                "revision": self._revision,
                "tables": {
                    name: rows_hash.hexdigest()
                    for name, rows_hash in sorted(self._hashes.items())
                },
            }

    def __init__(self, engine, project_home):
        """
        Manage the experiments of the given project.

        Args:
            engine:    The SQLAlchemy engine for the original's database.

            project_home(str):    The original's home directory.
        """

        self._engine = engine
        self._home = project_home

    @classmethod
    def is_operation_run(cls, row):
        """
        Is the run one of an operation on experiments, rather than processing?

        Args:
            row:    The run's ``pipeline_run`` row, as a mapping from column
                name to value.

        Returns:
            bool:    Whether its task is one of the operations': a merge keeps
            the original's such runs, and the fingerprint leaves them out.
        """

        return row["task"] in cls._operation_tasks.values()

    @staticmethod
    def _find_active_runs(engine, ignored_runs, run_ids=None):
        """
        Return the runs going in the project, on this machine and on others.

        By :func:`~autowisp.database.pipeline_runs.find_active_runs`, which
        records as finished the runs of this machine whose process is gone;
        that is committed. A run of another machine counts as going, there
        being no telling from here whether it has ended: the user, knowing it
        has, e.g. because that machine crashed, names it among the ignored.

        Args:
            engine:    The engine of the project's database: the original's, or
                an experiment's.

            ignored_runs:    The ids of the runs left out: those the user knows
                have ended, and the operation's own.

            run_ids:    The ids of the only runs to consider, or None for all.

        Returns:
            [PipelineRun]:    The runs going on this machine.

            [PipelineRun]:    The runs going on other machines.
        """

        with start_engine_session(engine, expire_on_commit=False) as db_session:
            found = find_active_runs(db_session)
        return tuple(
            [
                pipeline_run
                for pipeline_run in runs
                if pipeline_run.id not in ignored_runs
                and (run_ids is None or pipeline_run.id in run_ids)
            ]
            for runs in found
        )

    @staticmethod
    def _refuse_runs(engine, runs):
        """
        Raise if any runs are given, naming them, going in the project.

        Args:
            engine:    The engine of the project's database.

            runs([PipelineRun]):    The runs going, which refuse the operation.

        Raises:
            ExperimentError:    Naming each run, with its task and host.
        """

        if not runs:
            return
        raise ExperimentError(
            "Refusing while runs are going in "
            f"{engine.url.render_as_string(hide_password=True)}: "
            + ", ".join(
                f"{pipeline_run.id} ({pipeline_run.task}, on "
                f"{pipeline_run.host})"
                for pipeline_run in runs
            )
            + ". Wait for them to finish. A run known to have ended without "
            "recording it, e.g. on a machine that crashed, can be ignored by "
            "its id.",
            details={"runs": sorted(pipeline_run.id for pipeline_run in runs)},
        )

    @classmethod
    def refuse_active_runs(cls, engine, ignored_runs):
        """
        Raise if a run is going in the project, other than those ignored.

        See :meth:`_find_active_runs` for the runs counted. Public, for
        operations on whole projects besides those on experiments, e.g.
        cloning a project's database
        (:func:`~autowisp.database.existing_project.clone_server_database`).

        Raises:
            ExperimentError:    Naming each run going, with its task and host.
        """

        here, elsewhere = cls._find_active_runs(engine, ignored_runs)
        cls._refuse_runs(engine, here + elsewhere)

    @staticmethod
    def _read_experiments(db_session):
        """
        Return the experiments of the project, or None if it is an experiment.

        Every read of the ``experiment`` table goes through here: its absence
        is what marks a project as an experiment, and only here is that
        decided. Takes the session to read in, since a migration has to read
        on its own connection (see :meth:`keep_mergeable`).

        Args:
            db_session:    A session on the original's database.

        Returns:
            [Experiment] or None:    The project's experiments, by name; None
                if the project has no ``experiment`` table, i.e. is an
                experiment itself.
        """

        if not sa_inspect(db_session.connection()).has_table(
            Experiment.__tablename__
        ):
            return None
        return list(
            db_session.scalars(select(Experiment).order_by(Experiment.name))
        )

    def get_experiments(self):
        """
        Return the experiments of the project, or None if it is an experiment.

        Returns:
            [Experiment] or None:    The project's experiments, by name,
                detached from the session that read them; None if the project
                has no ``experiment`` table, i.e. is an experiment itself.
        """

        with start_engine_session(
            self._engine, expire_on_commit=False
        ) as db_session:
            return self._read_experiments(db_session)

    def describe_experiments(self):
        """
        Return each experiment, with its state and the operation working on it.

        An experiment in any state but those of :attr:`_openable_states` has
        an operation working on it, or interrupted doing so: its latest, whose
        run it refers to. That run is either going, on this machine or
        another, or ended without the operation completing. Runs of this
        machine whose process is gone are recorded as finished as they are
        found (see :meth:`_find_active_runs`), which is committed.

        Returns:
            [dict] or None:    By name, each experiment's ``name``, ``state``
            and ``created``, and its ``operation``: None if the experiment is
            ``current`` or ``stale``, otherwise the ``task`` of its latest
            operation's run (e.g. ``"reset experiment"``) and where it is
            ``running``: ``"here"``, the host of another machine, or None if
            it was interrupted. None if the project is itself an experiment.
        """

        experiments = self.get_experiments()
        if experiments is None:
            return None
        unsettled = {
            experiment.run_id
            for experiment in experiments
            if experiment.state not in self._openable_states
        }
        running = {}
        tasks = {}
        if unsettled:
            here, elsewhere = self._find_active_runs(
                self._engine, (), unsettled
            )
            running = {pipeline_run.id: "here" for pipeline_run in here}
            running.update(
                (pipeline_run.id, pipeline_run.host)
                for pipeline_run in elsewhere
            )
            with start_engine_session(self._engine) as db_session:
                tasks = dict(
                    db_session.execute(
                        select(PipelineRun.id, PipelineRun.task).where(
                            PipelineRun.id.in_(unsettled)
                        )
                    ).all()
                )
        return [
            {
                "name": experiment.name,
                "state": experiment.state,
                "created": experiment.created,
                "operation": (
                    {
                        "task": tasks[experiment.run_id],
                        "running": running.get(experiment.run_id),
                    }
                    if experiment.run_id in unsettled
                    else None
                ),
            }
            for experiment in experiments
        ]

    @classmethod
    def can_open(cls, experiment):
        """
        May the experiment be opened as a project?

        Not while an operation is working on it, nor once one was interrupted
        doing so: only in the states of :attr:`_openable_states`. The browser
        interface finds an experiment only through its original's row, so
        this is what keeps it from opening one being reset, merged or
        deleted. A command given the experiment's home directly is not
        stopped.

        Args:
            experiment(Experiment):    The experiment's row.

        Returns:
            bool:    Whether it may be opened.
        """

        return experiment.state in cls._openable_states

    def refuse_if_locked(self):
        """
        Raise if the project is locked: it has experiments that lock it.

        Any change to the original makes its experiments unmergeable, so it
        is refused while one could still be merged: one in any state but
        those of :attr:`_unlocked_states`. One still being created counts: it
        is locked from the moment its row is inserted. An experiment, having
        no ``experiment`` table, is never locked.

        Raises:
            ProjectLockedError:    Naming the experiments holding the lock.
        """

        experiments = self.get_experiments()
        if experiments is None:
            return
        locking = {
            experiment.name: experiment.state
            for experiment in experiments
            if experiment.state not in self._unlocked_states
        }
        if not locking:
            return
        raise ProjectLockedError(
            "The project cannot be changed while it has experiments that may "
            "still be merged: "
            + ", ".join(f"{name} ({state})" for name, state in locking.items())
            + ". Merge or delete them, or unlock the project "
            "('wisp-experiments unlock'), which marks the current ones stale; "
            "one still being created can only be deleted.",
            details={"experiments": locking},
        )

    def plan_unlock(self):
        """
        Return what unlocking would do, for confirming it first.

        Returns:
            [str]:    The names of the experiments :meth:`unlock` would mark
                stale: the current ones.

            dict:    The states of the experiments that would still lock the
                project, by name: those an operation is working on, or was
                interrupted doing so, e.g. one still being created.
        """

        experiments = self.get_experiments() or []
        return (
            [
                experiment.name
                for experiment in experiments
                if experiment.state == self._current_state
            ],
            {
                experiment.name: experiment.state
                for experiment in experiments
                if experiment.state
                not in {self._current_state, *self._unlocked_states}
            },
        )

    def unlock(self):
        """
        Release the lock by marking every current experiment stale.

        Their configurations are kept: resetting the processing of one brings
        it up to date with the original, and so locks the original again.

        Experiments still being created are left alone, and keep the lock:
        their copy is incomplete, which their state is the only record of. A
        stale experiment is a complete copy, which reset can use; one whose
        create crashed is only deleted.

        Returns:
            [str]:    The names of the experiments marked stale.
        """

        unlocked = self.plan_unlock()[0]
        if unlocked:
            # By name, so that an experiment created since the read above is
            # not marked stale without being reported.
            with start_engine_session(self._engine) as db_session:
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name.in_(unlocked))
                    .values(state=self._stale_state)
                )
        return unlocked

    def mark_changed_stale(self, connection=None, fingerprint=None):
        """
        Mark stale the current experiments whose original changed since.

        An experiment is current while the original is as it was when copied,
        which the fingerprint recorded in its row tells. The lock keeps it
        so; one found changed means the original changed despite it, or was
        moved to a database encoding its rows differently (see
        :class:`_ProjectHasher`), and the experiment could no longer be merged
        without losing the change.

        Args:
            connection:    A connection to the original's database, inside a
                transaction to do this in, e.g. a migration's; None for a
                transaction of its own.

            fingerprint(dict):    The original's fingerprint, if read
                elsewhere: in the database it was copied from, in the copy's
                pass. Read here if None, only if an experiment is current.

        Returns:
            [str]:    The names of the experiments marked stale.
        """

        if connection is None:
            with self._engine.begin() as own_connection:
                return self.mark_changed_stale(own_connection, fingerprint)

        # The session joins the connection's transaction, leaving committing
        # it to the caller.
        with Session(bind=connection) as db_session:
            current = [
                experiment
                for experiment in self._read_experiments(db_session) or []
                if experiment.state == self._current_state
            ]
            if not current:
                return []
            if fingerprint is None:
                fingerprint = self._get_fingerprint(connection)
            stale = [
                experiment.name
                for experiment in current
                if json.loads(experiment.fingerprint) != fingerprint
            ]
            if stale:
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name.in_(stale))
                    .values(state=self._stale_state)
                )
        return stale

    def is_original_unchanged(self, experiment):
        """Is the original as it was when the experiment was copied from it?

        Hashes the whole original, as a merge does. Public for the browser
        interface's merge confirmation, which warns differently when a
        current experiment's original changed despite the lock; asked there
        only, never while listing experiments.

        Args:
            experiment(Experiment):    The experiment's row.

        Returns:
            bool:    Whether the original's fingerprint is the one recorded.
        """

        with self._engine.connect() as connection:
            return self._get_fingerprint(connection) == json.loads(
                experiment.fingerprint
            )

    @contextmanager
    def keep_mergeable(self, connection, fingerprint=None):
        """
        Keep the experiments mergeable across the change in the block.

        A migration changes the original's rows, and a copy to another
        database the way they are encoded, either of which would make every
        experiment look stale and unmergeable for nothing. So around it:

        * on entering, the experiments the original changed since they were
          copied are marked stale (:meth:`mark_changed_stale`): they were
          unmergeable already;

        * on leaving normally, the original's new fingerprint is recorded in
          every experiment still current. Each migrates with the same
          revisions when next opened, so it stays mergeable; a copy changed
          no row.

        Without a current experiment, no fingerprint is read. One still being
        created is left alone, having no fingerprint to compare until its
        copy completes.

        Everything is done on the given connection, within its transaction,
        which commits or rolls back with the migration: another connection
        would wait on the lock the migration holds.

        Args:
            connection:    The connection the migration runs on, inside its
                transaction; for a copy, one to the copy.

            fingerprint(dict):    For a copy, the original's fingerprint in
                the database it was copied from, read in the copy's pass.

        Yields:
            [str]:    The names of the experiments marked stale.
        """

        stale = self.mark_changed_stale(connection, fingerprint)
        yield stale

        # The session joins the connection's transaction, leaving committing
        # it to the migration.
        with Session(bind=connection) as db_session:
            still_current = [
                experiment.name
                for experiment in self._read_experiments(db_session) or []
                if experiment.state == self._current_state
            ]
            if still_current:
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name.in_(still_current))
                    .values(
                        fingerprint=self._dump_fingerprint(
                            self._get_fingerprint(connection)
                        )
                    )
                )

    @staticmethod
    def _dump_fingerprint(fingerprint):
        """Return the fingerprint as stored in an experiment's row."""

        return json.dumps(fingerprint, sort_keys=True)

    def _make_hasher(self):
        """Return a consumer of the original's rows, for its fingerprint."""

        return self._ProjectHasher(
            self._unhashed_tables, self._unhashed_columns, self._unhashed_rows
        )

    def _get_fingerprint(self, connection):
        """
        Return the original's fingerprint: its revision, and a hash per table.

        Exact: deletions, updates within the same second and updates after
        the clock is set back all change it. It covers whatever the database
        holds, read by :func:`read_project` within one transaction, which
        ends when the caller's does; the revision is read in the same pass.

        Args:
            connection:    A connection to the original's database.

        Returns:
            dict:    Its Alembic revision, under ``"revision"``, and under
            ``"tables"``, by table name, the hex SHA-256 of the table's rows.
            It is stored as JSON.
        """

        hasher = self._make_hasher()
        read_project(connection, [hasher])
        return hasher.get_fingerprint()
