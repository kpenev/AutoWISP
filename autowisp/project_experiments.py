"""Create, reset, merge and delete a project's experiments.

The operations copy whole projects: their files with
:mod:`autowisp.project_tree`, and their databases with
:mod:`autowisp.database.project_copy`. They extend
:class:`~autowisp.database.experiments.ExperimentRecords`, which holds what
the database layer itself needs, the lock and the fingerprint. The split is
by what each module may import, not by database work and file work: the
records are imported by :mod:`autowisp.database.migrate`, so everything built
on it, the copy and the steps' path options, is here, database work
included.
"""

# Every operation on experiments, kept together in one class.
# pylint: disable=too-many-lines

import json
import logging
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import PurePath

from alembic.runtime.migration import MigrationContext
from sqlalchemy import delete, func, select, update

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    Configuration,
    Error,
    Experiment,
    Image,
    MasterType,
    Parameter,
    PipelineRun,
)

# pylint: enable=no-name-in-module
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import create_project_engine, get_sqlite_url
from autowisp.database.migrate import check_project_schema, locked_connection
from autowisp.database.pipeline_runs import record_run, stop_run
from autowisp.database.project_copy import copy_project_database
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import (
    ConfigurationError,
    DatabaseError,
    ExperimentError,
    ResourceError,
)
from autowisp.processing_steps import get_path_arguments
from autowisp.processing_steps.manual_util import split_path_values
from autowisp.project_paths import PROJHOME_MARKER
from autowisp.project_tree import (
    clear_tree,
    copy_tree,
    list_movable,
    move_tree,
    remove_path,
)

_logger = logging.getLogger(__name__)


class ProjectExperiments(ExperimentRecords):
    """
    The experiments of one project, and every operation on them.

    Adds to the records the operations, and what they are built on: the
    checks of the configured paths, and the copies of the projects' files and
    databases.
    """

    # A name an experiment may take, which is its directory's: letters,
    # digits, "-" and "_", so never a hidden one, nor a path.
    _name_pattern = re.compile("[A-Za-z0-9_-]+")

    # The start of the name of a merge's backup, under the experiments'
    # directory, followed by when the merge started (UTC) and its run. Not a
    # name an experiment may take.
    _backup_prefix = ".pre-merge-"

    # A file in a merge's backup while every file of the original it moves is
    # in the backup, so that what is in their place was copied from the
    # experiment (see _restore_backup).
    _moved_marker = ".all-moved"

    @staticmethod
    def _refuse_external_paths(db_session):
        """
        Raise if a configured path lies outside the project.

        An experiment shares with its original whatever lies outside its home,
        so a step writing there would overwrite the other's products. Every
        configured value of every path option, under any condition and in any
        version, must start with ``{PROJHOME}``, except those of the options
        declared ``path="external"``: files of other tools, only read, and the
        temporary storage, under which each process makes a directory of its
        own. Inputs are not told from outputs: one step's outputs are
        another's inputs.

        Stored paths are compared as strings, so only the literal marker
        counts: an equivalent spelling, e.g. ``{PROJHOME:s}``, is refused.

        Args:
            db_session:    A session on the project's database.

        Raises:
            ConfigurationError:    Naming each parameter and value refused.
        """

        path_kinds = {
            name: path_kind
            for name, path_kind in get_path_arguments().items()
            if path_kind != "external"
        }
        refused = sorted(
            {
                (name, value)
                for name, value in db_session.execute(
                    select(Parameter.name, Configuration.value)
                    .join(Configuration.parameter)
                    .where(Parameter.name.in_(path_kinds))
                )
                if value
                and any(
                    not path.startswith(PROJHOME_MARKER)
                    for _, path in split_path_values(value, path_kinds[name])
                )
            }
        )
        if refused:
            raise ConfigurationError(
                "Paths are configured outside the project, where an "
                "experiment and its original would share them: "
                + ", ".join(f"{name} = {value}" for name, value in refused)
                + f". Configure them under {PROJHOME_MARKER}.",
                details={"parameters": sorted({name for name, _ in refused})},
            )

    def refuse_run(self):
        """
        Raise unless a processing run may start in the project.

        Refused in a locked project (see :meth:`refuse_if_locked`), and in an
        experiment whose configuration has a path outside it (see
        :meth:`_refuse_external_paths`). An ordinary project may write outside
        its home.

        Raises:
            ProjectLockedError:    If the project is locked.

            ConfigurationError:    If the project is an experiment with a path
                configured outside it.
        """

        self.refuse_if_locked()
        with start_engine_session(self._engine) as db_session:
            if self._read_experiments(db_session) is None:
                self._refuse_external_paths(db_session)

    def _get_home(self, name):
        """Return the home of the named experiment."""

        return os.path.join(self._home, self._dirname, name)

    @staticmethod
    def _find_raw_images(db_session):
        """
        Return the raw images inside the project, relative to its home.

        Nothing writes to them, so copying the project links them.
        """

        prefix = PROJHOME_MARKER + "/"
        return {
            PurePath(raw_fname[len(prefix) :])
            for raw_fname in db_session.scalars(
                select(Image.raw_fname).where(
                    Image.raw_fname.startswith(prefix)
                )
            )
        }

    @staticmethod
    def _finish_run(engine, run_id):
        """Record the run as finished in the project's database, if not yet."""

        with engine.begin() as connection:
            connection.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id, PipelineRun.finished.is_(None))
                .values(finished=func.now())  # pylint: disable=not-callable
            )

    def _copy_database(self, home, run_id, kept_tables=None):
        """
        Copy the original's database into the experiment's home.

        The original is hashed in the same pass. A new copy's ``experiment``
        table, which the copy creates empty, is dropped: its absence is what
        makes the copy an experiment. The copy holds the run copying it,
        unfinished, which is recorded as finished in it: otherwise a later
        check from another machine would count the run as still going in the
        experiment.

        Args:
            home(str):    The experiment's home.

            run_id(int):    The id of the run copying it.

            kept_tables:    None to copy into a new database. Otherwise the
                rows of the experiment's database are replaced, except those
                of the tables named.

        Returns:
            dict:    The original's fingerprint, as copied.
        """

        hasher = self._make_hasher()
        engine = create_project_engine(get_sqlite_url(home))
        try:
            copy_project_database(
                self._engine,
                engine,
                replace=kept_tables is not None,
                kept_tables=kept_tables or (),
                table_consumers=[hasher],
            )
            Experiment.__table__.drop(engine, checkfirst=True)
            self._finish_run(engine, run_id)
        finally:
            engine.dispose()
        return hasher.get_fingerprint()

    def _complete_copy(self, name, fingerprint):
        """
        Record the copy of the original as complete, making it current.

        Under the migration lock, so that the original's revision cannot
        change between checking and recording it. If it is still the one
        copied, the experiment is current. Otherwise a newer AutoWISP migrated
        the original during the copy, and the experiment is stale: its copy is
        complete at the earlier revision, and resetting it with that version
        brings it up to date.

        Nothing else can have changed the original since it was copied: the
        experiment's row has locked it since before the copy.

        Args:
            name(str):    The experiment's name.

            fingerprint(dict):    The original's fingerprint, as copied.

        Returns:
            str:    The experiment's state.

        Raises:
            ExperimentError:    If the experiment was deleted meanwhile.
        """

        with locked_connection(self._engine) as connection:
            if (
                connection.scalar(
                    select(Experiment.id).where(Experiment.name == name)
                )
                is None
            ):
                raise ExperimentError(
                    f"Experiment {name!r} was deleted while being copied."
                )
            revision = MigrationContext.configure(
                connection
            ).get_current_revision()
            state = (
                self._current_state
                if revision == fingerprint["revision"]
                else self._stale_state
            )
            connection.execute(
                update(Experiment)
                .where(Experiment.name == name)
                .values(
                    state=state,
                    fingerprint=self._dump_fingerprint(fingerprint),
                )
            )
        if state == self._stale_state:
            _logger.warning(
                "Experiment %s is stale: a newer AutoWISP migrated the "
                "original while it was copied. Resetting the experiment with "
                "that version brings it up to date.",
                name,
            )
        return state

    def create(self, name, ignored_runs=()):
        """
        Create an experiment: a copy of the project at ``experiments/<name>``.

        Refused before anything is touched if the name is not one an
        experiment may take (:attr:`_name_pattern`) or is taken, if the
        project is itself an experiment, if a path is configured outside it
        (:meth:`_refuse_external_paths`), or if the experiment's directory
        exists and is not empty: deleting an experiment removes everything
        under it, so it never adopts files it did not put there.

        The experiment's row is inserted ``creating``, with the run creating
        it, which locks the original. A run going in the original
        (:meth:`_refuse_active_runs`), or too little free space, then refuses
        the create, which is undone. Otherwise the files are copied, then the
        database (:meth:`_copy_database`), and the experiment becomes current
        (:meth:`_complete_copy`). However the create ends, its run is recorded
        as finished: one interrupted leaves the experiment ``creating``, which
        only delete applies to.

        Args:
            name(str):    The experiment's name.

            ignored_runs:    The ids of runs not counted as going (see
                :meth:`_refuse_active_runs`).

        Returns:
            str:    The experiment's state: ``"current"``, or ``"stale"`` if
            the original was migrated during the copy.

        Raises:
            ExperimentError:    If refused, or if the experiment was deleted
                while being created.

            ConfigurationError:    If a path is configured outside the
                project.

            ResourceError:    If the copy would not fit in the free space.
        """

        if not self._name_pattern.fullmatch(name):
            raise ExperimentError(
                f"{name!r} cannot name an experiment: use only letters, "
                "digits, '-' and '_'."
            )
        home = self._get_home(name)
        if os.path.exists(home) and (
            not os.path.isdir(home) or os.listdir(home)
        ):
            raise ExperimentError(
                f"{home} exists and is not an empty directory: move it away, "
                "or choose another name."
            )
        made_home = not os.path.exists(home)
        with start_engine_session(self._engine) as db_session:
            experiments = self._read_experiments(db_session)
            if experiments is None:
                raise ExperimentError(
                    f"{self._home} is an experiment: create experiments from "
                    "its original."
                )
            if name in [experiment.name for experiment in experiments]:
                raise ExperimentError(
                    f"The project already has an experiment named {name!r}."
                )
            self._refuse_external_paths(db_session)
            linked_files = self._find_raw_images(db_session)
            run_id = record_run(db_session, self._operation_tasks["create"]).id
            db_session.add(
                Experiment(
                    name=name,
                    home=f"{PROJHOME_MARKER}/{self._dirname}/{name}",
                    created=func.now(),  # pylint: disable=not-callable
                    fingerprint="{}",
                    state=self._creating_state,
                    run_id=run_id,
                )
            )

        try:
            try:
                self._refuse_active_runs(self._engine, {*ignored_runs, run_id})
                os.makedirs(home, exist_ok=True)
                copy_tree(self._home, home, linked_files, self._dirname)
            except (ExperimentError, ResourceError):
                # Refused before anything was copied: undone, which releases
                # the lock.
                with start_engine_session(self._engine) as db_session:
                    db_session.execute(
                        delete(Experiment).where(Experiment.name == name)
                    )
                if made_home and os.path.isdir(home):
                    os.rmdir(home)
                raise
            return self._complete_copy(name, self._copy_database(home, run_id))
        finally:
            self._finish_run(self._engine, run_id)

    def _find_experiment(self, name):
        """
        Return the named experiment's row.

        Raises:
            ExperimentError:    If the project is itself an experiment, or has
                no experiment of that name.
        """

        experiments = self.get_experiments()
        if experiments is None:
            raise ExperimentError(
                f"{self._home} is an experiment: its original manages "
                "experiments."
            )
        for experiment in experiments:
            if experiment.name == name:
                return experiment
        raise ExperimentError(f"The project has no experiment named {name!r}.")

    def _set_state(self, name, state, run_id):
        """Set an experiment's state, and the run of the operation giving it."""

        with start_engine_session(self._engine) as db_session:
            db_session.execute(
                update(Experiment)
                .where(Experiment.name == name)
                .values(state=state, run_id=run_id)
            )

    def _refuse_copy_operation(self, name, experiment_engine):
        """
        Raise unless the original and the experiment can be copied between.

        Both must be at the code's revision: each is migrated on its own, with
        its backup and confirmation. And the experiment's configuration must
        have no path outside it (:meth:`_refuse_external_paths`).

        Args:
            name(str):    The experiment's name.

            experiment_engine:    The engine of the experiment's database.

        Raises:
            ExperimentError:    Naming the project not at the code's revision.

            ConfigurationError:    If a path is configured outside the
                experiment.
        """

        for engine, home in [
            (self._engine, self._home),
            (experiment_engine, self._get_home(name)),
        ]:
            try:
                check_project_schema(engine)
            except DatabaseError as error:
                raise ExperimentError(f"{home}: {error}") from error
        with start_engine_session(experiment_engine) as db_session:
            self._refuse_external_paths(db_session)

    def _refuse_reset(self, experiment, experiment_engine):
        """
        Raise if the experiment's processing cannot be reset.

        See :meth:`reset` for when.

        Args:
            experiment(Experiment):    The experiment's row.

            experiment_engine:    The engine of the experiment's database.
        """

        if experiment.state == self._creating_state:
            raise ExperimentError(
                f"Experiment {experiment.name!r} is incomplete: it is being "
                "created or reset, or that was interrupted. It can only be "
                "deleted."
            )
        self._refuse_copy_operation(experiment.name, experiment_engine)
        with start_engine_session(experiment_engine) as db_session:
            present = set(db_session.scalars(select(Condition.id)))
        with start_engine_session(self._engine) as db_session:
            missing = {
                condition_id
                for condition_id in db_session.scalars(
                    select(MasterType.condition_id).union(
                        select(MasterType.maker_image_split_condition_id)
                    )
                )
                if condition_id is not None
            } - present
        if missing:
            raise ExperimentError(
                f"Experiment {experiment.name!r} lacks the conditions "
                f"{sorted(missing)}, which the original's master types use: "
                "resetting it keeps its conditions, so its masters could not "
                "be matched to images."
            )

    def reset(self, name, ignored_runs=()):
        """
        Reset an experiment's processing to the original's, keeping its setup.

        Refused before anything is touched if the project has no such
        experiment, if the experiment is incomplete (``creating``) or its
        database is gone, either of which only delete applies to, or if
        either project is not at the code's revision: each is migrated on its
        own, with its backup and confirmation. Also refused if the
        experiment's configuration, which a reset keeps, has a path outside
        it (:meth:`_refuse_external_paths`) or lacks a condition the
        original's master types use.

        The reset's run is recorded, and the experiment, referring to it, is
        marked ``creating``, which locks the original until the copy is
        complete. A run going in either project then refuses the reset, which
        restores the experiment as it was: the lock comes first, so that a
        run starting meanwhile is either seen or refused by it. Otherwise the
        experiment's files are replaced by the original's, then the rows of
        its database, except its configuration's, and it becomes current, as
        a created one does. However the reset ends, its run is recorded as
        finished: one interrupted leaves the experiment ``creating``.

        Args:
            name(str):    The experiment's name.

            ignored_runs:    The ids of runs not counted as going (see
                :meth:`_refuse_active_runs`).

        Returns:
            str:    The experiment's state: ``"current"``, or ``"stale"`` if
            the original was migrated during the copy.

        Raises:
            ExperimentError:    If refused, or if the experiment was deleted
                during the reset.

            ConfigurationError:    If a path is configured outside the
                experiment.

            ResourceError:    If the original's files would not fit in the
                free space, which is checked once the experiment's are
                removed.
        """

        experiment = self._find_experiment(name)
        home = self._get_home(name)
        # Checked before opening it: opening a missing one would create it.
        if not os.path.isfile(os.path.join(home, "autowisp.db")):
            raise ExperimentError(
                f"Experiment {name!r} has no database in {home}: it can only "
                "be deleted."
            )
        experiment_engine = create_project_engine(get_sqlite_url(home))
        try:
            self._refuse_reset(experiment, experiment_engine)
            with start_engine_session(self._engine) as db_session:
                linked_files = self._find_raw_images(db_session)
                run_id = record_run(
                    db_session, self._operation_tasks["reset"]
                ).id
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name == name)
                    .values(
                        state=self._creating_state,
                        run_id=run_id,
                        created=func.now(),  # pylint: disable=not-callable
                    )
                )
            try:
                self._refuse_active_runs(self._engine, {*ignored_runs, run_id})
                self._refuse_active_runs(experiment_engine, ignored_runs)
            except ExperimentError:
                # Refused before anything was replaced: restored as it was.
                with start_engine_session(self._engine) as db_session:
                    db_session.execute(
                        update(Experiment)
                        .where(Experiment.name == name)
                        .values(
                            state=experiment.state,
                            run_id=experiment.run_id,
                            created=experiment.created,
                        )
                    )
                self._finish_run(self._engine, run_id)
                raise
        finally:
            experiment_engine.dispose()

        try:
            clear_tree(home)
            copy_tree(self._home, home, linked_files, self._dirname)
            return self._complete_copy(
                name,
                self._copy_database(
                    home,
                    run_id,
                    # The experiment's configuration.
                    kept_tables=[
                        "configuration",
                        "condition",
                        "condition_expression",
                        "diagnostic_expression",
                    ],
                ),
            )
        finally:
            self._finish_run(self._engine, run_id)

    def delete(self, name, ignored_runs=()):
        """
        Delete an experiment: its directory, then its row.

        Any experiment may be deleted, in any state but ``merging``: one whose
        create, reset or delete was interrupted, or whose directory is gone,
        too. One being merged is refused, whether the merge is going or was
        interrupted: its row is the only record that the original may need
        restoring from the merge's backup (see :meth:`restore_merge`). Its
        files all lie in its directory, since its configuration allows no
        path outside it, so removing the directory removes all it made.

        The delete's run is recorded, and the experiment marked ``deleting``
        and referred to it, which releases its lock on the original and keeps
        the browser interface from opening it. A run going on another
        machine, in the experiment or as its latest operation (e.g. copying
        the original into it), then refuses the delete, which restores the
        experiment's state and run: it cannot be stopped from here, and
        removing its files would make it fail, or recreate some of them.
        Runs going on this machine are stopped and waited for, since Windows
        cannot remove a file still open. However the delete ends, its run is
        recorded as finished: one interrupted leaves the experiment
        ``deleting``, and deleting it again completes it.

        Args:
            name(str):    The experiment's name.

            ignored_runs:    The ids of runs going on other machines that the
                user knows have ended (see :meth:`_find_active_runs`).

        Raises:
            ExperimentError:    If there is no such experiment, it is being
                merged, or a run is going in it on another machine.

            PipelineError:    If a run going on this machine cannot be stopped.
        """

        experiment = self._find_experiment(name)
        if experiment.state == self._merging_state:
            raise ExperimentError(
                f"Experiment {name!r} is being merged, or its merge was "
                "interrupted, which may have left the original's files part "
                "way into the merge's backup. Wait for the merge to finish, "
                "or restore the original from the backup "
                "(wisp-experiments restore) first."
            )
        home = self._get_home(name)
        with start_engine_session(self._engine) as db_session:
            run_id = record_run(db_session, self._operation_tasks["delete"]).id
            db_session.execute(
                update(Experiment)
                .where(Experiment.name == name)
                .values(state=self._deleting_state, run_id=run_id)
            )

        try:
            try:
                # The experiment's latest operation, if still going: its
                # create or reset, or an earlier delete.
                previous, elsewhere = self._find_active_runs(
                    self._engine, ignored_runs, {experiment.run_id}
                )
                self._refuse_runs(self._engine, elsewhere)
                in_experiment = []
                # Checked before opening it: opening a missing one would
                # create it.
                if os.path.isfile(os.path.join(home, "autowisp.db")):
                    experiment_engine = create_project_engine(
                        get_sqlite_url(home)
                    )
                    try:
                        in_experiment, elsewhere = self._find_active_runs(
                            experiment_engine, ignored_runs
                        )
                        self._refuse_runs(experiment_engine, elsewhere)
                    finally:
                        experiment_engine.dispose()
            except ExperimentError:
                # Refused before anything was removed: restored as it was.
                self._set_state(name, experiment.state, experiment.run_id)
                raise
            for pipeline_run in previous + in_experiment:
                stop_run(pipeline_run)
            # Recorded as ended now, rather than whenever the original's runs
            # are next looked at; those in the experiment go with it.
            for pipeline_run in previous:
                self._finish_run(self._engine, pipeline_run.id)
            if os.path.lexists(home):
                remove_path(home)
            with start_engine_session(self._engine) as db_session:
                db_session.execute(
                    delete(Experiment).where(Experiment.name == name)
                )
        finally:
            self._finish_run(self._engine, run_id)

    def delete_all(self, ignored_runs=()):
        """
        Delete every experiment of the project, as deleting the project needs.

        Each is deleted by :meth:`delete`, with its refusals. Those refused do
        not stop the others: every experiment that can be deleted is, and
        one error then names those refused. A project that is itself an
        experiment has none.

        Args:
            ignored_runs:    As for :meth:`delete`.

        Returns:
            [str]:    The names of the experiments deleted.

        Raises:
            ExperimentError:    Naming each experiment refused, and why.
        """

        deleted = []
        refused = {}
        for experiment in self.get_experiments() or []:
            try:
                self.delete(experiment.name, ignored_runs)
                deleted.append(experiment.name)
            except ExperimentError as error:
                refused[experiment.name] = str(error)
        if refused:
            raise ExperimentError(
                f"Experiments of {self._home} could not be deleted: "
                + " ".join(
                    f"{name}: {reason}" for name, reason in refused.items()
                ),
                details={"experiments": sorted(refused)},
            )
        return deleted

    def _back_up_database(self, backup):
        """
        Copy the original's database into a merge's backup, as autowisp.db.

        SQLite is copied with its backup API, which gives a consistent copy
        while others use the database. A server database is copied into
        SQLite by :func:`copy_project_database`, which leaves out the
        ``experiment`` table's rows: the merge changes only their states.

        Args:
            backup(str):    The backup's directory.

        Returns:
            None
        """

        if self._engine.dialect.name == "sqlite":
            source = self._engine.raw_connection()
            try:
                with closing(
                    sqlite3.connect(os.path.join(backup, "autowisp.db"))
                ) as target:
                    source.driver_connection.backup(target)
            finally:
                source.close()
            return
        backup_engine = create_project_engine(get_sqlite_url(backup))
        try:
            copy_project_database(self._engine, backup_engine)
        finally:
            backup_engine.dispose()

    def _restore_backup(self, backup, kept_files):
        """
        Put the original's files back from a merge's backup, then remove it.

        Wherever the merge stopped: the marker (:attr:`_moved_marker`) in the
        backup tells that all the original's files were moved there, so that
        what is in their place was copied from the experiment, and is removed
        first, then the marker. Without it, only some were moved, and those
        in place are the original's. Each step can be repeated, so a restore
        interrupted in turn is completed by restoring again.

        Its database needs nothing: a merge left ``merging`` never committed
        its transaction.

        Args:
            backup(str):    The backup's directory.

            kept_files(set):    The files the backup left in place (see
                :func:`~autowisp.project_tree.move_tree`).

        Returns:
            None
        """

        marker = os.path.join(backup, self._moved_marker)
        if os.path.exists(marker):
            for relative in list_movable(self._home, kept_files, self._dirname):
                remove_path(os.path.join(self._home, relative))
            os.remove(marker)
        move_tree(backup, self._home, kept_files, self._dirname)
        remove_path(backup)

    def _list_merge_backups(self):
        """Return the paths of the backups merges left, oldest first."""

        directory = os.path.join(self._home, self._dirname)
        if not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.startswith(self._backup_prefix)
        ]

    def _make_backup_dir(self, run_id):
        """
        Create the directory of a merge's backup, named for the time and run.

        The time (UTC) is for the user. The merge's run, which is unique,
        tells the backup of an interrupted merge from the others (see
        :meth:`_find_backup`).

        Args:
            run_id(int):    The id of the merge's run.

        Returns:
            str:    The directory created.
        """

        backup = os.path.join(
            self._home,
            self._dirname,
            self._backup_prefix
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-run{run_id}",
        )
        os.makedirs(backup)
        return backup

    def _find_backup(self, run_id):
        """Return the backup of the merge with the given run; None if none."""

        for backup in self._list_merge_backups():
            if backup.endswith(f"-run{run_id}"):
                return backup
        return None

    def _merge_database(self, name, experiment_engine, run_id):
        """
        Replace the original's rows with the experiment's, in one transaction.

        On SQLite, the connection's foreign keys are turned on, which the
        renumbering needs, and the transaction takes the write lock first, so
        that nothing writes between reading the runs and renumbering them.
        Within it:

        * Refused, even if forced, unless each of the original's processing
          runs is in the experiment, as the same run under the same id. The
          lock froze them at the copy, so anything else means the original
          was processed since, which the merge would lose.

        * The original's runs of operations whose ids the experiment gives to
          other runs, i.e. those recorded since the copy (e.g. creating
          another experiment, or this merge), are renumbered past both
          projects' ids. The rows referring to them follow (``ON UPDATE
          CASCADE``).

        * The errors recorded against the original's runs of operations are
          deleted. Kept with their runs, they could collide with the
          experiment's errors by id, and their files are in the backup.

        * The experiment's rows replace the original's
          (:func:`copy_project_database`), except the original's runs of
          operations and its ``experiment`` table.

        * The experiment becomes current, with the original's new
          fingerprint, read from the original, whose backend may encode rows
          differently from the experiment's. The other current experiments
          become stale.

        Args:
            name(str):    The experiment's name.

            experiment_engine:    The engine of the experiment's database.

            run_id(int):    The id of the merge's run in the original.

        Returns:
            int:    The id of the merge's run, as renumbered.

        Raises:
            ExperimentError:    If the processing runs differ.
        """

        run_columns = [
            PipelineRun.id,
            PipelineRun.task,
            PipelineRun.host,
            PipelineRun.process_id,
            PipelineRun.started,
        ]
        with self._engine.begin() as connection:
            if connection.dialect.name == "sqlite":
                # The renumbering relies on ON UPDATE CASCADE, whatever engine
                # the caller gave; the pragma is ignored within a transaction.
                connection.exec_driver_sql("PRAGMA foreign_keys = ON")
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            original_runs = {
                row["id"]: dict(row)
                for row in connection.execute(select(*run_columns)).mappings()
            }
            with experiment_engine.connect() as experiment_connection:
                experiment_runs = {
                    row["id"]: dict(row)
                    for row in experiment_connection.execute(
                        select(*run_columns)
                    ).mappings()
                }
            lost = sorted(
                original_id
                for original_id, pipeline_run in original_runs.items()
                if not self.is_operation_run(pipeline_run)
                and experiment_runs.get(original_id) != pipeline_run
            )
            if lost:
                raise ExperimentError(
                    f"The original's processing runs {lost} are not in "
                    f"experiment {name!r} as they are in the original: the "
                    "original was processed after the experiment was copied, "
                    "which merging would lose. Reset the experiment, or "
                    "delete it.",
                    details={"runs": lost},
                )
            # Only runs of operations are left to differ.
            next_id = max([*original_runs, *experiment_runs]) + 1
            for original_id, pipeline_run in sorted(original_runs.items()):
                if experiment_runs.get(original_id, pipeline_run) != (
                    pipeline_run
                ):
                    connection.execute(
                        update(PipelineRun)
                        .where(PipelineRun.id == original_id)
                        .values(id=next_id)
                    )
                    if original_id == run_id:
                        run_id = next_id
                    next_id += 1
            connection.execute(
                delete(Error).where(
                    Error.pipeline_run_id.in_(
                        select(PipelineRun.id).where(
                            PipelineRun.task.in_(self._operation_tasks.values())
                        )
                    )
                )
            )
            copy_project_database(
                experiment_engine,
                connection,
                replace=True,
                kept_rows={"pipeline_run": self.is_operation_run},
            )
            fingerprint = self._get_fingerprint(connection)
            connection.execute(
                update(Experiment)
                .where(
                    Experiment.name != name,
                    Experiment.state == self._current_state,
                )
                .values(state=self._stale_state)
            )
            connection.execute(
                update(Experiment)
                .where(Experiment.name == name)
                .values(
                    state=self._current_state,
                    fingerprint=self._dump_fingerprint(fingerprint),
                )
            )
        return run_id

    def merge(self, name, force=False, ignored_runs=()):
        """
        Merge an experiment into the original, which becomes a copy of it.

        Refused before anything is touched if the project has no such
        experiment, if an operation is working on it or was interrupted doing
        so (any state but those of :attr:`_openable_states`), if its database
        is gone, or if either project cannot be copied between
        (:meth:`_refuse_copy_operation`). A stale experiment is refused unless
        forced: the original changed after the experiment was copied, and the
        merge loses that change.

        The merge's run is recorded and the experiment marked ``merging``,
        which locks the original. A run going in either project then refuses
        the merge, as does the original's fingerprint differing from the one
        recorded, unless forced. Either restores the experiment's state.

        Otherwise the original is backed up into
        ``experiments/.pre-merge-<UTC time>-run<merge's run id>``: its
        database is copied there (:meth:`_back_up_database`), and its other
        files moved there, all but the experiments and the raw images inside
        it (:func:`~autowisp.project_tree.move_tree`), after which
        :attr:`_moved_marker` is created there. The experiment's files are
        copied into their place, then its rows (:meth:`_merge_database`),
        which makes it current: an exact copy of the new original. Should
        either fail, the original's files are put back
        (:meth:`_restore_backup`) and the experiment's state restored. Once
        merged, the marker and the earlier backups are deleted. However the
        merge ends, its run is recorded as finished; one interrupted leaves
        the experiment ``merging``, and :meth:`restore_merge` puts the
        original's files back from the backup.

        Args:
            name(str):    The experiment's name.

            force(bool):    Whether to merge even if the original changed
                after the experiment was copied, losing the change.

            ignored_runs:    The ids of runs not counted as going (see
                :meth:`_refuse_active_runs`).

        Returns:
            str:    The backup's directory, which :meth:`delete_merge_backup`
            removes.

        Raises:
            ExperimentError:    If refused.

            ConfigurationError:    If a path is configured outside the
                experiment.

            ResourceError:    If the experiment's files would not fit in the
                free space.
        """

        experiment = self._find_experiment(name)
        if experiment.state not in self._openable_states:
            raise ExperimentError(
                f"Experiment {name!r} is {experiment.state}: an operation is "
                "working on it, or was interrupted doing so."
            )
        if experiment.state == self._stale_state and not force:
            raise ExperimentError(
                f"Experiment {name!r} is stale: the original changed after it "
                "was copied, and merging would lose the change. Reset the "
                "experiment, or force the merge (--force) to lose it."
            )
        home = self._get_home(name)
        # Checked before opening it: opening a missing one would create it.
        if not os.path.isfile(os.path.join(home, "autowisp.db")):
            raise ExperimentError(
                f"Experiment {name!r} has no database in {home}: it can only "
                "be deleted."
            )
        experiment_engine = create_project_engine(get_sqlite_url(home))
        try:
            self._refuse_copy_operation(name, experiment_engine)
            with start_engine_session(self._engine) as db_session:
                kept_files = self._find_raw_images(db_session)
                run_id = record_run(
                    db_session, self._operation_tasks["merge"]
                ).id
                db_session.execute(
                    update(Experiment)
                    .where(Experiment.name == name)
                    .values(state=self._merging_state, run_id=run_id)
                )
            try:
                try:
                    self._refuse_active_runs(
                        self._engine, {*ignored_runs, run_id}
                    )
                    self._refuse_active_runs(experiment_engine, ignored_runs)
                    if not force:
                        with self._engine.connect() as connection:
                            fingerprint = self._get_fingerprint(connection)
                        if fingerprint != json.loads(experiment.fingerprint):
                            raise ExperimentError(
                                "The original changed after experiment "
                                f"{name!r} was copied, and merging would lose "
                                "the change. Reset the experiment, or force "
                                "the merge (--force) to lose it."
                            )
                except ExperimentError:
                    # Refused before anything was moved: restored as it was.
                    self._set_state(name, experiment.state, experiment.run_id)
                    raise
                with start_engine_session(experiment_engine) as db_session:
                    linked_files = self._find_raw_images(db_session)
                backup = self._make_backup_dir(run_id)
                try:
                    self._back_up_database(backup)
                    move_tree(self._home, backup, kept_files, self._dirname)
                    # Created only once everything is moved, which tells a
                    # restore what in the home is the experiment's.
                    with open(
                        os.path.join(backup, self._moved_marker),
                        "w",
                        encoding="ascii",
                    ):
                        pass
                    copy_tree(home, self._home, linked_files, self._dirname)
                    run_id = self._merge_database(
                        name, experiment_engine, run_id
                    )
                except BaseException:
                    self._restore_backup(backup, kept_files)
                    self._set_state(name, experiment.state, experiment.run_id)
                    raise
            finally:
                self._finish_run(self._engine, run_id)
        finally:
            experiment_engine.dispose()
        # Merged: the backup is left holding only the original's files.
        os.remove(os.path.join(backup, self._moved_marker))
        for earlier in self._list_merge_backups():
            if earlier != backup:
                remove_path(earlier)
        return backup

    def restore_merge(self, name, ignored_runs=()):
        """
        Restore the original from the backup of an interrupted merge.

        Refused unless the experiment is ``merging``. The restore's run is
        recorded, then a run going in the original refuses the restore: the
        merge itself, still going, or another restore. A merge is never
        stopped: stopped part way, it would leave what this restores.

        The merge's backup is the one named for the run the experiment refers
        to (:meth:`_find_backup`), which the restore leaves as it is, so that
        a restore interrupted in turn finds the backup again. Its files are
        put back (:meth:`_restore_backup`); without a backup, the merge
        stopped before moving anything. The original's database needs
        nothing: the merge's transaction never committed. The experiment then
        becomes current if the original's fingerprint is still the one it
        recorded, otherwise stale. However the restore ends, its run is
        recorded as finished; one interrupted leaves the experiment
        ``merging``, and restoring again completes it.

        Args:
            name(str):    The experiment's name.

            ignored_runs:    The ids of runs not counted as going, e.g. the
                merge's, on a machine that crashed (see
                :meth:`_refuse_active_runs`).

        Returns:
            str:    The experiment's state: ``"current"`` or ``"stale"``.

        Raises:
            ExperimentError:    If refused.
        """

        experiment = self._find_experiment(name)
        if experiment.state != self._merging_state:
            raise ExperimentError(
                f"Experiment {name!r} is {experiment.state}, not being "
                "merged: there is no merge to restore the original from."
            )
        with start_engine_session(self._engine) as db_session:
            kept_files = self._find_raw_images(db_session)
            run_id = record_run(db_session, self._operation_tasks["restore"]).id
        try:
            self._refuse_active_runs(self._engine, {*ignored_runs, run_id})
            backup = self._find_backup(experiment.run_id)
            if backup is not None:
                self._restore_backup(backup, kept_files)
            with self._engine.connect() as connection:
                fingerprint = self._get_fingerprint(connection)
            state = (
                self._current_state
                if fingerprint == json.loads(experiment.fingerprint)
                else self._stale_state
            )
            self._set_state(name, state, experiment.run_id)
            return state
        finally:
            self._finish_run(self._engine, run_id)

    def delete_merge_backup(self):
        """
        Delete the backups merges left, once the merge is known to be right.

        Refused while an experiment is being merged, or its merge was
        interrupted: its backup holds the original as it was before.

        Returns:
            [str]:    The directories of the backups deleted.

        Raises:
            ExperimentError:    Naming the experiments being merged.
        """

        merging = [
            experiment.name
            for experiment in self.get_experiments() or []
            if experiment.state == self._merging_state
        ]
        if merging:
            raise ExperimentError(
                "Refusing while experiments are being merged, or their merge "
                f"was interrupted: {', '.join(merging)}. Their backup holds "
                "the original as it was before."
            )
        backups = self._list_merge_backups()
        for backup in backups:
            remove_path(backup)
        return backups
