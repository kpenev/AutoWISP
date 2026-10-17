"""Create, reset, merge and delete a project's experiments.

The operations copy whole projects: their files, and their databases with
:mod:`autowisp.database.project_copy`. They extend
:class:`~autowisp.database.experiments.ExperimentRecords`, which holds what
the database layer itself needs, the lock and the fingerprint. The split is
by what each module may import, not by database work and file work: the
records are imported by :mod:`autowisp.database.migrate`, so everything built
on it, the copy and the steps' path options, is here, database work
included.
"""

import logging
import os
import re
import shutil
import stat
import sys
from pathlib import PurePath

from alembic.runtime.migration import MigrationContext
from sqlalchemy import delete, func, select, update

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    Configuration,
    Experiment,
    Image,
    MasterType,
    Parameter,
    PipelineRun,
)

# pylint: enable=no-name-in-module
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import (
    DB_URL_FNAME,
    create_project_engine,
    get_sqlite_url,
)
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

    @staticmethod
    def _is_database_file(name):
        """
        Is a name in a project's home one of its database's files?

        The database, SQLite's journals beside it, the backups migrating it
        leaves (``autowisp.db.pre-<revision>``), and the file naming a server
        database. Copying a project's files leaves them all out: its database
        is copied by
        :func:`~autowisp.database.project_copy.copy_project_database`.
        """

        return name in ("autowisp.db", DB_URL_FNAME) or name.startswith(
            ("autowisp.db-", "autowisp.db.pre-")
        )

    @classmethod
    def _list_tree(cls, home):
        """
        Return the directories and files a copy of a project's tree takes.

        Everything under the home, following symbolic links, except at the
        top the database's files (:meth:`_is_database_file`) and the
        experiments.

        Args:
            home(str):    The project's home.

        Returns:
            [PurePath]:    The directories, relative to the home, the home
            itself included.

            [PurePath]:    The files, relative to the home.
        """

        directories = []
        fnames = []
        for directory, subdirs, files in os.walk(home, followlinks=True):
            if directory == home:
                subdirs[:] = [
                    name
                    for name in subdirs
                    if name != cls._dirname and not cls._is_database_file(name)
                ]
                files = [
                    name for name in files if not cls._is_database_file(name)
                ]
            relative_dir = PurePath(directory).relative_to(home)
            directories.append(relative_dir)
            fnames.extend(relative_dir / name for name in files)
        return directories, fnames

    @classmethod
    def _copy_tree(cls, source_home, target_home, linked_files):
        """
        Copy a project's files into another home, all but its database's.

        What is copied is listed by :meth:`_list_tree`. Symbolic links are
        followed, so that nothing is shared through one, as nothing may be
        through a path configured outside the project.

        Files are copied with their metadata (``shutil.copy2``), over any
        already in the target: steps append to DR and lightcurve files in
        place, so each project needs its own. Only the files nothing writes,
        the raw images, are hard-linked instead, or copied where the
        filesystem cannot link them. One already in the target is left as it
        is, being the same file or a copy of it.

        The space the copies take is checked against the target's free space
        before anything is copied.

        Args:
            source_home(str):    The home of the project copied.

            target_home(str):    The home to copy it into, which exists.

            linked_files(set):    The files to hard-link, each a ``PurePath``
                relative to ``source_home``.

        Returns:
            None

        Raises:
            ResourceError:    If the copies would not fit in the free space.
        """

        directories, fnames = cls._list_tree(source_home)
        # Hard links are made only within one filesystem.
        same_device = os.stat(source_home).st_dev == os.stat(target_home).st_dev
        needed = sum(
            os.stat(os.path.join(source_home, fname)).st_size
            for fname in fnames
            if not (same_device and fname in linked_files)
        )
        free = shutil.disk_usage(target_home).free
        if needed > free:
            raise ResourceError(
                f"Copying {source_home} into {target_home} needs {needed:,} "
                f"bytes, but only {free:,} are free there.",
                details={"needed": needed, "free": free},
            )

        for directory in directories:
            os.makedirs(os.path.join(target_home, directory), exist_ok=True)
        for fname in fnames:
            source = os.path.join(source_home, fname)
            target = os.path.join(target_home, fname)
            if fname not in linked_files:
                shutil.copy2(source, target)
            elif not os.path.exists(target):
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)

    @staticmethod
    def _remove_path(path):
        """
        Remove a file, or a directory with everything in it.

        Windows refuses to remove a read-only file, so one that fails to be
        removed is made writable and removed again.
        """

        def remove_writable(remove, failed_path, _):
            os.chmod(failed_path, stat.S_IWRITE)
            remove(failed_path)

        if os.path.isdir(path) and not os.path.islink(path):
            if sys.version_info >= (3, 12):
                # pylint: disable-next=unexpected-keyword-arg
                shutil.rmtree(path, onexc=remove_writable)
            else:
                # Deprecated from 3.12, which has onexc instead.
                # pylint: disable-next=deprecated-argument
                shutil.rmtree(path, onerror=remove_writable)
        else:
            try:
                os.remove(path)
            except OSError as error:
                remove_writable(os.remove, path, error)

    @classmethod
    def _clear_tree(cls, home):
        """
        Remove everything in a project's home but its database's files.

        Resetting an experiment does this before copying its original's
        files in, so that the experiment's tree mirrors the original's.
        """

        for name in os.listdir(home):
            if not cls._is_database_file(name):
                cls._remove_path(os.path.join(home, name))

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
                    copy_run_id=run_id,
                )
            )

        try:
            try:
                self._refuse_active_runs(self._engine, {*ignored_runs, run_id})
                os.makedirs(home, exist_ok=True)
                self._copy_tree(self._home, home, linked_files)
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
        for engine, home in [
            (self._engine, self._home),
            (experiment_engine, self._get_home(experiment.name)),
        ]:
            try:
                check_project_schema(engine)
            except DatabaseError as error:
                raise ExperimentError(f"{home}: {error}") from error
        with start_engine_session(experiment_engine) as db_session:
            self._refuse_external_paths(db_session)
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
                        copy_run_id=run_id,
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
                            copy_run_id=experiment.copy_run_id,
                            created=experiment.created,
                        )
                    )
                self._finish_run(self._engine, run_id)
                raise
        finally:
            experiment_engine.dispose()

        try:
            self._clear_tree(home)
            self._copy_tree(self._home, home, linked_files)
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

        Any experiment may be deleted, in whatever state: one whose create,
        reset or delete was interrupted, or whose directory is gone, too. Its
        files all lie in its directory, since its configuration allows no
        path outside it, so removing the directory removes all it made.

        The delete's run is recorded and the experiment marked ``deleting``,
        which releases its lock on the original and keeps the browser
        interface from opening it. A run going on another machine, in the
        experiment or copying the original into it, then refuses the delete,
        which restores the experiment's state: it cannot be stopped from here,
        and removing its files would make it fail, or recreate some of them.
        Runs going on this machine are stopped and waited for, since Windows
        cannot remove a file still open. However the delete ends, its run is
        recorded as finished: one interrupted leaves the experiment
        ``deleting``, and deleting it again completes it.

        Args:
            name(str):    The experiment's name.

            ignored_runs:    The ids of runs going on other machines that the
                user knows have ended (see :meth:`_find_active_runs`).

        Raises:
            ExperimentError:    If there is no such experiment, or a run is
                going in it on another machine.

            PipelineError:    If a run going on this machine cannot be stopped.
        """

        experiment = self._find_experiment(name)
        home = self._get_home(name)
        with start_engine_session(self._engine) as db_session:
            run_id = record_run(db_session, self._operation_tasks["delete"]).id
            db_session.execute(
                update(Experiment)
                .where(Experiment.name == name)
                .values(state=self._deleting_state)
            )

        try:
            try:
                # The experiment's create or reset, if still going.
                copying, elsewhere = self._find_active_runs(
                    self._engine, ignored_runs, {experiment.copy_run_id}
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
                with start_engine_session(self._engine) as db_session:
                    db_session.execute(
                        update(Experiment)
                        .where(Experiment.name == name)
                        .values(state=experiment.state)
                    )
                raise
            for pipeline_run in copying + in_experiment:
                stop_run(pipeline_run)
            # Recorded as ended now, rather than whenever the original's runs
            # are next looked at; those in the experiment go with it.
            for pipeline_run in copying:
                self._finish_run(self._engine, pipeline_run.id)
            if os.path.lexists(home):
                self._remove_path(home)
            with start_engine_session(self._engine) as db_session:
                db_session.execute(
                    delete(Experiment).where(Experiment.name == name)
                )
        finally:
            self._finish_run(self._engine, run_id)
