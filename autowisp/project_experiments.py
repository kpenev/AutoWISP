"""Create, reset, merge and delete a project's experiments.

The operations copy whole projects: their files, and their databases with
:mod:`autowisp.database.project_copy`. What they record in the original's
database is managed by
:class:`~autowisp.database.experiments.ExperimentRecords`, which the database
layer uses on its own, being unable to import the steps and the copy.
"""

import os
import shutil
import stat
import sys
from pathlib import PurePath

from sqlalchemy import select

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Configuration, Parameter

# pylint: enable=no-name-in-module
from autowisp.database.experiments import ExperimentRecords
from autowisp.database.interface import DB_URL_FNAME
from autowisp.exceptions import ConfigurationError, ResourceError
from autowisp.processing_steps import get_path_arguments
from autowisp.processing_steps.manual_util import split_path_values
from autowisp.project_paths import PROJHOME_MARKER


class ProjectExperiments(ExperimentRecords):
    """
    The experiments of one project, and every operation on them.

    Adds to the records what reaches beyond the database: the configured
    paths, and the projects' files.
    """

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
        with self._begin_session() as db_session:
            if self._read_experiments(db_session) is None:
                self._refuse_external_paths(db_session)
