"""Copy, move and remove the files of a project's tree.

What creating, resetting and merging experiments
(:mod:`autowisp.project_experiments`) do to the projects' files. Their
databases' files are always left out: those are copied by
:func:`~autowisp.database.project_copy.copy_project_database`.
"""

import os
import shutil
import stat
import sys
from pathlib import PurePath

from autowisp.database.interface import DB_URL_FNAME
from autowisp.exceptions import ResourceError


def _is_database_file(name):
    """
    Is a name in a project's home one of its database's files?

    The database, SQLite's journals beside it, the backups migrating it
    leaves (``autowisp.db.pre-<revision>``), and the file naming a server
    database.
    """

    return name in ("autowisp.db", DB_URL_FNAME) or name.startswith(
        ("autowisp.db-", "autowisp.db.pre-")
    )


def _list_tree(home, excluded_dir):
    """
    Return the directories and files a copy of a project's tree takes.

    Everything under the home, following symbolic links, except at the top
    the database's files (:func:`_is_database_file`) and the excluded
    directory.

    Args:
        home(str):    The project's home.

        excluded_dir(str):    The name of the directory at the top left out:
            the experiments'.

    Returns:
        [PurePath]:    The directories, relative to the home, the home itself
        included.

        [PurePath]:    The files, relative to the home.
    """

    directories = []
    fnames = []
    for directory, subdirs, files in os.walk(home, followlinks=True):
        if directory == home:
            subdirs[:] = [
                name
                for name in subdirs
                if name != excluded_dir and not _is_database_file(name)
            ]
            files = [name for name in files if not _is_database_file(name)]
        relative_dir = PurePath(directory).relative_to(home)
        directories.append(relative_dir)
        fnames.extend(relative_dir / name for name in files)
    return directories, fnames


def copy_tree(source_home, target_home, linked_files, excluded_dir):
    """
    Copy a project's files into another home, all but its database's.

    What is copied is listed by :func:`_list_tree`. Symbolic links are
    followed, so that nothing is shared through one, as nothing may be
    through a path configured outside the project.

    Files are copied with their metadata (``shutil.copy2``), over any already
    in the target: steps append to DR and lightcurve files in place, so each
    project needs its own. Only the files nothing writes, the raw images, are
    hard-linked instead, or copied where the filesystem cannot link them. One
    already in the target is left as it is, being the same file or a copy of
    it.

    The space the copies take is checked against the target's free space
    before anything is copied.

    Args:
        source_home(str):    The home of the project copied.

        target_home(str):    The home to copy it into, which exists.

        linked_files(set):    The files to hard-link, each a ``PurePath``
            relative to ``source_home``.

        excluded_dir(str):    The name of the directory at the top of the
            source left out: the experiments'.

    Returns:
        None

    Raises:
        ResourceError:    If the copies would not fit in the free space.
    """

    directories, fnames = _list_tree(source_home, excluded_dir)
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


def remove_path(path):
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


def clear_tree(home):
    """
    Remove everything in a project's home but its database's files.

    Resetting an experiment does this before copying its original's files
    in, so that the experiment's tree mirrors the original's.
    """

    for name in os.listdir(home):
        if not _is_database_file(name):
            remove_path(os.path.join(home, name))


def list_movable(home, kept_files, excluded_dir):
    """
    Return what moving a project's tree aside moves, relative to its home.

    Everything under the home except, at the top, the database's files
    (:func:`_is_database_file`) and the excluded directory, and except the
    files kept. Each entry is the topmost holding nothing kept, so that a
    directory is moved whole where it can be. Symbolic links are moved as
    they are, never followed.

    Args:
        home(str):    The project's home.

        kept_files(set):    The files left in place, each a ``PurePath``
            relative to the home.

        excluded_dir(str):    The name of the directory at the top left in
            place: the experiments'.

    Returns:
        [PurePath]:    The files and directories to move, relative to the
        home.
    """

    kept_dirs = {
        directory for fname in kept_files for directory in fname.parents
    }
    movable = []
    pending = [PurePath()]
    while pending:
        directory = pending.pop()
        for name in os.listdir(os.path.join(home, directory)):
            relative = directory / name
            if relative in kept_files or (
                directory == PurePath()
                and (name == excluded_dir or _is_database_file(name))
            ):
                continue
            path = os.path.join(home, relative)
            if (
                relative in kept_dirs
                and os.path.isdir(path)
                and not os.path.islink(path)
            ):
                pending.append(relative)
            else:
                movable.append(relative)
    return movable


def move_tree(source_home, target_home, kept_files, excluded_dir):
    """
    Move a project's files into another directory, by renaming.

    What is moved is listed by :func:`list_movable`, and keeps its path
    relative to the home. The target must not hold any of it already, and
    must be on the same filesystem.

    Args:
        source_home(str):    The home to move the files out of.

        target_home(str):    The directory to move them into.

        kept_files(set):    The files left in place, each a ``PurePath``
            relative to the home.

        excluded_dir(str):    The name of the directory at the top left in
            place: the experiments'.

    Returns:
        None
    """

    for relative in list_movable(source_home, kept_files, excluded_dir):
        target = os.path.join(target_home, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.rename(os.path.join(source_home, relative), target)
