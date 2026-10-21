***********
Experiments
***********

Trying a different configuration on a project that has already been
processed is risky: if the new settings turn out worse, getting back to
where you were means reprocessing. An experiment avoids that. It is a copy
of the project, made inside it, which you process as you like while the
original stays exactly as it was. If the experiment turns out better, you
merge it back and the original becomes a copy of it; if not, you delete it
and nothing has changed.

Everything here is done with one command, run on the original::

    wisp-experiments <project home> <subcommand> ...

``wisp-experiments --help`` lists the subcommands, and ``--help`` after
any of them describes it.

Creating an experiment
======================

::

    wisp-experiments <project home> create <name>

The experiment lives in ``experiments/<name>`` under the original's home.
Its name may use letters, digits, ``-`` and ``_``. Creating it copies
everything under the original's home except ``experiments/`` itself, then
copies the original's database. Raw images inside the project are
hard-linked rather than copied, since nothing ever writes to them, so they
cost no space; everything else is copied, because processing writes to
its files in place. The free space is checked before anything is copied.

From then on the experiment is an ordinary project in its own right:
configure it, process it and look at its results as you would any other,
through its home.

Paths must stay inside the project
----------------------------------

An experiment and its original share whatever lies outside their homes,
so a step writing there in one would overwrite the other's results. For
that reason a project can only have experiments if every path its
configuration names, under every condition, starts with ``{PROJHOME}``.
Creating an experiment is refused otherwise, with the offending
parameters named. The exceptions are the paths of things AutoWISP only
reads or keeps apart by itself: :option:`anet-indices`,
:option:`subpixmap` and :option:`tempstore-dir`.

The same check is made each time processing starts in an experiment, so
pointing an experiment's configuration outside it is refused there too.

The lock
========

Merging replaces the original with the experiment, so anything that
changed the original meanwhile would be lost. While it has an experiment
that may still be merged, the original is therefore locked: it refuses to
start processing, to take new images or to save configuration changes,
and the interface says so. The experiment itself is not locked.

An experiment releases the lock when it is merged or deleted. To work on
the original while keeping an experiment, release the lock by hand::

    wisp-experiments <project home> unlock

This marks every current experiment *stale*: it is kept, with its
configuration, but can no longer be merged as it is, since the original
may now change. Resetting it (below) brings it up to date and locks the
original again.

What the lock cannot catch
--------------------------

The lock lives in the project's database, and only what records itself
there is subject to it. A step run on its own from a terminal (one of the
``wisp-*`` step commands, rather than processing started from the
interface) records nothing, so nothing stops it, and nothing notices what
it changed. Running one in an original that has experiments, or while an
experiment is being created or merged, is not supported.

The interface finds an experiment only through its original, and does not
open one while an operation is working on it. A command given the
experiment's home directly is not stopped that way: leave an experiment
alone while it is being reset, merged or deleted.

Listing experiments
===================

::

    wisp-experiments <project home> list

Each experiment is listed with its state and when it was created:

``current``
    A complete copy of the original, which has not changed since.

``stale``
    A complete copy, but the original may have changed since, because of
    ``unlock``. It can be reset, or merged by force.

``creating``, ``merging``, ``deleting``
    An operation is working on it. The list says whether that operation
    is still running, here or on another machine, or was interrupted, and
    in that case what to do next.

Resetting an experiment
=======================

::

    wisp-experiments <project home> reset <name>

Resetting starts an experiment's processing over: its files and database
are replaced by the original's, except its configuration, which it keeps.
This is how to try the same idea again from a clean start, and how to
bring a stale experiment up to date.

Both projects must be at the version of AutoWISP in use: an experiment is
migrated on its own when opened, or with ``wisp-migrate <experiment
home>``, and the reset says which project needs it. A reset is also
refused if the experiment lacks a configuration condition that the
original uses to match its master frames to images.

Merging an experiment
=====================

::

    wisp-experiments <project home> merge <name>

Merging makes the original a copy of the experiment: its files, its
processing and its configuration. The original keeps its record of its
experiments, and the experiment, now an exact copy of the new original,
stays current. Every other current experiment becomes stale.

A stale experiment is refused, as is one whose original has changed since
it was copied, since merging would lose that change. ``--force`` merges it
anyway, losing the change. Even forced, a merge is refused if the original
was processed after the copy, since that is processing the experiment
does not have.

Before anything is replaced, the original is backed up into
``experiments/.pre-merge-<time>-run<id>``. Its files are moved there, not
copied, so this is quick and needs no space; its raw images stay where
they are. Its database is copied there as ``autowisp.db``. If the merge
then fails, everything is put back as it was. Once satisfied that the
merge is right, delete the backup::

    wisp-experiments <project home> delete-backup

A later merge deletes earlier backups by itself.

The error log after a merge is the experiment's: failures recorded in the
original since the copy, including those of creating or deleting other
experiments, remain only in the backup.

An interrupted merge
--------------------

A merge stopped part way, by a crash or a power cut, leaves the
experiment ``merging`` and the original's files partly in the backup. The
original stays locked, and the experiment cannot be deleted, since its
record is the only sign that the original needs putting back. Restore it::

    wisp-experiments <project home> restore <name>

This puts the original's files back from the backup, wherever the merge
stopped. The original's database needs nothing: the merge had not
finished replacing it. Never stop a merge that is still running; let it
finish or fail, which puts everything back by itself.

Restoring relies on each file being moved in one step, which a crash
leaves either in the original or in the backup. The filesystems normally
used for data (ext4, XFS, Btrfs, APFS, NTFS) guarantee that. On one that
does not (FAT, exFAT, ext2), a crash during a merge may leave files the
restore cannot place, and it stops naming them; keep projects with
experiments on a filesystem that does.

Deleting an experiment
======================

::

    wisp-experiments <project home> delete <name>

Deleting removes the experiment's whole directory and its record, after
asking (``--yes`` skips the question). Processing running in the
experiment on this machine is stopped first, as is the experiment's
create or reset, if still running. An experiment in any state can be
deleted, one left behind by an interrupted operation included, except
while it is being merged.

A processing step run on its own in the experiment (see above) cannot be
found, and fails once its files are gone.

Deleting a project from the interface deletes its experiments first. If
any of them cannot be deleted, the project is left as it is and the
reason is shown.

Runs on other machines
======================

Several machines can work on a project that keeps its database on a
server. An operation that finds a run going on another machine refuses,
since it can neither stop that run nor tell whether it has ended, and
names the run by its id. If you know the run has ended without saying so,
for instance because its machine crashed, name it to be ignored::

    wisp-experiments <project home> merge <name> --ignore-run <id>

Every subcommand that checks for runs takes ``--ignore-run``, as many
times as needed.

Refusals and failures are recorded in the original's error log, like any
other (see :doc:`when_things_go_wrong`).

Upgrading AutoWISP
==================

Opening a project with a newer AutoWISP migrates its database, which
changes it, and would make every experiment unmergeable for nothing. So
migrating an original keeps its current experiments current: each is
migrated in turn when next opened, and remains a copy of the original. An
experiment that was already out of date is reported stale by the
migration. ``wisp-experiments`` migrates the original it is given as any
command opening a project does; a server database needs
``--assume-backed-up`` (see ``wisp-migrate``).
