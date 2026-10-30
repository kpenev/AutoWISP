*************************
When something goes wrong
*************************

A stage that fails does not simply print a complaint and vanish. The
failure is recorded in the project, along with enough of its surroundings
to work out afterwards what happened -- and, if that is not enough, to
hand the whole situation to someone else.

The error log
=============

A red button appears in the interface's navigation as soon as there is
anything to report, and is absent when there is not, so there is no need
to go looking. It leads to the list of recorded failures.

Opening one shows what went wrong: the message, the details gathered at
the moment of failure -- which stage was running, on which image, and the
files it was working with -- and the traceback. Once a failure has been
dealt with it can be marked resolved so it stops drawing attention, or
deleted outright.

Not every failure stops everything. A run that comes to grief part way
through has still recorded everything it finished, so setting processing
going again carries on from where it stopped rather than starting over.

Picking up after an interrupted run
===================================

Carrying on is not merely a matter of skipping what was finished. A run
that stops without warning -- a crash, a power cut, a job killed by hand
-- leaves whatever it was in the middle of half done, and a half-written
result is worse than none, because nothing downstream can tell it apart
from a complete one.

Rather than trying to undo everything centrally, each stage is
responsible for its own leavings. A stage says which states it can be
caught in and which it is able to resume from, and the ones that write
results in pieces provide a way to undo a partial attempt. When
processing next starts, anything recorded as begun but not finished is
handed back to the stage that began it, which clears up and reports the
state it has left things in; that state has to be one the stage can
resume from, and the pipeline refuses it otherwise. Aperture photometry,
for instance, deletes the measurements it had started writing into the
data reduction file, so the next attempt begins from a clean one.

Most stages are simple in this respect: they record only that they had
begun, so there is one state to be caught in and clearing up is
straightforward. Building light curves is the exception worth knowing
about, since it adds its points first and marks the files complete
afterwards, and so can be interrupted between the two -- in which case
the work is not thrown away, only finished off.

None of this needs anything from you. It happens as processing starts,
and the only sign is a warning in the log saying what was cleaned up.

Telling whether a run is still going
====================================

A run that stops without warning cannot record that it has stopped, so
the interface works it out: it offers to start processing only once the
last run has ended. A run started anyway, e.g. from a terminal, gives up
if another run is already processing the project; of runs started
together, the first to record itself goes on. Each run records the
machine it runs on and the process running it, with when that process
started, and a run whose process is gone is taken to have ended. The
start time is what tells the run's process from a later one given the
same number. The machine is known by an identifier the operating system
keeps for it, not by its name, which can change while a run goes on -- a
laptop renames itself as it moves between networks.

A project is processed on one machine at a time: processing it from
several, even with its database on a server they can all reach, is not
supported. A run of another machine can still turn up, in a project
moved or copied from it (see "Moving or copying a project" in
:doc:`configuration`) while a run there was going or after one was
interrupted. There is no telling from here whether such a run has ended,
so it is never taken to have: processing starts with a warning that the
run may still be going, while the experiment operations refuse until
told to ignore it (see :doc:`experiments`), and so does cloning the
project's database. Move a project only once its processing has
finished, and after an interruption, open its progress page on the
machine it is leaving first, which records the interrupted run as ended.

The logs
========

Every process writes two files into ``LOGS/`` under the project home: the
log proper, ending in ``.log``, and a second, ending in ``.outerr``,
holding whatever the process wrote to its output directly. Their names
are built from the stage, the type of image it worked on, the task, the
time and the process id, so a run leaves a separate trail for each stage
rather than one tangled file. (A ``wisp-*`` step run on its own has no
image type to name its logs by, and leaves it out.) :option:`logging-fname` and
:option:`std-out-err-fname` control the naming, and :option:`verbose`
decides how much detail goes in. A project created by an earlier version
keeps the names it was created with, which leave out the type of image:
the logs of, say, stacking the master zero and the master dark then
cannot be told apart, and the interface says so rather than showing them.
Adding ``{image_type:s}`` to both settings, after the stage, fixes that
for the runs that follow.

Reading these is worthwhile when a stage produced something odd rather
than failing outright, which is the case the error log cannot help with.

Sending a crash report
======================

If a failure is not something you can act on, the useful thing to send is
a crash report: one zip file holding the failure and everything around
it. The button is on the error's own page, labelled **Download crash
report**. The same thing from a terminal::

    wisp-crash-report /path/to/project/home --last

Name an error explicitly instead of ``--last`` to report on an older one;
the number is the one the error log shows.

The zip holds the error record, the details sidecar with the full
traceback, the logs belonging to that run, a copy of the project
database, and a note of the platform, Python and package versions in use.
A ``manifest.json`` lists what was gathered. The database copy is what
makes the report worth sending: it carries the configuration in force,
what had been processed already, and the equipment involved, which is
usually the difference between a guess and an answer.

**Credentials are removed.** Scrubbing is not optional and nothing is
written unscrubbed: values whose names mark them as secret -- the Gaia
archive login, the astrometry.net API key, anything called a password or
a token -- are replaced with ``***REDACTED***`` wherever they appear, in
the logs, in the configuration and in the copied database alike.

Collection is deliberately best-effort. A file that cannot be read, or
that is not there, is written down as a gap in the manifest rather than
being allowed to abandon the report -- so a report still arrives in the
case where the missing file is itself the problem. Logs are truncated to
the beginning and end of each file, half a megabyte in all; pass
``--max-log-bytes`` for more if it is asked for.

Keeping the log from growing
============================

Recorded failures stay until removed. To prune them::

    wisp-cleanup-errors /path/to/project/home --older-than 30d

The cutoff takes days, hours or weeks (``30d``, ``12h``, ``2w``). Run
without it, the command leaves the records alone and only tidies up after
itself, clearing sidecar files with no error still pointing at them and
references to sidecars that have gone.
