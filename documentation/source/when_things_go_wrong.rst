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

The logs
========

Every process writes two files into ``logs/`` under the project home: the
log proper, and a second holding whatever the process wrote to its output
directly. Their names are built from the stage, the task, the time and
the process id, so a run leaves a separate trail for each stage rather
than one tangled file. :option:`logging-fname` and
:option:`std-out-err-fname` control the naming, and :option:`verbose`
decides how much detail goes in.

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
