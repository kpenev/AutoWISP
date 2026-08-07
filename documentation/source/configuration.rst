****************************
How a setting gets its value
****************************

:doc:`wisp_options` lists every setting there is and says what each one
means. It does not say how AutoWISP decides what a setting *is* for any
particular image, which is a separate question and occasionally a
puzzling one -- settings can hold different values for different images,
some never appear in your configuration at all, and leaving one blank
does not mean what it looks like it means.

Settings belong to the project
==============================

A setting is not something you pass to a command. It is a row in the
project's database, written when the project is created: AutoWISP asks
each processing stage what it can be told, and records every answer. That
is why the options reference is generated from a freshly made project
rather than written by hand, and why a setting exists whether or not you
ever give it a value.

The important consequence is that **a setting belongs to the project, not
to a stage, and many settings are used by several stages at once**. There
is one :option:`verbose`, and it decides how much every stage writes to
its log; likewise the rest of the logging settings. There is one
:option:`num-parallel-processes`, and it decides how much work at a time
each stage that can use several processes will do.

That extends to settings that matter a great deal more than logging does:

* :option:`data-reduction-fname` says where each image's data reduction
  file lives. It is shared by ``find_stars``, ``fit_star_shape`` and
  ``measure_aperture_photometry`` -- and it has to be, since one writes
  what the others go on to read. Were they able to disagree, the later
  stages would simply not find the files.
* :option:`variables` names the light curve quantities that
  :option:`lc-points-filter-expression` and the fitting expressions are
  written in terms of. It is shared by all four detrending stages: EPD,
  TFA and the two that report on how well each of them did. If the
  reporting stages understood the names differently from the correcting
  ones, the statistics would describe something other than what was
  corrected.

Sharing is the point -- you say a thing once and every stage that needs
it is told -- but it does mean there is no such thing as changing a
setting "for one stage". Change it and everything using it changes.

This is worth keeping in mind because the configuration editor invites
the opposite impression. It can be narrowed to a single stage, so that
you see only the settings that stage uses instead of several hundred at
once. That filter decides what is shown and nothing else: values are
stored against the setting, not against the stage you happened to be
looking at, so editing one while filtered to ``tfa`` changes it for every
other stage that uses it too.

Running a stage yourself with ``wisp-calibrate`` and the like is the
exception to all of this. Those commands read a configuration file and
their own command line, not the database, which is what makes them useful
for one-off runs but also means a value you set there affects nothing
else.

A value, and when it applies
============================

Each value is stored together with a **condition**: a set of expressions
over the image header, all of which must hold for that value to be used.
Every project starts with one condition whose expression is simply
``True``, so a value stored against it applies to every image -- which is
what a setting with a single, unconditional value amounts to.

Give a setting more than one value and you are choosing between them by
condition. When a stage is about to run on an image, AutoWISP evaluates
the conditions' expressions against that image's header, and a value
applies if everything its condition asks for came out true. The values
are tried **in the order they are listed** and the first that applies
wins, so a general fallback belongs last: put it first and nothing after
it is ever reached. If no value applies at all, the run stops with an
error naming the setting, rather than guessing.

Two things about the matching are worth knowing. It happens per image,
so conditions can key off anything in the header -- the target, the
exposure length, the observing session. And it happens per colour
channel as well, so the same setting can take different values for the
red and blue channels of the same image.

Where the values come from, and what "no value" means
=====================================================

Every setting has a value from the moment a project exists. As the
project is created each one is given the value its stage would have used
by itself, and that is what goes into the database. You will want to
change a fair number of them, and most of this documentation is about
which ones and why; the point is that none of them are empty waiting for
you, and the stored value is what applies until you replace it.

For a fair number of settings, the value a stage would have used is
*nothing at all*. That is a real state in its own right, and not the same
as an empty box: such a setting is left out of the configuration handed
to the stage, and the stage then decides for itself when it runs.
:option:`epd-datasets` and :option:`tfa-datasets` are the clearest
example. Absent, they do not detrend nothing -- they detrend every
aperture the photometry produced, worked out from the photometric
reference at the time.

The distinction matters because only one of the two can be typed. The
editor's value boxes hold text, so clearing one leaves an empty string,
which is a value like any other rather than an absence. For settings
where an empty value means nothing sensible, that is an error and not a
fallback: emptying :option:`tfa-datasets` does not restore the behaviour
described above, it stops the step with a complaint that ``''`` cannot be
made sense of as a dataset to detrend.

Restoring a value to absent is therefore not something the value boxes
can do. It has to be done through the exported configuration described
next, where an absence appears as ``null``.

Carrying configuration between projects
=======================================

The configuration editor can write the whole configuration out as a JSON
file and read one back in. Importing merges what the file contains into
what the project already has and hands you the result in the editor, so
you can look it over -- and, if it is not what you wanted, leave without
saving. Nothing changes until you save.

That is the way to set up a second project like the first, to keep a copy
of a configuration that works, and to make the sort of edit the value
boxes cannot express.

**Use the JSON rather than a configuration file for this.** The two are
not equivalent. A configuration file has one line per setting, and no way
to say "this value under these conditions"; the JSON is a tree, and the
conditions are the branches. Anything conditional survives the JSON round
trip and cannot be written in a configuration file at all.

The same limitation applies to the configuration you can paste in when
creating a project. It is read as ordinary configuration lines, so
everything it sets is unconditional -- convenient for carrying the
settings of an instrument to a new project, but not the conditions you
may have built on top of them. Those have to come from an import
afterwards.

The configuration files that the individual ``wisp-*`` commands take are
different again: each is a flat list of settings for one run, which is
all such a command needs, since it works on what you hand it rather than
choosing per image.

Settings you will not find in your configuration
================================================

Some of what a stage reads is never stored as a setting at all. Anything
naming a master file -- the master bias, the single photometric
reference, the catalog collected for detrending -- is filled in after
your configuration has been read, by looking up which master matches the
image at hand in the tables AutoWISP maintains for the purpose. Those
settings appear in the options reference, because the stages accept them,
but not in the configuration you edit, and there is nothing to set: the
answer depends on the image.

This is also why such a setting behaves differently when you run a stage
by hand. There is no master lookup on the command line, so
:option:`single-photref-dr-fname` and its like have to be given
explicitly, and the test data's configuration file duly gives them.

What the values themselves are
==============================

Every value is stored as text, and what becomes of that text depends on
the setting. Most are used as they stand, or converted to a number or a
list. Some -- those identifying equipment, the observing time, the target
-- are expressions evaluated against the image header, which is what lets
them cope with headers that use different keywords for the same thing.
:doc:`bringing_your_own_data` covers those, including the quoting needed
to make one produce a fixed string rather than read a keyword.

A note on versions
==================

Values carry a version number, and the configuration editor offers a
version to work on. The intent is that a project's configuration can move
forward while what earlier runs used stays on record, so that results can
be traced back to the settings that produced them.

Treat it as unfinished. It has had no real use and is not exercised by
the tests, so it is not something to build a way of working around yet.
Leaving the version alone and editing the configuration in place is the
trodden path.
