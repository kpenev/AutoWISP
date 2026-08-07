Installation
============

Requirements
------------

AutoWISP is a Python program, so Python has to be on your computer before
you can install it:

* **Python 3.11 or newer** -- any newer version will do. To find out
  whether you already have it, open a terminal and run ``python3
  --version``. If that reports something older than 3.11, or the command
  is not found at all, install a current Python from `python.org
  <https://www.python.org/downloads/>`_.
* **Linux, macOS or Windows**, on any processor in common use today.

Everything else AutoWISP needs comes with it. Parts of it are written in
C++ for speed, but those arrive ready-built, so there is nothing to
compile and no other software to install first.

One thing does need attention before you can process any images. AutoWISP
has to work out where in the sky each of your images points, and for that
it needs either a program installed alongside it or an account on a free
web service. `Plate solving`_ below explains both; you will have to pick
one. `Gaia archive access`_, further down, is genuinely optional.

Installing the released version
-------------------------------

Python comes with a program called `pip <https://pip.pypa.io/en/stable/>`_
whose job is to fetch and install Python software. Open a terminal and
type::

    pip install autowisp

That downloads AutoWISP together with everything it relies on, which may
take a couple of minutes.

It is worth, though not necessary, installing AutoWISP into a *virtual
environment* first. That is a self-contained folder holding AutoWISP and
the packages it uses, kept apart from the rest of the Python software on
your computer so the two cannot interfere with one another::

    python3 -m venv autowisp-env
    source autowisp-env/bin/activate      # Windows: autowisp-env\Scripts\activate
    pip install autowisp

The middle line is what switches your terminal over to using that folder,
and it has to be repeated in every new terminal window from which you
intend to run AutoWISP.

What you just installed
-----------------------

The installation gives you three ways in, and most users only need the
first:

**The browser interface.** ``wisp-bui`` is the intended way to use
AutoWISP: it is where you create a project, point it at your images, set
the configuration described in the rest of this page, start processing,
and watch it progress. To start it, open a terminal and run::

    wisp-bui

It works on your computer generating a web-page with which you interact,
and opens that page in a browser tab for you. Nothing is sent anywhere:
the page is served by your own computer and only your own computer can
reach it. Leave the terminal open while you work -- closing it, or
pressing Ctrl-C in it, shuts the interface down. (Should you need the
page at a particular address, pass ``--hostname [<host>:]<port>``;
otherwise a free port is picked for you.) Nothing else in this section is
required to get going.

**The pipeline.** This is the part that does the real work on your
images: correcting them for the behaviour of the camera, finding the
stars, working out where each image points, measuring how bright every
star is, and gathering those measurements into a brightness history for
each star. It keeps track of which of those stages have been done for
which images, does whatever is still outstanding in the order the stages
depend on one another, and works on several images at once. Because the
record of what is finished is kept on disk, a run that you stop, or that
comes to grief part way through, carries on from where it got to rather
than starting over. Setting processing going in the browser interface is
what launches it.

**The individual steps.** Each stage of the pipeline is also a command in
its own right -- ``wisp-calibrate``, ``wisp-find-stars``,
``wisp-solve-astrometry`` and so on. These exist for running a single
stage by hand, or for stringing together a sequence of your own in a
script; you can ignore them entirely if you work through the browser
interface. See :doc:`processing_steps` for what each stage does, and
:doc:`wisp_options` for the settings they have in common.

Installing from source
----------------------

You can skip this section unless you want to run code that has not been
released yet, or to make changes to AutoWISP yourself. It assumes you are
comfortable with ``git``.

Install from a clone of the repository::

    git clone https://github.com/kpenev/AutoWISP.git
    cd AutoWISP
    pip install .

If you are making changes to AutoWISP, install in editable mode instead::

    pip install -e .

This matters for more than convenience: the ``wisp-*`` commands are
installed entry points, so a plain ``pip install .`` runs the code as it
was at install time. Without ``-e``, changes to your working copy are not
picked up until you reinstall.

Plate solving
-------------

Before AutoWISP can make anything of an image it has to know precisely
which patch of sky the image covers and where on the sensor each star
falls. It works that out by comparing the pattern of stars it has
detected against a reference map of the whole sky -- much as you might
recognise a constellation. This is called *plate solving*, and AutoWISP
leaves the job to a well-established program called `astrometry.net
<https://astrometry.net>`_.

There are two ways of running it: on your own computer, or through the
astrometry.net web service. **One of the two has to be set up before you
can process any images** -- there is no arrangement in which neither is
needed.

AutoWISP solves on your own computer when both of the following are true,
and turns to the web service otherwise:

#. the two folders of reference maps you pointed it at (``anet-indices``,
   described below) both exist, and
#. the ``solve-field`` program can be run.

Solving locally
~~~~~~~~~~~~~~~

Solving on your own computer is far faster, and puts no limit on how many
images you may solve, so it is the better choice for anything beyond a
handful. It needs two things.

**The solver itself**, which your operating system can install for you:

.. code-block:: bash

    # Debian / Ubuntu
    sudo apt-get install astrometry.net

    # macOS, with Homebrew
    brew install astrometry-net

On **Windows** there is no version of ``solve-field`` of its own. It comes
instead as part of `ANSVR <https://adgsoftware.com/ansvr/>`_, which runs
it inside a small Linux-like environment, with the result that Windows
never sees it as an ordinary program. AutoWISP knows this and looks for
ANSVR where it installs itself by default,
``%LOCALAPPDATA%\cygwin_ansvr\bin\bash.exe``. If you put ANSVR somewhere
else, say where by setting the ``ANSVR_BASH`` environment variable.

**The reference maps**, known as *index files*. These are what the solver
compares your images against. They are large, so they come separately and
you download only the ones you need: which those are depends on how much
sky a single one of your images covers, and the `astrometry.net
documentation <https://astrometry.net/doc/readme.html>`_ explains how the
sets are numbered and which to choose. Put them in two folders -- one for
the maps covering small areas of sky, one for large -- and then tell
AutoWISP where both folders are.

Any setting AutoWISP has, this one included, can be given in four
different ways. Use whichever suits you:

*In the browser interface*, on the settings page for the
``solve_astrometry`` stage. This is the usual choice: what you enter is
remembered as part of the project, so it applies to every run without
having to be given again.

*In a configuration file* -- a plain text file of settings, one per line,
which you hand to AutoWISP with ``--config-file``. A setting is written
without its leading dashes, and the two folders as a bracketed list::

    anet-indices = [/path/to/narrow, /path/to/wide]

*As an environment variable*, that is, a setting your terminal hands on
to every program started from it::

    export AUTOWISP_ANET_INDICES="[/path/to/narrow, /path/to/wide]"

Note the brackets and the comma -- the same list form as in the file.

*On the command line*, when running one of the individual stage
commands::

    wisp-solve-astrometry --anet-indices /path/to/narrow /path/to/wide ...

One thing to watch for, whichever way you choose: if either folder does
not exist -- and a typo in the path is enough -- AutoWISP quietly falls
back on the web service rather than complaining. What you notice is
processing that is unaccountably slow, not an error message.

Solving over the web
~~~~~~~~~~~~~~~~~~~~

This is what AutoWISP falls back on whenever it cannot solve locally. It
is not, however, something that works straight away: the service requires
an account, and AutoWISP signs in before submitting anything, so **a key
is not optional here**. Create a free account at
https://nova.astrometry.net, copy the key shown on your ``Profile`` page,
and give it to AutoWISP as ``--anet-api-key``, by any of the four means
described above -- for example::

    wisp-solve-astrometry --anet-api-key YOUR_KEY ...

Without a valid key the sign-in fails and no image can be solved.

Bear in mind also that the service takes one image at a time from each
user, and can spend a minute or more on each, so a single night's images
may keep it busy for many hours.

Gaia archive access
-------------------

To measure how bright a star is, AutoWISP first has to know which stars
to expect in an image and what is already known about them. It looks them
up, over the internet as it works, in the Gaia catalog -- a survey of
more than a billion stars published by the European Space Agency.

No account is needed for this: anyone may consult the archive
anonymously. Anonymous users are, however, held to tighter limits on how
much a single query may return and how many queries may run at once. If
you find yourself running into those, register for a free Gaia account
and give AutoWISP the username and password as ``--gaia-user`` and
``--gaia-password``, by any of the four means described above.

Each lookup is saved on your computer and reused, so a given patch of sky
is only ever fetched once, however many of your images cover it.

Run the tests (optional)
------------------------

AutoWISP comes with a set of self-checks that put a small set of images
through every processing stage and confirm the results come out as they
should. Running them is a good way of satisfying yourself that the
installation is sound::

    python3 -m autowisp.tests failed_test -vvvv

``failed_test`` is the name of a folder to put aside the working files of
any check that fails, so the failure can be looked into afterwards. If
nothing fails, no such folder is created.

The images to test on are fetched the first time you run this, so it
needs a working internet connection, and expect it to take a while.

It may take a while to run all the tests, so please be patient. In
particular, if you have not installed a local ``solve-field`` (see `Plate
solving`_), the astrometry tests go through the astrometry.net web
interface, which accepts only one image at a time per user and can take a
minute or more per image.
