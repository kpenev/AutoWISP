Installation
============

Requirements
------------

On Windows there is a second way in that needs none of what follows --
no Python, and nothing else to install but Docker. If that appeals, skip
ahead to `A ready-made setup for Windows`_.

Otherwise, AutoWISP is a Python program, so Python has to be on your
computer before you can install it:

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
interface. :doc:`test_data` walks through the whole sequence one command
at a time, explaining what each stage does and which of its settings
matter; :doc:`wisp_options` lists every setting there is.

A ready-made setup for Windows
------------------------------

Everything above assumes you install AutoWISP into a Python of your own.
On Windows there is an alternative that avoids all of it: AutoWISP is
also published as a Docker image, ``kpenev/wisp:latest``, with AutoWISP
and the astrometry.net solver already installed inside it, together with
a small launcher program that configures and starts that image for you.
The one thing you do need is `Docker Desktop
<https://www.docker.com/products/docker-desktop/>`_, installed and
running.

.. TODO: link the Zenodo record for the launcher download here once it
   is published, in place of the sentence below.

The launcher is distributed through Zenodo, as two files which have to
stay in the same folder as one another:

``compose.yaml``
    A description of how the container should be run: which folders on
    your computer it is allowed to see, and which port it serves the
    browser interface on.

the launcher itself
    A small window that fills that file in for you and starts
    everything.

Run the launcher. The first time, it notices that ``compose.yaml`` has
not been filled in yet and asks you to choose a **storage folder** before
letting you do anything else. This is the choice that matters: it is
where AutoWISP will keep your images and everything derived from them,
and it is the only part of your computer the container can see. Pick a
disk with room on it.

Choosing it fills in four more folders underneath, creating them if they
do not exist:

======================================  =====================================
``<storage>\tmp``                       scratch space during processing
``<storage>\BUI``                       the interface's own data and logs
``<storage>\astrometry\narrow``         narrow-field astrometry.net indices
``<storage>\astrometry\wide``           wide-field astrometry.net indices
======================================  =====================================

Any of these can be pointed elsewhere -- the two index folders in
particular, if you already have index files and would rather not copy
them. There is also a box for the port to serve on, in case something
else on your machine is already using the one picked for you. Each
change is written into ``compose.yaml`` as you make it, so the launcher
can be closed and reopened without losing anything.

Then press **Start AutoWISP**. A terminal window opens running ``docker
compose up``; the launcher waits until the interface answers and then
opens it in your browser. The first run has to download the image, so
expect it to take a while. Leave that terminal open -- closing it stops
AutoWISP.

The **Check for Update** button stops the container and fetches the
newest published image. It is worth pressing occasionally: it is what
takes the place of ``pip install --upgrade autowisp``.

Two consequences of running in a container are worth knowing:

* AutoWISP can only reach the folders listed above. Images kept anywhere
  else are invisible to it, so either put them under the storage folder
  or add a folder of your own to ``compose.yaml``.
* The astrometry.net solver comes already installed in the image, so
  `Solving locally`_ below is simpler than it reads: there is nothing to
  install, and all you have to do is put index files into those two
  folders. The launcher has already told AutoWISP where they are.

The launcher is built with `PyInstaller <https://pyinstaller.org/>`_ from
``docker/compose_gui.py`` in the repository. Nothing in it is specific to
Windows -- it is simply where this route is most useful -- so you can
equally well run that script with Python on any machine that has Docker,
keeping ``compose.yaml`` beside it. Note that the published image is
built for x86-64.

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

(If you took the Docker route above, the first of the two is already
done for you and the second has a folder waiting for it -- skip to **The
reference maps**.)

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
