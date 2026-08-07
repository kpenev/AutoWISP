**********************
Bringing your own data
**********************

The test data arrives with every decision already made for it: the images
are sorted, the camera and lens it was taken with are described, and the
settings that tell AutoWISP how to read the FITS headers were written by
the people who produced it. Your own images come with none of that
settled, and the pipeline cannot start until it is.

There are four things to arrange, and they are worth doing in this order:

#. sort the images by what kind of frame each one is;
#. describe the equipment that took them;
#. tell AutoWISP how to find, in your headers, the handful of quantities
   it needs;
#. decide which calibration frames you have, and what makes two frames
   similar enough to share a master.

All of it is done through the browser interface, and the built-in
tutorial (see :doc:`installation`) shows where each part lives. This page
explains what the choices mean, which the tutorial does not stop to do.

Sort the images by type
=======================

AutoWISP recognises four kinds of frame, and the simplest arrangement --
the one the test data uses -- is a directory per kind:

* **zero** -- bias frames: exposures as near zero length as the camera
  allows, measuring the level every pixel starts from before any light
  falls on it.
* **dark** -- exposures as long as your science frames but with no light
  reaching the detector, measuring the charge that collects in each pixel
  by itself.
* **flat** -- images of something as uniformly bright as you can manage,
  measuring how sensitive the system is to light arriving from each
  direction.
* **object** -- the images of the sky you actually want photometry from.

Sorting them before you start is a great deal easier than untangling them
afterwards. You do not have to sort them by hand, though. What decides an
image's type is :option:`image-type`: an expression evaluated against the
header, whose value is the name of one of the types above. If your camera
records the kind of frame in a keyword, ``IMAGETYP.lower()`` is often all
it takes.

A single expression need not cover every case. Like any other setting,
:option:`image-type` can be given different values under different
conditions, so frames that have to be recognised by different means --
one batch identified by a keyword, another by exposure length, a third by
which directory it came from -- can each get the expression that suits
them.

Whatever the expression produces is compared, in lower case, against the
types the project actually has. Anything else is an error, on the
principle that a frame nobody can identify is more likely to be a mistake
in the expression than a frame worth skipping. Set
:option:`ignore-unknown-image-types` if you would rather such frames were
passed over quietly.

Describe the equipment
======================

AutoWISP keeps a record of the instruments a project uses -- the
**survey** -- and every image is tied to entries in it. This is what lets
the pipeline tell two cameras apart when deciding which master bias
belongs to which frame, and it is why a project will not accept images
from hardware it has never been told about.

The survey holds five kinds of entry: **Cameras**, **Telescopes**,
**Mounts**, **Observers** and **Observatories**. Cameras, telescopes and
mounts are described twice over: once as a model (make, model, focal
length, pixel size and so forth) and then as one or more **devices** --
the individual physical units, each with a serial number. The serial
number is the part that matters most, because that is what your image
headers are matched against.

Despite the name, it need not be the manufacturer's serial number, or a
number at all. It is simply a label that tells one unit of a kind apart
from the others in the project, and any string will do as long as it is
unique among the devices of that kind and your headers can be made to
produce it. If the real serial number is recorded in your images, using
it saves inventing anything; if it is not -- which is common -- pick a
name that means something to you.

You can build the survey in the browser interface under **Edit Survey**,
or write it as JSON and import it::

    wisp-survey /path/to/project/home import -f survey_instruments.json

The same command exports what a project already has, which is the easiest
way to see the format and to copy a survey between projects::

    wisp-survey /path/to/project/home export -f survey_instruments.json

A camera entry additionally describes its **channels**. A monochrome
detector has one; a colour sensor has one per colour in its filter
mosaic, each defined by which pixels belong to it -- an offset and a step
in x and in y. Every stage of the pipeline runs on each channel
separately, which is why results are labelled by channel throughout, and
why a colour camera produces several light curves per star rather than
one.

If your images were taken with equipment whose serial number is not
recorded anywhere in the header -- which is common -- give the device any
serial number you like in the survey and have the expression in the next
section produce that same string.

Tell AutoWISP how to read your headers
======================================

FITS headers agree on very little between one camera and the next, so
AutoWISP does not assume keyword names. Instead each quantity it needs is
configured as an expression evaluated against the header, and the
defaults are simply the keywords that happened to suit the cameras it was
first used with:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Default expression
   * - :option:`camera-serial-number`
     - ``CAMSN``
   * - :option:`telescope-serial-number`
     - ``INTSN``
   * - :option:`mount-serial-number`
     - ``OBSERVER``
   * - :option:`observer`
     - ``ORIGIN``
   * - :option:`observatory-location`
     - ``LAT_OBS``, ``LONG_OBS``, ``ALT_OBS``
   * - :option:`target-ra`
     - ``RA_MNT``
   * - :option:`target-dec`
     - ``DEC_MNT``
   * - :option:`target-name`
     - ``FIELD``
   * - :option:`exposure-start-utc`
     - ``DATE_OBS + "T" + TIME_OBS``
   * - :option:`exposure-seconds`
     - ``EXPTIME``

An expression is not limited to naming a keyword. It is ordinary Python
evaluated with the header keywords as variables, so anything you can
compute from them is available. These are all from a working project::

    exposure-start-utc = IMAGEID.split("_")[-1]
    mount-serial-number = SEQID[SEQID.find('PAN'):SEQID.find('PAN')+6]
    observatory-location = [LAT_OBS, LONG_OBS, ELEV_OBS]

The first pulls a timestamp out of a compound identifier; the second digs
a mount name out of the middle of a sequence identifier; the third is
there only because that camera writes ``ELEV_OBS`` where the default
expects ``ALT_OBS``.

The serial numbers and the observer are matched against the survey, and
the values your expressions produce have to be entries that already
exist. An observatory can be given by name instead
(:option:`observatory`), in which case the location is not consulted; if
you go by location, the matching observatory has to be within about
100 km.

If you own one camera, one lens and one mount -- much the commonest
situation -- there is nothing to tell apart and the matching has no work
to do. You can switch it off by having the expression ignore the header
and return a fixed string, which then only has to agree with the name you
gave that device in the survey. In the browser interface, type the name
into the field *with quotes around it*::

    'my camera'

Writing it in a configuration file instead takes two sets of quotes, one
of which is stripped when the file is read::

    camera-serial-number = "'my camera'"

The quotes are not decoration. These settings are expressions, not
plain text, so a bare ``my camera`` is read as the name of a header
keyword; no such keyword exists and evaluation fails. Quoted, it is a
string, and the same string for every image.

Getting these right is the fiddliest part of setting up a project, and
the fastest way to do it is to open one of your own files and read its
header before you start guessing.

Decide which calibrations you have
==================================

When a project is created you say which of the three calibration frame
types you can supply. This is not merely a default that can be revisited
later: the types you turn off are removed from the project altogether,
along with the processing stages that would have produced them. A project
created without flats has no flat image type and no flat master, and will
not acquire them afterwards.

Say no to all three and processing still works. The photometry will not
be as good as it could have been, but for many applications -- possibly
most -- collecting calibration data is impractical, and doing without is
a perfectly ordinary way to use AutoWISP.

For each type you keep, two lists of header expressions control how
masters are matched to images, and they are easy to confuse:

* **split** decides which frames are combined *into* one master. Frames
  differing in any of these expressions go into separate masters.
  Splitting the bias frames by observing session, for instance, gives one
  master bias per night rather than one for the whole campaign.
* **match** decides which master is *applied to* a given image. A master
  is eligible for an image only if all of these expressions agree between
  them -- the same camera and the same colour channel, typically.

Both default to sensible choices and most projects never touch them. What
they must not be is too narrow: split by something that varies from frame
to frame and every master is built from a single image, which defeats the
point of stacking.

These choices, together, are what a project's ``master_config.json``
records. The browser interface can export it from a project you have set
up and load it into the next one, which saves repeating the whole
exercise for every new project on the same instrument.

Registering the images
======================

With the four decisions made, point the project at your images in the
browser interface and import them: each file is read, classified, matched
to its equipment, and recorded. From there processing proceeds exactly as
the tutorial shows for the test data.

Import a handful of frames first rather than the whole campaign. Nearly
every mistake in this page's subject matter surfaces on the first image
-- an expression naming a keyword that does not exist, a serial number
with no matching device, two type checks true at once -- and finding that
out on ten images is quicker than on ten thousand.
