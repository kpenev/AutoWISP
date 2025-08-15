2. Find Stars
=============

**Command:** ``wisp-find-stars``

Important parameters
--------------------

* :option:`brightness-threshold`. 
  
* :option:`filter-sources` (if needed)

* :option:`srcextract-max-sources` (if needed)

Description
-----------

Before AutoWISP can measure the brightnesses of the star in images, it needs to
know the coordinates of the stars in the image. Finding a reliable sets of
coordinates consists of two steps: running a source extraction algorithm to find
an initial (not extremely precise) list of stars and their coordinates, and
then matching these stars to an external high-precision and high-accuracy in
order to find a function that can convert the coordinates of the stars in the
catalogue to coordinates of the stars in the image.

The first of these steps is performed using the ``wisp-find-stars`` command.
Behind the scenes, this command runs the source extraction algorithm
from the `FITSH
<https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.1825P/abstract>`_ package. 

The browser interface provides a sandbox where source extraction can be tested
interactively on any calibrated image with different values of
:option:`brightness-threshold`, :option:`filter-sources`, and
:option:`srcextract-max-sources`.
