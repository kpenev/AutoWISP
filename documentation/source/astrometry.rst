3. Astrometry
=============

**Commnd:** ``wisp-solve-astrometry``

Important Parameters
--------------------

* :option:`astrometry-catalog-max-magnitude`

* :option:`max-srcmatch-distance`

* :option:`astrometry-order`

* :option:`min-match-fraction`

* :option:`max-rms-distance`

* :option:`frame-center-estimate`

* :option:`frame-fov-estimate`

* :option:`anet-api-key` if using the online astrometry.net service.

Description
-----------

The ``wisp-solve-astrometry`` command finds the transformation that converts sky
coordinates (right ascention and declination) into image coorditanes (x, y). This
allows the use of external catalogue data for more precise positions of the
sources than can be extracted from survey images and also the use of auxiliary
data provided in the catalogue about each source, in the subsequent processing
steps of the pipeline.

Astrometry is accomplished in 2 steps. First, AutoWISP uses `astrometry.net
<https://nova.astrometry.net/>`_ to find an initial match between the few tens
(to few hundred) brightest extracted source to the Gaia catalog, and then
iteratively refines this match and the transformation parameters to match almost
every single extracted source to its catalog counterpart. For wide-field images,
this means thousands of matches that are used to find the transfomation
parameters, allowing AutoWISP to model transformation involving signifiacant
image distortion, which often occurs in wide-field images.
