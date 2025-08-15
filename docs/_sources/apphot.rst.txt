5. Aperture Photometry
======================

**Command**: ``wisp-measure-aperture-photometry``

Important Parameters
--------------------

* :option:`apertures`

* :option:`subpixmap`: if processing images collected with a color camera

Description
-----------

For each source, ``wisp-measure-aperture-photometry`` sums-up the flux in the
image within a series of concentric circles centered on the projected source
position. In order to properly handle the inevitable pixels that are partiallly
within an aperture, knowledge of the distribution of light accross these pixels
as well as the sub-pixel sensitivy map is required.
