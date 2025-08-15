8. Creating Lightcurves
=======================

**Command:** ``wisp-create-lightcurves``

Important Parameters
--------------------

* :option:`lc-catalog-max-magnitude`

* :option:`max-memory`
 
* :option:`latitude-deg`

* :option:`longitude-deg`

* :option:`altitude-meters`

Description
-----------

This is a simple transpose operation. In all previous steps, the photometry is
extracted simultaneously for all sources in a given image or in a short series
of images. In order to study each source's individual variability, the
measurements from all frames for that source must be collected together. This
step simply performs that reorganization. For each catalogue source, all
available measurements from the individual frames are collected in a file,
possibly combined with earlier measurements from say a different but overlapping
pointing of the telescope or with another instrumental set-up.
