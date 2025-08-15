6. Fit PSF Map to Extracted Stars
=================================

**Command:** ``wisp-fit-source-extracted-psf-map``

Important Parameters
--------------------

* :option:`srcextract-psfmap-terms`


The various systematics removal steps in the pipeline can make use of
information about the shape of stars in the images (a.k.a. PRF). That can come
from two sources: :ref:`the find stars step <srcfind-section>` as part of the
process fits for a PRF model, and obviously the :ref:`PSF/PRF fitting step
<psffit-section>` does as well. At the moment, the latter does not produce
summary information (e.g.  full-width at half maximum, ellpticity, etc.).
Furthermore, in many applications, the :ref:`PSF/PRF fitting step
<psffit-section>` is too computationally expensive, and so it is replaced with a
dummy fit.  The result is that presently we rely on information from :ref:`the
find stars step <srcfind-section>`.

Fitting the PRF of each individual star (as :ref:`the find stars step
<srcfind-section>` does) results in quite noisy estimated parameters. In order
to reduce that noise, this step of the pipeline fits a smooth function of image
position and possibly stellar properties (brightness, color, etc. from the
catalog) to each of the relevant parameters. Because the fit is based on a large
number of stars, the noise gets averaged out to a large extent. The map is then
evaluated for each star to produce higher-quality estimates of the PRF
parameters.

The step is accomplished by the ``wisp-fit-source-extracted-psf-map`` command.

