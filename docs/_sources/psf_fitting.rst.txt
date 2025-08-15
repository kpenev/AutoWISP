4. PRF/PSF Fitting
==================

**Command:** ``wisp-fit-star-shape``

Important Parameters
--------------------

* :option:`photometry-catalog-max-magnitude`

* :option:`subpixmap`: if processing images collected with a color camera

* :option:`background-annulus`

* :option:`shapefit-smoothing`

* :option:`shape-grid`

* :option:`shape-mode`

* :option:`shape-terms-expression`

* :option:`map-variables`

Description
-----------

The first flux measurements AutoWISP generates for each image is one based on
fitting for the shape of stars in the image and the flux (as the proportionality
constant before the shape and the actual pixels belonging to that star on the
image), called PSF or PRF fitting. This is accomplished by the
``wisp-fit-star-shape`` command.

Each point source once it is imaged by our observing system produces a
particular distribution of light on the detector.  The idea of PRF and PSF
fitting is to model that distribution as some smooth parametric function
centered on the projected source position that has an integral of one. For each
star AutoWISP also fits for the amplitude that best scales this shape to the
observed pixel values. The amplitude of course is then a measure of the flux of
the source, while the parameters of the function specify its shape.

To review the terms:

    * Point Spread Function or PSF: PSF(dx, dy) is the amount of light that hits
      the surface of the detector offset by (dx, dy) from the projected position
      of the source. In order to actually predict what a particular detector
      pixel will measure, one computes the integral of the PSF times a sub-pixel
      sensitivity map over the area of the pixel.

    * Pixel Response Function or PRF: PRF(dx, dy) is the value that a pixel with
      a center offset by (dx, dy) from the projected source position will
      register.  Note that dx and dy can be arbitrary real values and not just
      integers. The PRF already folds in its definition the sub-pixel
      sensitivity map, and other detector characteristics. Further, since the
      PRF is the PSF convolved with the sub-pixel sensitiity map it is generally
      smoother than the PSF and thus easier to model.

In this pipeline we use `AstroWISP <https://github.com/kpenev/AstroWISP>`_ to
perform PSF and PRF fitting. For the details of how this is done, see the
`AstroWISP documentation <https://kpenev.github.io/AstroWISP/>`_. Briefly, the
PSF and PRF are modeled as piecewise bi-cubic functions with a number of free
parameters.  These parameters are in turn forced to vary smoothly as a function
of source and image properties across sources and across images.

The information from PSF fitting is then used in the next step
(:ref:`Aperture Photometry <apphot-section>`)
