***************************************
Low Level Aperture Photometry Interface
***************************************

One of the commonly used methods to measure fluxes of sources from an image
is to sum--up the flux within some circular aperture centered on each source
position. The reason for choosing circular apertures is that in most
applications, point sources produce roughly circularly symmetric profiles. In
general there is a trade--off that has to be made when an aperture is chosen.
Large apertures result in a larger area of sky being included which
contributes noise to the measurement. This particularly affects faint stars.
On the other hand, smaller apertures include less flux, and hence are subject
to larger Poisson noise. This is particularly important for bright stars,
since their PSF dominates over the sky brightness over a much larger area of
the detector. Luckily, flux measurements using many multiple apertures can
be performed, resulting in optimal photometry regardless of the brightnesses
of the sources.

If one were to simply add up the fluxes for any pixels that fully or
partially overlap with the aperture, the area over which the flux is added
will vary as the source mover around the pixel grid. As a result, unless the
positions of sources are kept fixed to much better than the size of a pixel
between consecutive images of a survey, this introduces scatter in the
measured flux. The only way to avoid this software noise source is to always
add up the flux over the same area around the source for each image. This
automatically implies that photometry tools have to handle pixels which only
partially overlap with the aperture.

AstroWISP performs aperture photometry, properly accounting for the PSF and
non-uniform pixel sensitivity for both pixels fully within the aperture and
pixels straddling the aperture boundary. If no PSF information is available, the
flux over each pixel is assumed uniformly distributed. If not pixel sensitivity
information is available, pixels are assumed uniformly sensitive.

The flux (:math:`F`) and its Poisson error (:math:`\delta F`) are estimated as
:math:`F=\sum_p k_p m_p - \pi r^2B` and :math:`\delta F=\sqrt{\sum_p k_p^2 m_p
g_p^{-1}} + \pi r^2\delta B^2`, where the :math:`p` index iterates over all
pixels which at least partially overlap with the aperture, :math:`m_p` are the
measured (after bias, dark and flat field corrections) values of the pixels in
ADU, :math:`g_p` are the gains of the pixels in electrons/ADU (including the
effects of the pre--applied flat field correction), :math:`k_p` are constants
which account for the sub--pixel sensitivity map and the partial overlaps
between the pixel and the aperture, :math:`r` is the size of the aperture, and
:math:`B` and :math:`\delta B` are the background estimate and its associated
error for the source.

If the sub--pixel sensitivity map is :math:`S(x,y)\ (0<x<1, 0<y<1)`, the PSF is
:math:`f(x,y)` (:math:`x` and :math:`y` are relative to the input position of
the source center), and :math:`l_p`/:math:`b_p` is the horizontal/vertical
distance between the left/bottom boundary of the :math:`p`-th pixel and the source
location, then the :math:`k_p` constants are given by:

.. math::

	k_p\equiv \frac{
        \int {\left[f(l_p+x, b_p+y) + B/A \right] dxdy}
	}{
        B/A + \int_0^1 dx \int_0^1 dy\,f(l_p+x, b_p+y)S(x,y)
	}

where the integral in the numerator is performed over the part of the pixel
that is inside the aperture, and :math:`A` is the overall scaling constant by which
the PSF must be multiplied in order to reproduce the pixel values on the
image (see :doc:`low_level_psf_fitting` for a description of
how :math:`A`, :math:`B` and the PSF parameters are estimated).
