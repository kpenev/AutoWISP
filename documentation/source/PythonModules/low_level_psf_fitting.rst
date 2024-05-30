***********************************
Low Level PSF/PRF Fitting Interface
***********************************

First to define our terms. The PSF is the distribution of light produced by the
optical system in the plane of the detector for a point source at infinity,
while the PRF gives the response of a detector pixel at some offset from the
"center" of the source. That is the PRF is the PSF convolved with the
sensitivity map of detector pixels.

PSF/PRF fitting is done using AstroWISP, which models both the PSF and the PRF
as follows: The area around the source location is split into a rectangular grid
of cells (unrelated to pixels) with, in general, non-uniform sizes, organized in
M rows and N columns. In each cell, the amount of light falling on the detector
is modeled as a bi--cubic function:

.. math::

    f(x,y)=A\sum_{m=0}^3\sum_{n=0}^3 C_{i,j,m,n} x^m y^n \label{eq: piecewise
    bicubic PSF}

where :math:`C_{i,j,m,n}` is the coefficient in front of :math:`x^my^n` for the
cell in the i-th column and j-th row.  The only restrictions imposed during PSF
fitting (but may be violated if manual PSF models are supplied) are common
sense: first: :math:`f(x,y)` should be continuously differentiable across cell
boundaries; and second: the PSF and all its derivatives should be zero at the
grid boundaries. Under these assumptions, the PB model is fully specified by the
values of :math:`f`, :math:`\frac{\partial f}{\partial x}`,
:math:`\frac{\partial f}{\partial y}` and :math:`\frac{\partial^2 f}{\partial x
\partial y}` on the :math:`(N-1)\times(M-1)` cell corners.

The PSF model is described by a set of shape parameters (the :math:`C_{i,j,m,n}`
coefficients) and an overall amplitude. The amplitudes are independent between
sources, while the shape parameters are assumed to vary smoothly as a function
of image position, outside temperature, color or brightness of the source, etc.
This smooth dependence is specified as a polynomial of a user specified order,
and the fit is performed for the coefficients of this polynomial. The fitting
compares measured pixel values with the value predicted from either the integral
of the product of the PSF and the sub--pixel sensitivity map or the value of the
PRF at the center of the pixel, weighing each pixel by its signal to noise. All
source pixels in all input images will be fit simultaneously.

If the amplitudes are fixed, the predicted pixel values are linear functions of
the polynomial expansion coefficients. As a result, PSF/PRF fitting splits into
two linear fits: one for the polynomial expansion coefficients of the shape
parameters and one for the amplitudes. The fit is performed by iterating between
the two, until the fractional change in the vector of best fit fluxes falls
below some threshold. It is highly advantageous to redefine the fitting
parameters such that the full integral of the PSF/PRF is restricted to always be
unity. This is straightforward to achieve, and does not break the linear
dependence of the predicted pixel values on the fitted parameters. The only
problem that remains is to derive a good initial guess for either the fluxes or
the shape parameters. The former can be achieved by performing aperture
photometry on the input sources, assuming a flat PSF model.

Since modelling every single image pixel and fitting for the fluxes of all
sources simultaneously is an overwhelming computational task, AstroWISP
processes only pixels near known sources (i.e. ones that at least partially
overlap with the PSF grid), discarding pixels to which multiple sources
contribute flux from the shape fit but fitting for the fluxes of such sources
simultaneously during the amplitude fitting step.

Given calibrated FITS frames of the night sky produced as described in
:doc:`low_level_image_calibration` and an astrometric solution produced as
described in :doc:`low_level_astrometry`, this module uses AstroWISP to carry
out the PSF/PRF fitting decsribed above in four steps:

  1. Querying the catalogue for all sources in an area fully covering the image,
     filtering those sources by specified criteria and using the astrometric
     solution to project those sources on the frame, discarding sources which
     project to locations outside the frame. This is the list of sources passed
     to AstroWISP for which PSF/PRF fitting is performed.

  2. Invoke AstroWISP to measure the background for each source in the above
     list.

  3. Invoke AstroWISP with the list of sources now each augmented by its
     background measurement to fir for the PSF or PRF.

  4. Add the results of the above 3 steps to the data reduction files for the
     frames processed, fully documenting the versions of each tool and module
     used as well as any configuration parameters.
