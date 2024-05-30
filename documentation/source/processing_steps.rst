**************
Pipeline Steps
**************

The pipeline operations can be broken down into 5 big steps each of which is
further broken down into multiple smaller steps:

1. Image calibration: 
=====================

Take the raw data and calibrate it for various instrumental effects.

See :doc:`PythonModules/image_calibration` for implementation
documentation.

1.1 Split raw frames by type:
-----------------------------

    * Calibration frames:

        * bias: Frames with zero exposure intended to measure the behavior of
          the A-to-D converter.

        * dark: Frames with no light falling on the detector intended to measure
          the rate of accumulation of charge in the detector pixels in the
          absence of light.

        * flat: Frames with uniform illumination falling on the detector
          intended to measure the sensitivity to light of the system coming from
          different directions. 

    * Object frames: Images of the night sky from which photometry is to be
      extracted. Those can further be split into sub-groups from which
      independent lightcurves need to be generated. For example if several
      different exposure times were used, or there could be a number of filters
      or other chages in the optical system between frames which may produce
      better results if processed independently.

1.2 Image Calibration
---------------------

Before raw images are used, they need to be calibrated. The sequence of steps
is: 

    * calibrate raw bias frames

    * generate master bias frames

    * calibrate raw dark frames using the master biases

    * generate master dark frames

    * calibrate raw flat frames using the master biases and master darks

    * generate master flat frames

    * calibrate raw object frames

1.2.1 Create mask
^^^^^^^^^^^^^^^^^

Create a mask image noting pixels which are saturated (i.e. near the full-well
capacity). Also marks pixels neighboring saturatied pixels in the leak direction
as recipients of leaked charge. Also transfers any masks in the masters used,
including taking separate mask-only files.

1.2.1 Overscan corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^

In many instances, the imaging device provides extra areas that attempt to
measure bias level and dark current, e.g. by continuing to read pixels past the
physical number of pixels in the device, thus measuring the bias or by having an
area of pixels which are somehow shielded from light, thus measuring the dark
level in real time. Such corrections are supierior to the master frames in that
they measure the instantaneous bias and dark level, which may vary over time due
to for example the temperature of the detector varying. However, bias level and
dark current in particular can vary from pixel to pixel, which is not captured
by these real-time areas. Hence, the best strategy is a combination of both, and
is different for different detectors.

The pipeline allows (but does not require) such areas to be used to estimate
some smooth function of image position to subtract from each raw image, and then
the masters are applied to the result. This works mathematically, because the
masters will also have their values corrected for the bias and dark measured by
these areas from the individual frames that were used to construct them. In this
scheme, the master frames are used only to capture the pixel to pixel
differences in bias and dark current. We refer to these areas as "overscan",
although that term really means only one type of such area.

While overscan corrections are applied to all raw frames,

1.2.2. Bias level and dark current is subtracted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step simply subtracts the master bias and the master dark from the target
image.

The master bias is not subtracted from raw bias frames (since the reason for
calibrating those is to generate the master bias), and for raw dark frames,
master bias corrections are applied, but master dark are not. All other image
types get the full set of corrections.

1.2.2. Flat field corrections are applied
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a very simple step which simply takes the ratio of the bias and dark
corrected frame and the master flat, pixel by pixel.

This step is skipped for raw bias, dark and flat frames, and applied to all
object frames.

1.2.3. The image is trimmed to only the image area
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step removes overscan, dark and other areas that are used during the
calibration process, but lose their meaning afterwards.

This step is apllied to all raw frames.

1.2.4. Individual pixel errors are calculated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the original image, the error in the value of each pixel is simply given by
(pixel value / gain)\ :sup:`0.5`. However, once the above corrections are
applied this is no longer true. The de-biasing and de-darking adds the (small)
noise in the master frames (and the overscan corrections), the scaling by the
flat introduces the error in the master flat, but also changes the "gain"
differently for ecah pixel. In order to properly handle all those, calibrating
raw frames in the pipeline produces two images: the calibrated image and an
error estimate image giving the so called 1-sigma error estimate (in 68% of the
cases the true amount of light that fell on the detector deviates no more than
the given amount from the reported value).

This is done for all raw frames.

1.3 Generate master frames:
---------------------------

Master frames are stacks of individual calibrated calibration frames. As a
result their signal to noise ratio is greatly increased, compared to individual
un-stacked frames, allowing for much better calibration. In each case, the
frames are split into groups in which the effect being measured is not expected
to var and the individual frames are stacked, with suspicious (outlier in some
way) frames are discarded.

2. Astrometry:
==============

Find a transformation that allows you to map sky coordinates (RA, Dec) into
image coorditanes. This allows the use of external catalogue data for more
precise positions of the sources than can be extracted from survey images and
also the use of auxiliary data provided in the catalogue about each source, in
the subsequent processing steps of the pipeline.

Astrometry is accomplished in 3 steps:

2.1 Extract sources:
--------------------

Find sources (stars) in the individual calibrated object frames.

2.2 Match to external catalogue.
--------------------------------

Match the extracted sources to the sources listed in an external
catalogue.

2.3 Solve for the transformation
--------------------------------

Find a smooth transformation that maps the catalogue (RA, Dec) coordinates to
the positions of the extracted sources as close as possible. The key word here
is smooth. That is the transformation should only have a few free parameters to
be tuned on thousands of sources. As a result the transformation parameters are
determined to very high accuracy and precision, thus providing more precise
image positions than source extraction by transforming high precision catalogue
positions through this high S/N transformation.

3. Photometry:
==============

For each calibrated object frames, extract flux measuruments for catalogue
sources which map to some position within the frame using the astrometric
transformation derived in the previous step. There are many flavors of
photomety. This pipeline supports three: PRF fitting, PSF fitting and aperture
photometry, with aperture photometry requiring PSF fitting.

3.1 PRF/PSF fitting:
--------------------

Each point source once it is imaged by our observing system produces a
particular distribution of light on the detector. The idea of PRF and PSF
fitting is to model that distribution as some smooth parametric function
centered on the projected source position that has an integral of 1 times an
amplitude. The amplitude of course is then a measure of the flux of the source,
while the parameters of the function specify its shape in some way.

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
perform PSF and PRF fitting. For the gory details of how this is done, see the
`AstroWISP documentation <https://kpenev.github.io/AstroWISP/>`_. Briefly, the
PSF and PRF are modeled as piecewise bi-cubic functions with a number of free
parameters.  These parameters are in turn forced to vary smoothly as a function
of source and image properties across sources and across images.

3.2 Aperture photometry:
------------------------

For each source, sum-up the flux in the image within a series of concentric
circles centered on the projected source position. In order to properly handle
the inevitable pixels that are partiallly within an aperture, knowledge of the
distribution of light accross these pixels as well as the sub-pixel sensitivy
map is required.

This taks is again carried out by `AstroWISP
<https://github.com/kpenev/AstroWISP>`_. See the `documentation
<https://kpenev.github.io/AstroWISP/>`_ for further details.

4. Magnitude fitting:
=====================

In ground based applications, the night sky is imaged through variable amount of
atmosphere, which itself is subject to changes (i.e. clouds, humidity, etc.). In
addition various instrumental effects are generally present. The purpose of the
magnitude fitting step is to eliminate as much as possible effects that modify
the measured source brightness within an image in a manner that depends
smoothly on the properties of the source.

In short, a reference frame is selected (and later generated). Then for each
individual frame (target frame from now on) a smooth multiplicative correction
is derived that when applied to the brightness measurements in the target frame
matches the brightness measurements in the reference frame as closely as
possible.

In the pipeline this is actually done twice. The first time, a single frame
which appears to be of very high quality (sharp PSF, high atmospheric
transparency, dark sky etc.) is used as the reference frame. The corrected
brightness measurements of the individua frames are then stacked to produce a
much highe signal to noise "master reference frame", which is then used in a
second iteration of the magnitude fitting process to generate the final fitted
magnitudes.

5. Dumping lightcurves:
=======================

This is a simple transpose operation. In all previous steps, the photometry is
extracted simultaneously for all sources in a given image or in a short series
of images. In order to study each source's individual variability, the
measurements from all frames for that source must be collected together. This
step simply performs that reorganization. For each catalogue source, all
available measurements from the individual frames are collected in a file,
possibly combined with earlier measurements from say a different but overlapping
pointing of the telescope or with another instrumental set-up.

6. Lightcurve post-processing:
==============================

Even though we have tried hard to eliminate as many "instrumental" effects as
possible from teh lightcurves generated above, there will still be some present.
Namely those that violate the assumptions behind magnitude fitting. Further, for
many applications, e.g. planet hunting, the goal is to identify a signal with a
very specific shape. In this case, it is desirable to filter out even real
astrophysical signals in order to boost the sensitivity to lower amplitude
effects. In order to achieve this, several post-processing steps are carried out
by the pipeline.

6.1 External Parameter Decorrelation (EPD):
-------------------------------------------

This simply removes from each individual lightcurve the linear combintion of
user specified instrumental and other time variable parameters that explain the
most variance. Clearly care must be taken when selecting the parameters to
decorrelate against, lest they vary on similar timescales as the target signal.
If this happens, this step will highly distort if not eliminate the target
signal.


6.2 Trend filtering algorithm (TFA):
------------------------------------

In this step signals which are shared by mulitple sources are removed from each
source's lightcurve. The idea is that most instrumental effects will affect
multiple sources in a similar way, and thus signals common to several sources
are suspected of being instrumental, rather than real astrophysical variability.
Again this steps has the potential to distort or eliminate target signals, so it
should be used with care. If the shape of the target signal is known, there are
versions of this procedure which tend to preserve it.
