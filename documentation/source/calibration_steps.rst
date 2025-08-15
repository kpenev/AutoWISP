1. Image Calibration
====================

**Commands:** ``wisp-calibrate``, ``wisp-stack-to-master``,
``wisp-stack-to-master-flat``

Important Parameters
--------------------

The most important parameters to set for the calibration are:

* :option:`exposure-start-utc` or :option:`exposure-start-jd`

* :option:`exposure-seconds`

* :option:`saturation-threshold`

* :option:`fnum`

* :option:`image-area`: If your camera generates images with overscan or
  other areas that are not part of the image.

* :option:`split-channels` or :option:`raw-hdu`: If using a color
  detector

* :option:`gain`: If known. Used for accurate error estimates.

Description
-----------

Before raw images are used, they need to be calibrated. The sequence of steps
is: 

    calibrate the raw bias images
        Accomplished using ``wisp-calibrate`` command with no master files
        specified with raw bias images as input.

    generate master bias image
        Use ``wisp-stack-to-master`` command with the calibrated bias images as
        input.

    calibrate raw dark images using the master bias
        ``wisp-calibrate`` with the :option:`master-bias` option set to the
        master bias file generated in the step above.

    generate master dark image
        Just like generating master bias but using the calibrated dark images.

    calibrate the raw flat images
        ``wisp-calibrate`` with the :option:`master-bias` option set to the
        master bias file and the :option:`master-dark` option set to the master
        dark.

    generate master flat image(s)
        ``wisp-stack-to-master-flat`` command. This is different from how master
        bias and dark are created, because AutoWISP is designed to allow using
        sky flats, which may be affected by clouds, or the sky is not perfectly
        uniformly bright. 

    calibrate raw object images
        ``wisp-calibrate`` with :option:`master-bias`,
        :option:`master-dark`, and :option:`master-flat` all specified.

No Calibration Data?
--------------------

In case calibration data is not available, only the last of these steps needs to
be performed, with no masters specified (see below). Even though in this case
the pixels values will not be corrected for any of the effects described above,
this step is still needed. It will add change the data format (float point
instead of integer), add required information in the headher, split the
different colors if using a color camera etc.

Overscan Corrections
--------------------

In many instances, the imaging device provides extra areas that attempt to
measure bias level and dark current, e.g. by continuing to read pixels past the
physical number of pixels in the device, thus measuring the bias or by having an
area of pixels which are somehow shielded from light, thus measuring the dark
level in real time. Such corrections can be supierior to the master frames in
that they measure the instantaneous bias and dark level, which may vary over
time due to for example the temperature of the detector varying. However, bias
level and dark current in particular can vary from pixel to pixel, which is not
captured by these real-time areas. Hence, the best strategy is a combination of
both, and is different for different detectors.

AutoWISP allows (but does not require) such areas to be used to estimate
some smooth function of image position to subtract from each raw image, and then
the masters are applied to the result. This works mathematically, because the
masters will also have their values corrected for the bias and dark measured by
these areas from the individual frames that were used to construct them. In this
scheme, the master frames are used only to capture the pixel to pixel
differences in bias and dark current. We refer to these areas as "overscan",
although that term really means only one type of such area.

Overscan area(s) can be specified using the :option:`overscans` option of the
``wisp-calibrate`` command.
