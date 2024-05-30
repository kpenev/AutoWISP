*************************************
Low Level Image Calibration Interface
*************************************

In the first release of the pipeline all images are assumed to be in FITS format
and only FITS images are produced. As described in the image calibration step of
the :doc:`../processing_steps` image calibration produces two images for each
input image. Those are stored as a two-extension FITS files. The first extension
is the calibrated image and the second extension are the error estimates for
each pixel. While the input images can be either integer or floating point, the
calibrated images are always floating point.

The :mod:`image_calibration<autowisp.image_calibration>` module
defines a class
(:class:`Calibrator<autowisp.image_calibration.Calibrator>`) that
provides the lowest level interface for performing calibrations. The calibration
requires specifying overscan area(s) to use, overscan correction method(s),
master bias/dark/flat, and gain to assume for the input raw frames (single
floating point value), the area in the image which actually contains image
pixels (must match the dimensions of the masters). We will refer to these as
calibration parameters from now on. 

The typical work flow is as follows:

    1. Create a Calibrator instance, optionally specifying calibration
       parameters.

    2. Optionally, specify further or overwrite previously specified calibration
       parameters as attributes to the object.

    3. Call the object with the  filename of the image to calibrate and the
       output filename for the calibrated image. All calibration parameters
       can be replaced, for this image only, through additional keyword
       arguments. Any masters not specified or set to None are not applied.
       Hence to calibrate a raw flat frame, set master_flat = None.

    4. Repeat steps 2 and 3 for all images which need calibrating. 

For example, in order to calibrate flat frames called ``raw1.fits`` and
``raw2.fits`` (with a resolution of 4096x4116) with overscan region consisting
of the first 20 rows applied by subtracting a simple median::

    from super_phot_pipeline.image_calibration import\
        Calibrator,\
        overscan_methods

    calibrate = Calibrator(
        saturation_threshold=64000.0,
        overscans=[dict(xmin = 0, xmax = 4096, ymin = 0, ymax = 20)],
        overscan_method=overscan_methos.median,
        master_bias='masters/master_bias1.fits',
        gain=16.0,
        image_area=dict(xmin=0, xmax=4096, ymin=20, ymax=4116
    )
    calibrate.set_masters(dark='masters/master_dark3.fits')
    calibrate(raw='raw1.fits', calibrated='calib1.fits')
    calibrate(raw='raw2.fits', calibrated='calib2.fits', gain=8.0)

Users can define their own overscan methods. All that is required is a function
that takes the input image (numpy array-like object), an overscans dictionary
defining the overscan area and returns the overscan corrected image as a numpy
floating point array-like object.
