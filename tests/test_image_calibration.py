"""Unit tests for the image_calibration module."""

import os.path
import sys
import unittest

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from superphot_pipeline.image_calibration import Calibrator, overscan_methods


git_id = '$Id$'

if __name__ == '__main__':
    calibrate = Calibrator(
        #The first 20 lines of the image are overscan area.
        overscans=[dict(xmin=0, xmax=4096, ymin=0, ymax=20)],

        #Overscan corrections should subtract the median of the values.
        overscan_method=overscan_methods.Median(),

        #The master bias frame to use.
        master_bias='masters/master_bias1.fits',

        #The gain (electrons/ADU) to assume for the input images.
        gain=16.0,

        #The area within the raw frame containing the image:
        image_area=dict(xmin=0, xmax=4096, ymin=20, ymax=4116)
    )

    #Specify a master dark after construction.
    calibrate.set_masters(dark='masters/master_dark3.fits')

    #Calibrate an image called 'raw1.fits', producing (or overwriting a
    #calibrated file called 'calib1.fits' using the previously specified
    #calibration parameters. Note that no flat correction is going to be
    #applied, since a master flat was never specified.
    calibrate(raw='raw1.fits', calibrated='calib1.fits')

    #Calibrate an image, changing the gain assumed for this image only and
    #disabling overscan correction for this image only.
    calibrate(raw='raw2.fits',
              calibrated='calib2.fits',
              gain=8.0,
              overscans=None)
