#!/usr/bin/env python3

"""Test the calibration of bias images."""

from tempfile import TemporaryDirectory
from os import path, makedirs
from shutil import copy
from sys import argv

import unittest

from autowisp.tests import AutoWISPTestCase
from autowisp.tests.get_test_data import get_test_data

# Automatically used by pytest
# pylint: disable=unused-import
from autowisp.tests.test_calibrate import TestCalibrate
from autowisp.tests.test_stack_to_master import TestStackToMaster
from autowisp.tests.test_find_stars import TestFindStars
from autowisp.tests.test_solve_astrometry import TestSolveAstrometry
from autowisp.tests.test_fit_star_shape import TestFitStarShape
from autowisp.tests.test_measure_aperture_photometry import (
    TestMeasureAperturePhotometry,
)
from autowisp.tests.test_fit_source_extracted_psf_map import (
    TestFitSourceExtractedPSFMap,
)
from autowisp.tests.test_fit_magnitudes import TestFitMagnitudes
from autowisp.tests.test_create_lightcurves import TestCreateLightcurves
from autowisp.tests.test_epd import TestEPD
from autowisp.tests.test_tfa import TestTFA
from autowisp.tests.test_detrending_stat import TestDetrendingStat

# pylint: enable=unused-import

if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        get_test_data(temp_dir)
        # temp_dir = (
        # "/Users/kpenev/projects/git/AutoWISP/autowisp/tests/test_data"
        # )
        processing_dir = path.join(temp_dir, "processing")
        makedirs(processing_dir, exist_ok=False)
        copy(
            path.join(temp_dir, "test.cfg"),
            path.join(processing_dir, "test.cfg"),
        )
        AutoWISPTestCase.set_test_directory(
            temp_dir, processing_dir, argv.pop(1)
        )
        unittest.main(failfast=True)
