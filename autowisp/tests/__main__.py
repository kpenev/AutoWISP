#!/usr/bin/env python3

"""Test the calibration of bias images."""

from tempfile import TemporaryDirectory
from os import path, makedirs
from subprocess import run

import unittest

from autowisp.tests import autowisp_dir
from autowisp.tests.get_test_data import get_test_data
from autowisp.tests.test_calibrate import TestCalibrate
from autowisp.tests.test_stack_to_master import TestStackToMaster
from autowisp.tests.test_find_stars import TestFindStars
from autowisp.tests.test_solve_astrometry import TestSolveAstrometry
from autowisp.tests.test_fit_star_shape import TestFitStarShape
from autowisp.tests.test_measure_aperture_photometry import (
    TestMeasureAperturePhotometry,
)


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        # temp_dir = (
        #    "/Users/kpenev/projects/git/AutoWISP/autowisp/tests/test_data"
        # )
        get_test_data(temp_dir)
        processing_dir = path.join(temp_dir, "processing")
        makedirs(processing_dir)
        run(
            [
                "python3",
                path.join(
                    autowisp_dir,
                    "database",
                    "initialize_database.py",
                ),
                "--drop-hdf5-structure-tables",
            ],
            cwd=processing_dir,
            check=True,
        )
        with open(
            path.join(temp_dir, "test.cfg"),
            "r",
            encoding="utf-8",
        ) as cfg_template, open(
            path.join(processing_dir, "test.cfg"), "w", encoding="utf-8"
        ) as cfg_file:
            cfg_file.write(
                cfg_template.read().replace("@@OUTDIR@@", processing_dir)
            )
        TestCalibrate.set_test_directory(temp_dir, processing_dir)
        TestStackToMaster.set_test_directory(temp_dir, processing_dir)
        TestFindStars.set_test_directory(temp_dir, processing_dir)
        TestSolveAstrometry.set_test_directory(temp_dir, processing_dir)
        TestFitStarShape.set_test_directory(temp_dir, processing_dir)
        TestMeasureAperturePhotometry.set_test_directory(
            temp_dir, processing_dir
        )
        unittest.main()
