#!/usr/bin/env python3

"""Test the calibration of bias images."""

from tempfile import TemporaryDirectory
from os import path, makedirs
from subprocess import run
import contextlib

import unittest

from autowisp.tests import autowisp_dir
from autowisp.tests.get_test_data import get_test_data
from autowisp.tests.test_calibrate import TestCalibrate
from autowisp.tests.test_stack_to_master import TestStackToMaster
from autowisp.tests.test_find_stars import TestFindStars

if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
    #with contextlib.suppress() as _:
        #temp_dir = path.join(autowisp_dir, "tests", "test_data")
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
            path.join(path.dirname(__file__), "test_data", "test.cfg"),
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
        unittest.main()
