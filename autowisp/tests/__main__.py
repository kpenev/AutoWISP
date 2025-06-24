#!/usr/bin/env python3

"""Test the calibration of bias images."""

from tempfile import TemporaryDirectory
from os import path, makedirs

import unittest

from autowisp.tests.get_test_data import get_test_data
from autowisp.tests.test_calibration import TestCalibration

if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        processing_dir = path.join(temp_dir, "processing")
        makedirs(processing_dir)
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
        get_test_data(temp_dir)
        TestCalibration.set_test_directory(temp_dir, processing_dir)
        unittest.main()
