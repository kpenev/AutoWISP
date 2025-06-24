"""Test cases for image calibration."""

from os import path
from glob import glob
from subprocess import run

from autowisp.tests import autowisp_dir, FITSTestCase


class TestCalibration(FITSTestCase):
    """Test cases for bias image calibration."""

    @classmethod
    def set_test_directory(cls, test_dirname, processing_dirname):
        """Set the directory where data to test against is located."""

        cls._test_directory = test_dirname
        cls._processing_directory = processing_dirname

    def setUp(self):
        """Make sure the data to compare against is defined."""

        self.assertTrue(
            hasattr(self, "_test_directory"), "No test data directory defined!"
        )
        self.assertTrue(
            hasattr(self, "_processing_directory"),
            "No processing directory defined!",
        )
        self.assertTrue(
            path.exists(self._test_directory),
            f"Test directory {self._test_directory} does not exist!",
        )

    def test_bias_calibration(self):
        """Check if bias calibration works as expected."""

        input_dir = path.join(self._test_directory, "RAW", "zero")
        calib_process = run(
            [
                "python3",
                path.join(autowisp_dir, "processing_steps", "calibrate.py"),
                "-c",
                path.join(self._processing_directory, "test.cfg"),
                path.join(input_dir, "*.fits.fz"),
                "RAW/zero/*.fits.fz",
            ],
            cwd=self._processing_directory,
            check=False,
        )
        self.assertTrue(
            calib_process.returncode == 0,
            f"Calibration processed failed in {self._processing_directory}!",
        )

        generated = sorted(
            glob(
                path.join(self._processing_directory, "CAL", "zero", "*.fits*")
            )
        )
        expected = sorted(
            glob(path.join(self._test_directory, "CAL", "zero", "*.fits.fz"))
        )
        self.assertTrue(
            [path.basename(fname) for fname in generated]
            == [path.basename(fname) for fname in expected],
            "Generated files do not match expected files!",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_fits_match(exp_fname, gen_fname)
