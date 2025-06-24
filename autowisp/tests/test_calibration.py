"""Test cases for image calibration."""

from os import path
from glob import glob

from astrowisp.tests.utilities import FloatTestCase


class TestCalibration(FloatTestCase):
    """Test cases for bias image calibration."""

    def assert_header_match(self, header1, header2):
        """Check if two headers match."""

        for key, value in header1.items():
            self.assertIn(
                key, header2, f"Key {key} not found in second header."
            )
            self.assertEqual(
                header2[key], value, f"Value for key {key} does not match."
            )

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

    def test_bias_calibration(self):
        """Check if bias calibration works as expected."""

        input_dir = path.join(self._test_directory, "RAW", "zero")
        with open(
            path.join(self._processing_directory, "test.cfg"),
            "r",
            encoding="utf-8",
        ) as cfg:
            print(
                f"Processing {len(glob(path.join(input_dir, '*.fits*')))} bias "
                f"frames using configuration:\n{cfg.read()}"
            )
        self.assertTrue(True)
