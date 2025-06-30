"""Test cases for image calibration."""

from os import path
from glob import glob
from shutil import rmtree

from autowisp.tests.fits_test_case import FITSTestCase


class TestCalibrate(FITSTestCase):
    """Test cases for image calibration."""

    def _test_calibration(self, input_imtype, **masters):
        """Perform a calibration step and test outputs match expectations."""

        input_dir = path.join(self.test_directory, "RAW", input_imtype)
        command = [
            "wisp-calibrate",
            "-c",
            path.join(self.processing_directory, "test.cfg"),
            path.join(input_dir, "*.fits.fz"),
        ]
        for master_type, master_fname in masters.items():
            command.extend([f"--master-{master_type}", master_fname])
        self.run_step(command)

        generated = sorted(
            glob(
                path.join(
                    self.processing_directory, "CAL", input_imtype, "*.fits*"
                )
            )
        )
        expected = sorted(
            glob(
                path.join(self.test_directory, "CAL", input_imtype, "*.fits.fz")
            )
        )
        self.assertTrue(
            [path.basename(fname) for fname in generated]
            == [path.basename(fname) for fname in expected],
            "Generated files do not match expected files!",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_fits_match(exp_fname, gen_fname)
        rmtree(path.join(self.processing_directory, "CAL", input_imtype))
        self.successful_test = True

    def test_bias_calibration(self):
        """Check if bias calibration works as expected."""

        self._test_calibration("zero")

    def test_dark_calibration(self):
        """Check if dark calibration works as expected."""

        self._test_calibration(
            "dark",
            bias="R:"
            + path.join(self.test_directory, "MASTERS", "zero_R.fits.fz"),
        )

    def test_flat_calibration(self):
        """Check if flat calibration works as expected."""

        self._test_calibration(
            "flat",
            bias="R:"
            + path.join(self.test_directory, "MASTERS", "zero_R.fits.fz"),
            dark="R:"
            + path.join(self.test_directory, "MASTERS", "dark_R.fits.fz"),
        )

    def test_object_calibration(self):
        """Check if flat calibration works as expected."""

        self._test_calibration(
            "object",
            bias="R:"
            + path.join(self.test_directory, "MASTERS", "zero_R.fits.fz"),
            dark="R:"
            + path.join(self.test_directory, "MASTERS", "dark_R.fits.fz"),
            flat="R:"
            + path.join(self.test_directory, "MASTERS", "flat_R.fits.fz"),
        )
