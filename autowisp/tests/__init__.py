"""Autowisp unit-test init."""

from os import path
from subprocess import run, PIPE, STDOUT

from astrowisp.tests.utilities import FloatTestCase

autowisp_dir = path.dirname(path.dirname(path.abspath(__file__)))
steps_dir = path.join(autowisp_dir, "processing_steps")

class AutoWISPTestCase(FloatTestCase):
    """Base class for AutoWISP tests."""

    @classmethod
    def set_test_directory(cls, test_dirname, processing_dirname):
        """Set the directory where data to test against is located."""

        cls.test_directory = test_dirname
        cls.processing_directory = processing_dirname

    def setUp(self):
        """Make sure the data to compare against is defined."""

        self.assertTrue(
            hasattr(self, "test_directory"), "No test data directory defined!"
        )
        self.assertTrue(
            hasattr(self, "processing_directory"),
            "No processing directory defined!",
        )
        self.assertTrue(
            path.exists(self.test_directory),
            f"Test directory {self.test_directory} does not exist!",
        )

    def run_calib_step(self, command):
        """Run a calibration step and check the return code."""

        calib_process = run(
            command,
            cwd=self.processing_directory,
            check=False,
            stdout=PIPE,
            stderr=STDOUT,
        )
        self.assertTrue(
            calib_process.returncode == 0,
            f"AutoWISP step command:\n{command!r}\n"
            f"Started from {self.processing_directory!r} "
            f"failed:\n{calib_process.stdout.decode('utf-8')}",
        )
