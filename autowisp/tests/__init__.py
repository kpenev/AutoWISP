"""Autowisp unit-test init."""

from os import path, makedirs
from subprocess import run, PIPE, STDOUT
from shutil import copytree, copy, rmtree
from glob import glob

from astrowisp.tests.utilities import FloatTestCase


class AutoWISPTestCase(FloatTestCase):
    """Base class for AutoWISP tests."""

    successful_test = False

    def get_inputs(self, inputs):
        """Get the input files for the test step and return what to clean up."""

        for product in inputs:
            for source in glob(path.join(self.test_directory, product)):
                destination = source.replace(
                    self.test_directory,
                    self.processing_directory,
                )
                assert path.exists(source)
                if path.isdir(source):
                    copytree(source, destination)
                else:
                    assert path.isfile(source)
                    destination = path.dirname(destination)
                    makedirs(destination, exist_ok=True)
                    copy(source, destination)

    @classmethod
    def set_test_directory(
        cls, test_dirname, processing_dirname, failed_test_dirname
    ):
        """Set the directory where data to test against is located."""

        cls.test_directory = test_dirname
        cls.processing_directory = processing_dirname
        cls.failed_test_directory = failed_test_dirname

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
        makedirs(self.processing_directory, exist_ok=True)
        copy(
            path.join(self.test_directory, "test.cfg"),
            path.join(self.processing_directory, "test.cfg"),
        )
        copy(
            path.join(self.test_directory, "autowisp.db"),
            path.join(self.processing_directory, "autowisp.db"),
        )

        self.successful_test = False

    def tearDown(self):
        """Remove the processing directory."""

        if self.successful_test:
            rmtree(self.processing_directory)
        else:
            rmtree(self.failed_test_directory, ignore_errors=True)
            copytree(self.processing_directory, self.failed_test_directory)

    def run_step(self, command):
        """Run a calibration step and check the return code."""

        calib_process = run(
            command,
            cwd=self.processing_directory,
            check=False,
            stdout=PIPE,
            stderr=STDOUT,
            timeout=600,
        )
        self.assertTrue(
            calib_process.returncode == 0,
            f"AutoWISP step command:\n{command!r}\n"
            f"Started from {self.processing_directory!r} "
            f"failed:\n{calib_process.stdout.decode('utf-8')}",
        )
