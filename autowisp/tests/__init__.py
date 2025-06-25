"""Autowisp unit-test init."""

from os import path
from unittest import TestCase

autowisp_dir = path.dirname(path.dirname(path.abspath(__file__)))

class AutoWISPTestCase(TestCase):
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
