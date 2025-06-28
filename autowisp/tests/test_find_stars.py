"""Unit tests for the find_stars step."""

from os import path

from autowisp.tests.h5_test_case import H5TestCase


class TestFindStars(H5TestCase):
    """Tests of the find_stars step."""

    def test_find_stars(self):
        """Run the find_stars step and check the outputs against expected."""

        self.run_step_test(
            "find_stars",
            path.join("CAL", "object"),
            ["SourceExtraction/Version000/Sources"],
        )
