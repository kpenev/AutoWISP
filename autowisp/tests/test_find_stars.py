"""Unit tests for the find_stars step."""

from os import path
from glob import glob

from autowisp.tests import steps_dir
from autowisp.tests.dr_test_case import DRTestCase


class TestFindStars(DRTestCase):
    """Tests of the find_stars step."""

    def test_find_stars(self):
        """Run the find_stars step and check the outputs against expected."""

        input_dir = path.join(self.test_directory, "CAL", "object")

        self.run_calib_step(
            [
                "python3",
                path.join(steps_dir, "find_stars.py"),
                "-c",
                "test.cfg",
                input_dir,
            ]
        )

        generated = sorted(
            glob(path.join(self.processing_directory, "DR", "*.h5*"))
        )
        expected = sorted(glob(path.join(self.test_directory, "DR", "*.h5")))
        self.assertTrue(
            [path.basename(fname) for fname in generated]
            == [path.basename(fname) for fname in expected],
            "Generated files do not match expected files!",
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_groups_match(
                gen_fname, exp_fname, "SourceExtraction/Version000/Sources"
            )
