"""Unit tests for the solve_astrometry step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestSolveAstrometry(H5TestCase):
    """Tests of the find_stars step."""

    def test_solve_astrometry(self):
        """Run the solve_astrometry step and check the outputs."""

        self.run_step_test(
            "solve_astrometry",
            "DR",
            ["CatalogueSources", "SkyToFrameTransformation"],
        )
