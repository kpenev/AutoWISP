"""Define test case for the fit_star_shape step."""

from os import path

from autowisp.tests.h5_test_case import H5TestCase


class TestFitStarShape(H5TestCase):
    """Tests of the fit_star_shape step."""

    def test_fit_star_shape(self):
        """Run the fit_star_shape step and check the outputs."""

        self.run_step_test(
            "fit_star_shape",
            [path.join("CAL", "object"), "DR"],
            ["Background", "ProjectedSources", "ShapeFit"],
        )
