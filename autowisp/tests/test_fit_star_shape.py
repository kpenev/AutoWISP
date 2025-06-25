"""Define test case for the fit_star_shape step."""

from autowisp.tests.dr_test_case import DRTestCase

class TestFitStarShape(DRTestCase):
    """Tests of the fit_star_shape step."""

    def test_fit_star_shape(self):
        """Run the fit_star_shape step and check the outputs."""

        self.run_step_test(
            "fit_star_shape",
            "CAL",
            ['Background', 'ProjectedSources', 'ShapeFit'],
        )
