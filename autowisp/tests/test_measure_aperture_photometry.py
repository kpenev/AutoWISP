"""Define test case for the fit_star_shape step."""

from autowisp.tests.dr_test_case import DRTestCase


class TestMeasureAperturePhotometry(DRTestCase):
    """Tests of the fit_star_shape step."""

    def test_measure_aperture_photometry(self):
        """Run the fit_star_shape step and check the outputs."""

        self.run_step_test(
            "measure_aperture_photometry", "CAL", ["AperturePhotometry"]
        )
