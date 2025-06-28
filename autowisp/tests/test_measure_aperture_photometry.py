"""Define test case for the fit_star_shape step."""

from os import path

from autowisp.tests.h5_test_case import H5TestCase


class TestMeasureAperturePhotometry(H5TestCase):
    """Tests of the fit_star_shape step."""

    @staticmethod
    def is_magfit(dr_path):
        """Check if the given path within a DR file is related to magfit."""

        return "FittedMagnitudes" in dr_path

    def test_measure_aperture_photometry(self):
        """Run the fit_star_shape step and check the outputs."""

        self.run_step_test(
            "measure_aperture_photometry",
            [path.join("CAL", "object"), "DR"],
            ["AperturePhotometry"],
            ignore=self.is_magfit,
        )
