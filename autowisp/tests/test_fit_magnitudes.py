"""Define test case for the fit_source_extracted_psf_map step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestFitMagnitudes(H5TestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def test_fit_magnitudes(self):
        """Run the fit_source_extracted_psf_map step and check the outputs."""

        self.run_step_test(
            "fit_magnitudes",
            "DR",
            [
                f"AperturePhotometry/Version000/Aperture{ap_ind:03d}/"
                "FittedMagnitudes"
                for ap_ind in range(4)
            ],
        )
