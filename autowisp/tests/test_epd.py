"""Define test case for the epd step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestEPD(H5TestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def test_epd(self):
        """Run the epd step and check the outputs."""

        self.run_step_test(
            "epd",
            "LC",
            [
                f"AperturePhotometry/Aperture{ap_ind:03d}/EPD"
                for ap_ind in range(4)
            ],
            output_type="LC",
        )
