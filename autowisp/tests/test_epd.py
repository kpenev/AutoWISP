"""Define test case for the epd step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestEPD(H5TestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def test_epd(self):
        """Run the epd step and check the outputs."""

        self.run_step_test(
            "epd",
            # The single photref DR is needed to derive the datasets to
            # detrend when ``epd-datasets`` is not configured.
            ["LC", "DR"],
            [
                f"AperturePhotometry/Aperture{ap_ind:03d}/EPD"
                for ap_ind in range(4)
            ],
            output_type="LC",
            ignore=lambda name: name.endswith("/EPD/FitProperties/Filter"),
        )
