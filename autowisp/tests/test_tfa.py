"""Define test case for the tfa step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestTFA(H5TestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def test_tfa(self):
        """Run the tfa step and check the outputs."""

        self.run_step_test(
            "tfa",
            ["LC", "DR", "MASTERS/epd_statistics.txt"],
            [
                f"AperturePhotometry/Aperture{ap_ind:03d}/TFA"
                for ap_ind in range(4)
            ],
            output_type="LC",
        )
