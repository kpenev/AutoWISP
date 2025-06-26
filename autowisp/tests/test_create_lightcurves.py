"""Define test case for the create_lightcurves step."""

from autowisp.tests.h5_test_case import H5TestCase


class TestCreateLightcurves(H5TestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def is_postprocessing(self, lc_path):
        """Check if the given LC path is related to postprocessing."""

        return (
            "/EPD/" in lc_path
            or "/TFA/" in lc_path
            or lc_path.endswith("/TFA")
            or lc_path.endswith("/EPD")
        )

    def test_create_lightcurves(self):
        """Run the create_lightcurves step and check the outputs."""

        self.run_step_test(
            "create_lightcurves",
            "DR",
            ["/"],
            output_type="LC",
            ignore=self.is_postprocessing,
        )
