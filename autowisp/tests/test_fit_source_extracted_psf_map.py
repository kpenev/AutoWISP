"""Define test case for the fit_source_extracted_psf_map step."""

from autowisp.tests.dr_test_case import DRTestCase


class TestFitSourceExtractedPSFMap(DRTestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    def test_fit_source_extracted_psf_map(self):
        """Run the fit_source_extracted_psf_map step and check the outputs."""

        self.run_step_test(
            "fit_source_extracted_psf_map",
            "DR",
            ["SourceExtraction/Version000/PSFMap"],
        )
