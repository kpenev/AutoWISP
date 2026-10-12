"""Define test case for the create_lightcurves step."""

from glob import glob
from os import path

from autowisp.tests.fits_test_case import FITSTestCase
from autowisp.tests.h5_test_case import H5TestCase


class TestCreateLightcurves(H5TestCase, FITSTestCase):
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

        # The lightcurve catalogues the step writes besides the lightcurves.
        generated, expected = (
            sorted(glob(path.join(directory, "MASTERS", "lc_catalog_*.fits")))
            for directory in (self.processing_directory, self.test_directory)
        )
        self.assertEqual(
            [path.basename(fname) for fname in generated],
            [path.basename(fname) for fname in expected],
        )
        for gen_fname, exp_fname in zip(generated, expected):
            self.assert_fits_match(exp_fname, gen_fname)
