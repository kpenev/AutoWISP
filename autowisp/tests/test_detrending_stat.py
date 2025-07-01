"""Define Testt class for the EPD and TFA statistics genaration."""

from os import path

import pandas

from autowisp.tests import AutoWISPTestCase


class TestDetrendingStat(AutoWISPTestCase):
    """Tests for the generate_(epd/tfa)_statistics steps."""

    def run_test(self, mode):
        """Run the test for the given detrending mode ("epd" or "tfa")."""

        self.get_inputs(["LC", "MASTERS/lc_catalog*.fits", "DR"])
        self.run_step(
            [
                f"wisp-generate-{mode}-statistics",
                "-c",
                "test.cfg",
                "LC",
            ]
        )
        generated, expected = (
            pandas.read_csv(
                path.join(dirname, "MASTERS", f"{mode}_statistics.txt"),
                sep=r'\s+',
                index_col="ID",
            ).sort_values(by="ID")
            for dirname in [
                self.processing_directory,
                self.test_directory,
            ]
        )
        self.assertApproxPandas(
            expected,
            generated,
            f"{mode.upper()} statistics",
        )
        self.successful_test = True

    def test_generate_epd_statistics(self):
        """Test the generation of EPD statistics."""

        self.run_test("epd")

    def test_generate_tfa_statistics(self):
        """Test the generation of TFA statistics."""

        self.run_test("tfa")
