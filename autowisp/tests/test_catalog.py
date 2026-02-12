"""Test cases for catalog queries."""

from os import path
from glob import glob

from astropy import units

from autowisp.catalog import read_catalog_file, create_catalog_file

from autowisp.tests.fits_test_case import FITSTestCase


class TestCatalog(FITSTestCase):
    """Compare catalog queries to expected results."""

    def _test_single_query(self, expected_fname):
        """Query a catalog per the header in given name and compare results."""

        header = read_catalog_file(expected_fname, return_metadata=True)[1]
        query_config = {
            k: header[k.upper()] * units.deg
            for k in ["ra", "dec", "width", "height"]
        }
        query_config["columns"] = [
            value[: -len("_orig")] if value.endswith("_orig") else value
            for key, value in header.items()
            if key.startswith("TTYPE")
            and value not in ["RA", "Dec", "magnitude"]
        ]
        query_config["epoch"] = header["EPOCH"] * units.yr
        query_config["magnitude_limit"] = header["MAGMAX"]
        query_config["magnitude_expression"] = header["MAGEXPR"]
        generated_fname = path.join(
            self.processing_directory,
            "catalog_tests",
            path.basename(expected_fname),
        )
        create_catalog_file(generated_fname, **query_config, verbose=True)
        self.assert_fits_match(expected_fname, generated_fname)
        self.successful_test = False

    def test_catalog_queries(self):
        """Verify each of the queries for which expected results are known."""

        test_glob = path.join(
            self.test_directory,
            "catalog_tests",
            "test_catalog_*.fits",
        )
        print(f"Test directory contents: {glob(self.test_directory + '/*')}")
        print(f"Looking for test catalogs matching {test_glob}")
        for expected_fname in glob(test_glob):
            print(f"Testing {expected_fname}")
            self._test_single_query(expected_fname)
