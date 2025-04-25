"""Define tests to verify master photref collector works correctly."""

import logging
from tempfile import TemporaryDirectory
from os import path
from shutil import copy

import unittest
import numpy
import pandas
from astropy.table import Table

from astrowisp.tests.utilities import FloatTestCase

from autowisp.magnitude_fitting.master_photref_collector import (
    MasterPhotrefCollector,
)
from autowisp.magnitude_fitting.tests import test_data_dir


class TestMphotrefCollector(FloatTestCase):
    """Tests of the `MasterPhotrefCollector`_ class."""

    _logger = logging.getLogger(__name__)

    # Following standard unittest assert naming convections
    # pylint: disable=invalid-name
    def assertStat(self, test_stat_fname, test_case, nphot):
        """Assert that the generated statistics matches the expected."""

        expected_stat_fname = path.join(
            test_data_dir, f"{test_case}_mfit_stat.txt"
        )
        stat_data = {
            key: pandas.read_csv(
                test_stat_fname,
                header=None,
                sep=r"\s+",
                names=(
                    ["src_id"]
                    + [
                        f"{q}_{stat}_{phot}"
                        for q in ["mag", "err"]
                        for phot in range(nphot)
                        for stat in [
                            "count",
                            "rcount",
                            "mag",
                            "meddev",
                            "medmeddev",
                        ]
                    ]
                ),
            )
            for key, fname in [
                ("test", test_stat_fname),
                ("expected", expected_stat_fname),
            ]
        }
        self.assertApproxPandas(
            stat_data["expected"], stat_data["test"], "MasterPhotrefCollector"
        )

    def assertMaster(self, test_master_fname, test_case):
        """Assert that the generated master references matches the expected."""

        self.assertApproxPandas(
            Table.read(
                path.join(test_data_dir, f"{test_case}_mphotref.fits")
            ).to_pandas(),
            Table.read(test_master_fname).to_pandas(),
        )

    # pylint: enable=invalid-name

    def test_tiny(self):
        """Test with 10 stars, 20 images, 5 photometries."""

        nstars = 10
        nimg = 20
        nphot = 5
        nmfit_iter = 1
        catalogue = {
            src_i
            + 1: numpy.array(
                (src_i / 2, src_i / 3), dtype=[("ra", float), ("dec", float)]
            )
            for src_i in range(nstars)
        }
        with TemporaryDirectory() as tempdir:
            stat_fname = path.join(tempdir, "mfit_stat.txt")
            master_fname = path.join(tempdir, "mphotref.fits")

            collector = MasterPhotrefCollector(
                statistics_fname=stat_fname,
                num_photometries=nphot,
                temp_directory=tempdir,
                source_name_format="{0:d}",
            )
            for img_i in range(nimg):
                phot = numpy.empty(
                    nstars,
                    dtype=[
                        ("source_id", int),
                        ("mag_err", numpy.float64, (nmfit_iter, nphot)),
                        ("phot_flag", numpy.uint, (nmfit_iter, nphot)),
                    ],
                )
                fitted = numpy.empty((nstars, nphot))
                for src_i in range(nstars):
                    phot["source_id"][src_i] = src_i + 1
                    phot["mag_err"][src_i] = [
                        [
                            0.01 + 0.1 * phot_i + 0.01 * fit_iter
                            for phot_i in range(nphot)
                        ]
                        for fit_iter in range(nmfit_iter)
                    ]
                    phot["phot_flag"][src_i] = [
                        [0 for _ in range(nphot)] for _ in range(nmfit_iter)
                    ]
                    fitted[src_i] = [
                        0.01 * img_i + phot_i**2 for phot_i in range(nphot)
                    ]
                collector.add_input([(phot, fitted)])
            collector.generate_master(
                master_reference_fname=master_fname,
                catalogue=catalogue,
                fit_terms_expression="O0{ra}",
                parse_source_id=None,
            )
            self.assertStat(stat_fname, "tiny", nphot)
            self.assertMaster(master_fname, "tiny")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(failfast=False)
