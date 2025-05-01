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

from autowisp.magnitude_fitting.master_photref_collector_zarr import (
    MasterPhotrefCollector,
)
from autowisp.magnitude_fitting.tests import test_data_dir


class TestMphotrefCollector(FloatTestCase):
    """Tests of the `MasterPhotrefCollector`_ class."""

    _logger = logging.getLogger(__name__)

    _dimensions = {
        "tiny": {"stars": 10, "images": 20, "photometries": 3, "mfit_iter": 1},
        "rotatestars": {
            "stars": 15,
            "images": 19,
            "photometries": 5,
            "mfit_iter": 1,
        },
    }
    _stars_in_image = {
        "tiny": 10,
        "rotatestars": 7,
    }

    # Following standard unittest assert naming convections
    # pylint: disable=invalid-name
    def _assertStat(self, test_stat_fname, test_case, nphot):
        """Assert that the generated statistics matches the expected."""

        self.set_tolerance(0.0, 1e-7)
        if not hasattr(self, f"_get_stat_{test_case}"):
            expected_stat_fname = path.join(
                test_data_dir, f"{test_case}_mfit_stat.txt"
            )
            read_stats = [
                ("test", test_stat_fname),
                ("expected", expected_stat_fname),
            ]
        else:
            read_stats = [("test", test_stat_fname)]

        stat_data = {
            key: pandas.read_csv(
                fname,
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
                            "med",
                            "meddev",
                            "medmeddev",
                        ]
                    ]
                ),
                index_col="src_id",
            ).sort_index()
            for key, fname in read_stats
        }
        if "expected" not in stat_data:
            stat_data["expected"] = getattr(self, f"_get_stat_{test_case}")()

        self.assertApproxPandas(
            stat_data["expected"], stat_data["test"], "MasterPhotrefCollector"
        )

    @staticmethod
    def _get_fits_df(fits_fname, hdu_index):
        return (
            Table.read(fits_fname, hdu=hdu_index)
            .to_pandas()
            .set_index("source_id")
            .sort_index()
        )

    def _assertMaster(self, test_master_fname, test_case):
        """Assert that the generated master references matches the expected."""

        self.set_tolerance(10.0, 1e-12)
        if hasattr(self, f"_get_master_{test_case}"):
            expected_master = getattr(self, f"_get_master_{test_case}")()
        else:
            expected_master = None

        for hdu_index in range(
            1, 1 + self._dimensions[test_case]["photometries"]
        ):
            self.assertApproxPandas(
                (
                    self._get_fits_df(
                        path.join(test_data_dir, f"{test_case}_mphotref.fits"),
                        hdu_index,
                    )
                    if expected_master is None
                    else expected_master[hdu_index - 1]
                ),
                self._get_fits_df(test_master_fname, hdu_index),
            )

    # pylint: enable=invalid-name

    def _get_tiny_catalog(self):
        """Return the catalog to use for the tiny test."""

        return {
            (src_i + 1): numpy.array(
                (src_i / 2, src_i / 3, src_i % 5),
                dtype=[
                    ("ra", float),
                    ("dec", float),
                    ("phot_g_mean_mag", float),
                ],
            )
            for src_i in range(self._dimensions["tiny"]["stars"])
        }

    def _get_rotatestars_catalog(self):
        """Return the catalog to use for the rotatestars test."""

        return {
            (src_i + 1): numpy.array(
                (src_i / 2, src_i / 3, src_i % 4),
                dtype=[
                    ("ra", float),
                    ("dec", float),
                    ("phot_g_mean_mag", float),
                ],
            )
            for src_i in range(self._dimensions["rotatestars"]["stars"])
        }

    def _get_collector_inputs_tiny(self, img_i):
        """Feed the collector with 10 stars, 20 images, 5 photometries."""

        phot = numpy.empty(
            self._dimensions["tiny"]["stars"],
            dtype=[
                ("source_id", int),
                (
                    "mag_err",
                    numpy.float64,
                    (
                        self._dimensions["tiny"]["mfit_iter"],
                        self._dimensions["tiny"]["photometries"],
                    ),
                ),
                (
                    "phot_flag",
                    numpy.uint,
                    (
                        self._dimensions["tiny"]["mfit_iter"],
                        self._dimensions["tiny"]["photometries"],
                    ),
                ),
            ],
        )
        fitted = numpy.empty(
            tuple(
                self._dimensions["tiny"][dim]
                for dim in ["stars", "photometries"]
            )
        )
        for src_i in range(self._dimensions["tiny"]["stars"]):
            phot["source_id"][src_i] = src_i + 1
            phot["mag_err"][src_i] = [
                [
                    0.01 + 0.1 * phot_i + 0.01 * fit_iter
                    for phot_i in range(
                        self._dimensions["tiny"]["photometries"]
                    )
                ]
                for fit_iter in range(self._dimensions["tiny"]["mfit_iter"])
            ]
            phot["phot_flag"][src_i] = [
                [0 for _ in range(self._dimensions["tiny"]["photometries"])]
                for _ in range(self._dimensions["tiny"]["mfit_iter"])
            ]
            fitted[src_i] = [
                0.01 * img_i + phot_i**2
                for phot_i in range(self._dimensions["tiny"]["photometries"])
            ]
        return phot, fitted

    def _get_collector_inputs_rotatestars(self, img_i):
        """Feed the collector with 10 stars, 20 images, 5 photometries."""

        stars_in_image = 7

        phot = numpy.empty(
            self._stars_in_image["rotatestars"],
            dtype=[
                ("source_id", int),
                (
                    "mag_err",
                    numpy.float64,
                    (
                        self._dimensions["rotatestars"]["mfit_iter"],
                        self._dimensions["rotatestars"]["photometries"],
                    ),
                ),
                (
                    "phot_flag",
                    numpy.uint,
                    (
                        self._dimensions["rotatestars"]["mfit_iter"],
                        self._dimensions["rotatestars"]["photometries"],
                    ),
                ),
            ],
        )
        fitted = numpy.empty(
            (
                self._stars_in_image["rotatestars"],
                self._dimensions["rotatestars"]["photometries"],
            )
        )
        for src_i in range(self._stars_in_image["rotatestars"]):
            phot["source_id"][src_i] = (src_i + img_i) % self._dimensions[
                "rotatestars"
            ]["stars"] + 1
            phot["mag_err"][src_i] = [
                [
                    0.01 + 0.1 * phot_i + 0.01 * fit_iter
                    for phot_i in range(
                        self._dimensions["rotatestars"]["photometries"]
                    )
                ]
                for fit_iter in range(
                    self._dimensions["rotatestars"]["mfit_iter"]
                )
            ]
            phot["phot_flag"][src_i] = [
                [
                    0
                    for _ in range(
                        self._dimensions["rotatestars"]["photometries"]
                    )
                ]
                for _ in range(self._dimensions["rotatestars"]["mfit_iter"])
            ]
            fitted[src_i] = [
                0.01 * img_i + phot_i**2
                for phot_i in range(
                    self._dimensions["rotatestars"]["photometries"]
                )
            ]
        return phot, fitted

    def _get_stat_rotatestars(self):
        """Return the expected statistics DataFrame for the rotatestars test."""

        dimensions = self._dimensions["rotatestars"]
        all_img_first_phot = numpy.arange(dimensions["images"]) * 0.01
        result = pandas.DataFrame(
            numpy.zeros(
                dimensions["stars"],
                dtype=(
                    [
                        ("source_id", numpy.uint64),
                    ]
                    + [
                        (
                            f"{q}_{stat}_{phot}",
                            (int if stat.endswith("count") else float),
                        )
                        for q in ["mag", "err"]
                        for phot in range(dimensions["photometries"])
                        for stat in [
                            "count",
                            "rcount",
                            "med",
                            "meddev",
                            "medmeddev",
                        ]
                    ]
                ),
            )
        ).set_index("source_id")
        result.index = numpy.arange(1, dimensions["stars"] + 1)

        for src_i in range(dimensions["stars"]):
            src_first_phot = numpy.copy(all_img_first_phot)
            src_in_image = (
                (src_i - numpy.arange(dimensions["images"]))
                % dimensions["stars"]
            ) < self._stars_in_image["rotatestars"]
            src_first_phot[numpy.logical_not(src_in_image)] = numpy.nan
            outliers = numpy.array([True])
            med_phot = {}
            diff_phot = {}
            rms_phot = {}
            while outliers.any():
                med_phot = numpy.nanmedian(src_first_phot)
                print(f"Med phot: {med_phot!r}")
                for avg in ["mean", "median"]:
                    diff_phot[avg] = src_first_phot - med_phot
                    rms_phot[avg] = numpy.sqrt(
                        getattr(numpy, f"nan{avg}")(
                            numpy.square(diff_phot[avg])
                        )
                    )
                    if src_i == 0:
                        print(f"Diff {avg}: {diff_phot[avg]!r}")
                        print(f"Rms {avg}: {rms_phot[avg]!r}")

                outliers = (
                    numpy.abs(diff_phot["median"]) > 5 * rms_phot["median"]
                )
                if src_i == 0:
                    print(f"Outliers: {outliers!r}")
                src_first_phot[outliers] = numpy.nan
            for phot_i in range(dimensions["photometries"]):
                result.at[src_i + 1, f"mag_count_{phot_i}"] = src_in_image.sum()
                result.at[src_i + 1, f"mag_rcount_{phot_i}"] = numpy.isfinite(
                    src_first_phot
                ).sum()
                result.at[src_i + 1, f"mag_med_{phot_i}"] = med_phot + phot_i**2
                result.at[src_i + 1, f"mag_meddev_{phot_i}"] = rms_phot["mean"]
                result.at[src_i + 1, f"mag_medmeddev_{phot_i}"] = rms_phot[
                    "median"
                ]
        for phot_i in range(dimensions["photometries"]):
            result[f"err_count_{phot_i}"] = result[f"mag_count_{phot_i}"]
            result[f"err_rcount_{phot_i}"] = result[f"mag_count_{phot_i}"]
            result[f"err_med_{phot_i}"] = 0.01 + 0.1 * phot_i
        return result

    def _get_master_rotatestars(self):
        """Return the expected master DataFrame for the rotatestars test."""

        statistics = self._get_stat_rotatestars()
        dimensions = self._dimensions["rotatestars"]
        result = []

        for phot_i in range(dimensions["photometries"]):
            master = pandas.DataFrame(
                numpy.empty(
                    dimensions["stars"],
                    dtype=(
                        [
                            ("source_id", numpy.uint64),
                            ("full_count", numpy.uint64),
                            ("rejected_count", numpy.uint64),
                            ("magnitude", numpy.float64),
                            ("mediandev", numpy.float64),
                            ("medianmeddev", numpy.float64),
                            ("scatter_excess", numpy.float64),
                        ]
                    ),
                )
            ).set_index("source_id")
            master.index = statistics.index
            master["full_count"][:] = statistics[f"mag_count_{phot_i}"]
            master["rejected_count"][:] = statistics[f"mag_rcount_{phot_i}"]
            master["magnitude"][:] = statistics[f"mag_med_{phot_i}"]
            master["mediandev"][:] = statistics[f"mag_meddev_{phot_i}"]
            master["medianmeddev"][:] = statistics[f"mag_medmeddev_{phot_i}"]
            master["scatter_excess"][:] = numpy.log10(
                master["medianmeddev"]
            ) - numpy.mean(numpy.log10(master["medianmeddev"]))
            result.append(master)
        return result

    def perform_test(self, test_name):
        """Run a single test."""

        with TemporaryDirectory() as tempdir:
            stat_fname = path.join(tempdir, "mfit_stat.txt")
            master_fname = path.join(tempdir, "mphotref.fits")

            collector = MasterPhotrefCollector(
                statistics_fname=stat_fname,
                num_photometries=self._dimensions[test_name]["photometries"],
                num_frames=self._dimensions[test_name]["images"],
                temp_directory=tempdir,
                source_name_format="{0:d}",
            )
            for img_i in range(self._dimensions[test_name]["images"]):
                collector.add_input(
                    [getattr(self, "_get_collector_inputs_" + test_name)(img_i)]
                )
            collector.generate_master(
                master_reference_fname=master_fname,
                catalog=getattr(self, f"_get_{test_name}_catalog")(),
                fit_terms_expression="O0{ra}",
                parse_source_id=None,
            )
            # copy(
            #    stat_fname,
            #    path.join(test_data_dir, "rotatestars_mfit_stat.txt"),
            # )
            # copy(
            #    master_fname,
            #    path.join(test_data_dir, "rotatestars_mphotref.fits")
            # )
            self._assertStat(
                stat_fname,
                test_name,
                self._dimensions[test_name]["photometries"],
            )
            self._assertMaster(master_fname, test_name)

    # def test_tiny(self):
    #    """Tiny super-fast test."""

    #    self.perform_test("tiny")

    def test_rotatestars(self):
        """Test with rotating collection of stars between images."""

        self.perform_test("rotatestars")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d",
    )
    unittest.main(failfast=False)
