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
        "big": {
            "stars": 1024,
            "images": 14 * 8 * 10,
            "photometries": 7,
            "mfit_iter": 1,
        },
    }
    _stars_in_image = {"tiny": 10, "rotatestars": 7, "big": 896}

    _catalog = None

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
                    ("xi", float),
                    ("eta", float),
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
                    ("xi", float),
                    ("eta", float),
                    ("phot_g_mean_mag", float),
                ],
            )
            for src_i in range(self._dimensions["rotatestars"]["stars"])
        }

    def _get_big_catalog(self):
        """Return the catalog to use for the big test."""

        return {
            (src_i + 1): numpy.array(
                (
                    0.3 * ((src_i + 1) % 34),
                    0.3 * (src_i // 34),
                    src_i % 5,
                ),
                dtype=[
                    ("xi", float),
                    ("eta", float),
                    ("phot_g_mean_mag", float),
                ],
            )
            for src_i in range((self._dimensions["big"]["stars"] * 9) // 8)
        }

    def _get_empty_collector_inputs(self, test_case):
        """Return empty arrays with correct dtype for collector inputs."""

        return (
            numpy.zeros(
                self._stars_in_image[test_case],
                dtype=[
                    ("source_id", int),
                    (
                        "mag_err",
                        numpy.float64,
                        (
                            self._dimensions[test_case]["mfit_iter"],
                            self._dimensions[test_case]["photometries"],
                        ),
                    ),
                    (
                        "phot_flag",
                        numpy.uint,
                        (
                            self._dimensions[test_case]["mfit_iter"],
                            self._dimensions[test_case]["photometries"],
                        ),
                    ),
                ],
            ),
            numpy.empty(
                (
                    self._stars_in_image[test_case],
                    self._dimensions[test_case]["photometries"],
                )
            ),
        )

    def _get_collector_inputs_tiny(self, img_i):
        """Feed the collector with 10 stars, 20 images, 5 photometries."""

        phot, fitted = self._get_empty_collector_inputs("tiny")
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
            fitted[src_i] = [
                0.01 * img_i + phot_i**2
                for phot_i in range(self._dimensions["tiny"]["photometries"])
            ]
        return phot, fitted

    def _get_collector_inputs_rotatestars(self, img_i):
        """Feed the collector with 10 stars, 20 images, 5 photometries."""

        phot, fitted = self._get_empty_collector_inputs("rotatestars")

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
            fitted[src_i] = [
                0.01 * img_i + phot_i**2
                for phot_i in range(
                    self._dimensions["rotatestars"]["photometries"]
                )
            ]
        return phot, fitted

    def _get_collector_inputs_big(self, img_i):
        """
        Feed the collector with the stars for the big test.

        One out of every 9 catalog stars will never appear in an image (128
        stars). Another one out of very 9 catalog stars will only appear in one
        out of 8 images. Another one of 9 catalog stars will be missing from
        1/8th of the images and 768 stars will appear in all images.

        For each star for each photometry one out of 14 images will have outlier
        flux (different image for different photometry). Ditto for photometry
        error estimates.
        """

        phot, fitted = self._get_empty_collector_inputs("big")
        dimensions = self._dimensions["big"]
        assert self._catalog is not None
        source_ind = 0
        # False positive
        # pylint: disable=not-an-iterable
        for src_id in self._catalog:
            # pylint: enable=not-an-iterable
            if (
                src_id % 9 == 0
                or (src_id % 9 == 3 and img_i % 8 != 0)
                or (src_id % 9 == 6 and img_i % 8 == 0)
            ):
                continue
            phot["source_id"][source_ind] = src_id
            if src_id % 9 == 3:
                src_img = img_i // 8
            elif src_id % 9 == 6:
                src_img = img_i - img_i // 8
            else:
                src_img = img_i
            outlier_ind = src_img % (2 * dimensions["photometries"])
            # False positive
            # pylint: disable=unsubscriptable-object
            cat_mag = self._catalog[src_id]["phot_g_mean_mag"]
            # pylint: enable=unsubscriptable-object
            phot["mag_err"][source_ind] = [
                0.003
                + 0.005 * (src_img % 10) * cat_mag
                + (
                    5.3
                    if outlier_ind == (phot_i + dimensions["photometries"])
                    else 0
                )
                for phot_i in range(dimensions["photometries"])
            ]
            fitted[source_ind] = [
                0.9 * cat_mag
                + 0.01 * (src_img % 10) * cat_mag
                + (0.1 if src_id % 9 == 5 else 0.01) * (src_img % 7)
                + ((3.0 + 2.0 * cat_mag) if outlier_ind == phot_i else 0)
                for phot_i in range(dimensions["photometries"])
            ]

            source_ind += 1
        assert source_ind == self._stars_in_image["big"]
        with numpy.printoptions(threshold=numpy.inf):
            self._logger.debug(
                "For image %d sources: %s", img_i, repr(phot["source_id"])
            )
        return phot, fitted

    def _get_empty_stat(self, test_case):
        """Return empty, but properly configured statistics DataFrame."""

        return pandas.DataFrame(
            numpy.zeros(
                self._dimensions[test_case]["stars"],
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
                        for phot in range(
                            self._dimensions[test_case]["photometries"]
                        )
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

    def _get_stat_rotatestars(self):
        """Return the expected statistics DataFrame for the rotatestars test."""

        dimensions = self._dimensions["rotatestars"]
        all_img_first_phot = numpy.arange(dimensions["images"]) * 0.01
        result = self._get_empty_stat("rotatestars")
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
                for avg in ["mean", "median"]:
                    diff_phot[avg] = src_first_phot - med_phot
                    rms_phot[avg] = numpy.sqrt(
                        getattr(numpy, f"nan{avg}")(
                            numpy.square(diff_phot[avg])
                        )
                    )

                outliers = (
                    numpy.abs(diff_phot["median"]) > 5 * rms_phot["median"]
                )
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

    def _get_stat_big(self):
        """Return the expected statistics file for the big test."""

        result = self._get_empty_stat("big")
        dimensions = self._dimensions["big"]
        cat_star_ids = numpy.arange(1, len(self._catalog) + 1)
        result.index = cat_star_ids[cat_star_ids % 9 > 0]

        for phot_i in range(dimensions["photometries"]):
            result[f"mag_count_{phot_i}"] = dimensions["images"]
            result.loc[result.index % 9 == 3, f"mag_count_{phot_i}"] = (
                dimensions["images"] // 8
            )
            result.loc[result.index % 9 == 6, f"mag_count_{phot_i}"] -= (
                dimensions["images"] // 8
            )
            result[f"mag_rcount_{phot_i}"] = result[
                f"mag_count_{phot_i}"
            ] - result[f"mag_count_{phot_i}"] // (
                2 * dimensions["photometries"]
            )

            for src_id in result.index:
                image_inds = numpy.arange(
                    result.loc[src_id, f"mag_count_{phot_i}"]
                )
                image_inds = image_inds[
                    image_inds % (2 * dimensions["photometries"]) != phot_i
                ]

                cat_mag = self._catalog[src_id]["phot_g_mean_mag"]
                mags = (0.9 + 0.01 * (image_inds % 10)) * cat_mag + (
                    0.1 if src_id % 9 == 5 else 0.01
                ) * (image_inds % 7)
                result.loc[src_id, f"mag_med_{phot_i}"] = numpy.median(mags)
                squaredev = numpy.square(
                    mags - result.loc[src_id, f"mag_med_{phot_i}"]
                )
                result.loc[src_id, f"mag_meddev_{phot_i}"] = (
                    numpy.mean(squaredev) ** 0.5
                )
                result.loc[src_id, f"mag_medmeddev_{phot_i}"] = (
                    numpy.median(squaredev) ** 0.5
                )

                image_inds = numpy.arange(
                    result.loc[src_id, f"mag_count_{phot_i}"]
                )
                image_inds = image_inds[
                    image_inds % (2 * dimensions["photometries"])
                    != (dimensions["photometries"] + phot_i)
                ]
                errors = 0.003 + 0.005 * (image_inds % 10) * cat_mag
                result.loc[src_id, f"err_med_{phot_i}"] = numpy.median(errors)
                squaredev = numpy.square(
                    errors - result.loc[src_id, f"err_med_{phot_i}"]
                )
                result.loc[src_id, f"err_meddev_{phot_i}"] = (
                    numpy.mean(squaredev) ** 0.5
                )
                result.loc[src_id, f"err_medmeddev_{phot_i}"] = (
                    numpy.median(squaredev) ** 0.5
                )

            result[f"err_count_{phot_i}"] = result[f"mag_count_{phot_i}"]
            result[f"err_rcount_{phot_i}"] = result[f"mag_rcount_{phot_i}"]

        return result

    def _get_empty_masters(self, test_name):
        """Return properly initalized masters with undefined entries."""

        return [
            pandas.DataFrame(
                numpy.empty(
                    self._dimensions[test_name]["stars"],
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
            for phot_i in range(self._dimensions[test_name]["photometries"])
        ]

    def _get_master_rotatestars(self):
        """Return the expected master DataFrame for the rotatestars test."""

        statistics = self._get_stat_rotatestars()
        result = self._get_empty_masters("rotatestars")

        for phot_i, master in enumerate(result):
            master.index = statistics.index
            master["full_count"][:] = statistics[f"mag_count_{phot_i}"]
            master["rejected_count"][:] = statistics[f"mag_rcount_{phot_i}"]
            master["magnitude"][:] = statistics[f"mag_med_{phot_i}"]
            master["mediandev"][:] = statistics[f"mag_meddev_{phot_i}"]
            master["medianmeddev"][:] = statistics[f"mag_medmeddev_{phot_i}"]
            master["scatter_excess"][:] = numpy.log10(
                master["medianmeddev"]
            ) - numpy.mean(numpy.log10(master["medianmeddev"]))

        return result

    def _get_master_big(self):
        """Return the expected master Data Frame for the big test."""

        statistics = self._get_stat_big()
        result = self._get_empty_masters("big")
        for phot_i, master in enumerate(result):
            master.index = statistics.index
            master.index = statistics.index
            master["full_count"][:] = statistics[f"mag_count_{phot_i}"]
            master["rejected_count"][:] = statistics[f"mag_rcount_{phot_i}"]
            master["magnitude"][:] = statistics[f"mag_med_{phot_i}"]
            master["mediandev"][:] = statistics[f"mag_meddev_{phot_i}"]
            master["medianmeddev"][:] = statistics[f"mag_medmeddev_{phot_i}"]

            drop = statistics.index[
                numpy.logical_or(
                    statistics.index % 9 == 5,
                    statistics["mag_count_0"]
                    < 0.8 * self._dimensions["big"]["images"],
                )
            ]
            master.drop(drop, inplace=True)
            cat_mag = numpy.array(
                [self._catalog[src]["phot_g_mean_mag"] for src in master.index]
            )
            log_dev = numpy.log10(master["medianmeddev"])
            linear_coef = (
                numpy.mean(log_dev * cat_mag)
                - numpy.mean(log_dev) * numpy.mean(cat_mag)
            ) / numpy.var(cat_mag)
            offset = numpy.mean(log_dev) - linear_coef * numpy.mean(cat_mag)

            master["scatter_excess"][:] = (
                log_dev - linear_coef * cat_mag - offset
            )

        return result

    def perform_test(self, test_name):
        """Run a single test."""

        pandas.options.display.min_rows = 60
        pandas.options.display.max_columns = 200
        self._catalog = getattr(self, f"_get_{test_name}_catalog")()
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
                catalog=self._catalog,
                fit_terms_expression=(
                    "O1{phot_g_mean_mag}" if test_name == "big" else "O0{xi}"
                ),
                fit_outlier_threshold=6.0,
                parse_source_id=None,
            )
            self._assertStat(
                stat_fname,
                test_name,
                self._dimensions[test_name]["photometries"],
            )
            self._assertMaster(master_fname, test_name)

    def test_tiny(self):
        """Tiny super-fast test."""

        self.perform_test("tiny")

    def test_rotatestars(self):
        """Test with rotating collection of stars between images."""

        self.perform_test("rotatestars")

    def test_big(self):
        """Big test."""

        self.perform_test("big")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d",
    )
    unittest.main(failfast=False)
