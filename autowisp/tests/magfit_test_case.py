"""Define a class comparing the masters magnitude fitting writes."""

from glob import glob
from os import path

import numpy
from astropy.io import fits

from autowisp.tests import AutoWISPTestCase


class MagfitMastersTestCase(AutoWISPTestCase):
    """Add asserts comparing master photometric references and statistics."""

    # The master photometric reference's header record of its population.
    _qc_keywords = {"QCRULE", "QCNIMG", "QCNEXCL"}

    def _assert_masters_match(self, expected_fname, master_fname):
        """Assert two masters match, apart from their population record."""

        with (
            fits.open(expected_fname) as expected,
            fits.open(master_fname) as master,
        ):
            # Blank cards are padding, e.g. left by converting paths in place.
            self.assertEqual(
                {
                    key: value
                    for key, value in expected[0].header.items()
                    if key.strip() and key not in self._qc_keywords
                },
                {
                    key: value
                    for key, value in master[0].header.items()
                    if key.strip() and key not in self._qc_keywords
                },
            )
            self.assertEqual(
                [hdu.name for hdu in expected], [hdu.name for hdu in master]
            )
            for expected_hdu, master_hdu in zip(expected[1:], master[1:]):
                if expected_hdu.name != "MPHOTREF":
                    continue
                # Rows follow the order in which images finished fitting.
                expected_data, master_data = (
                    numpy.sort(hdu.data, order="source_id")
                    for hdu in (expected_hdu, master_hdu)
                )
                for column in expected_data.dtype.names:
                    numpy.testing.assert_allclose(
                        master_data[column],
                        expected_data[column],
                        rtol=1e-8,
                        atol=0 if column == "source_id" else 1e-8,
                        err_msg=f"{master_fname}: {column}",
                    )

    def _assert_population(self, master_fname, expected):
        """Assert the master's INMASTER table holds ``{dr_fname: included}``."""

        with fits.open(master_fname) as master:
            population = master["INMASTER"].data
            self.assertEqual(
                dict(
                    zip(
                        population["dr_fname"],
                        population["included"].tolist(),
                    )
                ),
                expected,
                master_fname,
            )
            self.assertEqual(len(population), len(expected), master_fname)

    def _assert_magfit_stat_match(self, expected_fname, stat_fname):
        """Assert two magfit statistics files match, in any order of rows."""

        sorted_stats = []
        for fname in (expected_fname, stat_fname):
            source_ids = numpy.loadtxt(fname, usecols=0, dtype=numpy.uint64)
            order = numpy.argsort(source_ids)
            sorted_stats.append(
                (
                    source_ids[order],
                    numpy.loadtxt(fname, ndmin=2)[order, 1:],
                )
            )
        numpy.testing.assert_array_equal(
            sorted_stats[1][0], sorted_stats[0][0], err_msg=stat_fname
        )
        numpy.testing.assert_allclose(
            sorted_stats[1][1],
            sorted_stats[0][1],
            rtol=1e-8,
            atol=1e-8,
            err_msg=stat_fname,
        )

    def _assert_masters_as_expected(self):
        """Assert the masters and statistics written match the test data's."""

        masters_dir = path.join(self.processing_directory, "MASTERS")
        expected_dir = path.join(self.test_directory, "MASTERS")
        for pattern in ["mphotref_*.fits", "mfit_stat_*.txt"]:
            self.assertEqual(
                sorted(
                    map(path.basename, glob(path.join(masters_dir, pattern)))
                ),
                sorted(
                    map(path.basename, glob(path.join(expected_dir, pattern)))
                ),
            )

        for stat_fname in glob(path.join(masters_dir, "mfit_stat_*.txt")):
            self._assert_magfit_stat_match(
                path.join(expected_dir, path.basename(stat_fname)), stat_fname
            )

        for master_fname in glob(path.join(masters_dir, "mphotref_*.fits")):
            expected_fname = path.join(
                expected_dir, path.basename(master_fname)
            )
            self._assert_masters_match(expected_fname, master_fname)
            with (
                fits.open(expected_fname) as expected,
                fits.open(master_fname) as master,
            ):
                for keyword in self._qc_keywords:
                    self.assertEqual(
                        master[0].header[keyword],
                        expected[0].header[keyword],
                        f"{master_fname}: {keyword}",
                    )
                expected_population = expected["INMASTER"].data
                self._assert_population(
                    master_fname,
                    dict(
                        zip(
                            expected_population["dr_fname"],
                            expected_population["included"].tolist(),
                        )
                    ),
                )
