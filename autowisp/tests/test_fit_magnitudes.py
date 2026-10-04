"""Define test case for the fit_source_extracted_psf_map step."""

import filecmp
import re
from glob import glob
from os import makedirs, path, remove
from shutil import copy, move
from subprocess import PIPE, STDOUT, run

import h5py
import numpy
from astropy.io import fits

from autowisp.magnitude_fitting import get_master_photref
from autowisp.tests.h5_test_case import DRTestCase


class TestFitMagnitudes(DRTestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    _fitted_magnitudes = [
        f"AperturePhotometry/Version000/Aperture{ap_ind:03d}/FittedMagnitudes"
        for ap_ind in range(4)
    ]

    # The master photometric reference's header record of its population.
    _qc_keywords = {"QCRULE", "QCNIMG", "QCNEXCL"}

    def test_fit_magnitudes(self):
        """Run the fit_magnitudes step and check the outputs."""

        self.run_step_test("fit_magnitudes", "DR", self._fitted_magnitudes)
        self._assert_masters_as_expected()

    def _read_fits(self, dr_fname):
        """Return ``{path: (magnitudes, fit residual)}`` of every fit in DR."""

        result = {}
        with h5py.File(dr_fname, "r") as dr_file:
            for group in self._fitted_magnitudes:
                if group not in dr_file:
                    continue
                for version, iterations in dr_file[group].items():
                    for iteration, fitted in iterations.items():
                        result[f"{group}/{version}/{iteration}"] = (
                            fitted[()],
                            fitted.attrs["FitResidual"],
                        )
        return result

    def test_later_image_fit_like_master_builders(self):
        """An image fit against the final master matches its last pass.

        Images added after the master photometric reference exists are fit
        against it alone, and must get exactly the fit that the images which
        built it got in their last pass, at the same iteration index.
        """

        self.get_inputs(["DR", "MASTERS/mphotref_*.fits"])
        final_master = path.basename(
            sorted(
                glob(
                    path.join(
                        self.processing_directory, "MASTERS", "mphotref_*.fits"
                    )
                )
            )[-1]
        )
        last_pass = int(re.search(r"iter(\d+)\.fits$", final_master)[1]) + 1

        # Not the single photometric reference, and fit in every pass.
        later_dr = path.join(
            self.processing_directory, "DR", "10-465240_2_center.h5"
        )
        last_fits = {
            fit_path: fit
            for fit_path, fit in self._read_fits(later_dr).items()
            if fit_path.endswith(f"/Iteration{last_pass:03d}")
        }
        self.assertTrue(
            last_fits,
            f"No pass at iteration {last_pass}: the last pass was not fit "
            f"against the final master {final_master}.",
        )

        with h5py.File(later_dr, "a") as dr_file:
            for group in self._fitted_magnitudes:
                del dr_file[group]
        self.run_step(
            [
                "wisp-fit-magnitudes",
                "-c",
                "test.cfg",
                # Spelled as test.cfg's format expands it, which magfit
                # checks: PROJHOME is "." in the test data headers.
                "--master-photref-fname",
                f"./MASTERS/{final_master}",
                later_dr,
            ]
        )

        refits = self._read_fits(later_dr)
        # The DR file links the fit to every earlier iteration as well, so
        # that its iterations stay contiguous: only the last one is the fit.
        self.assertEqual(
            max(int(fit_path[-3:]) for fit_path in refits), last_pass
        )
        for fit_path, (magnitudes, residual) in last_fits.items():
            numpy.testing.assert_allclose(
                refits[fit_path][0],
                magnitudes,
                rtol=1e-8,
                atol=1e-8,
                err_msg=fit_path,
            )
            numpy.testing.assert_allclose(
                refits[fit_path][1], residual, rtol=1e-8, err_msg=fit_path
            )

    def _assert_master_reads(self, master_fname):
        """Assert the master reads as exactly its photometry tables."""

        reference = get_master_photref(master_fname)
        with fits.open(master_fname) as master:
            tables = [
                hdu.data
                for hdu in master[1:]
                if "magnitude" in hdu.columns.names
            ]
            self.assertEqual(
                next(iter(reference.values()))["mag"].shape,
                (1, len(tables)),
                master_fname,
            )
            for phot_ind, table in enumerate(tables):
                for source_id, magnitude in zip(
                    table["source_id"], table["magnitude"]
                ):
                    self.assertEqual(
                        reference[source_id]["mag"][0, phot_ind], magnitude
                    )

    def _assert_masters_match(self, expected_fname, master_fname):
        """Assert two masters match, apart from their population record."""

        with (
            fits.open(expected_fname) as expected,
            fits.open(master_fname) as master,
        ):
            self.assertEqual(
                {
                    key: value
                    for key, value in expected[0].header.items()
                    if key not in self._qc_keywords
                },
                {
                    key: value
                    for key, value in master[0].header.items()
                    if key not in self._qc_keywords
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

    def _fit_without_and_with_exclusions(self, excluded):
        """
        Fit the images not in ``excluded`` alone, then all excluding those.

        The list given to the second run names the first excluded file by a
        relative path, the second by an absolute one, and adds a file that is
        not among the inputs.

        Args:
            excluded([str]):    Two DR files, under the ``DR`` directory.

        Returns:
            [str]:
                The DR files the second run fit, those the bundle has fits for.

            str:
                The directory holding the DR files of the first run.

            str:
                The directory the masters of the first run were moved to.
        """

        dr_dir = path.join(self.processing_directory, "DR")
        subset_dir = path.join(self.processing_directory, "DR_subset")
        makedirs(subset_dir)
        fit_dr_fnames = []
        for dr_fname in glob(path.join(dr_dir, "*.h5")):
            with h5py.File(dr_fname, "a") as dr_file:
                if self._fitted_magnitudes[0] in dr_file:
                    fit_dr_fnames.append(dr_fname)
                for group in self._fitted_magnitudes:
                    if group in dr_file:
                        del dr_file[group]
            if dr_fname not in excluded:
                copy(dr_fname, subset_dir)

        masters_dir = path.join(self.processing_directory, "MASTERS")
        subset_masters_dir = path.join(masters_dir, "subset")
        makedirs(subset_masters_dir)
        self.run_step(["wisp-fit-magnitudes", "-c", "test.cfg", subset_dir])
        # Only the masters move: MASTERS/Gaia caches the catalog.
        for pattern in ["mphotref_*.fits", "mfit_stat_*.txt"]:
            for master_fname in glob(path.join(masters_dir, pattern)):
                move(master_fname, subset_masters_dir)

        exclusion_fname = path.join(self.processing_directory, "exclude.txt")
        with open(exclusion_fname, "w", encoding="utf-8") as exclusion_list:
            exclusion_list.write(
                "\n".join(
                    [
                        path.relpath(excluded[0], self.processing_directory),
                        excluded[1],
                        path.join(dr_dir, "not_an_input.h5"),
                    ]
                )
                + "\n"
            )
        self.run_step(
            [
                "wisp-fit-magnitudes",
                "-c",
                "test.cfg",
                "--qc-exclude-file",
                exclusion_fname,
                dr_dir,
            ]
        )
        return fit_dr_fnames, subset_dir, subset_masters_dir

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

        # The population lists DR files by the paths magfit was given, so it
        # is compared by name, the test data being fit in another directory.
        dr_dir = path.join(self.processing_directory, "DR")
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
                    {
                        path.join(dr_dir, path.basename(dr_fname)): included
                        for dr_fname, included in zip(
                            expected_population["dr_fname"],
                            expected_population["included"].tolist(),
                        )
                    },
                )

    def _assert_masters_built_without(
        self, excluded, fit_dr_fnames, subset_dir, subset_masters_dir
    ):
        """
        Assert the masters match the subset run's and record the populations.

        Args:
            excluded([str]):    The DR files excluded from the masters.

            fit_dr_fnames([str]):    The DR files fit, excluded included.

            subset_dir(str):    Where the DR files of the subset run are.

            subset_masters_dir(str):    Where the masters built from the kept
                images alone are.

        Returns:
            [str]:
                The masters, in order of iteration.
        """

        masters_dir = path.join(self.processing_directory, "MASTERS")
        for pattern in ["mphotref_*.fits", "mfit_stat_*.txt"]:
            self.assertEqual(
                sorted(
                    map(
                        path.basename,
                        glob(path.join(subset_masters_dir, pattern)),
                    )
                ),
                sorted(
                    map(path.basename, glob(path.join(masters_dir, pattern)))
                ),
            )
        masters = sorted(glob(path.join(masters_dir, "mphotref_*.fits")))
        for master_fname in masters:
            subset_master_fname = path.join(
                subset_masters_dir, path.basename(master_fname)
            )
            self._assert_masters_match(subset_master_fname, master_fname)
            with fits.open(master_fname) as master:
                self.assertEqual(master[0].header["QCRULE"], "")
                self.assertEqual(master[0].header["QCNIMG"], len(fit_dr_fnames))
                self.assertEqual(master[0].header["QCNEXCL"], len(excluded))
            self._assert_population(
                master_fname,
                {
                    dr_fname: dr_fname not in excluded
                    for dr_fname in fit_dr_fnames
                },
            )
            # The subset run is given only the kept files, and no exclusion
            # list, so it fits just those and builds from every one of them.
            self._assert_population(
                subset_master_fname,
                {
                    path.join(subset_dir, path.basename(dr_fname)): True
                    for dr_fname in fit_dr_fnames
                    if dr_fname not in excluded
                },
            )
        return masters

    def test_exclusions_kept_out_of_master(self):
        """Excluding images gives the master the kept images give alone.

        Fits the kept images by themselves, then all images with the others
        excluded, and compares: the masters and the kept images' fits must
        be the same, the excluded images fit at the last pass, and each
        master must list every image fit, flagging the ones it was built
        from. The list names one input by a relative path, one by an absolute
        path, and a file that is not among the inputs, which must not be
        recorded. Both the new master and one written before exclusions were
        recorded must read as their photometry tables alone.
        """

        self.get_inputs(["DR"])
        dr_dir = path.join(self.processing_directory, "DR")
        excluded = [
            path.join(dr_dir, "10-465241_2_center.h5"),
            path.join(dr_dir, "10-465243_2_center.h5"),
        ]
        fit_dr_fnames, subset_dir, subset_masters_dir = (
            self._fit_without_and_with_exclusions(excluded)
        )
        masters = self._assert_masters_built_without(
            excluded, fit_dr_fnames, subset_dir, subset_masters_dir
        )

        for subset_fname in glob(path.join(subset_dir, "*.h5")):
            dr_fname = path.join(dr_dir, path.basename(subset_fname))
            for group in self._fitted_magnitudes:
                self.assert_groups_match(subset_fname, dr_fname, group, None)
                self.assert_groups_match(dr_fname, subset_fname, group, None)

        last_pass = int(re.search(r"iter(\d+)\.fits$", masters[-1])[1]) + 1
        for dr_fname in excluded:
            self.assertEqual(
                max(
                    int(fit_path[-3:]) for fit_path in self._read_fits(dr_fname)
                ),
                last_pass,
                dr_fname,
            )

        self._assert_master_reads(masters[-1])
        for legacy_master in glob(
            path.join(self.test_directory, "legacy_masters", "*.fits")
        ):
            self._assert_master_reads(legacy_master)

    def test_existing_files_are_not_overwritten(self):
        """A file the first pass would write stops the fit before it starts.

        The master and the statistics of the first pass are each put in
        place alone, as a master built from another single photometric
        reference named alike would have left them. The step must fail
        without fitting any image and leave the file as it was.
        """

        self.get_inputs(["DR"])
        dr_fnames = glob(path.join(self.processing_directory, "DR", "*.h5"))
        for dr_fname in dr_fnames:
            with h5py.File(dr_fname, "a") as dr_file:
                for group in self._fitted_magnitudes:
                    if group in dr_file:
                        del dr_file[group]

        masters_dir = path.join(self.processing_directory, "MASTERS")
        makedirs(masters_dir, exist_ok=True)
        for pattern in ["mphotref_*_iter000.fits", "mfit_stat_*_iter000.txt"]:
            (existing,) = glob(
                path.join(self.test_directory, "MASTERS", pattern)
            )
            with self.subTest(existing=path.basename(existing)):
                in_place = copy(existing, masters_dir)
                fit = run(
                    [
                        "wisp-fit-magnitudes",
                        "-c",
                        "test.cfg",
                        path.join(self.processing_directory, "DR"),
                    ],
                    cwd=self.processing_directory,
                    check=False,
                    stdout=PIPE,
                    stderr=STDOUT,
                )
                self.assertNotEqual(
                    fit.returncode, 0, fit.stdout.decode("utf-8")
                )
                self.assertTrue(filecmp.cmp(existing, in_place, shallow=False))
                self.assertEqual(
                    [
                        dr_fname
                        for dr_fname in dr_fnames
                        if self._read_fits(dr_fname)
                    ],
                    [],
                )
                remove(in_place)
