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
from autowisp.tests.magfit_test_case import MagfitMastersTestCase


class TestFitMagnitudes(DRTestCase, MagfitMastersTestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    _fitted_magnitudes = [
        f"AperturePhotometry/Version000/Aperture{ap_ind:03d}/FittedMagnitudes"
        for ap_ind in range(4)
    ]

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

    def _assert_qc_included(self, dr_fname, included):
        """Assert every photometry of the DR file records ``included``."""

        with h5py.File(dr_fname, "r") as dr_file:
            flags = {
                f"{group}/{version}": fitted.attrs["QCIncluded"]
                for group in self._fitted_magnitudes
                if group in dr_file
                for version, fitted in dr_file[group].items()
            }
        self.assertEqual(len(flags), len(self._fitted_magnitudes), dr_fname)
        for flag_path, flag in flags.items():
            self.assertEqual(flag, included, f"{dr_fname}: {flag_path}")

    def test_later_image_fit_like_master_builders(self):
        """An image fit against the final master matches its last pass.

        Images added after the master photometric reference exists are fit
        against it alone, and must get exactly the fit that the images which
        built it got in their last pass, at the same iteration index. One of
        two such images is on the exclusion list, which changes nothing about
        its fit, only the quality flag it records.
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

        # Neither is the single photometric reference; both are fit in every
        # pass. The second is the one on the exclusion list.
        later_drs = [
            path.join(self.processing_directory, "DR", fname)
            for fname in ("10-465240_2_center.h5", "10-465244_2_center.h5")
        ]
        last_fits = {}
        for dr_fname in later_drs:
            last_fits[dr_fname] = {
                fit_path: fit
                for fit_path, fit in self._read_fits(dr_fname).items()
                if fit_path.endswith(f"/Iteration{last_pass:03d}")
            }
            self.assertTrue(
                last_fits[dr_fname],
                f"No pass at iteration {last_pass} in {dr_fname}: the last "
                f"pass was not fit against the final master {final_master}.",
            )
            with h5py.File(dr_fname, "a") as dr_file:
                for group in self._fitted_magnitudes:
                    del dr_file[group]

        exclusion_fname = path.join(self.processing_directory, "exclude.txt")
        with open(exclusion_fname, "w", encoding="utf-8") as exclusion_list:
            exclusion_list.write(later_drs[1] + "\n")
        self.run_step(
            [
                "wisp-fit-magnitudes",
                "-c",
                "test.cfg",
                "--master-photref-fname",
                f"{{PROJHOME}}/MASTERS/{final_master}",
                "--qc-exclude-file",
                exclusion_fname,
            ]
            + later_drs
        )

        for dr_fname, included in zip(later_drs, (True, False)):
            refits = self._read_fits(dr_fname)
            # The DR file links the fit to every earlier iteration as well, so
            # that its iterations stay contiguous: only the last one is the
            # fit.
            self.assertEqual(
                max(int(fit_path[-3:]) for fit_path in refits),
                last_pass,
                dr_fname,
            )
            for fit_path, (magnitudes, residual) in last_fits[dr_fname].items():
                numpy.testing.assert_allclose(
                    refits[fit_path][0],
                    magnitudes,
                    rtol=1e-8,
                    atol=1e-8,
                    err_msg=f"{dr_fname}: {fit_path}",
                )
                numpy.testing.assert_allclose(
                    refits[fit_path][1],
                    residual,
                    rtol=1e-8,
                    err_msg=f"{dr_fname}: {fit_path}",
                )
            self._assert_qc_included(dr_fname, included)

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
            # Listed in stored form, the processing directory being the
            # project home.
            self._assert_population(
                master_fname,
                {
                    f"{{PROJHOME}}/DR/{path.basename(dr_fname)}": (
                        dr_fname not in excluded
                    )
                    for dr_fname in fit_dr_fnames
                },
            )
            # The subset run is given only the kept files, and no exclusion
            # list, so it fits just those and builds from every one of them.
            subset_stored = "{PROJHOME}/" + path.relpath(
                subset_dir, self.processing_directory
            )
            self._assert_population(
                subset_master_fname,
                {
                    f"{subset_stored}/{path.basename(dr_fname)}": True
                    for dr_fname in fit_dr_fnames
                    if dr_fname not in excluded
                },
            )
        return masters

    def test_exclusions_kept_out_of_master(self):
        """Excluding images gives the master the kept images give alone.

        Fits the kept images by themselves, then all images with the others
        excluded, and compares: the masters and the kept images' fits must
        be the same, the excluded images fit at the last pass, each master
        must list every image fit, flagging the ones it was built from, and
        each image must record whether it passes the quality cut. The list
        names one input by a relative path, one by an absolute
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
        for dr_fname in fit_dr_fnames:
            self._assert_qc_included(dr_fname, dr_fname not in excluded)

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
