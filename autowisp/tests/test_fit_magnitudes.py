"""Define test case for the fit_source_extracted_psf_map step."""

import re
from glob import glob
from os import path

import h5py
import numpy

from autowisp.tests.h5_test_case import DRTestCase


class TestFitMagnitudes(DRTestCase):
    """Tests of the fit_source_extracted_psf_map step."""

    _fitted_magnitudes = [
        f"AperturePhotometry/Version000/Aperture{ap_ind:03d}/FittedMagnitudes"
        for ap_ind in range(4)
    ]

    def test_fit_magnitudes(self):
        """Run the fit_magnitudes step and check the outputs."""

        self.run_step_test("fit_magnitudes", "DR", self._fitted_magnitudes)

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
