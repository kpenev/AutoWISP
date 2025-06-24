"""Autowisp unit-test init."""

from os import path
from unittest import TestCase

import numpy

from autowisp.fits_utilities import read_image_components

autowisp_dir = path.dirname(path.dirname(path.abspath(__file__)))


class FITSTestCase(TestCase):
    """Add assert for comparing AutoWISP generated FITS files."""

    def assert_fits_match(self, fname1, fname2):
        """Check if two headers match."""

        fits_components = [
            dict(
                zip(
                    ["image", "error", "mask", "header"],
                    read_image_components(fits_fname),
                )
            )
            for fits_fname in [fname1, fname2]
        ]

        self.assertTrue(
            set(fits_components[0]["header"].keys())
            == set(fits_components[1]["header"].keys()),
            f"Headers of {fname1} and {fname2} do not have the same keys!",
        )
        for key, value in fits_components[0]["header"].items():
            if key == "COMMENT" or key.strip() == "":
                continue
            if key in [
                f"M{tp.upper()}FNM" for tp in ["bias", "dark", "flat"]
            ]:
                self.assertEqual(
                    path.basename(fits_components[1]["header"][key]),
                    path.basename(value),
                    f"Master {key[1:-3].lower()} does not match between "
                    f"{fname1} and {fname2}.",
                )
            else:
                self.assertEqual(
                    fits_components[1]["header"][key],
                    value,
                    f"Value for key {key!r} does not match between {fname1} and"
                    f" {fname2}.",
                )

        for component in ["image", "error"]:
            self.assertTrue(
                numpy.isclose(
                    fits_components[0][component],
                    fits_components[1][component],
                    rtol=1e-8,
                    atol=1e-8,
                ).all(),
                f"{component.title()} pixels in {fname1} do not match "
                f"{component} pixels in {fname2}!",
            )
        self.assertTrue(
            (fits_components[0]["mask"] == fits_components[1]["mask"]).all(),
            f"Pixel mask in {fname1} do not match pixel mask in {fname2}!",
        )
