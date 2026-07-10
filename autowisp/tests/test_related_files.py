"""Unit tests for the per-step ``related_files`` classifiers (item 9).

Each parallel/step site turns its work item (plus any batch-constant
auxiliary inputs) into :class:`RelatedFile` entries, so an error carries
the artifact(s) it was about. These test the classifier helpers in
isolation; the scoping mechanism itself is covered in
``test_error_context.py``.
"""

import unittest

from autowisp.exceptions import FileKind
from autowisp.processing_steps.calibrate import _calibration_related_files
from autowisp.processing_steps.fit_star_shape import _frame_set_related_files
from autowisp.light_curves.apply_correction import _detrending_related_files
from autowisp.magnitude_fitting.iterative_refit import _magfit_related_files


def _pairs(related):
    """(kind, posix path) tuples for order-independent assertions."""

    return [(rf.kind, rf.path.as_posix()) for rf in related]


class TestCalibrationRelatedFiles(unittest.TestCase):
    """calibrate: the raw image plus every master applied."""

    def test_channel_dependent_masters(self):
        """Channel-keyed master dicts contribute one entry per channel."""

        config = {
            "master_bias": {"R": "/M/bias_R.fits", "G": "/M/bias_G.fits"},
            "master_dark": None,  # not applied -> skipped
            "master_flat": "/M/flat.fits",  # bare filename also accepted
        }
        pairs = _pairs(_calibration_related_files("/RAW/img.fits", config))

        self.assertEqual(pairs[0], (FileKind.RAW_IMAGE, "/RAW/img.fits"))
        self.assertIn((FileKind.MASTER_BIAS, "/M/bias_R.fits"), pairs)
        self.assertIn((FileKind.MASTER_BIAS, "/M/bias_G.fits"), pairs)
        self.assertIn((FileKind.MASTER_FLAT, "/M/flat.fits"), pairs)
        # The un-applied dark contributes nothing.
        self.assertFalse(any(kind is FileKind.MASTER_DARK for kind, _ in pairs))

    def test_no_masters(self):
        """With no masters configured, only the raw image is attached."""

        pairs = _pairs(_calibration_related_files("/RAW/img.fits", {}))
        self.assertEqual(pairs, [(FileKind.RAW_IMAGE, "/RAW/img.fits")])


class TestMagfitRelatedFiles(unittest.TestCase):
    """fit_magnitudes: the DR item plus the single/master photref."""

    def test_with_master_photref(self):
        pairs = _pairs(
            _magfit_related_files(
                "/DR/x.h5",
                single_photref="/DR/sp.h5",
                master_photref="/M/mp.fits",
            )
        )
        self.assertEqual(
            pairs,
            [
                (FileKind.DR_FILE, "/DR/x.h5"),
                (FileKind.DR_FILE, "/DR/sp.h5"),
                (FileKind.MASTER_PHOTREF, "/M/mp.fits"),
            ],
        )

    def test_without_master_photref(self):
        """Before a master photref exists it is simply omitted."""

        pairs = _pairs(
            _magfit_related_files(
                "/DR/x.h5", single_photref="/DR/sp.h5", master_photref=None
            )
        )
        self.assertEqual(
            pairs,
            [(FileKind.DR_FILE, "/DR/x.h5"), (FileKind.DR_FILE, "/DR/sp.h5")],
        )


class TestFrameSetRelatedFiles(unittest.TestCase):
    """fit_star_shape: every frame in a simultaneous-fit set."""

    def test_all_frames(self):
        pairs = _pairs(_frame_set_related_files(["/CAL/a.fits", "/CAL/b.fits"]))
        self.assertEqual(
            pairs,
            [
                (FileKind.CALIBRATED_IMAGE, "/CAL/a.fits"),
                (FileKind.CALIBRATED_IMAGE, "/CAL/b.fits"),
            ],
        )


class TestDetrendingRelatedFiles(unittest.TestCase):
    """epd/tfa: the light curve plus its single photometric reference."""

    def test_with_photref(self):
        pairs = _pairs(
            _detrending_related_files(
                "/LC/src.h5", single_photref_dr_fname="/DR/sp.h5"
            )
        )
        self.assertEqual(
            pairs,
            [
                (FileKind.LIGHTCURVE, "/LC/src.h5"),
                (FileKind.DR_FILE, "/DR/sp.h5"),
            ],
        )

    def test_without_photref(self):
        pairs = _pairs(_detrending_related_files("/LC/src.h5"))
        self.assertEqual(pairs, [(FileKind.LIGHTCURVE, "/LC/src.h5")])


if __name__ == "__main__":
    unittest.main()
