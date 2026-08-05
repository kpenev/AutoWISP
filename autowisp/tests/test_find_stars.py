"""Unit tests for the find_stars step."""

import unittest
from os import path
from tempfile import TemporaryDirectory

import numpy
from astropy.io import fits

from autowisp.exceptions import FindStarsError, NoSourcesFoundError
from autowisp.processing_steps.find_stars import fail_reasons, find_stars_single
from autowisp.tests.h5_test_case import DRTestCase


class TestFindStars(DRTestCase):
    """Tests of the find_stars step."""

    def test_find_stars(self):
        """Run the find_stars step and check the outputs against expected."""

        self.run_step_test(
            "find_stars",
            path.join("CAL", "object"),
            ["SourceExtraction/Version000/Sources", "Provenance", "FITSHeader"],
        )


class TestStarlessFrameHandling(unittest.TestCase):
    """A starless frame must fail on its own, not take the batch down.

    A frame that is clouded over or badly defocused yields no extracted
    sources. That is a property of the frame, not a pipeline fault, so
    ``find_stars_single`` records it as a failed frame and returns;
    every *other* find_stars failure still propagates and stops the run.
    """

    def setUp(self):
        """Record ``mark_start`` / ``mark_end`` calls in ``self.calls``."""

        self.calls = []

    def _mark_start(self, input_fname, **kwargs):
        """Stand-in for the manager's progress-start callback."""

        self.calls.append(("start", input_fname, kwargs))

    def _mark_end(self, input_fname, status=1, **kwargs):
        """Stand-in for the manager's progress-end callback."""

        self.calls.append(("end", input_fname, status, kwargs))

    def _run_with_extractor(self, tmp, extractor):
        """Run ``find_stars_single`` on a throwaway frame with ``extractor``."""

        frame = path.join(tmp, "frame.fits")
        fits.PrimaryHDU(numpy.zeros((4, 4))).writeto(frame)
        find_stars_single(frame, extractor, 0, self._mark_start, self._mark_end)
        return frame

    def test_no_sources_marks_frame_failed(self):
        """``NoSourcesFoundError`` is recorded as a failure, not re-raised."""

        def extract_nothing(fits_fname):
            """Stand in for a source finder that extracts nothing."""

            raise NoSourcesFoundError(f"no sources in {fits_fname!r}")

        with TemporaryDirectory() as tmp:
            frame = self._run_with_extractor(tmp, extract_nothing)

        self.assertEqual(
            self.calls,
            [
                ("start", frame, {}),
                ("end", frame, fail_reasons["no sources extracted"], {}),
            ],
        )
        self.assertLess(fail_reasons["no sources extracted"], 0)

    def test_other_failures_still_propagate(self):
        """Any other find_stars failure must still stop the run."""

        def extract_broken(fits_fname):
            """Stand in for a source finder that cannot run at all."""

            raise FindStarsError(f"cannot run the extractor on {fits_fname!r}")

        with TemporaryDirectory() as tmp:
            with self.assertRaises(FindStarsError):
                self._run_with_extractor(tmp, extract_broken)

        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
