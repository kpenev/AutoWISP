"""Exercise the quantile-based brightness-threshold path in ``SourceFinder``.

When ``brightness_threshold`` is left unset, ``SourceFinder`` derives it from an
image quantile (``brightness_quantile_scale * numpy.quantile(...)``), producing
a ``numpy.float64``. Under numpy 2, ``repr(numpy.float64(x))`` is
``'np.float64(x)'`` -- which fistar rejects with "invalid command line
argument" -- so ``start_fistar`` coerces to a plain ``float`` before ``repr``.

The find_stars integration test only ever uses a fixed (Python-float) threshold
from ``test.cfg``, so this quantile path -- where the ``numpy.float64``
originates and reaches the command line -- was never exercised. This drives it
end to end with the real fistar (an astrowisp dependency) and checks that the
extraction actually succeeds.
"""

import os
import unittest
from tempfile import TemporaryDirectory

import numpy
from astropy.io import fits

from autowisp.source_finder import SourceFinder
from autowisp.exceptions import NoSourcesFoundError


def _make_test_frame(fits_path, source_positions=None):
    """Write a 3-HDU FITS (image, placeholder, saturation mask) with 5 sources.

    Three HDUs because ``SourceFinder`` reads the image from HDU 0 and the
    saturation mask from HDU 2. The five bright Gaussians sit well above the
    0.999 image quantile so fistar has clear sources to extract.

    Args:
        fits_path(str):    Where to write the frame.

        source_positions(list or None):    ``(x, y)`` of each source to
            inject; the default five are used if not given. Pass an empty
            list for a starless frame.
    """

    if source_positions is None:
        source_positions = [
            (50, 50),
            (120, 80),
            (160, 150),
            (30, 170),
            (100, 100),
        ]
    rng = numpy.random.default_rng(0)
    image = rng.normal(100.0, 5.0, (200, 200))
    grid_y, grid_x = numpy.mgrid[0:200, 0:200]
    for x0, y0 in source_positions:
        image += 5000.0 * numpy.exp(
            -((grid_x - x0) ** 2 + (grid_y - y0) ** 2) / (2 * 2.0**2)
        )
    saturation = numpy.zeros((200, 200), dtype=numpy.int32)
    saturation[0, 0] = saturation[-1, -1] = 1  # a few, away from the sources
    fits.HDUList(
        [
            fits.PrimaryHDU(data=image),
            fits.ImageHDU(data=numpy.zeros_like(image)),
            fits.ImageHDU(data=saturation),
        ]
    ).writeto(fits_path)


class TestQuantileBrightnessThreshold(unittest.TestCase):
    """The image-quantile threshold must drive a successful fistar run."""

    def test_quantile_threshold_extraction_succeeds(self):
        """Derive the threshold from the image quantile and run fistar for real.

        With ``brightness_threshold`` unset, ``SourceFinder`` computes it as a
        ``numpy.float64`` from the image and hands it to fistar. A malformed
        value (the pre-fix numpy repr) makes fistar exit with "invalid command
        line argument" and extract nothing, so a non-empty result confirms the
        threshold reached the CLI in a parseable form.
        """

        with TemporaryDirectory() as tmp:
            frame = os.path.join(tmp, "frame.fits")
            _make_test_frame(frame)
            finder = SourceFinder(
                tool="fistar",
                brightness_threshold=0,  # unset -> derive from the quantile
                brightness_quantile=0.999,
                brightness_quantile_scale=1.0,
            )
            sources = finder(frame)

        self.assertIsNotNone(sources)
        # The five injected sources are well above the threshold; the exact
        # count can vary with fistar, so just require a successful extraction.
        self.assertGreater(len(sources), 0)


class TestExtremeSourceCounts(unittest.TestCase):
    """Frames with one source or none must not crash the extractor.

    ``numpy.genfromtxt`` is what makes these two special: a single
    extracted source comes back as a **0-d** array (indexing it with the
    finite-value mask raised ``IndexError``), and no sources at all come
    back as a 1-d array whose *dtype has no field names* (``sort(order=
    "flux")`` then raised ``ValueError: unknown field name``).
    """

    def test_single_source_extraction_succeeds(self):
        """One source must come back as a normal 1-element source list."""

        with TemporaryDirectory() as tmp:
            frame = os.path.join(tmp, "one_source.fits")
            _make_test_frame(frame, source_positions=[(100, 100)])
            sources = SourceFinder(tool="fistar", brightness_threshold=1000.0)(
                frame
            )

        self.assertEqual(sources.ndim, 1)
        self.assertEqual(len(sources), 1)
        self.assertIn("flux", sources.dtype.names)

    def test_no_sources_raises_naming_the_frame(self):
        """A starless frame raises ``NoSourcesFoundError`` naming the file.

        The threshold is set far above anything in the (source-free)
        image so fistar reliably extracts nothing.
        """

        with TemporaryDirectory() as tmp:
            frame = os.path.join(tmp, "starless.fits")
            _make_test_frame(frame, source_positions=[])
            finder = SourceFinder(tool="fistar", brightness_threshold=1.0e6)
            with self.assertRaises(NoSourcesFoundError) as caught:
                finder(frame)

        # ``repr`` because the message interpolates the name with ``!r``,
        # which on Windows doubles the backslashes of the raw path.
        self.assertIn(repr(frame), str(caught.exception))
        self.assertEqual(caught.exception.details["image"], frame)
        self.assertEqual(
            caught.exception.details["brightness_threshold"], 1.0e6
        )


if __name__ == "__main__":
    unittest.main()
