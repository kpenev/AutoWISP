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


def _make_test_frame(fits_path):
    """Write a 3-HDU FITS (image, placeholder, saturation mask) with 5 sources.

    Three HDUs because ``SourceFinder`` reads the image from HDU 0 and the
    saturation mask from HDU 2. The five bright Gaussians sit well above the
    0.999 image quantile so fistar has clear sources to extract.
    """

    rng = numpy.random.default_rng(0)
    image = rng.normal(100.0, 5.0, (200, 200))
    grid_y, grid_x = numpy.mgrid[0:200, 0:200]
    for x0, y0 in [(50, 50), (120, 80), (160, 150), (30, 170), (100, 100)]:
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


if __name__ == "__main__":
    unittest.main()
