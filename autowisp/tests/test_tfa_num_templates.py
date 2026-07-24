"""Unit tests for variable TFA template counts and selection diagnostics.

These are deliberately self-contained (no downloaded test-data bundle): the
bundle cannot exercise a *varying* template count because every test channel
happens to select the same number, so the ragged-storage regression and the
short-count warning are covered here instead.
"""

import logging
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import h5py
import numpy

from autowisp.database.interface import set_project_home
from autowisp.light_curves.hashable_array import HashableArray
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.light_curves.tfa_correction import TFACorrection

#: The two config datasets whose per-config value is a variable-length array of
#: template star IDs.
_TEMPLATE_KEYS = (
    "shapefit.tfa.cfg.template_source_ids",
    "apphot.tfa.cfg.template_source_ids",
)


class TestTemplateSourceIdsVlen(unittest.TestCase):
    """Ragged storage of the TFA ``TemplateStarIDs`` config datasets."""

    @classmethod
    def setUpClass(cls):
        """Point the DB/project home at a throwaway directory."""

        cls._project_home = mkdtemp(prefix="autowisp_vlen_")
        set_project_home(cls._project_home)

    @classmethod
    def tearDownClass(cls):
        """Drop the throwaway project home."""

        rmtree(cls._project_home, ignore_errors=True)

    def test_dtype_override_is_vlen(self):
        """``get_dtype`` reports a vlen dtype (of the DB base) for the keys."""

        with LightCurveFile() as light_curve:
            for key in _TEMPLATE_KEYS:
                dtype = light_curve.get_dtype(key)
                self.assertEqual(
                    dtype.kind,
                    "O",
                    f"{key} should be stored as a variable-length dtype",
                )
                # The element type stays whatever the database declared
                # (uint64), only wrapped as vlen.
                self.assertEqual(
                    numpy.dtype(h5py.check_dtype(vlen=dtype)),
                    numpy.dtype(numpy.uint64),
                )

    def test_varying_template_counts_round_trip(self):
        """Appending configs of *different* template counts must not broadcast.

        Reproduces the production crash (``Can't broadcast (1, 12) ->
        (1, 10)``): one TFA run stores a width-10 row, a later run for the same
        source appends a width-12 row. With fixed-2D storage the append failed;
        as ragged rows it succeeds and round-trips.
        """

        key = "shapefit.tfa.cfg.template_source_ids"
        num_frames = 3
        template_id_sets = (
            numpy.arange(10, dtype=numpy.uint64),
            numpy.arange(100, 112, dtype=numpy.uint64),  # length 12
            numpy.arange(200, 208, dtype=numpy.uint64),  # length 8
        )

        with LightCurveFile() as light_curve:
            # Give the LC a defined length so the config-append path (rather
            # than fresh creation) is exercised -- that is where the broadcast
            # crash lived.
            light_curve.add_dataset(
                "skypos.BJD",
                numpy.arange(num_frames, dtype=float),
                unlimited=True,
            )
            light_curve.confirm_lc_length()

            for template_ids in template_id_sets:
                light_curve.add_configurations(
                    component="shapefit.tfa",
                    configurations=(
                        ((key, HashableArray(template_ids)),),
                    ),
                    config_indices=numpy.zeros(num_frames, dtype=numpy.uint),
                )

            dataset = light_curve[light_curve.get_element_path(key)]
            self.assertEqual(dataset.dtype.kind, "O")
            self.assertEqual(dataset.shape, (len(template_id_sets),))
            for index, expected in enumerate(template_id_sets):
                numpy.testing.assert_array_equal(dataset[index], expected)


class _TemplateSelectionHarness(TFACorrection):
    """A ``TFACorrection`` exposing just the template-selection step.

    The full ``TFACorrection.__init__`` reads per-light-curve template
    measurements and observation IDs (``_prepare_template_data``), which needs
    real data. ``_select_template_stars`` only consults the configuration and
    the (passed-in) EPD statistics, so this harness sets the configuration and
    skips the data-reading construction.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, configuration):
        self._configuration = configuration


class TestTemplateSelectionDiagnostics(unittest.TestCase):
    """The short-count warning emitted by ``_select_template_stars``."""

    _logger_name = "autowisp.light_curves.tfa_correction"

    @staticmethod
    def _make_epd_statistics(num_stars):
        """A single-photometry structured array that survives every cut.

        All stars are unsaturated, bright, long-LC, and follow a clean
        magnitude-RMS relation (small scatter so the typical-RMS fit keeps
        them), spread over a 2-D field.
        """

        dtype = numpy.dtype(
            [
                ("mag", "f8"),
                ("num_finite", "i8", (1,)),
                ("rms", "f8", (1,)),
                ("xi", "f8"),
                ("eta", "f8"),
            ]
        )
        epd_statistics = numpy.zeros(num_stars, dtype=dtype)
        magnitude = numpy.linspace(10.0, 11.0, num_stars)
        epd_statistics["mag"] = magnitude
        epd_statistics["num_finite"][:, 0] = 100
        scatter = numpy.random.default_rng(0).normal(0.0, 0.02, num_stars)
        epd_statistics["rms"][:, 0] = 10.0 ** (
            -2.0 - 0.1 * (magnitude - 10.0) + scatter
        )
        positions = numpy.linspace(0.0, 1.0, num_stars)
        epd_statistics["xi"] = positions
        epd_statistics["eta"] = positions[::-1]
        return epd_statistics

    @staticmethod
    def _configuration(sqrt_num_templates):
        """Config with cuts loose enough that every synthetic star passes."""

        return {
            "saturation_magnitude": 8.0,
            "faint_mag_limit": 12.0,
            "min_observations_quantile": 0.5,
            "min_observations_fraction": 0.5,
            "mag_rms_dependence_order": 1,
            "mag_rms_outlier_threshold": 5.0,
            "mag_rms_max_rej_iter": 3,
            "max_rms": 0.05,
            "allow_saturated_templates": False,
            "sqrt_num_templates": sqrt_num_templates,
        }

    def test_shortfall_emits_warning(self):
        """Fewer selected than ``sqrt_num_templates**2`` logs a warning.

        Ten eligible stars against a 4x4 (=16) grid: the grid-nearest-neighbor
        collapse yields fewer than 16 distinct templates, which must warn and
        report the per-stage funnel.
        """

        num_stars = 10
        sqrt_num_templates = 4
        grid_size = sqrt_num_templates**2
        selector = _TemplateSelectionHarness(
            self._configuration(sqrt_num_templates)
        )

        with self.assertLogs(self._logger_name, logging.WARNING) as captured:
            selected = selector._select_template_stars(  # pylint: disable=protected-access
                self._make_epd_statistics(num_stars)
            )

        self.assertEqual(len(selected), 1)
        self.assertLess(len(selected[0]), grid_size)

        warnings = [
            record.getMessage()
            for record in captured.records
            if record.levelno == logging.WARNING
        ]
        self.assertEqual(len(warnings), 1)
        message = warnings[0]
        self.assertIn(f"of the expected {grid_size}", message)
        self.assertIn("channel 0", message)
        # The funnel counts are reported so the shortfall is explainable.
        self.assertIn(f"{num_stars} stars total", message)

    def test_full_count_no_warning(self):
        """Enough distinct templates for the grid must not warn."""

        # A 1x1 grid needs a single template; ten eligible stars easily clear
        # it, so no shortfall warning should fire.
        selector = _TemplateSelectionHarness(
            self._configuration(sqrt_num_templates=1)
        )

        with self.assertNoLogs(self._logger_name, logging.WARNING):
            selected = selector._select_template_stars(  # pylint: disable=protected-access
                self._make_epd_statistics(10)
            )
        self.assertEqual(len(selected[0]), 1)


if __name__ == "__main__":
    unittest.main()
