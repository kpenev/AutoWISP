"""EPD and TFA fitting without the listed observations, but correcting them."""

import contextlib
import io
import unittest
from itertools import cycle
from os import path

import numpy

from autowisp.exceptions import ConfigurationError
from autowisp.light_curves.apply_correction import (
    recalculate_correction_statistics,
)
from autowisp.light_curves.correction import Correction
from autowisp.light_curves.epd_correction import EPDCorrection
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.tests.synthetic_light_curve_test_case import (
    AllTemplatesTFA,
    SyntheticLightCurveTestCase,
)


class TestDetrendingExclusions(SyntheticLightCurveTestCase):
    """EPD and TFA with an exclusion list on two-channel lightcurves.

    Each lightcurve follows an exact model, except at the bad observations:
    the first channel of the early frames, where it follows another one. So
    a fit that leaves exactly those out has no residual beyond storage
    rounding, flattens the rest, and leaves at the bad observations exactly
    what was added to them. The same lightcurves corrected without the list
    are the control: the same check must fail for them.

    The list also names the first channel of a frame the EPD filter does not
    select and the TFA target lacks, and an observation in no lightcurve.
    The second channel of every frame is never listed.

    Every other listed observation names the photometry fit, aperture 0, and
    the rest name none, which applies to every photometry. Some good
    observations are listed too, under other photometries only, which must
    not leave them out of the fit of aperture 0.

    Outlier rejection is off: it would otherwise discard some of the bad
    observations on its own, list or no list.
    """

    _num_frames = 40
    _rule = "(cloud[0] > 0.3) | (sky['G0'] > 100)"

    @classmethod
    def _read_result(cls, source_id, mode, variables, points_filter):
        """
        Return what a fit left in a source's lightcurve, and its statistics.

        Args:
            source_id(int):    The source whose lightcurve to read.

            mode(str):    Either ``'epd'`` or ``'tfa'``.

            variables(dict):    The variables ``points_filter`` uses.

            points_filter(str or None):    Selects the points the fit
                corrected. None if it corrected them all.

        Returns:
            dict:
                The fit's datasets by the tail of their pipeline key, and
                under ``'statistics'`` the performance of the fit.
        """

        with LightCurveFile(cls._get_lc_fname(source_id), "r") as light_curve:
            result = {
                quantity: light_curve.get_dataset(
                    f"apphot.{mode}.{quantity}", **cls._dataset[1]
                )
                for quantity in [
                    "magnitude",
                    "qc_excluded",
                    "fit_residual",
                    "num_fit_points",
                    "num_qc_excluded_points",
                    "cfg.exclusion_rule",
                ]
            }
        result["statistics"] = recalculate_correction_statistics(
            [cls._get_lc_fname(source_id)],
            fit_datasets=[cls._dataset + (f"apphot.{mode}.magnitude",)],
            variables=variables,
            lc_points_filter_expression=points_filter,
            calculate_average=numpy.nanmedian,
            calculate_scatter=numpy.nanmean,
            outlier_threshold=1e3,
            max_outlier_rejections=1,
        )[0]
        return result

    @classmethod
    def _run_epd(cls, exclusions):
        """Create the EPD lightcurve twice; correct one copy with the list."""

        rng = numpy.random.default_rng(1)
        zenith_distance = rng.uniform(10.0, 60.0, cls._fnums.size)
        cls._epd_extra = numpy.where(cls._bad, 0.3 + 0.01 * zenith_distance, 0)

        cls._epd_result = {}
        for source_id, corrector_exclusions in [(101, {}), (102, exclusions)]:
            cls._write_light_curve(
                source_id,
                0.02 * zenith_distance + cls._epd_extra,
                extra_datasets={
                    "skypos.zenith_distance": zenith_distance,
                    "skypos.hour_angle": numpy.where(
                        cls._outside_filter, 1.0, -1.0
                    ),
                },
            )
            # EPD reports its fit target on stdout.
            with contextlib.redirect_stdout(io.StringIO()):
                EPDCorrection(
                    used_variables={
                        "z": ("skypos.zenith_distance", {}),
                        "h": ("skypos.hour_angle", {}),
                    },
                    fit_points_filter_expression="h < 0",
                    fit_terms_expression="O1{z}",
                    fit_datasets=[cls._dataset + ("apphot.epd.magnitude",)],
                    error_avg="nanmedian",
                    rej_level=5.0,
                    max_rej_iter=0,
                    fit_identifier="EPD",
                    mark_progress=lambda *_: None,
                    observation_id=(
                        "fitsheader.fnum",
                        "fitsheader.cfg.clrchnl",
                    ),
                    **corrector_exclusions,
                )(cls._get_lc_fname(source_id))
            cls._epd_result[bool(corrector_exclusions)] = cls._read_result(
                source_id, "epd", {"h": ("skypos.hour_angle", {})}, "h < 0"
            )

    @classmethod
    def _run_tfa(cls, exclusions):
        """Create the templates and two targets; correct one with the list."""

        rng = numpy.random.default_rng(2)
        weights = numpy.array([0.5, -1.0, 2.0, 0.3])
        templates = rng.normal(size=(weights.size, cls._fnums.size))
        template_extra = numpy.where(
            cls._listed, rng.normal(size=templates.shape), 0.0
        )
        for template_index, template in enumerate(templates + template_extra):
            cls._write_light_curve(template_index + 1, template)

        cls._tfa_points = numpy.logical_not(cls._outside_filter & cls._listed)
        # Correcting subtracts the templates as they are, bad parts included.
        cls._tfa_extra = numpy.where(cls._listed, 0.3, 0.0) - numpy.dot(
            weights, template_extra
        )

        cls._tfa_result = {}
        for source_id, corrector_exclusions in [(201, {}), (202, exclusions)]:
            cls._write_light_curve(
                source_id,
                numpy.dot(weights, templates)
                + numpy.where(cls._listed, 0.3, 0),
                cls._tfa_points,
            )
            configuration = cls._get_tfa_configuration(**corrector_exclusions)
            # TFA reports the template data it verifies on stdout.
            with contextlib.redirect_stdout(io.StringIO()):
                AllTemplatesTFA(
                    cls._get_template_statistics(weights.size),
                    configuration,
                    verify_template_data=True,
                    error_avg=configuration["detrend_error_avg"],
                    rej_level=configuration["detrend_rej_level"],
                    max_rej_iter=0,
                    fit_identifier="TFA",
                    mark_progress=lambda *_: None,
                )(cls._get_lc_fname(source_id))
            # No filter, as for the fit: TFA corrected every point.
            cls._tfa_result[bool(corrector_exclusions)] = cls._read_result(
                source_id, "tfa", {}, None
            )

    @classmethod
    def setUpClass(cls):
        """Create the lightcurves and correct them with and without a list."""

        super().setUpClass()

        frame_indices = cls._fnums - cls._fnums[0]
        cls._outside_filter = frame_indices >= 30
        cls._bad = (frame_indices < 15) & (cls._channel_indices == 0)
        cls._listed = cls._bad | (
            (frame_indices == 35) & (cls._channel_indices == 0)
        )

        exclude_fname = path.join(cls._project_home, "exclude.txt")
        other_photometries = (
            (frame_indices >= 20)
            & (frame_indices < 25)
            & (cls._channel_indices == 0)
        )
        with open(exclude_fname, "w", encoding="utf-8") as exclude_file:
            for index, fnum in enumerate(cls._fnums[cls._listed]):
                exclude_file.write(f"{fnum} G0{' ap0' * (index % 2)}\n")
            for fnum, photometry in zip(
                cls._fnums[other_photometries],
                cycle(("ap1", "shapefit")),
            ):
                exclude_file.write(f"{fnum} G0 {photometry}\n")
            exclude_file.write("\n9999 G0\n")
        exclusions = {
            "qc_exclude_file": exclude_fname,
            "exclusion_rule": cls._rule,
        }

        cls._run_epd(exclusions)
        cls._run_tfa(exclusions)

    def _assert_corrected_to_model(self, result, corrected, bad, extra):
        """
        Check that the fit found the model the points that are not bad follow.

        Args:
            result(dict):    What the fit left in the lightcurve.

            corrected(bool array):    The points the fit corrects.

            bad(bool array):    The points that do not follow the model.

            extra(array):    What was added to the model at each point.
        """

        # LC magnitudes are stored to about 1e-5.
        self.assertLess(result["fit_residual"].item(), 1e-4)
        level = numpy.median(result["magnitude"][corrected & ~bad])
        numpy.testing.assert_allclose(
            result["magnitude"][corrected], level + extra[corrected], atol=1e-4
        )
        # Flat only without the bad points, which are corrected to the extra.
        self.assertLess(result["statistics"]["rms"][0], 1e-4)

    def _assert_deviates_from_model(self, result, corrected, bad, extra):
        """
        Check that the fit missed the model the points that are not bad follow.

        Every way _assert_corrected_to_model() checks for the model must
        show the miss, each on its own.

        Args:
            See _assert_corrected_to_model().
        """

        self.assertGreater(result["fit_residual"].item(), 0.01)
        level = numpy.median(result["magnitude"][corrected & ~bad])
        self.assertGreater(
            numpy.abs(
                result["magnitude"][corrected] - level - extra[corrected]
            ).max(),
            0.01,
        )
        self.assertGreater(result["statistics"]["rms"][0], 0.01)

    def _assert_recorded(self, result, corrected, excluded, rule):
        """
        Check what the fit recorded about the points it left out.

        Args:
            result(dict):    What the fit left in the lightcurve.

            corrected(bool array):    The points the fit corrects.

            excluded(bool array):    The points expected to be left out.

            rule(str):    The rule expected to be recorded.
        """

        self.assertEqual(
            result["num_fit_points"].item(), (corrected & ~excluded).sum()
        )
        self.assertEqual(
            result["statistics"]["num_finite"][0],
            (corrected & ~excluded).sum(),
        )
        numpy.testing.assert_array_equal(result["qc_excluded"], excluded)
        self.assertEqual(
            result["num_qc_excluded_points"].item(), excluded.sum()
        )
        self.assertEqual(result["cfg.exclusion_rule"][0], rule.encode("utf-8"))

    def test_epd_ignores_excluded(self):
        """EPD fits without the listed points its filter selects."""

        result = self._epd_result[True]
        corrected = numpy.logical_not(self._outside_filter)
        self._assert_corrected_to_model(
            result, corrected, self._bad, self._epd_extra
        )
        self._assert_recorded(result, corrected, self._bad, self._rule)
        self.assertFalse(numpy.isfinite(result["magnitude"][~corrected]).any())

    def test_epd_without_list(self):
        """Without a list EPD fits the bad points too, and misses the model."""

        result = self._epd_result[False]
        corrected = numpy.logical_not(self._outside_filter)
        self._assert_deviates_from_model(
            result, corrected, self._bad, self._epd_extra
        )
        self._assert_recorded(
            result, corrected, numpy.zeros(corrected.shape, dtype=bool), ""
        )

    def test_tfa_ignores_excluded(self):
        """TFA fits without the listed observations the target has."""

        result = self._tfa_result[True]
        corrected = numpy.ones(self._tfa_points.sum(), dtype=bool)
        listed = self._listed[self._tfa_points]
        self._assert_corrected_to_model(
            result, corrected, listed, self._tfa_extra[self._tfa_points]
        )
        self._assert_recorded(result, corrected, listed, self._rule)

    def test_tfa_without_list(self):
        """Without a list TFA fits the bad points too, and misses the model."""

        result = self._tfa_result[False]
        corrected = numpy.ones(self._tfa_points.sum(), dtype=bool)
        self._assert_deviates_from_model(
            result,
            corrected,
            self._listed[self._tfa_points],
            self._tfa_extra[self._tfa_points],
        )
        self._assert_recorded(
            result, corrected, numpy.zeros(corrected.shape, dtype=bool), ""
        )

    def test_a_list_naming_no_photometry_is_refused(self):
        """A last value that is not a photometry, or a value too many."""

        exclude_fname = path.join(self._project_home, "bad_exclude.txt")
        for line in ("101 G0 ap", "101 G0 G1", "101 G0 ap0 ap1"):
            with self.subTest(line=line):
                with open(exclude_fname, "w", encoding="utf-8") as bad_list:
                    bad_list.write(line + "\n")
                with self.assertRaises(ConfigurationError):
                    # pylint: disable-next=protected-access
                    Correction._read_exclusions(exclude_fname, 2)


if __name__ == "__main__":
    unittest.main()
