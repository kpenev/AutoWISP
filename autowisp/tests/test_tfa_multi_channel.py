"""TFA on lightcurves collecting points from two channels of each frame."""

import contextlib
import io
import unittest

import numpy

from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.tests.synthetic_light_curve_test_case import (
    AllTemplatesTFA,
    SyntheticLightCurveTestCase,
)


class TestTFAMultiChannel(SyntheticLightCurveTestCase):
    """TFA with the default observation id on two-channel lightcurves.

    Both channels of a frame share its frame number, so only the channel
    tells their points apart. Two targets are corrected: an exact
    combination of the templates, and one of the templates itself, which
    TFA must fit without its own template. Template 1 and both targets lack
    one (frame, channel) point the other templates have.
    """

    @classmethod
    def setUpClass(cls):
        """Create the lightcurves and correct both targets."""

        super().setUpClass()

        rng = numpy.random.default_rng(1)
        templates = rng.normal(size=(4, cls._fnums.size))
        # Not exactly a combination of the others, or the templates would be
        # degenerate; fit by the others, it leaves a residual of about the
        # noise.
        templates[3] = numpy.dot([0.7, -0.4, 1.2], templates[:3]) + rng.normal(
            scale=0.01, size=cls._fnums.size
        )

        all_points = numpy.ones(cls._fnums.size, dtype=bool)
        lacking_one = numpy.copy(all_points)
        lacking_one[2 * 7 + 1] = False

        cls._write_light_curve(1, templates[0], lacking_one)
        for template_index in range(1, templates.shape[0]):
            cls._write_light_curve(
                template_index + 1, templates[template_index], all_points
            )
        cls._combination_id = 99
        cls._write_light_curve(
            cls._combination_id,
            numpy.dot([0.5, -1.0, 2.0, 0.3], templates),
            lacking_one,
        )
        cls._num_combination_points = int(lacking_one.sum())
        cls._template_target_id = 4

        configuration = cls._get_tfa_configuration()
        cls._fit_residual = {}
        cls._num_fit_points = {}
        # TFA reports the template data it verifies on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            correct = AllTemplatesTFA(
                cls._get_template_statistics(templates.shape[0]),
                configuration,
                verify_template_data=True,
                error_avg=configuration["detrend_error_avg"],
                rej_level=configuration["detrend_rej_level"],
                max_rej_iter=configuration["detrend_max_rej_iter"],
                reject_scale_floor=configuration["detrend_reject_scale_floor"],
                fit_identifier="TFA",
                mark_progress=lambda *_: None,
            )
            for target_id in [cls._combination_id, cls._template_target_id]:
                lc_fname = cls._get_lc_fname(target_id)
                correct(lc_fname)
                with LightCurveFile(lc_fname, "r") as light_curve:
                    cls._fit_residual[target_id] = light_curve.get_dataset(
                        "apphot.tfa.fit_residual", **cls._dataset[1]
                    ).item()
                    cls._num_fit_points[target_id] = light_curve.get_dataset(
                        "apphot.tfa.num_fit_points", **cls._dataset[1]
                    ).item()

    def test_combination_of_templates_is_fit_exactly(self):
        """An exact combination of templates leaves only storage rounding.

        LC magnitudes are stored to about 1e-5, which bounds the residual.
        """

        self.assertLess(self._fit_residual[self._combination_id], 1e-4)
        self.assertEqual(
            self._num_fit_points[self._combination_id],
            self._num_combination_points,
        )

    def test_template_is_fit_by_the_other_templates(self):
        """A target among the templates is fit by the rest, to its noise."""

        self.assertLess(self._fit_residual[self._template_target_id], 0.02)


if __name__ == "__main__":
    unittest.main()
