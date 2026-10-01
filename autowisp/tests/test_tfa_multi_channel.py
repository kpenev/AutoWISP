"""TFA on lightcurves collecting points from two channels of each frame.

Self-contained (no downloaded test-data bundle): the bundle has a single
channel, so it cannot show that the default observation id keeps the
channels of one frame apart.
"""

import contextlib
import io
import unittest
from os import path
from shutil import rmtree
from tempfile import mkdtemp

import numpy

from autowisp.database.interface import set_project_home
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp.light_curves.tfa_correction import TFACorrection
from autowisp.processing_steps import tfa


class _AllTemplatesTFA(TFACorrection):
    """A ``TFACorrection`` using every star in the statistics as template.

    Template selection has its own tests; here it would only obscure which
    stars the fit uses.
    """

    def _select_template_stars(self, epd_statistics):
        return [numpy.arange(epd_statistics.size)]


class TestTFAMultiChannel(unittest.TestCase):
    """TFA with the default observation id on two-channel lightcurves.

    Both channels of a frame share its frame number, so only the channel
    tells their points apart. Two targets are corrected: an exact
    combination of the templates, and one of the templates itself, which
    TFA must fit without its own template. Template 1 and both targets lack
    one (frame, channel) point the other templates have.
    """

    _dataset = ("apphot.magfit.magnitude", {"aperture_index": 0})

    @classmethod
    def _write_light_curve(cls, source_id, magnitudes, points):
        """Create the lightcurve of one source with the given points."""

        with LightCurveFile(
            path.join(cls._project_home, f"{source_id}.h5"),
            "a",
            source_ids={"Gaia DR3": str(source_id)},
        ) as light_curve:
            light_curve.extend_dataset("fitsheader.fnum", cls._fnums[points])
            light_curve.extend_dataset(
                cls._dataset[0], 10.0 + magnitudes[points], **cls._dataset[1]
            )
            light_curve.add_configurations(
                "fitsheader",
                (
                    (("fitsheader.cfg.clrchnl", b"G0"),),
                    (("fitsheader.cfg.clrchnl", b"G1"),),
                ),
                cls._channel_indices[points],
            )
            light_curve.confirm_lc_length()

    @classmethod
    def _get_configuration(cls):
        """Return the step's default configuration, as ``tfa()`` adapts it."""

        # The parser reports its defaults on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            configuration = tfa.parse_command_line([])
        for param in list(configuration.keys()):
            if param.startswith("tfa_"):
                configuration[param[4:]] = configuration.pop(param)
        configuration.update(
            lc_fname=path.join(cls._project_home, "{:d}.h5"),
            fit_datasets=[cls._dataset + ("apphot.tfa.magnitude",)],
            fit_points_filter_expression=None,
            variables={},
        )
        return configuration

    @classmethod
    def setUpClass(cls):
        """Create the lightcurves and correct both targets."""

        cls._project_home = mkdtemp(prefix="autowisp_tfa_channels_")
        set_project_home(cls._project_home)

        num_frames = 30
        cls._fnums = numpy.repeat(numpy.arange(100, 100 + num_frames), 2)
        cls._channel_indices = numpy.tile([0, 1], num_frames)

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

        epd_statistics = numpy.zeros(
            templates.shape[0],
            dtype=[
                ("ID", numpy.uint64),
                ("rms", numpy.float64, (1,)),
                ("num_finite", numpy.uint64, (1,)),
            ],
        )
        epd_statistics["ID"] = numpy.arange(1, templates.shape[0] + 1)

        configuration = cls._get_configuration()
        cls._fit_residual = {}
        cls._num_fit_points = {}
        # TFA reports the template data it verifies on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            correct = _AllTemplatesTFA(
                epd_statistics,
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
                lc_fname = configuration["lc_fname"].format(target_id)
                correct(lc_fname)
                with LightCurveFile(lc_fname, "r") as light_curve:
                    cls._fit_residual[target_id] = light_curve.get_dataset(
                        "apphot.tfa.fit_residual", **cls._dataset[1]
                    ).item()
                    cls._num_fit_points[target_id] = light_curve.get_dataset(
                        "apphot.tfa.num_fit_points", **cls._dataset[1]
                    ).item()

    @classmethod
    def tearDownClass(cls):
        """Drop the throwaway project home."""

        rmtree(cls._project_home, ignore_errors=True)

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
