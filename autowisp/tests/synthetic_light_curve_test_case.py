"""Define a base class for tests on lightcurves built to a known answer."""

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


class AllTemplatesTFA(TFACorrection):
    """A ``TFACorrection`` using every star in the statistics as template.

    Template selection has its own tests; here it would only obscure which
    stars the fit uses.
    """

    def _select_template_stars(self, epd_statistics):
        return [numpy.arange(epd_statistics.size)]


class SyntheticLightCurveTestCase(unittest.TestCase):
    """Tests on two-channel lightcurves created in a throwaway project.

    Self-contained (no downloaded test-data bundle): the bundle has a single
    channel, and its lightcurves follow no model a result could be checked
    against. Here both channels of a frame share its frame number, so only
    the channel tells their points apart.
    """

    _dataset = ("apphot.magfit.magnitude", {"aperture_index": 0})
    _num_frames = 30

    @classmethod
    def setUpClass(cls):
        """Create the project and the frames the lightcurves sample."""

        cls._project_home = mkdtemp(prefix="autowisp_synthetic_lc_")
        set_project_home(cls._project_home)
        cls._fnums = numpy.repeat(numpy.arange(100, 100 + cls._num_frames), 2)
        cls._channel_indices = numpy.tile([0, 1], cls._num_frames)

    @classmethod
    def tearDownClass(cls):
        """Drop the throwaway project home."""

        rmtree(cls._project_home, ignore_errors=True)

    @classmethod
    def _get_lc_fname(cls, source_id):
        """Return the filename of the lightcurve of the given source."""

        return path.join(cls._project_home, f"{source_id}.h5")

    @classmethod
    def _write_light_curve(
        cls, source_id, magnitudes, points=slice(None), extra_datasets=None
    ):
        """
        Create the lightcurve of one source with the given points.

        Args:
            source_id(int):    Identifies the source and names its lightcurve.

            magnitudes(array):    The brightness of the source at every
                channel of every frame, relative to a constant.

            points(slice or bool array):    The points the lightcurve has.

            extra_datasets(dict or None):    Further datasets to create,
                with their values at every channel of every frame indexed by
                pipeline key.
        """

        with LightCurveFile(
            cls._get_lc_fname(source_id),
            "a",
            source_ids={"Gaia DR3": str(source_id)},
        ) as light_curve:
            light_curve.extend_dataset("fitsheader.fnum", cls._fnums[points])
            for dataset_key, values in (extra_datasets or {}).items():
                light_curve.extend_dataset(dataset_key, values[points])
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
    def _get_tfa_configuration(cls, **overrides):
        """Return the step's default configuration, as ``tfa()`` adapts it."""

        # The parser reports its defaults on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            configuration = tfa.parse_command_line([])
        for param in list(configuration.keys()):
            if param.startswith("tfa_"):
                configuration[param[4:]] = configuration.pop(param)
        configuration.update(
            # Set by the engine, or from the command line.
            project_home=cls._project_home,
            # Not "{:d}.h5": Windows takes "{:" for a drive, and joining a
            # path with a drive of its own discards the directory.
            lc_fname=path.join(cls._project_home, "{0:d}.h5"),
            fit_datasets=[cls._dataset + ("apphot.tfa.magnitude",)],
            fit_points_filter_expression=None,
            variables={},
            **overrides,
        )
        return configuration

    @staticmethod
    def _get_template_statistics(num_templates):
        """Return EPD statistics listing sources 1 to ``num_templates``."""

        result = numpy.zeros(
            num_templates,
            dtype=[
                ("ID", numpy.uint64),
                ("rms", numpy.float64, (1,)),
                ("num_finite", numpy.uint64, (1,)),
            ],
        )
        result["ID"] = numpy.arange(1, num_templates + 1)
        return result
