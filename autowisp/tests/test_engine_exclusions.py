"""Tests for the engine turning exclusion rules into the steps' lists.

Before ``fit_magnitudes`` the engine lists the DR files of a batch the
``magfit-exclusion-rule`` leaves out, and before ``epd`` and ``tfa`` the
observations of a single photometric reference the step's rule leaves out,
by the values of the ``tfa-observation-id`` datasets. What a rule decides is
tested in ``test_exclusion_rules``; these check what the engine asks it and
what it makes of the answer.

The project is the one of ``test_photref_binding``, with a background that
differs between the channels, and a DR file for every observation, whose
header the engine reads the observation ids from.
"""

import logging
import unittest

from astropy.io import fits
from sqlalchemy import delete, select

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.lightcurve_processing import (
    LightCurveProcessingManager,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageMasterSelection,
    MasterType,
    PhotometryDiagnostics,
)

# pylint: enable=no-name-in-module
from autowisp.exceptions import ConfigurationError
from autowisp.tests.test_photref_binding import PhotrefBindingProject


class TestEngineExclusions(PhotrefBindingProject):
    """The lists the engine hands ``fit_magnitudes``, ``epd`` and ``tfa``."""

    #: ``bg_center`` per image and channel; ``longer`` records none in G.
    _backgrounds = {
        ("near", "R"): 10.0,
        ("near", "G"): 50.0,
        ("far", "R"): 50.0,
        ("far", "G"): 10.0,
        ("longer", "R"): 10.0,
    }

    #: ``photometry_mag_offset`` per image and channel, in apertures 0 and 1.
    _offsets = {
        ("near", "R"): (10.0, 50.0),
        ("near", "G"): (50.0, 10.0),
        ("far", "R"): (50.0, 50.0),
        ("longer", "R"): (10.0, 10.0),
    }

    #: The ``FNUM`` of each image, in the header of its DR files.
    _fnums = {"near": 101, "far": 102, "longer": 103}

    #: The observation id of the default configuration.
    _observation_id = ("fitsheader.fnum", "fitsheader.cfg.clrchnl")

    @classmethod
    def _fill_database(cls):
        """Add the diagnostics, and a DR file for every observation."""

        super()._fill_database()
        with start_db_session() as db_session:
            for (name, channel), value in cls._backgrounds.items():
                cls._add_diagnostics(
                    db_session,
                    cls._image_ids[name],
                    channel,
                    {"bg_center": value},
                )
            offset_id = db_session.scalar(
                select(DiagnosticType.id).filter_by(
                    name="photometry_mag_offset"
                )
            )
            db_session.add_all(
                # pylint: disable=not-callable
                PhotometryDiagnostics(
                    image_id=cls._image_ids[name],
                    channel=channel,
                    photometry_id=photometry,
                    diagnostic_id=offset_id,
                    value=value,
                )
                # pylint: enable=not-callable
                for (name, channel), values in cls._offsets.items()
                for photometry, value in enumerate(values)
            )

        # A manager without a pipeline run switches logging off entirely.
        logging_disabled = logging.root.manager.disable
        processing = ImageProcessingManager(pipeline_run_id=None)
        logging.disable(logging_disabled)

        cls._dr_fnames = {}
        with start_db_session() as db_session:
            for name, image_id in cls._image_ids.items():
                processing.evaluate_expressions_image(
                    db_session.get(Image, image_id), db_session
                )
                for channel in ("R", "G"):
                    cls._dr_fnames[name, channel] = (
                        processing.get_product_fname(image_id, channel, "dr")
                    )
                    with DataReductionFile(
                        cls._dr_fnames[name, channel], "w"
                    ) as dr_file:
                        dr_file.initialize(
                            fits.Header(
                                {
                                    "RAWFNAME": name,
                                    "CLRCHNL": channel,
                                    "FNUM": cls._fnums[name],
                                }
                            )
                        )

    def magfit_excluded(self, rule, members, *, master=None, step=None):
        """
        Return what the engine lists for ``fit_magnitudes`` of a batch.

        Args:
            rule(str or None):    The ``magfit-exclusion-rule``.

            members:    ``(image name, channel)`` of the batch.

            master(str or None):    The existing master photometric
                reference the batch is fit against, if any.

            step(str or None):    The step to ask for, by default
                ``fit_magnitudes``.
        """

        # pylint: disable=protected-access
        with start_db_session() as db_session:
            for name, channel in members:
                image = db_session.get(Image, self._image_ids[name])
                self._processing.evaluate_expressions_image(image, db_session)
                self._processing._init_processed_ids(image, [channel], "dr")

        return self._processing._get_magfit_exclusions(
            step or "fit_magnitudes",
            {"magfit_exclusion_rule": rule, "master_photref_fname": master},
            [self._dr_fnames[member] for member in members],
        )
        # pylint: enable=protected-access

    def detrending_excluded(self, step, configuration, bound):
        """
        Return what the engine lists for *step* of the single photref.

        Args:
            step(str):    The lightcurve step to ask for.

            configuration(dict):    Its configuration.

            bound:    ``(image name, channel)`` bound to the single photref
                while the engine is asked, and no longer.
        """

        with start_db_session() as db_session:
            photref_type_id = db_session.scalar(
                select(MasterType.id).filter_by(name="single_photref")
            )
            db_session.add_all(
                # pylint: disable=not-callable
                ImageMasterSelection(
                    image_id=self._image_ids[name],
                    channel=channel,
                    master_type_id=photref_type_id,
                    master_file_id=self._photref_id,
                )
                # pylint: enable=not-callable
                for name, channel in bound
            )

        try:
            logging_disabled = logging.root.manager.disable
            processing = LightCurveProcessingManager(pipeline_run_id=None)
            logging.disable(logging_disabled)

            # pylint: disable-next=protected-access
            return processing._get_detrending_exclusions(
                step, configuration, self._photref_fname
            )
        finally:
            with start_db_session() as db_session:
                db_session.execute(delete(ImageMasterSelection))

    def dr(self, *members):
        """Return the DR files of the given ``(image name, channel)``."""

        return sorted(self._dr_fnames[member] for member in members)

    def test_magfit_lists_the_dr_files_a_slot_excludes(self):
        """Each channel is decided on its own background.

        Against an existing master too: nothing is built from the batch then,
        but its DR files record the verdict.
        """

        batch = [("near", "R"), ("near", "G"), ("far", "R"), ("far", "G")]

        for master in (None, "/masters/mphotref_R.fits"):
            with self.subTest(master=master):
                self.assertEqual(
                    self.magfit_excluded(
                        "bg_center[0] > 30",
                        batch + [("longer", "R")],
                        master=master,
                    ),
                    self.dr(("near", "G"), ("far", "R")),
                )

    def test_magfit_lists_every_channel_of_an_image_excluded(self):
        """A rule quoting a channel decides for every channel in the batch."""

        self.assertEqual(
            self.magfit_excluded(
                "bg_center['G'] > 30",
                [("near", "R"), ("near", "G"), ("far", "R"), ("longer", "R")],
            ),
            self.dr(("near", "R"), ("near", "G")),
        )

    def test_magfit_without_a_rule_to_apply_lists_nothing(self):
        """Unset, or for another step."""

        batch = [("near", "R"), ("far", "R")]
        for rule, kwargs in (
            (None, {}),
            ("", {}),
            ("bg_center[0] > 30", {"step": "fit_star_shape"}),
        ):
            with self.subTest(rule=rule, **kwargs):
                self.assertIsNone(self.magfit_excluded(rule, batch, **kwargs))

    def test_magfit_cannot_read_its_own_diagnostics(self):
        """The engine says it is deciding before magnitude fitting."""

        with self.assertRaises(ConfigurationError):
            self.magfit_excluded(
                "photometry_mag_offset[0][0] > 1", [("near", "R")]
            )

    def test_detrending_lists_observation_ids(self):
        """Of the observations bound to the photref, in any channel.

        ``far`` in G is excluded by the rule but bound to nothing, so is not
        fit with this photref, and is not listed.
        """

        for step in ("epd", "tfa"):
            with self.subTest(step=step):
                self.assertEqual(
                    self.detrending_excluded(
                        step,
                        {
                            f"{step}_exclusion_rule": "bg_center[0] > 30",
                            "tfa_observation_id": self._observation_id,
                        },
                        [
                            ("near", "R"),
                            ("near", "G"),
                            ("far", "R"),
                            ("longer", "R"),
                        ],
                    ),
                    ["101 G", "102 R"],
                )

    def test_detrending_lists_the_photometry_a_slot_decides(self):
        """Each observation under every photometry it is excluded in.

        ``far`` in R is excluded in both apertures, so listed twice, and
        ``longer`` in neither.
        """

        self.assertEqual(
            self.detrending_excluded(
                "epd",
                {
                    "epd_exclusion_rule": "photometry_mag_offset[0][0] > 30",
                    "tfa_observation_id": self._observation_id,
                },
                [("near", "R"), ("near", "G"), ("far", "R"), ("longer", "R")],
            ),
            ["101 G ap0", "101 R ap1", "102 R ap0", "102 R ap1"],
        )

    def test_detrending_without_a_rule_lists_nothing(self):
        """Unset, or set only for the other step, or a step taking none."""

        rule = "bg_center[0] > 30"
        for step, configuration in (
            ("epd", {"epd_exclusion_rule": None, "tfa_exclusion_rule": rule}),
            ("tfa", {"epd_exclusion_rule": rule, "tfa_exclusion_rule": None}),
            ("generate_epd_statistics", {"epd_exclusion_rule": rule}),
        ):
            with self.subTest(step=step):
                self.assertIsNone(
                    self.detrending_excluded(
                        step,
                        dict(
                            configuration,
                            tfa_observation_id=self._observation_id,
                        ),
                        [("near", "R"), ("far", "R")],
                    )
                )

    def test_detrending_ids_must_be_header_keywords(self):
        """Which is all the engine can read for an image."""

        with self.assertRaises(ConfigurationError):
            self.detrending_excluded(
                "epd",
                {
                    "epd_exclusion_rule": "bg_center[0] > 30",
                    "tfa_observation_id": ("skypos.bjd",),
                },
                [("far", "R")],
            )


if __name__ == "__main__":
    unittest.main()
