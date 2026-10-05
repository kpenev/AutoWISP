"""Tests for deciding which observations an exclusion rule leaves out.

The fixture is the one of ``test_series_references``, a night fit against
two references with a frame bound differently in its two channels and one
bound to nothing, with what a rule needs added: the camera and its channels,
a background in the second channel, a library, and a second night. The
second night is fit against ``ref1`` too, and is far brighter in everything,
so a rule comparing to an aggregate decides differently if the nights are
pooled.
"""

import unittest

from sqlalchemy import select, update

from autowisp.database.interface import start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticExpression,
    DiagnosticType,
    ImageDiagnostics,
    ObservingSession,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraChannel,
    CameraType,
)

# pylint: enable=no-name-in-module
from autowisp.database.user_interface import count_cameras_lacking
from autowisp.diagnostics.exclusion_rules import (
    get_excluded,
    get_rule_reads,
)
from autowisp.exceptions import ConfigurationError
from autowisp.tests.test_series_references import ReferenceProject


class TestExclusionRules(ReferenceProject):
    """What a rule excludes among observations fit together."""

    #: ``bg_center`` in ``B`` on the first night, falling where ``R`` rises.
    _b_backgrounds = [206.0, 205.0, 204.0, 203.0, 202.0, 201.0, 200.0]

    _library = {
        "offset": "photometry_mag_offset[0]['ap0']",
        "green": "bg_center['G0']",
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._extend_database()

    @classmethod
    def _add_cameras(cls, db_session, channels, num_cameras):
        """
        Add a camera type with the given channels, and cameras of it.

        Args:
            db_session:    The session to add them in.

            channels(iterable of str):    The names of the type's channels.

            num_cameras(int):    How many cameras of the type to add.

        Returns:
            [int]:    The ids of the cameras.
        """

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        camera_type = CameraType(
            make="make",
            model="model",
            version="1",
            x_resolution=100,
            y_resolution=100,
            pixel_size=5.0,
        )
        db_session.add(camera_type)
        db_session.flush()
        cameras = [
            Camera(
                camera_type_id=camera_type.id,
                serial_number=f"{camera_type.id}-{index}",
                notes="",
            )
            for index in range(num_cameras)
        ]
        db_session.add_all(cameras)
        db_session.add_all(
            CameraChannel(
                camera_type_id=camera_type.id,
                name=name,
                x_offset=offset,
                y_offset=offset,
            )
            for offset, name in enumerate(channels)
        )
        # pylint: enable=not-callable
        db_session.flush()
        return [camera.id for camera in cameras]

    @classmethod
    def _extend_database(cls):
        """Add a night, the camera, the ``B`` backgrounds and the library."""

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            # No background in B, and every frame bound to ref1.
            _, cls._later_image_ids = cls._add_night(
                db_session,
                "later_night",
                day=1,
                backgrounds=[500.0, 501.0, 502.0],
                offsets={
                    "R": [200.0, 300.0, 100.0],
                    "B": [400.0, 600.0, 500.0],
                },
                bound_to={"R": ["ref1"] * 3, "B": ["ref1"] * 3},
            )
            db_session.execute(
                update(ObservingSession).values(
                    camera_id=cls._add_cameras(db_session, "RB", 1)[0]
                )
            )

            bg_center_id = db_session.scalar(
                select(DiagnosticType.id).where(
                    DiagnosticType.name == "bg_center"
                )
            )
            db_session.add_all(
                ImageDiagnostics(
                    image_id=image_id,
                    channel="B",
                    diagnostic_id=bg_center_id,
                    value=background,
                )
                for image_id, background in zip(
                    cls.image_ids, cls._b_backgrounds
                )
            )
            db_session.add_all(
                DiagnosticExpression(
                    name=name, expression=expression, description=""
                )
                for name, expression in cls._library.items()
            )
        # pylint: enable=not-callable

    def first(self, channel, *frames):
        """Return the given frames of the first night in *channel*."""

        return {(self.image_ids[frame], channel) for frame in frames}

    def later(self, channel, *frames):
        """Return the given frames of the second night in *channel*."""

        return {(self._later_image_ids[frame], channel) for frame in frames}

    def everything(self):
        """Return every frame of both nights in both channels."""

        return set().union(
            *(
                self.first(channel, *range(len(self.image_ids)))
                | self.later(channel, *range(len(self._later_image_ids)))
                for channel in "RB"
            )
        )

    def excluded(self, rule, members=None, **kwargs):
        """Return what *rule* excludes among *members*, by default all."""

        with start_db_session() as db_session:
            return get_excluded(
                rule,
                self.everything() if members is None else members,
                db_session,
                **kwargs,
            )

    def test_a_slot_decides_each_channel_within_its_night(self):
        """Above the median of its own night and channel, not of them all."""

        self.assertEqual(
            self.excluded("bg_center[0] > nanmedian(bg_center[0])"),
            {
                None: self.first("R", 4, 5, 6)
                | self.first("B", 0, 1, 2)
                # Nothing is recorded in B on the second night, and what is
                # not recorded is kept.
                | self.later("R", 2)
            },
        )

    def test_quoted_channels_decide_for_every_channel_fit(self):
        """One verdict per image, applied to those of its channels fit."""

        members = self.everything() - self.first("B", 5)

        self.assertEqual(
            self.excluded("bg_center['R'] > 103", members),
            {
                None: self.first("R", 4, 5, 6)
                | self.first("B", 4, 6)
                | self.later("R", 0, 1, 2)
                | self.later("B", 0, 1, 2)
            },
        )

    def test_a_magfit_read_is_judged_within_its_reference(self):
        """Each reference of each night is a population of its own.

        The frame bound to nothing is in none of them, and is kept.
        """

        self.assertEqual(
            self.excluded("offset[0] > nanmedian(offset[0])"),
            {
                None: self.first("R", 2, 5)
                # The frame bound to ref2 in R is bound to ref1 in B.
                | self.first("B", 2, 5, 4)
                | self.later("R", 1)
                | self.later("B", 1)
            },
        )

    def test_a_quoted_magfit_channel_is_split_by_its_reference(self):
        """Although the rule binds no channel to hold the reference."""

        self.assertEqual(
            self.excluded("photometry_mag_offset['B']['ap0'] > 45"),
            {
                None: self.first("R", 5)
                | self.first("B", 5)
                | self.later("R", 0, 1, 2)
                | self.later("B", 0, 1, 2)
            },
        )

    def test_a_photometry_slot_decides_each_photometry(self):
        """Every one recorded is listed, with what is excluded in it."""

        first_night = self.first("R", *range(7)) | self.first("B", *range(7))

        self.assertEqual(
            self.excluded("photometry_mag_offset[0][0] > 1002.5", first_night),
            {
                0: set(),
                # One frame records nothing in R in this aperture.
                2: self.first("R", 2, 3, 4, 5) | self.first("B", *range(6)),
            },
        )

    def test_magfit_cannot_be_read_before_it(self):
        """Directly or through the library; anything else can."""

        for rule in ("photometry_mag_offset[0][0] > 1", "offset[0] > 1"):
            with self.subTest(rule=rule):
                with self.assertRaises(ConfigurationError):
                    self.excluded(rule, before_magfit=True)

        self.assertEqual(
            self.excluded("bg_center['R'] > 105.5", before_magfit=True),
            {
                None: self.first("R", 6)
                | self.first("B", 6)
                | self.later("R", 0, 1, 2)
                | self.later("B", 0, 1, 2)
            },
        )

    def test_a_channel_the_camera_lacks_is_refused(self):
        """Also where only an expression the rule uses quotes it."""

        for rule in (
            "bg_center['G0'] > 1",
            "green > 1",
            "bg_center[0] > green",
        ):
            with self.subTest(rule=rule):
                with self.assertRaises(ConfigurationError):
                    self.excluded(rule)

    def test_a_photometry_not_extracted_is_refused(self):
        """It would read as not recorded, and keep every image."""

        with self.assertRaises(ConfigurationError):
            self.excluded("photometry_mag_offset[0]['ap7'] > 1")

    def test_what_is_no_rule_is_refused(self):
        """Two slots, where one is bound; and a value where a verdict goes."""

        for rule in ("bg_center[0] > bg_center[1]", "bg_center[0] - 100"):
            with self.subTest(rule=rule):
                with self.assertRaises(ConfigurationError):
                    self.excluded(rule)

    def test_excluding_most_of_a_fit_is_a_warning(self):
        """The fraction is reported either way, conspicuously if large."""

        first_night = self.first("R", *range(7))
        for limit, level in ((105.5, "INFO"), (100.5, "WARNING")):
            with self.subTest(limit=limit):
                with self.assertLogs(
                    "autowisp.diagnostics.exclusion_rules", "INFO"
                ) as logged:
                    self.excluded(f"bg_center[0] > {limit}", first_night)
                self.assertEqual(
                    [record.levelname for record in logged.records], [level]
                )

    def test_cameras_lacking_a_channel_are_counted(self):
        """Per name some camera lacks, counting cameras rather than types.

        Beside the fixture's camera with ``R`` and ``B``, two cameras with
        ``R`` and ``G0``, and a type defining ``X`` that no camera is of.
        Added in this test's session alone and rolled back, so the other
        tests never see them.
        """

        with start_db_session() as db_session:
            self._add_cameras(db_session, ["R", "G0"], 2)
            self._add_cameras(db_session, ["X"], 0)
            self.assertEqual(
                count_cameras_lacking(
                    ["R", "B", "G0", "X", "Y", "B"], db_session
                ),
                {"B": (2, 3), "G0": (1, 3), "X": (3, 3), "Y": (3, 3)},
            )
            db_session.rollback()

    def test_rules_are_read_with_the_slots_they_take(self):
        """Each slot bound once; what takes more, or is broken, is no rule.

        Whether one gives true or false is not known without evaluating it,
        so one giving numbers is listed too.
        """

        self.assertEqual(
            get_rule_reads(
                {
                    "quoted": "bg_center['R'] > 103",
                    "per_channel": "bg_center[0] > 103",
                    "per_photometry": "photometry_mag_offset['R'][0] > 1",
                    "per_both": "photometry_mag_offset[0][0] > 1",
                    "numbers": "bg_center[0] - 100",
                    "two_channels": "bg_center[0] > bg_center[1]",
                    "broken": "bg_center[0] >",
                }
            ),
            {
                "quoted": "quoted",
                "per_channel": "per_channel[0]",
                "per_photometry": "per_photometry[()][0]",
                "per_both": "per_both[0][0]",
                "numbers": "numbers[0]",
            },
        )


if __name__ == "__main__":
    unittest.main()
