"""Tests for evaluating quantities over a photref group's members.

A photref group is given as its members, ``(image, channel)`` pairs that may
span sessions, and each member is one point, with a quantity's slot bound to
that member's own channel.

The fixture is one group over two nights, mixing what has to be told apart:
an image that is a member in both ``G0`` and ``G1``; one that is a member only
in ``G1`` but has a ``G0`` value too, which must not be read; a non-member on
the first night, whose outlying values would shift any aggregate they leaked
into; a member on the second night; and a member with nothing recorded in its
channel.
"""

import tempfile
import unittest
from datetime import datetime

import numpy

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageType,
    ObservingSession,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.expression_series import get_custom_group_values
from autowisp.exceptions import PipelineError


class TestCustomGroupValues(unittest.TestCase):
    """Quantities over an explicit list of members, one point each."""

    #: Per image, in order of Julian date: the night it was taken on and its
    #: ``s_center`` per channel. Distinct everywhere, so that a value read in
    #: the wrong channel, or for the wrong image, cannot pass by luck.
    _images = [
        ("night1", {"G0": 1.0, "G1": 2.0}),
        ("night1", {"G0": 40.0, "G1": 4.0}),
        ("night1", {"G0": 100.0, "G1": 200.0}),
        ("night2", {"G0": 8.0, "G1": 80.0}),
        ("night2", {"G0": 90.0}),
    ]

    #: ``bg_center`` in ``R``, per image, read by quoting the channel.
    _backgrounds = [10.0, 11.0, 12.0, 13.0, 14.0]

    #: The group, as ``(image index, channel)``; image 2 is not in it.
    _members = [(0, "G0"), (0, "G1"), (1, "G1"), (3, "G0"), (4, "G1")]

    @classmethod
    def setUpClass(cls):
        # Closed in tearDownClass rather than by a context manager, which a
        # fixture spanning every test of the class cannot use.
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        set_project_home(cls._tmp.name)
        cls._fill_database()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _add_sessions(db_session):
        """Add the two nights, returning ``{label: ObservingSession}``."""

        sessions = {}
        for day, label in enumerate(("night1", "night2")):
            # False positive: the declarative models are callable.
            # pylint: disable-next=not-callable
            sessions[label] = ObservingSession(
                observer_id=1,
                camera_id=1,
                telescope_id=1,
                mount_id=1,
                observatory_id=1,
                target_id=1,
                label=label,
                start_time_utc=datetime(2023, 3, 1 + day, 20, 0, 0),
                end_time_utc=datetime(2023, 3, 1 + day, 23, 0, 0),
            )
            db_session.add(sessions[label])
        db_session.flush()
        return sessions

    @classmethod
    def _fill_database(cls):
        """Two nights of object frames, holding the group and one other.

        Provenance foreign keys are left dangling, as SQLite does not
        enforce them and nothing here reads them.
        """

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            diagnostic_ids = {}
            for name in ("bg_center", "s_center"):
                diagnostic = DiagnosticType(name=name, description=name)
                db_session.add(diagnostic)
                db_session.flush()
                diagnostic_ids[name] = diagnostic.id

            image_type = ImageType(name="object", description="objects")
            db_session.add(image_type)
            sessions = cls._add_sessions(db_session)

            cls._image_ids = []
            for index, ((night, s_centers), background) in enumerate(
                zip(cls._images, cls._backgrounds)
            ):
                image = Image(
                    raw_fname=f"/data/raw/frame_{index}.fits",
                    image_type_id=image_type.id,
                    observing_session_id=sessions[night].id,
                    jd=2460005.5 + 0.3 * index,
                )
                db_session.add(image)
                db_session.flush()
                cls._image_ids.append(image.id)

                readings = [("bg_center", "R", background)] + [
                    ("s_center", channel, value)
                    for channel, value in s_centers.items()
                ]
                for name, channel, value in readings:
                    db_session.add(
                        ImageDiagnostics(
                            image_id=image.id,
                            channel=channel,
                            diagnostic_id=diagnostic_ids[name],
                            value=value,
                        )
                    )
        # pylint: enable=not-callable

    def _evaluate(self, quantities, expressions=None, members=None):
        """Return :func:`get_custom_group_values` over the group, in a session.

        Members are handed over in reverse, so that the order they come
        back in is the function's doing.
        """

        members = self._members if members is None else members
        with start_db_session() as db_session:
            return get_custom_group_values(
                [
                    (self._image_ids[index], channel)
                    for index, channel in reversed(members)
                ],
                quantities,
                expressions or {},
                db_session,
            )

    def _expected(self, per_member):
        """Return *per_member* of the group's members, as a list."""

        return [per_member(index, channel) for index, channel in self._members]

    def _assert_values(self, actual, expected):
        """Compare arrays holding NaN, which never equals itself."""

        numpy.testing.assert_array_equal(actual, numpy.array(expected))

    def test_each_member_reads_its_own_channel(self):
        """Image 0 twice, in each channel; image 1 in G1 only."""

        values, members = self._evaluate(["s_center"])

        self.assertEqual(
            members,
            [
                (self._image_ids[index], channel)
                for index, channel in self._members
            ],
        )
        self._assert_values(
            values["s_center"],
            self._expected(
                lambda index, channel: self._images[index][1].get(
                    channel, numpy.nan
                )
            ),
        )

    def test_an_aggregate_spans_the_whole_group(self):
        """Both nights and both channels, and nothing outside the group.

        The median of the members' 1, 2, 4 and 8 is 3; image 2's 100 and
        200 would move it, and so would image 1's G0 value.
        """

        values, _ = self._evaluate(
            ["rel_s"], {"rel_s": "s_center[0] - nanmedian(s_center[0])"}
        )

        self._assert_values(values["rel_s"], [-2.0, -1.0, 1.0, 5.0, numpy.nan])

    def test_a_quoted_channel_is_read_for_every_member(self):
        """Whatever the member's own channel, including for image 0 twice."""

        values, _ = self._evaluate(
            ["background"], {"background": "bg_center['R']"}
        )

        self._assert_values(
            values["background"],
            self._expected(lambda index, _: self._backgrounds[index]),
        )

    def test_the_time_is_read_per_member(self):
        """Channel-free, so the same for both of image 0's entries."""

        values, _ = self._evaluate(["jd"])

        self._assert_values(
            values["jd"],
            self._expected(lambda index, _: 2460005.5 + 0.3 * index),
        )

    def test_a_quantity_comparing_channels_is_refused(self):
        """A member has one channel to bind, its own."""

        with self.assertRaises(PipelineError):
            self._evaluate(["colour"], {"colour": "s_center[0] - s_center[1]"})


if __name__ == "__main__":
    unittest.main()
