"""Tests for binding a channel to each column of the series table.

Its own module because these need a project holding *two* channels, and a
session per kind of camera: a colour one with a choice to offer and a
monochrome one without.  The fixture in ``test_diagnostics_views`` records
a single channel, and giving it a second would change what every
series-table test there sees.
"""

import tempfile
import unittest
from datetime import datetime

import matplotlib

# The backend has to be selected before anything imports pyplot, which the
# view module under test does at import time.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
from sqlalchemy import select

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageType,
    ObservingSession,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraChannel,
    CameraType,
)

# pylint: enable=no-name-in-module
from autowisp.browser_interface.diagnostics.image_diagnostics_views import (
    collect_series_data,
    get_available_series,
    get_axes_slot_needs,
    get_series_data,
    get_slot_options,
    make_row_id,
    plan_spare_row,
    split_row_id,
)

# pylint: enable=wrong-import-position


#: JD of the colour session's first frame.
_first_jd = 2460000.5

#: The monochrome session is a night later, so the two cannot overlap.
_night_separation = 1.0


class TestTwoChannelRow(unittest.TestCase):
    """A row binding a channel per axis, which is what the columns are for.

    Its own fixture rather than the shared one, which records a single
    channel: giving that one a second would change what every other
    series-table test sees.  Two sessions, one per kind of camera -- a
    colour one with a choice to offer and a monochrome one without -- since
    the table treats them differently and a project may hold both.
    """

    #: ``bg_center`` per channel, distinct so that reading the wrong one
    #: cannot pass by coincidence.
    values_of = {"R": [8.0, 6.0, 4.0], "B": [2.0, 3.0, 8.0]}

    #: The monochrome session's frames, in its camera's one channel.
    mono_values = [11.0, 12.0]

    @classmethod
    def jd_values(cls):
        """Return the Julian dates the fixture gave its frames, in order."""

        return [
            _first_jd + 0.05 * index for index in range(len(cls.values_of["R"]))
        ]

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        set_project_home(cls._tmp.name)
        cls._fill_database()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _add_camera(cls, db_session, channels):
        """Describe a camera defining *channels*, and return its id."""

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        camera_type = CameraType(
            make="Test",
            model=f"{len(channels)}-channel",
            version="1",
            x_resolution=16,
            y_resolution=16,
            pixel_size=1.0,
            notes="",
        )
        db_session.add(camera_type)
        db_session.flush()

        for offset, name in enumerate(channels):
            db_session.add(
                CameraChannel(
                    camera_type_id=camera_type.id,
                    name=name,
                    x_offset=offset,
                    y_offset=0,
                    x_step=len(channels),
                    y_step=1,
                )
            )

        camera = Camera(
            camera_type_id=camera_type.id,
            serial_number=f"serial-{camera_type.id}",
            notes="",
        )
        db_session.add(camera)
        db_session.flush()
        # pylint: enable=not-callable

        return camera.id

    @classmethod
    def _add_session(cls, db_session, label, camera_id, jds):
        """Add one session of object frames and return its id."""

        # pylint: disable=not-callable
        session = ObservingSession(
            observer_id=1,
            camera_id=camera_id,
            telescope_id=1,
            mount_id=1,
            observatory_id=1,
            target_id=1,
            label=label,
            start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
            end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
        )
        db_session.add(session)
        db_session.flush()

        image_type_id = db_session.execute(select(ImageType.id)).scalar()
        for index, jd in enumerate(jds):
            db_session.add(
                Image(
                    raw_fname=f"/data/raw/{label}_{index}.fits",
                    image_type_id=image_type_id,
                    observing_session_id=session.id,
                    jd=jd,
                )
            )
        db_session.flush()
        # pylint: enable=not-callable

        return session.id

    @classmethod
    def _fill_database(cls):
        """Two sessions: a colour camera's frames, and a monochrome one's."""

        # pylint: disable=not-callable
        with start_db_session() as db_session:
            db_session.add(
                DiagnosticType(name="bg_center", description="Background")
            )
            db_session.add(ImageType(name="object", description="objects"))
            db_session.flush()
            diagnostic_id = db_session.execute(
                select(DiagnosticType.id)
            ).scalar()

            cls.session_id = cls._add_session(
                db_session,
                "colour_night",
                cls._add_camera(db_session, ("R", "B")),
                cls.jd_values(),
            )
            cls.mono_session_id = cls._add_session(
                db_session,
                "mono_night",
                cls._add_camera(db_session, ("R",)),
                [
                    _first_jd + _night_separation + 0.05 * index
                    for index in range(len(cls.mono_values))
                ],
            )

            recorded = {
                cls.session_id: cls.values_of,
                cls.mono_session_id: {"R": cls.mono_values},
            }
            for session_id, per_channel in recorded.items():
                image_ids = [
                    row[0]
                    for row in db_session.execute(
                        # pylint: disable=no-member
                        select(Image.id)
                        .where(Image.observing_session_id == session_id)
                        .order_by(Image.jd)
                        # pylint: enable=no-member
                    ).all()
                ]
                for channel, values in per_channel.items():
                    for image_id, value in zip(image_ids, values):
                        db_session.add(
                            ImageDiagnostics(
                                image_id=image_id,
                                channel=channel,
                                diagnostic_id=diagnostic_id,
                                value=value,
                            )
                        )
        # pylint: enable=not-callable

    def rows_for(self, x_diagnostic, y_diagnostic):
        """Return ``{session_id: row}`` for an axis pair, one type here."""

        with start_db_session() as db_session:
            context = get_available_series(
                x_diagnostic, y_diagnostic, {}, db_session
            )

        return {
            split_row_id(row["id"])[0]: row
            for row in context["diagnostics_list"]
        }

    def test_a_colour_session_offers_a_choice(self):
        """Both channels, with the images each would draw."""

        row = self.rows_for("jd", "bg_center")[self.session_id]

        self.assertEqual(len(row["slots"]), 1)
        self.assertFalse(row["slots"][0]["fixed"])
        self.assertEqual(
            row["slots"][0]["options"],
            [
                (channel, len(values))
                for channel, values in sorted(self.values_of.items())
            ],
        )
        self.assertEqual(row["count"], "-")

    def test_a_monochrome_session_arrives_bound(self):
        """One channel to the camera's name is no choice at all.

        Demanding a click with one possible outcome before anything can be
        drawn is ceremony, so the row is bound and counted at render --
        which is exactly the table this page had before a series could bind
        more than one channel.
        """

        row = self.rows_for("jd", "bg_center")[self.mono_session_id]

        self.assertTrue(row["slots"][0]["fixed"])
        self.assertEqual(row["slots"][0]["value"], "R")
        self.assertEqual(row["channels"], ["R"])
        self.assertEqual(row["count"], len(self.mono_values))

    def test_a_column_per_axis_that_binds_one(self):
        """Two axes over a diagnostic ask for two channels, not one."""

        row = self.rows_for("bg_center", "bg_center")[self.session_id]

        self.assertEqual(len(row["slots"]), 2)

    def row(self, ordinal):
        """Return the id of one row of the colour session's group."""

        return make_row_id(self.session_id, "object", None, ordinal)

    def plan_spare(self, bound_id, datasets):
        """Return the spare row planned when *bound_id* is completed."""

        with start_db_session() as db_session:
            slot_needs = get_axes_slot_needs("jd", "bg_center", {}, None)
            options, labels = get_slot_options(slot_needs, db_session)

        return plan_spare_row(bound_id, slot_needs, options, labels, datasets)

    def test_a_completed_row_summons_a_spare(self):
        """So that a second binding of the same series can be built."""

        spare = self.plan_spare(self.row(0), {self.row(0): {"channels": ["R"]}})

        self.assertEqual(spare["id"], self.row(1))
        self.assertEqual(spare["channels"], [])
        self.assertEqual(spare["count"], "-")

    def test_no_spare_once_every_binding_is_present(self):
        """Two channels, two rows: a third could only repeat one."""

        self.assertIsNone(
            self.plan_spare(
                self.row(1),
                {
                    self.row(0): {"channels": ["R"]},
                    self.row(1): {"channels": ["B"]},
                },
            )
        )

    def test_rebinding_an_earlier_row_summons_nothing(self):
        """A spare is already waiting below it, and one is enough."""

        self.assertIsNone(
            self.plan_spare(
                self.row(0),
                {
                    self.row(0): {"channels": ["B"]},
                    self.row(1): {"channels": []},
                },
            )
        )

    def test_which_rows_are_drawn(self):
        """Every row is posted, so four of them have to be skipped here.

        Two are the user saying not to draw it, and two name no data: a
        row still to be bound, and one from a page whose script predates
        the channel columns, which posts no channels at all. That last is
        skipped rather than refused -- a stale page should draw nothing,
        not turn the response into an error page. Defaulting ``selected``
        to true keeps a payload stored before the table posted every row
        still plotting.
        """

        rows = {
            "bound and selected": {"channels": ["R"], "selected": True},
            "not selected": {"channels": ["R"], "selected": False},
            "no marker": {"channels": ["R"], "marker": " "},
            "not yet bound": {"channels": [""], "selected": True},
            "from a page with no channel columns": {"channels": []},
            "from an older payload": {"channels": ["B"]},
        }
        series_list = [
            {
                "id": self.row(ordinal),
                "marker": "o",
                "color": "#ffffff",
                "label": description,
                **config,
            }
            for ordinal, (description, config) in enumerate(rows.items())
        ]

        with start_db_session() as db_session:
            drawn = collect_series_data(
                series_list, "jd", "bg_center", {}, db_session
            )

        self.assertEqual(
            [series["label"] for series, *_ in drawn],
            ["bound and selected", "from an older payload"],
        )

    def test_one_quantity_on_both_axes_in_two_channels(self):
        """``bg_center`` against ``bg_center``: the plainest colour plot.

        The two axes name one quantity and differ only in what the row
        bound each column to. Resolving them by name alone would draw one
        binding on both axes -- a perfect diagonal, and no error anywhere.
        """

        with start_db_session() as db_session:
            x_values, y_values, image_ids = get_series_data(
                {
                    "id": make_row_id(self.session_id, "object", None, 0),
                    "channels": ["R", "B"],
                },
                "bg_center",
                "bg_center",
                {},
                db_session,
            )

        self.assertEqual(x_values.tolist(), self.values_of["R"])
        self.assertEqual(y_values.tolist(), self.values_of["B"])
        self.assertEqual(image_ids.size, len(self.values_of["R"]))

    def test_the_columns_are_split_between_the_axes(self):
        """An axis takes as many channels as the quantity it draws.

        ``jd`` takes none, so the row's one channel belongs to the other
        axis -- and a slice taken from the wrong end would bind it to the
        time instead and refuse, or worse, silently shift.
        """

        with start_db_session() as db_session:
            x_values, y_values, _ = get_series_data(
                {
                    "id": make_row_id(self.session_id, "object", None, 0),
                    "channels": ["B"],
                },
                "jd",
                "bg_center",
                {},
                db_session,
            )

        self.assertEqual(y_values.tolist(), self.values_of["B"])
        self.assertEqual(x_values.tolist(), self.jd_values())
