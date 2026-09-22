"""Tests for binding a channel to each column of the series table.

Its own module because these need a project recording *two* channels, and
a session per kind of camera: a colour one with a choice to offer and a
monochrome one without.  The fixture in ``test_diagnostics_views`` records
a single channel, and giving it a second would change what every
series-table test there sees.

Django is configured here, unlike in the other series-table tests, because
what a rebinding answers includes the row's channel cells as rendered
HTML, and rendering them needs the template engine.
"""

import os
import tempfile
import unittest
from datetime import datetime

import django
import matplotlib

# The backend has to be selected before anything imports pyplot, which the
# view module under test does at import time.
matplotlib.use("Agg")

# Against the throwaway user data directory that ``autowisp.tests``
# installs, as ``test_bui_models`` does it, so that the developer's real
# browser-interface database is never touched. Nothing here reads that
# database -- rendering a template does not -- but configuring Django at
# all would otherwise point at it.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "autowisp.browser_interface.django_project.settings",
)
django.setup()

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
    get_series_data,
)
from autowisp.browser_interface.diagnostics.series_table import (
    get_available_series,
    get_table_response,
    make_id,
    split_pair_id,
)

# pylint: enable=wrong-import-position


#: JD of the colour session's first frame.
_first_jd = 2460000.5

#: The monochrome session is a night later, so the two cannot overlap.
_night_separation = 1.0


class TwoChannelProject(unittest.TestCase):
    """The fixture the classes below share, holding no tests of its own.

    Its own project rather than the shared one, which records a single
    channel: giving that one a second would change what every other
    series-table test sees.  Two sessions, one per kind of camera -- a
    colour one with a choice to offer and a monochrome one without --
    since the table treats them differently and a project may hold both.
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

    def table_for(self, x_quantity, y_quantity):
        """Return what one section's table offers for an axis pair."""

        with start_db_session() as db_session:
            return get_available_series(
                x_quantity, y_quantity, {}, db_session, marker="o"
            )

    def first_row(self, x_quantity, y_quantity):
        """Return the row the table starts with, on the colour session.

        Both sessions begin at the same moment in this fixture, so the
        pairs are ordered by their text and ``colour_night`` comes first.
        """

        return self.table_for(x_quantity, y_quantity)["diagnostics_list"][0]

    def moved_to(self, row, session_id, *channels):
        """Return *row* as the client posts it after moving it to a session."""

        return {
            **row,
            "pair": make_id(session_id, "object"),
            "channels": list(channels),
        }


class TestChannelColumns(TwoChannelProject):
    """What each column of a row offers, given what the project records."""

    def test_a_colour_session_offers_a_choice(self):
        """Both channels, with the images each would draw."""

        row = self.first_row("jd", "bg_center")

        self.assertEqual(split_pair_id(row["pair"])[0], self.session_id)
        self.assertEqual(len(row["slots"]), 1)
        self.assertFalse(row["slots"][0]["fixed"])
        self.assertEqual(
            [
                (option["value"], option["text"])
                for option in row["slots"][0]["options"]
                if option["value"]
            ],
            [
                (channel, f"{channel} ({len(values)})")
                for channel, values in sorted(self.values_of.items())
            ],
        )
        self.assertEqual(row["count"], "-")

    def test_a_column_per_axis_that_binds_one(self):
        """Two axes over a diagnostic ask for two channels, not one."""

        self.assertEqual(
            len(self.first_row("bg_center", "bg_center")["slots"]), 2
        )


class TestRebinding(TwoChannelProject):
    """What the server answers when a row's binding changes.

    Its own class rather than more of the one above: what a column offers
    is a question about the project, where this is a question about one
    edit and what the figure is drawn from afterwards. They share only the
    fixture.
    """

    def rebinding(self, row, *channels, session_id=None, automatic=True):
        """Return ``(rows to draw, fields)`` for *row* moved and rebound.

        The monochrome session unless told otherwise, that being the move
        which changes what the columns may offer. *automatic* says what
        the client reports about the row's colour and label, which is what
        decides whether the server replaces them.
        """

        posted = {
            **self.moved_to(
                row,
                self.mono_session_id if session_id is None else session_id,
                *channels,
            ),
            "automatic_color": automatic,
            "automatic_label": automatic,
        }
        with start_db_session() as db_session:
            return get_table_response(
                {"bind": posted["id"], "datasets": {posted["id"]: posted}},
                x_quantity="jd",
                expressions={},
                db_session=db_session,
            )

    def rebind(self, row, *channels, session_id=None):
        """Return only what a rebinding answers, discarding the rows."""

        return self.rebinding(row, *channels, session_id=session_id)[1]

    def test_a_pair_change_is_answered_with_the_channels_unset(self):
        """Which channels a column may offer depends on the pair, so the
        cells are re-rendered before anything is bound in them."""

        response = self.rebind(
            self.first_row("jd", "bg_center"), "", session_id=self.session_id
        )

        self.assertEqual(response["count"], "-")
        self.assertIn("B (3)", response["slot_cells"])
        self.assertIn("R (3)", response["slot_cells"])

    def test_a_session_recording_one_channel_settles_on_it(self):
        """One channel to choose from is no choice at all, so the cell
        states what it binds and the row is counted at once."""

        response = self.rebind(self.first_row("jd", "bg_center"), "")

        self.assertIn('data-channel="R"', response["slot_cells"])
        self.assertEqual(response["count"], len(self.mono_values))

    def test_a_rebinding_carries_the_session_times(self):
        """Read-only cells that the client cannot work out for itself."""

        response = self.rebind(self.first_row("jd", "bg_center"), "R")

        self.assertEqual(response["start"], "2023-03-01 20:00")
        self.assertEqual(response["end"], "2023-03-01 23:00")

    def test_a_channel_the_new_pair_offers_is_kept(self):
        """The user chose it, and the new session records it too."""

        response = self.rebind(self.first_row("jd", "bg_center"), "R")

        self.assertEqual(response["count"], len(self.mono_values))
        self.assertEqual(response["label"], "bg_center mono_night object R")

    def test_a_label_names_its_quantity_first(self):
        """So that a legend entry says which section a series belongs to.

        With one section that is redundant, but prefixing always keeps the
        rule simple -- and the label is the user's to rewrite either way.
        """

        label = self.first_row("jd", "bg_center")["label"]

        self.assertTrue(
            label.startswith("bg_center "),
            f"expected the quantity first, got {label!r}",
        )

    def test_a_channel_the_new_pair_lacks_is_dropped(self):
        """``B`` is recorded for the colour session alone.

        Carrying it over would leave the row naming data that is not
        there; the column takes the one channel it does offer instead.
        """

        response = self.rebind(self.first_row("jd", "bg_center"), "B")

        self.assertNotIn("B", response["slot_cells"])
        self.assertIn('data-channel="R"', response["slot_cells"])

    def test_nothing_is_answered_without_a_rebound_row(self):
        """Which is every other redraw: a colour, a marker, a row toggled.

        The rows still come back, those being what the figure is drawn
        from whether or not anything was rebound.
        """

        with start_db_session() as db_session:
            self.assertEqual(
                get_table_response(
                    {"datasets": {}},
                    x_quantity="jd",
                    expressions={},
                    db_session=db_session,
                ),
                ([], {}),
            )

    def test_an_automatic_colour_is_corrected_before_drawing(self):
        """The whole point of answering before the figure is made.

        A row rebound onto another channel is drawn in that channel's
        colour, with a legend naming it, in the same round trip that
        writes them into the table -- rather than drawn in the colour of
        the binding it just left while the table shows the new one.
        """

        row = self.first_row("jd", "bg_center")
        drawn, fields = self.rebinding(row, "R")

        self.assertEqual(len(drawn), 1)
        self.assertEqual(drawn[0]["color"], fields["color"])
        self.assertEqual(drawn[0]["label"], fields["label"])
        self.assertNotEqual(drawn[0]["color"], row["color"])

    def test_a_colour_the_user_chose_survives_the_rebinding(self):
        """Theirs, in the figure as well as in the table.

        Colouring by quantity rather than by channel is a thing a user may
        want, and a rebinding must not quietly undo it.
        """

        row = {**self.first_row("jd", "bg_center"), "color": "#123456"}
        drawn, fields = self.rebinding(row, "R", automatic=False)

        self.assertEqual(drawn[0]["color"], "#123456")
        # What the binding *would* make default is still reported, for the
        # client to apply or ignore by the same test the server just made.
        self.assertNotEqual(fields["color"], "#123456")

    def test_no_row_but_the_rebound_one_is_touched(self):
        """A rebinding answers for the row that was rebound, and no other."""

        row = self.first_row("jd", "bg_center")
        other = {
            **self.moved_to(row, self.session_id, "B"),
            "id": make_id("bg_center", 1),
            "color": "#ffffff",
            "label": "left alone",
            "automatic_color": True,
            "automatic_label": True,
        }
        bound = {
            **self.moved_to(row, self.mono_session_id, "R"),
            "automatic_color": True,
            "automatic_label": True,
        }

        with start_db_session() as db_session:
            drawn, _ = get_table_response(
                {
                    "bind": bound["id"],
                    "datasets": {bound["id"]: bound, other["id"]: other},
                },
                x_quantity="jd",
                expressions={},
                db_session=db_session,
            )

        untouched = next(r for r in drawn if r["id"] == other["id"])
        self.assertEqual(untouched["color"], "#ffffff")
        self.assertEqual(untouched["label"], "left alone")


class TestAddedRow(TwoChannelProject):
    """What a press of ``+`` earns: a copy of the row it was pressed in."""

    def add_below(self, row, *others):
        """Return ``(rows to draw, fields)`` for ``+`` pressed in *row*."""

        datasets = {
            entry["id"]: entry for entry in (row,) + others if entry is not None
        }
        with start_db_session() as db_session:
            return get_table_response(
                {"add": row["id"], "datasets": datasets},
                x_quantity="jd",
                expressions={},
                db_session=db_session,
            )

    def test_the_copy_carries_the_pair_and_channels(self):
        """The next series usually differs in one thing, so copying
        leaves only that thing to change."""

        row = self.moved_to(
            self.first_row("jd", "bg_center"), self.session_id, "B"
        )
        drawn, fields = self.add_below(row)

        self.assertEqual(fields["after"], row["id"])
        self.assertEqual(len(drawn), 2)
        self.assertEqual(drawn[1]["pair"], row["pair"])
        self.assertEqual(drawn[1]["channels"], ["B"])

    def test_the_copy_is_drawn_with_the_figure_that_carries_it(self):
        """Rather than the table gaining a row the plot catches up with."""

        row = self.moved_to(
            self.first_row("jd", "bg_center"), self.session_id, "B"
        )
        drawn, _ = self.add_below(row)

        self.assertEqual(drawn[1]["count"], len(self.values_of["B"]))

    def test_the_copy_takes_the_next_ordinal(self):
        """And one past the highest, so a gap left by a removal is not
        reused while the row that had it may still be on the page."""

        row = self.moved_to(
            self.first_row("jd", "bg_center"), self.session_id, "B"
        )
        third = {**row, "id": make_id("bg_center", 2)}

        _, fields = self.add_below(row, third)

        self.assertIn(
            'id="' + make_id("bg_center", 3) + '"', fields["added_row"]
        )

    def test_the_copy_is_rendered_as_a_row(self):
        """Through the one row template, so it arrives able to do
        everything a row rendered with the page can."""

        row = self.moved_to(
            self.first_row("jd", "bg_center"), self.session_id, "B"
        )
        _, fields = self.add_below(row)

        for expected in (
            'class="diagnostic-row active"',
            'class="add-row"',
            'class="remove-row"',
            'class="pair-select"',
            'class="slot-cell"',
            'class="series-count"',
        ):
            self.assertIn(expected, fields["added_row"])


class TestSeriesValues(TwoChannelProject):
    """Which of the posted rows are drawn, and what each one reads."""

    def test_which_rows_are_drawn(self):
        """Every row is posted, so five of them have to be skipped here.

        Two are the user saying not to draw it, and three name no data: a
        row still to be bound, one from a page whose script predates the
        channel columns and posts no channels at all, and one from before
        the session and type were chosen in the row, which names no
        population to read. The stale two are skipped rather than refused
        -- such a page should draw nothing, not turn the response into an
        error page. Defaulting ``selected`` to true keeps a payload stored
        before the table posted every row still plotting.
        """

        rows = {
            "bound and selected": {"channels": ["R"], "selected": True},
            "not selected": {"channels": ["R"], "selected": False},
            "no marker": {"channels": ["R"], "marker": " "},
            "not yet bound": {"channels": [""], "selected": True},
            "from a page with no channel columns": {"channels": []},
            "from a page with no pair dropdown": {
                "channels": ["R"],
                "pair": None,
            },
            "from an older payload": {"channels": ["B"]},
        }
        series_list = [
            {
                "id": make_id("bg_center", ordinal),
                "pair": make_id(self.session_id, "object"),
                "marker": "o",
                "color": "#ffffff",
                "label": description,
                **config,
            }
            for ordinal, (description, config) in enumerate(rows.items())
        ]

        with start_db_session() as db_session:
            drawn = collect_series_data(series_list, "jd", {}, db_session)

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
                    "id": make_id("bg_center", 0),
                    "pair": make_id(self.session_id, "object"),
                    "channels": ["R", "B"],
                },
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
                    "id": make_id("bg_center", 0),
                    "pair": make_id(self.session_id, "object"),
                    "channels": ["B"],
                },
                "jd",
                {},
                db_session,
            )

        self.assertEqual(y_values.tolist(), self.values_of["B"])
        self.assertEqual(x_values.tolist(), self.jd_values())
