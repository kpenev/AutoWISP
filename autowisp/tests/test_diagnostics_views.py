"""Tests for the BUI diagnostics view modules.

What the series table offers for a pair of axes -- the (session, image
type) pairs a row may be drawn for, and the one row the table starts with
-- what a row means once the client posts it back, and how the figure
below offsets and groups what those rows draw.

Uses a throwaway project database, following ``test_error_render``.
"""

import tempfile
import unittest
from datetime import datetime
from unittest import mock

import matplotlib

# The backend has to be selected before anything imports pyplot, which the
# view module under test does at import time.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import numpy
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

# pylint: enable=no-name-in-module

# Imported after set_project_home is available; these only touch the DB
# through start_db_session, so no Django configuration is needed.
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    get_canonical_images,
)
from autowisp.exceptions import PipelineError
from autowisp.browser_interface.diagnostics.image_diagnostics_views import (
    create_diagnostics_figure,
    get_series_data,
    group_series_by_x_overlap,
)
from autowisp.browser_interface.diagnostics.series_table import (
    get_available_diagnostics,
    get_available_expressions,
    get_available_series,
    get_recorded_diagnostics,
    get_series_key,
    make_id,
    make_slot_cells,
    next_row_id,
    split_pair_id,
    split_row_id,
    unset_option_text,
)

# pylint: enable=wrong-import-position


#: JD of the first image of the first night.
_first_jd = 2460000.5

#: Nights are one day apart, so their JD ranges cannot overlap.
_night_separation = 1.0

#: Recorded for object frames alone, which is what makes them the case of
#: a diagnostic that is not drawable for every image type.
_quantile_names = ("pixel_q99", "pixel_q999")

#: Every diagnostic the fixture records.  All are created explicitly: the
#: lazy database initialization behind ``set_project_home`` creates the
#: schema but seeds no ``diagnostic_type`` rows, and depending on that would
#: couple these tests to project-creation behaviour they are not about.
_diagnostic_names = ("bg_center",) + _quantile_names

#: Frames of each type per night.  Only the second night is mixed, which is
#: what lets these tests tell a per-type series from one that lumps a whole
#: session together; leaving the first night single-type keeps the plain
#: one-series-per-night cases readable.
_frames_per_night = ({"object": 3}, {"object": 3, "flat": 2})

#: ``bg_center`` of the first frame of each type.  Far enough apart that a
#: median over one type cannot be confused with a median over the mixture.
_first_bg_center = {"object": 100.0, "flat": 500.0}


class DiagnosticsViewTestCase(unittest.TestCase):
    """Base creating one throwaway project database holding two nights."""

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

    #: ``{(night, image_type): [image_id, ...]}``, in JD order, so a test can
    #: say which images a series is supposed to be built from.
    images_of = {}

    def table_for(self, x_diagnostic, y_diagnostic, expressions=None):
        """Return what the series table offers for an axis pair."""

        with start_db_session() as db_session:
            return get_available_series(
                x_diagnostic, y_diagnostic, expressions or {}, db_session
            )

    def pairs_for(self, x_diagnostic, y_diagnostic, expressions=None):
        """Return the ``(session_id, image_type)`` pairs a row may name.

        What the table used to answer with a row each, and now answers
        with the options of one dropdown.
        """

        return [
            split_pair_id(option["value"])
            for option in self.table_for(
                x_diagnostic, y_diagnostic, expressions
            )["pair_options"]
        ]

    def first_row(self, x_diagnostic, y_diagnostic, expressions=None):
        """Return the row the table starts with, on the earliest session."""

        return self.table_for(x_diagnostic, y_diagnostic, expressions)[
            "diagnostics_list"
        ][0]

    @staticmethod
    def bind(row, session_id, image_type, *channels):
        """Return the row as the client posts it with its dropdowns set.

        A test asks for a particular series the way the table does -- by
        choosing a pair and a channel per column -- rather than by picking
        a row out of a list, there being one row to start from.
        """

        return {
            **row,
            "pair": make_id(session_id, image_type),
            "channels": list(channels),
        }

    @classmethod
    def _fill_database(cls):
        """Create two observing sessions, the second holding two image types.

        Every frame records ``bg_center`` in channel ``R``; only object
        frames record the quantiles, which is the ordinary case of a
        diagnostic that is not defined for every type.  Provenance foreign
        keys are left dangling, as SQLite does not enforce them and the
        diagnostics queries only ever join back to ``observing_session`` and
        ``image_type``.
        """

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            for name in _diagnostic_names:
                db_session.add(
                    DiagnosticType(
                        name=name, description=f"Test diagnostic {name}"
                    )
                )
            for name in ("object", "flat"):
                db_session.add(ImageType(name=name, description=f"{name}s"))
            db_session.flush()

            diagnostic_ids = dict(
                db_session.execute(
                    select(DiagnosticType.name, DiagnosticType.id).where(
                        DiagnosticType.name.in_(_diagnostic_names)
                    )
                ).all()
            )
            image_type_ids = dict(
                db_session.execute(select(ImageType.name, ImageType.id)).all()
            )

            for night, frame_counts in enumerate(_frames_per_night):
                session = ObservingSession(
                    observer_id=1,
                    camera_id=1,
                    telescope_id=1,
                    mount_id=1,
                    observatory_id=1,
                    target_id=1,
                    label=f"night_{night}",
                    start_time_utc=datetime(2023, 3, 1 + night, 20, 0, 0),
                    end_time_utc=datetime(2023, 3, 1 + night, 23, 0, 0),
                )
                db_session.add(session)
                db_session.flush()

                jd = _first_jd + night * _night_separation
                for image_type, count in frame_counts.items():
                    cls.images_of[night, image_type] = []
                    for index in range(count):
                        image = Image(
                            raw_fname=(
                                f"/data/raw/n{night}_{image_type}_{index}.fits"
                            ),
                            image_type_id=image_type_ids[image_type],
                            observing_session_id=session.id,
                            jd=jd,
                        )
                        db_session.add(image)
                        db_session.flush()
                        cls.images_of[night, image_type].append(image.id)
                        jd += 0.05

                        values = {
                            "bg_center": _first_bg_center[image_type] + index
                        }
                        if image_type == "object":
                            values["pixel_q99"] = 200.0 + index
                            values["pixel_q999"] = 300.0 + index

                        for name, value in values.items():
                            db_session.add(
                                ImageDiagnostics(
                                    image_id=image.id,
                                    channel="R",
                                    diagnostic_id=diagnostic_ids[name],
                                    value=value,
                                )
                            )
        # pylint: enable=not-callable


class TestRowId(unittest.TestCase):
    """The row id, a round trip through the client and back.

    It becomes an HTML element id, five more element ids are built from
    it, and it keys the ``datasets`` object the client posts back -- so it
    has to survive all of that as an opaque string. What it deliberately
    does *not* carry is anything the row can edit: the session, the image
    type and the channels are all chosen in the row, and an id that
    changed as they were would take every element id with it.
    """

    def test_a_row_names_its_quantity_and_its_place(self):
        """And nothing else, those being the only things it cannot edit."""

        self.assertEqual(
            split_row_id(make_id("bg_center", 0)), ("bg_center", 0)
        )

    def test_underscores_are_harmless(self):
        """A diagnostic name carries them, and the separator is not one.

        This is what an earlier encoding could not do without guessing
        which underscores separated fields and which belonged to a name.
        """

        self.assertEqual(
            split_row_id(make_id("pixel_q999", 3)), ("pixel_q999", 3)
        )

    def test_an_ambiguous_field_is_refused(self):
        """Failing loudly beats an id that silently pairs wrong data."""

        with self.assertRaises(ValueError):
            make_id("we|rd", 0)

    def test_the_quantity_is_part_of_the_identity(self):
        """Two rows drawing different quantities must not collide."""

        self.assertNotEqual(make_id("bg_center", 0), make_id("smooth_bg", 0))

    def test_siblings_differ_by_their_ordinal(self):
        """One quantity may be drawn by any number of rows."""

        self.assertNotEqual(make_id("bg_center", 0), make_id("bg_center", 1))

    def test_an_added_row_takes_the_next_ordinal(self):
        """``+`` copies a row, and the copy needs an id of its own."""

        self.assertEqual(
            next_row_id(
                {"id": make_id("bg_center", 0)}, [make_id("bg_center", 0)]
            ),
            make_id("bg_center", 1),
        )

    def test_an_ordinal_freed_by_a_removal_is_not_reused(self):
        """One past the highest in use, not the first gap in the run.

        The id is the suffix of five element ids, so a second row
        answering to them would show up as one row's colour arriving on
        another's.
        """

        self.assertEqual(
            next_row_id(
                {"id": make_id("bg_center", 0)},
                [make_id("bg_center", 0), make_id("bg_center", 3)],
            ),
            make_id("bg_center", 4),
        )

    def test_another_quantity_does_not_crowd_the_ordinals(self):
        """They count per quantity, the two together making the id."""

        self.assertEqual(
            next_row_id(
                {"id": make_id("bg_center", 0)},
                [make_id("bg_center", 0), make_id("smooth_bg", 7)],
            ),
            make_id("bg_center", 1),
        )

    def test_a_pair_id_round_trips(self):
        """The dropdown's value, opaque to the client exactly as a row id is."""

        self.assertEqual(
            split_pair_id(make_id(7, "twilight_flat")), (7, "twilight_flat")
        )

    def test_the_key_comes_from_what_the_client_posts(self):
        """All of it: the pair is edited in the row as the channels are.

        Reading the session or the type from the id would read what the
        row was rendered with rather than what its dropdown now says.
        """

        self.assertEqual(
            get_series_key(
                {
                    "id": make_id("bg_center", 2),
                    "pair": make_id(7, "object"),
                    "channels": ["R", "B"],
                }
            ),
            SeriesKey(7, "object", ("R", "B")),
        )


class TestAvailableQuantities(DiagnosticsViewTestCase):
    """What the two axis selectors offer, given what this project holds.

    Availability rather than validity: every stored expression is valid in
    every project, so the only question a selector can usefully ask is
    whether the diagnostics an expression reaches have actually been
    recorded here.
    """

    def _recorded(self):
        """Return the raw in-use diagnostic names."""

        with start_db_session() as db_session:
            return get_recorded_diagnostics(db_session)

    def test_recorded_names_are_raw(self):
        """The individual quantiles, not the family that replaces them.

        This is the set an expression is judged against, and one may
        reference a concrete ``pixel_q999``, so the collapse must happen
        after rather than before.
        """

        self.assertEqual(sorted(self._recorded()), sorted(_diagnostic_names))

    def test_selector_offers_the_family_not_its_members(self):
        """One entry expanding to a series per quantile, and ``jd`` first."""

        available = get_available_diagnostics(self._recorded(), {})

        self.assertEqual(available[0], "jd")
        self.assertIn("pixel_quantiles", available)
        self.assertIn("bg_center", available)
        for name in _quantile_names:
            self.assertNotIn(name, available)

    def test_expressions_join_the_same_flat_list(self):
        """Not a second list beside it.

        An axis reads one name, and a recorded diagnostic is an expression
        of itself as far as everything downstream is concerned -- which is
        the same flat name space that stops an expression taking a
        diagnostic's name.
        """

        available = get_available_diagnostics(
            self._recorded(), {"rel_bg": "bg_center * 2"}
        )

        self.assertIn("rel_bg", available)
        self.assertIn("bg_center", available)

    def test_expression_offered_where_its_inputs_are_recorded(self):
        """The ordinary case, and the composed one behind it."""

        library = {
            "rel_bg": "bg_center - nanmedian(bg_center)",
            "twice_rel_bg": "rel_bg * 2",
        }

        self.assertEqual(
            get_available_expressions(library, self._recorded()),
            ["rel_bg", "twice_rel_bg"],
        )

    def test_expression_hidden_where_an_input_is_not_recorded(self):
        """Not an error -- ``astrom_residual`` is real, merely unrecorded.

        The distinction the whole design rests on: this expression is valid
        here and would be offered in a project that had run plate solving.
        """

        library = {"rel": "astrom_residual / diagonal_fov"}

        self.assertEqual(
            get_available_expressions(library, self._recorded()), []
        )

    def test_a_dependency_makes_its_dependents_unavailable(self):
        """Availability follows the whole subtree, not the direct names."""

        library = {
            "rel": "astrom_residual / diagonal_fov",
            "scaled": "rel * 2",
        }

        self.assertEqual(
            get_available_expressions(library, self._recorded()), []
        )

    def test_a_concrete_quantile_is_judged_against_the_raw_names(self):
        """Which is why the family collapse happens after this check.

        Against the collapsed list ``pixel_q999`` would look unrecorded and
        a perfectly drawable expression would go missing from the selector.
        """

        library = {"contrast": "pixel_q999 / pixel_q99"}

        self.assertEqual(
            get_available_expressions(library, self._recorded()), ["contrast"]
        )

    def test_a_jd_only_expression_is_always_available(self):
        """``jd`` exists for every image of the canonical list."""

        library = {"night_time": "jd - nanmin(jd)"}

        self.assertEqual(
            get_available_expressions(library, self._recorded()), ["night_time"]
        )

    def test_a_broken_expression_is_hidden_rather_than_raised(self):
        """A stored cycle must not stop the plot page rendering.

        Saying what is wrong with it belongs to the management page; here
        the only sane answer is not to offer it.
        """

        library = {"a": "b + 1", "b": "a + 1", "rel_bg": "bg_center * 2"}

        self.assertEqual(
            get_available_expressions(library, self._recorded()), ["rel_bg"]
        )

    def test_an_unknown_name_is_hidden_too(self):
        """A typo no version of AutoWISP defines, rather than a cycle."""

        library = {"typo": "bg_centre * 2"}

        self.assertEqual(
            get_available_expressions(library, self._recorded()), []
        )


class TestPairOptions(DiagnosticsViewTestCase):
    """The (session, image type) pairs a row may be drawn for.

    What the table answered with a row each when it listed them, and now
    answers with the options of one dropdown.
    """

    def test_a_pair_needs_every_column_to_have_a_channel(self):
        """Only object frames record the quantiles, so only they are offered."""

        self.assertEqual(
            {image_type for _, image_type in self.pairs_for("jd", "pixel_q99")},
            {"object"},
        )

    def test_listed_by_session_start_time(self):
        """Labels are free-form, so the times are what orders them."""

        options = self.table_for("jd", "bg_center")["pair_options"]

        self.assertEqual(
            [option["start"] for option in options],
            sorted(option["start"] for option in options),
        )
        self.assertEqual(options[0]["text"], "night_0 object")

    def test_an_option_names_its_session_and_type(self):
        """Otherwise the two rows of the mixed night would read alike."""

        self.assertEqual(
            {
                option["text"]
                for option in self.table_for("jd", "bg_center")["pair_options"]
            },
            {"night_0 object", "night_1 object", "night_1 flat"},
        )

    def test_the_times_sort_as_text(self):
        """Which is why they are formatted rather than left as datetimes."""

        first = self.table_for("jd", "bg_center")["pair_options"][0]

        self.assertEqual(first["start"], "2023-03-01 20:00")
        self.assertEqual(first["end"], "2023-03-01 23:00")


class TestInitialRow(DiagnosticsViewTestCase):
    """The one row a table starts with, so the page draws without a click."""

    def test_exactly_one_row(self):
        """The table lists nothing; the rest are built by the user."""

        self.assertEqual(
            len(self.table_for("jd", "bg_center")["diagnostics_list"]), 1
        )

    def test_on_the_first_pair_offered(self):
        """The earliest session, that being how the options are ordered."""

        table = self.table_for("jd", "bg_center")

        self.assertEqual(
            table["diagnostics_list"][0]["pair"],
            table["pair_options"][0]["value"],
        )
        self.assertEqual(
            table["diagnostics_list"][0]["pair_sort"], "night_0 object"
        )

    def test_it_names_the_quantity_it_draws(self):
        """Which is what lets the figure read a y per row rather than a page."""

        self.assertEqual(
            split_row_id(self.first_row("jd", "bg_center")["id"]),
            ("bg_center", 0),
        )

    def test_one_channel_recorded_arrives_bound(self):
        """A column with one channel to offer is no choice at all.

        The fixture records ``R`` alone, so the row is bound and counted
        at render and draws the moment the page loads.
        """

        row = self.first_row("jd", "bg_center")

        self.assertEqual(row["channels"], ["R"])
        self.assertEqual(row["count"], _frames_per_night[0]["object"])


class TestSlotCells(unittest.TestCase):
    """What each channel column of a row offers, and what a move keeps.

    Pure, so the rules can be checked without a database or a browser.
    """

    def test_a_column_with_one_channel_is_settled(self):
        """Asking for a click with one possible outcome is ceremony."""

        cells = make_slot_cells([{"R": 3}], ())

        self.assertTrue(cells[0]["fixed"])
        self.assertEqual(cells[0]["value"], "R")
        self.assertEqual(cells[0]["sort"], "R")

    def test_a_column_with_a_choice_starts_unset(self):
        """Nothing picks one of several channels on the user's behalf."""

        cells = make_slot_cells([{"R": 3, "B": 2}], ())

        self.assertFalse(cells[0]["fixed"])
        self.assertEqual(cells[0]["value"], "")
        self.assertEqual(cells[0]["sort"], unset_option_text)

    def test_a_channel_the_pair_still_offers_is_kept(self):
        """The user chose it, and it remains an answer here."""

        self.assertEqual(
            make_slot_cells([{"R": 3, "B": 2}], ("B",))[0]["value"], "B"
        )

    def test_a_channel_the_pair_does_not_offer_is_cleared(self):
        """Rather than quietly bound to something the user never chose."""

        self.assertEqual(
            make_slot_cells([{"R": 3, "G": 1}], ("B",))[0]["value"], ""
        )

    def test_each_column_is_decided_on_its_own(self):
        """One axis may have a channel to choose where the other has none."""

        cells = make_slot_cells([{"R": 3}, {"R": 3, "B": 2}], ("R", "B"))

        self.assertEqual([cell["fixed"] for cell in cells], [True, False])

    def test_an_option_carries_the_text_it_shows(self):
        """One place decides it, since the cell sorts by that same text."""

        self.assertEqual(
            [
                (option["value"], option["text"])
                for option in make_slot_cells([{"R": 3, "B": 2}], ())[0][
                    "options"
                ]
            ],
            [("", unset_option_text), ("B", "B (2)"), ("R", "R (3)")],
        )


class TestSharedTimeOffset(DiagnosticsViewTestCase):
    """The x-offset is one value for the whole figure, not per series."""

    def _plotted_x_values(self):
        """Return the x arrays that reach the per-series plotting call.

        ``plot_image_diagnostic_series`` is mocked out, which both captures
        the offset values and avoids ``reverse()`` needing Django settings.
        """

        table = self.table_for("jd", "bg_center")
        # A row per (session, image type), built the way the user builds
        # them: the table starts with one row, and the rest are that row on
        # the other pairs its dropdown offers, each with an id of its own.
        series_list = [
            {
                **self.bind(
                    table["diagnostics_list"][0],
                    *split_pair_id(option["value"]),
                    "R",
                ),
                "id": make_id("bg_center", ordinal),
            }
            for ordinal, option in enumerate(table["pair_options"])
        ]
        # Night 0 object, night 1 object, night 1 flat.
        self.assertEqual(len(series_list), 3)

        with start_db_session() as db_session:
            target = (
                "autowisp.browser_interface.diagnostics"
                ".image_diagnostics_views.plot_image_diagnostic_series"
            )
            with mock.patch(target) as plot_series:
                create_diagnostics_figure(
                    series_list,
                    x_diagnostic="jd",
                    expressions={},
                    db_session=db_session,
                    # Nothing is drawn once plotting is mocked, so asking
                    # for a legend only produces a warning.
                    figure_config={"show_legend": False},
                )

        return [call.args[1] for call in plot_series.call_args_list]

    def _series_starts(self):
        """Return where each plotted series begins on the shared x axis."""

        return sorted(float(min(values)) for values in self._plotted_x_values())

    def test_only_the_earliest_series_starts_at_zero(self):
        """One offset for the figure, so exactly one series lands on 0."""

        starts = self._series_starts()
        self.assertAlmostEqual(starts[0], 0.0, places=6)
        for start in starts[1:]:
            self.assertGreater(start, 0.0)

    def test_offset_is_not_per_series(self):
        """Guard the exact regression the merge could introduce.

        Zeroing each series on its own would start every one of them at 0,
        collapsing the day between the two nights.  The second night's
        series keep that day, wherever in the night each one begins.
        """

        for start in self._series_starts()[1:]:
            self.assertGreaterEqual(
                start,
                _night_separation,
                msg="a second-night series was zeroed on its own -- the "
                "offset became per-series instead of shared",
            )


class TestImageTypeSplit(DiagnosticsViewTestCase):
    """A session holding several image types yields a series per type."""

    def test_each_type_is_offered_separately(self):
        """The mixed night offers object and flat as two pairs, not one."""

        mixed = {
            image_type
            for session_id, image_type in self.pairs_for("jd", "bg_center")
            if session_id == 2
        }
        self.assertEqual(mixed, {"object", "flat"})

    def test_a_type_without_the_diagnostic_is_absent(self):
        """Only object frames record the quantiles, so only they appear."""

        self.assertEqual(
            {
                image_type
                for _, image_type in self.pairs_for("jd", "pixel_q999")
            },
            {"object"},
        )

    def test_the_type_is_shown_in_the_table(self):
        """Otherwise two pairs of the mixed night would read identically."""

        self.assertIn(
            "Session and Type",
            self.table_for("jd", "bg_center")["diagnostics_fields"],
        )

    def test_canonical_list_holds_only_its_own_type(self):
        """The alignment the whole design rests on is per type.

        Every array is padded onto this list, so if it mixed types then so
        would every quantity built against it.
        """

        with start_db_session() as db_session:
            for image_type in ("object", "flat"):
                image_ids, _ = get_canonical_images(
                    SeriesKey(2, image_type, ("R",)), db_session
                )
                self.assertEqual(
                    image_ids.tolist(), self.images_of[1, image_type]
                )

    def test_values_are_not_taken_across_types(self):
        """The point of the split: an aggregate sees one population.

        A series covering the whole night would hand ``nanmedian`` all five
        frames and return the object median, since the objects outnumber the
        flats -- silently, and wrongly.
        """

        series = self.bind(self.first_row("jd", "bg_center"), 2, "flat", "R")
        with start_db_session() as db_session:
            _, y_values, image_ids = get_series_data(
                series, "jd", {}, db_session
            )

        flat_values = [
            _first_bg_center["flat"] + index
            for index in range(_frames_per_night[1]["flat"])
        ]
        self.assertEqual(y_values.tolist(), flat_values)
        self.assertEqual(image_ids.tolist(), self.images_of[1, "flat"])
        self.assertAlmostEqual(
            float(numpy.nanmedian(y_values)),
            numpy.median(flat_values),
            places=6,
        )


class TestExpressionAxis(DiagnosticsViewTestCase):
    """An expression selected for an axis, as a diagnostic would be.

    The library is passed in rather than stored, which is the arrangement
    that lets these run against a project database alone: what the view does
    with an expression does not depend on where it was kept.
    """

    #: Referenced by every test here; ``bg_center`` is recorded for both
    #: image types, so the availability answer is interesting.
    library = {
        "rel_bg": "bg_center[1] - nanmedian(bg_center[1])",
        "scaled_bg": "rel_bg[1] * 10",
        "q_ratio": "pixel_q999[1] / pixel_q99[1]",
    }

    def test_offered_wherever_its_diagnostics_are(self):
        """Availability follows what the expression reaches, not its name.

        Nothing records a diagnostic called ``rel_bg``; the series it can be
        drawn for are those recording the ``bg_center`` it is built from.
        """

        self.assertEqual(
            self.pairs_for("jd", "rel_bg", self.library),
            self.pairs_for("jd", "bg_center", self.library),
        )

    def test_a_composed_expression_reaches_through(self):
        """``scaled_bg`` needs what ``rel_bg`` needs, transitively."""

        self.assertEqual(
            self.pairs_for("jd", "scaled_bg", self.library),
            self.pairs_for("jd", "bg_center", self.library),
        )

    def test_restricted_to_the_types_recording_its_inputs(self):
        """Only object frames record the quantiles, so only they are offered."""

        self.assertEqual(
            {
                image_type
                for _, image_type in self.pairs_for(
                    "jd", "q_ratio", self.library
                )
            },
            {"object"},
        )

    def test_the_values_are_the_expression_evaluated(self):
        """End to end: an expression axis produces its own numbers."""

        series = self.bind(
            self.first_row("jd", "rel_bg", self.library), 2, "object", "R"
        )
        with start_db_session() as db_session:
            _, y_values, _ = get_series_data(
                series, "jd", self.library, db_session
            )

        # bg_center is 100, 101, 102 for these frames.
        self.assertEqual(y_values.tolist(), [-1.0, 0.0, 1.0])

    def test_an_expression_against_a_diagnostic(self):
        """Both axes at once, one of each kind, sharing a query."""

        series = self.bind(
            self.first_row("bg_center", "rel_bg", self.library),
            2,
            "object",
            "R",
            "R",
        )
        with start_db_session() as db_session:
            x_values, y_values, _ = get_series_data(
                series, "bg_center", self.library, db_session
            )

        self.assertEqual(x_values.tolist(), [100.0, 101.0, 102.0])
        self.assertEqual(y_values.tolist(), [-1.0, 0.0, 1.0])

    def test_an_unknown_name_is_refused(self):
        """Neither a diagnostic nor an expression, so nothing to plot."""

        with self.assertRaises(PipelineError):
            self.pairs_for("jd", "no_such_thing", self.library)


class TestSeriesGrouping(unittest.TestCase):
    """``group_series_by_x_overlap`` splits only non-overlapping ranges."""

    @staticmethod
    def _entry(jd_values):
        """Build the tuple shape the grouping helper consumes."""

        return ({}, numpy.asarray(jd_values), None, None)

    def test_disjoint_ranges_split(self):
        """Two nights a day apart occupy separate subplots."""

        groups = group_series_by_x_overlap(
            [
                self._entry([_first_jd, _first_jd + 0.1]),
                self._entry(
                    [
                        _first_jd + _night_separation,
                        _first_jd + _night_separation + 0.1,
                    ]
                ),
            ]
        )
        self.assertEqual(len(groups), 2)

    def test_overlapping_ranges_merge(self):
        """Overlapping ranges share one subplot.

        This is the case a non-time x axis reduces to once the grouping is
        generalized from JD to arbitrary x, so it must keep holding.
        """

        groups = group_series_by_x_overlap(
            [
                self._entry([0.0, 10.0]),
                self._entry([5.0, 15.0]),
                self._entry([12.0, 20.0]),
            ]
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)


if __name__ == "__main__":
    unittest.main()
