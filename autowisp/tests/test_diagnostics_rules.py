"""Tests for the diagnostics table's rules, which need no project.

The decisions a series table makes that are questions about names and
arithmetic rather than about an observing project: what a row id means,
what a channel column offers, what a section header says, and which
marker a section starts with.  They are pure, so they are checked here
without the throwaway project database ``test_diagnostics_views`` builds
-- which is the point of having kept them pure.

The grouping of series into subplots keeps them company: it takes arrays
rather than a database, and splitting it from the rules it sits beside
would buy nothing.
"""

import unittest

import matplotlib

# The backend has to be selected before anything imports pyplot, which the
# view module under test does at import time.
matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import numpy

from autowisp.diagnostics.expression_series import SeriesKey
from autowisp.browser_interface.diagnostics.image_diagnostics_views import (
    assign_y_axes,
    group_series_by_x_overlap,
)
from autowisp.browser_interface.diagnostics.quantities import (
    describe_quantity,
    next_section_marker,
    section_markers,
)
from autowisp.browser_interface.diagnostics.series_table import (
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


class TestQuantityDescription(unittest.TestCase):
    """What a section header says about the quantity it draws.

    A collapsed section shows only this, so it has to say what is plotted
    without the rows below it being visible.
    """

    library = {"smooth_bg": "median_filter(bg_center[0], 5)"}
    descriptions = {
        "bg_center": "Background at the frame centre.",
        "smooth_bg": "The background, smoothed.",
    }

    def describe(self, name):
        """Return the header entry for one quantity."""

        return describe_quantity(name, self.library, self.descriptions)

    def test_a_recorded_diagnostic_shows_no_expression(self):
        """It is a measurement rather than a formula, so there is none."""

        self.assertEqual(
            self.describe("bg_center"),
            {
                "name": "bg_center",
                "expression": "",
                "description": "Background at the frame centre.",
            },
        )

    def test_an_expression_shows_what_it_is(self):
        """A name its author chose says little on its own."""

        described = self.describe("smooth_bg")

        self.assertEqual(described["expression"], self.library["smooth_bg"])
        self.assertEqual(described["description"], "The background, smoothed.")

    def test_the_time_describes_itself(self):
        """``jd`` is recorded by nothing, so nothing else can say."""

        self.assertTrue(self.describe("jd")["description"])

    def test_an_undescribed_quantity_is_not_an_error(self):
        """A diagnostic recorded without a description still draws."""

        self.assertEqual(
            describe_quantity("num_extracted_src", {}, {}),
            {"name": "num_extracted_src", "expression": "", "description": ""},
        )


class TestSectionMarkers(unittest.TestCase):
    """Which marker a section's rows start with.

    A starting point only: every row's marker stays editable, so this
    decides what a section looks like before anyone touches it.
    """

    def test_the_first_section_takes_the_first_marker(self):
        """With nothing taken there is nothing to avoid."""

        self.assertEqual(next_section_marker([]), section_markers[0])

    def test_a_section_avoids_the_markers_in_use(self):
        """Two sections drawn alike would defeat the point of the default."""

        self.assertEqual(
            next_section_marker(section_markers[:3]), section_markers[3]
        )

    def test_a_marker_freed_by_a_removal_is_taken_up(self):
        """Rather than left idle while a later section doubles up."""

        taken = [section_markers[0], section_markers[2]]

        self.assertEqual(next_section_marker(taken), section_markers[1])

    def test_markers_cycle_once_every_one_is_taken(self):
        """A ninth section repeats the first, a tenth the second.

        Rather than every further section piling onto the same marker,
        which is what picking the first would do.
        """

        taken = list(section_markers)

        self.assertEqual(next_section_marker(taken), section_markers[0])
        self.assertEqual(
            next_section_marker(taken + [section_markers[0]]),
            section_markers[1],
        )


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


class TestYAxisAssignment(unittest.TestCase):
    """Which quantities share a y axis, and in what order the axes come.

    A user says which *number* each quantity should be on, that being far
    easier to say than an ordering; turning those numbers into axes is
    what this does.
    """

    def test_everything_shares_one_axis_by_default(self):
        """Sharing is the safe answer, and the usual one.

        Two quantities wrongly sharing an axis show it at once -- one of
        them is flattened -- where two wrongly separated are each rescaled
        to fill the height and invite a comparison that is not there.
        """

        self.assertEqual(
            assign_y_axes(["bg_center", "smooth_bg"], {}),
            [["bg_center", "smooth_bg"]],
        )

    def test_a_quantity_asked_onto_its_own_axis_gets_one(self):
        """Different units on a shared x is what the second axis is for."""

        self.assertEqual(
            assign_y_axes(
                ["bg_center", "num_extracted_src"],
                {"bg_center": 1, "num_extracted_src": 2},
            ),
            [["bg_center"], ["num_extracted_src"]],
        )

    def test_a_number_nothing_uses_is_skipped(self):
        """Asking for 1 and 3 draws two axes, not three with a gap.

        The number groups and orders; it does not count the axes, which
        would make an empty one in the middle.
        """

        self.assertEqual(
            assign_y_axes(
                ["bg_center", "smooth_bg"],
                {"bg_center": 1, "smooth_bg": 3},
            ),
            [["bg_center"], ["smooth_bg"]],
        )

    def test_an_axis_keeps_its_quantities_in_section_order(self):
        """Which is the order of the legend and of everything else."""

        self.assertEqual(
            assign_y_axes(
                ["a", "b", "c"],
                {"a": 2, "b": 1, "c": 2},
            ),
            [["b"], ["a", "c"]],
        )

    def test_a_quantity_that_is_not_drawn_makes_no_axis(self):
        """A section switched off entirely leaves no empty scale behind."""

        self.assertEqual(
            assign_y_axes(["bg_center"], {"bg_center": 1, "smooth_bg": 2}),
            [["bg_center"]],
        )

    def test_a_quantity_is_named_once_however_many_rows_draw_it(self):
        """Rows of one section share its axis and its label."""

        self.assertEqual(
            assign_y_axes(["bg_center", "bg_center"], {}), [["bg_center"]]
        )

    def test_nothing_drawn_makes_no_axes_at_all(self):
        """The caller then has nothing to label or to draw a legend for."""

        self.assertEqual(assign_y_axes([], {"bg_center": 2}), [])

    def test_an_unanswerable_number_falls_to_the_first_axis(self):
        """A plot is not worth failing over a malformed request.

        The client sends what a dropdown said, and a page from before
        these existed sends nothing at all.
        """

        self.assertEqual(
            assign_y_axes(["a", "b"], {"a": "not a number", "b": None}),
            [["a", "b"]],
        )


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
