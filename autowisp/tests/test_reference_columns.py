"""Tests for the series table over magfit diagnostics.

A diagnostic ``fit_magnitudes`` produces is read in a photometric reference
and a photometry as well as a channel, so the table chooses all three: the
reference with the channel, in the channel columns, and the photometry in
columns of its own. These need the night of ``test_series_references``,
fit against two references and recorded in two apertures, rather than the
fixture of ``test_channel_binding``, which records neither.

Django is configured here because what a rebinding answers includes the
row's cells as rendered HTML, and rendering them needs the template engine.
"""

import os

import django
import matplotlib
import numpy

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
from autowisp.database.interface import start_db_session
from autowisp.browser_interface.diagnostics.image_diagnostics_views import (
    collect_series_data,
)
from autowisp.browser_interface.diagnostics.series_table import (
    get_available_series,
    get_table_response,
    make_id,
)
from autowisp.tests.test_series_references import ReferenceProject

# pylint: enable=wrong-import-position


class TestReferenceColumns(ReferenceProject):
    """The table over magfit diagnostics: references and photometries
    chosen in its columns.

    On the night of :mod:`autowisp.tests.test_series_references`, fit
    against two references in each of two channels, with one frame bound
    differently in the two and one bound to nothing, and the offsets
    recorded in two apertures, one of them missing a frame in R.
    """

    _expressions = {
        # The offset in its slot's channel and, by quoting it, in B, so
        # that a row needs a reference for each; both in one photometry.
        "offset_to_b": "photometry_mag_offset[0][0]"
        " - photometry_mag_offset['B'][0]",
        # Two photometries of one channel, so two photometry columns.
        "apertures": "photometry_mag_offset[0][0]"
        " - photometry_mag_offset[0][1]",
        # A photometry already bound, so no column at all.
        "in_ap2": "photometry_mag_offset[0]['ap2']",
    }

    def _table(self, y_quantity):
        """Return the table a section drawing *y_quantity* against jd has."""

        with start_db_session() as db_session:
            return get_available_series(
                "jd", y_quantity, self._expressions, db_session, marker="o"
            )

    def _value(self, channel, reference):
        """Return what a column posts for *reference* in *channel*."""

        return make_id(channel, self.photref[reference, channel])

    def test_a_magfit_column_offers_its_references(self):
        """Each (channel, photref) the night's frames are bound to.

        What each is called, and how many frames it draws, are for the page
        and for the counting tests respectively.
        """

        cell = self._table("photometry_mag_offset")["diagnostics_list"][0][
            "channel_slots"
        ][0]

        self.assertEqual(
            [option["value"] for option in cell["options"]][1:],
            [
                self._value("B", "ref1"),
                self._value("B", "ref2"),
                self._value("R", "ref1"),
                self._value("R", "ref2"),
            ],
        )

    def test_a_quoted_magfit_read_gets_a_column_of_its_own(self):
        """After the slot's, offering only the quoted channel's references."""

        channel_slots = self._table("offset_to_b")["diagnostics_list"][0][
            "channel_slots"
        ]

        self.assertEqual(len(channel_slots), 2)
        self.assertEqual(
            [option["value"] for option in channel_slots[1]["options"]][1:],
            [self._value("B", "ref1"), self._value("B", "ref2")],
        )

    def test_a_rebound_row_counts_the_frames_of_its_binding(self):
        """Those fit against both references, as the series will draw, and
        recording the offset in the photometry bound: in aperture 2, one of
        them does not in R."""

        row_id = make_id("offset_to_b", 0)
        population = self.selected(
            range(len(self.image_ids)), R="ref1", B="ref1"
        )
        for photometry in self.shifts:
            with (
                self.subTest(photometry=photometry),
                start_db_session() as db_session,
            ):
                _, answer = get_table_response(
                    {
                        "datasets": {
                            row_id: {
                                "pair": make_id(self.session_id, "object"),
                                "channels": [
                                    self._value("R", "ref1"),
                                    self._value("B", "ref1"),
                                ],
                                "photometries": [str(photometry)],
                            }
                        },
                        "bind": row_id,
                    },
                    x_quantity="jd",
                    expressions=self._expressions,
                    db_session=db_session,
                )

                self.assertEqual(
                    answer["count"],
                    len(
                        [
                            index
                            for index in population
                            if ("R", index, photometry) != self.unrecorded
                        ]
                    ),
                )

    def test_a_row_is_drawn_from_its_references(self):
        """Its tail column posted too, and only the frames fit against both.

        Here the one frame fit against ref2 in R but ref1 in B. A click on
        a point opens it in the row's channel, R, not in ``R|…``.
        """

        with start_db_session() as db_session:
            drawn = collect_series_data(
                [
                    {
                        "id": make_id("offset_to_b", 0),
                        "pair": make_id(self.session_id, "object"),
                        "channels": [
                            self._value("R", "ref2"),
                            self._value("B", "ref1"),
                        ],
                        "photometries": ["0"],
                        "marker": "o",
                    }
                ],
                "jd",
                self._expressions,
                db_session,
            )

        # Drawn at all: a row posting more values than its axes take was
        # once skipped as not fully bound.
        self.assertEqual(len(drawn), 1)
        series, _, y_values, image_ids = drawn[0]
        self.assertEqual(series["channel"], "R")
        self.assertEqual(
            image_ids.tolist(),
            self.selected(self.image_ids, R="ref2", B="ref1"),
        )
        self.assertEqual(
            y_values.tolist(),
            [
                r_offset - b_offset
                for r_offset, b_offset in zip(
                    self.selected(self.offsets["R"], R="ref2", B="ref1"),
                    self.selected(self.offsets["B"], R="ref2", B="ref1"),
                )
            ],
        )

    def test_each_photometry_parameter_gets_a_column(self):
        """However the photometries are reached: a diagnostic takes one, an
        expression one per photometry parameter, and a quoted photometry
        none. Each column offers both of the night's apertures."""

        expected = {
            "photometry_mag_offset": 1,
            "offset_to_b": 1,
            "apertures": 2,
            "in_ap2": 0,
        }
        for quantity, column_count in expected.items():
            with self.subTest(quantity=quantity):
                cells = self._table(quantity)["diagnostics_list"][0][
                    "photometry_slots"
                ]
                self.assertEqual(
                    [
                        [option["value"] for option in cell["options"]][1:]
                        for cell in cells
                    ],
                    [[str(photometry) for photometry in sorted(self.shifts)]]
                    * column_count,
                )

    def test_two_photometries_of_one_channel_are_compared(self):
        """Each read in the photometry its column binds.

        Aperture 2 against aperture 0 differs by their shift throughout,
        but on the frame aperture 2 did not record in R, where it is
        undefined. A row beside it posting one photometry for two columns
        names no binding, and is not drawn.
        """

        def posted(ordinal, photometries):
            """Return a row of ``apertures`` on R, fit against ref1."""

            return {
                "id": make_id("apertures", ordinal),
                "pair": make_id(self.session_id, "object"),
                "channels": [self._value("R", "ref1")],
                "photometries": photometries,
                "marker": "o",
            }

        with start_db_session() as db_session:
            drawn = collect_series_data(
                [posted(0, ["2", "0"]), posted(1, ["2"])],
                "jd",
                self._expressions,
                db_session,
            )

        self.assertEqual(
            [series["id"] for series, *_ in drawn], [make_id("apertures", 0)]
        )
        numpy.testing.assert_allclose(
            drawn[0][2],
            [
                (
                    numpy.nan
                    if ("R", index, 2) == self.unrecorded
                    else self.shifts[2] - self.shifts[0]
                )
                for index in self.selected(range(len(self.image_ids)), R="ref1")
            ],
            equal_nan=True,
        )
