"""Characterization tests for the BUI diagnostics view modules.

These pin the behaviours that must survive merging
``image_diagnostics_views`` and ``diag_vs_diag_views`` into a single
x-versus-y path: how the ``quantiles`` pseudo-name expands into series, and
how the time-series figure offsets and groups its series.  They are written
against the *current* code, so they must pass before the merge starts.

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
    get_available_series,
    get_series_data,
    group_series_by_x_overlap,
)

# pylint: enable=wrong-import-position


#: JD of the first image of the first night.
_first_jd = 2460000.5

#: Nights are one day apart, so their JD ranges cannot overlap.
_night_separation = 1.0

#: The quantile diagnostics, which expand to one series each.
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


class TestSeriesId(unittest.TestCase):
    """The id encoding, a round trip through the client and back.

    It becomes an HTML element id, four more element ids are built from it,
    and it keys the ``datasets`` object the client posts back -- so it has
    to survive all of that as an opaque string.
    """

    def round_trip(self, *args):
        """Return the key recovered from the id built from *args*."""

        return SeriesKey.from_id(SeriesKey(*args).to_id())

    def test_plain_series(self):
        """No quantile: the field is empty rather than missing."""

        key = SeriesKey(7, "object", "R")
        self.assertEqual(self.round_trip(*key), key)

    def test_quantile_series(self):
        """The quantile survives despite containing underscores.

        This is what the previous encoding could not do without guessing
        which underscores separated fields and which belonged to the name.
        """

        key = SeriesKey(7, "object", "R", "pixel_q999")
        self.assertEqual(self.round_trip(*key), key)

    def test_underscores_anywhere_are_harmless(self):
        """Neither the channel nor the image type has to avoid them."""

        key = SeriesKey(7, "twilight_flat", "odd_channel", "pixel_q999")
        self.assertEqual(self.round_trip(*key), key)

    def test_an_ambiguous_field_is_refused(self):
        """Failing loudly beats an id that silently pairs wrong data."""

        with self.assertRaises(ValueError):
            SeriesKey(7, "object", "we|rd").to_id()

    def test_the_image_type_is_part_of_the_identity(self):
        """Two types in one session must not collide on one id."""

        self.assertNotEqual(
            SeriesKey(7, "object", "R").to_id(),
            SeriesKey(7, "flat", "R").to_id(),
        )


class TestQuantileSeriesExpansion(DiagnosticsViewTestCase):
    """``quantiles`` expands to one series per ``pixel_q*``, either axis."""

    def _series_ids(self, x_diagnostic, y_diagnostic):
        """Return the series ids offered for the given axis pair."""

        with start_db_session() as db_session:
            context = get_available_series(
                x_diagnostic, y_diagnostic, {}, db_session
            )
        return [series["id"] for series in context["diagnostics_list"]]

    def test_quantiles_on_x_axis(self):
        """One series per quantile per (session, channel), name in the id."""

        series_ids = self._series_ids("quantiles", "bg_center")

        self.assertEqual(len(series_ids), 2 * len(_quantile_names))
        for name in _quantile_names:
            self.assertEqual(
                sum(series_id.endswith(name) for series_id in series_ids),
                2,
                f"expected one {name} series per night, got {series_ids!r}",
            )

    def test_quantiles_on_y_axis(self):
        """Reversing the axes yields the same expansion."""

        self.assertEqual(
            sorted(self._series_ids("bg_center", "quantiles")),
            sorted(self._series_ids("quantiles", "bg_center")),
        )

    def test_quantile_column_present(self):
        """A quantile pairing gains the extra ``Quantile`` table column."""

        with start_db_session() as db_session:
            context = get_available_series(
                "quantiles", "bg_center", {}, db_session
            )
        self.assertIn("Quantile", context["diagnostics_fields"])


class TestSharedTimeOffset(DiagnosticsViewTestCase):
    """The x-offset is one value for the whole figure, not per series."""

    def _plotted_x_values(self):
        """Return the x arrays that reach the per-series plotting call.

        ``plot_image_diagnostic_series`` is mocked out, which both captures
        the offset values and avoids ``reverse()`` needing Django settings.
        """

        with start_db_session() as db_session:
            context = get_available_series("jd", "bg_center", {}, db_session)
            series_list = context["diagnostics_list"]
            # One per (night, image type): night 0 object, night 1 object,
            # night 1 flat.
            self.assertEqual(len(series_list), 3)

            target = (
                "autowisp.browser_interface.diagnostics"
                ".image_diagnostics_views.plot_image_diagnostic_series"
            )
            with mock.patch(target) as plot_series:
                create_diagnostics_figure(
                    series_list,
                    x_diagnostic="jd",
                    y_diagnostic="bg_center",
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

    def _series_for(self, x_diagnostic, y_diagnostic):
        """Return ``{SeriesKey: series}`` offered for an axis pair."""

        with start_db_session() as db_session:
            context = get_available_series(
                x_diagnostic, y_diagnostic, {}, db_session
            )
        return {
            SeriesKey.from_id(series["id"]): series
            for series in context["diagnostics_list"]
        }

    def test_each_type_gets_its_own_series(self):
        """The mixed night offers object and flat separately."""

        types = {
            key.image_type
            for key in self._series_for("jd", "bg_center")
            if key.session_id == 2
        }
        self.assertEqual(types, {"object", "flat"})

    def test_a_type_without_the_diagnostic_is_absent(self):
        """Only object frames record the quantiles, so only they appear."""

        types = {
            key.image_type for key in self._series_for("quantiles", "bg_center")
        }
        self.assertEqual(types, {"object"})

    def test_the_type_is_shown_in_the_table(self):
        """Otherwise two rows of the mixed night would look identical."""

        with start_db_session() as db_session:
            context = get_available_series("jd", "bg_center", {}, db_session)
        self.assertIn("Type", context["diagnostics_fields"])

    def test_canonical_list_holds_only_its_own_type(self):
        """The alignment the whole design rests on is per type.

        Every array is padded onto this list, so if it mixed types then so
        would every quantity built against it.
        """

        with start_db_session() as db_session:
            for image_type in ("object", "flat"):
                image_ids, _ = get_canonical_images(
                    SeriesKey(2, image_type, "R"), db_session
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

        series = self._series_for("jd", "bg_center")[SeriesKey(2, "flat", "R")]
        with start_db_session() as db_session:
            _, y_values, image_ids = get_series_data(
                series, "jd", "bg_center", {}, db_session
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
        "rel_bg": "bg_center - nanmedian(bg_center)",
        "scaled_bg": "rel_bg * 10",
        "q_ratio": "pixel_q999 / pixel_q99",
    }

    def _series_for(self, x_diagnostic, y_diagnostic):
        """Return ``{SeriesKey: series}`` offered for an axis pair."""

        with start_db_session() as db_session:
            context = get_available_series(
                x_diagnostic, y_diagnostic, self.library, db_session
            )
        return {
            SeriesKey.from_id(series["id"]): series
            for series in context["diagnostics_list"]
        }

    def test_offered_wherever_its_diagnostics_are(self):
        """Availability follows what the expression reaches, not its name.

        Nothing records a diagnostic called ``rel_bg``; the series it can be
        drawn for are those recording the ``bg_center`` it is built from.
        """

        self.assertEqual(
            sorted(self._series_for("jd", "rel_bg")),
            sorted(self._series_for("jd", "bg_center")),
        )

    def test_a_composed_expression_reaches_through(self):
        """``scaled_bg`` needs what ``rel_bg`` needs, transitively."""

        self.assertEqual(
            sorted(self._series_for("jd", "scaled_bg")),
            sorted(self._series_for("jd", "bg_center")),
        )

    def test_restricted_to_the_types_recording_its_inputs(self):
        """Only object frames record the quantiles, so only they are offered."""

        self.assertEqual(
            {key.image_type for key in self._series_for("jd", "q_ratio")},
            {"object"},
        )

    def test_the_values_are_the_expression_evaluated(self):
        """End to end: an expression axis produces its own numbers."""

        series = self._series_for("jd", "rel_bg")[SeriesKey(2, "object", "R")]
        with start_db_session() as db_session:
            _, y_values, _ = get_series_data(
                series, "jd", "rel_bg", self.library, db_session
            )

        # bg_center is 100, 101, 102 for these frames.
        self.assertEqual(y_values.tolist(), [-1.0, 0.0, 1.0])

    def test_an_expression_against_a_diagnostic(self):
        """Both axes at once, one of each kind, sharing a query."""

        series = self._series_for("bg_center", "rel_bg")[
            SeriesKey(2, "object", "R")
        ]
        with start_db_session() as db_session:
            x_values, y_values, _ = get_series_data(
                series, "bg_center", "rel_bg", self.library, db_session
            )

        self.assertEqual(x_values.tolist(), [100.0, 101.0, 102.0])
        self.assertEqual(y_values.tolist(), [-1.0, 0.0, 1.0])

    def test_an_unknown_name_is_refused(self):
        """Neither a diagnostic nor an expression, so nothing to plot."""

        with self.assertRaises(PipelineError):
            self._series_for("jd", "no_such_thing")


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
