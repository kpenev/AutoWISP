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
    ObservingSession,
)

# pylint: enable=no-name-in-module

# Imported after set_project_home is available; these only touch the DB
# through start_db_session, so no Django configuration is needed.
from autowisp.browser_interface.diagnostics.image_diagnostics_views import (
    create_diagnostics_figure,
    get_available_series,
    group_series_by_x_overlap,
    make_series,
    split_series_id,
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


class DiagnosticsViewTestCase(unittest.TestCase):
    """Base creating one throwaway project database holding two nights."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        set_project_home(cls._tmp.name)
        cls._fill_database()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _fill_database(cls):
        """Create two observing sessions of three images each.

        Both nights record ``bg_center`` and both quantiles in channel ``R``,
        so every combination under test has data.  Provenance foreign keys are
        left dangling, as SQLite does not enforce them and the diagnostics
        queries only ever join back to ``observing_session``.
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
            db_session.flush()

            diagnostic_ids = dict(
                db_session.execute(
                    select(DiagnosticType.name, DiagnosticType.id).where(
                        DiagnosticType.name.in_(_diagnostic_names)
                    )
                ).all()
            )

            for night in range(2):
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

                for index in range(3):
                    image = Image(
                        raw_fname=f"/data/raw/n{night}_{index}.fits",
                        image_type_id=1,
                        observing_session_id=session.id,
                        jd=(
                            _first_jd + night * _night_separation + index * 0.05
                        ),
                    )
                    db_session.add(image)
                    db_session.flush()

                    for name, value in [
                        ("bg_center", 100.0 + index),
                        ("pixel_q99", 200.0 + index),
                        ("pixel_q999", 300.0 + index),
                    ]:
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
    """The id encoding, which is a contract between two functions only.

    It is also an HTML element id, four more element ids are built from it,
    and it keys the ``datasets`` object the client posts back -- so it has
    to survive a round trip through all of that as an opaque string.
    """

    def round_trip(self, session_id, channel, quantile_name=None):
        """Build a series and read its identity back out of the id."""

        series = make_series("night_0", session_id, channel, 3, quantile_name)
        return split_series_id(series)

    def test_plain_series(self):
        """No quantile: the field is empty rather than missing."""

        self.assertEqual(self.round_trip(7, "R"), (7, "R", None))

    def test_quantile_series(self):
        """The quantile survives despite containing underscores.

        This is what the previous encoding could not do without guessing
        which underscores separated fields and which belonged to the name.
        """

        self.assertEqual(
            self.round_trip(7, "R", "pixel_q999"), (7, "R", "pixel_q999")
        )

    def test_channel_containing_an_underscore(self):
        """Underscores anywhere are now harmless."""

        self.assertEqual(
            self.round_trip(7, "odd_channel", "pixel_q999"),
            (7, "odd_channel", "pixel_q999"),
        )

    def test_an_ambiguous_field_is_refused(self):
        """Failing loudly beats an id that silently pairs wrong data."""

        with self.assertRaises(ValueError):
            make_series("night_0", 7, "we|rd", 3)


class TestQuantileSeriesExpansion(DiagnosticsViewTestCase):
    """``quantiles`` expands to one series per ``pixel_q*``, either axis."""

    def _series_ids(self, x_diagnostic, y_diagnostic):
        """Return the series ids offered for the given axis pair."""

        with start_db_session() as db_session:
            context = get_available_series(
                x_diagnostic, y_diagnostic, db_session
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
            context = get_available_series("quantiles", "bg_center", db_session)
        self.assertIn("Quantile", context["diagnostics_fields"])


class TestSharedTimeOffset(DiagnosticsViewTestCase):
    """The x-offset is one value for the whole figure, not per series."""

    def _plotted_x_values(self):
        """Return the x arrays that reach the per-series plotting call.

        ``plot_image_diagnostic_series`` is mocked out, which both captures
        the offset values and avoids ``reverse()`` needing Django settings.
        """

        with start_db_session() as db_session:
            context = get_available_series("jd", "bg_center", db_session)
            series_list = context["diagnostics_list"]
            self.assertEqual(len(series_list), 2, "expected one series/night")

            target = (
                "autowisp.browser_interface.diagnostics"
                ".image_diagnostics_views.plot_image_diagnostic_series"
            )
            with mock.patch(target) as plot_series:
                create_diagnostics_figure(
                    series_list,
                    x_diagnostic="jd",
                    y_diagnostic="bg_center",
                    db_session=db_session,
                    # Nothing is drawn once plotting is mocked, so asking
                    # for a legend only produces a warning.
                    figure_config={"show_legend": False},
                )

        return [call.args[1] for call in plot_series.call_args_list]

    def test_one_series_starts_at_zero(self):
        """Only the earliest night is zeroed; the offset is shared."""

        x_values = self._plotted_x_values()
        self.assertEqual(len(x_values), 2)

        starts = sorted(float(min(values)) for values in x_values)
        self.assertAlmostEqual(starts[0], 0.0, places=6)
        self.assertAlmostEqual(starts[1], _night_separation, places=6)

    def test_offset_is_not_per_series(self):
        """Guard the exact regression the merge could introduce."""

        starts = [float(min(values)) for values in self._plotted_x_values()]
        self.assertNotAlmostEqual(
            starts[0],
            starts[1],
            places=6,
            msg="both nights zeroed independently -- the offset became "
            "per-series instead of shared across the figure",
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
