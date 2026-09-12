"""Tests for reading one series' values out of the project database.

These cover tier 2 of the expression layer: the part that knows the project
database but not Django, and turns a :class:`SeriesKey` into the
``{name: array}`` tier 1 evaluates against.  What is asserted here is
mostly *alignment* -- that index *i* means the same image in every array --
because every other property in the design rests on it and a violation of
it produces a plot that looks entirely reasonable and is wrong.

The two-night fixture is the one ``test_diagnostics_views`` builds, imported
rather than repeated: it is a project-database fixture rather than a view
one, and the view tests and these want exactly the same rows.
"""

import re
import tempfile
import unittest
from datetime import datetime

import numpy
from sqlalchemy import event, select

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
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    _diagnostic_values_query,
    count_images_with_all,
    get_canonical_images,
    get_diagnostic_values,
    get_expression_availability,
    get_quantity_values,
    get_series_values,
)
from autowisp.tests.test_diagnostics_views import DiagnosticsViewTestCase


class SeriesValuesTestCase(DiagnosticsViewTestCase):
    """The two-night fixture, with the keys these tests ask about."""

    #: The mixed night's object frames: three, all recording everything.
    objects = SeriesKey(2, "object", ("R",))

    #: The same night's flats: two, recording ``bg_center`` and no quantile.
    flats = SeriesKey(2, "flat", ("R",))


class TestCanonicalImages(SeriesValuesTestCase):
    """The list every array is padded onto."""

    def test_one_type_only(self):
        """The list is per type, which is what confines an aggregate."""

        with start_db_session() as db_session:
            image_ids, _ = get_canonical_images(self.objects, db_session)
            flat_ids, _ = get_canonical_images(self.flats, db_session)

        self.assertEqual(list(image_ids), self.images_of[1, "object"])
        self.assertEqual(list(flat_ids), self.images_of[1, "flat"])

    def test_the_channel_is_not_consulted(self):
        """Channels share an index space, so the list cannot depend on one."""

        with start_db_session() as db_session:
            for channel in ("R", "G", "B", "no-such-channel"):
                image_ids, _ = get_canonical_images(
                    self.objects._replace(channels=(channel,)), db_session
                )
                self.assertEqual(list(image_ids), self.images_of[1, "object"])

    def test_ordered_by_time(self):
        """Ordered, because every array is aligned to this one by position."""

        with start_db_session() as db_session:
            _, jd_values = get_canonical_images(self.objects, db_session)

        self.assertEqual(list(jd_values), sorted(jd_values))


class TestDiagnosticValues(SeriesValuesTestCase):
    """Padding several diagnostics onto that list in one query.

    What is read is stated as ``{name: {channel tuples}}`` -- the shape the
    walk that decides it produces -- because a diagnostic may be wanted in
    more than one channel at once.
    """

    #: The fixture records everything in this one channel.
    recorded = ("R",)

    def test_values_land_against_their_own_images(self):
        """The fixture makes each value say which image it belongs to."""

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects,
                {name: {self.recorded} for name in ("bg_center", "pixel_q99")},
                db_session,
            )

        self.assertEqual(list(image_ids), self.images_of[1, "object"])
        self.assertEqual(
            list(values["bg_center"][self.recorded]), [100.0, 101.0, 102.0]
        )
        self.assertEqual(
            list(values["pixel_q99"][self.recorded]), [200.0, 201.0, 202.0]
        )

    def test_a_diagnostic_the_type_lacks_is_all_nan(self):
        """Flats record no quantiles, and still owe a full-length column.

        This is the case the padding exists for: an expression over a
        diagnostic some frames lack must produce a series of the right
        length with holes, not a shorter one that silently misaligns.
        """

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.flats,
                {name: {self.recorded} for name in ("bg_center", "pixel_q99")},
                db_session,
            )

        quantile = values["pixel_q99"][self.recorded]
        self.assertEqual(quantile.size, image_ids.size)
        self.assertTrue(numpy.all(numpy.isnan(quantile)))
        self.assertEqual(
            list(values["bg_center"][self.recorded]), [500.0, 501.0]
        )

    def test_a_channel_nothing_was_recorded_in_is_all_nan(self):
        """Asking for a channel this camera never had is not an error.

        It is the same padding one step further out: the join finds
        nothing, so the column is NaN throughout rather than absent, and an
        expression over it is undefined rather than broken.
        """

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects,
                {"bg_center": {self.recorded, ("no-such-channel",)}},
                db_session,
            )

        missing = values["bg_center"][("no-such-channel",)]
        self.assertEqual(missing.size, image_ids.size)
        self.assertTrue(numpy.all(numpy.isnan(missing)))
        self.assertEqual(
            list(values["bg_center"][self.recorded]), [100.0, 101.0, 102.0]
        )

    def test_one_diagnostic_read_in_two_channels(self):
        """What an expression comparing channels needs, in one query.

        Both columns come back against the same images, which is what
        makes a ratio between them meaningful without any joining.
        """

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects,
                {"bg_center": {("R",), ("G",)}},
                db_session,
            )

        self.assertEqual(sorted(values["bg_center"]), [("G",), ("R",)])
        for array in values["bg_center"].values():
            self.assertEqual(array.size, image_ids.size)

    def test_every_array_is_the_same_length(self):
        """Alignment is by position, so a short column would be a bug."""

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects,
                {
                    "bg_center": {self.recorded},
                    "pixel_q99": {self.recorded},
                    "pixel_q999": {self.recorded},
                    "jd": {()},
                },
                db_session,
            )

        for by_channels in values.values():
            for array in by_channels.values():
                self.assertEqual(array.size, image_ids.size)

    def test_time_comes_from_the_image_row(self):
        """``jd`` is not in image_diagnostics and needs no row there."""

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects, {"jd": {()}}, db_session
            )
            _, jd_values = get_canonical_images(self.objects, db_session)

        self.assertEqual(sorted(values), ["jd"])
        self.assertEqual(list(values["jd"][()]), list(jd_values))
        self.assertEqual(values["jd"][()].size, image_ids.size)

    def test_asking_for_nothing_still_gives_the_images(self):
        """The caller needs the image list even with no diagnostic wanted."""

        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                self.objects, {}, db_session
            )

        self.assertEqual(values, {})
        self.assertEqual(list(image_ids), self.images_of[1, "object"])

    def test_each_channel_is_asked_for_by_its_whole_key(self):
        """The shape that keeps this affordable however big the archive.

        A session holds a manageable number of images; the ``image`` table
        will not, so the feature rests on anchoring to one observing
        session and reaching ``image_diagnostics`` by its unique index on
        ``(image_id, channel, diagnostic_id)``. Reading several channels
        adds a join each, and every one has to pin all three columns --
        drop the channel and the join matches every channel's row, which
        is both wrong and a scan.

        Asserted on the statement rather than on a query plan, because the
        predicates are what this module decides; which index to use is the
        database's business, and SQLite's answer would say nothing about
        the MariaDB servers that hold the large archives.
        """

        statement = str(
            _diagnostic_values_query(
                self.objects,
                ["bg_center", "pixel_q99"],
                ["G", "R"],
            ).compile(compile_kwargs={"literal_binds": True})
        )

        self.assertIn("image.observing_session_id = 2", statement)

        # Read the alias names out rather than assuming them: what
        # SQLAlchemy calls an anonymous alias is its own business and has
        # changed before.
        aliases = re.findall(
            r"LEFT OUTER JOIN image_diagnostics AS (\w+)", statement
        )
        self.assertEqual(len(aliases), 2, statement)

        for alias in aliases:
            for column in ("image_id", "diagnostic_id", "channel"):
                self.assertEqual(
                    statement.count(f"{alias}.{column} ="), 1, statement
                )


class TestSeriesValues(SeriesValuesTestCase):
    """Resolving quantities, which is where expressions enter."""

    def test_plain_diagnostics_need_no_library(self):
        """The browser interface's present caller passes an empty one."""

        with start_db_session() as db_session:
            values, image_ids = get_series_values(
                self.objects, ["jd", "bg_center"], {}, db_session
            )

        self.assertEqual(sorted(values), ["bg_center", "jd"])
        self.assertEqual(list(image_ids), self.images_of[1, "object"])

    def test_an_expression_is_evaluated_per_series(self):
        """The aggregate sees this series' images and no others.

        ``bg_center`` runs 100, 101, 102 for these frames and 500, 501 for
        the flats of the same night, so a median taken across the two would
        be nowhere near zero.
        """

        with start_db_session() as db_session:
            values, _ = get_series_values(
                self.objects,
                ["rel_bg"],
                {"rel_bg": "bg_center - nanmedian(bg_center)"},
                db_session,
            )

        self.assertEqual(list(values["rel_bg"]), [-1.0, 0.0, 1.0])

    def test_both_axes_resolve_together(self):
        """Two quantities, one call -- the point of taking a sequence."""

        with start_db_session() as db_session:
            values, _ = get_series_values(
                self.objects,
                ["jd", "twice_bg"],
                {"twice_bg": "bg_center * 2"},
                db_session,
            )

        self.assertEqual(sorted(values), ["jd", "twice_bg"])
        self.assertEqual(list(values["twice_bg"]), [200.0, 202.0, 204.0])

    def test_a_quantity_may_be_asked_for_twice(self):
        """Plotting something against itself is a request, not a mistake."""

        with start_db_session() as db_session:
            values, _ = get_series_values(
                self.objects, ["bg_center", "bg_center"], {}, db_session
            )

        self.assertEqual(list(values["bg_center"]), [100.0, 101.0, 102.0])

    def test_a_composed_expression_resolves_its_dependency(self):
        """Tier 1 orders them; this checks the values reach it to do so."""

        with start_db_session() as db_session:
            values, _ = get_series_values(
                self.objects,
                ["scaled"],
                {
                    "rel_bg": "bg_center - nanmedian(bg_center)",
                    "scaled": "rel_bg * 10",
                },
                db_session,
            )

        self.assertEqual(list(values["scaled"]), [-10.0, 0.0, 10.0])


class TestAvailability(SeriesValuesTestCase):
    """Which series an expression can be drawn for, counted in SQL."""

    def test_it_counts_what_the_expression_reaches(self):
        """An expression is available wherever its diagnostics are."""

        with start_db_session() as db_session:
            available = get_expression_availability(
                "twice_bg", {"twice_bg": "bg_center * 2"}, db_session
            )
            directly = count_images_with_all({"bg_center"}, db_session)

        self.assertEqual(available, directly)

    def test_a_quantile_expression_is_offered_for_objects_only(self):
        """Only object frames record the quantiles in the fixture."""

        with start_db_session() as db_session:
            available = get_expression_availability(
                "q_ratio",
                {"q_ratio": "pixel_q999 / pixel_q99"},
                db_session,
            )

        self.assertEqual({row[2] for row in available}, {"object"})


class TestCrossChannelValues(unittest.TestCase):
    """An expression reading one diagnostic in two channels.

    Its own fixture rather than the shared one, which records a single
    channel: giving that one a second would change what every series-table
    test sees.  Two channels of three frames is the smallest thing that can
    tell a cross-channel quantity from a per-channel one.
    """

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

    #: ``bg_center`` per channel, chosen so that every pairing of them
    #: gives a distinct ratio -- reading the wrong channel cannot land on
    #: the right answer by luck.
    values_of = {"R": [8.0, 6.0, 4.0], "B": [2.0, 3.0, 1.0]}

    #: A channel this camera never recorded, for the undefined case.
    missing_channel = "G"

    @classmethod
    def _fill_database(cls):
        """Three frames, each recording ``bg_center`` in two channels."""

        # False positive: the declarative models are callable.
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
            image_type_id = db_session.execute(select(ImageType.id)).scalar()

            session = ObservingSession(
                observer_id=1,
                camera_id=1,
                telescope_id=1,
                mount_id=1,
                observatory_id=1,
                target_id=1,
                label="colour",
                start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
                end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
            )
            db_session.add(session)
            db_session.flush()

            for index in range(3):
                image = Image(
                    raw_fname=f"/data/raw/colour_{index}.fits",
                    image_type_id=image_type_id,
                    observing_session_id=session.id,
                    jd=2460000.5 + 0.05 * index,
                )
                db_session.add(image)
                db_session.flush()
                for channel, values in cls.values_of.items():
                    db_session.add(
                        ImageDiagnostics(
                            image_id=image.id,
                            channel=channel,
                            diagnostic_id=diagnostic_id,
                            value=values[index],
                        )
                    )
            cls.session_id = session.id
        # pylint: enable=not-callable

    def key(self, channels):
        """Return the series binding *channels*."""

        return SeriesKey(self.session_id, "object", channels)

    def test_a_ratio_between_two_channels(self):
        """The quantity the whole of section 10 exists for.

        Both columns are read against the same images, so the ratio is
        meaningful without anything being joined or matched up.
        """

        with start_db_session() as db_session:
            values, image_ids = get_quantity_values(
                self.key(("R", "B")),
                {"sky_color": ("R", "B")},
                {"sky_color": "bg_center[1] / bg_center[2]"},
                db_session,
            )

        self.assertEqual(image_ids.size, 3)
        self.assertEqual(list(values["sky_color"]), [4.0, 2.0, 4.0])

    def test_the_binding_decides_which_way_round(self):
        """The same expression, the channels swapped, is the reciprocal."""

        with start_db_session() as db_session:
            values, _ = get_quantity_values(
                self.key(("B", "R")),
                {"sky_color": ("B", "R")},
                {"sky_color": "bg_center[1] / bg_center[2]"},
                db_session,
            )

        self.assertEqual(list(values["sky_color"]), [0.25, 0.5, 0.25])

    def test_an_aggregate_spans_one_channel(self):
        """Each slot is read on its own, so a median is per channel.

        Were the two pooled, the median of all six values would be 5,
        making this 0 rather than 4, and nothing would say so.
        """

        with start_db_session() as db_session:
            values, _ = get_quantity_values(
                self.key(("R", "B")),
                {"relative": ("R", "B")},
                {
                    "relative": (
                        "nanmedian(bg_center[1]) - nanmedian(bg_center[2])"
                    )
                },
                db_session,
            )

        self.assertEqual(list(values["relative"]), [4.0, 4.0, 4.0])

    def test_a_channel_never_recorded_is_undefined(self):
        """Not an error: the expression is simply NaN throughout.

        One library is shared by every project, so an expression naming a
        channel this camera does not have is an ordinary thing to meet.
        """

        with start_db_session() as db_session:
            values, _ = get_quantity_values(
                self.key(("R", self.missing_channel)),
                {"sky_color": ("R", self.missing_channel)},
                {"sky_color": "bg_center[1] / bg_center[2]"},
                db_session,
            )

        self.assertTrue(numpy.all(numpy.isnan(values["sky_color"])))

    def test_two_axes_of_different_kinds_resolve_in_one_query(self):
        """A plain diagnostic against a cross-channel expression.

        Counting the statements as well as checking the numbers, because
        one read of the union is why both axes are asked for at once: two
        axes over the same diagnostic in overlapping channels must not
        become two reads of it, and a figure draws many series.
        """

        issued = []

        def record(_conn, _cursor, statement, *_args):
            issued.append(statement)

        with start_db_session() as db_session:
            event.listen(db_session.bind, "before_cursor_execute", record)
            try:
                values, image_ids = get_quantity_values(
                    self.key(("R", "B")),
                    {"bg_center": ("R",), "sky_color": ("R", "B")},
                    {"sky_color": "bg_center[1] / bg_center[2]"},
                    db_session,
                )
            finally:
                event.remove(db_session.bind, "before_cursor_execute", record)

        self.assertEqual(list(values["bg_center"]), self.values_of["R"])
        self.assertEqual(list(values["sky_color"]), [4.0, 2.0, 4.0])
        self.assertEqual(image_ids.size, 3)
        self.assertEqual(
            len(issued), 1, f"both axes should share one read: {issued}"
        )


class TestTiedJulianDates(unittest.TestCase):
    """Two images of one session sharing a ``jd``.

    Alignment is by position, so the order images come back in has to be
    total.  Ordering by ``jd`` alone leaves a tie for the database to break
    however it likes, and two queries breaking one differently would pair a
    value with the wrong image -- silently, since both plots look fine.  The
    fixture here is the smallest thing that would expose it: two frames at
    the same instant with values that say which is which.
    """

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

    #: ``bg_center`` of each image, by id, so a mispairing is visible.
    value_of = {}

    @classmethod
    def _fill_database(cls):
        """Four frames, two of which share a Julian date."""

        # False positive: the declarative models are callable.
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
            image_type_id = db_session.execute(select(ImageType.id)).scalar()

            session = ObservingSession(
                observer_id=1,
                camera_id=1,
                telescope_id=1,
                mount_id=1,
                observatory_id=1,
                target_id=1,
                label="tied",
                start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
                end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
            )
            db_session.add(session)
            db_session.flush()

            # The middle two share a jd.
            tied_jds = [2460000.5, 2460000.6, 2460000.6, 2460000.7]
            for index, jd in enumerate(tied_jds):
                image = Image(
                    raw_fname=f"/data/raw/tied_{index}.fits",
                    image_type_id=image_type_id,
                    observing_session_id=session.id,
                    jd=jd,
                )
                db_session.add(image)
                db_session.flush()
                value = 10.0 * index
                cls.value_of[image.id] = value
                db_session.add(
                    ImageDiagnostics(
                        image_id=image.id,
                        channel="R",
                        diagnostic_id=diagnostic_id,
                        value=value,
                    )
                )
            cls.session_id = session.id
        # pylint: enable=not-callable

    def test_values_stay_with_their_own_images(self):
        """The assertion the tiebreak exists for."""

        key = SeriesKey(self.session_id, "object", ("R",))
        with start_db_session() as db_session:
            values, image_ids = get_diagnostic_values(
                key, {"bg_center": {("R",)}}, db_session
            )

        self.assertEqual(
            list(values["bg_center"][("R",)]),
            [self.value_of[image_id] for image_id in image_ids],
        )

    def test_the_order_is_repeatable(self):
        """Two calls must not disagree about which image comes first."""

        key = SeriesKey(self.session_id, "object", ("R",))
        with start_db_session() as db_session:
            first, _ = get_canonical_images(key, db_session)
            _, second_ids = get_diagnostic_values(
                key, {"bg_center": {("R",)}}, db_session
            )

        self.assertEqual(list(first), list(second_ids))


if __name__ == "__main__":
    unittest.main()
