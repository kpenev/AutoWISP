"""Tests for offering and ranking the candidates for a single photref.

The photometric reference selection page offers the images of a photref
group that still need a reference, best first by a merit expression from the
project's library. These check which entries count as still needing one, which
a newly chosen one reports binding, and how the offered ones are ranked, on
the throwaway project the binding tests
use: one camera with channels ``R`` and ``G``, and images ``near``, ``far``
and ``longer``.
"""

import unittest
from os import path

import numpy
from sqlalchemy import select

from autowisp.database.image_processing import find_raw_image_id
from autowisp.database.interface import start_db_session
from autowisp.database.photref_selection import (
    bind_images_to_photref,
    get_merit_expressions,
    get_unbound_entries,
    rank_photref_candidates,
)

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Image,
    ImageType,
    MasterType,
    ObservingSession,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.expression_library import (
    default_expressions,
    photref_merit,
)
from autowisp.tests.test_photref_binding import PhotrefBindingProject


class TestUnboundEntries(PhotrefBindingProject):
    """Which entries of a photref group still need a reference."""

    def test_bindings_are_checked_per_channel(self):
        """An image bound in one channel still needs its other one bound.

        As under a ``must_match`` putting both channels in one group. With
        ``near`` bound in ``R`` and ``far`` in ``G``, the group's first entry
        being in ``R`` must not decide the channel checked for all.
        """

        self.bind_in_advance("near", "R")
        self.bind_in_advance("far", "G")
        with start_db_session() as db_session:
            entries = [
                (db_session.get(Image, self._image_ids[name]), channel, None)
                for name, channel in [
                    ("near", "R"),
                    ("near", "G"),
                    ("far", "R"),
                    ("far", "G"),
                    ("longer", "R"),
                ]
            ]
            unbound = get_unbound_entries(
                entries,
                db_session.scalar(
                    select(MasterType.id).filter_by(name="single_photref")
                ),
                db_session,
            )
            name_of = {
                image_id: name for name, image_id in self._image_ids.items()
            }
            unbound = [
                (name_of[image.id], channel) for image, channel, _ in unbound
            ]

        self.assertEqual(
            unbound, [("near", "G"), ("far", "R"), ("longer", "R")]
        )


class TestRecordedBindings(PhotrefBindingProject):
    """What binding a group to a newly chosen photref reports binding."""

    def test_the_entries_bound_are_returned(self):
        """Exactly those written: the far image stays unbound.

        The page drops what this returns from the group, so an entry
        returned but not bound would never be offered a reference again.
        """

        returned = bind_images_to_photref(
            self._photref_fname,
            [
                (f"{name}.fits", f"{name}_R.h5", self._image_ids[name], "R")
                for name in ("near", "far")
            ],
        )
        with start_db_session() as db_session:
            bindings = self.bindings(db_session)

        self.assertEqual(returned, [(self._image_ids["near"], "R")])
        self.assertEqual(bindings, {("near", "R"): self._photref_id})


class TestFindRawImage(PhotrefBindingProject):
    """Which image the RAWFNAME of a DR header names."""

    #: Raw file names stored besides the fixture's, by what they test, in the
    #: order they are added, each joined with the system's own separator (a
    #: backslash on Windows, which a pattern assuming ``/`` never matched).
    #: The lookalike comes first: a LIKE pattern takes ``_`` for any
    #: character, so a lookup taking the first match of one finds it instead
    #: of frame_1.
    _stored = {
        label: path.join("data", "RAW", fname)
        for label, fname in [
            ("lookalike", "frameX1.fits"),
            ("extended", "frame_10.fits"),
            ("target", "frame_1.fits.fz"),
        ]
    }

    @classmethod
    def _fill_database(cls):
        """Add the images with the names above."""

        super()._fill_database()
        with start_db_session() as db_session:
            cls._stored_ids = {}
            for label, raw_fname in cls._stored.items():
                # False positive: the declarative models are callable.
                # pylint: disable-next=not-callable
                image = Image(
                    raw_fname=raw_fname,
                    image_type_id=db_session.scalar(
                        select(ImageType.id).filter_by(name="object")
                    ),
                    observing_session_id=db_session.scalar(
                        select(ObservingSession.id)
                    ),
                    jd=2460005.5,
                )
                db_session.add(image)
                db_session.flush()
                cls._stored_ids[label] = image.id

    def test_each_name_finds_its_own_image(self):
        """Exactly the image named: not one its name is part of, or like."""

        with start_db_session() as db_session:
            found = {
                raw_fname_keyword: find_raw_image_id(
                    raw_fname_keyword, db_session
                )
                for raw_fname_keyword in ("frame_1", "frame", "nope")
            }

        self.assertEqual(
            found,
            {
                "frame_1": self._stored_ids["target"],
                "frame": None,
                "nope": None,
            },
        )


class TestPhotrefRanking(PhotrefBindingProject):
    """The offered candidates of a group, best first by the default merit."""

    #: ``(image name, channel): (s_center, bg_center)`` of the group, in the
    #: order of the batch. ``near`` is a member in both channels. ``longer``
    #: is not offered, and its outlying values would change every rank if
    #: they leaked into the population. ``nobg`` has no background, so no
    #: merit.
    _readings = {
        ("near", "R"): (3.0, 10.0),
        ("near", "G"): (2.0, 10.0),
        ("far", "R"): (1.0, 30.0),
        ("longer", "R"): (100.0, 0.0),
        ("nobg", "R"): (5.0, None),
    }

    #: The default merit over the offered entries, worked out by hand:
    #: ``s_center`` ranks 0.75, 0.5, 0.25, 1 and ``bg_center`` 0.5, 0.5, 1
    #: among the finite three.
    _merits = {
        ("near", "R"): 1.0 / (0.25**2 + 0.5**2),
        ("near", "G"): 1.0 / (0.5**2 + 0.5**2),
        ("far", "R"): 1.0 / (0.75**2 + 1.0**2),
        ("nobg", "R"): numpy.nan,
    }

    @classmethod
    def _fill_database(cls):
        """Add the ``nobg`` image and the readings the merit uses."""

        super()._fill_database()
        with start_db_session() as db_session:
            cls._image_ids["nobg"] = cls._add_raw_image(
                db_session,
                "nobg",
                db_session.scalar(select(ObservingSession.id)),
                30.0,
            )
            for (name, channel), (s_center, bg_center) in cls._readings.items():
                cls._add_diagnostics(
                    db_session,
                    cls._image_ids[name],
                    channel,
                    {
                        diagnostic: value
                        for diagnostic, value in [
                            ("s_center", s_center),
                            ("bg_center", bg_center),
                        ]
                        if value is not None
                    },
                )

    def _rank(self, merit):
        """Return the ranked ``(name, channel)`` entries and the values."""

        batch = [
            (
                f"{name}.fits",
                f"{name}_{channel}.h5",
                self._image_ids[name],
                channel,
            )
            for name, channel in self._readings
        ]
        with start_db_session() as db_session:
            positions, values = rank_photref_candidates(
                batch,
                merit,
                [name != "longer" for name, _ in self._readings],
                db_session,
            )
        entries = list(self._readings)
        return [entries[position] for position in positions], values

    def test_the_seeded_merit_ranks_the_offered_entries(self):
        """Best first, no merit last, ranked among the offered alone."""

        ranked, values = self._rank(photref_merit)

        self.assertEqual(
            ranked, [("near", "R"), ("near", "G"), ("far", "R"), ("nobg", "R")]
        )
        numpy.testing.assert_allclose(
            values[photref_merit], [self._merits[entry] for entry in ranked]
        )
        numpy.testing.assert_array_equal(
            values["s_center"],
            [self._readings[entry][0] for entry in ranked],
        )

    def test_every_diagnostic_recorded_for_the_group_is_returned(self):
        """Those of the members' images, not of the reference's own."""

        self.assertEqual(
            set(self._rank(photref_merit)[1]),
            {"bg_center", "dec_center", "ra_center", "s_center", photref_merit},
        )

    def test_without_a_merit_the_order_is_by_time(self):
        """Every image shares one Julian date, so by image, then channel."""

        ranked, values = self._rank(None)

        self.assertEqual(
            ranked, [("near", "G"), ("near", "R"), ("far", "R"), ("nobg", "R")]
        )
        self.assertNotIn(photref_merit, values)


class TestMeritExpressions(unittest.TestCase):
    """Which library expressions candidates can be ranked by."""

    def test_only_one_value_per_entry_qualifies(self):
        """At most one channel, and no photometry."""

        library = {
            photref_merit: default_expressions[photref_merit]["expression"],
            "colour": "s_center[0] - s_center[1]",
            "night_bg": "nanmedian(bg_center['R'])",
            "offset": "photometry_mag_offset[0][0]",
        }

        self.assertEqual(
            get_merit_expressions(library), ["night_bg", photref_merit]
        )


if __name__ == "__main__":
    unittest.main()
