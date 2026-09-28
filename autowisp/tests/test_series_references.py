"""Tests for splitting a series by the photometric reference its images used.

A diagnostic ``fit_magnitudes`` produces depends on the reference an image
was fit against as well as on the image, so a series reading one is
restricted to the images bound to one reference in each channel it reads it
in. These cover the key carrying those references, the restriction and the
guard refusing to read magfit diagnostics without one, and finding which
reference populations a session holds.

The fixture is one night fit against two references, ``ref1`` and ``ref2``,
in both of its channels, ``R`` and ``B``: the first three frames against
``ref1``, the next two against ``ref2``, one frame against ``ref2`` in ``R``
but ``ref1`` in ``B`` -- rare, and exactly what a split has to keep apart --
and a last frame magfit-ed but bound to nothing, as processing before every
magfit-ed image was bound could leave one.
"""

import tempfile
import unittest
from datetime import datetime

from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageMasterSelection,
    ImageType,
    MasterFile,
    MasterType,
    ObservingSession,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    count_unbound_images,
    get_canonical_images,
    get_quantity_values,
    split_series,
)
from autowisp.diagnostics.image_counts import (
    count_images_with_all,
    count_images_with_channels,
)
from autowisp.exceptions import PipelineError


class TestSeriesKeyReferences(unittest.TestCase):
    """What a key accepts for its references, and what it makes of them."""

    def test_omitted_is_none_in_every_slot(self):
        """A key built as before equals the one spelling the Nones out."""

        key = SeriesKey(3, "object", ("R", "B"))

        self.assertEqual(key.photrefs, (None, None))
        self.assertEqual(key, SeriesKey(3, "object", ("R", "B"), (None, None)))
        self.assertEqual(key.reference_pairs, ())

    def test_one_reference_per_slot(self):
        """A length differing from the channels' is refused."""

        with self.assertRaises(ValueError):
            SeriesKey(3, "object", ("R", "B"), (12,))

    def test_coerced_to_a_hashable_key(self):
        """References posted as a list of strings still make a usable key."""

        key = SeriesKey(3, "object", ["R"], ["12"])

        self.assertEqual(key.photrefs, (12,))
        self.assertEqual({key: 1}[SeriesKey(3, "object", ("R",), (12,))], 1)

    def test_a_channel_carries_one_reference(self):
        """A slot left None takes the reference another slot on its channel has.

        What makes a producer filling only the slots that read magfit build
        the same key as one filling every slot on the channel.
        """

        self.assertEqual(
            SeriesKey(3, "object", ("R", "R", "B"), (12, None, None)).photrefs,
            (12, 12, None),
        )

    def test_two_references_on_a_channel_are_left_alone(self):
        """Nothing could be meant by filling from one of them."""

        key = SeriesKey(3, "object", ("R", "R", "R"), (12, 15, None))

        self.assertEqual(key.photrefs, (12, 15, None))
        self.assertEqual(key.reference_pairs, (("R", 12), ("R", 15)))

    def test_replace_goes_through_the_checks(self):
        """Replacing the channels alone leaves the references too short."""

        key = SeriesKey(3, "object", ("R",))

        self.assertEqual(
            key._replace(photrefs=[12]).photrefs,  # pylint: disable=no-member
            (12,),
        )
        with self.assertRaises(ValueError):
            key._replace(channels=("R", "B"))  # pylint: disable=no-member


class ReferenceProject(unittest.TestCase):
    """The fixture the classes below share, holding no tests of its own.

    Its own project rather than the shared one: references, bindings and
    magfit diagnostics there would change what every other series test
    sees.
    """

    #: The reference each frame is bound to, per channel; ``None`` for the
    #: frame magfit-ed but bound to nothing. The sixth is bound differently
    #: in the two channels.
    bound_to = {
        "R": ["ref1", "ref1", "ref1", "ref2", "ref2", "ref2", None],
        "B": ["ref1", "ref1", "ref1", "ref2", "ref2", "ref1", None],
    }

    #: ``photometry_mag_offset`` per channel and frame, distinct everywhere
    #: so that a value landing on the wrong image cannot pass by luck.
    offsets = {
        "R": [1.0, 2.0, 3.0, 10.0, 20.0, 50.0, 99.0],
        "B": [4.0, 5.0, 6.0, 30.0, 40.0, 60.0, 98.0],
    }

    #: ``bg_center`` in ``R``, which does not depend on the reference.
    backgrounds = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]

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

    @classmethod
    def _fill_database(cls):
        """One night of seven object frames in two channels, two references.

        Provenance foreign keys, and the master type's condition, are left
        dangling, as SQLite does not enforce them and nothing here reads
        them.
        """

        # False positive: the declarative models are callable.
        # pylint: disable=not-callable
        with start_db_session() as db_session:
            diagnostic_ids = {}
            for name in ("bg_center", "photometry_mag_offset"):
                diagnostic = DiagnosticType(name=name, description=name)
                db_session.add(diagnostic)
                db_session.flush()
                diagnostic_ids[name] = diagnostic.id

            image_type = ImageType(name="object", description="objects")
            photref_type = MasterType(
                name="single_photref", condition_id=1, description="photref"
            )
            session = ObservingSession(
                observer_id=1,
                camera_id=1,
                telescope_id=1,
                mount_id=1,
                observatory_id=1,
                target_id=1,
                label="split_night",
                start_time_utc=datetime(2023, 3, 1, 20, 0, 0),
                end_time_utc=datetime(2023, 3, 1, 23, 0, 0),
            )
            db_session.add_all([image_type, photref_type, session])
            db_session.flush()
            cls.session_id = session.id

            #: ``{(reference, channel): MasterFile id}``.
            cls.photref = {}
            for reference in ("ref1", "ref2"):
                for channel in ("R", "B"):
                    master = MasterFile(
                        type_id=photref_type.id,
                        filename=f"/photref/{reference}_{channel}.h5",
                        enabled=True,
                    )
                    db_session.add(master)
                    db_session.flush()
                    cls.photref[reference, channel] = master.id

            cls.image_ids = []
            for index, background in enumerate(cls.backgrounds):
                image = Image(
                    raw_fname=f"/data/raw/frame_{index}.fits",
                    image_type_id=image_type.id,
                    observing_session_id=session.id,
                    jd=2460005.5 + 0.05 * index,
                )
                db_session.add(image)
                db_session.flush()
                cls.image_ids.append(image.id)

                db_session.add(
                    ImageDiagnostics(
                        image_id=image.id,
                        channel="R",
                        diagnostic_id=diagnostic_ids["bg_center"],
                        value=background,
                    )
                )
                for channel, values in cls.offsets.items():
                    db_session.add(
                        ImageDiagnostics(
                            image_id=image.id,
                            channel=channel,
                            diagnostic_id=diagnostic_ids[
                                "photometry_mag_offset"
                            ],
                            value=values[index],
                        )
                    )
                    reference = cls.bound_to[channel][index]
                    if reference is not None:
                        db_session.add(
                            ImageMasterSelection(
                                image_id=image.id,
                                channel=channel,
                                master_type_id=photref_type.id,
                                master_file_id=cls.photref[reference, channel],
                            )
                        )
        # pylint: enable=not-callable

    def key(self, channels, *references):
        """Return the night's series binding *channels* to *references*.

        Each reference is ``"ref1"``, ``"ref2"`` or ``None``, per slot, and
        is looked up in that slot's channel.
        """

        return SeriesKey(
            self.session_id,
            "object",
            channels,
            (
                tuple(
                    (
                        None
                        if reference is None
                        else self.photref[reference, channel]
                    )
                    for channel, reference in zip(channels, references)
                )
                if references
                else None
            ),
        )

    def selected(self, values, **references):
        """Return the *values* of the frames bound as *references* say.

        Args:
            values(list):    One entry per frame, in order.

            references:    ``channel=reference`` for every channel the
                frames have to be bound in, ``reference`` possibly ``None``.
        """

        return [
            value
            for index, value in enumerate(values)
            if all(
                self.bound_to[channel][index] == reference
                for channel, reference in references.items()
            )
        ]


class TestReferenceRestriction(ReferenceProject):
    """A key naming references reads only the images fit against them."""

    def test_the_image_list_is_the_reference_population(self):
        """The canonical list itself shrinks, so every array does too."""

        with start_db_session() as db_session:
            image_ids, _ = get_canonical_images(
                self.key(("R",), "ref2"), db_session
            )

        self.assertEqual(
            list(image_ids), self.selected(self.image_ids, R="ref2")
        )

    def test_an_aggregate_sees_one_reference(self):
        """A median taken over both references would be neither's."""

        with start_db_session() as db_session:
            values, image_ids = get_quantity_values(
                self.key(("R",), "ref1"),
                {"rel_offset": {("R",)}},
                {
                    "rel_offset": "photometry_mag_offset[0]"
                    " - nanmedian(photometry_mag_offset[0])"
                },
                db_session,
            )

        self.assertEqual(
            list(image_ids), self.selected(self.image_ids, R="ref1")
        )
        self.assertEqual(list(values["rel_offset"][("R",)]), [-1.0, 0.0, 1.0])

    def test_a_magfit_read_needs_a_reference(self):
        """Without one, the series would mix the night's two references."""

        with (
            start_db_session() as db_session,
            self.assertRaises(PipelineError),
        ):
            get_quantity_values(
                self.key(("R",)),
                {"photometry_mag_offset": {("R",)}},
                {},
                db_session,
            )

    def test_a_reference_is_needed_in_every_channel_read(self):
        """One in R does not cover the magfit read in B."""

        with (
            start_db_session() as db_session,
            self.assertRaises(PipelineError),
        ):
            get_quantity_values(
                self.key(("R", "B"), "ref1", None),
                {"colour": {("R", "B")}},
                {
                    "colour": "photometry_mag_offset[0]"
                    " - photometry_mag_offset[1]"
                },
                db_session,
            )

    def test_a_reference_restricts_what_does_not_need_one(self):
        """The background of the frames fit against ref2, say."""

        with start_db_session() as db_session:
            values, image_ids = get_quantity_values(
                self.key(("R",), "ref2"),
                {"bg_center": {("R",)}},
                {},
                db_session,
            )

        self.assertEqual(
            list(image_ids), self.selected(self.image_ids, R="ref2")
        )
        self.assertEqual(
            list(values["bg_center"][("R",)]),
            self.selected(self.backgrounds, R="ref2"),
        )

    def test_a_quoted_read_takes_its_reference_from_a_tail_slot(self):
        """The quoted channel binds nothing, but still needs its reference.

        The quantity takes no parameter, so its only slot is the tail one
        for the channel it quotes.
        """

        with start_db_session() as db_session:
            values, image_ids = get_quantity_values(
                self.key(("B",), "ref1"),
                {"offset_b": {()}},
                {"offset_b": "photometry_mag_offset['B']"},
                db_session,
            )

        self.assertEqual(
            list(image_ids), self.selected(self.image_ids, B="ref1")
        )
        self.assertEqual(
            list(values["offset_b"][()]),
            self.selected(self.offsets["B"], B="ref1"),
        )

    def test_a_quoted_read_without_its_slot_is_refused(self):
        """No tail slot means no reference for the quoted channel."""

        with (
            start_db_session() as db_session,
            self.assertRaises(PipelineError),
        ):
            get_quantity_values(
                self.key(()),
                {"offset_b": {()}},
                {"offset_b": "photometry_mag_offset['B']"},
                db_session,
            )

    def test_two_references_on_one_channel_draw_nothing(self):
        """No image is bound to both, since each has one per channel."""

        with start_db_session() as db_session:
            image_ids, _ = get_canonical_images(
                self.key(("R", "R"), "ref1", "ref2"), db_session
            )

        self.assertEqual(image_ids.size, 0)


class TestSplitSeries(ReferenceProject):
    """Finding the reference populations a session holds."""

    #: Read in both channels, so split by the references of both.
    colour = {"colour": "photometry_mag_offset[0] - photometry_mag_offset[1]"}

    def split(self, key, wanted, expressions=None):
        """Return :func:`split_series` for *key*, in a session of its own."""

        with start_db_session() as db_session:
            return split_series(key, wanted, expressions or {}, db_session)

    def test_one_key_per_reference(self):
        """The night splits into the frames fit against ref1 and ref2."""

        self.assertEqual(
            self.split(self.key(("R",)), {"photometry_mag_offset": {("R",)}}),
            [self.key(("R",), "ref1"), self.key(("R",), "ref2")],
        )

    def test_a_frame_fit_differently_per_channel_is_its_own_population(self):
        """Split by R and B, the frame fit against ref2 and ref1 stands alone.

        Neither with the frames fit against ref1 in both channels, nor with
        those fit against ref2 in both: an aggregate over either would mix
        references in one of its channels. And only combinations some frame
        has come back -- never ref1 in R with ref2 in B.
        """

        keys = self.split(
            self.key(("R", "B")), {"colour": {("R", "B")}}, self.colour
        )

        self.assertEqual(
            keys,
            [
                self.key(("R", "B"), "ref1", "ref1"),
                self.key(("R", "B"), "ref2", "ref1"),
                self.key(("R", "B"), "ref2", "ref2"),
            ],
        )
        with start_db_session() as db_session:
            image_ids, _ = get_canonical_images(keys[1], db_session)
        self.assertEqual(
            list(image_ids), self.selected(self.image_ids, R="ref2", B="ref1")
        )

    def test_each_key_draws_its_own_population(self):
        """What the engine does with the keys: one evaluation each."""

        wanted = {"photometry_mag_offset": {("R",)}}
        with start_db_session() as db_session:
            for key, reference in zip(
                split_series(self.key(("R",)), wanted, {}, db_session),
                ("ref1", "ref2"),
            ):
                with self.subTest(reference=reference):
                    values, _ = get_quantity_values(key, wanted, {}, db_session)
                    self.assertEqual(
                        list(values["photometry_mag_offset"][("R",)]),
                        self.selected(self.offsets["R"], R=reference),
                    )

    def test_a_reference_the_key_names_is_kept(self):
        """Pinned to ref1 in B, the split of R meets both of R's references.

        The frames fit against ref1 in B include the one fit against ref2 in
        R, so the split within them is not the same as pinning R too.
        """

        self.assertEqual(
            self.split(
                self.key(("R", "B"), None, "ref1"),
                {"colour": {("R", "B")}},
                self.colour,
            ),
            [
                self.key(("R", "B"), "ref1", "ref1"),
                self.key(("R", "B"), "ref2", "ref1"),
            ],
        )

    def test_every_slot_on_a_split_channel_gets_its_reference(self):
        """The background slot shares the offset slot's channel, R."""

        self.assertEqual(
            self.split(
                self.key(("R", "R")),
                {"mixed": {("R", "R")}},
                {"mixed": "photometry_mag_offset[0] - bg_center[1]"},
            ),
            [
                self.key(("R", "R"), "ref1", "ref1"),
                self.key(("R", "R"), "ref2", "ref2"),
            ],
        )

    def test_nothing_to_split_without_a_magfit_read(self):
        """The background does not depend on the reference."""

        key = self.key(("R",))

        self.assertEqual(self.split(key, {"bg_center": {("R",)}}), [key])

    def test_a_quoted_channel_needs_its_tail_slot(self):
        """Its reference would have nowhere to go."""

        with self.assertRaises(PipelineError):
            self.split(
                self.key(()),
                {"offset_b": {()}},
                {"offset_b": "photometry_mag_offset['B']"},
            )

    def test_a_quoted_channel_is_split_in_its_tail_slot(self):
        """With the slot there, it splits like any other."""

        self.assertEqual(
            self.split(
                self.key(("B",)),
                {"offset_b": {()}},
                {"offset_b": "photometry_mag_offset['B']"},
            ),
            [self.key(("B",), "ref1"), self.key(("B",), "ref2")],
        )


class TestCountUnboundImages(ReferenceProject):
    """The frames a split leaves out, for want of a binding."""

    def count(self, key, wanted, expressions=None):
        """Return :func:`count_unbound_images`, in a session of its own."""

        with start_db_session() as db_session:
            return count_unbound_images(
                key, wanted, expressions or {}, db_session
            )

    def test_the_unbound_frame_is_counted(self):
        """Magfit-ed in both channels, bound in neither: counted once."""

        self.assertEqual(
            self.count(
                self.key(("R", "B")),
                {"colour": {("R", "B")}},
                {
                    "colour": "photometry_mag_offset[0]"
                    " - photometry_mag_offset[1]"
                },
            ),
            1,
        )

    def test_it_is_in_none_of_the_split_keys(self):
        """Which is what makes it worth reporting."""

        wanted = {"photometry_mag_offset": {("R",)}}
        with start_db_session() as db_session:
            drawn = {
                image_id
                for key in split_series(
                    self.key(("R",)), wanted, {}, db_session
                )
                for image_id in get_canonical_images(key, db_session)[0]
            }

        self.assertEqual(
            drawn,
            set(self.image_ids) - set(self.selected(self.image_ids, R=None)),
        )

    def test_nothing_is_counted_without_a_magfit_read(self):
        """Nothing is split, so nothing is left out."""

        self.assertEqual(
            self.count(self.key(("R",)), {"bg_center": {("R",)}}), 0
        )

    def test_nothing_is_counted_where_the_key_names_the_reference(self):
        """There is no split to be left out of."""

        self.assertEqual(
            self.count(
                self.key(("R",), "ref1"),
                {"photometry_mag_offset": {("R",)}},
            ),
            0,
        )


class TestReferenceCounts(ReferenceProject):
    """What the series table counts: per reference, and per bound row."""

    #: What a row comparing the two channels' offsets reads.
    colour_reads = {
        ("photometry_mag_offset", "R"),
        ("photometry_mag_offset", "B"),
    }

    def test_options_are_counted_per_reference(self):
        """One count per (channel, photref) the frames are bound to.

        The frame bound differently in its two channels counts under ref2
        in R and under ref1 in B; the frame bound to nothing, under neither.
        """

        with start_db_session() as db_session:
            rows = count_images_with_all(
                {"photometry_mag_offset"}, db_session, by_reference=True
            )

        self.assertEqual(
            [row[3:] for row in rows],
            sorted(
                (
                    channel,
                    self.photref[reference, channel],
                    len(self.selected(self.image_ids, **{channel: reference})),
                )
                for channel in ("B", "R")
                for reference in ("ref1", "ref2")
            ),
        )

    def test_a_bound_row_counts_what_it_draws(self):
        """For each reference population, the images its series reads.

        Unrestricted, the count is the whole night, the frame bound to
        nothing included, which no population holds.
        """

        colour = {
            "colour": "photometry_mag_offset[0] - photometry_mag_offset[1]"
        }
        with start_db_session() as db_session:
            self.assertEqual(
                [
                    row[3]
                    for row in count_images_with_channels(
                        self.colour_reads, db_session
                    )
                ],
                [len(self.image_ids)],
            )
            for key in split_series(
                self.key(("R", "B")),
                {"colour": {("R", "B")}},
                colour,
                db_session,
            ):
                with self.subTest(photrefs=key.photrefs):
                    self.assertEqual(
                        [
                            row[3]
                            for row in count_images_with_channels(
                                self.colour_reads,
                                db_session,
                                key.reference_pairs,
                            )
                        ],
                        [get_canonical_images(key, db_session)[0].size],
                    )


if __name__ == "__main__":
    unittest.main()
