"""Tests for photometry slots in diagnostic expressions.

The second subscript, ``magfit_residual[0][1]``, binding the photometries of
a quantity read per photometry as the first binds its channels. Kept apart
from :mod:`autowisp.tests.test_expressions` only for length; the fixture is
its :class:`~autowisp.tests.test_expressions.SlotTestCase`, with a
per-photometry diagnostic added.
"""

import unittest

import numpy

from autowisp.diagnostics.diagnostic_types import (
    shapefit_photometry,
    time_quantity,
)
from autowisp.diagnostics.expressions import (
    check_expression,
    check_rule,
    get_channel_arity,
    get_indexed_names,
    get_photometry_arity,
    get_photometry_parameters,
)
from autowisp.tests.test_expressions import SlotTestCase


class TestPhotometrySlots(SlotTestCase):
    """Photometries, bound through a second subscript as channels are
    through the first.

    The library adds to the channel one: two photometries of one channel;
    one formal photometry against a quoted aperture; an expression passing
    its photometries on the other way round; one quoting both its channel
    and its photometry, the shape fit; and a channel-free one taking a
    photometry, and one reading that.
    """

    library = dict(
        SlotTestCase.library,
        apertures="magfit_residual[0][0] - magfit_residual[0][1]",
        against_ap4="magfit_residual[0][0] - magfit_residual[0]['ap4']",
        swapped="apertures[1][1, 0] + bg_center[1]",
        g_shapefit="magfit_residual['G0']['shapefit']",
        g_residual="magfit_residual['G0'][0]",
        doubled_g="g_residual[()][0] * 2",
    )

    #: ``magfit_residual`` per channel and photometry, distinct everywhere so
    #: that reading the wrong one cannot pass by coincidence.
    residual = {
        (channel, photometry): numpy.array([1.0, 2.0])
        * (10 * index + photometry + 2)
        for index, channel in enumerate(("B", "R", "G0"))
        for photometry in (shapefit_photometry, 0, 2, 3, 4)
    }

    #: One of each shape above, all wanted at once.
    wanted = {
        "swapped": {(("R",), (2, shapefit_photometry))},
        "against_ap4": {(("B",), (0,))},
        "g_shapefit": {((), ())},
        "doubled_g": {((), (3,))},
    }

    def fetch(self, wanted):
        """As for any diagnostic, with ``magfit_residual`` per photometry."""

        fetched = super().fetch(wanted)
        if "magfit_residual" in fetched:
            fetched["magfit_residual"] = {
                binding: self.residual[binding[0][0], binding[1][0]]
                for binding in fetched["magfit_residual"]
            }
        return fetched

    def test_parameters_and_arity(self):
        """Derived from the second subscripts, as channels from the first."""

        for name, parameters, arity in (
            ("apertures", (0, 1), 2),
            ("against_ap4", (0,), 1),
            ("swapped", (0, 1), 2),
            ("g_shapefit", (), 0),
            ("g_residual", (0,), 1),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    get_photometry_parameters(self.library[name]), parameters
                )
                self.assertEqual(
                    get_photometry_arity(name, self.library), arity
                )
        for name, arity in (
            ("magfit_residual", 1),
            ("bg_center", 0),
            (time_quantity, 0),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    get_photometry_arity(name, self.library), arity
                )
        self.assertEqual(get_channel_arity("g_residual", self.library), 0)

    def test_reads_carry_their_photometries(self):
        """The second subscript is part of the read, not a read of its own."""

        self.assertEqual(
            get_indexed_names(self.library["swapped"]),
            [("apertures", (1,), (1, 0)), ("bg_center", (1,), ())],
        )
        self.assertEqual(
            get_indexed_names(self.library["doubled_g"]),
            [("g_residual", (), (0,))],
        )

    def test_needed_in_every_photometry_read(self):
        """Formal ones bound, quoted ones read as the ids they spell."""

        self.assertEqual(
            self.needed(self.wanted),
            {
                "magfit_residual": {
                    (("R",), (shapefit_photometry,)),
                    (("R",), (2,)),
                    (("B",), (0,)),
                    (("B",), (4,)),
                    (("G0",), (shapefit_photometry,)),
                    (("G0",), (3,)),
                },
                "bg_center": {(("R",), ())},
            },
        )

    def test_evaluated_in_the_photometries_bound(self):
        """``swapped`` at ``(2, shapefit)`` is ``apertures`` at the reverse."""

        drawn = self.evaluate(self.wanted)

        for quantity, binding, expected in (
            (
                "swapped",
                (("R",), (2, shapefit_photometry)),
                self.residual["R", shapefit_photometry]
                - self.residual["R", 2]
                + self.bg["R"],
            ),
            (
                "against_ap4",
                (("B",), (0,)),
                self.residual["B", 0] - self.residual["B", 4],
            ),
            (
                "g_shapefit",
                ((), ()),
                self.residual["G0", shapefit_photometry],
            ),
            ("doubled_g", ((), (3,)), 2 * self.residual["G0", 3]),
        ):
            with self.subTest(quantity=quantity):
                numpy.testing.assert_allclose(
                    drawn[quantity][binding], expected
                )

    def test_the_library_accepts_them(self):
        """Every shape above is a valid expression."""

        for name in self.wanted:
            with self.subTest(name=name):
                self.assertEqual(
                    check_expression(name, self.library[name], self.library),
                    [],
                )

    def test_a_read_must_match_what_it_reads(self):
        """A photometry subscript exactly where one is taken, naming one."""

        for text in (
            "magfit_residual[0]",
            "bg_center[0][0]",
            "apertures[0][0]",
            "magfit_residual[0]['ap']",
            "g_residual",
            "g_residual[0][0]",
        ):
            with self.subTest(text=text):
                self.assertTrue(check_expression("bad", text, self.library))

    def test_a_rule_takes_at_most_one_photometry_slot(self):
        """The other photometries a rule reads are quoted."""

        self.assertEqual(
            check_rule(
                "magfit_residual[0][0] - magfit_residual[0]['ap4'] > 1",
                self.library,
            ),
            [],
        )
        problems = check_rule("apertures[0][0, 1] > 1", self.library)
        self.assertEqual(len(problems), 1)
        self.assertIn("photometry", problems[0])


if __name__ == "__main__":
    unittest.main()
