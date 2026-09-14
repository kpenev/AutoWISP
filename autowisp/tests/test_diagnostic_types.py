"""Tests for the diagnostic vocabulary.

The catalogue was moved out of ``_init_diagnostic_types()`` so that an
expression can be validated without opening a project.  The risk that
creates is drift: the seeder and the catalogue are now two things that
must agree, where before they were one.  :class:`TestSeeding` is the test
that keeps them honest, and is the reason the rest of this file exists.
"""

import tempfile
import unittest

from autowisp.database.initialize_database import _init_diagnostic_types
from autowisp.database.interface import set_project_home, start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import DiagnosticType

# pylint: enable=no-name-in-module
from autowisp.diagnostics.diagnostic_types import (
    is_diagnostic,
    is_known_quantity,
    is_quantile_diagnostic,
    standard_diagnostic_names,
    standard_diagnostic_types,
    time_quantity,
)


class TestCatalogue(unittest.TestCase):
    """The shape and content of the static catalogue."""

    def test_it_cannot_be_mutated(self):
        """A shared catalogue one caller can edit is not shared, it is a bug."""

        with self.assertRaises(TypeError):
            standard_diagnostic_types()["invented"] = "nonsense"

        self.assertIsInstance(standard_diagnostic_names(), frozenset)

    def test_names_are_exactly_the_mapping_keys(self):
        """The two accessors cannot disagree about what exists."""

        self.assertEqual(
            standard_diagnostic_names(), set(standard_diagnostic_types())
        )

    def test_descriptions_are_fully_interpolated(self):
        """No description may contain an unexpanded placeholder.

        Guards a real defect the extraction fixed: the continuation line of
        the ``*_map_residual`` description was not an f-string, so every one
        of them was seeded reading "and smoothed {param.upper()} map"
        literally.  A missing ``f`` prefix is silent, so only a test of the
        rendered text catches it coming back.
        """

        for name, description in standard_diagnostic_types().items():
            with self.subTest(diagnostic=name):
                self.assertNotIn("{", description)
                self.assertTrue(description.strip())

    def test_it_is_not_empty(self):
        """A catalogue that silently emptied would pass every other test."""

        self.assertGreater(len(standard_diagnostic_types()), 10)


class TestRuntimePatterns(unittest.TestCase):
    """The names that are described rather than listed.

    ``is_quantile_diagnostic`` is the single definition of what a quantile
    is called: ``_save_image_diagnostics`` asks it before creating a row,
    and expression validation asks it before accepting a name. The two used
    to disagree -- a loose ``startswith("pixel_q")`` against an anchored
    pattern -- which is exactly the drift one predicate prevents.
    """

    def test_it_matches_quantile_diagnostics(self):
        """``calibrate`` names them ``pixel_q`` followed by the percentile."""

        for name in ("pixel_q1", "pixel_q99", "pixel_q999"):
            with self.subTest(name=name):
                self.assertTrue(is_quantile_diagnostic(name))
                self.assertTrue(is_diagnostic(name))

    def test_it_does_not_swallow_plausible_names(self):
        """The anchor and the required digits both matter.

        Without them ``pixel_quality`` would be taken for a quantile, and a
        real diagnostic by that name could then never be referenced -- an
        expression naming it would be told it is shadowing something.
        """

        for name in ("pixel_quality", "pixel_q", "pixel_q9x", "bg_center"):
            with self.subTest(name=name):
                self.assertFalse(is_quantile_diagnostic(name))


class TestVocabulary(unittest.TestCase):
    """What an expression may reference, without opening any project."""

    def test_catalogue_entries_are_diagnostics(self):
        """The seeded names, recorded or not in any particular project."""

        for name in ("bg_center", "astrom_residual", "diagonal_fov"):
            with self.subTest(name=name):
                self.assertTrue(is_diagnostic(name))

    def test_time_is_a_quantity_but_not_a_diagnostic(self):
        """``jd`` is a column of the image row, not an ``image_diagnostics``
        value, yet it is a variable in the same flat name space."""

        self.assertFalse(is_diagnostic(time_quantity))
        self.assertTrue(is_known_quantity(time_quantity))

    def test_invented_names_are_neither(self):
        """The vocabulary is closed, which is what makes it checkable."""

        for name in ("bg_centre", "no_such_thing", ""):
            with self.subTest(name=name):
                self.assertFalse(is_known_quantity(name))


class TestSeeding(unittest.TestCase):
    """The catalogue and the rows it seeds must stay identical."""

    def test_seeded_rows_match_the_catalogue(self):
        """The whole point of the extraction, and the only silent failure.

        Descriptions are compared as well as names: they are what the
        catalogue carries beyond what validation needs, so a mismatch there
        is exactly the drift that splitting the two invites.
        """

        with tempfile.TemporaryDirectory() as project_home:
            set_project_home(project_home)
            _init_diagnostic_types()
            with start_db_session() as db_session:
                seeded = {
                    row.name: row.description
                    for row in db_session.query(DiagnosticType).all()
                }

        self.assertEqual(seeded, dict(standard_diagnostic_types()))


if __name__ == "__main__":
    unittest.main()
