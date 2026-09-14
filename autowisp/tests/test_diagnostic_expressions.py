"""Tests for stored diagnostic expressions.

Only what is specific to the model lives here.  That it carries ``created``
and ``modified``, and that they are maintained however the row is written,
is covered once for every browser-interface model by ``test_bui_models``.
"""

import os
import sys
import unittest
from pathlib import Path

import django

_browser_interface = Path(__file__).resolve().parents[1] / "browser_interface"
if str(_browser_interface) not in sys.path:
    sys.path.insert(0, str(_browser_interface))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_project.settings")
django.setup()

# pylint: disable=wrong-import-position
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db.utils import IntegrityError

from diagnostics.expression_data import get_expressions
from diagnostics.models import DiagnosticExpression

# pylint: enable=wrong-import-position


class DiagnosticExpressionTestCase(unittest.TestCase):
    """Base migrating the throwaway browser-interface database."""

    @classmethod
    def setUpClass(cls):
        assert "autowisp_tests_" in str(
            settings.DATABASES["default"]["NAME"]
        ), "refusing to run against a real browser-interface database"
        call_command("migrate", verbosity=0)

    def setUp(self):
        # pylint: disable=no-member
        DiagnosticExpression.objects.all().delete()

    def make(self, name, expression="astrom_residual / diagonal_fov"):
        """Store one expression."""

        # pylint: disable=no-member
        return DiagnosticExpression.objects.create(
            name=name, expression=expression
        )


class TestNameSpace(DiagnosticExpressionTestCase):
    """Names have to behave like the diagnostic names they sit beside."""

    def test_name_is_unique(self):
        """Two expressions cannot share a name.

        The name is what a selector and a URL carry, and what other
        expressions reference, so a duplicate would be ambiguous in three
        places at once.
        """

        self.make("rel_astrom_residual")
        with self.assertRaises(IntegrityError):
            self.make("rel_astrom_residual", "bg_center")

    def test_name_must_survive_a_url(self):
        """Rejected unless it is a slug.

        Expressions are selected through ``image/<slug:x>/vs/<slug:y>``, so
        a name outside the slug charset could be stored but never plotted.
        """

        expression = DiagnosticExpression(
            name="not a slug!", expression="bg_center"
        )
        with self.assertRaises(ValidationError):
            expression.full_clean()

    def test_a_slug_name_is_accepted(self):
        """The names the documentation suggests actually validate."""

        for name in ("rel_astrom_residual", "bg-relative", "pixel_q999_ratio"):
            with self.subTest(name=name):
                DiagnosticExpression(
                    name=name, expression="bg_center"
                ).full_clean()


class TestStoredFields(DiagnosticExpressionTestCase):
    """What the model keeps, and what it deliberately does not check."""

    def test_description_is_optional(self):
        """Most expressions are self-explanatory from their text."""

        self.make("terse").full_clean()

    def test_unknown_names_are_not_rejected_here(self):
        """The model stores text; resolving names is not its job.

        An expression may legitimately reference diagnostics the open
        project has never recorded -- it is then simply not offered there --
        so validating against a project database at this level would be
        wrong.
        """

        self.make("references_nothing_real", "no_such_diagnostic * 2")
        # pylint: disable=no-member
        self.assertEqual(
            DiagnosticExpression.objects.get(
                name="references_nothing_real"
            ).expression,
            "no_such_diagnostic * 2",
        )

    def test_ordering_is_by_name(self):
        """The management page lists them alphabetically."""

        for name in ("zeta", "alpha", "mu"):
            self.make(name)
        # pylint: disable=no-member
        self.assertEqual(
            [row.name for row in DiagnosticExpression.objects.all()],
            ["alpha", "mu", "zeta"],
        )

    def test_str_is_the_name(self):
        """What the admin and any error message will show."""

        self.assertEqual(str(self.make("readable")), "readable")


class TestLibraryAccess(DiagnosticExpressionTestCase):
    """``get_expressions`` -- the whole of tier 3.

    What it produces is the ``{name: expression}`` dictionary tiers 1 and 2
    take as an argument, so these assert the *shape* of that hand-off rather
    than anything about expressions, which is tested where the rules live.
    """

    def test_empty_library_is_a_dictionary(self):
        """Not ``None``: the tiers below iterate it without checking."""

        self.assertEqual(get_expressions(), {})

    def test_names_map_to_their_text(self):
        """The shape tiers 1 and 2 expect, and nothing more."""

        self.make("rel_bg", "bg_center - nanmedian(bg_center)")
        self.make("twice_bg", "bg_center * 2")

        self.assertEqual(
            get_expressions(),
            {
                "rel_bg": "bg_center - nanmedian(bg_center)",
                "twice_bg": "bg_center * 2",
            },
        )

    def test_the_whole_library_regardless_of_what_resolves(self):
        """Filtering by project would need a project, which tier 3 lacks.

        An expression naming a diagnostic nothing has recorded is not an
        error; it is simply not offered where it cannot be drawn, and
        deciding that belongs to whoever holds the project's names.
        """

        self.make("references_nothing_real", "no_such_diagnostic * 2")

        self.assertIn("references_nothing_real", get_expressions())


if __name__ == "__main__":
    unittest.main()
