"""Tests for the diagnostic expression library stored in a project.

The library is read and written through
:mod:`autowisp.diagnostics.expression_library`, against a throwaway project
database. What an expression *means* is tested where those rules live, in
the expression layer's own tests; here it is only what gets stored, and
that references survive a rename.

The management form is tested here too, for the one check it makes itself
rather than delegating: that a name is a slug and is not already taken.
"""

import os
import tempfile
import unittest

import django
from sqlalchemy.exc import IntegrityError

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "autowisp.browser_interface.django_project.settings",
)
django.setup()

# pylint: disable=wrong-import-position
from autowisp.browser_interface.diagnostics.forms import (
    DiagnosticExpressionForm,
)
from autowisp.database.interface import set_project_home, start_db_session
from autowisp.diagnostics.expression_library import (
    delete_expressions,
    get_expression_descriptions,
    get_expression_entries,
    get_expressions,
    store_expression,
    write_expressions,
)

# pylint: enable=wrong-import-position


class ExpressionLibraryTestCase(unittest.TestCase):
    """Base creating one throwaway project, emptied before every test."""

    @classmethod
    def setUpClass(cls):
        # Closed in tearDownClass rather than by a context manager, which a
        # fixture spanning every test of the class cannot use.
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        # pylint: enable=consider-using-with
        set_project_home(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        with start_db_session() as db_session:
            delete_expressions(get_expressions(db_session), db_session)

    @staticmethod
    def make(name, expression="astrom_residual[0] / diagonal_fov[0]", **kw):
        """Store one expression, returning what the store reported."""

        with start_db_session() as db_session:
            return store_expression(
                db_session, name=name, expression=expression, **kw
            )

    @staticmethod
    def library():
        """The stored library, ``{name: expression}``."""

        with start_db_session() as db_session:
            return get_expressions(db_session)


class TestNameSpace(ExpressionLibraryTestCase):
    """Names have to behave like the diagnostic names they sit beside."""

    def test_name_is_unique(self):
        """Two expressions cannot share a name.

        The name is what a selector and a URL carry, and what other
        expressions and exclusion rules reference, so a duplicate would be
        ambiguous everywhere at once. The form refuses one first; this is
        the database guaranteeing it regardless.
        """

        self.make("rel_astrom_residual")
        with self.assertRaises(IntegrityError):
            self.make("rel_astrom_residual", "bg_center[0]")


class TestStoredFields(ExpressionLibraryTestCase):
    """What the library keeps, and what it deliberately does not check."""

    def test_description_is_optional(self):
        """Most expressions are self-explanatory from their text."""

        self.make("terse")
        with start_db_session() as db_session:
            self.assertEqual(
                get_expression_descriptions(db_session), {"terse": ""}
            )

    def test_unknown_names_are_not_rejected_here(self):
        """The library stores text; resolving names is not its job.

        An expression may legitimately reference diagnostics the project has
        never recorded -- it is then simply not offered -- so refusing it
        here would be wrong.
        """

        self.make("references_nothing_real", "no_such_diagnostic * 2")
        self.assertEqual(
            self.library(),
            {"references_nothing_real": "no_such_diagnostic * 2"},
        )


class TestLibraryAccess(ExpressionLibraryTestCase):
    """What reading hands to the expression layer and to an export."""

    def test_empty_library_is_a_dictionary(self):
        """Not ``None``: the layers below iterate it without checking."""

        self.assertEqual(self.library(), {})

    def test_names_map_to_their_text(self):
        """The shape the expression layer expects, and nothing more."""

        self.make("rel_bg", "bg_center[0] - nanmedian(bg_center[0])")
        self.make("twice_bg", "bg_center[0] * 2")

        self.assertEqual(
            self.library(),
            {
                "rel_bg": "bg_center[0] - nanmedian(bg_center[0])",
                "twice_bg": "bg_center[0] * 2",
            },
        )

    def test_entries_for_export(self):
        """Every stored field, by name, skipping names not stored."""

        self.make("zeta", "bg_center[0]", description="last")
        self.make("alpha", "s_center[0]")

        with start_db_session() as db_session:
            entries = get_expression_entries(
                ["zeta", "alpha", "no_such_expression"], db_session
            )

        self.assertEqual(
            entries,
            [
                {
                    "name": "alpha",
                    "expression": "s_center[0]",
                    "description": "",
                },
                {
                    "name": "zeta",
                    "expression": "bg_center[0]",
                    "description": "last",
                },
            ],
        )


class TestStoreExpression(ExpressionLibraryTestCase):
    """Adding, replacing and renaming one expression."""

    def test_replacing_keeps_the_name(self):
        """An edit that keeps the name changes the text in place."""

        self.make("rel_bg", "bg_center[0]")
        self.assertEqual(
            self.make("rel_bg", "bg_center[0] * 2", replacing="rel_bg"), []
        )
        self.assertEqual(self.library(), {"rel_bg": "bg_center[0] * 2"})

    def test_rename_carries_the_dependents(self):
        """What referenced the old name references the new one.

        Leaving them naming something that no longer exists would break
        them, and refusing the rename would not help: a dependent cannot be
        pointed at the new name before it exists.
        """

        self.make("rel_bg", "bg_center[0] - nanmedian(bg_center[0])")
        self.make("twice_rel_bg", "rel_bg[0] * 2")
        self.make("unrelated", "s_center[0]")

        updated = self.make(
            "bg_offset",
            "bg_center[0] - nanmedian(bg_center[0])",
            replacing="rel_bg",
        )

        self.assertEqual(updated, ["twice_rel_bg"])
        self.assertEqual(
            self.library(),
            {
                "bg_offset": "bg_center[0] - nanmedian(bg_center[0])",
                "twice_rel_bg": "bg_offset[0] * 2",
                "unrelated": "s_center[0]",
            },
        )

    def test_replacing_a_name_not_stored_adds(self):
        """A stale edit link adds the expression rather than losing it."""

        self.assertEqual(self.make("rel_bg", replacing="gone"), [])
        self.assertIn("rel_bg", self.library())


class TestDeleteAndWrite(ExpressionLibraryTestCase):
    """Deleting several, and writing an import's worth at once."""

    def test_delete_only_the_named(self):
        """Other expressions, and names not stored, are left alone."""

        for name in ("alpha", "mu", "zeta"):
            self.make(name)

        with start_db_session() as db_session:
            delete_expressions(["alpha", "zeta", "not_stored"], db_session)

        self.assertEqual(list(self.library()), ["mu"])

    def test_write_counts_added_and_replaced(self):
        """New names are added, existing ones overwritten, and counted."""

        self.make("rel_bg", "bg_center[0]")

        with start_db_session() as db_session:
            counts = write_expressions(
                {
                    "rel_bg": {
                        "expression": "bg_center[0] * 2",
                        "description": "replaced",
                    },
                    "rel_s": {
                        "expression": "s_center[0]",
                        "description": "",
                    },
                },
                db_session,
            )
            descriptions = get_expression_descriptions(db_session)

        self.assertEqual(counts, (1, 1))
        self.assertEqual(
            self.library(),
            {"rel_bg": "bg_center[0] * 2", "rel_s": "s_center[0]"},
        )
        self.assertEqual(descriptions, {"rel_bg": "replaced", "rel_s": ""})


class TestExpressionForm(unittest.TestCase):
    """The name checks the management form makes itself."""

    library = {"rel_bg": "bg_center[0] - nanmedian(bg_center[0])"}

    def form(self, name, replacing=None):
        """Bind the form to one proposed expression."""

        return DiagnosticExpressionForm(
            {"name": name, "expression": "bg_center[0] * 2"},
            expressions=self.library,
            replacing=replacing,
        )

    def test_name_must_survive_a_url(self):
        """Rejected unless it is a slug.

        Expressions are selected through ``image/<slug:x>/vs/<slug:y>``, so
        a name outside the slug charset could be stored but never plotted.
        """

        form = self.form("not a slug!")
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_a_slug_name_is_accepted(self):
        """The names the documentation suggests actually validate."""

        for name in ("rel_astrom_residual", "bg-relative", "pixel_q999_ratio"):
            with self.subTest(name=name):
                self.assertTrue(self.form(name).is_valid())

    def test_a_taken_name_is_refused(self):
        """Adding under a stored name would replace it without asking."""

        form = self.form("rel_bg")
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_editing_keeps_its_own_name(self):
        """The expression being edited does not clash with itself."""

        self.assertTrue(self.form("rel_bg", replacing="rel_bg").is_valid())

    def test_renaming_onto_another_is_refused(self):
        """A rename may not take a name some other expression has."""

        form = DiagnosticExpressionForm(
            {"name": "rel_bg", "expression": "s_center[0]"},
            expressions=dict(self.library, rel_s="s_center[0]"),
            replacing="rel_s",
        )
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)


if __name__ == "__main__":
    unittest.main()
