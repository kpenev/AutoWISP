"""Tests for the browser-interface model base and its timestamp triggers.

Importing this module configures Django against the throwaway user data
directory that ``autowisp.tests`` installs, so the developer's real
``bui_db.sqlite3`` -- which holds their project list -- is never touched.
That redirect happens when the parent package is imported, which is before
this module runs, so nothing here needs to arrange it.
"""

import os
import sys
import time
import unittest
from pathlib import Path

import django

_browser_interface = Path(__file__).resolve().parents[1] / "browser_interface"
if str(_browser_interface) not in sys.path:
    sys.path.insert(0, str(_browser_interface))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_project.settings")
django.setup()

# pylint: disable=wrong-import-position
from django.apps import apps
from django.conf import settings
from django.core.management import call_command
from django.db import connection

from core.models import BuiModelBase
from core.timestamp_triggers import timestamped_models
from home.models import Project

# pylint: enable=wrong-import-position

#: Enough for the millisecond-resolution timestamps the trigger writes to
#: differ between two updates.
_tick = 0.05


class BuiModelTestCase(unittest.TestCase):
    """Base migrating the throwaway browser-interface database."""

    @classmethod
    def setUpClass(cls):
        assert "autowisp_tests_" in str(
            settings.DATABASES["default"]["NAME"]
        ), "refusing to run against a real browser-interface database"
        call_command("migrate", verbosity=0)

    def make_project(self, name):
        """Create and return a project row to experiment on."""

        # False positive: Django adds the manager at class-creation time.
        # pylint: disable=no-member
        return Project.objects.create(
            name=name, description="", path=f"/tmp/{name}"
        )

    def stamp(self, project):
        """Re-read *project*'s ``modified`` from the database."""

        # pylint: disable=no-member
        return Project.objects.get(pk=project.pk).modified


class TestModelBase(BuiModelTestCase):
    """Every browser-interface model carries the timestamp columns."""

    def test_project_has_both_columns(self):
        """``Project`` inherits ``created`` and ``modified``."""

        fields = {
            field.name
            for field in Project._meta.fields  # pylint: disable=protected-access
        }
        self.assertIn("created", fields)
        self.assertIn("modified", fields)

    def test_selection_matches_the_base_class(self):
        """Exactly the models deriving from the base are selected.

        Comparing the two sets, rather than checking one known model, is
        what makes this hold for models added later: the triggers are
        installed on the strength of a ``modified`` field being present,
        and this asserts that predicate agrees with deriving from
        :class:`core.models.BuiModelBase`.  A model that gained the column
        some other way, or one that derived from the base but shadowed the
        field, would show up here.
        """

        derived = {
            model
            for app_config in apps.get_app_configs()
            for model in app_config.get_models()
            if issubclass(model, BuiModelBase)
        }
        covered = {
            model
            for app_config in apps.get_app_configs()
            for model in timestamped_models(app_config)
        }

        self.assertEqual(derived, covered)
        # Guard against both being empty, which would pass vacuously.
        self.assertIn(Project, derived)

    def test_contrib_models_are_left_alone(self):
        """Django's own tables get no trigger; they have no such column."""

        self.assertEqual(
            list(timestamped_models(apps.get_app_config("sessions"))), []
        )


class TestModifiedIsMaintained(BuiModelTestCase):
    """``modified`` advances however the row is written."""

    def test_every_covered_model_has_its_trigger(self):
        """``migrate`` leaves a trigger on each selected model's table."""

        if connection.vendor != "sqlite":
            self.skipTest("trigger introspection here is SQLite-specific")

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
            names = {row[0] for row in cursor.fetchall()}

        covered = [
            model
            for app_config in apps.get_app_configs()
            for model in timestamped_models(app_config)
        ]
        self.assertIn(Project, covered)
        for model in covered:
            table = model._meta.db_table  # pylint: disable=protected-access
            self.assertIn(f"update_{table}_modified", names)

    def _assert_advances(self, project, write):
        """Assert *write* moves ``modified`` forward."""

        before = self.stamp(project)
        time.sleep(_tick)
        write()
        self.assertGreater(
            self.stamp(project),
            before,
            "modified did not advance",
        )

    def test_save(self):
        """The ordinary path, covered by ``auto_now`` alone."""

        project = self.make_project("save")
        self._assert_advances(project, project.save)

    def test_queryset_update(self):
        """``QuerySet.update`` bypasses ``auto_now`` entirely."""

        project = self.make_project("update")
        self._assert_advances(
            project,
            # pylint: disable=no-member
            lambda: Project.objects.filter(pk=project.pk).update(
                description="changed"
            ),
        )

    def test_bulk_update(self):
        """``bulk_update`` would otherwise write back a *stale* value.

        Django reads the in-memory attribute rather than calling
        ``pre_save``, so without the trigger this does not merely skip the
        stamp -- it restores the timestamp the object was loaded with.
        """

        project = self.make_project("bulk")
        # pylint: disable=no-member
        loaded = Project.objects.get(pk=project.pk)
        loaded.description = "changed"
        self._assert_advances(
            project,
            lambda: Project.objects.bulk_update([loaded], ["description"]),
        )

    def test_raw_sql(self):
        """Statements issued outside the ORM are covered too."""

        project = self.make_project("raw")

        def write():
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE home_project SET description='raw' WHERE id=%s",
                    [project.pk],
                )

        self._assert_advances(project, write)


class TestMigrationCarriesExistingRows(BuiModelTestCase):
    """Upgrading an existing database keeps its data and stamps it."""

    def test_existing_row_keeps_created_and_gains_modified(self):
        """The column is added to a populated table without loss.

        Also exercises migrating *backwards* across the change, which is
        what first revealed that SQLite refuses to alter a column while a
        trigger names it -- hence dropping them in ``pre_migrate``.
        """

        original = "2020-01-02 03:04:05"
        call_command("migrate", "home", "0001", verbosity=0)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO home_project"
                    " (name, description, path, created)"
                    " VALUES ('legacy', 'pre-existing', '/tmp/legacy', %s)",
                    [original],
                )
        finally:
            call_command("migrate", "home", verbosity=0)

        # pylint: disable=no-member
        legacy = Project.objects.get(name="legacy")
        self.assertEqual(legacy.created.strftime("%Y-%m-%d %H:%M:%S"), original)
        self.assertIsNotNone(legacy.modified)


if __name__ == "__main__":
    unittest.main()
