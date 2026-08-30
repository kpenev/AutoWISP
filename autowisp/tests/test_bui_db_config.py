"""Tests for resolving which database the browser interface uses.

These exercise ``django_project.db_config`` directly rather than through
``settings``, because the setting is read once at import time and cannot be
re-resolved afterwards.
"""

import sys
import tempfile
import unittest
from os import environ, path
from pathlib import Path

_browser_interface = Path(__file__).resolve().parents[1] / "browser_interface"
if str(_browser_interface) not in sys.path:
    sys.path.insert(0, str(_browser_interface))

# pylint: disable=wrong-import-position
from django_project import db_config

# pylint: enable=wrong-import-position


class DbConfigTestCase(unittest.TestCase):
    """Base giving each test an empty data directory and no environment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name
        self._saved = environ.pop(db_config.url_env_var, None)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        """Put back whatever the environment held before the test."""

        if self._saved is None:
            environ.pop(db_config.url_env_var, None)
        else:
            environ[db_config.url_env_var] = self._saved

    def write_url_file(self, url):
        """Write *url* to the data directory's URL file."""

        with open(
            path.join(self.data_dir, db_config.url_fname),
            "w",
            encoding="utf-8",
        ) as url_file:
            url_file.write(url)


class TestDiscovery(DbConfigTestCase):
    """Where the URL is looked for, and in what order."""

    def test_nothing_configured(self):
        """No environment and no file means no URL."""

        self.assertIsNone(db_config.get_database_url(self.data_dir))

    def test_url_file(self):
        """A URL file is read, and surrounding whitespace ignored."""

        self.write_url_file("  sqlite:////tmp/from-file.sqlite3\n")
        self.assertEqual(
            db_config.get_database_url(self.data_dir),
            "sqlite:////tmp/from-file.sqlite3",
        )

    def test_environment_wins(self):
        """The environment takes precedence over the file."""

        self.write_url_file("sqlite:////tmp/from-file.sqlite3")
        environ[db_config.url_env_var] = "sqlite:////tmp/from-env.sqlite3"
        self.assertEqual(
            db_config.get_database_url(self.data_dir),
            "sqlite:////tmp/from-env.sqlite3",
        )

    def test_empty_values_are_ignored(self):
        """An empty variable or file falls through to the default."""

        environ[db_config.url_env_var] = "   "
        self.write_url_file("\n")
        self.assertIsNone(db_config.get_database_url(self.data_dir))


class TestDefaultDatabase(DbConfigTestCase):
    """An unconfigured installation keeps working untouched."""

    def test_defaults_to_sqlite_in_the_data_directory(self):
        """This is what every existing installation already has."""

        databases = db_config.get_databases(self.data_dir)
        self.assertEqual(
            databases["default"]["ENGINE"], "django.db.backends.sqlite3"
        )
        self.assertEqual(
            databases["default"]["NAME"],
            path.join(self.data_dir, db_config.sqlite_fname),
        )


class TestUrlTranslation(DbConfigTestCase):
    """A URL becomes Django's DATABASES dictionary."""

    def test_sqlite_url_supplies_the_path(self):
        """An explicit SQLite file may live outside the data directory."""

        self.write_url_file("sqlite:////elsewhere/bui.sqlite3")
        databases = db_config.get_databases(self.data_dir)
        self.assertEqual(
            databases["default"]["ENGINE"], "django.db.backends.sqlite3"
        )
        self.assertEqual(databases["default"]["NAME"], "/elsewhere/bui.sqlite3")

    def test_mysql_url_is_mapped_field_by_field(self):
        """The spelling matches the project database's own URLs."""

        environ[db_config.url_env_var] = (
            "mysql+pymysql://someone:secret@db.example.org:3307/autowisp_bui"
        )
        databases = db_config.get_databases(self.data_dir)["default"]
        self.assertEqual(databases["ENGINE"], "django.db.backends.mysql")
        self.assertEqual(databases["NAME"], "autowisp_bui")
        self.assertEqual(databases["USER"], "someone")
        self.assertEqual(databases["PASSWORD"], "secret")
        self.assertEqual(databases["HOST"], "db.example.org")
        self.assertEqual(databases["PORT"], "3307")

    def test_mariadb_uses_the_same_backend(self):
        """MariaDB is MySQL as far as Django is concerned."""

        environ[db_config.url_env_var] = (
            "mariadb+pymysql://u:p@host/autowisp_bui"
        )
        databases = db_config.get_databases(self.data_dir)["default"]
        self.assertEqual(databases["ENGINE"], "django.db.backends.mysql")
        self.assertEqual(databases["PORT"], "")

    def test_unsupported_backend_is_refused(self):
        """Refused rather than half-supported.

        ``core.timestamp_triggers`` implements ``modified`` triggers for
        SQLite and MySQL only, so accepting another backend would leave the
        column silently unmaintained.
        """

        self.write_url_file("postgresql://u:p@host/autowisp_bui")
        with self.assertRaises(ValueError) as caught:
            db_config.get_databases(self.data_dir)
        self.assertIn("postgresql", str(caught.exception))


class TestMysqlDriver(DbConfigTestCase):
    """Django's MySQL backend is made to work with the driver present."""

    def test_pymysql_is_accepted(self):
        """AutoWISP documents ``mysql+pymysql`` but depends on no driver.

        Django's backend imports ``MySQLdb``, so ``pymysql`` has to be
        installed under that name; skipped where neither is present.
        """

        try:
            import pymysql  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.skipTest("no MySQL driver installed")

        # pylint: disable=protected-access
        db_config._ensure_mysql_driver()
        self.assertIs(sys.modules["MySQLdb"], pymysql)


if __name__ == "__main__":
    unittest.main()
