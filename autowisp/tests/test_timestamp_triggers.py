"""Tests for the DDL that installs the update-timestamp triggers.

Its own module rather than part of ``test_database_migration``: the
migration code *reinstates* these triggers, but what they say is decided
in :mod:`autowisp.database.data_model`, and that is what is checked here.
"""

import unittest

from autowisp.database.data_model import timestamp_trigger_ddl
from autowisp.database.data_model.base import DataModelBase


class TestTriggerQuoting(unittest.TestCase):
    """Every identifier a trigger names is quoted.

    ``condition`` is a table here and a reserved word in MariaDB, so an
    unquoted ``ON condition`` is a syntax error. Over the metadata rather
    than over that one name, and on both dialects, since the next
    reserved word to become a table would be as silent as this one was on
    SQLite -- which reserves a different set and forgave it.
    """

    def tables(self):
        """Return the tables the timestamp triggers are installed on."""

        return [
            table.name
            for table in DataModelBase.metadata.tables.values()
            if "timestamp" in table.columns and table.primary_key.columns
        ]

    def test_mysql_quotes_what_it_names(self):
        """The table and the column written; it names no key columns."""

        for table in self.tables():
            with self.subTest(table=table):
                statement = timestamp_trigger_ddl(table, ["id"], "mysql")
                self.assertIn(f"ON `{table}`", statement)
                self.assertIn("NEW.`timestamp`", statement)

    def test_sqlite_quotes_what_it_names(self):
        """Including the key columns the row is addressed by."""

        for table in self.tables():
            with self.subTest(table=table):
                statement = timestamp_trigger_ddl(table, ["id"], "sqlite")
                self.assertIn(f'ON "{table}"', statement)
                self.assertIn(f'UPDATE "{table}"', statement)
                self.assertIn('SET "timestamp"', statement)
                self.assertIn('"id" = NEW."id"', statement)
