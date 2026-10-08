"""Unit tests for the guard against creating a project over another one.

Project creation drops and recreates every AutoWISP table, so pointing a
new project at a database that already holds them destroys whichever
project lives there. These exercise ``set_project_home(new_project=True)``
against throwaway SQLite databases -- the check is backend-independent
(it compares table names), so SQLite covers the centralised case too.
"""

import os
import tempfile
import unittest

from autowisp.database.interface import (
    DB_URL_FNAME,
    get_db_engine,
    set_project_home,
)
from autowisp.exceptions import DatabaseError


class TestNewProjectGuard(unittest.TestCase):
    """``new_project=True`` refuses a non-empty AutoWISP database."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.project_home = self._tmp.name

    def _url_file(self, project_home=None):
        return os.path.join(project_home or self.project_home, DB_URL_FNAME)

    def test_creation_in_empty_home_succeeds(self):
        """A fresh directory is a valid home for a new project."""

        set_project_home(self.project_home, new_project=True)

        self.assertTrue(
            os.path.isfile(os.path.join(self.project_home, "autowisp.db"))
        )

    def test_recreating_over_existing_project_refused(self):
        """A second creation against the same database is rejected."""

        set_project_home(self.project_home, new_project=True)

        with self.assertRaises(DatabaseError) as caught:
            set_project_home(self.project_home, new_project=True)

        self.assertIn("already contains", str(caught.exception))

    def test_opening_existing_project_still_allowed(self):
        """Without new_project the same call just reconnects."""

        set_project_home(self.project_home, new_project=True)
        set_project_home(self.project_home)

        self.assertIsNotNone(get_db_engine())

    def test_refused_centralised_url_is_not_persisted(self):
        """A rejected db_url leaves no autowisp_db.url behind.

        Otherwise the half-created project home would keep pointing at
        the database it was just refused.
        """

        shared = os.path.join(self._tmp.name, "shared.db")
        first_home = tempfile.mkdtemp(dir=self._tmp.name)
        second_home = tempfile.mkdtemp(dir=self._tmp.name)
        db_url = f"sqlite:///{shared}"

        set_project_home(first_home, db_url=db_url, new_project=True)
        self.assertTrue(os.path.isfile(self._url_file(first_home)))

        with self.assertRaises(DatabaseError):
            set_project_home(second_home, db_url=db_url, new_project=True)

        self.assertFalse(os.path.isfile(self._url_file(second_home)))


if __name__ == "__main__":
    unittest.main()
