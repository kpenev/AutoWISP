"""Tests of what the experiments menu offers, which needs no project."""

import unittest
from datetime import datetime

from autowisp.browser_interface.processing.experiments_menu import (
    plan_experiments_menu,
)


class TestExperimentsMenu(unittest.TestCase):
    """The actions offered for each experiment, and the attention flag."""

    @staticmethod
    def _describe(name, state, task=None, running=None):
        """Return an experiment as ``describe_experiments`` gives it."""

        return {
            "name": name,
            "state": state,
            "created": datetime(2026, 10, 24),
            "operation": (
                None if task is None else {"task": task, "running": running}
            ),
        }

    def _plan(self, described, selected=None):
        """Return each row's offers by name, and the attention flag."""

        rows, needs_attention = plan_experiments_menu(described, selected)
        return {
            row["name"]: {
                offer
                for offer in ("selectable", "offer_delete", "offer_restore")
                if row[offer]
            }
            for row in rows
        }, needs_attention

    def test_offers(self):
        """
        Each state and operation gets its own offers; the selected is left out.

        Includes a merge whose restore was interrupted in turn, which is
        offered restoring again, and a delete interrupted, offered deleting
        again.
        """

        described = [
            self._describe("cur", "current"),
            self._describe("old", "stale"),
            self._describe("making", "creating", "create experiment", "here"),
            self._describe("away", "creating", "reset experiment", "otherhost"),
            self._describe("half_merged", "merging", "merge experiment"),
            self._describe(
                "merging_now", "merging", "merge experiment", "here"
            ),
            self._describe("half_restored", "merging", "restore experiment"),
            self._describe("half_deleted", "deleting", "delete experiment"),
            self._describe("me", "current"),
        ]
        offers, needs_attention = self._plan(described, selected="me")
        self.assertEqual(
            offers,
            {
                "cur": {"selectable", "offer_delete"},
                "old": {"selectable", "offer_delete"},
                "making": {"offer_delete"},
                "away": set(),
                "half_merged": {"offer_restore"},
                "merging_now": set(),
                "half_restored": {"offer_restore"},
                "half_deleted": {"offer_delete"},
            },
        )
        self.assertTrue(needs_attention)

    def test_attention(self):
        """
        Only a stale experiment or an interrupted operation needs attention.

        A running operation does not, and the selected experiment counts
        though it has no row.
        """

        settled = [
            self._describe("cur", "current"),
            self._describe("making", "creating", "create experiment", "here"),
            self._describe(
                "merging_now", "merging", "merge experiment", "otherhost"
            ),
        ]
        self.assertFalse(self._plan(settled)[1])
        self.assertTrue(
            self._plan(settled + [self._describe("me", "stale")], "me")[1]
        )
        self.assertTrue(
            self._plan(
                settled
                + [self._describe("half", "creating", "reset experiment")]
            )[1]
        )
