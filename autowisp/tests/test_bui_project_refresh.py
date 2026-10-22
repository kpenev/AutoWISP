"""Refreshing selection pages must retain the active project.

Only the refresh branches are exercised: no images or pipeline jobs are
needed to check what survives into the redirected request.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from autowisp.browser_interface.processing import (
    select_photref_views,
    tune_starfind_views,
)
from autowisp.browser_interface import set_project_middleware


class TestProjectRefresh(unittest.TestCase):
    """Page caches are rebuilt while project and other page state survive."""

    def request(self, **session):
        """Return a request with an active project and optional cached data."""

        return SimpleNamespace(
            method="GET",
            POST={},
            session={
                "project_home": "/projects/example",
                "project_name": "Example",
                "unrelated": "keep",
                **session,
            },
        )

    def assert_project_active(self, request):
        """The redirected request can still activate the same database."""

        self.assertEqual(request.session["project_name"], "Example")
        self.assertEqual(request.session["unrelated"], "keep")
        response = object()
        with mock.patch.object(
            set_project_middleware, "set_project_home"
        ) as activate:
            middleware = set_project_middleware.set_project_middleware(
                lambda request: response
            )
            self.assertIs(middleware(request), response)
        activate.assert_called_once_with("/projects/example")

    def test_starfind_refresh_retains_project(self):
        """Refresh drops stale batches and image data, not project identity."""

        request = self.request(
            starfind={"pending": {"object": ["stale"]}},
            fits_display={"image_index": 9},
            extracted={"x": [1], "y": [2]},
            photref_merit="custom_merit",
        )
        with mock.patch.object(tune_starfind_views, "redirect") as redirect:
            response = tune_starfind_views.select_starfind_batch(
                request, refresh=True
            )
        self.assertIs(response, redirect.return_value)
        redirect.assert_called_once_with("processing:select_starfind_batch")
        for key in ("starfind", "fits_display", "extracted"):
            self.assertNotIn(key, request.session)
        self.assertEqual(request.session["photref_merit"], "custom_merit")
        self.assert_project_active(request)

    def test_photref_refresh_retains_project_and_merit(self):
        """GET keeps the merit; a posted choice replaces it on refresh."""

        for method, posted, expected in (
            ("GET", {}, "previous_merit"),
            ("POST", {"merit": "new_merit"}, "new_merit"),
            ("POST", {"merit": ""}, "previous_merit"),
        ):
            with self.subTest(method=method, posted=posted):
                request = self.request(
                    need_photref={"master_values": ["stale"]},
                    demo=True,
                    fits_display={"image_index": 9},
                    photref_merit="previous_merit",
                    starfind={"pending": {}},
                )
                request.method = method
                request.POST = posted
                with mock.patch.object(
                    select_photref_views, "redirect"
                ) as redirect:
                    response = select_photref_views.select_photref_target(
                        request, recalc=True
                    )
                self.assertIs(response, redirect.return_value)
                redirect.assert_called_once_with(
                    "processing:select_photref_target"
                )
                for key in ("need_photref", "demo", "fits_display"):
                    self.assertNotIn(key, request.session)
                self.assertIn("starfind", request.session)
                self.assertEqual(request.session["photref_merit"], expected)
                self.assert_project_active(request)

    def test_refresh_without_cached_state(self):
        """Refreshing twice or before visiting an image remains valid."""

        request = self.request()
        for _ in range(2):
            with mock.patch.object(tune_starfind_views, "redirect"):
                tune_starfind_views.select_starfind_batch(request, refresh=True)
            with mock.patch.object(select_photref_views, "redirect"):
                select_photref_views.select_photref_target(request, recalc=True)
        self.assert_project_active(request)

    def test_merit_post_without_refresh_keeps_groups(self):
        """Changing the ranking alone does not rebuild the target groups."""

        groups = {"master_values": ["cached"]}
        request = self.request(need_photref=groups)
        request.method = "POST"
        request.POST = {"merit": "new_merit"}
        with mock.patch.object(select_photref_views, "redirect"):
            select_photref_views.select_photref_target(request)
        self.assertIs(request.session["need_photref"], groups)
        self.assertEqual(request.session["photref_merit"], "new_merit")
        self.assert_project_active(request)
