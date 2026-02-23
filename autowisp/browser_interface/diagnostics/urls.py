"""Define the URL paths used by the diagnostics BUI app."""

from django.urls import path

from . import views

app_name = "diagnostics"

urlpatterns = [
    path(
        "detrending/<slug:step>/<slug:imtype>",
        views.display_detrending_diagnostics,
        name="diagnostics",
    ),
    path(
        "detrending/<slug:step>/<slug:imtype>/<slug:master_ids>",
        views.display_detrending_diagnostics,
        name="diagnostics",
    ),
    path(
        "display_detrending_diagnostics",
        views.display_detrending_diagnostics,
        name="display_detrending_diagnostics",
    ),
    path(
        "refresh_detrending_diagnostics",
        views.refresh_detrending_diagnostics,
        name="refresh_diagnostics",
    ),
    path(
        "update_detrending_diagnostics_plot",
        views.update_detrending_diagnostics_plot,
        name="update_diagnostics_plot",
    ),
    path(
        "download_detrending_diagnostics_plot",
        views.download_detrending_diagnostics_plot,
        name="download_diagnostics_plot",
    ),
    path(
        "image/<slug:diagnostic_name>",
        views.display_image_diagnostics,
        name="display_image_diagnostics",
    ),
    path(
        "image/<slug:diagnostic_name>/update_plot",
        views.update_image_diagnostics_plot,
        name="update_image_diagnostics_plot",
    ),
]
