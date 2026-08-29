"""Define the URL paths used by the diagnostics BUI app."""

import functools

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
    # Kept so links predating the axis merge keep working; redirects onto
    # the pair route below with x=jd.
    path(
        "image/<slug:diagnostic_name>",
        views.display_image_diagnostics,
        name="display_image_diagnostics",
    ),
    path(
        "image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>",
        views.display_diagnostics,
        name="display_diagnostics",
    ),
    path(
        "image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>/update_plot",
        functools.partial(
            views.update_plot_view,
            figure_factory=views.create_diagnostics_figure,
            session_key="diagnostics_last",
        ),
        name="update_diagnostics_plot",
    ),
    path(
        "image/<slug:x_diagnostic>/vs/<slug:y_diagnostic>/download_plot",
        functools.partial(
            views.download_plot_view,
            figure_factory=views.create_diagnostics_figure,
            session_key="diagnostics_last",
        ),
        name="download_diagnostics_plot",
    ),
    path(
        "preview_calibrated/<int:image_id>/<slug:color_channel>",
        views.preview_calibrated_image,
        name="preview_calibrated_image",
    ),
    path(
        "preview_calibrated/<int:image_id>/<slug:color_channel>"
        "/overlay/<slug:overlay_type>",
        views.get_image_overlay,
        name="get_image_overlay",
    ),
]
