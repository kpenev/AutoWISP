"""Collect all views for the diagnostics app.

Also where the stored expression library is fetched and handed to the
plotting code.  That code takes it as an argument and never looks it up, so
that it needs no browser-interface database -- the same reason the tiers
below it take one too -- which leaves this module, already the app's Django
side, as the place the two meet.
"""

# This module should collect all views
# pylint: disable=unused-import
from .detrending_diagnostics_views import (
    display_detrending_diagnostics,
    refresh_detrending_diagnostics,
    update_detrending_diagnostics_plot,
    download_detrending_diagnostics_plot,
)
from .image_diagnostics_views import (
    display_image_diagnostics,
    update_plot_view,
    download_plot_view,
    create_diagnostics_figure,
    plot_session_key,
)
from .preview_calibrated import preview_calibrated_image, get_image_overlay

# pylint: enable=unused-import

from . import image_diagnostics_views
from .expression_data import get_expressions


def display_diagnostics(request, x_diagnostic, y_diagnostic):
    """Show the series table for an axis pair, expressions included."""

    return image_diagnostics_views.display_diagnostics(
        request, x_diagnostic, y_diagnostic, get_expressions()
    )


def update_diagnostics_plot(request, x_diagnostic, y_diagnostic):
    """Redraw the figure, with the library available to both axes."""

    return update_plot_view(
        request,
        create_diagnostics_figure,
        session_key=plot_session_key,
        x_diagnostic=x_diagnostic,
        y_diagnostic=y_diagnostic,
        expressions=get_expressions(),
    )


def download_diagnostics_plot(request, x_diagnostic, y_diagnostic):
    """Regenerate the last figure as a PDF, library and all."""

    return download_plot_view(
        request,
        create_diagnostics_figure,
        session_key=plot_session_key,
        x_diagnostic=x_diagnostic,
        y_diagnostic=y_diagnostic,
        expressions=get_expressions(),
    )
