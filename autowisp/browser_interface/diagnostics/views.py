"""Collect all views for the diagnostics app.

Also where the stored expression library is fetched and handed to the
plotting code.  That code takes it as an argument and never looks it up, so
that its tests can write the library they need -- the same reason the tiers
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
    update_plot_view,
    download_plot_view,
    create_diagnostics_figure,
    plot_session_key,
)
from .series_table import get_table_response
from .preview_calibrated import preview_calibrated_image, get_image_overlay
from .expression_views import (
    list_expressions,
    save_expression,
    delete_expressions,
    export_expressions,
    import_expressions,
    confirm_import_expressions,
)

# pylint: enable=unused-import

from . import image_diagnostics_views
from .expression_data import get_expressions, get_expression_descriptions


def display_diagnostics(request, x_quantity, y_quantities):
    """Show a section per y quantity, the library and its prose included."""

    return image_diagnostics_views.display_diagnostics(
        request,
        x_quantity,
        y_quantities,
        get_expressions(),
        get_expression_descriptions(),
    )


def diagnostics_section(request, x_quantity, y_quantity):
    """Render one section, for the y selector to append to the page."""

    return image_diagnostics_views.diagnostics_section(
        request,
        x_quantity,
        y_quantity,
        get_expressions(),
        get_expression_descriptions(),
    )


def update_diagnostics_plot(request, x_quantity):
    """Redraw the figure, with the library available to every row."""

    return update_plot_view(
        request,
        create_diagnostics_figure,
        session_key=plot_session_key,
        extra=get_table_response,
        x_quantity=x_quantity,
        expressions=get_expressions(),
    )


def download_diagnostics_plot(request, x_quantity):
    """Regenerate the last figure as a PDF, library and all."""

    return download_plot_view(
        request,
        create_diagnostics_figure,
        session_key=plot_session_key,
        x_quantity=x_quantity,
        expressions=get_expressions(),
    )
