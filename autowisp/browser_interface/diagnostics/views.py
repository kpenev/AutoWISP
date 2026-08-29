"""Collect all views for the diagnostics app."""

# This module should collect all views
# pylint: disable=unused-import
from .detrending_diagnostics_views import (
    display_detrending_diagnostics,
    refresh_detrending_diagnostics,
    update_detrending_diagnostics_plot,
    download_detrending_diagnostics_plot,
)
from .image_diagnostics_views import (
    display_diagnostics,
    display_image_diagnostics,
    update_plot_view,
    download_plot_view,
    create_diagnostics_figure,
)
from .preview_calibrated import preview_calibrated_image, get_image_overlay

# pylint: enable=unused-import
