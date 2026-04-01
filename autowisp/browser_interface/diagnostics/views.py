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
    display_image_diagnostics,
    update_image_diagnostics_plot,
)
from .preview_calibrated import preview_calibrated_image, get_image_overlay
from .diag_vs_diag_views import display_diag_vs_diag, update_diag_vs_diag_plot

# pylint: enable=unused-import
