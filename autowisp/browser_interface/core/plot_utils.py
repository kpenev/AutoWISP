"""Shared plot constants and utilities for BUI views."""

import json
from io import StringIO

import matplotlib
from matplotlib import pyplot

from django.http import JsonResponse

from autowisp.database.interface import start_db_session


channel_colors = {"R": "#ff0000", "G": "#00ff00", "B": "#0000ff"}

edge_only_markers = set("x+.,1234|_")


def setup_svg_matplotlib():
    """Configure matplotlib to render SVG with dark background."""
    matplotlib.use("svg")
    pyplot.style.use("dark_background")


def figure_to_svg_response(fig, **extra):
    """Serialize a matplotlib figure to SVG and return a JsonResponse.

    Args:
        fig:    The matplotlib Figure to serialize and close.
        **extra: Additional key/value pairs to include in the JSON response
                 alongside ``plot_data``.

    Returns:
        JsonResponse with ``plot_data`` containing the SVG string.
    """
    with StringIO() as svg_stream:
        fig.savefig(svg_stream, bbox_inches="tight", format="svg")
        pyplot.close(fig)
        return JsonResponse({"plot_data": svg_stream.getvalue(), **extra})
