"""Shared plot constants and utilities for BUI views."""

import json
from io import StringIO

import matplotlib
from matplotlib import pyplot

from django.http import JsonResponse

from autowisp.database.interface import start_db_session

channel_colors = {"R": "#ff0000", "G": "#00ff00", "B": "#0000ff"}

#: The styles that draw a series as a curve through its points rather than
#: as points, which is what a smoothed quantity wants over the raw one it
#: smooths.
#:
#: Keyed by the character the style menu is written with, and holding the
#: linestyle matplotlib draws for it. The menu is a string walked one
#: character at a time, so each style needs a character of its own, and
#: matplotlib spells dashed with two -- hence ``|`` standing for it here
#: rather than the style passing straight through. ``|`` is a marker to
#: matplotlib, so nothing may hand one of these keys to it unmapped.
line_styles = {"-": "-", "|": "--", ":": ":"}


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
