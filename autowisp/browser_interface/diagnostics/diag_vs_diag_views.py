"""Views for diagnostic-vs-diagnostic scatter plots."""

import json
from io import StringIO

import matplotlib
from matplotlib import pyplot
from sqlalchemy import select, func
from sqlalchemy.orm import aliased

from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse

from autowisp.database.interface import start_db_session

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    ImageDiagnostics,
    Image,
    ObservingSession,
)

# pylint: enable=no-name-in-module

from .image_diagnostics_views import get_available_diagnostics


def get_available_series_for_pair(x_diagnostic, y_diagnostic, db_session):
    """
    Return the (observing session, channel) pairs that have both diagnostics.

    Queries for distinct (observing_session, channel) pairs for which at least
    one image has a measured value for *both* ``x_diagnostic`` and
    ``y_diagnostic``.

    Args:
        x_diagnostic(str):  Name of the diagnostic to use as the X axis.

        y_diagnostic(str):  Name of the diagnostic to use as the Y axis.

        db_session:  An active SQLAlchemy database session.

    Returns:
        dict with keys ``diagnostics_fields`` and ``diagnostics_list``,
        in the same format expected by ``diagnostics_app.html``.
    """

    x_diag = aliased(ImageDiagnostics)
    y_diag = aliased(ImageDiagnostics)
    x_type = aliased(DiagnosticType)
    y_type = aliased(DiagnosticType)

    query = (
        select(
            ObservingSession.label,
            ObservingSession.id,
            x_diag.channel,
            func.count(x_diag.id),  # pylint: disable=not-callable
        )
        .select_from(Image)
        .join(x_diag, x_diag.image_id == Image.id)  # pylint: disable=no-member
        .join(
            ObservingSession,
            ObservingSession.id == Image.observing_session_id,  # pylint: disable=no-member
        )
        .join(x_type, x_type.id == x_diag.diagnostic_id)
        .join(
            y_diag,
            (y_diag.image_id == Image.id)  # pylint: disable=no-member
            & (y_diag.channel == x_diag.channel),
        )
        .join(y_type, y_type.id == y_diag.diagnostic_id)
        .where(x_type.name == x_diagnostic, y_type.name == y_diagnostic)
        .group_by(ObservingSession.id, x_diag.channel)
        .order_by(ObservingSession.label, x_diag.channel)
    )

    channel_colors = {"R": "#ff0000", "G": "#00ff00", "B": "#0000ff"}
    diagnostics_list = []
    for session_label, session_id, channel, count in db_session.execute(
        query
    ).all():
        diagnostics_list.append(
            {
                "id": f"{session_id}_{channel}",
                "color": channel_colors.get(
                    channel[0].upper() if channel else "", "#ffffff"
                ),
                "marker": "o",
                "scale": "1.0",
                "label": f"{session_label} {channel}",
                "info": [session_label, channel, count],
            }
        )

    return {
        "diagnostics_fields": ["Observing Session", "Channel", "Count"],
        "diagnostics_list": diagnostics_list,
    }


def get_xy_series_data(series, x_diagnostic, y_diagnostic, db_session):
    """
    Query paired x/y diagnostic values for a single (session, channel) series.

    Args:
        series(dict):  A series entry as produced by
            :func:`get_available_series_for_pair`.

        x_diagnostic(str):  Name of the X-axis diagnostic.

        y_diagnostic(str):  Name of the Y-axis diagnostic.

        db_session:  An active SQLAlchemy database session.

    Returns:
        tuple:  ``(x_values, y_values, image_ids)`` as tuples of floats/ints.
            Empty tuples if no paired data is found.
    """

    parts = series["id"].split("_")
    session_id = int(parts[0])
    channel = parts[1]

    x_diag = aliased(ImageDiagnostics)
    y_diag = aliased(ImageDiagnostics)
    x_type = aliased(DiagnosticType)
    y_type = aliased(DiagnosticType)

    rows = db_session.execute(
        select(x_diag.value, y_diag.value, Image.id)  # pylint: disable=no-member
        .select_from(Image)
        .join(x_diag, x_diag.image_id == Image.id)  # pylint: disable=no-member
        .join(x_type, x_type.id == x_diag.diagnostic_id)
        .join(
            y_diag,
            (y_diag.image_id == Image.id)  # pylint: disable=no-member
            & (y_diag.channel == x_diag.channel),
        )
        .join(y_type, y_type.id == y_diag.diagnostic_id)
        .where(
            Image.observing_session_id == session_id,  # pylint: disable=no-member
            x_diag.channel == channel,
            x_type.name == x_diagnostic,
            y_type.name == y_diagnostic,
        )
    ).all()

    if not rows:
        return (), (), ()

    return zip(*rows)


def create_diag_vs_diag_figure(
    series_list, x_diagnostic, y_diagnostic, db_session, figure_config=None
):
    """
    Create a scatter plot of *x_diagnostic* versus *y_diagnostic*.

    Args:
        series_list(list):  Series entries (as produced by
            :func:`get_available_series_for_pair`) to plot.  Only entries
            whose ``marker`` is non-empty are plotted.

        x_diagnostic(str):  Name of the X-axis diagnostic.

        y_diagnostic(str):  Name of the Y-axis diagnostic.

        db_session:  An active SQLAlchemy database session.

        figure_config(dict):  Optional overrides.  Recognised key:

            ``aspect_ratio`` (float):  Width / height of the display area.
            Defaults to 1.0.

    Returns:
        matplotlib.figure.Figure:  The completed figure.
    """

    config = {"aspect_ratio": 1.0}
    config.update(figure_config or {})

    fig_width = 10.0
    fig, axes = pyplot.subplots(
        1, 1, figsize=(fig_width, fig_width / config["aspect_ratio"])
    )

    edge_only_markers = set("x+.,1234|_")
    has_data = False
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        x_values, y_values, _ = get_xy_series_data(
            series, x_diagnostic, y_diagnostic, db_session
        )
        x_values = list(x_values)
        y_values = list(y_values)
        if not x_values:
            continue

        has_data = True
        marker = series["marker"]
        color = series["color"]
        size = float(series.get("scale", 1.0))

        axes.scatter(
            x_values,
            y_values,
            marker=marker,
            s=size * 20,
            edgecolors=color if marker in edge_only_markers else "none",
            facecolors="none" if marker in edge_only_markers else color,
            label=series["label"],
        )

    if not has_data:
        axes.text(
            0.5,
            0.5,
            "Select diagnostics to display",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
    else:
        axes.set_xlabel(x_diagnostic)
        axes.set_ylabel(y_diagnostic)
        axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


def update_diag_vs_diag_plot(request, x_diagnostic, y_diagnostic):
    """Generate the diag-vs-diag plot for the active series (AJAX endpoint).

    Expects a JSON POST body with a ``datasets`` dict keyed by series id,
    each value containing ``color``, ``marker``, ``scale``, and ``label``.
    An optional ``figure_config`` dict may supply an ``aspect_ratio``.
    """

    post_data = json.loads(request.body.decode())
    datasets = post_data.get("datasets", {})
    series_list = [
        {"id": series_id, **config} for series_id, config in datasets.items()
    ]
    figure_config = post_data.get("figure_config")

    matplotlib.use("svg")
    pyplot.style.use("dark_background")

    with start_db_session() as db_session:
        fig = create_diag_vs_diag_figure(
            series_list, x_diagnostic, y_diagnostic, db_session, figure_config
        )

    with StringIO() as svg_stream:
        fig.savefig(svg_stream, bbox_inches="tight", format="svg")
        pyplot.close(fig)
        return JsonResponse({"plot_data": svg_stream.getvalue()})


def display_diag_vs_diag(request, x_diagnostic, y_diagnostic):
    """Display the diagnostic-vs-diagnostic scatter plot page."""

    with start_db_session() as db_session:
        context = get_available_series_for_pair(
            x_diagnostic, y_diagnostic, db_session
        )
        context["available_diagnostics"] = get_available_diagnostics(db_session)

    context["x_diagnostic"] = x_diagnostic
    context["y_diagnostic"] = y_diagnostic
    context["diagnostics_title"] = f"{x_diagnostic} vs {y_diagnostic}"
    context["update_plot_url"] = reverse(
        "diagnostics:update_diag_vs_diag_plot",
        kwargs={
            "x_diagnostic": x_diagnostic,
            "y_diagnostic": y_diagnostic,
        },
    )

    return render(request, "diagnostics/diag_vs_diag.html", context)
