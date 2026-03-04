"""Views for displaying per-image diagnostics."""

import json
import math
from io import StringIO

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
from sqlalchemy import select, func
import numpy

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


# pylint: enable=no-name-in-module
def get_available_diagnostic_series(diagnostic_name, db_session):
    """
    Return the observing sessions and channels with data for a diagnostic.

    Queries for distinct (observing_session, channel) pairs that have at
    least one value for the given diagnostic name.

    Args:
        diagnostic_name(str):    The name of the diagnostic to query
            (must match a :class:`DiagnosticType` row).

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:
            A dictionary with two keys:

            ``diagnostics_fields``:
                A list of column header strings for the extra table columns
                in the ``diagnostics_app.html`` template.

            ``diagnostics_list``:
                A list of dicts, one per (observing session, channel) pair,
                each containing the keys ``id``, ``color``, ``marker``,
                ``scale``, ``label``, and ``info`` (a list of values
                matching ``diagnostics_fields``).
    """

    is_quantile = diagnostic_name == "quantiles"

    query = (
        select(
            ObservingSession.label,
            ObservingSession.id,
            ImageDiagnostics.channel,
            func.count(ImageDiagnostics.id),  # pylint: disable=not-callable
        )
        .join(
            Image,
            Image.id == ImageDiagnostics.image_id,  # pylint: disable=no-member
        )
        .join(
            ObservingSession,
            ObservingSession.id
            == Image.observing_session_id,  # pylint: disable=no-member
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
    )

    if is_quantile:
        query = (
            query.add_columns(DiagnosticType.name)
            .where(DiagnosticType.name.like("pixel_q%"))
            .group_by(
                ObservingSession.id,
                ImageDiagnostics.channel,
                DiagnosticType.id,
            )
            .order_by(
                ObservingSession.label,
                ImageDiagnostics.channel,
                DiagnosticType.name,
            )
        )
    else:
        query = (
            query.where(DiagnosticType.name == diagnostic_name)
            .group_by(ObservingSession.id, ImageDiagnostics.channel)
            .order_by(ObservingSession.label, ImageDiagnostics.channel)
        )

    channel_colors = {"R": "#ff0000", "G": "#00ff00", "B": "#0000ff"}

    diagnostics_list = []
    for row in db_session.execute(query).all():
        session_label, session_id, channel, count = row[:4]
        series = {
            "color": channel_colors.get(
                channel[0].upper() if channel else "", "#ffffff"
            ),
            "marker": "o",
            "scale": "1.0",
        }
        if is_quantile:
            quantile_name = row[4]
            quantile_label = "0." + quantile_name[len("pixel_q") :]
            series["id"] = f"{session_id}_{channel}_{quantile_name}"
            series["label"] = f"{session_label} {channel} {quantile_label}"
            series["info"] = [session_label, channel, quantile_label, count]
        else:
            series["id"] = f"{session_id}_{channel}"
            series["label"] = f"{session_label} {channel}"
            series["info"] = [session_label, channel, count]

        diagnostics_list.append(series)

    fields = ["Observing Session", "Channel"]
    if is_quantile:
        fields.append("Quantile")
    fields.append("Count")

    return {
        "diagnostics_fields": fields,
        "diagnostics_list": diagnostics_list,
    }


def get_diagnostic_series_data(series, diagnostic_name, db_session):
    """
    Query the JD and diagnostic values for a single series.

    Args:
        series(dict):    A series entry as produced by
            :func:`get_available_diagnostic_series`.

        diagnostic_name(str):    The diagnostic type name to query, or
            ``"quantiles"`` (in which case the quantile name is extracted
            from the series ``id``).

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    ``(jd_values, diag_values)`` as tuples of floats,
            ordered by JD.  Empty tuples if no data is found.
    """

    parts = series["id"].split("_")
    session_id = int(parts[0])
    channel = parts[1]

    if diagnostic_name == "quantiles":
        query_diag_name = "_".join(parts[2:])
    else:
        query_diag_name = diagnostic_name

    rows = db_session.execute(
        select(Image.jd, ImageDiagnostics.value)  # pylint: disable=no-member
        .join(
            ImageDiagnostics,
            ImageDiagnostics.image_id == Image.id,  # pylint: disable=no-member
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
        .where(
            Image.observing_session_id  # pylint: disable=no-member
            == session_id,
            ImageDiagnostics.channel == channel,
            DiagnosticType.name == query_diag_name,
            Image.jd.is_not(None),  # pylint: disable=no-member
        )
        .order_by(Image.jd)  # pylint: disable=no-member
    ).all()

    if not rows:
        return (), ()

    return zip(*rows)


def plot_image_diagnostic_series(axes, time_values, diag_values, config):
    """
    Plot a single image diagnostic series on the given axes.

    Args:
        axes:    A matplotlib Axes to plot on.

        time_values:    Sequence of Julian date x-coordinates.

        diag_values:    Sequence of diagnostic y-coordinates.

        config(dict):    Configuration for the plotting usually produce by
            :func:`get_available_diagnostic_series`. Should contain keys
            ``color``, ``marker``, ``scale``, and ``label``.
    """

    axes.plot(
        time_values,
        diag_values,
        linestyle="none",
        marker=config["marker"],
        markersize=float(config.get("scale", 1.0)),
        markeredgecolor=(
            config["color"] if config["marker"] in "x+.,1234|_" else "none"
        ),
        markerfacecolor=config["color"],
        label=config["label"],
    )


def group_series_by_jd_overlap(series_data):
    """
    Group diagnostic series into sets that share overlapping JD ranges.

    Series with the same diagnostic whose JD ranges overlap are grouped
    together (to be plotted on the same axes).  Series with non-overlapping
    ranges end up in separate groups.

    Args:
        series_data(list):    A list of ``(series, jd_values, diag_values)``
            tuples, where *jd_values* are ordered sequences of Julian dates.

    Returns:
        list:    A list of lists, each inner list containing
            ``(series, jd_values, diag_values)`` tuples that should be
            plotted on the same axes.
    """

    groups = []
    group_ranges = []
    for entry in series_data:
        jd_values = entry[1]
        if not jd_values:
            continue
        jd_min = min(jd_values)
        jd_max = max(jd_values)

        overlapping = [
            i
            for i, (g_min, g_max) in enumerate(group_ranges)
            if jd_min <= g_max and jd_max >= g_min
        ]

        if not overlapping:
            groups.append([entry])
            group_ranges.append((jd_min, jd_max))
        else:
            target = overlapping[0]
            groups[target].append(entry)
            merged_min = min(jd_min, group_ranges[target][0])
            merged_max = max(jd_max, group_ranges[target][1])
            for i in reversed(overlapping[1:]):
                groups[target].extend(groups.pop(i))
                merged_min = min(merged_min, group_ranges[i][0])
                merged_max = max(merged_max, group_ranges[i][1])
                group_ranges.pop(i)
            group_ranges[target] = (merged_min, merged_max)

    return groups


def create_figure(num_plots, plot_height_frac, aspect_ratio, num_columns):
    """Create the figure for the diagnostics plot per given configuration."""

    if num_plots == 0:
        fig = Figure(figsize=(10, 2))
        axes = fig.add_subplot(111)
        axes.text(
            0.5,
            0.5,
            "Select diagnostics to display",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        return fig, None

    fig_width = 10
    num_rows = math.ceil(num_plots / num_columns)
    plot_height_frac = max(plot_height_frac, 1.0 / num_rows)
    row_height = fig_width / aspect_ratio * plot_height_frac
    fig_height = row_height * num_rows

    fig, all_axes = pyplot.subplots(
        num_rows,
        num_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for group_index in range(num_plots, num_rows * num_columns):
        row = group_index // num_columns
        col = group_index % num_columns
        all_axes[row][col].set_visible(False)

    return fig, all_axes


def create_image_diagnostics_figure(
    series_list,
    diagnostic_name,
    db_session,
    overwrite_figure_config=None,
):
    """
    Create a multi-panel figure for the selected image diagnostic series.

    Args:
        series_list(list):    Series entries (as produced by
            :func:`get_available_diagnostic_series`) to plot.  Only
            entries whose ``marker`` is non-empty are plotted.

        diagnostic_name(str):    The diagnostic type name, or
            ``"quantiles"``.

        db_session:    An active SQLAlchemy database session.

        figure_config(dict):    Configuration for the layout of the figure.
            Should define:

                plot_height_frac(float):    Height of each subplot row as a
                    fraction of the available screen area.

                num_columns(int):    Number of columns in the subplot grid.

                aspect_ratio(float):    Width / height of the available screen
                    area.
            By default, height fraction is 1/3, number of columns is 1, and
            aspect ratio is 5.0.

    Returns:
        matplotlib.figure.Figure:    The completed figure.
    """

    figure_config = {
        "plot_height_frac": 1.0 / 3.0,
        "num_columns": 1,
        "aspect_ratio": 3.0,
    }
    figure_config.update(overwrite_figure_config or {})

    series_data = []
    min_jd = numpy.inf
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        jd_values, diag_values = get_diagnostic_series_data(
            series, diagnostic_name, db_session
        )
        jd_values = numpy.atleast_1d(jd_values)
        if jd_values.size:
            min_jd = min(min_jd, numpy.nanmin(jd_values))
            series_data.append((series, jd_values, diag_values))

    groups = group_series_by_jd_overlap(series_data)
    fig, all_axes = create_figure(len(groups), **figure_config)
    if all_axes is None:
        return fig

    for axes, group in zip(all_axes.flatten(), groups):
        for series, jd_values, diag_values in group:
            plot_image_diagnostic_series(
                axes, jd_values - min_jd, diag_values, series
            )
        axes.set_xlabel(f"JD - {min_jd!r}")
        axes.set_ylabel(diagnostic_name)
        axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


def update_image_diagnostics_plot(request, diagnostic_name):
    """Generate the image diagnostics plot for the series selected by the user.

    Expects a JSON POST body with a ``datasets`` dict keyed by series id,
    each value containing ``color``, ``marker``, ``scale``, and ``label``.
    An optional ``figure_config`` dict may override the default layout.
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
        fig = create_image_diagnostics_figure(
            series_list, diagnostic_name, db_session, figure_config
        )

    with StringIO() as svg_stream:
        fig.savefig(svg_stream, bbox_inches="tight", format="svg")
        pyplot.close(fig)
        return JsonResponse({"plot_data": svg_stream.getvalue()})


def get_available_diagnostics(db_session):
    """Return the list of diagnostic names that have at least one value."""

    names = [
        row[0]
        for row in db_session.execute(
            select(DiagnosticType.name)
            .join(
                ImageDiagnostics,
                ImageDiagnostics.diagnostic_id == DiagnosticType.id,
            )
            .group_by(DiagnosticType.id)
            .order_by(DiagnosticType.id)
        ).all()
    ]

    quantile_names = [n for n in names if n.startswith("pixel_q")]
    other_names = [n for n in names if not n.startswith("pixel_q")]

    result = other_names[:]
    if quantile_names:
        result.append("quantiles")

    return result


def display_image_diagnostics(request, diagnostic_name):
    """View displaying the table of available series for an image diagnostic."""

    with start_db_session() as db_session:
        context = get_available_diagnostic_series(diagnostic_name, db_session)
        context["available_diagnostics"] = get_available_diagnostics(db_session)
    context["diagnostics_title"] = diagnostic_name
    context["update_plot_url"] = reverse(
        "diagnostics:update_image_diagnostics_plot",
        kwargs={"diagnostic_name": diagnostic_name},
    )

    return render(
        request,
        "diagnostics/diagnostics_app.html",
        context,
    )
