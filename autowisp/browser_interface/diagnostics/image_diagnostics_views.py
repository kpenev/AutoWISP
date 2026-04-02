"""Views for displaying per-image diagnostics."""

from io import BytesIO
import json
import math

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
from sqlalchemy import select, func
import numpy

from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse

from autowisp.browser_interface.core.plot_utils import (
    channel_colors,
    edge_only_markers,
    setup_svg_matplotlib,
    figure_to_svg_response,
)
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

    diagnostics_list = []
    for row in db_session.execute(query).all():
        session_label, session_id, channel, count = row[:4]
        series = {
            "channel": channel,
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
        tuple:    ``(jd_values, diag_values, image_ids)`` as tuples of
            floats/ints, ordered by JD.  Empty tuples if no data is found.
    """

    parts = series["id"].split("_")
    session_id = int(parts[0])
    channel = series["channel"]

    if diagnostic_name == "quantiles":
        query_diag_name = "_".join(parts[2:])
    else:
        query_diag_name = diagnostic_name

    rows = db_session.execute(
        select(  # pylint: disable=no-member
            Image.jd,  # pylint: disable=no-member
            ImageDiagnostics.value,
            Image.id,  # pylint: disable=no-member
        )
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
        return (), (), ()

    return zip(*rows)


def plot_image_diagnostic_series(
    axes, time_values, diag_values, image_ids, config
):
    """
    Plot a single image diagnostic series on the given axes.

    Args:
        axes:    A matplotlib Axes to plot on.

        time_values:    Sequence of Julian date x-coordinates.

        diag_values:    Sequence of diagnostic y-coordinates.

        config(dict):    Configuration for the plotting usually produce by
            :func:`get_available_diagnostic_series`. Should contain keys
            ``channel``, ``color``, ``marker``, ``scale``, and ``label``.
    """

    marker = config["marker"]
    color = config["color"]
    size = float(config.get("scale", 1.0))

    collection = axes.scatter(
        time_values,
        diag_values,
        marker=marker,
        s=size * 20,
        edgecolors=color if marker in edge_only_markers else "none",
        facecolors="none" if marker in edge_only_markers else color,
        label=config["label"],
    )
    collection.set_urls([
        reverse(
            "diagnostics:preview_calibrated_image",
            kwargs={"image_id": img_id, "color_channel": config["channel"]},
        )
        for img_id in image_ids
    ])


def group_series_by_jd_overlap(series_data):
    """
    Group diagnostic series into sets that share overlapping JD ranges.

    Series with the same diagnostic whose JD ranges overlap are grouped
    together (to be plotted on the same axes).  Series with non-overlapping
    ranges end up in separate groups.

    Args:
        series_data(list):    A list of ``(series, jd_values, diag_values, image_ids)``
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
        if not jd_values.size:
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
    *,
    diagnostic_name,
    db_session,
    figure_config=None,
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

    figure_config = figure_config or {}
    show_legend = figure_config.pop("show_legend", True)
    figure_config.setdefault("plot_height_frac", 1.0 / 3.0)
    figure_config.setdefault("num_columns", 1)
    figure_config.setdefault("aspect_ratio", 3.0)

    series_data = []
    min_jd = numpy.inf
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        jd_values, diag_values, image_ids = get_diagnostic_series_data(
            series, diagnostic_name, db_session
        )
        jd_values = numpy.atleast_1d(jd_values)
        if jd_values.size:
            min_jd = min(min_jd, numpy.nanmin(jd_values))
            series_data.append((series, jd_values, diag_values, image_ids))

    groups = group_series_by_jd_overlap(series_data)
    fig, all_axes = create_figure(len(groups), **figure_config)
    if all_axes is None:
        return fig

    for axes, group in zip(all_axes.flatten(), groups):
        for series, jd_values, diag_values, image_ids in group:
            plot_image_diagnostic_series(
                axes, jd_values - min_jd, diag_values, image_ids, series
            )
        axes.set_xlabel(f"JD - {min_jd!r}")
        axes.set_ylabel(diagnostic_name)
        if show_legend:
            axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


def update_plot_view(request, figure_factory, session_key=None, **url_kwargs):
    """Common handler for diagnostics AJAX plot-update views.

    Parses the JSON POST body, calls ``figure_factory`` to produce the figure,
    and returns an SVG ``JsonResponse``.

    Args:
        request:        Django HTTP request whose body is a JSON object with a
                        ``datasets`` dict (keyed by series id) and an optional
                        ``figure_config`` dict.
        figure_factory: Callable accepting ``series_list``, ``db_session``,
                        ``figure_config``, plus any URL kwargs as keyword
                        arguments.
        session_key:    If given, the raw POST data is stored in the session
                        under this key so a download view can retrieve it.

    Returns:
        JsonResponse with ``plot_data`` containing the SVG string.
    """
    post_data = json.loads(request.body.decode())
    if session_key:
        request.session[session_key] = post_data
        request.session.modified = True
    series_list = [
        {"id": series_id, **config}
        for series_id, config in post_data.get("datasets", {}).items()
    ]
    figure_config = post_data.get("figure_config")

    setup_svg_matplotlib()

    with start_db_session() as db_session:
        fig = figure_factory(
            series_list,
            db_session=db_session,
            figure_config=figure_config,
            **url_kwargs,
        )

    return figure_to_svg_response(fig)


def download_plot_view(request, figure_factory, session_key, **url_kwargs):
    """Return the last-plotted figure as a PDF download.

    Reads the plot configuration stored in the session by a previous call to
    :func:`update_plot_view` and regenerates the figure in PDF format.

    Args:
        request:        Django HTTP request.
        figure_factory: Same factory used by the corresponding update view.
        session_key:    Session key where :func:`update_plot_view` stored the
                        last POST data.

    Returns:
        HttpResponse with PDF content.
    """
    post_data = request.session.get(session_key, {})
    series_list = [
        {"id": series_id, **config}
        for series_id, config in post_data.get("datasets", {}).items()
    ]
    figure_config = post_data.get("figure_config", {}).copy()
    figure_config.pop("show_legend", None)

    matplotlib.use("pdf")
    pyplot.style.use("default")

    with start_db_session() as db_session:
        fig = figure_factory(
            series_list,
            db_session=db_session,
            figure_config=figure_config,
            **url_kwargs,
        )

    with BytesIO() as pdf_stream:
        fig.savefig(pdf_stream, bbox_inches="tight", format="pdf")
        pyplot.close(fig)
        return HttpResponse(
            pdf_stream.getvalue(),
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="diagnostics.pdf"',
            },
        )


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
    context["download_pdf_url"] = reverse(
        "diagnostics:download_image_diagnostics_plot",
        kwargs={"diagnostic_name": diagnostic_name},
    )

    return render(
        request,
        "diagnostics/diagnostics_app.html",
        context,
    )
