"""Views for displaying per-image diagnostics."""

from collections import defaultdict
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

CLOUD_BRIGHTNESS_DIAGNOSTIC = "local_sky_brightness_minmax_frac"
CLOUD_BRIGHTNESS_MINMAX_THRESHOLD = 0.0168
CLOUD_FLAG_COLOR = "#f6b44b"


def _diagnostic_exists(db_session, diagnostic_name):
    """Return whether the named diagnostic has at least one value."""

    return (
        db_session.scalar(
            select(func.count(ImageDiagnostics.id))
            .join(
                DiagnosticType,
                DiagnosticType.id == ImageDiagnostics.diagnostic_id,
            )
            .where(DiagnosticType.name == diagnostic_name)
        )
        > 0
    )


def _collect_saved_cloud_metric(db_session, diagnostic_name):
    """Return per-session saved cloud metric values."""

    rows = db_session.execute(
        select(
            ImageDiagnostics.image_id,
            Image.observing_session_id,
            Image.jd,
            ImageDiagnostics.value,
        )
        .join(Image, Image.id == ImageDiagnostics.image_id)
        .join(DiagnosticType, DiagnosticType.id == ImageDiagnostics.diagnostic_id)
        .where(DiagnosticType.name == diagnostic_name)
    ).all()

    by_session = defaultdict(list)
    for image_id, session_id, jd, value in rows:
        by_session[session_id].append((image_id, jd, float(value)))
    return by_session


def _format_metric_entry(entry):
    """Return a compact serializable description of one metric sample."""

    image_id, jd, value = entry
    return {
        "image_id": int(image_id),
        "jd": None if jd is None else float(jd),
        "value": round(float(value), 6),
    }


def _get_cloud_detection(db_session, selected_image_ids=None):
    """Return cloudy image ids and a debug report for the quantiles page."""

    selected_image_ids = set(selected_image_ids or [])
    evaluated = _diagnostic_exists(db_session, CLOUD_BRIGHTNESS_DIAGNOSTIC)
    report = {
        "evaluated": evaluated,
        "signal": "saved local sky brightness range" if evaluated else None,
        "signal_description": (
            "fractional range of star-suppressed sky brightness across "
            "image blocks"
            if evaluated
            else None
        ),
        "rule": (
            "flag frames when "
            f"{CLOUD_BRIGHTNESS_DIAGNOSTIC} >= "
            f"{CLOUD_BRIGHTNESS_MINMAX_THRESHOLD:g}"
            if evaluated
            else None
        ),
        "sessions_evaluated": 0,
        "usable_pairs": 0,
        "frames_flagged": 0,
        "selected_flagged": 0,
        "groups": [],
        "reasons": [],
    }

    if not evaluated:
        report["reasons"].append(
            f"no saved {CLOUD_BRIGHTNESS_DIAGNOSTIC} metric"
        )
        return set(), report

    metric_by_session = _collect_saved_cloud_metric(
        db_session, CLOUD_BRIGHTNESS_DIAGNOSTIC
    )
    if not any(metric_by_session.values()):
        report["reasons"].append(
            f"no same-image {CLOUD_BRIGHTNESS_DIAGNOSTIC} values"
        )
        return set(), report

    cloudy_ids = set()
    for session_id, entries in sorted(metric_by_session.items()):
        entries = sorted(entries, key=lambda entry: (entry[1] is None, entry[1]))
        values = numpy.array([value for _, _, value in entries], dtype=float)
        flagged_entries = [
            entry
            for entry in entries
            if float(entry[2]) >= CLOUD_BRIGHTNESS_MINMAX_THRESHOLD
        ]
        for image_id, _, _ in flagged_entries:
            cloudy_ids.add(image_id)

        group = {
            "session_id": int(session_id) if session_id is not None else None,
            "usable_pairs": int(values.size),
            "flagged": len(flagged_entries),
            "baseline": None,
            "threshold": CLOUD_BRIGHTNESS_MINMAX_THRESHOLD,
            "mad": None,
            "min_ratio": (
                round(float(numpy.min(values)), 6) if values.size else None
            ),
            "max_ratio": (
                round(float(numpy.max(values)), 6) if values.size else None
            ),
            "examples": [_format_metric_entry(entry) for entry in entries[:3]],
            "flagged_examples": [
                _format_metric_entry(entry) for entry in flagged_entries[:3]
            ],
            "reason": "",
        }
        if not flagged_entries:
            group["reason"] = "all brightness ranges below threshold"
        report["groups"].append(group)
        report["usable_pairs"] += int(values.size)

    report["sessions_evaluated"] = len(metric_by_session)
    report["frames_flagged"] = len(cloudy_ids)
    report["selected_flagged"] = len(cloudy_ids & selected_image_ids)

    if not cloudy_ids:
        report["reasons"].append("all brightness ranges below threshold")
    elif selected_image_ids and not (cloudy_ids & selected_image_ids):
        report["reasons"].append("selected rows do not include flagged points")

    return cloudy_ids, report


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

    return tuple(zip(*rows))


def plot_image_diagnostic_series(
    axes,
    time_values,
    diag_values,
    image_ids,
    config,
    *,
    cloudy_ids=None,
    cloudy_color=CLOUD_FLAG_COLOR,
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

    image_ids_list = list(image_ids)
    point_colors = color
    if cloudy_ids:
        point_colors = [
            cloudy_color if int(img_id) in cloudy_ids else color
            for img_id in image_ids_list
        ]

    collection = axes.scatter(
        time_values,
        diag_values,
        marker=marker,
        s=size * 20,
        c=point_colors,
        label=config["label"],
    )
    collection.set_urls([
        reverse(
            "diagnostics:preview_calibrated_image",
            kwargs={"image_id": img_id, "color_channel": config["channel"]},
        )
        for img_id in image_ids_list
    ])


def group_series_by_jd_overlap(series_data):
    """
    Group diagnostic series into sets that share overlapping JD ranges.

    Series with the same diagnostic whose JD ranges overlap are grouped
    together (to be plotted on the same axes).  Series with non-overlapping
    ranges end up in separate groups.

    Args:
        series_data(list):    A list of
            ``(series, jd_values, diag_values, image_ids)``
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

    series_data = []
    cloudy_ids = set()
    min_jd = numpy.inf
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        jd_values, diag_values, image_ids = get_diagnostic_series_data(
            series, diagnostic_name, db_session
        )
        jd_values = numpy.atleast_1d(jd_values)
        diag_values = numpy.atleast_1d(diag_values)
        image_ids = numpy.atleast_1d(image_ids)
        if jd_values.size:
            min_jd = min(min_jd, numpy.nanmin(jd_values))
            series_data.append((series, jd_values, diag_values, image_ids))

    if diagnostic_name == "quantiles":
        selected_ids = {
            int(image_id)
            for _, _, _, image_ids in series_data
            for image_id in image_ids
        }
        cloudy_ids, _ = _get_cloud_detection(db_session, selected_ids)

    groups = group_series_by_jd_overlap(series_data)
    fig, all_axes = create_figure(
        len(groups),
        plot_height_frac=figure_config.get("plot_height_frac", 1.0 / 3.0),
        aspect_ratio=figure_config.get("aspect_ratio", 3.0),
        num_columns=figure_config.get("num_columns", 1),
    )
    if all_axes is None:
        return fig

    for axes, group in zip(all_axes.flatten(), groups):
        for series, jd_values, diag_values, image_ids in group:
            plot_image_diagnostic_series(
                axes,
                jd_values - min_jd,
                diag_values,
                image_ids,
                series,
                cloudy_ids=cloudy_ids,
            )
        axes.set_xlabel(f"JD - {min_jd!r}")
        axes.set_ylabel(diagnostic_name)
        if figure_config.get("show_legend", True):
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
    figure_config = post_data.get("figure_config")

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
    if diagnostic_name == "quantiles":
        context["cloudy_note"] = (
            "Frames flagged as cloudy use the highlight color "
            "(uses the saved local star-suppressed sky brightness range "
            "across image blocks)."
        )
        context["cloudy_color"] = CLOUD_FLAG_COLOR
    context["diagnostics_title"] = diagnostic_name
    context["y_diagnostic"] = diagnostic_name
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
