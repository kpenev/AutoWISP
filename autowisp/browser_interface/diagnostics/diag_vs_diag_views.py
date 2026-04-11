"""Views for diagnostic-vs-diagnostic scatter plots."""

from matplotlib import pyplot
from sqlalchemy import select, func
from sqlalchemy.orm import aliased

from django.shortcuts import render
from django.urls import reverse

from autowisp.browser_interface.core.plot_utils import channel_colors
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

from .image_diagnostics_views import (
    get_available_diagnostics,
    plot_image_diagnostic_series,
)


def _get_series_query(x_diagnostic, y_diagnostic):
    """Return DB query needed to find available series for plotting."""

    x_quantile = x_diagnostic == "quantiles"
    y_quantile = y_diagnostic == "quantiles"

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
            ObservingSession.id
            == Image.observing_session_id,  # pylint: disable=no-member
        )
        .join(x_type, x_type.id == x_diag.diagnostic_id)
        .join(
            y_diag,
            (y_diag.image_id == Image.id)  # pylint: disable=no-member
            & (y_diag.channel == x_diag.channel),
        )
        .join(y_type, y_type.id == y_diag.diagnostic_id)
        .where(
            (
                x_type.name.like("pixel_q%")
                if x_quantile
                else x_type.name == x_diagnostic
            ),
            (
                y_type.name.like("pixel_q%")
                if y_quantile
                else y_type.name == y_diagnostic
            ),
        )
    )

    if x_quantile:
        query = (
            query.add_columns(x_type.name)
            .group_by(ObservingSession.id, x_diag.channel, x_type.id)
            .order_by(ObservingSession.label, x_diag.channel, x_type.name)
        )
    elif y_quantile:
        query = (
            query.add_columns(y_type.name)
            .group_by(ObservingSession.id, x_diag.channel, y_type.id)
            .order_by(ObservingSession.label, x_diag.channel, y_type.name)
        )
    else:
        query = query.group_by(ObservingSession.id, x_diag.channel).order_by(
            ObservingSession.label, x_diag.channel
        )
    return query


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

    selected_quantile = (
        x_diagnostic == "quantiles" or y_diagnostic == "quantiles"
    )

    query = _get_series_query(x_diagnostic, y_diagnostic)

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
        if selected_quantile:
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
    if selected_quantile:
        fields.append("Quantile")
    fields.append("Count")

    return {
        "diagnostics_fields": fields,
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
    channel = series["channel"]

    quantile_name = (
        "_".join(parts[2:])
        if x_diagnostic == "quantiles" or y_diagnostic == "quantiles"
        else None
    )
    x_diag_name = quantile_name if x_diagnostic == "quantiles" else x_diagnostic
    y_diag_name = quantile_name if y_diagnostic == "quantiles" else y_diagnostic

    x_diag = aliased(ImageDiagnostics)
    y_diag = aliased(ImageDiagnostics)
    x_type = aliased(DiagnosticType)
    y_type = aliased(DiagnosticType)

    rows = db_session.execute(
        select(
            x_diag.value, y_diag.value, Image.id  # pylint: disable=no-member
        )  # pylint: disable=no-member
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
            Image.observing_session_id  # pylint: disable=no-member
            == session_id,  # pylint: disable=no-member
            x_diag.channel == channel,
            x_type.name == x_diag_name,
            y_type.name == y_diag_name,
        )
    ).all()

    if not rows:
        return (), (), ()

    return zip(*rows)


def create_diag_vs_diag_figure(
    series_list,
    *,
    x_diagnostic,
    y_diagnostic,
    db_session,
    figure_config=None,
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

    figure_config = figure_config or {}
    show_legend = figure_config.get("show_legend", True)
    aspect_ratio = figure_config.get("aspect_ratio", 1.0)

    fig_width = 10.0
    fig, axes = pyplot.subplots(
        1, 1, figsize=(fig_width, fig_width / aspect_ratio)
    )

    has_data = False
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        x_values, y_values, image_ids = get_xy_series_data(
            series, x_diagnostic, y_diagnostic, db_session
        )
        x_values = list(x_values)
        y_values = list(y_values)
        if not x_values:
            continue

        has_data = True
        plot_image_diagnostic_series(
            axes,
            x_values,
            y_values,
            image_ids,
            series,
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
        if show_legend:
            axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


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
    context["download_pdf_url"] = reverse(
        "diagnostics:download_diag_vs_diag_plot",
        kwargs={
            "x_diagnostic": x_diagnostic,
            "y_diagnostic": y_diagnostic,
        },
    )

    return render(request, "diagnostics/diagnostics_app.html", context)
