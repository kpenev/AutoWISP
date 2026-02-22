"""Views for displaying per-image diagnostics."""

from sqlalchemy import select, func

from django.shortcuts import render

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
            func.count(ImageDiagnostics.id),
        )
        .join(Image, Image.id == ImageDiagnostics.image_id)
        .join(
            ObservingSession,
            ObservingSession.id == Image.observing_session_id,
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
    )

    if is_quantile:
        query = (
            query
            .add_columns(DiagnosticType.name)
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
            query
            .where(DiagnosticType.name == diagnostic_name)
            .group_by(ObservingSession.id, ImageDiagnostics.channel)
            .order_by(ObservingSession.label, ImageDiagnostics.channel)
        )

    rows = db_session.execute(query).all()

    channel_colors = {"R": "#ff0000", "G": "#00ff00", "B": "#0000ff"}

    diagnostics_list = []
    for row in rows:
        session_label, session_id, channel, count = row[:4]
        if is_quantile:
            quantile_name = row[4]
            quantile_label = "0." + quantile_name[len("pixel_q"):]
            series_id = f"{session_id}_{channel}_{quantile_name}"
            label = f"{session_label} {channel} {quantile_label}"
            info = [session_label, channel, quantile_label, count]
        else:
            series_id = f"{session_id}_{channel}"
            label = f"{session_label} {channel}"
            info = [session_label, channel, count]

        diagnostics_list.append(
            {
                "id": series_id,
                "color": channel_colors.get(
                    channel[0].upper() if channel else "", "#ffffff"
                ),
                "marker": "o",
                "scale": "1.0",
                "label": label,
                "info": info,
            }
        )

    fields = ["Observing Session", "Channel"]
    if is_quantile:
        fields.append("Quantile")
    fields.append("Count")

    return {
        "diagnostics_fields": fields,
        "diagnostics_list": diagnostics_list,
    }


def display_image_diagnostics(request, diagnostic_name):
    """View displaying the table of available series for an image diagnostic."""

    with start_db_session() as db_session:
        context = get_available_diagnostic_series(
            diagnostic_name, db_session
        )
    context["diagnostics_title"] = diagnostic_name

    return render(
        request,
        "diagnostics/diagnostics_app.html",
        context,
    )
