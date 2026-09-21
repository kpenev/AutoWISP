"""Views for displaying per-image diagnostics.

One quantity may be plotted against another, where a quantity is a
``DiagnosticType`` name, the ``pixel_quantiles`` pseudo-name expanding to
one series per ``pixel_q*``, or ``jd``.  Plotting against time is not a separate
mode: it is ``x="jd"``, which resolves through the same path as everything
else because the canonical image list already carries the Julian dates.

The figure half of the page: reading the values a row asks for, drawing
them, and the Django views that serve it. What the *table* above the plot
offers, and what a row binds, is :mod:`series_table` -- which never
evaluates anything, where everything here does.
"""

from io import BytesIO
import json
import math

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
import numpy

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from autowisp.browser_interface.core.plot_utils import (
    line_styles,
    setup_svg_matplotlib,
    figure_to_svg_response,
)
from autowisp.database.interface import start_db_session
from autowisp.diagnostics.expression_series import (
    get_quantity_values,
    time_quantity,
)
from autowisp.diagnostics.expressions import get_quantity_arity

from .series_table import (
    get_available_diagnostics,
    get_available_series,
    get_recorded_diagnostics,
    get_series_key,
    posted_rows,
    resolve_quantity,
    split_row_id,
)


def get_series_data(series, x_diagnostic, expressions, db_session):
    """
    Query the paired x/y values for a single series.

    Both axes are resolved in one call, which is what makes them share a
    query for the diagnostics they need and one symbol table, so a
    subexpression common to the two is evaluated once.  They are returned
    unmasked; the single finite mask lives in
    :func:`plot_image_diagnostic_series`.

    Args:
        series(dict):    One row as the client posted it back, holding the
            id it was rendered with, and the pair and channels its
            dropdowns say.

        x_diagnostic(str):    Quantity on the X axis, which the page
            supplies: it is the one thing a row does not choose.  The y
            comes from the row's own id, which names the quantity it
            draws.

        expressions(dict):    The library, ``{name: expression}``, passed in
            rather than fetched so that nothing below the view has to know
            it came from the browser-interface database.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    ``(x_values, y_values, image_ids)``, all of equal length.
    """

    series_key = get_series_key(series)
    y_diagnostic, _ = split_row_id(series["id"])
    quantities = [
        resolve_quantity(quantity_name, series_key.quantile_name)
        for quantity_name in (x_diagnostic, y_diagnostic)
    ]

    # The row's channels are the two axes' bindings laid end to end, in
    # the order the columns are, so each axis takes as many as the
    # quantity it draws has parameters.
    bindings = []
    taken = 0
    for quantity in quantities:
        arity = get_quantity_arity(quantity, expressions)
        bindings.append(series_key.channels[taken : taken + arity])
        taken += arity

    wanted = {}
    for quantity, channels in zip(quantities, bindings):
        wanted.setdefault(quantity, set()).add(channels)

    values, image_ids = get_quantity_values(
        series_key, wanted, expressions, db_session
    )

    # By quantity *and* binding: the two axes may name one quantity, read
    # either in the same channels -- a plot of it against itself -- or in
    # two, which is how it is compared between them.
    return (
        values[quantities[0]][bindings[0]],
        values[quantities[1]][bindings[1]],
        image_ids,
    )


def plot_image_diagnostic_series(axes, x_values, y_values, image_ids, config):
    """
    Plot a single series on the given axes.

    Args:
        axes:    A matplotlib Axes to plot on.

        x_values:    Sequence of x coordinates.

        y_values:    Sequence of y coordinates.

        image_ids:    The image each point belongs to, used for the
            click-through URLs.

        config(dict):    Configuration for the plotting, usually produced by
            :func:`get_available_series`. Should contain keys ``channel``,
            ``color``, ``marker``, ``scale``, and ``label``. A ``marker``
            naming one of
            :data:`~autowisp.browser_interface.core.plot_utils.line_styles`
            draws a curve instead of points, reading the scale as the line
            width where points read it as the marker size.
    """

    # The arrays arrive NaN-padded to the canonical image list. Dropping the
    # non-finite entries here is what used to be an inner join between the
    # two axes, and image_ids must be masked with them so the per-point
    # click-through stays aligned with the drawn markers.
    x_values = numpy.atleast_1d(x_values)
    y_values = numpy.atleast_1d(y_values)
    keep = numpy.isfinite(x_values) & numpy.isfinite(y_values)
    x_values, y_values = x_values[keep], y_values[keep]
    image_ids = numpy.asarray(image_ids)[keep]

    if config["marker"] in line_styles:
        # A curve has to be walked in x order. Against jd the canonical
        # image list already is ordered, but nothing orders an arbitrary
        # quantity, and an unsorted line is a scribble rather than a curve.
        # No per-point URLs either: a Line2D carries one for the whole
        # artist, so clicking through to a frame stays the business of the
        # series drawn as points beneath it.
        in_x_order = numpy.argsort(x_values, kind="stable")
        axes.plot(
            x_values[in_x_order],
            y_values[in_x_order],
            line_styles[config["marker"]],
            linewidth=float(config.get("scale", 1.0)),
            color=config["color"],
            label=config["label"],
        )
        return

    collection = axes.scatter(
        x_values,
        y_values,
        marker=config["marker"],
        s=float(config.get("scale", 1.0)) * 20,
        c=config["color"],
        label=config["label"],
    )
    collection.set_urls(
        [
            reverse(
                "diagnostics:preview_calibrated_image",
                kwargs={
                    "image_id": img_id,
                    "color_channel": config["channel"],
                },
            )
            for img_id in image_ids
        ]
    )


def group_series_by_x_overlap(series_data):
    """
    Group series into sets whose x ranges overlap.

    Series whose x ranges overlap share axes; disjoint ones get their own.
    For a time axis this separates observing nights, which is what it was
    written for.  For any other quantity the ranges normally overlap, so
    everything collapses onto a single set of axes.

    Args:
        series_data(list):    ``(series, x_values, y_values, image_ids)``
            tuples.

    Returns:
        list:    Lists of the entries that should share one set of axes.
    """

    groups = []
    group_ranges = []
    for entry in series_data:
        finite = entry[1][numpy.isfinite(entry[1])]
        if not finite.size:
            continue
        x_min = finite.min()
        x_max = finite.max()

        overlapping = [
            i
            for i, (g_min, g_max) in enumerate(group_ranges)
            if x_min <= g_max and x_max >= g_min
        ]

        if not overlapping:
            groups.append([entry])
            group_ranges.append((x_min, x_max))
        else:
            target = overlapping[0]
            groups[target].append(entry)
            merged_min = min(x_min, group_ranges[target][0])
            merged_max = max(x_max, group_ranges[target][1])
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


def collect_series_data(series_list, x_diagnostic, expressions, db_session):
    """
    Read the selected series, dropping those with nothing to draw.

    Every row of the table is posted rather than only the drawn ones, so
    that the server can see what the whole table binds.  Four of them are
    skipped here: one the user has not selected, one whose marker is
    blank, one still missing a channel, and one naming no session and
    image type -- the last two naming no data to read.

    Args:
        series_list(list):    Every row as the client posted it back.

        x_diagnostic(str):    Quantity on the X axis, shared by the whole
            figure.  Each row names its own y through its id.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(series, x_values, y_values, image_ids)`` tuples for the
            series having at least one point where both axes are finite.
    """

    series_data = []
    for series in series_list:
        # Defaulting to selected, so that a payload from before the table
        # posted every row still draws what it was asked to.
        if not series.get("selected", True):
            continue
        if not series.get("marker", "").strip():
            continue
        # ``not channels`` as well as ``all``, which an empty list passes:
        # a page whose script predates the channel columns posts none at
        # all, and binding nothing is not a binding.  A payload predating
        # the chosen pair names no population either.  Both are skipped
        # rather than refused -- a stale page should draw nothing, not
        # turn the response into an error page.
        channels = series.get("channels", ())
        if not channels or not all(channels) or not series.get("pair"):
            continue
        x_values, y_values, image_ids = get_series_data(
            series, x_diagnostic, expressions, db_session
        )
        x_values = numpy.atleast_1d(x_values)
        y_values = numpy.atleast_1d(y_values)
        # A padded array is full length even when every value is NaN, so
        # its size no longer tells us whether anything will be drawn.
        if numpy.any(numpy.isfinite(x_values) & numpy.isfinite(y_values)):
            # Two things the figure reads off the binding rather than off
            # the client: the channel a click on a point opens the frame
            # in -- the first of them, for want of a better answer once a
            # series can bind several -- and the quantity the row draws,
            # which its y axis is labelled for.
            drawn = {
                **series,
                "channel": channels[0] if channels else "",
                "quantity": split_row_id(series["id"])[0],
            }
            series_data.append((drawn, x_values, y_values, image_ids))

    return series_data


def draw_series_group(axes, group, x_offset):
    """
    Plot every series sharing one set of axes.

    Args:
        axes:    The matplotlib Axes the group was assigned.

        group(list):    ``(series, x_values, y_values, image_ids)`` tuples,
            as grouped by :func:`group_series_by_x_overlap`.

        x_offset(float):    Subtracted from every x value.  Shared by the
            whole figure so the series keep their spacing relative to each
            other.

    Returns:
        None
    """

    for series, x_values, y_values, image_ids in group:
        plot_image_diagnostic_series(
            axes, x_values - x_offset, y_values, image_ids, series
        )


def create_diagnostics_figure(
    series_list,
    *,
    x_diagnostic,
    expressions,
    db_session,
    figure_config=None,
):
    """
    Create the figure for the selected series, against one x quantity.

    Args:
        series_list(list):    Every row as the client posted it back.
            Only those with a non-empty ``marker`` are plotted.

        x_diagnostic(str):    Quantity on the X axis, which every series
            shares.  What each draws against it comes from its own id.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

        figure_config(dict):    Layout of the figure, defining
            ``plot_height_frac``, ``num_columns`` and ``aspect_ratio``.

    Returns:
        matplotlib.figure.Figure:    The completed figure.
    """

    figure_config = figure_config or {}
    against_time = x_diagnostic == time_quantity

    series_data = collect_series_data(
        series_list, x_diagnostic, expressions, db_session
    )

    # Julian dates are large numbers spanning a tiny range, so the axis is
    # offset to stay readable. One offset for the whole figure, not one per
    # series: nights must keep their spacing relative to each other.
    x_offset = 0.0
    if against_time and series_data:
        x_offset = min(numpy.nanmin(entry[1]) for entry in series_data)

    groups = group_series_by_x_overlap(series_data)
    fig, all_axes = create_figure(
        len(groups),
        plot_height_frac=figure_config.get("plot_height_frac", 1.0 / 3.0),
        aspect_ratio=figure_config.get(
            "aspect_ratio", 3.0 if against_time else 1.0
        ),
        num_columns=figure_config.get("num_columns", 1),
    )
    if all_axes is None:
        return fig

    for axes, group in zip(all_axes.flatten(), groups):
        draw_series_group(axes, group, x_offset)
        axes.set_xlabel(f"JD - {x_offset!r}" if against_time else x_diagnostic)
        # Named for what this subplot drew rather than for the page: a row
        # carries the quantity it draws, and a subplot holds the rows whose
        # x ranges overlap, which need not be all of them.  In the order
        # they were drawn, and each named once however many rows drew it.
        axes.set_ylabel(
            ", ".join(dict.fromkeys(series["quantity"] for series, *_ in group))
        )
        if figure_config.get("show_legend", True):
            axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


def update_plot_view(
    request, figure_factory, session_key=None, extra=None, **url_kwargs
):
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
        session_key:    If given, the posted plot configuration is stored in
                        the session under this key so a download view can
                        retrieve it.
        extra:          Optional callable given the whole POST, the database
                        session and the URL kwargs, returning ``(rows,
                        fields)``: the rows to draw, and what a redraw
                        answers *besides* the figure -- a rebound row's
                        count and cells -- which rides along in the
                        response so that one action costs one round trip.
                        It decides what is drawn as well as what is said,
                        because a row whose binding has just changed has to
                        be drawn with the colour and label of the binding
                        it now has.

    Returns:
        JsonResponse with ``plot_data`` containing the SVG string.
    """
    post_data = json.loads(request.body.decode())
    figure_config = post_data.get("figure_config")

    setup_svg_matplotlib()

    with start_db_session() as db_session:
        # Before the figure, not after it: the rows it hands back are the
        # ones drawn. Answering afterwards left the figure showing a
        # rebound row in the colour and legend of the binding it had just
        # left, while the table beside it showed the new ones.
        if extra is None:
            series_list, alongside = posted_rows(post_data), {}
        else:
            series_list, alongside = extra(
                post_data, db_session=db_session, **url_kwargs
            )

        if session_key:
            # Stored after ``extra`` and from what it returned, so that
            # Download Figure straight after a rebinding regenerates what
            # is on the screen rather than what the client had posted.
            request.session[session_key] = {
                **post_data,
                "datasets": {
                    row["id"]: {
                        key: value for key, value in row.items() if key != "id"
                    }
                    for row in series_list
                },
            }
            request.session.modified = True

        fig = figure_factory(
            series_list,
            db_session=db_session,
            figure_config=figure_config,
            **url_kwargs,
        )

    return figure_to_svg_response(fig, **alongside)


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
    series_list = posted_rows(post_data)
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


#: Where the last posted plot configuration is kept, so that the download
#: view can regenerate exactly what was on screen.
plot_session_key = "diagnostics_last"


def display_diagnostics(request, x_diagnostic, y_diagnostic, expressions):
    """View displaying the table of available series for an axis pair.

    The library arrives as an argument rather than being fetched here: it
    is the one thing on this page that comes from the browser-interface
    database, and keeping it out means everything in this module can be
    tested against a project database alone.  ``views.py`` supplies it.
    """

    with start_db_session() as db_session:
        context = get_available_series(
            x_diagnostic, y_diagnostic, expressions, db_session
        )
        context["available_diagnostics"] = get_available_diagnostics(
            get_recorded_diagnostics(db_session), expressions
        )

    context["x_diagnostic"] = x_diagnostic
    context["y_diagnostic"] = y_diagnostic
    context["diagnostics_title"] = (
        y_diagnostic
        if x_diagnostic == time_quantity
        else f"{x_diagnostic} vs {y_diagnostic}"
    )
    # No y in either: every posted row names the quantity it draws, so
    # redrawing and downloading ask only for the x they share.
    context["update_plot_url"] = reverse(
        "diagnostics:update_diagnostics_plot",
        kwargs={"x_diagnostic": x_diagnostic},
    )
    context["download_pdf_url"] = reverse(
        "diagnostics:download_diagnostics_plot",
        kwargs={"x_diagnostic": x_diagnostic},
    )

    return render(request, "diagnostics/diagnostics_app.html", context)


def display_image_diagnostics(_request, diagnostic_name):
    """Redirect the pre-merge time-series URL onto the merged view.

    Kept so that links built before the merge keep working, including the
    six ``{% url %}`` tags in ``processing/progress.html``.
    """

    return redirect(
        "diagnostics:display_diagnostics",
        x_diagnostic=time_quantity,
        y_diagnostic=diagnostic_name,
    )
