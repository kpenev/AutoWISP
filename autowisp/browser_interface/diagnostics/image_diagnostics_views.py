"""Views for displaying per-image diagnostics.

Quantities may be plotted against one another, where a quantity is a
``DiagnosticType`` name, an expression over those, or ``jd``.  Plotting
against time is not a separate mode: it is ``x="jd"``, which resolves
through the same path as everything else because the canonical image list
already carries the Julian dates.

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
from django.shortcuts import render
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

from .quantities import (
    describe_quantity,
    get_available_diagnostics,
    get_diagnostic_descriptions,
    get_recorded_diagnostics,
    next_section_marker,
    section_markers,
)
from .series_table import (
    get_available_series,
    get_series_key,
    posted_rows,
    split_row_id,
)


def get_series_data(series, x_quantity, expressions, db_session):
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

        x_quantity(str):    Quantity on the X axis, which the page
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
    y_quantity, _ = split_row_id(series["id"])
    quantities = [x_quantity, y_quantity]

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


def collect_series_data(series_list, x_quantity, expressions, db_session):
    """
    Read the selected series, dropping those with nothing to draw.

    Every row of the table is posted rather than only the drawn ones, so
    that the server can see what the whole table binds.  Four of them are
    skipped here: one the user has not selected, one whose marker is
    blank, one still missing a channel, and one naming no session and
    image type -- the last two naming no data to read.

    Args:
        series_list(list):    Every row as the client posted it back.

        x_quantity(str):    Quantity on the X axis, shared by the whole
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
            series, x_quantity, expressions, db_session
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


def assign_y_axes(drawn, requested):
    """
    Return the quantities to put on each y axis, in the order drawn.

    Sharing an axis is the safe default and the usual answer: two
    quantities wrongly sharing one show it at once, since one of them is
    flattened, where two wrongly given separate axes are rescaled to fill
    the same height and invite a reader to compare what cannot be
    compared.

    What a user asks for is an axis *number* per quantity, which is easier
    to say than an ordering. Turning numbers into axes is what happens
    here: numbers nothing drawn uses are skipped, so asking for 1 and 3
    draws two axes rather than three with an empty one between them, and
    the number is only ever a way of grouping and ordering.

    Args:
        drawn(iterable):    The quantities actually drawn, in section
            order. Repeats are ignored: a quantity is on one axis however
            many rows draw it.

        requested(dict):    ``{quantity: axis number}``, as the client
            posts it. A quantity missing from it, or carrying something
            that is not a number, falls to the first axis -- an
            unanswered question is not worth failing a plot over.

    Returns:
        list:    One list of quantity names per axis, the first being the
            host. Each list is in section order, and the axes are ordered
            by the number asked for. Empty when nothing is drawn.
    """

    wanted = {}
    for quantity in dict.fromkeys(drawn):
        try:
            number = int(requested.get(quantity, 1))
        except (TypeError, ValueError):
            number = 1
        wanted.setdefault(number, []).append(quantity)

    return [wanted[number] for number in sorted(wanted)]


def create_diagnostics_figure(
    series_list,
    *,
    x_quantity,
    expressions,
    db_session,
    figure_config=None,
):
    """
    Create the figure for the selected series, against one x quantity.

    Args:
        series_list(list):    Every row as the client posted it back.
            Only those with a non-empty ``marker`` are plotted.

        x_quantity(str):    Quantity on the X axis, which every series
            shares.  What each draws against it comes from its own id.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

        figure_config(dict):    Layout of the figure, defining
            ``plot_height_frac``, ``num_columns`` and ``aspect_ratio``.

    Returns:
        matplotlib.figure.Figure:    The completed figure.
    """

    figure_config = figure_config or {}
    against_time = x_quantity == time_quantity

    series_data = collect_series_data(
        series_list, x_quantity, expressions, db_session
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

    # Decided from everything drawn rather than per subplot, so that a
    # quantity keeps the same axis, the same side and the same label in
    # every night. Built per subplot, a night drawing only the second
    # quantity would put it on the host, its label crossing to the other
    # side of the plot from where the night above has it. The scales still
    # differ between nights -- each subplot autoscales to what it holds,
    # which is deliberate -- so what this keeps steady is where to look,
    # not what the heights mean.
    per_axis = assign_y_axes(
        (series["quantity"] for series, *_ in series_data),
        figure_config.get("y_axes", {}),
    )

    for host, group in zip(all_axes.flatten(), groups):
        draw_group_on_axes(
            host,
            group,
            per_axis,
            x_offset,
            figure_config.get("show_legend", True),
        )
        host.set_xlabel(f"JD - {x_offset!r}" if against_time else x_quantity)
        # The host alone: twin grids interleave into a mesh that says
        # nothing about either scale.
        host.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


def draw_group_on_axes(host, group, per_axis, x_offset, show_legend):
    """
    Draw one night's series, each on the axis its quantity was given.

    Args:
        host:    The subplot this night was assigned, which carries the
            first y axis, the x axis and the grid.

        group(list):    ``(series, x_values, y_values, image_ids)`` tuples
            for this night.

        per_axis(list):    What :func:`assign_y_axes` returned: the
            quantities on each axis, the first being the host's.

        x_offset(float):    Subtracted from every x value.

        show_legend(bool):    Whether to draw the merged legend.

    Returns:
        None
    """

    axes_for = {}
    axes = []
    for number, quantities in enumerate(per_axis):
        # A twin shares the x axis and brings a y scale of its own. From
        # the third onwards its spine is pushed outwards, so that the
        # scales stand side by side instead of on top of one another.
        this = host if number == 0 else host.twinx()
        if number > 1:
            this.spines.right.set_position(("axes", 1 + 0.12 * (number - 1)))
        this.set_ylabel(", ".join(quantities))
        axes.append(this)
        for quantity in quantities:
            axes_for[quantity] = this

    for series, x_values, y_values, image_ids in group:
        plot_image_diagnostic_series(
            axes_for.get(series["quantity"], host),
            x_values - x_offset,
            y_values,
            image_ids,
            series,
        )

    if show_legend:
        draw_merged_legend(axes)


def draw_merged_legend(axes):
    """
    Draw one legend for a subplot, naming what every one of its axes drew.

    On the topmost axis rather than the host, and gathering the handles by
    hand, because neither happens by itself: each axis offers only the
    series drawn on it, and a legend belonging to the host is painted
    underneath the twins that sit above it.

    Args:
        axes(list):    The subplot's axes, host first, as
            :func:`draw_group_on_axes` built them.

    Returns:
        None
    """

    handles, labels = [], []
    for each in axes:
        each_handles, each_labels = each.get_legend_handles_labels()
        handles += each_handles
        labels += each_labels

    if handles:
        axes[-1].legend(handles, labels)


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
        # Before the figure, because the rows it returns are the ones
        # drawn: a row whose binding has just changed has to reach the
        # figure with the colour and label of the binding it now has,
        # rather than the ones the client posted with it.
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


# Each names one thing a section cannot be built without, and the caller
# holds all of them.
# pylint: disable=too-many-arguments
def build_section(
    section_y, *, x_quantity, expressions, descriptions, marker, db_session
):
    """
    Return everything one section of the page is rendered from.

    A section is the table for the pair ``(x, section_y)``, headed by what
    that quantity is. Its channel columns depend on the quantity it draws,
    which is what a section is *for*: quantities binding different numbers
    of channels cannot share one table, so each gets its own.

    Args:
        section_y(str):    The quantity this section draws.

        x_quantity(str):    Quantity on the X axis, shared by the page.

        expressions(dict):    The library, ``{name: expression}``.

        descriptions(dict):    What each quantity means, the recorded
            diagnostics' and the expressions' together.

        marker(str):    What this section's rows start out drawn with.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    The section, as ``_series_section.html`` renders it.
    """

    section = get_available_series(
        x_quantity, section_y, expressions, db_session, marker=marker
    )
    section["quantity"] = describe_quantity(
        section_y, expressions, descriptions
    )
    section["marker"] = marker

    return section


# pylint: enable=too-many-arguments


def display_diagnostics(
    request, x_quantity, y_quantities, expressions, expression_descriptions
):
    """View displaying a section per y quantity, all against one x.

    Args:
        request:    The Django request.

        x_quantity(str):    Quantity on the X axis. The page has one, and
            a user wanting a second opens another tab: everything drawn
            here shares it.

        y_quantities(list):    What the sections draw, in the order the
            URL names them, which is the order they appear in and the
            order they take their markers in.

        expressions(dict):    The library.  It arrives as an argument
            rather than being fetched here because it comes from the
            browser-interface database, and keeping that out means
            everything in this module can be tested against a project
            database alone.  ``views.py`` supplies it.

        expression_descriptions(dict):    What each expression is for,
            from the same database and passed in for the same reason. The
            recorded diagnostics describe themselves in the project one,
            and the two are merged here.
    """

    with start_db_session() as db_session:
        recorded = get_recorded_diagnostics(db_session)
        # An expression may not take a recorded diagnostic's name, so
        # neither can shadow the other and the merge is unambiguous.
        descriptions = {
            **get_diagnostic_descriptions(db_session),
            **expression_descriptions,
        }
        sections = [
            build_section(
                section_y,
                x_quantity=x_quantity,
                expressions=expressions,
                descriptions=descriptions,
                marker=marker,
                db_session=db_session,
            )
            for section_y, marker in zip(y_quantities, section_markers)
        ]
        available = get_available_diagnostics(recorded, expressions)

    context = {
        "sections": sections,
        "available_diagnostics": available,
    }
    context["x_quantity"] = x_quantity
    context["y_quantities"] = y_quantities
    # Named for the x alone. The sections come and go without the page
    # being reloaded, so a title naming them would be wrong as soon as one
    # was added.
    context["diagnostics_title"] = (
        "Diagnostics"
        if x_quantity == time_quantity
        else f"Diagnostics against {x_quantity}"
    )
    # No y in either: every posted row names the quantity it draws, so
    # redrawing and downloading ask only for the x they share.
    context["update_plot_url"] = reverse(
        "diagnostics:update_diagnostics_plot",
        kwargs={"x_quantity": x_quantity},
    )
    context["download_pdf_url"] = reverse(
        "diagnostics:download_diagnostics_plot",
        kwargs={"x_quantity": x_quantity},
    )

    return render(request, "diagnostics/diagnostics_app.html", context)


def diagnostics_section(
    request, x_quantity, y_quantity, expressions, expression_descriptions
):
    """Return one rendered section, for adding one without a reload.

    A request of its own rather than something a redraw carries, unlike
    the row ``+`` adds: a new section's row is bound only where each of
    its channel columns has a single channel recorded, so on a colour
    camera it arrives unbound, draws nothing, and the figure is unchanged.
    Asking for the section alone leaves that figure alone; folding this
    into a redraw would rebuild it to look exactly as it already does.

    The markers the page's sections already start with arrive as
    ``taken``, comma separated, since sections have come and gone in the
    browser since the page was built and the server cannot know otherwise
    which markers are still spoken for.
    """

    taken = request.GET.get("taken", "")

    with start_db_session() as db_session:
        section = build_section(
            y_quantity,
            x_quantity=x_quantity,
            expressions=expressions,
            descriptions={
                **get_diagnostic_descriptions(db_session),
                **expression_descriptions,
            },
            marker=next_section_marker(taken.split(",") if taken else []),
            db_session=db_session,
        )

    return render(
        request, "diagnostics/_series_section.html", {"section": section}
    )
