"""Views for displaying per-image diagnostics.

One quantity may be plotted against another, where a quantity is a
``DiagnosticType`` name, the ``pixel_quantiles`` pseudo-name expanding to
one series per ``pixel_q*``, or ``jd``.  Plotting against time is not a separate
mode: it is ``x="jd"``, which resolves through the same path as everything
else because the canonical image list already carries the Julian dates.
"""

from io import BytesIO
import json
import math

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
from sqlalchemy import select
import numpy

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from autowisp.browser_interface.core.plot_utils import (
    channel_colors,
    setup_svg_matplotlib,
    figure_to_svg_response,
)
from autowisp.database.interface import start_db_session
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    count_images_with_all,
    get_series_values,
    time_quantity,
)
from autowisp.diagnostics.diagnostic_types import (
    is_quantile_diagnostic,
    quantiles_quantity,
)
from autowisp.diagnostics.expressions import order_expressions
from autowisp.exceptions import PipelineError

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    ImageDiagnostics,
)

# pylint: enable=no-name-in-module


def make_series(session_label, series_key, count):
    """
    Build the entry describing one plottable series.

    Args:
        session_label(str):    The label of the observing session.

        series_key(SeriesKey):    What the series identifies.

        count(int):    The number of images contributing to the series.

    Returns:
        dict:    A series entry with the keys expected by
            ``diagnostics_app.html`` and
            :func:`plot_image_diagnostic_series`.
    """

    describe = [session_label, series_key.image_type, series_key.channel]
    if series_key.quantile_name is not None:
        describe.append("0." + series_key.quantile_name[len("pixel_q") :])

    return {
        "channel": series_key.channel,
        "color": channel_colors.get(
            series_key.channel[0].upper() if series_key.channel else "",
            "#ffffff",
        ),
        "marker": "o",
        "scale": "1.0",
        "id": series_key.to_id(),
        "label": " ".join(describe),
        "info": describe + [count],
    }


def get_recorded_diagnostics(db_session):
    """
    Return the ``DiagnosticType`` names anything has recorded in this project.

    A per-type ``EXISTS`` probe rather than a ``GROUP BY`` over the whole of
    ``image_diagnostics``: the question is only which names are in use, and
    the grouped form has to walk every row to answer it.

    The names come back raw, individual ``pixel_q*`` entries included --
    before :func:`get_available_diagnostics` collapses them into the family
    name.  That is what an expression has to be judged against, since one
    may reference a concrete quantile.

    Args:
        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    The names in use, in ``DiagnosticType`` order.
    """

    names = []
    for type_id, name in db_session.execute(
        select(DiagnosticType.id, DiagnosticType.name).order_by(
            DiagnosticType.id
        )
    ).all():
        in_use = db_session.execute(
            select(
                select(ImageDiagnostics.id)
                .where(ImageDiagnostics.diagnostic_id == type_id)
                .exists()
            )
        ).scalar()
        if in_use:
            names.append(name)

    return names


def get_available_diagnostics(recorded, expressions):
    """
    Return every quantity an axis may be set to.

    One flat list rather than diagnostics and expressions kept apart: an
    axis reads a name, and a recorded diagnostic is simply an expression of
    itself as far as anything downstream is concerned.  Sharing one name
    space is what lets the selectors, the URL and the series table treat
    all of them alike, and it is why an expression may not take a
    diagnostic's name.

    Args:
        recorded(list):    What :func:`get_recorded_diagnostics` found.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        list:    ``jd``, then every recorded diagnostic -- with the
            individual quantiles standing down in favour of the family name
            that expands to one series per member -- then the expressions
            this project has the data to draw.
    """

    result = [time_quantity] + [
        name for name in recorded if not is_quantile_diagnostic(name)
    ]
    if any(is_quantile_diagnostic(name) for name in recorded):
        result.append(quantiles_quantity)

    return result + get_available_expressions(expressions, recorded)


def get_available_expressions(expressions, recorded):
    """
    Return the expressions this project has the data to draw.

    Availability, not validity.  Every stored expression is valid in every
    project -- the vocabulary is the same everywhere, see
    :mod:`autowisp.diagnostics.diagnostic_types` -- so filtering by
    :func:`~autowisp.diagnostics.expressions.check_expression` would filter
    nothing and offer all of them everywhere.  What decides whether one is
    offered *here* is whether the diagnostics it reaches, transitively, have
    actually been recorded.

    Args:
        expressions(dict):    The library, ``{name: expression}``.

        recorded(list):    What :func:`get_recorded_diagnostics` found.
            The raw names, since an expression may reference a concrete
            ``pixel_q*`` rather than the family.

    Returns:
        list:    The names whose every diagnostic is recorded here,
            alphabetically.
    """

    recorded = set(recorded)

    available = []
    for name in sorted(expressions):
        try:
            _, needed = order_expressions([name], expressions)
        except PipelineError:
            # A stored cycle, or a name no version of AutoWISP defines.
            # Saying so is the management page's business; here it is
            # merely not offered, so that one broken expression cannot stop
            # the plot page rendering.
            continue
        # jd is known for every image of the canonical list, so it never
        # counts against availability.
        if needed - {time_quantity} <= recorded:
            available.append(name)

    return available


def resolve_quantity(quantity_name, quantile_name):
    """
    Map an axis name onto the concrete quantity for one series.

    ``pixel_quantiles`` names a family rather than a quantity: each series picks
    one ``pixel_q*`` member of it, recorded in the series id.  Resolving
    that here, once, is what lets everything downstream handle a single
    concrete name -- leaving ``jd`` as the only quantity that still needs a
    branch anywhere, because it alone comes from the image table rather than
    from ``image_diagnostics``.

    Args:
        quantity_name(str):    The name an axis was selected as.

        quantile_name(str):    The ``pixel_q*`` this series stands for, or
            ``None`` outside a quantile expansion.

    Returns:
        str:    The quantity to actually read.
    """

    if quantity_name == quantiles_quantity:
        return quantile_name
    return quantity_name


def get_quantile_names(db_session):
    """Return the ``pixel_q*`` diagnostic names in use, quantile order."""

    return [
        row[0]
        for row in db_session.execute(
            select(DiagnosticType.name)
            .where(DiagnosticType.name.like("pixel_q%"))
            .order_by(DiagnosticType.name)
        ).all()
    ]


def get_available_series(x_diagnostic, y_diagnostic, expressions, db_session):
    """
    Return the (session, image type, channel) series plottable for an axis pair.

    The count is the number of images recording every diagnostic both axes
    need -- for an expression, every diagnostic it reaches transitively.  It
    is an upper bound on the number of drawn points, since arithmetic can
    still yield NaN, so the column is labelled for the inputs rather than
    for the points.  Nothing is evaluated to produce it: the count is a
    question about rows, and stays a SQL aggregate.

    Args:
        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``, passed in
            rather than fetched so that nothing below the view has to know
            it came from the browser-interface database.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``diagnostics_fields`` and ``diagnostics_list``, in the
            format ``diagnostics_app.html`` expects.

    Raises:
        PipelineError:    If an axis names nothing that resolves.
    """

    has_quantiles = quantiles_quantity in (x_diagnostic, y_diagnostic)

    quantile_names = get_quantile_names(db_session) if has_quantiles else [None]

    rows = []
    for quantile_name in quantile_names:
        # What an axis *needs* is not what it names: an expression needs the
        # diagnostics it reaches, transitively, and a series is offered only
        # where every one of them is recorded.  For a plain diagnostic the
        # walk returns it unchanged, and jd drops out either way -- it is
        # known for every image of the session and so constrains nothing.
        _, needed = order_expressions(
            [
                resolve_quantity(quantity_name, quantile_name)
                for quantity_name in (x_diagnostic, y_diagnostic)
            ],
            expressions,
        )
        needed = needed - {time_quantity}
        rows.extend(
            (
                session_label,
                SeriesKey(session_id, image_type, channel, quantile_name),
                count,
            )
            for session_label, session_id, image_type, channel, count in (
                count_images_with_all(needed, db_session)
            )
        )

    # Session, then type, then channel, then quantile.
    rows.sort(
        key=lambda row: (
            row[0],
            row[1].image_type,
            row[1].channel,
            row[1].quantile_name or "",
        )
    )

    fields = ["Observing Session", "Type", "Channel"]
    if has_quantiles:
        fields.append("Quantile")
    fields.append("Count")

    return {
        "diagnostics_fields": fields,
        "diagnostics_list": [make_series(*row) for row in rows],
    }


def get_series_data(
    series, x_diagnostic, y_diagnostic, expressions, db_session
):
    """
    Query the paired x/y values for a single series.

    Both axes are resolved in one call, which is what makes them share a
    query for the diagnostics they need and one symbol table, so a
    subexpression common to the two is evaluated once.  They are returned
    unmasked; the single finite mask lives in
    :func:`plot_image_diagnostic_series`.

    Args:
        series(dict):    An entry from :func:`get_available_series`.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``, passed in
            rather than fetched so that nothing below the view has to know
            it came from the browser-interface database.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    ``(x_values, y_values, image_ids)``, all of equal length.
    """

    # From the id rather than from the entry's own ``channel``, which the
    # client echoes back: one source of truth for what the series is.
    series_key = SeriesKey.from_id(series["id"])
    quantities = [
        resolve_quantity(quantity_name, series_key.quantile_name)
        for quantity_name in (x_diagnostic, y_diagnostic)
    ]

    values, image_ids = get_series_values(
        series_key, quantities, expressions, db_session
    )

    # Indexed rather than unpacked: the two axes may name one quantity,
    # which is a plot of it against itself rather than a mistake.
    return values[quantities[0]], values[quantities[1]], image_ids


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
            ``color``, ``marker``, ``scale``, and ``label``.
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


def collect_series_data(
    series_list, x_diagnostic, y_diagnostic, expressions, db_session
):
    """
    Read the selected series, dropping those with nothing to draw.

    Args:
        series_list(list):    Entries from :func:`get_available_series`.
            Only those with a non-empty ``marker`` are read.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(series, x_values, y_values, image_ids)`` tuples for the
            series having at least one point where both axes are finite.
    """

    series_data = []
    for series in series_list:
        if not series.get("marker", "").strip():
            continue
        x_values, y_values, image_ids = get_series_data(
            series, x_diagnostic, y_diagnostic, expressions, db_session
        )
        x_values = numpy.atleast_1d(x_values)
        y_values = numpy.atleast_1d(y_values)
        # A padded array is full length even when every value is NaN, so
        # its size no longer tells us whether anything will be drawn.
        if numpy.any(numpy.isfinite(x_values) & numpy.isfinite(y_values)):
            series_data.append((series, x_values, y_values, image_ids))

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


# All but the first are keyword-only, and each names one thing the figure
# cannot be drawn without.  Grouping them into an object would hide what the
# URL layer has to supply rather than simplify it.
# pylint: disable=too-many-arguments
def create_diagnostics_figure(
    series_list,
    *,
    x_diagnostic,
    y_diagnostic,
    expressions,
    db_session,
    figure_config=None,
):
    """
    Create the figure for the selected series of an axis pair.

    Args:
        series_list(list):    Entries from :func:`get_available_series`.
            Only those with a non-empty ``marker`` are plotted.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

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
        series_list, x_diagnostic, y_diagnostic, expressions, db_session
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
        axes.set_ylabel(y_diagnostic)
        if figure_config.get("show_legend", True):
            axes.legend()
        axes.grid(True, linewidth=0.2)

    fig.tight_layout()
    return fig


# pylint: enable=too-many-arguments


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
    context["update_plot_url"] = reverse(
        "diagnostics:update_diagnostics_plot",
        kwargs={
            "x_diagnostic": x_diagnostic,
            "y_diagnostic": y_diagnostic,
        },
    )
    context["download_pdf_url"] = reverse(
        "diagnostics:download_diagnostics_plot",
        kwargs={
            "x_diagnostic": x_diagnostic,
            "y_diagnostic": y_diagnostic,
        },
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
