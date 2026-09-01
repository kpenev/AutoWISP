"""Views for displaying per-image diagnostics.

One quantity may be plotted against another, where a quantity is a
``DiagnosticType`` name, the ``quantiles`` pseudo-name expanding to one
series per ``pixel_q*``, or ``jd``.  Plotting against time is not a separate
mode: it is ``x="jd"``, which resolves through the same path as everything
else because the canonical image list already carries the Julian dates.
"""

from io import BytesIO
from typing import NamedTuple
import json
import math

import matplotlib
from matplotlib import pyplot
from matplotlib.figure import Figure
from sqlalchemy import select, func
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

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    ImageDiagnostics,
    Image,
    ImageType,
    ObservingSession,
)

# pylint: enable=no-name-in-module

#: The mid-exposure Julian date, available for every image of a session
#: without consulting ``image_diagnostics``.  Plain JD, not barycentric:
#: barycentric correction depends on where on the sky one points, so BJD is
#: per-source and undefined for a per-image diagnostic.
_time_quantity = "jd"

#: Pseudo-quantity expanding to one series per ``pixel_q*`` diagnostic.
_quantiles_quantity = "quantiles"


class SeriesKey(NamedTuple):
    """What one plotted series is, and what its id encodes.

    The image type is part of the key because a session holds frames of
    several types and a diagnostic rarely means the same thing across them
    -- some are only defined for object frames, and one recorded for both
    would have its aggregates taken over a mixture, making
    ``nanmedian(bg_center)`` a median of object and flat frames together.

    Everything downstream of :func:`get_available_series` takes one of these
    rather than the four values separately, so a caller cannot pair a
    channel with the wrong session by getting an argument order wrong.
    """

    session_id: int
    image_type: str
    channel: str
    quantile_name: str = None

    #: Separates the fields of an id.  Not the underscore the encoding used
    #: to use: ``pixel_q*`` names contain those, so unpacking had to guess
    #: which underscores were separators, and adding a field would have made
    #: the guess wrong.  A session id, a channel and a diagnostic name can
    #: none of them contain this one.  Not annotated, so it stays a class
    #: attribute rather than becoming a fifth field.
    id_separator = "|"

    def to_id(self):
        """
        Return the opaque string identifying this series to the client.

        It becomes an HTML element id, four more element ids are built from
        it, and it keys the ``datasets`` object the client posts back, so it
        has to survive that round trip unchanged.

        Raises:
            ValueError:    If a field contains :data:`id_separator`, which
                would make the id ambiguous.  Worth failing on rather than
                trusting, since a channel naming scheme is not this module's
                to control and the alternative is plots that silently pair
                the wrong data.
        """

        fields = (
            str(self.session_id),
            self.image_type,
            self.channel,
            self.quantile_name or "",
        )
        ambiguous = [field for field in fields if self.id_separator in field]
        if ambiguous:
            raise ValueError(
                f"Cannot build a series id from {fields!r}: "
                f"{', '.join(repr(field) for field in ambiguous)} contains "
                f"the {self.id_separator!r} that separates its fields."
            )
        return self.id_separator.join(fields)

    @classmethod
    def from_id(cls, series_id):
        """Return the key an id was built from, the inverse of `to_id`."""

        session_id, image_type, channel, quantile_name = series_id.split(
            cls.id_separator
        )
        return cls(int(session_id), image_type, channel, quantile_name or None)


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


def get_available_diagnostics(db_session):
    """Return the quantity names that can be plotted in this project.

    A per-type ``EXISTS`` probe rather than a ``GROUP BY`` over the whole of
    ``image_diagnostics``: the question is only which names are in use, and
    the grouped form has to walk every row to answer it.
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

    result = [_time_quantity] + [
        name for name in names if not name.startswith("pixel_q")
    ]
    if any(name.startswith("pixel_q") for name in names):
        result.append(_quantiles_quantity)

    return result


def get_canonical_images(series_key, db_session):
    """
    Return ``(image_ids, jd_values)`` for one session and image type, by JD.

    Every array plotted for this series is built against this list, with
    ``NaN`` wherever a value does not exist, so index *i* is the same image
    in every array.  Alignment is then structural and no join is needed
    between two quantities.

    The channel of *series_key* is deliberately not used -- the list is the
    same for every channel -- but the image type is: frames of different
    types are different populations, and mixing them would put a flat frame
    and an object frame in one array for an aggregate to average over.

    Args:
        series_key(SeriesKey):    The series to list the images of.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    Arrays of image IDs and of Julian dates, of equal length.
    """

    rows = db_session.execute(
        select(Image.id, Image.jd)  # pylint: disable=no-member
        .join(
            ImageType,
            ImageType.id == Image.image_type_id,  # pylint: disable=no-member
        )
        .where(
            # pylint: disable=no-member
            Image.observing_session_id == series_key.session_id,
            ImageType.name == series_key.image_type,
            Image.jd.is_not(None),
            # pylint: enable=no-member
        )
        .order_by(Image.jd)  # pylint: disable=no-member
    ).all()

    if not rows:
        return numpy.empty(0, dtype=int), numpy.empty(0, dtype=float)

    image_ids, jd_values = zip(*rows)
    return (
        numpy.asarray(image_ids, dtype=int),
        numpy.asarray(jd_values, dtype=float),
    )


def resolve_quantity(quantity_name, quantile_name):
    """
    Map an axis name onto the concrete quantity for one series.

    ``quantiles`` names a family rather than a quantity: each series picks
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

    if quantity_name == _quantiles_quantity:
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


def count_images_with_all(needed, db_session):
    """
    Count images holding all of *needed*, per (session, type, channel).

    Args:
        needed(set):    ``DiagnosticType`` names that must all be recorded
            for an image to count.  An empty set means no quantity
            constrains the result, which only happens when both axes are
            :data:`_time_quantity`; nothing is plottable then.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(session_label, session_id, image_type, channel, count)``
            tuples.
    """

    if not needed:
        return []

    per_image = (
        select(
            Image.observing_session_id.label(  # pylint: disable=no-member
                "session_id"
            ),
            Image.image_type_id.label(  # pylint: disable=no-member
                "image_type_id"
            ),
            ImageDiagnostics.channel.label("channel"),
        )
        .join(
            Image,
            Image.id == ImageDiagnostics.image_id,  # pylint: disable=no-member
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
        .where(
            DiagnosticType.name.in_(needed),
            Image.jd.is_not(None),  # pylint: disable=no-member
        )
        .group_by(ImageDiagnostics.image_id, ImageDiagnostics.channel)
        .having(
            # pylint: disable=not-callable
            func.count(func.distinct(DiagnosticType.id))
            == len(needed)
        )
        .subquery()
    )

    return db_session.execute(
        select(
            ObservingSession.label,
            ObservingSession.id,
            ImageType.name,
            per_image.c.channel,
            func.count(),  # pylint: disable=not-callable
        )
        .select_from(per_image)
        .join(ObservingSession, ObservingSession.id == per_image.c.session_id)
        .join(ImageType, ImageType.id == per_image.c.image_type_id)
        .group_by(ObservingSession.id, ImageType.id, per_image.c.channel)
        .order_by(ObservingSession.label, ImageType.name, per_image.c.channel)
    ).all()


def get_available_series(x_diagnostic, y_diagnostic, db_session):
    """
    Return the (session, image type, channel) series plottable for an axis pair.

    The count is the number of images recording every diagnostic both axes
    need.  It is an upper bound on the number of drawn points, since
    arithmetic can still yield NaN, so the column is labelled for the inputs
    rather than for the points.

    Args:
        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``diagnostics_fields`` and ``diagnostics_list``, in the
            format ``diagnostics_app.html`` expects.
    """

    has_quantiles = _quantiles_quantity in (x_diagnostic, y_diagnostic)

    quantile_names = get_quantile_names(db_session) if has_quantiles else [None]

    rows = []
    for quantile_name in quantile_names:
        # Anything but jd has to be recorded for an image to count; jd is
        # known for every image of the session and so constrains nothing.
        needed = {
            name
            for name in (
                resolve_quantity(x_diagnostic, quantile_name),
                resolve_quantity(y_diagnostic, quantile_name),
            )
            if name != _time_quantity
        }
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


def get_quantity_values(quantity_name, series_key, canonical, db_session):
    """
    Return one quantity NaN-padded onto the canonical image list.

    Args:
        quantity_name(str):    The quantity to read, already resolved by
            :func:`resolve_quantity` so that it names either
            :data:`_time_quantity` or a single ``DiagnosticType``.

        series_key(SeriesKey):    The series to read the values of.  Its
            image type narrows the query: values for the session's other
            types would be discarded anyway, being absent from the canonical
            list, but excluding them in SQL keeps the work proportional to
            the series actually plotted.

        canonical(tuple):    ``(image_ids, jd_values)`` from
            :func:`get_canonical_images`.

        db_session:    An active SQLAlchemy database session.

    Returns:
        numpy.ndarray:    Values in canonical order, ``NaN`` where the
            quantity is not recorded for that image.
    """

    image_ids, jd_values = canonical
    if quantity_name == _time_quantity:
        return jd_values

    values = numpy.full(image_ids.size, numpy.nan)
    if not image_ids.size:
        return values

    row_of_image = {
        image_id: index for index, image_id in enumerate(image_ids.tolist())
    }
    for image_id, value in db_session.execute(
        select(ImageDiagnostics.image_id, ImageDiagnostics.value)
        .join(
            Image,
            Image.id == ImageDiagnostics.image_id,  # pylint: disable=no-member
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
        .join(
            ImageType,
            ImageType.id == Image.image_type_id,  # pylint: disable=no-member
        )
        .where(
            # pylint: disable=no-member
            Image.observing_session_id == series_key.session_id,
            ImageType.name == series_key.image_type,
            ImageDiagnostics.channel == series_key.channel,
            DiagnosticType.name == quantity_name,
            Image.jd.is_not(None),
            # pylint: enable=no-member
        )
    ).all():
        index = row_of_image.get(image_id)
        if index is not None:
            values[index] = value

    return values


def get_series_data(series, x_diagnostic, y_diagnostic, db_session):
    """
    Query the paired x/y values for a single series.

    Both axes are built against the same canonical image list, so they are
    aligned by construction and are returned unmasked; the single finite
    mask lives in :func:`plot_image_diagnostic_series`.

    Args:
        series(dict):    An entry from :func:`get_available_series`.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    ``(x_values, y_values, image_ids)``, all of equal length.
    """

    # From the id rather than from the entry's own ``channel``, which the
    # client echoes back: one source of truth for what the series is.
    series_key = SeriesKey.from_id(series["id"])
    canonical = get_canonical_images(series_key, db_session)

    x_values, y_values = (
        get_quantity_values(
            resolve_quantity(quantity_name, series_key.quantile_name),
            series_key,
            canonical,
            db_session,
        )
        for quantity_name in (x_diagnostic, y_diagnostic)
    )

    return x_values, y_values, canonical[0]


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


def collect_series_data(series_list, x_diagnostic, y_diagnostic, db_session):
    """
    Read the selected series, dropping those with nothing to draw.

    Args:
        series_list(list):    Entries from :func:`get_available_series`.
            Only those with a non-empty ``marker`` are read.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

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
            series, x_diagnostic, y_diagnostic, db_session
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


def create_diagnostics_figure(
    series_list,
    *,
    x_diagnostic,
    y_diagnostic,
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

        db_session:    An active SQLAlchemy database session.

        figure_config(dict):    Layout of the figure, defining
            ``plot_height_frac``, ``num_columns`` and ``aspect_ratio``.

    Returns:
        matplotlib.figure.Figure:    The completed figure.
    """

    figure_config = figure_config or {}
    against_time = x_diagnostic == _time_quantity

    series_data = collect_series_data(
        series_list, x_diagnostic, y_diagnostic, db_session
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


def display_diagnostics(request, x_diagnostic, y_diagnostic):
    """View displaying the table of available series for an axis pair."""

    with start_db_session() as db_session:
        context = get_available_series(x_diagnostic, y_diagnostic, db_session)
        context["available_diagnostics"] = get_available_diagnostics(db_session)

    context["x_diagnostic"] = x_diagnostic
    context["y_diagnostic"] = y_diagnostic
    context["diagnostics_title"] = (
        y_diagnostic
        if x_diagnostic == _time_quantity
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
        x_diagnostic=_time_quantity,
        y_diagnostic=diagnostic_name,
    )
