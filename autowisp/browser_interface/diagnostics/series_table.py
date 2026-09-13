"""What the table above a diagnostics plot offers, and what it binds.

The half of the plot page that knows what a *row* means: which series can
be drawn at all, which channel each column of a row may be bound to, how
many images a binding would draw, and what a row is called while the user
chooses. It never evaluates an expression and never draws anything --
:mod:`image_diagnostics_views` does both -- which is not merely a tidy
division. There is a row per observing session and image type, so
evaluating one to fill the table would be work proportional to the whole
image collection; every question asked here is answered by a SQL
aggregate instead.
"""

import math

from sqlalchemy import select

from django.template.loader import render_to_string

from autowisp.browser_interface.core.plot_utils import channel_colors
from autowisp.diagnostics.diagnostic_types import (
    is_quantile_diagnostic,
    quantiles_quantity,
)
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    count_images_with_all,
    count_images_with_channels,
    time_quantity,
)
from autowisp.diagnostics.expressions import (
    get_expression_parameters,
    get_needed_values,
    get_quantity_arity,
    order_expressions,
)
from autowisp.exceptions import PipelineError

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    ImageDiagnostics,
    ObservingSession,
)
from autowisp.database.data_model.provenance import Camera, CameraChannel

# pylint: enable=no-name-in-module


#: Separates the fields of a row id.  Not the underscore an earlier
#: encoding used: ``pixel_q*`` names contain those, so unpacking had to
#: guess which underscores separated fields.  Neither a session id, an
#: image type nor a diagnostic name can contain this one.
row_id_separator = "|"


def make_row_id(session_id, image_type, quantile_name, ordinal):
    """
    Return the id of one row of the series table.

    A row outlives the channels bound in it -- binding one is an edit made
    *in* the row -- so its id names the group it belongs to and its place
    among the rows of that group, and says nothing about channels. That is
    what lets the four element ids derived from it (``plot-color:``,
    ``marker-button:``, ``scale:``, ``label:``) and the key the client
    posts it under stay put while the user chooses.

    Args:
        session_id(int):    The observing session.

        image_type(str):    The frame type.

        quantile_name(str):    The ``pixel_q*`` this row stands for, or
            ``None`` outside a quantile expansion.

        ordinal(int):    Distinguishes the rows of one group, which differ
            only by what they bind.

    Returns:
        str:    The opaque id, which has to survive a round trip through
            the client unchanged.

    Raises:
        ValueError:    If a part contains the separator, which would make
            the id ambiguous. Worth failing on rather than trusting, since
            the alternative is a plot that silently draws the wrong rows.
    """

    parts = (
        str(session_id),
        image_type,
        quantile_name or "",
        str(ordinal),
    )
    ambiguous = [part for part in parts if row_id_separator in part]
    if ambiguous:
        raise ValueError(
            f"Cannot build a row id from {parts!r}: "
            f"{', '.join(repr(part) for part in ambiguous)} contains "
            f"{row_id_separator!r}, which separates its parts."
        )

    return row_id_separator.join(parts)


def split_row_id(row_id):
    """Return ``(session_id, image_type, quantile_name)`` from a row id.

    The *group* a row belongs to, which is what almost everything wants:
    the ordinal only tells sibling rows apart, and is read separately by
    the one caller that needs it.
    """

    session_id, image_type, quantile_name, _ = row_id.split(row_id_separator)

    return int(session_id), image_type, quantile_name or None


def row_ordinal(row_id):
    """Return which of its group's rows this is, counting from zero."""

    return int(row_id.rsplit(row_id_separator, 1)[1])


def get_series_key(series):
    """
    Return what one posted row draws: its group, and what it binds.

    Half of this comes from the row id and half from the client, and the
    split is not an inconsistency. Session, type and quantile are not
    editable in the table, so taking them from the id keeps one source of
    truth for them; the channels *are* edited there, and reading them from
    anywhere but the dropdown the user just changed would be reading a
    stale value.

    Args:
        series(dict):    One row as the client posted it back.

    Returns:
        SeriesKey:    What the row identifies, with empty channels where
            its dropdowns are still unset.
    """

    session_id, image_type, quantile_name = split_row_id(series["id"])

    return SeriesKey(
        session_id,
        image_type,
        tuple(series.get("channels", ())),
        quantile_name,
    )


def make_series(row_id, session_label, series_key, slots, count):
    """
    Build the entry describing one row of the series table.

    Args:
        row_id(str):    What :func:`make_row_id` returned for this row.

        session_label(str):    The label of the observing session.

        series_key(SeriesKey):    What the row draws once bound. Its
            ``channels`` are empty for a row still to be bound.

        slots(list):    One entry per channel the axes bind, as
            :func:`get_slot_options` describes them.

        count(int):    The number of images contributing, or ``None``
            where nothing is bound yet and there is nothing to count.

    Returns:
        dict:    A series entry with the keys expected by
            ``diagnostics/_series_row.html`` and
            :func:`plot_image_diagnostic_series`.
    """

    quantile = (
        None
        if series_key.quantile_name is None
        else "0." + series_key.quantile_name[len("pixel_q") :]
    )
    describe = [session_label, series_key.image_type]
    if quantile is not None:
        describe.append(quantile)
    describe.extend(series_key.channels)

    return {
        "id": row_id,
        "channels": list(series_key.channels),
        "color": channel_colors.get(
            series_key.channel[0].upper() if series_key.channel else "",
            "#ffffff",
        ),
        "marker": "o",
        "scale": "1.0",
        "label": " ".join(describe),
        "session_label": session_label,
        "image_type": series_key.image_type,
        "quantile": quantile,
        "slots": slots,
        "count": "-" if count is None else count,
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


def get_axis_slots(quantity, expressions):
    """
    Return the diagnostics read in each channel one axis binds.

    An axis binds one channel per parameter of the quantity it draws, and
    two axes never share one: their numbers are formal parameters, so the
    first slot of the x quantity and the first of the y quantity are
    unrelated, and tying them together would silently couple the axes.

    Args:
        quantity(str):    The concrete quantity the axis draws, the
            quantile family already resolved to a member.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        list:    One ``(parameter, needed)`` pair per channel to bind, in
            column order: the slot number the definition writes, which
            names the column, and the diagnostics read in it. The
            parameter is ``None`` for a diagnostic, which has no numbering
            of its own. Empty for an axis over the time alone, which binds
            nothing.
    """

    if quantity in expressions:
        parameters = get_expression_parameters(expressions[quantity])
    else:
        # A diagnostic binds one channel and numbers it nothing; the time
        # binds none. Asked rather than derived, neither arity coming from
        # a definition -- and asking is also what refuses a name that
        # resolves to nothing at all.
        parameters = (None,) * get_quantity_arity(quantity, expressions)

    if not parameters:
        return []

    # Walked with a *column index* standing in for each channel. A
    # reference's arguments are matched against the definition's
    # parameters positionally, so whatever is passed here is what comes
    # back at the leaves: passing columns asks which column each
    # diagnostic is read in, whatever numbers the expression happened to
    # write. ``bg_center[3] / pixel_q99[7]`` reports column 0 and column 1.
    columns = tuple(range(len(parameters)))
    reads = {column: set() for column in columns}
    for name, bindings in get_needed_values(
        {quantity: {columns}}, expressions
    ).items():
        for binding in bindings:
            for column in binding:
                reads[column].add(name)

    return list(zip(parameters, (frozenset(reads[c]) for c in columns)))


def get_slot_headings(axis, axis_name, slots):
    """
    Return the column heading for each channel one axis binds.

    Named for the axis as well as the quantity because the two axes may
    name one quantity -- comparing a diagnostic between channels is
    exactly that -- and two columns headed alike would say nothing. Where
    an axis binds several channels the heading is the reference as the
    definition writes it, so a column can be matched to the text it fills
    in.

    Args:
        axis(str):    Which axis these columns belong to.

        axis_name(str):    The quantity as the selector names it, which
            for the quantile family is the family rather than a member:
            the table has one header for all of them.

        slots(list):    What :func:`get_axis_slots` returned for the axis.

    Returns:
        list:    One heading per channel to bind.
    """

    if len(slots) == 1:
        return [f"{axis}: {axis_name}"]

    return [f"{axis}: {axis_name}[{parameter}]" for parameter, _ in slots]


def get_slot_options(slot_needs, db_session):
    """
    Return what each slot of the table may be bound to, and the labels.

    One aggregate per *distinct* set of diagnostics among the slots --
    usually one for the whole table, since the commonest axis pairs read
    the same diagnostics in every slot. Nothing is evaluated: which
    channels a slot may offer is a question about rows.

    Args:
        slot_needs(list):    What each slot reads, from
            :func:`get_axis_slots` for each axis in turn.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    One entry per distinct set of needs, holding
                ``{(session_id, image_type): {channel: count}}``.

            dict:    ``{session_id: label}``, the same whatever is read.
    """

    labels = {}
    options = {}
    for needed in set(slot_needs):
        by_group = {}
        for (
            label,
            session_id,
            image_type,
            channel,
            count,
        ) in count_images_with_all(needed, db_session):
            labels[session_id] = label
            by_group.setdefault((session_id, image_type), {})[channel] = count
        options[needed] = by_group

    return options, labels


def count_bound_images(slot_needs, channels, db_session):
    """
    Count the images one binding draws on, for every group at once.

    The exact question, where :func:`get_slot_options` answers the looser
    one that fills the dropdowns: an image counts when its rows cover
    every (diagnostic, channel) pair *between them*, which is what a
    quantity comparing channels needs and what reading each channel on its
    own cannot say.

    Every group is counted in one aggregate rather than one per row, since
    a binding is usually shared -- by every row of a monochrome project at
    render, and by nothing much afterwards, when one row is rebound at a
    time.

    Args:
        slot_needs(list):    What each slot reads, in column order.

        channels(tuple):    The channel bound in each of those slots.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``{(session_id, image_type): count}``, holding only the
            groups with anything to draw.
    """

    return {
        (session_id, image_type): count
        for _, session_id, image_type, count in count_images_with_channels(
            {
                (name, channel)
                for needed, channel in zip(slot_needs, channels)
                for name in needed
            },
            db_session,
        )
    }


def get_session_channels(db_session):
    """
    Return the channels each observing session's camera defines.

    Not the channels it has *recorded*: a Bayer camera part way through
    processing has one channel with data and three still to come, and
    treating that one as the only possibility would be wrong by tomorrow.
    What the camera defines does not move.

    Args:
        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``{session_id: [channel name, ...]}``, in name order.
            A session whose camera is not described is simply absent.
    """

    channels = {}
    # pylint: disable=no-member
    for session_id, name in db_session.execute(
        select(ObservingSession.id, CameraChannel.name)
        .select_from(ObservingSession)
        .join(Camera, Camera.id == ObservingSession.camera_id)
        .join(
            CameraChannel, CameraChannel.camera_type_id == Camera.camera_type_id
        )
        .order_by(ObservingSession.id, CameraChannel.name)
    ).all():
        channels.setdefault(session_id, []).append(name)
    # pylint: enable=no-member

    return channels


def build_group_rows(quantile_name, slot_needs, session_channels, db_session):
    """
    Return one row per (session, image type) the axes can be drawn for.

    A group is offered where every slot has at least one channel to choose
    from; the choosing then happens in the table, and nothing arrives
    bound. Pre-binding instead would mean enumerating a cartesian product
    of the channels across the axes' parameters.

    The exception is a camera defining a single channel -- a monochrome
    one, zero not being a working configuration -- where there is nothing
    to choose. Demanding a click with one possible outcome before anything
    can be drawn is ceremony, so those rows arrive bound and counted.

    Args:
        quantile_name(str):    Which ``pixel_q*`` these rows stand for,
            when an axis was selected as the ``pixel_quantiles`` *family*
            and the table expanded it into one group of rows per recorded
            member. ``None`` otherwise -- including for an expression
            naming concrete quantiles, such as ``pixel_q99[0] /
            pixel_q50[0]``, which is an ordinary quantity reading two
            diagnostics and expands into nothing.

        slot_needs(list):    What each slot reads, in column order.

        session_channels(dict):    What :func:`get_session_channels`
            returned.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    Series entries, as :func:`make_series` builds them.
    """

    if not slot_needs:
        return []

    options, labels = get_slot_options(slot_needs, db_session)
    groups = sorted(
        set.intersection(*(set(options[needed]) for needed in slot_needs))
    )

    bound = {
        session_id: session_channels[session_id][0]
        for session_id, _ in groups
        if len(session_channels.get(session_id, ())) == 1
    }
    counts = {
        channel: count_bound_images(
            slot_needs, (channel,) * len(slot_needs), db_session
        )
        for channel in set(bound.values())
    }

    rows = []
    for group in groups:
        session_id, image_type = group
        channel = bound.get(session_id)
        rows.append(
            make_series(
                make_row_id(session_id, image_type, quantile_name, 0),
                labels[session_id],
                SeriesKey(
                    session_id,
                    image_type,
                    () if channel is None else (channel,) * len(slot_needs),
                    quantile_name,
                ),
                [
                    {
                        "options": sorted(options[needed][group].items()),
                        "value": "" if channel is None else channel,
                        "fixed": channel is not None,
                    }
                    for needed in slot_needs
                ],
                None if channel is None else counts[channel].get(group, 0),
            )
        )

    return rows


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
    session_channels = get_session_channels(db_session)

    headings = []
    rows = []
    for quantile_name in quantile_names:
        # What an axis *needs* is not what it names: an expression needs
        # the diagnostics it reaches, transitively, and per slot, since a
        # slot is offered only the channels its own are recorded in. jd
        # drops out either way -- it is known for every image of the
        # session and so constrains nothing.
        per_axis = [
            get_axis_slots(
                resolve_quantity(quantity_name, quantile_name), expressions
            )
            for quantity_name in (x_diagnostic, y_diagnostic)
        ]
        # One header for the whole table, so it is written from the axis
        # names rather than from a member: every member of the family is a
        # diagnostic binding one channel, so the shape does not vary.
        headings = [
            heading
            for axis, axis_name, slots in zip(
                ("x", "y"), (x_diagnostic, y_diagnostic), per_axis
            )
            for heading in get_slot_headings(axis, axis_name, slots)
        ]
        rows.extend(
            build_group_rows(
                quantile_name,
                [needed for slots in per_axis for _, needed in slots],
                session_channels,
                db_session,
            )
        )

    # Session, then type, then quantile.  Not by channel: an unbound row
    # has none, and nothing reorders the table once it is drawn.
    rows.sort(
        key=lambda row: (
            row["session_label"],
            row["image_type"],
            row["quantile"] or "",
        )
    )

    fields = ["Observing Session", "Type"] + headings
    if has_quantiles:
        fields.append("Quantile")
    fields.append("Count")

    return {
        "diagnostics_fields": fields,
        # Named separately so the header can mark them unsortable: they
        # hold a dropdown, which sorts by nothing anyone wants.
        "slot_headings": headings,
        "diagnostics_list": rows,
    }


def get_axes_slot_needs(x_diagnostic, y_diagnostic, expressions, quantile_name):
    """Return what each channel column of the table reads, in column order.

    The x quantity's slots followed by the y quantity's, concatenated
    rather than merged: an expression's numbers are formal parameters, so
    the two axes' slots are unrelated even when written alike.

    Args:
        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``.

        quantile_name(str):    The ``pixel_q*`` a row stands for, or
            ``None`` outside a quantile expansion.

    Returns:
        list:    One ``frozenset`` of diagnostic names per column.
    """

    return [
        needed
        for quantity_name in (x_diagnostic, y_diagnostic)
        for _, needed in get_axis_slots(
            resolve_quantity(quantity_name, quantile_name), expressions
        )
    ]


def plan_spare_row(row_id, slot_needs, options, labels, datasets):
    """
    Return the entry for a fresh unbound row, or ``None`` for no spare.

    Completing a row's channels summons one below it, so that a second
    binding of the same series can be built and drawn on the same figure.
    Two cases earn none: a spare is already waiting -- this row is not the
    last of its group, so it was rebound rather than newly bound -- and a
    group whose every possible binding is already present, where a further
    row could only repeat one.

    Kept apart from rendering it so that the decision can be tested
    without Django, which is where every other rule in this module is
    checked.

    Args:
        row_id(str):    The row that was just bound. What it binds does
            not matter here -- a spare binds nothing -- so only its group
            and its ordinal are read from it.

        slot_needs(list):    What each channel column reads.

        options(dict):    What :func:`get_slot_options` returned.

        labels(dict):    The session labels from the same call.

        datasets(dict):    Every row of the table as the client posted it.

    Returns:
        dict or None:    A series entry, as :func:`make_series` builds it.
    """

    session_id, image_type, quantile_name = split_row_id(row_id)
    group = (session_id, image_type)
    ordinal = row_ordinal(row_id)
    siblings = [
        other_id
        for other_id in datasets
        if split_row_id(other_id) == split_row_id(row_id)
    ]

    # A spare is already waiting below this one, so the row was rebound
    # rather than newly bound.
    if any(row_ordinal(other_id) > ordinal for other_id in siblings):
        return None

    bindings = {
        tuple(datasets[other_id].get("channels", ()))
        for other_id in siblings
        if all(datasets[other_id].get("channels", ()))
    }
    if len(bindings) >= math.prod(
        len(options[needed][group]) for needed in slot_needs
    ):
        return None

    return make_series(
        make_row_id(session_id, image_type, quantile_name, ordinal + 1),
        labels[session_id],
        SeriesKey(session_id, image_type, (), quantile_name),
        [
            {
                "options": sorted(options[needed][group].items()),
                "value": "",
                "fixed": False,
            }
            for needed in slot_needs
        ],
        None,
    )


def get_binding_response(
    post_data, *, x_diagnostic, y_diagnostic, expressions, db_session
):
    """
    Return what a row that has just been fully bound earns.

    Merged into the figure's JSON response, so that choosing the last of a
    row's channels costs one round trip rather than two. Nothing here
    re-renders the table: the client updates one cell and appends at most
    one row, which is what lets every other row keep its node, and with it
    what was typed into it and its place under whatever sort is in force.

    Args:
        post_data(dict):    The whole POST, holding every row's state and
            ``bind``, the id of the row whose channels were completed.

        x_diagnostic(str):    Quantity on the X axis.

        y_diagnostic(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``bind`` echoed back with the row's ``count`` and the
            defaults that follow from its channels, plus ``spare_row``
            where a further binding is still to be built. Empty when
            nothing was bound, which is every other redraw.
    """

    row_id = post_data.get("bind")
    datasets = post_data.get("datasets", {})
    if not row_id or row_id not in datasets:
        return {}

    series_key = get_series_key({"id": row_id, **datasets[row_id]})
    if not series_key.channels or not all(series_key.channels):
        return {}

    group = (series_key.session_id, series_key.image_type)
    slot_needs = get_axes_slot_needs(
        x_diagnostic, y_diagnostic, expressions, series_key.quantile_name
    )
    options, labels = get_slot_options(slot_needs, db_session)
    if any(group not in options[needed] for needed in slot_needs):
        # The table was rendered against a project that has since changed.
        return {}

    entry = make_series(
        row_id,
        labels[series_key.session_id],
        series_key,
        [],
        count_bound_images(slot_needs, series_key.channels, db_session).get(
            group, 0
        ),
    )

    spare = plan_spare_row(row_id, slot_needs, options, labels, datasets)

    return {
        "bind": row_id,
        "count": entry["count"],
        "color": entry["color"],
        "label": entry["label"],
        "spare_row": (
            None
            if spare is None
            else render_to_string(
                "diagnostics/_series_row.html", {"diagnostic": spare}
            )
        ),
    }
