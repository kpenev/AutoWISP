"""What the table above a diagnostics plot offers, and what it binds.

The half of the plot page that knows what a *row* means: which observing
session and image type a row may be drawn for, which channel each of its
columns may be bound to, how many images a binding would draw, and what a
row is called while the user chooses. It never evaluates an expression and
never draws anything -- :mod:`image_diagnostics_views` does both -- which
is not merely a tidy division. The dropdowns a row is built from span every
observing session and image type in the project, so evaluating anything to
fill them would be work proportional to the whole image collection; every
question asked here is answered by a SQL aggregate instead.
"""

from sqlalchemy import select

from django.template.loader import render_to_string

from autowisp.browser_interface.core.plot_utils import channel_colors
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    count_images_with_all,
    count_images_with_channels,
)
from autowisp.diagnostics.expressions import (
    get_expression_parameters,
    get_needed_values,
    get_quantity_arity,
)

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import ObservingSession

# pylint: enable=no-name-in-module

#: Separates the fields of the two ids the client round-trips: a row's,
#: ``quantity|ordinal``, and the value of one option of the session and
#: type dropdown, ``session id|image type``.  Not the underscore an earlier
#: encoding used: ``pixel_q*`` names contain those, so unpacking had to
#: guess which underscores separated fields.  Neither a session id, an
#: image type nor a diagnostic name can contain this one.
row_id_separator = "|"

#: What a channel dropdown shows while it is unset.  Named here rather than
#: written into the template because a cell sorts by the text of its
#: selected option, so one place has to decide what that text is.
unset_option_text = "\N{EM DASH}"


def make_id(*parts):
    """
    Return one of the ids the client round-trips, built from *parts*.

    One function for both of them -- a row's ``quantity|ordinal`` and a
    pair's ``session id|image type`` -- because the encoding is the same
    and all either needs of it is that it can be taken apart again.  Where
    the two differ is in the reading: :func:`split_row_id` and
    :func:`split_pair_id`.

    Returns:
        str:    The opaque id, which has to survive a round trip through
            the client unchanged.

    Raises:
        ValueError:    If a part contains the separator, which would make
            the id ambiguous. Worth failing on rather than trusting, since
            the alternative is a plot that silently draws the wrong rows.
    """

    parts = tuple(str(part) for part in parts)
    ambiguous = [part for part in parts if row_id_separator in part]
    if ambiguous:
        raise ValueError(
            f"Cannot build an id from {parts!r}: "
            f"{', '.join(repr(part) for part in ambiguous)} contains "
            f"{row_id_separator!r}, which separates its parts."
        )

    return row_id_separator.join(parts)


def split_row_id(row_id):
    """
    Return ``(quantity, ordinal)`` from the id of one row.

    A row outlives everything chosen in it -- its session, its image type
    and its channels are all edits made *in* the row -- so its id names
    only what cannot be edited there: the quantity the row draws, and its
    place among the rows drawing it.  That is what lets the five element
    ids derived from it (``plot-color:``, ``marker-button:``, ``marker:``,
    ``scale:``, ``label:``) and the key the client posts it under stay put
    while the user chooses.

    The quantity is part of it because two rows drawing different
    quantities may otherwise agree on everything: a row is told apart by
    what it draws, not by where it draws it.
    """

    quantity, ordinal = row_id.split(row_id_separator)

    return quantity, int(ordinal)


def split_pair_id(pair_id):
    """
    Return ``(session_id, image_type)`` from what a row's dropdown says.

    The pair is the unit everything on this page is counted by, so it
    travels as one value and is taken apart only here.
    """

    session_id, image_type = pair_id.split(row_id_separator)

    return int(session_id), image_type


def get_series_key(series):
    """
    Return the population one posted row draws from, and what it binds.

    All of it comes from the client and none from the row id: the session,
    the image type and the channels are chosen *in* the row, so reading any
    of them from anywhere but the dropdown the user just changed would be
    reading a stale value.  What the id carries instead is the quantity,
    which the row cannot change.

    Args:
        series(dict):    One row as the client posted it back.

    Returns:
        SeriesKey:    What the row identifies, with empty channels where
            its dropdowns are still unset.
    """

    session_id, image_type = split_pair_id(series["pair"])

    return SeriesKey(
        session_id,
        image_type,
        tuple(series.get("channels", ())),
    )


def posted_rows(post_data):
    """
    Return the rows of a posted table as a list, each carrying its id.

    The client posts them keyed by id, since that is what it has to look
    them up by; everything on the server wants a row to be one object that
    knows its own id. The single place that turns one shape into the other.
    """

    return [
        {"id": row_id, **config}
        for row_id, config in post_data.get("datasets", {}).items()
    ]


def next_row_id(row, row_ids):
    """
    Return the id for a row added beside *row*, drawing the same quantity.

    One past the highest ordinal among that quantity's rows rather than
    one past *row*'s own, so that the id stays unique after a removal has
    left a gap: reusing a freed ordinal would collide with nothing on the
    page, but the id is also the suffix of five element ids, and a second
    row answering to them while the first is still being edited is the
    kind of bug that shows up as one row's colour landing on another's.

    Args:
        row(dict):    The row the new one is built beside, as the client
            posted it. Only the quantity in its id is read here.

        row_ids(iterable):    Every row id on the page, the new one having
            to be unique among all of them.

    Returns:
        str:    The id, as :func:`make_id` builds it.
    """

    quantity, _ = split_row_id(row["id"])
    taken = [
        ordinal
        for other_quantity, ordinal in map(split_row_id, row_ids)
        if other_quantity == quantity
    ]

    return make_id(quantity, max(taken, default=-1) + 1)


# Six things a row is made of, each named at the call site. Grouping them
# would only move the list somewhere less visible.
# pylint: disable=too-many-arguments
def make_series(row_id, series_key, options, slots, count, *, marker):
    """
    Build the entry describing one row of the series table.

    Args:
        row_id(str):    What :func:`make_id` returned for this row.

        series_key(SeriesKey):    What the row draws once bound. Its
            ``channels`` are empty for a row still to be bound.

        options(list):    Every (session, image type) pair to allow the user to
            select, as :func:`get_pair_options` returns them.  Carried by
            the row rather than read from the page's context, so that a row
            can be rendered on its own.

        slots(list):    One cell per channel the axes bind, as
            :func:`make_slot_cells` builds them.

        count(int):    The number of images contributing, or ``None``
            where nothing is bound yet and there is nothing to count.

        marker(str):    What the row is drawn with to begin with, which
            its section decides: the default tells quantities apart by
            shape, as the colour tells channels apart. Editable
            afterwards, like the colour.

    Returns:
        dict:    A series entry with the keys expected by
            ``diagnostics/_series_row.html`` and
            :func:`plot_image_diagnostic_series`.
    """

    pair = make_id(series_key.session_id, series_key.image_type)
    chosen = next(option for option in options if option["value"] == pair)

    return {
        "id": row_id,
        "channels": list(series_key.channels),
        "color": channel_colors.get(
            series_key.channel[0].upper() if series_key.channel else "",
            "#ffffff",
        ),
        "marker": marker,
        "scale": "1.0",
        # The quantity first, so that a legend entry says which of the
        # page's sections it belongs to. With one section that is
        # redundant, but prefixing always keeps the rule simple, and the
        # label is the user's to rewrite either way.
        "label": " ".join(
            [split_row_id(row_id)[0], chosen["text"], *series_key.channels]
        ),
        "pair": pair,
        # The cell sorts by what it shows -- the label rather than the id
        # behind it -- so that sorting by this column puts the sessions in
        # the order the dropdown lists them rather than in insertion order.
        "pair_sort": chosen["text"],
        "start": chosen["start"],
        "end": chosen["end"],
        "options": options,
        "slots": slots,
        "count": "-" if count is None else count,
    }


# pylint: enable=too-many-arguments


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


def get_session_times(session_ids, db_session):
    """
    Return when each observing session began and ended, as text.

    Formatted here rather than in the template because the table sorts a
    column by the text in it: ``YYYY-MM-DD HH:MM`` UTC sorts into
    chronological order, which is the whole reason these two columns
    exist. Session labels are free-form, so nothing else on the row can be
    sorted on to the same effect.

    Args:
        session_ids(iterable):    The sessions to look up, read in one
            query however many of them there are.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``{session_id: (start, end)}``, with an empty string
            wherever a session records no such time.
    """

    session_ids = set(session_ids)
    if not session_ids:
        return {}

    def as_text(when):
        """Return the timestamp as the table shows and sorts it."""

        return "" if when is None else when.strftime("%Y-%m-%d %H:%M")

    # pylint: disable=no-member
    return {
        session_id: (as_text(start), as_text(end))
        for session_id, start, end in db_session.execute(
            select(
                ObservingSession.id,
                ObservingSession.start_time_utc,
                ObservingSession.end_time_utc,
            ).where(ObservingSession.id.in_(session_ids))
        ).all()
    }
    # pylint: enable=no-member


def make_slot_cells(available, channels):
    """
    Return what each channel column of one row offers, and what it says.

    A column offering a single channel is settled rather than asked
    about: that is the whole of a monochrome camera, and of a colour one
    whose other channels are not processed yet, and demanding a click with
    one possible outcome before anything can be drawn is ceremony. Such a
    cell shows its channel as text, which is what lets a row be drawn the
    moment it appears. Should another channel be recorded later, the
    column has a choice to offer and asks for one.

    What a column may offer depends on the (session, image type) pair the
    row names, so choosing another pair asks this again for all of them.
    A channel the new pair still offers is kept -- the user chose it, and
    it remains an answer -- and one it does not is cleared rather than
    quietly bound to something else.

    Pure, so that both rules can be tested without a database or a
    browser.

    Args:
        available(list):    What each column may be bound to under this
            row's pair, ``{channel: count}`` in column order.

        channels(tuple):    What the row binds now, in the same order.
            Empty, or short, for a row whose columns are not all set.

    Returns:
        list:    One cell per column, as ``_slot_cells.html`` renders it.
    """

    cells = []
    for column, offered in enumerate(available):
        settled = len(offered) == 1
        if settled:
            value = next(iter(offered))
        else:
            value = channels[column] if column < len(channels) else ""
            if value not in offered:
                value = ""

        options = [{"value": "", "text": unset_option_text}] + [
            {"value": channel, "text": f"{channel} ({count})"}
            for channel, count in sorted(offered.items())
        ]
        cells.append(
            {
                "options": options,
                "value": value,
                "fixed": settled,
                # What the column sorts by: what the cell shows, so that
                # sorting follows the channels chosen rather than the
                # run-together text of a dropdown.
                "sort": (
                    value
                    if settled
                    else next(
                        option["text"]
                        for option in options
                        if option["value"] == value
                    )
                ),
            }
        )

    return cells


def get_pair_options(slot_needs, options, labels, db_session):
    """
    Return the (session, image type) pairs a row may be drawn for.

    A pair is offered where every channel column has at least one channel
    to bind: unless each of them can be read somewhere in that session's
    frames of that type, a row on it could name no data at all.  Which
    channel a column takes is chosen in the row afterwards, so a pair is
    offered once rather than once per binding it could carry.

    Args:
        slot_needs(list):    What each channel column reads, in column
            order.

        options(dict):    What :func:`get_slot_options` returned.

        labels(dict):    The session labels from the same call.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``{"value", "text", "start", "end"}`` per pair, by session
            start time and then image type -- which is the order the
            dropdown lists them in, and why the first of them is the
            earliest session. Empty where the axes bind no channel at all,
            there being nothing to draw and so nothing to offer.
    """

    if not slot_needs:
        return []

    pairs = set.intersection(*(set(options[needed]) for needed in slot_needs))
    times = get_session_times(
        {session_id for session_id, _ in pairs}, db_session
    )

    listed = [
        {
            "value": make_id(session_id, image_type),
            "text": f"{labels[session_id]} {image_type}",
            "start": times.get(session_id, ("", ""))[0],
            "end": times.get(session_id, ("", ""))[1],
        }
        for session_id, image_type in pairs
    ]
    # By text rather than by type alone, so that the pairs of one session
    # stay together: it begins with the session label, which is what the
    # start time ties on.
    listed.sort(key=lambda option: (option["start"], option["text"]))

    return listed


# Six things, none derivable from the others here, and both callers hold
# all of them. Bundling them would hide what a row depends on.
# pylint: disable=too-many-arguments
def make_row_for_pair(
    row_id,
    series_key,
    *,
    slot_needs,
    options,
    pair_options,
    marker,
    db_session,
):
    """
    Return one row of the series table, bound as far as its pair allows.

    Shared by the first row of a table and by a row whose pair has just
    changed, both asking the same question: what a row on this pair
    offers, which of the channels it names survive there, and how many
    images the result draws.

    Args:
        row_id(str):    What :func:`make_id` returned for this row.

        series_key(SeriesKey):    The pair the row names, and the channels
            it would keep -- those the pair does not offer are dropped.

        slot_needs(list):    What each channel column reads.

        options(dict):    What :func:`get_slot_options` returned.

        pair_options(list):    What :func:`get_pair_options` returned.

        marker(str):    The marker the row's section starts its rows with.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    A series entry, as :func:`make_series` builds it, counted
            where the pair leaves it fully bound and uncounted where it
            does not.
    """

    pair = (series_key.session_id, series_key.image_type)
    slots = make_slot_cells(
        [options[needed][pair] for needed in slot_needs],
        series_key.channels,
    )

    channels = tuple(slot["value"] for slot in slots)
    if not all(channels):
        # A partly bound row binds nothing: it names no data to read, and
        # an empty binding is what the colour, the label and the count all
        # branch on.
        channels = ()

    return make_series(
        row_id,
        SeriesKey(*pair, channels),
        pair_options,
        slots,
        (
            None
            if not channels
            else count_bound_images(slot_needs, channels, db_session).get(
                pair, 0
            )
        ),
        marker=marker,
    )


# pylint: enable=too-many-arguments


def get_available_series(
    x_quantity, y_quantity, expressions, db_session, *, marker
):
    """
    Return what one section's table offers, and the row it starts with.

    A row is a series the user builds, so what this answers is what its
    dropdowns may offer: the (session, image type) pairs where every
    channel column has a channel to bind.  The table then starts with one
    row on the first of them -- the earliest session -- so that the page
    has a series without the user having to build one.

    The count is the number of images recording every diagnostic both axes
    need -- for an expression, every diagnostic it reaches transitively.  It
    is an upper bound on the number of drawn points, since arithmetic can
    still yield NaN, so the column is labelled for the inputs rather than
    for the points.  Nothing is evaluated to produce it: the count is a
    question about rows, and stays a SQL aggregate.

    Args:
        x_quantity(str):    Quantity on the X axis.

        y_quantity(str):    Quantity on the Y axis, which is what the
            rows draw and therefore what their ids name.

        expressions(dict):    The library, ``{name: expression}``, passed in
            rather than fetched so that nothing below the view has to know
            where it is stored.

        db_session:    An active SQLAlchemy database session.

        marker(str):    What this section's rows are drawn with to begin
            with, chosen for the section rather than for the row so that
            the quantities on a plot are told apart by shape.

    Returns:
        dict:    ``diagnostics_fields``, ``pair_options`` and
            ``diagnostics_list``, in the format ``_series_section.html``
            expects.

    Raises:
        PipelineError:    If an axis names nothing that resolves.
    """

    # What an axis *needs* is not what it names: an expression needs the
    # diagnostics it reaches, transitively, and per slot, since a slot is
    # offered only the channels its own are recorded in. jd drops out
    # either way -- it is known for every image of the session and so
    # constrains nothing.
    per_axis = [
        get_axis_slots(quantity_name, expressions)
        for quantity_name in (x_quantity, y_quantity)
    ]
    headings = [
        heading
        for axis, axis_name, slots in zip(
            ("x", "y"), (x_quantity, y_quantity), per_axis
        )
        for heading in get_slot_headings(axis, axis_name, slots)
    ]
    slot_needs = [needed for slots in per_axis for _, needed in slots]

    options, labels = (
        get_slot_options(slot_needs, db_session) if slot_needs else ({}, {})
    )
    pair_options = get_pair_options(slot_needs, options, labels, db_session)

    rows = []
    if pair_options:
        session_id, image_type = split_pair_id(pair_options[0]["value"])
        rows.append(
            make_row_for_pair(
                make_id(y_quantity, 0),
                SeriesKey(session_id, image_type, ()),
                slot_needs=slot_needs,
                options=options,
                pair_options=pair_options,
                marker=marker,
                db_session=db_session,
            )
        )

    return {
        "diagnostics_fields": (
            ["Session and Type", "Start (UTC)", "End (UTC)"]
            + headings
            + ["Count"]
        ),
        "pair_options": pair_options,
        "diagnostics_list": rows,
    }


def get_axes_slot_needs(x_quantity, y_quantity, expressions):
    """Return what each channel column of the table reads, in column order.

    The x quantity's slots followed by the y quantity's, concatenated
    rather than merged: an expression's numbers are formal parameters, so
    the two axes' slots are unrelated even when written alike.

    Args:
        x_quantity(str):    Quantity on the X axis.

        y_quantity(str):    Quantity on the Y axis.

        expressions(dict):    The library, ``{name: expression}``.

    Returns:
        list:    One ``frozenset`` of diagnostic names per column.
    """

    return [
        needed
        for quantity_name in (x_quantity, y_quantity)
        for _, needed in get_axis_slots(quantity_name, expressions)
    ]


def get_row_options(y_quantity, *, x_quantity, expressions, db_session):
    """
    Return what a row drawing one quantity against the page's x is built from.

    The three things that have to be worked out before any row can be
    built: what each channel column reads, what each may be bound to, and
    the (session, image type) pairs the row may name.

    Args:
        y_quantity(str):    The quantity the row draws, from its id.

        x_quantity(str):    Quantity on the X axis.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    ``(slot_needs, options, pair_options)``, as
            :func:`get_axes_slot_needs`, :func:`get_slot_options` and
            :func:`get_pair_options` return them.
    """

    slot_needs = get_axes_slot_needs(x_quantity, y_quantity, expressions)
    options, labels = (
        get_slot_options(slot_needs, db_session) if slot_needs else ({}, {})
    )

    return (
        slot_needs,
        options,
        get_pair_options(slot_needs, options, labels, db_session),
    )


def get_table_response(post_data, *, x_quantity, expressions, db_session):
    """
    Return the rows to draw, and what the edit behind this redraw earns.

    Three kinds of redraw arrive here. A row was rebound (``bind``) --
    its last channel chosen, or another (session, image type) pair, which
    is a rebinding as much as the first since which channels a column may
    offer depends on the pair, and so is answered even while the channels
    are unset. Or ``+`` was pressed in a row (``add``), which earns a copy
    of it. Or nothing structural happened at all -- a colour, a marker, a
    row switched off -- which earns the figure and nothing else.

    Every one of them rides on the redraw the edit causes anyway, so each
    costs one round trip rather than two, and the table and the figure are
    answered together. Nothing here re-renders the table: the client
    replaces a row's channel cells, or inserts one row, which is what lets
    every other row keep its node, and with it what was typed into it and
    its place under whatever sort is in force.

    Args:
        post_data(dict):    The whole POST, holding every row's state and,
            where an edit is being answered, ``bind`` or ``add``: the id
            of the row it happened in.

        x_quantity(str):    Quantity on the X axis.  The y comes from
            the row id, each row naming the quantity it draws.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            list:    The rows the figure is drawn from: those posted, plus
                an added row, and with a rebound row's colour and label
                replaced by the defaults of its new binding wherever the
                client reports them still automatic. Drawing from these is
                what stops a rebound row appearing in the colour and
                legend of the binding it has just left while the table
                beside it shows the new ones.

            dict:    For a rebinding, ``bind`` echoed back with the row's
                ``count``, its channel cells rendered for the pair it now
                names, that session's ``start`` and ``end``, and the
                colour and label its binding makes default. For an
                addition, ``added_row``, the markup to insert, and
                ``after``, the row to insert it below. Empty when nothing
                was edited, which is every other redraw.
    """

    rows = posted_rows(post_data)
    datasets = post_data.get("datasets", {})
    adding = bool(post_data.get("add"))
    edited = post_data.get("add") if adding else post_data.get("bind")
    if not edited or edited not in datasets:
        return rows, {}

    source = {"id": edited, **datasets[edited]}
    series_key = get_series_key(source)
    slot_needs, options, pair_options = get_row_options(
        split_row_id(edited)[0],
        x_quantity=x_quantity,
        expressions=expressions,
        db_session=db_session,
    )
    if make_id(series_key.session_id, series_key.image_type) not in {
        option["value"] for option in pair_options
    }:
        # The row names a pair this table cannot offer, the project having
        # changed since the table was rendered.
        return rows, {}

    entry = make_row_for_pair(
        next_row_id(source, datasets) if adding else edited,
        series_key,
        slot_needs=slot_needs,
        options=options,
        pair_options=pair_options,
        # A copy starts with its section's marker rather than the marker
        # of the row it was copied from, which may have been set by hand:
        # the client posts what the section's is. A rebound row keeps
        # whatever it is already drawn with, the client sending it back
        # unchanged, so what is passed here reaches only the copy.
        marker=(post_data.get("section_marker") or source.get("marker") or "o"),
        db_session=db_session,
    )

    if adding:
        # Drawn from the response that carries it, so the row and its
        # series arrive together rather than the table gaining a row the
        # figure only catches up with on the next redraw.
        return rows + [entry], {
            "added_row": render_to_string(
                "diagnostics/_series_row.html", {"diagnostic": entry}
            ),
            "after": edited,
        }

    # The colour and label of the *new* binding, applied before the figure
    # is drawn rather than written into the table after it. Only where the
    # client reports the field still automatic, which is the same test
    # ``applyTableResponse`` makes before accepting them -- both sides
    # judging the same state, so exactly the fields the server corrects
    # here are the ones the client will take.
    automatic = {
        field: entry[field]
        for field in ("color", "label")
        if datasets[edited].get("automatic_" + field)
    }

    return (
        [row if row["id"] != edited else {**row, **automatic} for row in rows],
        {
            "bind": edited,
            "count": entry["count"],
            "color": entry["color"],
            "label": entry["label"],
            "start": entry["start"],
            "end": entry["end"],
            "slot_cells": render_to_string(
                "diagnostics/_slot_cells.html", {"diagnostic": entry}
            ),
        },
    )
