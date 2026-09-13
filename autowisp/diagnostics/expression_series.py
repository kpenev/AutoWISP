"""Values for one series of images, read from the project database.

Tier 2 of the expression layer: it knows the project database and nothing
else. Above it, the browser interface adds Django and a way of editing the
library; below it, :mod:`autowisp.diagnostics.expressions` knows what an
expression *means* and has no database at all. This module is the join
between them -- it turns a session, an image type and the channels a series
binds into the ``{name: {channels: array}}`` that tier 1 evaluates against.

Everything here is built on **one canonical image list per session and image
type**, ordered by Julian date, with ``NaN`` wherever a value is not
recorded. Alignment is then structural: index *i* is the same image in every
array, so two quantities need no join to be plotted against each other, and
an aggregate is taken over one population rather than over a mixture of
frame types.

The list deliberately does not depend on the channel, which is what makes
reading one diagnostic in several of them cheap: the columns arrive side by
side against the same images, so a quantity comparing channels is ordinary
arithmetic rather than a join.
"""

from typing import NamedTuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import aliased
import numpy

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageType,
    ObservingSession,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.diagnostic_types import time_quantity
from autowisp.diagnostics.expressions import (
    evaluate_quantities,
    get_needed_values,
)


class _SeriesKeyFields(NamedTuple):
    """The fields of a :class:`SeriesKey`, kept apart only to be checked.

    ``typing.NamedTuple`` prohibits ``__new__`` and ``__init__`` in a class
    body, so the check below cannot go there; deriving from this is what
    gives :class:`SeriesKey` somewhere to put it.
    """

    session_id: int
    image_type: str
    channels: tuple
    quantile_name: str = None


class SeriesKey(_SeriesKeyFields):
    """What one series is: a population of images and a binding.

    The image type is part of the key because a session holds frames of
    several types and a diagnostic rarely means the same thing across them
    -- some are only defined for object frames, and one recorded for both
    would have its aggregates taken over a mixture, making
    ``nanmedian(bg_center[0])`` a median of object and flat frames
    together.

    ``channels`` holds one channel per parameter of what the series draws,
    in the order those parameters are numbered: one for an ordinary
    diagnostic, several where an expression compares channels, and none for
    a quantity over the time alone.

    Every function here takes one of these rather than the fields
    separately, so a caller cannot pair a channel with the wrong session by
    getting an argument order wrong.

    ``quantile_name`` is the odd one out: it says which ``pixel_q*`` a
    series stands for when a caller has expanded the ``pixel_quantiles``
    family into one series per member, and by the time values are read the
    quantity it selects is already a concrete name. Nothing in this module
    consults it -- as nothing but the image list consults the channels --
    but it belongs to the identity of the series.
    """

    __slots__ = ()

    def __new__(cls, session_id, image_type, channels, quantile_name=None):
        """
        Build the key, refusing a bare string where a tuple belongs.

        ``__new__`` rather than a check further on because it has to
        *coerce* as well: bindings reach this from a JSON post as a list,
        and a list in that field makes the key unhashable, which is how it
        is used everywhere.

        The string case is worth refusing loudly because it fails silently
        and selectively: ``channels="R"`` leaves ``channels[0]`` reading
        ``"R"`` and joins to the same id, so a one-character channel
        behaves correctly, while ``"G1"`` becomes the two channels ``G``
        and ``1`` somewhere much later.
        """

        if isinstance(channels, str):
            raise TypeError(
                f"channels={channels!r} is a string: a series binds a "
                "*tuple* of channels, one per parameter of what it draws."
            )
        return super().__new__(
            cls, session_id, image_type, tuple(channels), quantile_name
        )

    @property
    def channel(self):
        """
        The one channel a whole series can be said to belong to, or ``""``.

        There is not really such a thing once a series can bind several --
        that is the point of ``channels`` -- but two things need one
        anyway, and neither is about the data: the frame a click on a point
        opens, and the colour the series is drawn in. Both take the first,
        for want of a better answer. A series binding no channel at all has
        none to give.
        """

        return self.channels[0] if self.channels else ""


def _of_one_type(series_key):
    """Return the WHERE terms selecting one session's frames of one type."""

    return (
        # pylint: disable=no-member
        Image.observing_session_id == series_key.session_id,
        ImageType.name == series_key.image_type,
        Image.jd.is_not(None),
        # pylint: enable=no-member
    )


#: The canonical order, by Julian date and then by id. The id is not
#: decoration: two images of a session can share a ``jd``, and everything
#: here is aligned by position, so an order leaving ties unresolved would let
#: two queries return the same images in different orders and pair a value
#: with the wrong image. That failure is silent -- a plot that looks right
#: and is wrong -- which is worth one more sort key.
# pylint: disable=no-member
_image_order = (Image.jd, Image.id)
# pylint: enable=no-member


def _as_arrays(rows):
    """Return ``(image_ids, jd_values)`` for rows starting ``(id, jd, …)``."""

    if not rows:
        return numpy.empty(0, dtype=int), numpy.empty(0, dtype=float)

    return (
        numpy.fromiter((row[0] for row in rows), dtype=int, count=len(rows)),
        numpy.fromiter((row[1] for row in rows), dtype=float, count=len(rows)),
    )


def get_canonical_images(series_key, db_session):
    """
    Return ``(image_ids, jd_values)`` for one session and image type, by JD.

    Every array built for this series is padded onto this list, so index *i*
    is the same image in each of them and alignment needs no join.

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

    return _as_arrays(
        db_session.execute(
            select(Image.id, Image.jd)  # pylint: disable=no-member
            .select_from(Image)
            .join(
                ImageType,
                # pylint: disable=no-member
                ImageType.id == Image.image_type_id,
                # pylint: enable=no-member
            )
            .where(*_of_one_type(series_key))
            .order_by(*_image_order)
        ).all()
    )


def _diagnostic_values_query(series_key, names, channels):
    """
    Return the statement reading *names* in *channels* for one series.

    Separate from running it so that what it asks the database for can be
    inspected without a database: the predicates below are what keep this
    affordable on an archive too large to scan, and they are this module's
    to get right, unlike which index a particular server then chooses.

    Args:
        series_key(SeriesKey):    The series, for its session and type.

        names(list):    The ``diagnostic_type`` names to read.

        channels(list):    The channels to read them in, one outer join
            each.

    Returns:
        The SQLAlchemy select, ordered by name and then canonically.
    """

    # One alias per channel, so a diagnostic wanted in several arrives as
    # several columns of the one row rather than as several queries. Each
    # pins all three columns of the unique index -- image, channel and
    # diagnostic -- since dropping the channel would match every channel's
    # row, which is both wrong and a scan.
    reads = {channel: aliased(ImageDiagnostics) for channel in channels}

    query = (
        select(
            Image.id,  # pylint: disable=no-member
            Image.jd,  # pylint: disable=no-member
            DiagnosticType.name,
            *(reads[channel].value for channel in channels),
        )
        # Explicit, because diagnostic_type joins on no relation to any of
        # the others and SQLAlchemy cannot pick the left side on its own.
        .select_from(Image).join(
            ImageType,
            ImageType.id == Image.image_type_id,  # pylint: disable=no-member
        )
        # No ON condition but the name filter: this is the cross join that
        # turns "the values that exist" into "one row per image per name",
        # which is what makes the result paddable.
        .join(DiagnosticType, DiagnosticType.name.in_(names))
    )

    for channel in channels:
        read = reads[channel]
        query = query.join(
            read,
            (
                (read.image_id == Image.id)  # pylint: disable=no-member
                & (read.diagnostic_id == DiagnosticType.id)
                & (read.channel == channel)
            ),
            isouter=True,
        )

    return (
        query.where(*_of_one_type(series_key))
        # Name first: the blocks the result is read back in are per name,
        # and order_by appends rather than replaces, so a name added after
        # the image order would sort within it instead of above it.
        .order_by(DiagnosticType.name, *_image_order)
    )


def get_diagnostic_values(series_key, needed, db_session):
    """
    Return the wanted diagnostics for one series, NaN-padded and aligned.

    One query, and nothing to match up afterwards. A cross join pairs every
    wanted diagnostic with every image of the series, and an outer join
    attaches the values, leaving ``NULL`` where nothing was recorded -- so
    the padding is what the database returns rather than something assembled
    from it. The unique index on ``(image_id, channel, diagnostic_id)`` is
    what makes that sound: no image contributes two rows for one diagnostic
    in one channel, so the result is exactly one row per image per name.

    **Several channels at once, still one query.** An expression comparing
    channels needs the same diagnostic read more than once, so there is one
    outer join per distinct channel, each with the channel pinned. That
    keeps the result a rectangle -- the joins only widen it, adding a value
    column per channel rather than rows -- and keeps every probe on the
    unique index, whose second column is exactly what is being pinned.

    Being a rectangle is what lets the values become arrays in one step: a
    column is read out whole and reshaped into one row per name, rather
    than accumulated name by name. Each block's name is taken from its
    first row rather than from a sorted list of the names asked for, so
    nothing depends on the database's collation ordering strings the way
    Python does.

    The image ids come back alongside, because the same query already
    carries them and a caller that needs them should not have to ask again
    -- nor risk a second query disagreeing about the order of images sharing
    a ``jd``. The dates are not returned separately: :data:`time_quantity`
    is asked for by name like anything else, and arrives in the dictionary.

    Args:
        series_key(SeriesKey):    The series to read the values of. Its
            channels are not consulted; what to read in is *needed*, since
            a quantity may be bound to channels other than the series'
            own.

        needed(dict):    ``{name: set of channel tuples}``, from
            :func:`~autowisp.diagnostics.expressions.get_needed_values`.
            May include :data:`time_quantity` at the empty tuple, which is
            taken from the image row rather than from ``image_diagnostics``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{name: {channels: array}}``, keyed exactly as
                *needed* asked, every array the length of the canonical
                image list and ``NaN`` where nothing is recorded. A name no
                ``diagnostic_type`` has is all ``NaN``.

            numpy.ndarray:    The image ids, in canonical order.
    """

    names = sorted(set(needed) - {time_quantity})
    channels = sorted(
        {
            channel
            for name in names
            for combination in needed[name]
            for channel in combination
        }
    )

    if not names:
        image_ids, jd_values = get_canonical_images(series_key, db_session)
        return (
            {time_quantity: {(): jd_values}} if time_quantity in needed else {},
            image_ids,
        )

    rows = db_session.execute(
        _diagnostic_values_query(series_key, names, channels)
    ).all()

    # From the rows rather than from len(names): a name no diagnostic_type
    # has contributes no block at all, and would otherwise throw the shape
    # out for every other name.
    blocks = len({row[2] for row in rows})
    per_block = len(rows) // blocks if blocks else 0

    read_names = [
        rows[start][2] for start in range(0, len(rows), per_block or 1)
    ]
    columns = {
        channel: numpy.fromiter(
            (
                numpy.nan if row[3 + offset] is None else row[3 + offset]
                for row in rows
            ),
            dtype=float,
            count=len(rows),
        ).reshape(blocks or 0, per_block)
        for offset, channel in enumerate(channels)
    }

    values = {
        name: {
            combination: (
                columns[combination[0]][read_names.index(name)]
                if name in read_names
                else numpy.full(per_block, numpy.nan)
            )
            for combination in needed[name]
        }
        for name in names
    }

    image_ids, jd_values = _as_arrays(rows[:per_block])
    if time_quantity in needed:
        values[time_quantity] = {(): jd_values}

    return values, image_ids


def get_quantity_values(series_key, wanted, expressions, db_session):
    """
    Return the quantities of one series as the table bound them.

    *wanted* holds both axes rather than one, because resolving them one at
    a time would waste the two properties this arrangement exists for: the
    diagnostics both axes read are fetched in **one** query for their
    union, and an instantiation the two share is evaluated **once**.

    The image ids come from the same query as the values, so no two results
    have to agree about the order of images sharing a Julian date.

    Args:
        series_key(SeriesKey):    The series to read the values of.

        wanted(dict):    ``{quantity: set of channel tuples}``, each tuple
            holding one channel per parameter of that quantity. A set of
            them because the two axes may be one quantity read in two
            channels, which is how a diagnostic is compared between them.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{quantity: {channels: array}}``, all of the same
                length, and unmasked -- dropping the non-finite entries is
                the caller's business, since the mask has to be taken
                across both axes at once and the image ids masked with it.

            numpy.ndarray:    The image ids that length runs over.

    Raises:
        PipelineError:    If a quantity names nothing, if the expressions
            reference each other in a cycle, or if a binding is the wrong
            length for what it binds.
    """

    values, image_ids = get_diagnostic_values(
        series_key, get_needed_values(wanted, expressions), db_session
    )

    return evaluate_quantities(wanted, expressions, values), image_ids


def _count_images(matching, required, per_channel, db_session):
    """
    Count images whose diagnostic rows satisfy *matching*, per series.

    The shared half of the two counting questions below, which differ only
    in what they match, how many matches an image owes, and whether a
    channel is part of the answer or fixed by the caller.

    Counting rows rather than distinct diagnostics is sound for both,
    because the unique index on ``(image_id, channel, diagnostic_id)``
    admits no duplicate: an image satisfying *n* of what was asked for
    contributes exactly *n* rows to its group.

    Args:
        matching:    The ``WHERE`` selecting the rows that count.

        required(int):    How many matched rows an image must have.

        per_channel(bool):    Whether an image has to satisfy the
            requirement **within one channel** -- which also makes the
            channel something the result varies over -- or may satisfy it
            across several.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(session_label, session_id, image_type[, channel],
            count)`` tuples, the channel present only when *per_channel*.
    """

    grouped = [ImageDiagnostics.image_id]
    carried = [
        Image.observing_session_id.label(  # pylint: disable=no-member
            "session_id"
        ),
        Image.image_type_id.label("image_type_id"),  # pylint: disable=no-member
    ]
    if per_channel:
        grouped.append(ImageDiagnostics.channel)
        carried.append(ImageDiagnostics.channel.label("channel"))

    per_image = (
        select(*carried)
        # Explicit, because which columns are selected varies with
        # *per_channel* and SQLAlchemy would otherwise infer the left side
        # from them -- and infer ``image`` when the channel is not among
        # them, leaving it nothing to join ``image`` to.
        .select_from(ImageDiagnostics)
        .join(
            Image,
            Image.id == ImageDiagnostics.image_id,  # pylint: disable=no-member
        )
        .join(
            DiagnosticType,
            DiagnosticType.id == ImageDiagnostics.diagnostic_id,
        )
        .where(
            matching,
            Image.jd.is_not(None),  # pylint: disable=no-member
        )
        .group_by(*grouped)
        .having(func.count() == required)  # pylint: disable=not-callable
        .subquery()
    )

    # The channel is a column, a grouping and an ordering of the result, or
    # none of the three.
    channel = [per_image.c.channel] if per_channel else []

    return db_session.execute(
        select(
            ObservingSession.label,
            ObservingSession.id,
            ImageType.name,
            *channel,
            func.count(),  # pylint: disable=not-callable
        )
        .select_from(per_image)
        .join(ObservingSession, ObservingSession.id == per_image.c.session_id)
        .join(ImageType, ImageType.id == per_image.c.image_type_id)
        .group_by(ObservingSession.id, ImageType.id, *channel)
        .order_by(ObservingSession.label, ImageType.name, *channel)
    ).all()


def count_images_with_all(needed, db_session):
    """
    Count images holding all of *needed*, per (session, type, channel).

    What a slot **may** be bound to: the channels a quantity could be read
    in, and how many images each would draw. Deliberately spans every
    observing session, that being what the series table lists -- so there
    is no one session to anchor it to, and its cost is proportional to the
    images carrying the diagnostic, which *Scaling* names as a standing
    limit rather than something an index could remove.

    Args:
        needed(set):    ``DiagnosticType`` names that must all be recorded
            for an image to count, **in the same channel**. An empty set
            means no quantity constrains the result, which only happens
            when every quantity is :data:`time_quantity`; nothing is
            plottable then.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(session_label, session_id, image_type, channel, count)``
            tuples.
    """

    if not needed:
        return []

    return _count_images(
        DiagnosticType.name.in_(needed),
        len(needed),
        True,
        db_session,
    )


def count_images_with_channels(requirements, db_session):
    """
    Count images holding every ``(diagnostic, channel)`` pair, per series.

    What a *binding* actually draws, once the channels are chosen -- the
    exact question, where :func:`count_images_with_all` answers the looser
    one that fills the dropdowns. The difference is a line of SQL and the
    whole of the meaning: an image counts when its rows cover every pair
    *between them*, so one requirement may be met in R and another in B,
    which is what a quantity comparing channels needs.

    Args:
        requirements:    ``(diagnostic_name, channel)`` pairs that must all
            be recorded for an image to count. Deduplicated here, since the
            two axes of one plot may read the same diagnostic in the same
            channel. Empty means nothing constrains the result.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(session_label, session_id, image_type, count)`` tuples.
            No channel among them: the binding names the channels, so they
            are not what the rows vary over.
    """

    requirements = {tuple(pair) for pair in requirements}
    if not requirements:
        return []

    return _count_images(
        or_(
            *(
                and_(
                    DiagnosticType.name == name,
                    ImageDiagnostics.channel == channel,
                )
                for name, channel in requirements
            )
        ),
        len(requirements),
        False,
        db_session,
    )
