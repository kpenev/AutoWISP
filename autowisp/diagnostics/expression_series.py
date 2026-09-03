"""Values for one series of images, read from the project database.

Tier 2 of the expression layer: it knows the project database and nothing
else. Above it, the browser interface adds Django and a way of editing the
library; below it, :mod:`autowisp.diagnostics.expressions` knows what an
expression *means* and has no database at all. This module is the join
between them -- it turns a session, image type and channel into the
``{name: array}`` that tier 1 evaluates against.

Everything here is built on **one canonical image list per session and image
type**, ordered by Julian date, with ``NaN`` wherever a value is not
recorded. Alignment is then structural: index *i* is the same image in every
array, so two quantities need no join to be plotted against each other, and
an aggregate is taken over one population rather than over a mixture of
frame types.
"""

from typing import NamedTuple

from sqlalchemy import select, func
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
    evaluate_expressions,
    order_expressions,
)


class SeriesKey(NamedTuple):
    """What one series is, and what its id encodes.

    The image type is part of the key because a session holds frames of
    several types and a diagnostic rarely means the same thing across them
    -- some are only defined for object frames, and one recorded for both
    would have its aggregates taken over a mixture, making
    ``nanmedian(bg_center)`` a median of object and flat frames together.

    Every function here takes one of these rather than the fields
    separately, so a caller cannot pair a channel with the wrong session by
    getting an argument order wrong.

    ``quantile_name`` is the odd one out: it says which ``pixel_q*`` a
    series stands for when a caller has expanded the ``pixel_quantiles`` family
    into one series per member, and by the time values are read the
    quantity it selects is already a concrete name. Nothing in this module
    consults it -- as nothing but the image list consults the channel -- but
    it belongs to the identity of the series, and so to its id.
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
        Return the opaque string identifying this series to a client.

        The browser interface makes it an HTML element id, builds four more
        element ids from it, and keys by it the ``datasets`` object the
        client posts back, so it has to survive that round trip unchanged.

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


def get_diagnostic_values(series_key, names, db_session):
    """
    Return the named diagnostics for one series, NaN-padded and aligned.

    One query, and nothing to match up afterwards. A cross join pairs every
    wanted diagnostic with every image of the series, and an outer join
    attaches the values, leaving ``NULL`` where nothing was recorded -- so
    the padding is what the database returns rather than something assembled
    from it. The unique index on ``(image_id, channel, diagnostic_id)`` is
    what makes that sound: no image contributes two rows for one diagnostic,
    so the result is exactly one row per image per name.

    Being a rectangle is what lets the values become arrays in one step: the
    column is read out whole and reshaped into one row per name, rather than
    accumulated name by name. Each block's name is taken from its first row
    rather than from a sorted list of the names asked for, so nothing
    depends on the database's collation ordering strings the way Python
    does.

    The image ids come back alongside, because the same query already
    carries them and a caller that needs them should not have to ask again
    -- nor risk a second query disagreeing about the order of images sharing
    a ``jd``. The dates are not returned separately: :data:`time_quantity`
    is asked for by name like anything else, and arrives in the dictionary.

    Args:
        series_key(SeriesKey):    The series to read the values of.

        names:    The ``diagnostic_type`` names wanted. May include
            :data:`time_quantity`, which is taken from the image row rather
            than from ``image_diagnostics``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    An array per name, every one the length of the
                canonical image list, ``NaN`` where the diagnostic is not
                recorded. A name no ``diagnostic_type`` has is all ``NaN``.

            numpy.ndarray:    The image ids, in canonical order.
    """

    wanted = set(names) - {time_quantity}

    if not wanted:
        image_ids, jd_values = get_canonical_images(series_key, db_session)
        return (
            {time_quantity: jd_values} if time_quantity in names else {},
            image_ids,
        )

    rows = db_session.execute(
        select(
            Image.id,  # pylint: disable=no-member
            Image.jd,  # pylint: disable=no-member
            DiagnosticType.name,
            ImageDiagnostics.value,
        )
        # Explicit, because diagnostic_type joins on no relation to any of
        # the others and SQLAlchemy cannot pick the left side on its own.
        .select_from(Image)
        .join(
            ImageType,
            ImageType.id == Image.image_type_id,  # pylint: disable=no-member
        )
        # No ON condition but the name filter: this is the cross join that
        # turns "the values that exist" into "one row per image per name",
        # which is what makes the result paddable.
        .join(DiagnosticType, DiagnosticType.name.in_(wanted))
        .join(
            ImageDiagnostics,
            (
                # pylint: disable=no-member
                (ImageDiagnostics.image_id == Image.id)
                # pylint: enable=no-member
                & (ImageDiagnostics.diagnostic_id == DiagnosticType.id)
                & (ImageDiagnostics.channel == series_key.channel)
            ),
            isouter=True,
        )
        .where(*_of_one_type(series_key))
        # Name first: the blocks this is read back in are per name, and
        # order_by appends rather than replaces, so a name added after the
        # image order would sort within it instead of above it.
        .order_by(DiagnosticType.name, *_image_order)
    ).all()

    # From the rows rather than from len(wanted): a name no diagnostic_type
    # has contributes no block at all, and would otherwise throw the shape
    # out for every other name.
    blocks = len({row[2] for row in rows})
    per_block = len(rows) // blocks if blocks else 0

    values = dict(
        zip(
            (rows[start][2] for start in range(0, len(rows), per_block or 1)),
            numpy.fromiter(
                (numpy.nan if row[3] is None else row[3] for row in rows),
                dtype=float,
                count=len(rows),
            ).reshape(blocks or 0, per_block),
        )
    )
    image_ids, jd_values = _as_arrays(rows[:per_block])

    for name in wanted - set(values):
        values[name] = numpy.full(per_block, numpy.nan)
    if time_quantity in names:
        values[time_quantity] = jd_values

    return values, image_ids


def get_series_values(series_key, quantities, expressions, db_session):
    """
    Return every wanted quantity for one series, and the images behind them.

    *quantities* is a sequence rather than a single name because the figure
    wants both axes of the same series, and resolving them one at a time
    would waste the two properties this arrangement exists for: the
    diagnostics both axes need are read in **one** query for their union,
    and a subexpression the two axes share is evaluated **once**, in one
    symbol table, rather than once per axis.

    The image ids come from the same query as the values, so no two results
    have to agree about the order of images sharing a Julian date.

    The arrays are returned unmasked. Dropping the non-finite entries is the
    caller's business, because the mask has to be taken across both axes at
    once and the image ids masked with it.

    Args:
        series_key(SeriesKey):    The series to read the values of.

        quantities:    The names to resolve: diagnostics, expressions or
            :data:`time_quantity`, in any mixture. Repeats are harmless --
            plotting a quantity against itself asks for one array twice.

        expressions(dict):    The library, ``{name: expression}``. Empty
            where the caller has none, which resolves plain diagnostics and
            :data:`time_quantity` and nothing else.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{quantity: array}``, all of the same length.

            numpy.ndarray:    The image ids that length runs over.

    Raises:
        PipelineError:    If a quantity names nothing, or the expressions
            reference each other in a cycle.
    """

    _, needed = order_expressions(quantities, expressions)
    values, image_ids = get_diagnostic_values(series_key, needed, db_session)

    return evaluate_expressions(quantities, expressions, values), image_ids


def count_images_with_all(needed, db_session):
    """
    Count images holding all of *needed*, per (session, type, channel).

    Args:
        needed(set):    ``DiagnosticType`` names that must all be recorded
            for an image to count. An empty set means no quantity
            constrains the result, which only happens when every quantity is
            :data:`time_quantity`; nothing is plottable then.

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


def get_expression_availability(name, expressions, db_session):
    """
    Return the series one expression can be plotted for, and how many images.

    The count comes from the SQL aggregate rather than from evaluating
    anything: the question is how many images record every diagnostic the
    expression reaches, which is a question about rows. It is an upper bound
    on the points drawn, since the arithmetic can still yield ``NaN``.

    Args:
        name(str):    The expression to report on.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    ``(session_label, session_id, image_type, channel, count)``
            tuples, empty where the expression needs nothing recorded.

    Raises:
        PipelineError:    If the expression references a name that resolves
            to nothing, or takes part in a cycle.
    """

    _, needed = order_expressions([name], expressions)

    return count_images_with_all(needed, db_session)
