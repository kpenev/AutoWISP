"""How many images each series would draw, read from the project database.

The series table's question, asked before anything is read or evaluated:
which sessions, image types and channels a quantity could be bound to, and
how many images a binding would draw. Answered by counting
``image_diagnostics`` rows alone, never by evaluating an expression, since
there is a table row per observing session and image type and evaluating
per row would be work proportional to the whole image collection. Reading
the values of a series is :mod:`autowisp.diagnostics.expression_series`.
"""

from sqlalchemy import and_, func, or_, select

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
        list:    One tuple per series, holding the session label, the
            session id, the image type, the channel where *per_channel*,
            and the count.
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
    images carrying the diagnostic. That is a standing limit rather than
    something an index could remove: enumerating the sessions *is* the
    question being asked.

    Args:
        needed(set):    ``DiagnosticType`` names that must all be recorded
            for an image to count, **in the same channel**. An empty set
            means no quantity constrains the result, which only happens
            when every quantity is
            :data:`~autowisp.diagnostics.diagnostic_types.time_quantity`;
            nothing is plottable then.

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
