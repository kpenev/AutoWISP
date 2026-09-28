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
from sqlalchemy.orm import aliased

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageMasterSelection,
    ImageType,
    ObservingSession,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.expression_series import (
    photref_binding,
    restrict_to_references,
)


def _count_images(
    restrict, required, per_channel, db_session, *, by_reference=False
):
    """
    Count images whose diagnostic rows *restrict* keeps, per series.

    The shared half of the two counting questions below, which differ in
    which rows count, how many of them an image owes, and whether a channel
    -- and with it a photometric reference -- is part of the answer or
    fixed by the caller.

    Counting rows rather than distinct diagnostics is sound for both,
    because the unique index on ``(image_id, channel, diagnostic_id)``
    admits no duplicate: an image satisfying *n* of what was asked for
    contributes exactly *n* rows to its group. Joining the image's binding
    in a row's channel keeps that so, there being at most one.

    Args:
        restrict:    Function restricting a query over ``image_diagnostics``
            rows, joined to their ``image`` and ``diagnostic_type``, to the
            rows that count.

        required(int):    How many rows an image must have kept.

        per_channel(bool):    Whether an image has to satisfy the
            requirement **within one channel** -- which also makes the
            channel something the result varies over -- or may satisfy it
            across several.

        db_session:    An active SQLAlchemy database session.

        by_reference(bool):    Whether the result varies over the
            photometric reference as well: the one each image is bound to
            in the channel of its rows, so only with *per_channel*. An
            image bound to none there is not counted.

    Returns:
        list:    One tuple per series, holding the session label, the
            session id, the image type, the channel where *per_channel*,
            the photref's ``MasterFile`` id where *by_reference*, and the
            count.
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
    binding = aliased(ImageMasterSelection)
    if by_reference:
        grouped.append(binding.master_file_id)
        carried.append(binding.master_file_id.label("photref"))

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
    )
    if by_reference:
        per_image = per_image.join(
            binding, photref_binding(binding, ImageDiagnostics.channel)
        )
    per_image = (
        restrict(per_image)
        .where(Image.jd.is_not(None))  # pylint: disable=no-member
        .group_by(*grouped)
        .having(func.count() == required)  # pylint: disable=not-callable
        .subquery()
    )

    # What the result varies over besides the session and the type -- the
    # channel, then the reference -- is a column, a grouping and an
    # ordering of it, or none of the three.
    split = [per_image.c.channel] if per_channel else []
    if by_reference:
        split.append(per_image.c.photref)

    return db_session.execute(
        select(
            ObservingSession.label,
            ObservingSession.id,
            ImageType.name,
            *split,
            func.count(),  # pylint: disable=not-callable
        )
        .select_from(per_image)
        .join(ObservingSession, ObservingSession.id == per_image.c.session_id)
        .join(ImageType, ImageType.id == per_image.c.image_type_id)
        .group_by(ObservingSession.id, ImageType.id, *split)
        .order_by(ObservingSession.label, ImageType.name, *split)
    ).all()


def count_images_with_all(needed, db_session, *, by_reference=False):
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

        by_reference(bool):    Whether to count per photometric reference
            too, for a slot reading a diagnostic ``fit_magnitudes``
            produces, whose values mean something only within one
            reference: the one each image is bound to in the channel. An
            image bound to none there is not counted.

    Returns:
        list:    ``(session_label, session_id, image_type, channel, count)``
            tuples, with the photref's ``MasterFile`` id before the count
            where *by_reference*.
    """

    if not needed:
        return []

    return _count_images(
        lambda query: query.where(DiagnosticType.name.in_(needed)),
        len(needed),
        True,
        db_session,
        by_reference=by_reference,
    )


def count_images_with_channels(requirements, db_session, references=()):
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

        references:    ``(channel, photref)`` pairs an image must also be
            bound to, the photref a ``MasterFile`` id, as a series key's
            ``reference_pairs`` gives them. What restricts a series to the
            images fit against its references restricts its count the same
            way.

    Returns:
        list:    ``(session_label, session_id, image_type, count)`` tuples.
            No channel among them: the binding names the channels, so they
            are not what the rows vary over.
    """

    requirements = {tuple(pair) for pair in requirements}
    if not requirements:
        return []

    return _count_images(
        lambda query: restrict_to_references(query, references).where(
            or_(
                *(
                    and_(
                        DiagnosticType.name == name,
                        ImageDiagnostics.channel == channel,
                    )
                    for name, channel in requirements
                )
            )
        ),
        len(requirements),
        False,
        db_session,
    )
