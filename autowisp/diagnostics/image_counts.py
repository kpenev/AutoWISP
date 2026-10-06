"""How many images each series would draw, read from the project database.

The series table's question, asked before anything is read or evaluated:
which sessions, image types, channels and photometries a quantity could be
bound to, and how many images a binding would draw. Answered by counting
diagnostic rows alone -- in ``image_diagnostics``, or in
``photometry_diagnostics`` for a diagnostic recorded per photometry --
never by evaluating an expression, since there is a table row per observing
session and image type and evaluating per row would be work proportional to
the whole image collection. Reading the values of a series is
:mod:`autowisp.diagnostics.expression_series`.
"""

from sqlalchemy import and_, func, or_, select, union_all
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
    PhotometryDiagnostics,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.diagnostic_types import photometry_diagnostic_names
from autowisp.diagnostics.expression_series import (
    photref_binding,
    restrict_to_references,
)


def _recorded(per_image, per_photometry, columns=(), *, distinct=False):
    """
    Return the diagnostic rows the counts are made of, as one subquery.

    The rows of ``image_diagnostics`` *per_image* keeps, and those of
    ``photometry_diagnostics`` *per_photometry* keeps, side by side. Each
    table is filtered before the union rather than after it, so each branch
    is a probe of its own table's indexes, whatever a database makes of a
    filter over a union.

    Args:
        per_image:    The condition, over ``image_diagnostics`` and the
            ``diagnostic_type`` it is joined to, keeping the rows that
            count; ``None`` where none does. At least one of the two
            conditions must be given.

        per_photometry:    The same, over ``photometry_diagnostics``.

        columns:    The columns carried besides ``image_id`` and
            ``diagnostic_id``: those the count varies over, and no others,
            since which columns a row has decides which rows are alike.

        distinct(bool):    Whether rows alike in every column carried are
            one row: those of one diagnostic in several photometries, where
            the photometry is not carried and any of them will do.

    Returns:
        The subquery, with ``image_id``, ``diagnostic_id`` and *columns*.
    """

    branches = []
    for table, condition in (
        (ImageDiagnostics, per_image),
        (PhotometryDiagnostics, per_photometry),
    ):
        if condition is None:
            continue
        branch = (
            select(
                table.image_id,
                table.diagnostic_id,
                *(getattr(table, column) for column in columns),
            )
            .join(DiagnosticType, DiagnosticType.id == table.diagnostic_id)
            .where(condition)
        )
        branches.append(branch.distinct() if distinct else branch)

    return (
        union_all(*branches) if len(branches) > 1 else branches[0]
    ).subquery()


# Each keyword is named at every call site, and bundling them would only
# move the list somewhere less visible.
# pylint: disable=too-many-arguments
def _count_images(
    rows, required, split_by, db_session, *, by_reference=False, references=()
):
    """
    Count images with *required* of *rows* each, per series.

    The shared half of the three counting questions below, which differ in
    which rows count, how many of them an image owes, and what the result
    varies over besides the session and the type: the channel -- and with
    it, possibly, a photometric reference -- the photometry, or neither.

    Counting rows rather than distinct diagnostics is sound for all three,
    because each makes a row stand for one requirement: the unique index of
    each table admits no duplicate, and rows of several photometries that
    stand for one are merged by :func:`_recorded`. An image satisfying *n*
    of what was asked for contributes exactly *n* rows to its group. Joining
    the image's binding in a row's channel keeps that so, there being at
    most one.

    Args:
        rows:    What :func:`_recorded` returned: the rows that count.

        required(int):    How many rows an image must have.

        split_by(tuple):    The columns of *rows* the result varies over,
            which an image then has to satisfy the requirement within:
            ``()``, ``("channel",)`` or ``("photometry_id",)``.

        db_session:    An active SQLAlchemy database session.

        by_reference(bool):    Whether the result varies over the
            photometric reference as well: the one each image is bound to
            in the channel of its rows, so only when split by channel. An
            image bound to none there is not counted.

        references:    ``(channel, photref)`` pairs an image must also be
            bound to, as :func:`count_images_with_channels` takes them.

    Returns:
        list:    One tuple per series, holding the session label, the
            session id, the image type, the *split_by* columns, the
            photref's ``MasterFile`` id where *by_reference*, and the count.
    """

    grouped = [rows.c.image_id, *(rows.c[column] for column in split_by)]
    carried = [
        Image.observing_session_id.label(  # pylint: disable=no-member
            "session_id"
        ),
        Image.image_type_id.label("image_type_id"),  # pylint: disable=no-member
        *(rows.c[column] for column in split_by),
    ]
    binding = aliased(ImageMasterSelection)
    if by_reference:
        grouped.append(binding.master_file_id)
        carried.append(binding.master_file_id.label("photref"))

    per_image = (
        select(*carried)
        # Explicit, because which columns are selected varies with
        # *split_by* and SQLAlchemy would otherwise infer the left side
        # from them -- and infer ``image`` when none of *rows* is among
        # them, leaving it nothing to join ``image`` to.
        .select_from(rows).join(
            Image, Image.id == rows.c.image_id  # pylint: disable=no-member
        )
    )
    if by_reference:
        per_image = per_image.join(
            binding, photref_binding(binding, rows.c.channel)
        )
    per_image = (
        restrict_to_references(per_image, references)
        .where(Image.jd.is_not(None))  # pylint: disable=no-member
        .group_by(*grouped)
        .having(func.count() == required)  # pylint: disable=not-callable
        .subquery()
    )

    # What the result varies over besides the session and the type -- the
    # channel or the photometry, then the reference -- is a column, a
    # grouping and an ordering of it, or none of the three.
    split = [per_image.c[column] for column in split_by]
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


# pylint: enable=too-many-arguments


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

    The photometry is not among what the result varies over: it is chosen
    apart from the channel, and :func:`count_images_per_photometry` counts
    its options. So a diagnostic recorded per photometry is there when it
    is there in any photometry of the channel.

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

    per_photometry = set(needed) & photometry_diagnostic_names()
    per_image = set(needed) - per_photometry
    return _count_images(
        _recorded(
            DiagnosticType.name.in_(per_image) if per_image else None,
            (
                DiagnosticType.name.in_(per_photometry)
                if per_photometry
                else None
            ),
            ("channel",),
            distinct=True,
        ),
        len(needed),
        ("channel",),
        db_session,
        by_reference=by_reference,
    )


def count_images_with_channels(requirements, db_session, references=()):
    """
    Count images holding every diagnostic read, per series.

    What a *binding* actually draws, once the channels and photometries are
    chosen -- the exact question, where :func:`count_images_with_all` and
    :func:`count_images_per_photometry` answer the looser ones that fill
    the dropdowns. The difference is a line of SQL and the whole of the
    meaning: an image counts when its rows cover every read *between them*,
    so one requirement may be met in R and another in B, or one in aperture
    0 and another in aperture 2, which is what a quantity comparing
    channels or photometries needs.

    Args:
        requirements:    ``(diagnostic_name, channel, photometry)`` reads
            that must all be recorded for an image to count, the photometry
            an id for a diagnostic recorded per photometry and ``None`` for
            one recorded per image. Deduplicated here, since the two axes of
            one plot may read the same diagnostic in the same channel and
            photometry. Empty means nothing constrains the result.

        db_session:    An active SQLAlchemy database session.

        references:    ``(channel, photref)`` pairs an image must also be
            bound to, the photref a ``MasterFile`` id, as a series key's
            ``reference_pairs`` gives them. What restricts a series to the
            images fit against its references restricts its count the same
            way.

    Returns:
        list:    ``(session_label, session_id, image_type, count)`` tuples.
            No channel or photometry among them: the binding names those,
            so they are not what the rows vary over.
    """

    requirements = {tuple(read) for read in requirements}
    if not requirements:
        return []

    per_image = [
        and_(DiagnosticType.name == name, ImageDiagnostics.channel == channel)
        for name, channel, photometry in requirements
        if photometry is None
    ]
    per_photometry = [
        and_(
            DiagnosticType.name == name,
            PhotometryDiagnostics.channel == channel,
            PhotometryDiagnostics.photometry_id == photometry,
        )
        for name, channel, photometry in requirements
        if photometry is not None
    ]
    return _count_images(
        _recorded(
            or_(*per_image) if per_image else None,
            or_(*per_photometry) if per_photometry else None,
        ),
        len(requirements),
        (),
        db_session,
        references=references,
    )


def count_images_per_photometry(needed, db_session):
    """
    Count images holding all of *needed*, per (session, type, photometry).

    What a photometry column **may** be bound to: the photometries a
    quantity could be read in, and how many images each would draw -- the
    photometry's counterpart of :func:`count_images_with_all`. Which
    channels the reads are in is for the channel columns to choose, so an
    image counts in a photometry where it records every one of *needed*,
    in whichever channels. Only the names of
    :func:`~autowisp.diagnostics.diagnostic_types.photometry_diagnostic_names`
    are recorded per photometry, so no other can be among them.

    Args:
        needed(set):    Diagnostic names that must all be recorded in the
            same photometry for an image to count. Empty means nothing is
            read per photometry, and there is nothing to count.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    One tuple per photometry of each session and image type,
            ``(session_label, session_id, image_type, photometry, count)``,
            the photometry an id as ``fit_magnitudes`` records it.
    """

    if not needed:
        return []

    return _count_images(
        _recorded(
            None,
            DiagnosticType.name.in_(needed),
            ("photometry_id",),
            distinct=True,
        ),
        len(needed),
        ("photometry_id",),
        db_session,
    )
