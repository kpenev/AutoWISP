"""Values for one series of images, read from the project database.

Tier 2 of the expression layer: it knows the project database and nothing
else. Above it, the browser interface adds Django and a way of editing the
library; below it, :mod:`autowisp.diagnostics.expressions` knows what an
expression *means* and has no database at all. This module is the join
between them -- it turns a session, an image type and the channels and
photometries a series binds into the ``{name: {(channels, photometries):
array}}`` that tier 1 evaluates against.

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

It does depend on the photometric references a series names. A diagnostic
``fit_magnitudes`` produces depends on the reference an image was fit
against, so a series reading one is restricted to the images bound to one
reference in each channel it reads it in, and :func:`split_series` finds
which such populations a session holds.

A photref group is the one population not identified by a session: it is
given as its members, ``(image, channel)`` pairs, and
:func:`get_custom_group_values` reads and evaluates over those, one point per
member, with the same query and evaluation.
"""

# Over pylint's 1000-line default, and left there: everything here is one
# path -- a population, one query over it, one evaluation -- which the
# series and the custom group share, and splitting it would leave each half
# importing the other's private helpers. Counting images, which shares none
# of it, is already apart in :mod:`autowisp.diagnostics.image_counts`.
# pylint: disable=too-many-lines

from typing import NamedTuple

from sqlalchemy import func, select, union_all
from sqlalchemy.orm import aliased
import numpy

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageMasterSelection,
    ImageType,
    MasterType,
    PhotometryDiagnostics,
)

# pylint: enable=no-name-in-module
from autowisp.diagnostics.diagnostic_types import (
    magfit_diagnostic_names,
    time_quantity,
)
from autowisp.diagnostics.expressions import (
    evaluate_quantities,
    get_channel_arity,
    get_needed_values,
    get_photometry_arity,
)
from autowisp.exceptions import PipelineError


class _SeriesKeyFields(NamedTuple):
    """The fields of a :class:`SeriesKey`, kept apart only to be checked.

    ``typing.NamedTuple`` prohibits ``__new__`` and ``__init__`` in a class
    body, so the check below cannot go there; deriving from this is what
    gives :class:`SeriesKey` somewhere to put it.
    """

    session_id: int
    image_type: str
    channels: tuple
    photometries: tuple
    photrefs: tuple


def _normalize_photrefs(channels, photrefs):
    """Return *photrefs* with every slot on a channel given its reference.

    A slot left ``None`` takes the reference another slot on its channel
    has, where there is exactly one; a channel given two different ones is
    left alone, selecting nothing, as it would anyway.
    """

    by_channel = {}
    for channel, photref in zip(channels, photrefs):
        if photref is not None:
            by_channel.setdefault(channel, set()).add(photref)

    return tuple(
        (
            next(iter(by_channel[channel]))
            if photref is None and len(by_channel.get(channel, ())) == 1
            else photref
        )
        for channel, photref in zip(channels, photrefs)
    )


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
    a quantity over the time alone. After them come the quoted channels a
    diagnostic of :func:`magfit_diagnostic_names` is read in, if any, in
    the order the text first quotes them: they bind nothing, but need a
    reference like any other channel magfit is read in, and so a slot to
    hold it. Whatever turns ``channels`` into bindings slices it from the
    front, by arity, and never reaches them.

    ``photometries`` holds one photometry id per photometry parameter of
    what the series draws, in the same order -- the axes' parameters laid
    end to end -- and is empty where nothing it draws is read per
    photometry. It is independent of ``channels``: a photometry belongs to
    no channel slot, every image having every photometry in every channel.

    ``photrefs`` holds, per slot, the ``single_photref`` master file the
    series' images are bound to in that slot's channel, restricting the
    population to the images fit against it there; ``None`` where the
    population is not restricted in that channel. A reference is a
    property of a channel, not of a slot -- an image has one binding per
    channel -- so slots on one channel carry the same one. A channel
    something reads a magfit diagnostic in must have one (see
    :func:`get_diagnostic_values`), since its values then depend on the
    reference as well as on the image.

    Every function here takes one of these rather than the fields
    separately, so a caller cannot pair a channel with the wrong session by
    getting an argument order wrong.
    """

    __slots__ = ()

    def __new__(
        cls, session_id, image_type, channels, *, photometries=(), photrefs=None
    ):
        """
        Build the key, coercing what arrives in the wrong shape.

        ``__new__`` rather than a check further on because it has to
        *coerce* as well: bindings reach this from a JSON post as a list,
        and a list in that field makes the key unhashable, which is how it
        is used everywhere.

        A bare string for the channels is refused loudly because it fails
        silently and selectively: ``channels="R"`` leaves ``channels[0]``
        reading ``"R"`` and joins to the same id, so a one-character
        channel behaves correctly, while ``"G1"`` becomes the two channels
        ``G`` and ``1`` somewhere much later.

        *photometries* and *photrefs* are keyword-only: both are tuples of
        ids, so one passed in the other's place would be accepted and read
        wrongly rather than refused.

        Omitted, *photrefs* is ``None`` in every slot, so a key built
        without it equals one spelling that out. Given, it must be as long
        as *channels*, and a slot left ``None`` on a channel another slot
        has a reference for takes it: a producer filling only the slots
        that read magfit then builds the same key as one filling every
        slot on the channel.
        """

        if isinstance(channels, str):
            raise TypeError(
                f"channels={channels!r} is a string: a series binds a "
                "*tuple* of channels, one per parameter of what it draws."
            )
        channels = tuple(channels)

        if photrefs is None:
            photrefs = (None,) * len(channels)
        photrefs = tuple(
            None if photref is None else int(photref) for photref in photrefs
        )
        if len(photrefs) != len(channels):
            raise ValueError(
                f"photrefs={photrefs!r} gives {len(photrefs)} references for "
                f"{len(channels)} channels {channels!r}: a series has one "
                "per slot, None where it is not restricted by reference."
            )

        return super().__new__(
            cls,
            session_id,
            image_type,
            channels,
            tuple(int(photometry) for photometry in photometries),
            _normalize_photrefs(channels, photrefs),
        )

    # False positive: pylint models the inherited method as taking each field
    # as a parameter, where it really takes ``self, /, **kwargs`` like this.
    def _replace(self, /, **changes):  # pylint: disable=arguments-differ
        """Build through :meth:`__new__`, which the inherited one skips."""

        return type(self)(**{**self._asdict(), **changes})

    @property
    def reference_pairs(self):
        """
        The ``(channel, photref)`` pairs restricting the population.

        One per distinct pair with a reference, sorted so that the joins
        built from them come out the same every time. A channel given two
        references contributes both, which between them select nothing.
        """

        return tuple(
            sorted(
                {
                    (channel, photref)
                    for channel, photref in zip(self.channels, self.photrefs)
                    if photref is not None
                }
            )
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


def _in_series(query, series_key):
    """
    Return *query*, over ``Image``, keeping only the images of one series.

    One session's frames of one type, bound to the key's references. The
    query must already select from ``Image``, which is what gets joined to
    its type.
    """

    return restrict_to_references(
        query.join(
            ImageType,
            ImageType.id == Image.image_type_id,  # pylint: disable=no-member
        ),
        series_key.reference_pairs,
    ).where(
        # pylint: disable=no-member
        Image.observing_session_id == series_key.session_id,
        ImageType.name == series_key.image_type,
        Image.jd.is_not(None),
        # pylint: enable=no-member
    )


def photref_binding(binding, channel):
    """
    Return the ON terms matching *binding* to an image's photref row.

    Everything but the master file, which is what the callers vary: pinned
    to one, or read out to see which it is. All three columns of the
    primary key are pinned, so each probe is one index lookup. Public
    because the image counts of
    :mod:`autowisp.diagnostics.image_counts` join the same way.

    Args:
        binding:    An alias of ``ImageMasterSelection``.

        channel:    The channel to match it in: a name, or a column holding
            one, such as the channel of the diagnostic row being read.
    """

    return (
        (binding.image_id == Image.id)  # pylint: disable=no-member
        & (binding.channel == channel)
        & (
            binding.master_type_id
            == select(MasterType.id)
            .where(MasterType.name == "single_photref")
            .scalar_subquery()
        )
    )


def restrict_to_references(query, reference_pairs):
    """
    Return *query*, over ``Image``, keeping images bound to the references.

    One inner join per pair, so an image stays only if it was fit against
    every one of them in its channel; no pairs, and the query comes back
    unchanged. Public because the image counts of
    :mod:`autowisp.diagnostics.image_counts` restrict the same way.

    Args:
        query:    A select over ``Image``.

        reference_pairs:    ``(channel, photref)`` pairs, the photref a
            ``MasterFile`` id, as :attr:`SeriesKey.reference_pairs` gives
            them for a series.
    """

    for channel, photref in reference_pairs:
        binding = aliased(ImageMasterSelection)
        query = query.join(
            binding,
            photref_binding(binding, channel)
            & (binding.master_file_id == photref),
        )
    return query


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

    The channels of *series_key* are deliberately not used -- the list is
    the same for every channel -- but the image type is: frames of different
    types are different populations, and mixing them would put a flat frame
    and an object frame in one array for an aggregate to average over. Its
    references are used too, for the same reason: images fit against
    different references are different populations of magfit values.

    Args:
        series_key(SeriesKey):    The series to list the images of.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:    Arrays of image IDs and of Julian dates, of equal length.
    """

    return _list_images(lambda query: _in_series(query, series_key), db_session)


def _list_images(population, db_session):
    """
    Return ``(image_ids, jd_values)`` for a population, in canonical order.

    Args:
        population:    Function restricting a query over ``Image`` to the
            images wanted, as :func:`_in_series` does to a series'.

        db_session:    An active SQLAlchemy database session.
    """

    # pylint: disable=no-member
    query = population(select(Image.id, Image.jd).select_from(Image))
    # pylint: enable=no-member
    return _as_arrays(db_session.execute(query.order_by(*_image_order)).all())


def _diagnostic_values_query(population, names, bindings):
    """
    Return the statement reading *names* in each of *bindings*.

    Separate from running it so that what it asks the database for can be
    inspected without a database: the predicates below are what keep this
    affordable on an archive too large to scan, and they are this module's
    to get right, unlike which index a particular server then chooses.

    Args:
        population:    Function restricting a query over ``Image`` to the
            images to read, as :func:`_in_series` does to a series'.

        names(list):    The ``diagnostic_type`` names to read.

        bindings(list):    The diagnostic bindings to read them in, one
            outer join each: ``((channel,), ())`` on ``image_diagnostics``,
            and ``((channel,), (photometry,))`` on
            ``photometry_diagnostics``. A diagnostic takes one channel and
            at most one photometry, so there is nothing else a binding here
            can be.

    Returns:
        The SQLAlchemy select, ordered by name and then canonically.
    """

    # One alias per binding, so a diagnostic wanted in several arrives as
    # several columns of the one row rather than as several queries. Each
    # pins every column of its table's unique index -- image, channel and
    # diagnostic, and the photometry where there is one -- since dropping
    # the channel or the photometry would match every channel's or every
    # photometry's row, which is both wrong and a scan.
    aliases = [
        aliased(PhotometryDiagnostics if photometries else ImageDiagnostics)
        for _, photometries in bindings
    ]

    query = population(
        select(
            Image.id,  # pylint: disable=no-member
            Image.jd,  # pylint: disable=no-member
            DiagnosticType.name,
            *(alias.value for alias in aliases),
        )
        # Explicit, because diagnostic_type joins on no relation to any of
        # the others and SQLAlchemy cannot pick the left side on its own.
        .select_from(Image)
    )
    # No ON condition but the name filter: this is the cross join that
    # turns "the values that exist" into "one row per image per name",
    # which is what makes the result paddable.
    query = query.join(DiagnosticType, DiagnosticType.name.in_(names))

    for ((channel,), photometries), alias in zip(bindings, aliases):
        condition = (
            (alias.image_id == Image.id)  # pylint: disable=no-member
            & (alias.diagnostic_id == DiagnosticType.id)
            & (alias.channel == channel)
        )
        if photometries:
            condition &= alias.photometry_id == photometries[0]
        query = query.join(alias, condition, isouter=True)

    # Name first: the blocks the result is read back in are per name, and
    # order_by appends rather than replaces, so a name added after the image
    # order would sort within it instead of above it.
    return query.order_by(DiagnosticType.name, *_image_order)


def _magfit_channels(needed):
    """Return the channels *needed* reads a magfit diagnostic in, sorted.

    Args:
        needed(dict):    ``{name: set of (channels, photometries)}``, from
            :func:`~autowisp.diagnostics.expressions.get_needed_values`,
            which has already resolved slots, quoted channels and library
            expressions alike, so a channel is here whichever way it is read.
    """

    magfit = magfit_diagnostic_names()
    return sorted(
        {
            channel
            for name, bindings in needed.items()
            if name in magfit
            for channels, _ in bindings
            for channel in channels
        }
    )


def _unreferenced_magfit_channels(series_key, needed):
    """Return the channels *needed* reads magfit in with no reference."""

    referenced = {channel for channel, _ in series_key.reference_pairs}
    return [
        channel
        for channel in _magfit_channels(needed)
        if channel not in referenced
    ]


def _check_references(series_key, needed):
    """
    Refuse to read magfit diagnostics over images fit against unknown refs.

    A magfit diagnostic depends on the reference an image was fit against
    as well as on the image, so read in a channel the key names no
    reference for, the population could mix images fit against several,
    and an aggregate over it -- or a jump in a plot of it -- would mean
    nothing. A reference on a channel nothing magfit reads is fine: it
    merely restricts the population.

    Raises:
        PipelineError:    If a channel *needed* reads a magfit diagnostic in
            has no reference in *series_key*.
    """

    unreferenced = _unreferenced_magfit_channels(series_key, needed)
    if unreferenced:
        raise PipelineError(
            "Magnitude fitting diagnostics are read in "
            + ", ".join(unreferenced)
            + ", but the series names no photometric reference there: "
            "their values depend on the reference each image was fit "
            "against, so the series would mix references.",
            details={
                "channels": unreferenced,
                "series": series_key._asdict(),
            },
        )


def _read_diagnostics(population, names, bindings, db_session):
    """
    Return *names* in each of *bindings* for a population, NaN-padded.

    The reading :func:`get_diagnostic_values` describes, for any
    population: one query, read back as a rectangle.

    Args:
        population:    Function restricting a query over ``Image`` to the
            images to read, as :func:`_in_series` does to a series'.

        names(list):    The ``diagnostic_type`` names to read. May be
            empty, leaving only the images to list.

        bindings(list):    The diagnostic bindings to read every one of
            them in, as :func:`_diagnostic_values_query` takes them.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{name: {binding: array}}`` for every name and
                binding asked for, over the images in canonical order.

            numpy.ndarray:    The image ids, in canonical order.

            numpy.ndarray:    Their Julian dates.
    """

    if not names:
        return {}, *_list_images(population, db_session)

    rows = db_session.execute(
        _diagnostic_values_query(population, names, bindings)
    ).all()

    # From the rows rather than from names: a name no diagnostic_type has
    # contributes no block at all, and would otherwise throw the shape out
    # for every other name. In the order the rows bring them, which the
    # ordering by name makes one contiguous block each, so nothing depends
    # on the database's collation ordering strings the way Python does.
    read_names = list(dict.fromkeys(row[2] for row in rows))
    per_block = len(rows) // len(read_names) if read_names else 0

    columns = {
        binding: numpy.fromiter(
            (
                numpy.nan if row[3 + offset] is None else row[3 + offset]
                for row in rows
            ),
            dtype=float,
            count=len(rows),
        ).reshape(len(read_names), per_block)
        for offset, binding in enumerate(bindings)
    }

    return (
        {
            name: {
                binding: (
                    columns[binding][read_names.index(name)]
                    if name in read_names
                    else numpy.full(per_block, numpy.nan)
                )
                for binding in bindings
            }
            for name in names
        },
        *_as_arrays(rows[:per_block]),
    )


def get_diagnostic_values(series_key, needed, db_session):
    """
    Return the wanted diagnostics for one series, NaN-padded and aligned.

    One query, and nothing to match up afterwards. A cross join pairs every
    wanted diagnostic with every image of the series, and an outer join
    attaches the values, leaving ``NULL`` where nothing was recorded -- so
    the padding is what the database returns rather than something assembled
    from it. The unique indices make that sound: on ``(image_id, channel,
    diagnostic_id)`` for a diagnostic recorded per image, and on
    ``(image_id, channel, photometry_id, diagnostic_id)`` for one recorded
    per photometry. No image contributes two rows for one diagnostic in one
    binding, so the result is exactly one row per image per name.

    **Several bindings at once, still one query.** An expression comparing
    channels or photometries needs the same diagnostic read more than once,
    so there is one outer join per distinct binding, each with its channel,
    and its photometry where it has one, pinned. That keeps the result a
    rectangle -- the joins only widen it, adding a value column per binding
    rather than rows -- and keeps every probe on a unique index, whose
    columns are exactly what is being pinned.

    Being a rectangle is what lets the values become arrays in one step: a
    column is read out whole and reshaped into one row per name, rather
    than accumulated name by name. The blocks' names are taken from the
    rows, in the order they arrive, rather than from a sorted list of the
    names asked for, so nothing depends on the database's collation
    ordering strings the way Python does.

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

        needed(dict):    ``{name: set of (channels, photometries)}``, from
            :func:`~autowisp.diagnostics.expressions.get_needed_values`.
            May include :data:`time_quantity` at ``((), ())``, which is
            taken from the image row rather than from a diagnostics table.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{name: {(channels, photometries): array}}``, keyed
                exactly as *needed* asked, every array the length of the
                canonical image list and ``NaN`` where nothing is recorded.
                A name no ``diagnostic_type`` has is all ``NaN``.

            numpy.ndarray:    The image ids, in canonical order.

    Raises:
        PipelineError:    If *needed* reads a magfit diagnostic in a
            channel *series_key* names no reference for.
    """

    _check_references(series_key, needed)

    names = sorted(set(needed) - {time_quantity})
    by_binding, image_ids, jd_values = _read_diagnostics(
        lambda query: _in_series(query, series_key),
        names,
        sorted({binding for name in names for binding in needed[name]}),
        db_session,
    )

    values = {
        name: {binding: by_binding[name][binding] for binding in needed[name]}
        for name in names
    }
    if time_quantity in needed:
        values[time_quantity] = {((), ()): jd_values}

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

        wanted(dict):    ``{quantity: set of (channels, photometries)}``,
            each holding one channel per channel parameter of that quantity
            and one photometry id per photometry parameter. A set of them
            because the two axes may be one quantity read in two channels,
            which is how a diagnostic is compared between them.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{quantity: {(channels, photometries): array}}``, all
                of the same
                length, and unmasked -- dropping the non-finite entries is
                the caller's business, since the mask has to be taken
                across both axes at once and the image ids masked with it.

            numpy.ndarray:    The image ids that length runs over.

    Raises:
        PipelineError:    If a quantity names nothing, if the expressions
            reference each other in a cycle, if a binding is the wrong
            length for what it binds, or if a magfit diagnostic is read in
            a channel *series_key* names no reference for.
    """

    values, image_ids = get_diagnostic_values(
        series_key, get_needed_values(wanted, expressions), db_session
    )

    return evaluate_quantities(wanted, expressions, values), image_ids


def split_series(series_key, wanted, expressions, db_session):
    """
    Return the fit populations of a series: one key per reference combination.

    A session may have been fit against several photometric references --
    split by the separation limit, say -- and a quantity reading a magfit
    diagnostic means something only within one of them. This finds which
    references the series' images were actually bound to, in every channel
    *wanted* reads a magfit diagnostic in, and returns one key per
    combination that occurs, each restricted to the images having it.

    If the first 200 images of a session were fit against A and the rest
    against B, ``photometry_mag_offset[0]`` bound to ``R`` splits into keys
    with ``photrefs=(A_R,)`` and ``(B_R,)``. Read in ``R`` and ``B`` it
    normally splits into ``(A_R, A_B)`` and ``(B_R, B_B)``: only
    combinations some image has, never every pairing of the references
    found per channel.

    References the key already names are kept, so the split happens within
    the images bound to them. Each slot on a split channel gets that
    channel's reference.

    An image bound in none of those channels -- only possible from
    processing before every magfit-ed image was bound -- is in none of the
    keys; :func:`count_unbound_images` says how many there are.

    Args:
        series_key(SeriesKey):    The series to split. It needs a slot for
            every channel a magfit diagnostic is read in, those read only
            in quoted channels included, since that is where the reference
            goes.

        wanted(dict):    ``{quantity: set of (channels, photometries)}``, as for
            :func:`get_quantity_values`.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    :class:`SeriesKey` per combination present, ordered by the
            references. *series_key* alone where there is nothing to split
            by.

    Raises:
        PipelineError:    If a channel *wanted* reads a magfit diagnostic in
            has no slot in *series_key*, or as
            :func:`~autowisp.diagnostics.expressions.get_needed_values`
            does.
    """

    needed = get_needed_values(wanted, expressions)
    slotless = [
        channel
        for channel in _magfit_channels(needed)
        if channel not in series_key.channels
    ]
    if slotless:
        raise PipelineError(
            "Magnitude fitting diagnostics are read in "
            + ", ".join(slotless)
            + ", which the series has no slot for to hold a photometric "
            "reference: a channel read only by quoting it needs one after "
            "the parameters' slots.",
            details={"channels": slotless, "series": series_key._asdict()},
        )

    split_by = _unreferenced_magfit_channels(series_key, needed)
    if not split_by:
        return [series_key]

    bindings = {channel: aliased(ImageMasterSelection) for channel in split_by}
    photref_columns = [bindings[channel].master_file_id for channel in split_by]
    query = _in_series(select(*photref_columns).select_from(Image), series_key)
    for channel in split_by:
        query = query.join(
            bindings[channel], photref_binding(bindings[channel], channel)
        )

    result = []
    for combination in db_session.execute(
        query.distinct().order_by(*photref_columns)
    ).all():
        found = dict(zip(split_by, combination))
        # pylint: disable=no-member
        result.append(
            series_key._replace(
                photrefs=tuple(
                    found.get(channel, photref)
                    for channel, photref in zip(
                        series_key.channels, series_key.photrefs
                    )
                )
            )
        )
        # pylint: enable=no-member
    return result


def count_unbound_images(series_key, wanted, expressions, db_session):
    """
    Count the images :func:`split_series` leaves out for want of a binding.

    Those with a magfit diagnostic recorded in a channel *wanted* reads one
    in and *series_key* names no reference for, but no photref bound there:
    magfit-ed, so their values depend on a reference, with nothing saying
    which. Only processing from before every magfit-ed image was bound
    leaves any. Counted rather than listed, for the engine to log and the
    browser interface to show.

    Args:
        series_key(SeriesKey):    The series, as given to
            :func:`split_series`.

        wanted(dict):    ``{quantity: set of (channels, photometries)}``.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        int:    The number of images, each counted once however many
            channels it lacks a binding in.
    """

    split_by = _unreferenced_magfit_channels(
        series_key, get_needed_values(wanted, expressions)
    )
    if not split_by:
        return 0

    # A magfit diagnostic is recorded per image or per photometry, depending
    # on which it is; a row in either says the image was magfit-ed in that
    # channel.
    recorded = union_all(
        *(
            select(table.image_id, table.channel, table.diagnostic_id)
            for table in (ImageDiagnostics, PhotometryDiagnostics)
        )
    ).subquery()
    binding = aliased(ImageMasterSelection)
    return db_session.scalar(
        _in_series(
            select(
                func.count(Image.id.distinct())  # pylint: disable=not-callable
            ).select_from(Image),
            series_key,
        )
        .join(
            recorded,
            (recorded.c.image_id == Image.id)  # pylint: disable=no-member
            & recorded.c.channel.in_(split_by),
        )
        .join(
            DiagnosticType,
            (DiagnosticType.id == recorded.c.diagnostic_id)
            & DiagnosticType.name.in_(magfit_diagnostic_names()),
        )
        .join(
            binding,
            photref_binding(binding, recorded.c.channel),
            isouter=True,
        )
        .where(binding.image_id.is_(None))
    )


#: The channel a member's slot is bound to while its quantities are resolved:
#: not a channel, but a stand-in for each member's own, which the values are
#: then read in. The angle brackets keep it from ever naming a real one.
_own_channel = "<own channel>"


def _custom_group_bindings(quantities, expressions):
    """
    Return ``wanted`` for :func:`get_custom_group_values`.

    Every slot is bound to :data:`_own_channel`, the stand-in for each
    member's own channel.

    Raises:
        PipelineError:    If a quantity takes more than one channel, or any
            photometry.
    """

    wanted = {}
    for quantity in quantities:
        arity = get_channel_arity(quantity, expressions)
        if arity > 1:
            raise PipelineError(
                f"{quantity} takes {arity} channels, but a photref group "
                "binds only one: each member's own.",
                details={"quantity": quantity, "channels": arity},
            )
        if get_photometry_arity(quantity, expressions):
            raise PipelineError(
                f"{quantity} is read per photometry, but a photref group has "
                "not been magnitude-fit, so it has no photometry to bind.",
                details={"quantity": quantity},
            )
        wanted[quantity] = {((_own_channel,) * arity, ())}
    return wanted


def _read_custom_group(members, needed, db_session):
    """
    Return the values *needed* asks for, one entry per member.

    The members' images are read as any population is, in every binding
    *needed* asks for, one on :data:`_own_channel` in every channel a member
    has; each member then takes the column of its own channel wherever
    *needed* asks for :data:`_own_channel`.

    Returns:
        tuple:
            dict:    ``{name: {(channels, photometries): array}}``, as
                :func:`get_diagnostic_values` returns for a series.

            list:    The members the arrays run over, in canonical order.
    """

    # Tuples, since entries kept in a Django session come back as lists.
    members = set(map(tuple, members))
    names = sorted(set(needed) - {time_quantity})
    image_ids = sorted({image_id for image_id, _ in members})
    member_channels = {channel for _, channel in members}

    # Each name a diagnostic, so each binding holds exactly one channel.
    by_binding, read_ids, jd_values = _read_diagnostics(
        lambda query: query.where(
            Image.id.in_(image_ids),  # pylint: disable=no-member
            Image.jd.is_not(None),  # pylint: disable=no-member
        ),
        names,
        sorted(
            {
                ((channel_read,), photometries)
                for name in names
                for (channel,), photometries in needed[name]
                for channel_read in (
                    member_channels if channel == _own_channel else (channel,)
                )
            }
        ),
        db_session,
    )

    position = {image_id: index for index, image_id in enumerate(read_ids)}
    members = sorted(
        (member for member in members if member[0] in position),
        key=lambda member: (position[member[0]], member[1]),
    )
    rows = numpy.array([position[image_id] for image_id, _ in members], int)
    own = numpy.array([channel for _, channel in members])

    def read(name, binding):
        """Return *name* in *binding* per member, in each one's own channel
        where it is bound to :data:`_own_channel`."""

        (channel,), photometries = binding
        if channel != _own_channel:
            return by_binding[name][binding][rows]
        result = numpy.empty(len(members))
        for channel_read in set(own):
            mine = own == channel_read
            result[mine] = by_binding[name][(channel_read,), photometries][
                rows[mine]
            ]
        return result

    values = {
        name: {binding: read(name, binding) for binding in needed[name]}
        for name in names
    }
    if time_quantity in needed:
        values[time_quantity] = {((), ()): jd_values[rows]}

    return values, members


def get_custom_group_values(members, quantities, expressions, db_session):
    """
    Return quantities over an explicit list of members, one point each.

    The photref-group scope. A photref group -- images sharing configuration
    conditions and ``single_photref`` ``must_match`` values, possibly across
    sessions -- is found only by the engine's condition grouping, so it
    arrives as its members rather than as a key. A member is an image in one
    channel: under a ``must_match`` such as ``CLRCHNL[0].upper()``, one
    image is a member in ``G0`` and again in ``G1``.

    Each quantity takes at most one channel, bound per member to that
    member's own, so ``nanrank(s_center[0])`` ranks every member's
    ``s_center`` in its own channel against all the others'. A quoted
    channel is read in that channel for every member. No reference is
    required, as it is of a series: the group is being chosen one.

    Args:
        members:    ``(image_id, channel)`` pairs, in any order.

        quantities:    The names of the diagnostics and expressions to
            evaluate, each taking at most one channel.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{quantity: array}``, one entry per member.

            list:    The members the arrays run over, as ``(image_id,
                channel)`` tuples ordered by Julian date, then image id,
                then channel. A member whose image has no Julian date is
                left out, as it is from every series.

    Raises:
        PipelineError:    If a quantity takes more than one channel, or as
            :func:`get_quantity_values` does.
    """

    wanted = _custom_group_bindings(quantities, expressions)
    values, members = _read_custom_group(
        members, get_needed_values(wanted, expressions), db_session
    )

    return (
        {
            quantity: next(iter(by_binding.values()))
            for quantity, by_binding in evaluate_quantities(
                wanted, expressions, values
            ).items()
        },
        members,
    )
