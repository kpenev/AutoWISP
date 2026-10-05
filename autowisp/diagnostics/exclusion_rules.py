"""Which observations an exclusion rule leaves out of a fit.

An exclusion rule is a boolean expression over the diagnostics and the
project's expression library, true for what a fit should be derived without
(see :func:`~autowisp.diagnostics.expressions.check_rule`). The pipeline
engine evaluates the step's rule here just before running ``fit_magnitudes``,
``epd`` or ``tfa``, and hands the step the result as its exclusion list.
Nothing is stored: the rule is evaluated again whenever a step is prepared.

The observations decided for are those **fit together**, given as ``(image,
channel)`` pairs: a magnitude fitting batch, or the lightcurve points of one
single photometric reference. The rule is not evaluated over that group,
though, but over the populations an expression is always evaluated over, one
session's frames of one type, split by photometric reference where the rule
reads something ``fit_magnitudes`` produces. An aggregate in a rule therefore
means what it does in a plot of the same expression, and the group merely
selects which verdicts are used.

What a verdict applies to follows from what the rule takes:

* a channel slot: the channel it was bound to. Without one the rule reads
  only quoted channels, and its verdict applies to every channel of the image;
* a photometry slot: the photometry it was bound to. Without one the verdict
  applies to every photometry.
"""

import logging

from sqlalchemy import select

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Image,
    ImageType,
    ObservingSession,
    PhotometryDiagnostics,
)
from autowisp.database.data_model.provenance import Camera, CameraChannel

# pylint: enable=no-name-in-module
from autowisp.diagnostics.diagnostic_types import (
    magfit_diagnostic_names,
    photometry_literal,
)
from autowisp.diagnostics.expression_library import get_expressions
from autowisp.diagnostics.expression_series import (
    SeriesKey,
    count_unbound_images,
    get_magfit_channels,
    get_quantity_values,
    split_series,
)
from autowisp.diagnostics.expressions import (
    check_rule,
    get_channel_arity,
    get_channel_parameters,
    get_logical_keywords,
    get_needed_values,
    get_photometry_arity,
    get_photometry_parameters,
    rule_quantity,
)
from autowisp.exceptions import ConfigurationError, PipelineError

_logger = logging.getLogger(__name__)


def _get_populations(members, db_session):
    """
    Split the given members by (session, image type) ready for evaluation.

    Args:
        members:    ``(image_id, channel)`` pairs.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``{(session_id, image_type): {image_id: [channel, ...]}}``.
    """

    member_channels = {}
    for image_id, channel in members:
        member_channels.setdefault(image_id, []).append(channel)

    result = {}
    # pylint: disable=no-member
    for image_id, session_id, image_type in db_session.execute(
        select(Image.id, Image.observing_session_id, ImageType.name)
        .join(ImageType, ImageType.id == Image.image_type_id)
        .where(Image.id.in_(sorted(member_channels)))
    ).all():
        # pylint: enable=no-member
        result.setdefault((session_id, image_type), {})[image_id] = (
            member_channels[image_id]
        )
    return result


def _get_camera_channels(session_id, db_session):
    """Return the names of the channels the camera of a session has."""

    return set(
        db_session.scalars(
            select(CameraChannel.name)
            .join(Camera, Camera.camera_type_id == CameraChannel.camera_type_id)
            .join(ObservingSession, ObservingSession.camera_id == Camera.id)
            .where(ObservingSession.id == session_id)
        ).all()
    )


def _get_recorded_photometries(session_id, image_type, db_session):
    """Return the photometry IDs with photometry diagnostics for population."""

    # pylint: disable=no-member
    return set(
        db_session.scalars(
            select(PhotometryDiagnostics.photometry_id)
            .join(Image, Image.id == PhotometryDiagnostics.image_id)
            .join(ImageType, ImageType.id == Image.image_type_id)
            .where(
                Image.observing_session_id == session_id,
                ImageType.name == image_type,
            )
            .distinct()
        ).all()
    )
    # pylint: enable=no-member


def _check_readable(rule, needed, camera_channels, recorded_photometries):
    """
    Refuse a rule reading a channel or a photometry its population lacks.

    Values not recorded are padded with NaN, which compares false, so a rule
    reading a channel the camera does not have, or a photometry that was
    never extracted, would otherwise keep every image without saying so.

    Args:
        rule(str):    The rule, for the message.

        needed(dict):    ``{name: set of (channels, photometries)}``: what
            the rule reads, quoted or bound, itself or through the library.

        camera_channels(set):    The channels the population's camera has.

        recorded_photometries(set):    The photometry ids recorded for the
            population.

    Raises:
        ConfigurationError:    Naming what is missing.
    """

    read_channels = {
        channel
        for bindings in needed.values()
        for channel_list, _ in bindings
        for channel in channel_list
    }
    read_photometries = {
        photometry
        for bindings in needed.values()
        for _, photometry_list in bindings
        for photometry in photometry_list
    }

    missing = [
        f"channel {channel!r}"
        for channel in sorted(read_channels - camera_channels)
    ] + [
        f"photometry {photometry_literal(photometry)!r}"
        for photometry in sorted(read_photometries - recorded_photometries)
    ]
    if missing:
        raise ConfigurationError(
            f"The exclusion rule {rule!r} reads "
            + ", ".join(missing)
            + ", which the images it is applied to do not have: channels "
            + ", ".join(map(repr, sorted(camera_channels)))
            + " and photometries "
            + ", ".join(
                repr(photometry_literal(photometry))
                for photometry in sorted(recorded_photometries)
            )
            + ". Every image would be kept. Give this equipment a rule of "
            "its own through a condition on the parameter.",
            details={"rule": rule, "missing": missing},
        )


def _evaluate(series_key, wanted, library, db_session):
    """
    Return the verdicts of the rule in *library* over one population.

    Args:
        series_key(SeriesKey):    The population to evaluate over.

        wanted(dict):    ``{rule_quantity: set of (channels, photometries)}``.

        library(dict):    The project's library with the rule added.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            dict:    ``{(channels, photometries): bool array}``, true where
                the rule excludes.

            numpy.ndarray:    The image ids the arrays run over.

    Raises:
        ConfigurationError:    If the rule does not give true or false.
    """

    rule = library[rule_quantity]
    try:
        values, image_ids = get_quantity_values(
            series_key, wanted, library, db_session
        )
    except Exception:
        keywords = sorted(get_logical_keywords(rule))
        if keywords:
            _logger.error(
                "The exclusion rule %r uses %s, which fail on arrays: "
                "element-wise logic is written |, & and ~.",
                rule,
                ", ".join(keywords),
            )
        raise

    verdicts = values[rule_quantity]
    for verdict in verdicts.values():
        if verdict.dtype != bool:
            raise ConfigurationError(
                f"The exclusion rule {rule!r} gives {verdict.dtype} values "
                "where it has to say true or false for each image: compare "
                "what it calculates to a limit.",
                details={"rule": rule},
            )
    return verdicts, image_ids


# Everything the loop over channels and populations needs to share.
# pylint: disable-next=too-many-locals
def _exclude_in_population(
    library, population, member_channels, db_session, *, excluded
):
    """
    Add to *excluded* the members of one population that the rule excludes.

    Args:
        library(dict):    The project's library with the rule added.

        population(tuple):    The session id and the name of the image type.

        member_channels(dict):    ``{image_id: [channel, ...]}``: the
            observations of the population being decided for.

        db_session:    An active SQLAlchemy database session.

        excluded(dict):    ``{photometry: set of (image_id, channel)}``, as
            :func:`get_excluded` returns, updated in place.
    """

    rule = library[rule_quantity]
    per_channel = bool(get_channel_parameters(rule))
    per_photometry = bool(get_photometry_parameters(rule))

    camera_channels = _get_camera_channels(population[0], db_session)
    recorded_photometries = _get_recorded_photometries(*population, db_session)

    for channel in (
        sorted({name for names in member_channels.values() for name in names})
        if per_channel
        else [None]
    ):
        slot_channels = (channel,) if per_channel else ()
        wanted = {
            rule_quantity: {
                (slot_channels, (photometry,) if per_photometry else ())
                for photometry in (
                    recorded_photometries if per_photometry else [None]
                )
            }
        }
        needed = get_needed_values(wanted, library)
        _check_readable(rule, needed, camera_channels, recorded_photometries)

        # A channel magfit is read in needs a slot to hold its reference,
        # which a quoted one gets after those the rule binds.
        series_key = SeriesKey(
            *population,
            slot_channels
            + tuple(
                read_channel
                for read_channel in get_magfit_channels(needed)
                if read_channel not in slot_channels
            ),
        )
        num_unbound = count_unbound_images(
            series_key, wanted, library, db_session
        )
        if num_unbound:
            _logger.warning(
                "The exclusion rule %r decides nothing for %d images of "
                "session %d, which were magnitude fit but are bound to no "
                "photometric reference in a channel it reads the fit in.",
                rule,
                num_unbound,
                population[0],
            )

        for reference_key in split_series(
            series_key, wanted, library, db_session
        ):
            verdicts, image_ids = _evaluate(
                reference_key, wanted, library, db_session
            )
            for (_, photometries), verdict in verdicts.items():
                excluded.setdefault(
                    photometries[0] if per_photometry else None, set()
                ).update(
                    (image_id, member_channel)
                    for image_id in image_ids[verdict].tolist()
                    for member_channel in member_channels.get(image_id, ())
                    if not per_channel or member_channel == channel
                )


#: More than this fraction of a fit left out is conspicuous: the engine logs
#: it as a warning, and the BUI highlights it.
conspicuous_fraction = 0.5


def summarize_excluded(excluded, num_members):
    """
    Return how much of what is fit together each verdict of a rule excludes.

    Args:
        excluded(dict):    What :func:`get_excluded` returned.

        num_members(int):    How many observations are fit together.

    Returns:
        list:    A dict per photometry, in order of id: ``photometry``, its
            id, None for a rule deciding for every photometry at once;
            ``excluded``, how many observations it excludes; ``fraction``,
            what fraction that is of those fit together; and
            ``conspicuous``, whether it is above
            :data:`conspicuous_fraction`.
    """

    result = []
    for photometry, observations in sorted(excluded.items()):
        fraction = len(observations) / num_members if num_members else 0.0
        result.append(
            {
                "photometry": photometry,
                "excluded": len(observations),
                "fraction": fraction,
                "conspicuous": fraction > conspicuous_fraction,
            }
        )
    return result


def _report_fractions(rule, excluded, num_members):
    """Log how much of what is fit together *rule* excludes."""

    for verdict in summarize_excluded(excluded, num_members):
        _logger.log(
            logging.WARNING if verdict["conspicuous"] else logging.INFO,
            "The exclusion rule %r excludes %.0f%% (%d of %d) of the "
            "observations fit together%s.",
            rule,
            100 * verdict["fraction"],
            verdict["excluded"],
            num_members,
            (
                ""
                if verdict["photometry"] is None
                else " in photometry "
                + photometry_literal(verdict["photometry"])
            ),
        )


def get_excluded(
    rule, members, db_session, *, before_magfit=False, report=True
):
    """
    Return the observations fit together that *rule* leaves out of the fit.

    Args:
        rule(str):    The exclusion rule.

        members:    ``(image_id, channel)`` pairs: the observations fit
            together, which are the ones decided for.

        db_session:    An active SQLAlchemy database session.

        before_magfit(bool):    Whether the fit is the magnitude fit itself,
            which nothing it produces can be read ahead of.

        report(bool):    Whether to log the fraction excluded, as the engine
            does for every fit. A preview shows it instead.

    Returns:
        dict:    ``{photometry: set of (image_id, channel)}``: the excluded
            members. Under the single key ``None`` if the rule takes no
            photometry slot, its verdict then applying to every photometry,
            and otherwise under each photometry id recorded for the members,
            those nothing is excluded in included.

    Raises:
        ConfigurationError:    If the rule is malformed, reads a channel or a
            photometry the images do not have, reads what magnitude fitting
            produces when *before_magfit*, or does not give true or false.
    """

    members = set(map(tuple, members))
    library = get_expressions(db_session)
    problems = check_rule(rule, library)
    if problems:
        raise ConfigurationError(
            f"The exclusion rule {rule!r} cannot be used. "
            + " ".join(problems),
            details={"rule": rule, "problems": problems},
        )
    library[rule_quantity] = rule

    per_photometry = bool(get_photometry_parameters(rule))
    if before_magfit:
        # Resolved with stand-ins for the slots: what is read does not
        # depend on what they are bound to.
        read_magfit = sorted(
            magfit_diagnostic_names()
            & set(
                get_needed_values(
                    {
                        rule_quantity: {
                            (
                                ("",) * len(get_channel_parameters(rule)),
                                (0,) * per_photometry,
                            )
                        }
                    },
                    library,
                )
            )
        )
        if read_magfit:
            raise ConfigurationError(
                f"The exclusion rule {rule!r} for magnitude fitting reads "
                + ", ".join(read_magfit)
                + ", which magnitude fitting itself produces, so there is "
                "nothing to read yet.",
                details={"rule": rule, "diagnostics": read_magfit},
            )

    excluded = {} if per_photometry else {None: set()}
    for population, member_channels in sorted(
        _get_populations(members, db_session).items()
    ):
        _exclude_in_population(
            library, population, member_channels, db_session, excluded=excluded
        )

    if report:
        _report_fractions(rule, excluded, len(members))
    return excluded


def get_rule_reads(library):
    """
    Return the library expressions usable as exclusion rules, as configured.

    A rule names a library expression by a read binding the slots it takes,
    the channel slot to the channel being decided for and the photometry
    slot to the photometry: ``cloudy``, ``cloudy[0]``, ``cloudy[()][0]`` or
    ``cloudy[0][0]``. Whether an expression gives true or false is not
    known until it is evaluated, so one giving numbers is listed too.

    Args:
        library(dict):    The project's library, ``{name: expression}``.

    Returns:
        dict:    ``{name: read}`` for each expression :func:`check_rule`
            accepts read that way.
    """

    reads = {}
    for name in library:
        try:
            num_channels = get_channel_arity(name, library)
            num_photometries = get_photometry_arity(name, library)
        except SyntaxError, PipelineError:
            # Broken, which the library page says; not a rule either way.
            continue
        if num_channels > 1 or num_photometries > 1:
            continue
        read = name
        if num_channels or num_photometries:
            read += "[0]" if num_channels else "[()]"
        if num_photometries:
            read += "[0]"
        if not check_rule(read, library):
            reads[name] = read
    return reads


def preview_excluded(rule, image_ids, db_session, *, channel, photometry):
    """
    Return the images *rule* excludes, as the engine would decide for them.

    Each image is decided for in the given channel and photometry, which
    the rule's slots are bound to, and within the population the engine
    evaluates it over, so that what a preview shows is what a fit would
    leave out. Nothing is refused for reading what magnitude fitting
    produces: that only matters for the rule of the magnitude fit itself,
    which the engine checks when it runs it.

    Args:
        rule(str):    The exclusion rule.

        image_ids(iterable):    The images to decide for.

        db_session:    An active SQLAlchemy database session.

        channel(str or None):    The channel the rule's channel slot is
            bound to, None for a rule without one.

        photometry(int or None):    The photometry id the rule's photometry
            slot is bound to, None for a rule without one.

    Returns:
        set:    The ids of the images excluded.

    Raises:
        ConfigurationError:    As :func:`get_excluded` does, or if the rule
            takes a slot nothing is bound to.
    """

    per_photometry = bool(get_photometry_parameters(rule))
    for kind, slots, bound in [
        ("channel", get_channel_parameters(rule), channel),
        ("photometry", per_photometry, photometry),
    ]:
        if slots and bound is None:
            raise ConfigurationError(
                f"The exclusion rule {rule!r} decides per {kind}: choose the "
                f"{kind} to decide for.",
                details={"rule": rule},
            )

    excluded = get_excluded(
        rule,
        [(image_id, channel) for image_id in image_ids],
        db_session,
        report=False,
    )
    return {
        image_id
        for image_id, _ in excluded.get(
            photometry if per_photometry else None, ()
        )
    }
