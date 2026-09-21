"""Which quantities this project can draw, and what each one is.

An axis, a row and a section all name a *quantity*: a recorded
``DiagnosticType``, an expression built on top of one, or ``jd``. What
exists to be named is a question about the open project rather than about
any row, and it is asked once per page rather than once per row, which is
why it is answered here and not in :mod:`series_table` -- that module is
about what a row of the table means.

Nothing here evaluates a quantity or knows what one is worth: reading
values belongs to :mod:`autowisp.diagnostics.expression_series`, and what
an expression *means* to :mod:`autowisp.diagnostics.expressions`. What is
asked here is only which names are on offer.
"""

from sqlalchemy import select

from autowisp.diagnostics.diagnostic_types import (
    is_quantile_diagnostic,
    quantiles_quantity,
)
from autowisp.diagnostics.expression_series import time_quantity
from autowisp.diagnostics.expressions import order_expressions
from autowisp.exceptions import PipelineError

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import DiagnosticType, ImageDiagnostics

# pylint: enable=no-name-in-module


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


def get_diagnostic_descriptions(db_session):
    """
    Return ``{name: description}`` for every recorded diagnostic type.

    One query for the whole table rather than one per section: a page may
    show any number of sections, and the table is small enough that asking
    for all of it costs less than asking repeatedly.

    Args:
        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    The descriptions, with the empty string where a type
            records none.
    """

    return {
        name: description or ""
        for name, description in db_session.execute(
            select(DiagnosticType.name, DiagnosticType.description)
        ).all()
    }


def describe_quantity(name, expressions, descriptions):
    """
    Return what a section header says about the quantity it draws.

    A collapsed section still has to say what is plotted in it, so that
    the answer can be read rather than guessed from a name -- which for an
    expression is whatever its author called it.

    Args:
        name(str):    The quantity, as the selectors and the URL name it.

        expressions(dict):    The library, ``{name: expression}``.

        descriptions(dict):    What each name means: the recorded
            diagnostics' descriptions and the expressions' together, as
            :func:`get_diagnostic_descriptions` and
            ``expression_data.get_expression_descriptions`` return them.

    Returns:
        dict:    ``name``, ``expression`` and ``description``. The
            expression is empty for anything not built from one -- a
            recorded diagnostic is a measurement rather than a formula,
            and has none to show.
    """

    return {
        "name": name,
        "expression": expressions.get(name, ""),
        "description": descriptions.get(
            name,
            "Julian date of the exposure." if name == time_quantity else "",
        ),
    }


#: The markers a section's rows start with, in the order sections take
#: them.  Points first and in falling order of how readily one is told
#: from another at a glance, so that the commonest case -- two or three
#: sections -- gets the clearest pairs.
section_markers = "os^v<>x+"


def next_section_marker(taken):
    """
    Return the marker a new section's rows should start with.

    Telling quantities apart by shape and channels apart by colour is only
    a starting point: every row's marker stays editable, so a user who
    would rather tell them apart some other way sets it. What this decides
    is what a section looks like before anyone touches it.

    The first marker no section is using, so that one freed by a removed
    section is taken up again rather than left idle while a later section
    doubles up on one still in use. Once every marker is spoken for, they
    cycle by the number of sections -- the ninth section repeating the
    first marker, the tenth the second -- rather than piling every further
    section onto the same one.

    Args:
        taken(iterable):    The markers the sections already on the page
            start with, one per section. Counted rather than deduplicated,
            since the count is what the cycle turns on.

    Returns:
        str:    The marker to start with.
    """

    taken = list(taken)
    for marker in section_markers:
        if marker not in taken:
            return marker

    return section_markers[len(taken) % len(section_markers)]


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
