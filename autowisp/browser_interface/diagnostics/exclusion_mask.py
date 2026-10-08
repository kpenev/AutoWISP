"""The exclusion mask the diagnostics plot can apply to every row.

Any library expression usable as an exclusion rule can be chosen in the
footer of the plot page, with the channel and photometry its slots are
bound to. The points it would exclude are then drawn faintly, and each row
says how many there are, so that a rule can be judged against the very
diagnostics it reads before a step is run with it. It is decided by the
engine's own machinery, so a preview shows what a fit would leave out.
"""

import numpy

from autowisp.database.user_interface import list_channels
from autowisp.diagnostics.diagnostic_types import photometry_literal
from autowisp.diagnostics.exclusion_rules import (
    get_rule_reads,
    preview_excluded,
)
from autowisp.diagnostics.expressions import (
    get_channel_parameters,
    get_logical_keywords,
    get_photometry_parameters,
)
from autowisp.exceptions import ConfigurationError

from .quantities import get_recorded_photometries


def get_mask_options(expressions, db_session):
    """
    Return what the footer of the plot page offers to mask with.

    Args:
        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``exclusion_masks``, each expression usable as a rule with
            the read it is configured with and which slots that read binds;
            ``mask_channels``, every channel some camera has; and
            ``mask_photometries``, each recorded photometry with its label.
    """

    return {
        "exclusion_masks": [
            {
                "read": read,
                "channel_slot": bool(get_channel_parameters(read)),
                "photometry_slot": bool(get_photometry_parameters(read)),
            }
            for _, read in sorted(get_rule_reads(expressions).items())
        ],
        "mask_channels": sorted(list_channels(db_session)),
        "mask_photometries": [
            {"id": photometry, "label": photometry_literal(photometry)}
            for photometry in get_recorded_photometries(db_session)
        ],
    }


def decide_mask(mask, image_ids, drawn, *, expressions, db_session):
    """
    Return what an exclusion mask excludes of one row's drawn points.

    The images are decided for in the channel and photometry the mask
    names, not in any the row binds: a row may bind several channels, and
    none of them need be the one the rule was meant for.

    Args:
        mask(dict):    The ``rule``, as configured, and the ``channel`` and
            ``photometry`` its slots are bound to, as the page posts them.

        image_ids(numpy.ndarray):    The images the row's values run over.

        drawn(numpy.ndarray):    Which of them are drawn, the only ones
            decided for.

        expressions(dict):    The library, ``{name: expression}``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        numpy.ndarray:    Per image, whether it is drawn and excluded; none
            where nothing could be decided.

        dict:    What the row's Excluded cell says: ``text``, how many of
            its points are excluded or briefly why none could be decided
            for, and ``title``, shown on hovering over it, the whole of the
            why.
    """

    nothing = numpy.zeros(image_ids.shape, dtype=bool)
    photometry = mask.get("photometry")
    try:
        excluded = preview_excluded(
            mask["rule"],
            image_ids[drawn].tolist(),
            db_session,
            channel=mask.get("channel") or None,
            photometry=None if photometry in (None, "") else int(photometry),
        )
    except ConfigurationError as error:
        # What the images lack is the usual refusal, and short enough to
        # say in the cell; anything else only fits the hover text.
        missing = error.details.get("missing")
        return nothing, {
            "text": "no " + ", ".join(missing) if missing else "refused",
            "title": str(error),
        }
    except (ValueError, TypeError) as error:
        # Raised evaluating the rule, and the usual cause is a logical
        # keyword, which numpy's message does not name.
        title = f"{mask['rule']} failed: {error}"
        keywords = sorted(
            get_logical_keywords(
                expressions.get(mask["rule"].split("[", 1)[0], mask["rule"])
            )
        )
        if keywords:
            title += (
                f" It uses {', '.join(keywords)}, which fail on arrays: "
                "write |, & and ~."
            )
        return nothing, {"text": "failed", "title": title}

    return (
        numpy.isin(image_ids, sorted(excluded)) & drawn,
        {"text": f"{len(excluded)} of {drawn.sum()}", "title": ""},
    )
