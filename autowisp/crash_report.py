"""Build a shareable, scrubbed crash report for a recorded error.

The report gathers everything needed to diagnose a failure -- the error
record and its sidecar, the relevant per-process logs, the configuration
in effect, and environment provenance -- into a single zip the user can
hand to the maintainers.

Because logs and configuration can contain credentials (e.g. the Gaia
archive user/password threaded through the process configuration), every
text artifact is passed through the scrubbing helpers here before it
enters the report. Scrubbing is mandatory: nothing is written unscrubbed.
"""

import logging
import os
import re

from sqlalchemy import select, update

from autowisp.database.interface import start_db_session
from autowisp.database.image_processing import ImageProcessingManager

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    Image,
    ImageProcessingProgress,
    Parameter,
    Step,
)

# pylint: enable=no-name-in-module

git_id = "$Id$"

_logger = logging.getLogger(__name__)

#: Replacement written in place of a redacted secret value.
REDACTED = "***REDACTED***"  # pylint: disable=invalid-name

# Substring matched (case-insensitively) against a key to decide it names
# a secret. Covers the Gaia credentials this pipeline threads through the
# process config plus the usual credential-like names.
_SECRET = (  # pylint: disable=invalid-name
    r"gaia[_-]?(?:user|password)"
    r"|password|passwd|secret|token"
    r"|api[_-]?key|access[_-]?key|credentials?"
)

# A `key <:|=> value` assignment whose key names a secret, as it appears
# in dict reprs ('gaia_password': 'x'), JSON ("gaia_password": "x"), and
# ini/yaml-style config (gaia-password = x). The value -- quoted, or bare
# up to a comma or end of line -- is what gets redacted.
_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(['\"]?\b(?:{_SECRET})\b['\"]?\s*[:=]\s*)"
    r"('[^']*'|\"[^\"]*\"|[^\n,]*)"
)

# Matches a secret name anywhere in a mapping key.
_SECRET_KEY = re.compile(rf"(?i)(?:{_SECRET})")


def scrub_text(text):
    """Redact secret values from a blob of text (a log or config file).

    Replaces the value of any ``key: value`` / ``key = value`` assignment
    whose key names a secret with :data:`REDACTED`, leaving the key (and
    everything else) intact. Best-effort and never raises.

    Args:
        text(str):    The text to scrub.

    Returns:
        str:    The text with secret values redacted.
    """

    if not text:
        return text
    return _SECRET_ASSIGNMENT.sub(r"\1" + REDACTED, text)


def scrub_mapping(mapping):
    """Return a copy of ``mapping`` with secret-keyed values redacted.

    Recurses into nested dictionaries. A value is redacted when its key
    name matches a secret (e.g. ``gaia_password``, ``api_key``); other
    values are copied through unchanged.

    Args:
        mapping(dict):    The mapping to scrub.

    Returns:
        dict:    A scrubbed copy.
    """

    scrubbed = {}
    for key, value in mapping.items():
        if isinstance(key, str) and _SECRET_KEY.search(key):
            scrubbed[key] = REDACTED
        elif isinstance(value, dict):
            scrubbed[key] = scrub_mapping(value)
        else:
            scrubbed[key] = value
    return scrubbed


def scrub_config_values(db_session):
    """Redact secret configuration values in a database, in place.

    The project configuration stores credentials as ordinary rows (e.g. a
    ``gaia-password`` parameter), which a binary database file or SQL dump
    cannot be text-scrubbed for. This redacts the ``Configuration`` value
    of every parameter whose name names a secret.

    Intended for a *copy* of the project database destined for a crash
    report -- never the live database -- since it mutates the rows.

    Args:
        db_session:    A session connected to the database copy to scrub.

    Returns:
        int:    The number of configuration values redacted.
    """

    secret_param_ids = [
        param_id
        for param_id, name in db_session.execute(
            select(Parameter.id, Parameter.name)  # pylint: disable=no-member
        ).all()
        if name and _SECRET_KEY.search(name)
    ]
    if not secret_param_ids:
        return 0
    result = db_session.execute(
        update(Configuration)
        .where(
            Configuration.parameter_id.in_(  # pylint: disable=no-member
                secret_param_ids
            )
        )
        .values(value=REDACTED)
    )
    return result.rowcount or 0


# --- Locating the logs and processing record for an error. ------------


def find_error_progress(error_row, db_session=None):
    """Return the ``ImageProcessingProgress`` an error belongs to, or None.

    Resolves a step error to its processing record via run + step (and the
    image's type, when the error names an image), so the BUI can link the
    error to the matching log-review page. A pipeline/BUI error -- or a
    step with no recorded progress -- yields ``None`` rather than a wrong
    match.

    Args:
        error_row:    The ``Error`` row.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        ImageProcessingProgress or None
    """

    if db_session is None:
        with start_db_session() as own_session:
            return find_error_progress(error_row, own_session)

    if error_row.pipeline_run_id is None or not error_row.step_name:
        return None

    # pylint: disable=no-member
    step_id = db_session.scalar(
        select(Step.id).where(Step.name == error_row.step_name)
    )
    if step_id is None:
        return None

    query = select(ImageProcessingProgress).where(
        ImageProcessingProgress.run_id == error_row.pipeline_run_id,
        ImageProcessingProgress.step_id == step_id,
    )
    if error_row.image_id is not None:
        image_type_id = db_session.scalar(
            select(Image.image_type_id).where(Image.id == error_row.image_id)
        )
        if image_type_id is not None:
            query = query.where(
                ImageProcessingProgress.image_type_id == image_type_id
            )
    return db_session.scalars(
        query.order_by(ImageProcessingProgress.id.desc())
    ).first()
    # pylint: enable=no-member


def select_error_logs(error_row, db_session=None):
    """Return the per-process log files relevant to an error.

    Reuses the pipeline's own log-locating machinery
    (``ImageProcessingManager.find_processing_outputs``), so the
    configured ``logging_fname`` / ``std_out_err_fname`` naming is honored
    rather than assumed. Only the logs for the error's run and step are
    returned (the main-process log/outerr and the run's worker logs), not
    the whole log directory. Best-effort: returns the existing files it
    finds, or an empty list.

    Args:
        error_row:    The ``Error`` row.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        list[str]:    Absolute paths of the matching log files.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return select_error_logs(error_row, own_session)

    progress = find_error_progress(error_row, db_session)
    if progress is None:
        return []

    try:
        main_logs, worker_logs = ImageProcessingManager(
            pipeline_run_id=None
        ).find_processing_outputs(progress, db_session)
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "Could not locate logs for error %s",
            getattr(error_row, "id", None),
            exc_info=True,
        )
        return []

    candidates = list(main_logs)
    for entry in worker_logs:
        candidates.extend(entry)
    return sorted(
        {path for path in candidates if path and os.path.exists(path)}
    )
