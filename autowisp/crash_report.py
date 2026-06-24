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

import re

from sqlalchemy import select, update

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Configuration, Parameter

# pylint: enable=no-name-in-module

git_id = "$Id$"

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
