"""Define context processors for the entire interface."""

import logging

from autowisp.error_render import error_count

_logger = logging.getLogger(__name__)


def global_variables(request):
    """Set global variables available to all templates."""

    return {
        "project_name": request.session.get("project_name", ""),
        "error_count": _project_error_count(request),
    }


def _project_error_count(request):
    """Return the recorded-error count for the active project, else 0.

    Runs on every page, so it must never raise: with no project selected
    (or any query failure) it quietly reports 0 rather than breaking the
    page.
    """

    if not request.session.get("project_home"):
        return 0
    try:
        return error_count()
    except Exception:  # pylint: disable=broad-except
        _logger.debug("Could not count errors for the badge.", exc_info=True)
        return 0
