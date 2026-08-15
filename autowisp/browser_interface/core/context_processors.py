"""Define context processors for the entire interface."""

import logging

from autowisp.error_render import open_and_total_error_count

_logger = logging.getLogger(__name__)


def global_variables(request):
    """Set global variables available to all templates."""

    open_errors, total_errors = _project_error_counts(request)
    return {
        "project_name": request.session.get("project_name", ""),
        "error_count": open_errors,
        "total_error_count": total_errors,
    }


def _project_error_counts(request):
    """Return ``(open, total)`` recorded errors for the active project.

    Runs on every page, so it must never raise: with no project selected
    (or any query failure) it quietly reports zeros rather than breaking
    the page.
    """

    if not request.session.get("project_home"):
        return 0, 0
    try:
        return open_and_total_error_count()
    except Exception:  # pylint: disable=broad-except
        _logger.debug("Could not count errors for the badge.", exc_info=True)
        return 0, 0
