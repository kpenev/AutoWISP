"""Django middleware recording errors that escape a BUI view.

When a view raises, this captures a lightweight request snapshot onto the
error and records it as a queryable ``Error`` row plus detail sidecar
(via ``persist_error``), so web failures become the same durable records
as pipeline failures. Django's normal exception handling (the 500
response) is left untouched.

The request snapshot records *keys only* -- never query/POST values -- to
avoid persisting user-entered secrets.
"""

import logging
from datetime import datetime, timezone

from autowisp.exceptions import AutoWISPError, ViewError
from autowisp.error_persistence import persist_error

_logger = logging.getLogger(__name__)


def _request_snapshot(request):
    """Capture path / view / parameter *keys* (never values) from a request.

    Best-effort: any attribute that cannot be read is simply omitted, so
    snapshotting a request can never itself raise.

    Args:
        request:    The Django request (or any object exposing the same
            attributes).

    Returns:
        dict:    The request context to attach to the error.
    """

    snapshot = {}
    for key, getter in (
        ("path", lambda: request.path),
        ("method", lambda: request.method),
        (
            "view_name",
            lambda: getattr(request.resolver_match, "view_name", None),
        ),
        ("query_keys", lambda: sorted(request.GET.keys())),
        ("post_keys", lambda: sorted(request.POST.keys())),
        ("session_key", lambda: request.session.session_key),
    ):
        try:
            snapshot[key] = getter()
        except Exception:  # pylint: disable=broad-except
            continue
    return snapshot


class ErrorCaptureMiddleware:
    """Persist an :class:`AutoWISPError` (or wrapped view error) on failure.

    Uses Django's ``process_exception`` hook, which fires when a view
    raises before the exception is converted to a 500 response.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        """Record ``exception`` with a request snapshot; return ``None``.

        Returning ``None`` leaves Django to render its usual error
        response. Recording is best-effort and never raises (an error
        while recording an error must not mask the original).

        Args:
            request:    The Django request being handled.

            exception(BaseException):    The exception the view raised.

        Returns:
            None
        """

        try:
            if isinstance(exception, AutoWISPError):
                error = exception
            else:
                error = ViewError(str(exception) or type(exception).__name__)
                error.__cause__ = exception
                error.__traceback__ = exception.__traceback__

            error.details.setdefault("bui_request", _request_snapshot(request))
            if error.crashed is None:
                error.crashed = datetime.now(timezone.utc)

            persist_error(error)
        except Exception:  # pylint: disable=broad-except
            _logger.exception("Failed to record BUI error.")
        # Falls through returning None per Django's process_exception
        # contract, leaving Django to render its usual error response.
