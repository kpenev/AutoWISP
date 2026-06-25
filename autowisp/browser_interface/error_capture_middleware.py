"""Django middleware recording errors that escape a BUI view.

When a view raises, this captures a lightweight request snapshot onto the
error and records it as a queryable ``Error`` row plus detail sidecar
(via ``persist_error``), so web failures become the same durable records
as pipeline failures.

Rather than letting Django render its technical exception page, it then
keeps the user *inside the BUI*: it queues a dismissible message and
redirects back to where the user was, so a failure becomes something the
user can read and act on (the full technical detail stays available on
the error-detail page). Django's own ``Http404`` / ``PermissionDenied``
control-flow exceptions are deliberately left for Django to handle.

The request snapshot records *keys only* -- never query/POST values -- to
avoid persisting user-entered secrets.
"""

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse

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


def _safe_return_url(request):
    """Pick an in-BUI URL to send the user back to after an error.

    Prefers the page the request came from (``HTTP_REFERER``) so the user
    stays where they were, but only when it is same-host and not a GET page
    pointing back at itself (which would redirect-loop on a page that
    always errors). Falls back to the BUI home page.

    Args:
        request:    The Django request being handled.

    Returns:
        str:    A URL safe to redirect to within the BUI.
    """

    referer = request.META.get("HTTP_REFERER")
    if referer:
        parsed = urlparse(referer)
        same_host = not parsed.netloc or parsed.netloc == request.get_host()
        self_loop = request.method == "GET" and parsed.path == request.path
        if same_host and not self_loop:
            return referer
    return reverse("home:home")


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
        """Record ``exception`` and keep the user inside the BUI.

        Records the (possibly wrapped) error with a request snapshot, then
        -- instead of returning ``None`` and letting Django render its
        technical exception page -- queues a dismissible message (linking
        to the error-detail page) and redirects back to where the user
        was. Django's ``Http404`` / ``PermissionDenied`` are passed through
        untouched so their normal 404/403 handling stands. Everything here
        is best-effort: if recording or redirecting itself fails, fall back
        to ``None`` so Django handles the original error.

        Args:
            request:    The Django request being handled.

            exception(BaseException):    The exception the view raised.

        Returns:
            HttpResponseRedirect to stay in the BUI, or ``None`` to defer
            to Django (for 404/403 and on internal failure).
        """

        if isinstance(exception, (Http404, PermissionDenied)):
            return None

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

            error_id = persist_error(error)

            # The error id rides in extra_tags so the template can link the
            # banner to the full error-detail page (empty when persistence
            # failed, in which case the banner shows the message only).
            messages.error(
                request,
                error.user_message,
                extra_tags=str(error_id) if error_id is not None else "",
            )
            return HttpResponseRedirect(_safe_return_url(request))
        except Exception:  # pylint: disable=broad-except
            _logger.exception("Failed to record/redirect BUI error.")
            return None
