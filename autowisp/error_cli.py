"""CLI-side error reporting: persist, render to stderr, set exit code.

The top-level CLI handlers (``run_pipeline`` and, via
:func:`cli_entry_point`, each standalone ``wisp-*`` step entry) funnel an
escaping :class:`~autowisp.exceptions.AutoWISPError` through
:func:`report_error`, which records it as an ``Error`` row plus sidecar
and renders the stored record to stderr.
"""

import functools
import logging
import sys

from autowisp.database.interface import start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error

# pylint: enable=no-name-in-module
from autowisp.exceptions import AutoWISPError, Component
from autowisp.error_context import capture_errors, get_error_context
from autowisp.error_persistence import persist_error
from autowisp.error_render import (
    error_detail,
    error_summary,
    format_detail_text,
)

git_id = "$Id$"

_logger = logging.getLogger(__name__)


def exit_code_for(component):
    """Map an error's component to a process exit code.

    Distinct codes let shell scripts tell a step failure from a pipeline
    (orchestration/config) failure.

    Args:
        component(Component):    The failing component.

    Returns:
        int:    A non-zero exit code.
    """

    return {
        Component.STEP: 2,
        Component.PIPELINE: 3,
        Component.BUI: 4,
    }.get(component, 1)


def _wants_traceback():
    """Whether the invocation asked for the full developer traceback.

    Looks for ``--traceback`` on the command line (kept distinct from the
    steps' ``-v`` logging-verbosity flag to avoid clashing).
    """

    return "--traceback" in sys.argv


def report_error(exc, *, developer=False, stream=None):
    """Persist ``exc`` and render the stored record to ``stream``.

    The default view is the one-line :func:`error_summary`; ``developer``
    switches to the full :func:`format_detail_text`. Either way a pointer
    line names the ``Error`` id and the crash-report command. If
    persistence fails (e.g. no project database), a minimal line is
    printed instead so the user still sees something. Never raises.

    Args:
        exc(AutoWISPError):    The error to report.

        developer(bool):    Render the full technical detail.

        stream:    Output stream; defaults to ``sys.stderr``.

    Returns:
        int:    The exit code for ``exc`` (see :func:`exit_code_for`).
    """

    stream = stream or sys.stderr
    error_id = persist_error(exc)

    if error_id is None:
        # Could not record it (e.g. no project DB); still tell the user.
        print(f"[{exc.component.value}] {exc.user_message}", file=stream)
        print("(this error could not be recorded to the database)", file=stream)
        return exit_code_for(exc.component)

    with start_db_session() as db_session:
        row = db_session.get(Error, error_id)
        if developer:
            body = format_detail_text(
                error_detail(row, db_session, developer=True)
            )
        else:
            body = error_summary(row, db_session)
        sidecar_path = row.sidecar_path

    print(body, file=stream)
    pointer = f"Error {error_id}"
    if sidecar_path:
        pointer += f"; details in {sidecar_path}"
    pointer += f"; run 'wisp-crash-report {error_id}' for a shareable report."
    print(pointer, file=stream)
    return exit_code_for(exc.component)


def cli_entry_point(*, component):
    """Decorate a standalone ``wisp-*`` ``main()`` as a CLI error boundary.

    Wraps the callable so that any escaping exception is converted to the
    right :class:`AutoWISPError` (via :func:`capture_errors`), stamped
    with the ambient pipeline-run context if it lacks one, reported with
    :func:`report_error`, and turned into a non-zero ``SystemExit``.

    Args:
        component(Component):    The component the entry point belongs to.

    Returns:
        Callable:    The decorator.
    """

    def decorate(func):
        captured = capture_errors(component=component)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return captured(*args, **kwargs)
            except AutoWISPError as exc:
                if exc.pipeline_run is None:
                    exc.with_pipeline_run(get_error_context().pipeline_run)
                sys.exit(report_error(exc, developer=_wants_traceback()))

        return wrapper

    return decorate
