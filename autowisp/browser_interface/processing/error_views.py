"""Views for browsing recorded pipeline/step/BUI errors."""

from datetime import datetime, timezone

from django.http import Http404
from django.shortcuts import redirect, render

from autowisp.database.interface import start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error

# pylint: enable=no-name-in-module
from autowisp.error_render import error_detail, error_list_rows


def error_list(request):
    """Show every recorded error for the project, newest first.

    Backed by the queryable columns only -- no detail sidecar is opened --
    so the list stays cheap. An optional ``run`` query parameter restricts
    to a single pipeline run.
    """

    run = request.GET.get("run")
    pipeline_run_id = int(run) if run and run.isdigit() else None
    step_name = request.GET.get("step") or None
    with start_db_session() as db_session:
        rows = error_list_rows(
            db_session,
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
        )
    return render(
        request,
        "processing/error_list.html",
        {
            "errors": rows,
            "filtered_run": pipeline_run_id,
            "filtered_step": step_name,
        },
    )


def error_detail_view(request, error_id):
    """Show one error in full.

    Defaults to the user-facing view; ``?developer=1`` reveals the
    technical fields (message, traceback, details, worker/host info)
    loaded from the detail sidecar.
    """

    developer = request.GET.get("developer") == "1"
    with start_db_session() as db_session:
        row = db_session.get(Error, error_id)
        if row is None:
            raise Http404(f"No recorded error with id {error_id}.")
        detail = error_detail(row, db_session, developer=developer)
        resolved = row.resolved
    return render(
        request,
        "processing/error_detail.html",
        {
            "detail": detail,
            "developer": developer,
            "error_id": error_id,
            "resolved": resolved,
        },
    )


def toggle_error_resolved(request, error_id):
    """Mark an error resolved, or reopen it, then return to the list.

    Resolving sets the ``resolved`` timestamp (the error drops out of the
    badge, the progress-grid markers, and the start-processing gate, but
    stays in the list as history); reopening clears it.
    """

    if request.method != "POST":
        raise Http404("Use POST to change an error's resolved state.")
    with start_db_session() as db_session:
        row = db_session.get(Error, error_id)
        if row is None:
            raise Http404(f"No recorded error with id {error_id}.")
        row.resolved = (
            None if row.resolved is not None else datetime.now(timezone.utc)
        )
    return redirect(request.POST.get("next") or "processing:error_list")
