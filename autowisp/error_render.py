"""Human-readable projections of a persisted :class:`Error` record.

The CLI and BUI never format an exception directly -- they render the
persisted ``Error`` row (and, on demand, its detail sidecar) through the
functions here, so the two front-ends cannot drift. Rendering is a
projection of the stored record, never of a live exception.
"""

import logging

from sqlalchemy import func, or_, select

from autowisp.database.interface import start_db_session

# pylint: disable=no-name-in-module
from autowisp.database.data_model import Error, Image, MasterFile, PipelineRun

# pylint: enable=no-name-in-module
from autowisp.exceptions import Component
from autowisp.error_persistence import load_sidecar

git_id = "$Id$"

_logger = logging.getLogger(__name__)


def _describe_artifact(error_row, db_session):
    """Return a short description of the artifact the error is about.

    Resolves the row's artifact FK to a path; falls back to the bare id if
    the artifact row is gone, and to ``None`` when no artifact is linked.

    Args:
        error_row(Error):    The error row.

        db_session:    Active database session for resolving the FK.

    Returns:
        str or None:    e.g. ``"image '/data/raw/x.fits'"``.
    """

    if error_row.image_id is not None:
        raw_fname = db_session.scalar(
            select(Image.raw_fname).where(  # pylint: disable=no-member
                Image.id == error_row.image_id  # pylint: disable=no-member
            )
        )
        return (
            f"image {raw_fname!r}"
            if raw_fname
            else f"image #{error_row.image_id}"
        )
    if error_row.master_file_id is not None:
        filename = db_session.scalar(
            select(MasterFile.filename).where(  # pylint: disable=no-member
                MasterFile.id  # pylint: disable=no-member
                == error_row.master_file_id
            )
        )
        return (
            f"master {filename!r}"
            if filename
            else f"master #{error_row.master_file_id}"
        )
    return None


def _run_provenance(error_row, db_session):
    """Return ``{host, process_id, code_version}`` for the error's run.

    Empty dict when the error has no pipeline run (standalone CLI/BUI) or
    the run row is gone.
    """

    if error_row.pipeline_run_id is None:
        return {}
    run = db_session.get(PipelineRun, error_row.pipeline_run_id)
    if run is None:
        return {}
    return {
        "host": run.host,
        "process_id": run.process_id,
        "code_version": run.code_version,
    }


def error_summary(error_row, db_session=None):
    """Return a one-line human summary of an error row.

    Format: ``[component:step] <artifact>: <user_message>`` (the artifact
    clause is omitted when none is linked).

    Args:
        error_row(Error):    The error row to summarize.

        db_session:    Optional active session; one is opened if omitted
            (only needed to resolve the artifact FK).

    Returns:
        str:    The one-line summary.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return error_summary(error_row, own_session)

    label = error_row.component
    if error_row.step_name:
        label = f"{label}:{error_row.step_name}"

    artifact = _describe_artifact(error_row, db_session)
    head = f"[{label}]"
    if artifact:
        head = f"{head} {artifact}:"
    return f"{head} {error_row.user_message}"


def error_list_rows(db_session=None, *, pipeline_run_id=None, step_name=None):
    """Return the rows for a list view, newest first, from inline columns.

    Reads only the queryable columns -- never a sidecar -- so a list view
    stays cheap regardless of how many errors there are.

    Args:
        db_session:    Optional active session; one is opened if omitted.

        pipeline_run_id(int or None):    If given, restrict to that run.

        step_name(str or None):    If given, restrict to that step.

    Returns:
        list[dict]:    One dict per error with ``id``, ``created``,
            ``component``, ``step_name``, ``artifact``, ``user_message``,
            and a one-line ``summary``, ordered newest first.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return error_list_rows(
                own_session,
                pipeline_run_id=pipeline_run_id,
                step_name=step_name,
            )

    # Open errors first, then resolved (dimmed in the UI); newest first
    # within each group.
    query = select(Error).order_by(
        Error.resolved.isnot(None),  # pylint: disable=no-member
        Error.created.desc(),  # pylint: disable=no-member
        Error.id.desc(),  # pylint: disable=no-member
    )
    if pipeline_run_id is not None:
        query = query.where(
            Error.pipeline_run_id  # pylint: disable=no-member
            == pipeline_run_id
        )
    if step_name is not None:
        query = query.where(
            Error.step_name == step_name  # pylint: disable=no-member
        )

    return [
        {
            "id": row.id,
            "created": row.created,
            "component": row.component,
            "step_name": row.step_name,
            "artifact": _describe_artifact(row, db_session),
            "user_message": row.user_message,
            "summary": error_summary(row, db_session),
            "resolved": row.resolved,
        }
        for row in db_session.scalars(query).all()
    ]


def error_count(db_session=None):
    """Return the number of open (unresolved) errors.

    This is what the error badge shows -- resolved errors are kept as
    history but no longer counted.

    Args:
        db_session:    Optional active session; one is opened if omitted.

    Returns:
        int:    The open-error count (0 if none).
    """

    if db_session is None:
        with start_db_session() as own_session:
            return error_count(own_session)
    # pylint: disable=not-callable,no-member
    return (
        db_session.scalar(
            select(func.count())
            .select_from(Error)
            .where(Error.resolved.is_(None))
        )
        or 0
    )


def error_counts_by_step(db_session=None):
    """Return ``{step_name: count}`` for open errors that name a step.

    Powers the per-step markers on the progress grid; resolved errors and
    errors with no step (pipeline/BUI) are excluded.

    Args:
        db_session:    Optional active session; one is opened if omitted.

    Returns:
        dict:    Mapping of step name to its open-error count.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return error_counts_by_step(own_session)
    # pylint: disable=not-callable,no-member
    rows = db_session.execute(
        select(Error.step_name, func.count())
        .where(Error.step_name.isnot(None), Error.resolved.is_(None))
        .group_by(Error.step_name)
    ).all()
    # pylint: enable=not-callable,no-member
    return dict(rows)


def open_error_count_for_steps(step_names, db_session=None):
    """Return how many open errors would gate launching ``step_names``.

    Counts open errors that bear on running the pipeline:

    - every open **pipeline** error (an orchestration/config failure is
      run-level, so it gates any launch until resolved), and
    - open **step** errors for the steps about to run (all steps for a
      full run, i.e. an empty ``step_names``).

    Open **BUI** errors are excluded -- they are web-interface issues, not
    a reason to hold back processing. Used by the start-processing gate.

    Args:
        step_names(iterable):    Step names about to be run; empty means a
            full run (every step).

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        int:    The number of open errors relevant to the launch.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return open_error_count_for_steps(step_names, own_session)

    # pylint: disable=not-callable,no-member
    step_names = list(step_names)
    relevant_step = Error.component == Component.STEP.value
    if step_names:
        relevant_step = relevant_step & Error.step_name.in_(step_names)

    return (
        db_session.scalar(
            select(func.count())
            .select_from(Error)
            .where(
                Error.resolved.is_(None),
                or_(Error.component == Component.PIPELINE.value, relevant_step),
            )
        )
        or 0
    )


def error_detail(error_row, db_session=None, *, developer=False):
    """Return the full human view of an error row as a dict.

    Lazily loads the sidecar. With ``developer=False`` the result holds
    the user-facing fields (summary, message, artifact, and remediation
    if the error provided one). With ``developer=True`` it adds the
    technical fields: exception class, full message, traceback, details,
    ``subprocess_id``, the run's host/PID/`code_version`, and the
    related-file list. A missing sidecar degrades gracefully (the
    sidecar-backed fields are simply absent / empty).

    Args:
        error_row(Error):    The error row to render.

        db_session:    Optional active session; one is opened if omitted.

        developer(bool):    Include the technical fields.

    Returns:
        dict:    The structured detail view.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return error_detail(error_row, own_session, developer=developer)

    sidecar = load_sidecar(error_row) or {}
    sidecar_details = sidecar.get("details") or {}

    detail = {
        "id": error_row.id,
        "summary": error_summary(error_row, db_session),
        "component": error_row.component,
        "step_name": error_row.step_name,
        "user_message": error_row.user_message,
        "artifact": _describe_artifact(error_row, db_session),
        "created": error_row.created,
    }

    remediation = sidecar_details.get("remediation")
    if remediation:
        detail["remediation"] = remediation

    if developer:
        detail.update(
            exception_class=error_row.exception_class,
            message=sidecar.get("message", error_row.user_message),
            traceback=sidecar.get("traceback"),
            details=sidecar_details,
            related_files=sidecar.get("related_files", []),
            subprocess_id=error_row.subprocess_id,
            sidecar_available=bool(sidecar),
            **_run_provenance(error_row, db_session),
        )

    return detail


def format_detail_text(detail):
    """Render an :func:`error_detail` dict as plain text for a terminal.

    Keeps all formatting in this module (the front-ends never format
    fields themselves). Only the keys present in ``detail`` are shown, so
    the same function serves both the user and developer views.

    Args:
        detail(dict):    The result of :func:`error_detail`.

    Returns:
        str:    A multi-line, human-readable rendering.
    """

    lines = [detail["summary"]]

    def add(label, key):
        if detail.get(key) not in (None, "", [], {}):
            lines.append(f"  {label}: {detail[key]}")

    add("Error id", "id")
    add("Remediation", "remediation")
    # Developer-only fields (absent unless developer=True was used).
    add("Exception", "exception_class")
    add("Host", "host")
    add("Process", "process_id")
    add("Subprocess", "subprocess_id")
    add("Code version", "code_version")
    if detail.get("related_files"):
        lines.append("  Related files:")
        for related in detail["related_files"]:
            lines.append(
                f"    - [{related.get('kind')}/{related.get('role')}] "
                f"{related.get('path')}"
            )
    if detail.get("message") and detail["message"] != detail.get(
        "user_message"
    ):
        lines.append(f"  Message: {detail['message']}")
    if detail.get("traceback"):
        lines.append("  Traceback:")
        lines.extend(
            f"    {line}" for line in detail["traceback"].rstrip().splitlines()
        )
    return "\n".join(lines)
