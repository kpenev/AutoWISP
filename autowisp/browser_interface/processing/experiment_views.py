"""Confirm and start operations on the experiments of the selected project."""

import os
from contextlib import contextmanager
from sys import executable

from django.contrib import messages
from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from autowisp.browser_interface.core.background import launch_detached
from autowisp.browser_interface.home.models import Project
from autowisp.database.interface import (
    create_project_engine,
    get_db_engine,
    get_project_url,
    get_sqlite_url,
)
from autowisp.database.pipeline_runs import find_active_runs
from autowisp.database.sessions import start_engine_session
from autowisp.exceptions import ExperimentError
from autowisp.project_experiments import ProjectExperiments
from .experiments_menu import plan_experiments_menu


def select_original(session):
    """Make the selected experiment's original the selected project, if not."""

    if session.pop("experiment_name", None) is None:
        return
    original = Project.objects.get(  # pylint: disable=no-member
        id=session["original_id"]
    )
    session["project_home"] = original.path
    session["project_name"] = original.name


@contextmanager
def _get_project_experiments(session):
    """
    Yield the experiments of the selected project's original, and its row.

    With an experiment selected, the project's engine is the experiment's,
    so the original is read through an engine of its own, disposed of once
    done.

    Yields:
        ProjectExperiments:    The original's experiments.

        Project:    The original's registered project.
    """

    original = Project.objects.get(  # pylint: disable=no-member
        id=session["original_id"]
    )
    if session.get("experiment_name") is None:
        yield ProjectExperiments(get_db_engine(), original.path), original
        return
    engine = create_project_engine(get_project_url(original.path))
    try:
        yield ProjectExperiments(engine, original.path), original
    finally:
        engine.dispose()


def get_experiments_menu(session):
    """
    Return what the progress page's experiments menu shows, or None.

    Read from the original's rows: one query of its experiments, and in the
    original a look at its experiments' directory for the backups merges
    left, which only the original's menu offers deleting. Nothing is hashed.

    Returns:
        dict or None:    ``rows`` and ``needs_attention``, as
        :func:`plan_experiments_menu` returns them; ``selected``, the
        selected experiment's name, None in the original; ``original``, the
        original's registered ``Project``; and ``has_backups``. None for a
        project registered from an experiment's home, which has no menu:
        its original manages its experiments.
    """

    selected = session.get("experiment_name")
    with _get_project_experiments(session) as (experiments, original):
        described = experiments.describe_experiments()
        has_backups = selected is None and bool(
            experiments.list_merge_backups()
        )
    if described is None:
        return None
    rows, needs_attention = plan_experiments_menu(described, selected)
    return {
        "rows": rows,
        "needs_attention": needs_attention,
        "selected": selected,
        "original": original,
        "has_backups": has_backups,
    }


def _find_stopped_runs(experiments, described):
    """
    Return what deleting the experiment would stop on this machine.

    Its operation, if one is running here, e.g. its create. Otherwise, if it
    may be opened, the runs going here in its own database: its processing.
    Runs going on other machines refuse the delete instead, which says so.

    Args:
        experiments(ProjectExperiments):    The original's experiments.

        described(dict):    The experiment, as ``describe_experiments``
            gives it.

    Returns:
        [str]:    Each run stopped, by its task, and its id for a run of
        processing.
    """

    operation = described["operation"]
    if operation is not None:
        return [operation["task"]] if operation["running"] == "here" else []
    home = experiments.get_openable_home(described["name"])
    # Checked before opening it: opening a missing one would create it.
    if not os.path.isfile(os.path.join(home, "autowisp.db")):
        return []
    engine = create_project_engine(get_sqlite_url(home))
    try:
        with start_engine_session(engine) as db_session:
            return [
                f"{pipeline_run.task} (run {pipeline_run.id})"
                for pipeline_run in find_active_runs(db_session)[0]
            ]
    finally:
        engine.dispose()


def confirm_experiment_operation(request):
    """
    Ask to confirm an operation that loses something, saying what.

    For resetting, merging or deleting the experiment named by the GET's
    ``command`` and ``name``, and deleting the backups merges left
    (``command`` alone). A merge's warning depends on why it would lose a
    change to the original: the experiment is stale, or it is current but
    the original changed despite the lock, which only hashing the original
    tells -- done here, never in the menu. A delete names the runs it would
    stop.
    """

    command = request.GET.get("command")
    if command not in ("reset", "merge", "delete", "delete-backup"):
        return HttpResponseBadRequest(f"Nothing to confirm for {command!r}.")
    name = request.GET.get("name")
    context = {"command": command, "name": name}
    try:
        with _get_project_experiments(request.session) as (
            experiments,
            original,
        ):
            context["original"] = original
            if command == "delete-backup":
                context["backups"] = [
                    os.path.basename(backup)
                    for backup in experiments.list_merge_backups()
                ]
            elif command == "merge":
                experiment = next(
                    (
                        row
                        for row in experiments.get_experiments() or []
                        if row.name == name
                    ),
                    None,
                )
                if experiment is None:
                    raise ExperimentError(
                        f"The project has no experiment named {name!r}."
                    )
                if experiment.state == "stale":
                    context["merge_case"] = "stale"
                elif experiments.is_original_unchanged(experiment):
                    context["merge_case"] = "unchanged"
                else:
                    context["merge_case"] = "changed"
            elif command == "delete":
                described = next(
                    (
                        row
                        for row in experiments.describe_experiments() or []
                        if row["name"] == name
                    ),
                    None,
                )
                if described is None:
                    raise ExperimentError(
                        f"The project has no experiment named {name!r}."
                    )
                context["stopped"] = _find_stopped_runs(experiments, described)
    except ExperimentError as error:
        messages.error(request, str(error))
        return redirect("processing:progress")
    return render(
        request, "processing/confirm_experiment_operation.html", context
    )


@require_POST
def run_experiment_operation(request):
    """
    Start an operation on the experiments, and show it as a run.

    It runs ``wisp-experiments`` on the original, in the background, and
    records its run there, so the original becomes the selected project and
    its progress page follows the run. Even when started from inside the
    experiment: while the operation works on it, it cannot be opened anyway.
    A refusal or a failure is recorded as an error of the original.

    The POST gives the ``command``, a ``wisp-experiments`` command, the
    experiment's ``name``, and ``force`` for a merge that is to go ahead
    although it loses a change to the original. Deleting the backups merges
    left records no run, so it is done at once instead
    (:func:`delete_merge_backups`).
    """

    # The commands the interface starts, with the options each is given:
    # deleting is confirmed by the interface itself.
    command_options = {
        "create": [],
        "reset": [],
        "merge": [],
        "restore": [],
        "delete": ["--yes"],
    }

    command = request.POST.get("command")
    if command not in command_options:
        return HttpResponseBadRequest(f"Unknown operation {command!r}.")
    select_original(request.session)

    cmd = [
        executable,
        "-m",
        "autowisp.experiments_cli",
        request.session["project_home"],
        command,
        *command_options[command],
    ]
    if command == "merge" and request.POST.get("force"):
        cmd.append("--force")
    # After "--": a name may start with "-", which is not an option.
    cmd.extend(["--", request.POST["name"]])

    launch_detached(cmd, "wisp_experiments.out", request.session)
    return redirect("processing:progress", await_start=0)


@require_POST
def delete_merge_backups(request):
    """
    Delete the backups merges left, at once, and say how it went.

    Not a background operation: it records no run for the progress page to
    follow. A delete interrupted part way leaves the rest of a backup under
    its name, which is offered for deleting again.
    """

    with _get_project_experiments(request.session) as (experiments, _):
        try:
            deleted = experiments.delete_merge_backup()
        except ExperimentError as error:
            messages.error(request, str(error))
        else:
            messages.success(
                request,
                "Deleted the merge backups: "
                + ", ".join(os.path.basename(backup) for backup in deleted)
                + ".",
            )
    return redirect("processing:progress")
