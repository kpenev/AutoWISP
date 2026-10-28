"""Confirm and start operations on the experiments of the selected project."""

import os
from contextlib import contextmanager
from sys import executable

import platformdirs
import psutil
from django.contrib import messages
from django.http import HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from autowisp.browser_interface.core.background import (
    find_since_launch,
    launch_detached,
)
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

# The session key under which the operation launched last is kept, until the
# progress page reports how it ended (report_finished_operation).
_awaited_key = "awaited_operation"


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
    stop. Unlocking the project (``command`` alone, ``unlock``) names the
    experiments it makes stale, and those that would still lock it.
    """

    command = request.GET.get("command")
    if command not in ("reset", "merge", "delete", "delete-backup", "unlock"):
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
            elif command == "unlock":
                context["marked_stale"], context["still_locking"] = (
                    experiments.plan_unlock()
                )
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

    request.session[_awaited_key] = {
        "command": command,
        "name": request.POST["name"],
        "original_id": request.session["original_id"],
        # Its own copy, which a later launch elsewhere does not replace.
        # The process started is the operation itself: wisp-experiments
        # does its work in it, so its end is the operation's.
        "launch": launch_detached(cmd, "wisp_experiments.out", request.session),
    }
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


@require_POST
def unlock_project(request):
    """
    Unlock the original, at once, by marking its current experiments stale.

    Quick, and recording no run, so not a background operation. Experiments
    an operation is working on keep the lock, which the progress page then
    goes on showing.
    """

    with _get_project_experiments(request.session) as (experiments, original):
        unlocked = experiments.unlock()
    if unlocked:
        messages.success(
            request,
            f"Unlocked {original.name}, marking stale: "
            + ", ".join(unlocked)
            + ".",
        )
    else:
        messages.info(request, "No experiment was current to mark stale.")
    return redirect("processing:progress")


def _is_running(process_id):
    """
    Is the process of an operation still running?

    The process started is the operation itself, wisp-experiments handing
    its work to no other, so once it has exited the operation has recorded
    all it ever will: its run as finished, then the error that ended it, if
    any. The run alone would not tell: it is finished first.
    """

    try:
        # Reaped by launch_detached once it ends, so never left a zombie
        # for long.
        return psutil.Process(process_id).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _describe_done(command, name, original_name):
    """Return what an operation that succeeded did, as reported."""

    if command == "create":
        return f"Created experiment {name!r} of {original_name}."
    if command == "reset":
        return (
            f"Reset experiment {name!r} to the processing of "
            f"{original_name}."
        )
    if command == "merge":
        return (
            f"Merged experiment {name!r} into {original_name}, which is "
            "backed up until the merge backups are deleted."
        )
    if command == "restore":
        return (
            f"Restored {original_name} from the interrupted merge of "
            f"experiment {name!r}."
        )
    return f"Deleted experiment {name!r} of {original_name}."


def report_finished_operation(request):
    """
    Report how the operation launched last ended, once it has.

    Reported once, as a message, whatever is selected by then: the error
    that ended it, linked to its details; or that it ended without recording
    anything, e.g. failing to start; or what it did. Read from the original's
    database, through an engine of its own. A created experiment is then
    selected, if its original still is. Forgotten if the original is gone.

    Returns:
        HttpResponse or None:    Where to go instead of the progress page:
        the experiment just created. None to show the progress page.
    """

    awaited = request.session.get(_awaited_key)
    if awaited is None or _is_running(awaited["launch"]["process"]):
        return None
    del request.session[_awaited_key]
    original = Project.objects.filter(  # pylint: disable=no-member
        id=awaited["original_id"]
    ).first()
    if original is None:
        return None

    engine = create_project_engine(get_project_url(original.path))
    try:
        with start_engine_session(engine) as db_session:
            runs, errors = find_since_launch(awaited["launch"], db_session)
            failure = (errors[0].id, errors[0].user_message) if errors else None
    finally:
        engine.dispose()

    command, name = awaited["command"], awaited["name"]
    if failure is not None:
        messages.error(
            request,
            f"{command.capitalize()} experiment {name!r} of {original.name} "
            f"failed: {failure[1]}",
            extra_tags=str(failure[0]),
        )
        return None
    if not runs:
        messages.error(
            request,
            f"{command.capitalize()} experiment {name!r} of {original.name} "
            "ended without recording anything: see wisp_experiments.out in "
            f"{platformdirs.user_data_dir('autowisp')}.",
        )
        return None

    messages.success(request, _describe_done(command, name, original.name))
    if (
        command == "create"
        and request.session.get("original_id") == original.id
        and request.session.get("experiment_name") is None
    ):
        return redirect(
            "home:select_experiment", project_id=original.id, name=name
        )
    return None
