"""Start operations on the experiments of the selected project."""

from sys import executable

from django.http import HttpResponseBadRequest
from django.shortcuts import redirect
from django.views.decorators.http import require_POST

from autowisp.browser_interface.core.background import launch_detached
from autowisp.browser_interface.home.models import Project


def select_original(session):
    """Make the selected experiment's original the selected project, if not."""

    original_id = session.pop("original_id", None)
    if original_id is None:
        return
    session.pop("experiment_name", None)
    original = Project.objects.get(id=original_id)  # pylint: disable=no-member
    session["project_home"] = original.path
    session["project_name"] = original.name


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
    experiment's ``name`` (except for ``delete-backup``), and ``force`` for
    a merge that is to go ahead although it loses a change to the original.
    """

    # The commands the interface starts, with the options each is given:
    # deleting is confirmed by the interface itself.
    command_options = {
        "create": [],
        "reset": [],
        "merge": [],
        "restore": [],
        "delete": ["--yes"],
        "delete-backup": ["--yes"],
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
    if command != "delete-backup":
        # After "--": a name may start with "-", which is not an option.
        cmd.extend(["--", request.POST["name"]])

    launch_detached(cmd, "wisp_experiments.out")
    return redirect("processing:progress", await_start=0)
