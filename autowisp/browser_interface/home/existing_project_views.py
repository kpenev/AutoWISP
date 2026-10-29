"""Define the views adding a project moved or copied by hand.

Its home is chosen, then its name and description. A project with a
database of its own is registered at once. One naming a server database is
registered once that is cloned, or once using it as it is is confirmed (see
:mod:`autowisp.database.existing_project`). A clone runs in the background,
followed through its record (:mod:`autowisp.database.pending_projects`),
and the project list registers the project once it finds the clone done
(:func:`settle_pending`).
"""

import os
from sys import executable

import platformdirs
from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST
from sqlalchemy.engine import make_url

from autowisp.browser_interface.core.background import launch_detached
from autowisp.browser_interface.core.walk_fs_view import WalkFSView
from autowisp.database.existing_project import check_existing_project
from autowisp.database.interface import DB_URL_FNAME, has_project_database
from autowisp.database.pending_projects import (
    add_pending,
    dismiss_pending,
    get_log_fname,
    get_pending_fname,
    list_pending,
    remove_pending,
)
from autowisp.exceptions import DatabaseError, ProjectStateError

from .models import Project

# The fields of the add page, kept in the session, under the key with
# "existing-" in front, while the home is chosen.
_form_fields = ("name", "description", "choice", "target-url")


class SelectExistingHomeView(WalkFSView):
    """Choose the home of a project to add: a directory holding one."""

    template = "home/select_existing_home.html"
    url_name = "home:select_existing_home"
    cancel_url_name = "home:add_existing_project"

    def _get_context(self, config, search_dir):
        """Return the template's context, offering only a project's home."""

        context = super()._get_context(config, search_dir)
        # Shown, but not selectable: a home is a directory.
        context["unselectable"] = context.pop("file_list")
        context["file_list"] = []
        current_dir = context["currentdir"]
        context["valid_home"] = current_dir != "Computer" and (
            has_project_database(current_dir)
        )
        context["invalid_home_message"] = (
            f"{current_dir} holds no AutoWISP project: it has neither "
            f"autowisp.db nor {DB_URL_FNAME}."
        )
        return context

    def post(self, request, *_args, **_kwargs):
        """Take the directory shown as the home if it can be added, or say why.

        Any other submission shows the directory entered.
        """

        home = request.POST["currentdir"]
        if "set-project-home" in request.POST:
            try:
                _check_home(home)
            except (DatabaseError, ProjectStateError) as error:
                messages.error(request, str(error))
                return redirect("home:select_existing_home", dirname=home)
            request.session["existing-home"] = home
            return redirect("home:add_existing_project")
        return redirect("home:select_existing_home", dirname=home)


def _save_form(request):
    """Keep the add page's fields in the session, while the home is chosen."""

    for field in _form_fields:
        request.session["existing-" + field] = request.POST.get(field, "")
    home = request.session.get("existing-home")
    if home and request.session["existing-name"] == os.path.basename(home):
        # The default, which follows the home if another is chosen.
        request.session["existing-name"] = ""


def _forget_form(session):
    """Forget the add page's fields, once the project is added."""

    for field in ("home", *_form_fields):
        session.pop("existing-" + field, None)


def _check_home(home):
    """
    Return the project in the home, as :func:`check_existing_project` does.

    Raises:
        DatabaseError:    As :func:`check_existing_project` does.

        ProjectStateError:    If the project is registered already, or is
            being added already.
    """

    project = check_existing_project(
        home,
        [
            registered.path
            for registered in Project.objects.all()  # pylint: disable=no-member
        ],
    )
    for pending in list_pending():
        if os.path.normcase(pending["home"]) == os.path.normcase(
            project["home"]
        ):
            raise ProjectStateError(
                f"The project in {home!r} is being added already, as "
                f"{pending['name']!r}: see the project list."
            )
    return project


def _get_page_context(session):
    """
    Return the context of the add page, or its confirmation's.

    The home is checked each time: things may have changed since it was
    chosen. Its ``project`` is None until one is chosen.

    Raises:
        DatabaseError, ProjectStateError:    As :func:`_check_home` does.
    """

    home = session.get("existing-home", "")
    return {
        "home": home,
        "name": session.get("existing-name") or os.path.basename(home),
        "description": session.get("existing-description", ""),
        "choice": session.get("existing-choice") or "clone-sqlite",
        "target_url": session.get("existing-target-url", ""),
        "project": _check_home(home) if home else None,
    }


def _send_back(request, error):
    """Say why the home chosen cannot be added, where homes are chosen.

    It is forgotten, so that leaving the choice does not come back here.
    """

    home = request.session.pop("existing-home")
    messages.error(request, str(error))
    return redirect("home:select_existing_home", dirname=home)


def _register(request, home, name, description):
    """Register the project, and go to the project list."""

    Project(name=name, path=home, description=description).save()
    _forget_form(request.session)
    messages.success(request, f"Added project {name}, in {home}.")
    return redirect("home:home")


def _launch_clone(request, context, target_url):
    """Start cloning the project's database, to register it once done."""

    pending_id = add_pending(
        context["name"],
        context["description"],
        context["project"]["home"],
        target_url,
    )
    try:
        launch_detached(
            [
                executable,
                "-m",
                "autowisp.clone_existing_project",
                get_pending_fname(pending_id),
            ],
            os.path.relpath(
                get_log_fname(pending_id),
                platformdirs.user_data_dir("autowisp"),
            ),
            # No project's session: the clone records only in its record.
            {},
        )
    except OSError as error:
        remove_pending(pending_id)
        messages.error(
            request,
            f"Cloning the database of {context['name']} could not be "
            f"started: {error}",
        )
        return redirect("home:add_existing_project")
    _forget_form(request.session)
    messages.info(
        request,
        f"Cloning the database of {context['name']}: the project is added "
        "once that is done.",
    )
    return redirect("home:home")


def _add_server_project(request, context):
    """Add a project naming a server database, as the page's choice says."""

    if context["choice"] == "use-existing":
        return redirect("home:confirm_existing_database")
    target_url = None
    if context["choice"] == "clone-server":
        target_url = context["target_url"].strip()
        if not target_url:
            messages.error(
                request, "Give the URL of the empty database to clone into."
            )
            return redirect("home:add_existing_project")
    return _launch_clone(request, context, target_url)


def add_existing_project(request):
    """Show the page adding an existing project, or add it as asked."""

    if request.method == "POST":
        _save_form(request)
        home = request.session.get("existing-home")
        if "select-home" in request.POST or not home:
            if home:
                return redirect("home:select_existing_home", dirname=home)
            return redirect("home:select_existing_home")

    try:
        context = _get_page_context(request.session)
    except (DatabaseError, ProjectStateError) as error:
        return _send_back(request, error)
    if request.method != "POST":
        return render(request, "home/add_existing_project.html", context)

    project = context["project"]
    if project["db_url"] is None:
        return _register(
            request, project["home"], context["name"], context["description"]
        )
    return _add_server_project(request, context)


def confirm_existing_database(request):
    """Ask before adding a project on the database it names, or add it."""

    try:
        context = _get_page_context(request.session)
    except (DatabaseError, ProjectStateError) as error:
        return _send_back(request, error)
    project = context["project"]
    if project is None or project["db_url"] is None:
        # No home chosen, or its database is its own after all.
        return redirect("home:add_existing_project")
    if request.method == "POST":
        return _register(
            request, project["home"], context["name"], context["description"]
        )
    return render(request, "home/confirm_existing_database.html", context)


@require_POST
def dismiss_pending_project(request, pending_id):
    """Give up on adding a project whose clone did not finish."""

    try:
        dismiss_pending(pending_id)
    except FileNotFoundError:
        # Dismissed already, e.g. by a second click.
        pass
    except (ProjectStateError, ValueError) as error:
        messages.error(request, str(error))
    return redirect("home:home")


def settle_pending(request):
    """
    Register each project whose clone is done; return the other additions.

    Returns:
        [dict]:    The additions not done, oldest first, each with its
        ``id``, ``name``, ``home``, ``state`` (see :func:`list_pending`), the
        ``message`` of a failure, the ``log`` of its clone, and the
        ``target`` it clones into, its password hidden, None for SQLite.
    """

    unsettled = []
    for pending in list_pending():
        if pending["state"] != "done":
            unsettled.append(
                {
                    **{
                        key: pending.get(key)
                        for key in ("id", "name", "home", "state", "message")
                    },
                    "log": get_log_fname(pending["id"]),
                    "target": pending["target_url"]
                    and make_url(pending["target_url"]).render_as_string(
                        hide_password=True
                    ),
                }
            )
            continue
        try:
            # First, so that of two requests settling it, one registers it.
            remove_pending(pending["id"])
        except FileNotFoundError:
            continue
        Project(
            name=pending["name"],
            path=pending["home"],
            description=pending["description"],
        ).save()
        if pending["stale"]:
            messages.warning(
                request,
                f"Added project {pending['name']}, its database cloned. The "
                "project had changed since these experiments were copied, "
                f"which are now stale: {', '.join(pending['stale'])}.",
            )
        else:
            messages.success(
                request,
                f"Added project {pending['name']}, its database cloned.",
            )
    return unsettled
