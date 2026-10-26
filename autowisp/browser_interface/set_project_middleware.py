"""Define middleware to handle multiple AutoWISP BUI projects."""

from django.contrib import messages
from django.shortcuts import redirect
from django.urls import Resolver404, resolve

from autowisp.browser_interface.home.models import Project
from autowisp.database.interface import (
    create_project_engine,
    get_project_url,
    has_project_database,
    set_project_home,
)
from autowisp.exceptions import DatabaseError, ExperimentError
from autowisp.project_experiments import ProjectExperiments


def _refuse_experiment(session):
    """Return why the selected experiment may not be opened now, or None.

    An operation may have started on it since it was selected, e.g. from
    another tab or the command line, and its original's row says so. One
    query on the original's database, through an engine of its own, the
    current project being the experiment. This narrows the window rather
    than closing it: an operation can still start right after.
    """

    name = session.get("experiment_name")
    if name is None:
        return None
    original = Project.objects.filter(  # pylint: disable=no-member
        id=session["original_id"]
    ).first()
    if original is None:
        return f"The project experiment {name!r} belongs to is gone."
    engine = create_project_engine(get_project_url(original.path))
    try:
        ProjectExperiments(engine, original.path).get_openable_home(name)
    except ExperimentError as error:
        return str(error)
    finally:
        engine.dispose()
    return None


def _needs_project(request):
    """
    Does the request go to a page working on the selected project?

    Every application's but the project list's (``home``), where projects
    are created, selected and deleted, and the admin's. A path that resolves
    to nothing is left to give its 404.
    """

    try:
        namespace = resolve(request.path_info).namespace
    except Resolver404:
        return False
    return namespace not in ("home", "admin")


def set_project_middleware(get_response):
    """Middleware to set the active BUI project for processing requests."""

    def activate_project(request):
        """Set the correct database before processing the request.

        The project comes from the session, which outlives a restart of the
        interface, so it may name a project whose database this version
        cannot use as it stands -- typically one needing a migration that
        arrived with an upgrade -- or one deleted since it was selected.
        Only selecting a project migrates it, and failing here would fail
        every page, the project list included, so the project is dropped
        from the session instead and the user sent to that list with the
        reason. A page working on the selected project is sent there too
        when none is selected, rather than failing for the lack of one.
        """

        project_home = request.session.get("project_home")
        if project_home is not None:
            # Checked first: the project may have been deleted since it was
            # selected, e.g. from another tab, and opening a home without a
            # database would create an empty project there.
            if not has_project_database(project_home):
                refusal = (
                    f"Project {request.session.get('project_name')!r} is no "
                    f"longer there: {project_home!r} holds no project "
                    "database."
                )
            elif "original_id" not in request.session:
                # Selected by a version that did not keep it, which every
                # page may now read.
                refusal = (
                    f"Select project {request.session.get('project_name')!r} "
                    "again: it was selected before this version of AutoWISP "
                    "was installed."
                )
            else:
                refusal = _refuse_experiment(request.session)
            if refusal is None:
                try:
                    set_project_home(project_home)
                except DatabaseError as error:
                    refusal = str(error)
            if refusal is not None:
                for key in [
                    "project_home",
                    "project_name",
                    "original_id",
                    "experiment_name",
                ]:
                    request.session.pop(key, None)
                messages.error(request, refusal)
                return redirect("home:home")
        elif _needs_project(request):
            # E.g. a form posted from a page shown before the session lost
            # its project.
            messages.info(request, "Select a project first.")
            return redirect("home:home")

        response = get_response(request)

        return response

    return activate_project
