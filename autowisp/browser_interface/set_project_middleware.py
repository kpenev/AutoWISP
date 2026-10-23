"""Define middleware to handle multiple AutoWISP BUI projects."""

from django.contrib import messages
from django.shortcuts import redirect

from autowisp.database.interface import has_project_database, set_project_home
from autowisp.exceptions import DatabaseError


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
        reason.
        """

        project_home = request.session.get("project_home")
        if project_home is not None:
            refusal = None
            # Checked first: the project may have been deleted since it was
            # selected, e.g. from another tab, and opening a home without a
            # database would create an empty project there.
            if has_project_database(project_home):
                try:
                    set_project_home(project_home)
                except DatabaseError as error:
                    refusal = str(error)
            else:
                refusal = (
                    f"Project {request.session.get('project_name')!r} is no "
                    f"longer there: {project_home!r} holds no project "
                    "database."
                )
            if refusal is not None:
                request.session.pop("project_home", None)
                request.session.pop("project_name", None)
                messages.error(request, refusal)
                return redirect("home:home")

        response = get_response(request)

        return response

    return activate_project
