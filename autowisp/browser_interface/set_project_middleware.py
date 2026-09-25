"""Define middleware to handle multiple AutoWISP BUI projects."""

from django.contrib import messages
from django.shortcuts import redirect

from autowisp.database.interface import set_project_home
from autowisp.exceptions import DatabaseError


def set_project_middleware(get_response):
    """Middleware to set the active BUI project for processing requests."""

    def activate_project(request):
        """Set the correct database before processing the request.

        The project comes from the session, which outlives a restart of the
        interface, so it may name a project whose database this version
        cannot use as it stands -- typically one needing a migration that
        arrived with an upgrade. Only selecting a project migrates it, and
        failing here would fail every page, the project list included, so
        the project is dropped from the session instead and the user sent
        to that list with the reason.
        """

        project_home = request.session.get("project_home")
        if project_home is not None:
            try:
                set_project_home(project_home)
            except DatabaseError as error:
                request.session.pop("project_home", None)
                request.session.pop("project_name", None)
                messages.error(request, str(error))
                return redirect("home:home")

        response = get_response(request)

        return response

    return activate_project
