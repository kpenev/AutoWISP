"""Define middleware to handle multiple AutoWISP BUI projects."""

from os import path
# from autowisp.database.interface import set_project_home
from autowisp.database.interface import set_project_home


def set_project_middleware(get_response):
    """Middleware to set the active BUI project for processing requests."""

    def activate_project(request):
        """Set the correct database before processing the request."""

        project_home = request.session.get("project_home")
        if project_home is not None:
            set_project_home(project_home)

        response = get_response(request)

        return response

    return activate_project
