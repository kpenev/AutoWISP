"""Define middleware to handle multiple AutoWISP BUI projects."""

from autowisp.database.interface import set_sqlite_database


def set_project_middleware(get_response):
    """Middleware to set the active BUI project for processing requests."""

    def activate_project(request):
        """Set the correct database before processing the request."""

        db_path = request.session.get("project_db_path")
        if db_path is not None:
            set_sqlite_database(db_path)
        response = get_response(request)

        return response

    return activate_project
