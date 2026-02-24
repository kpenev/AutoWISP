"""Define context processors for the entire interface."""


def global_variables(request):
    """Set global variables available to all templates."""

    return {"project_name": request.session.get("project_name", "")}
