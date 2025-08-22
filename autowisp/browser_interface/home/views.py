"""Define the vies available on the home page."""

import os.path

from django.shortcuts import render, redirect

from autowisp.database.interface import set_sqlite_database
from .create_project_view import (  # pylint: disable=unused-import
    CreateProjectView,
)
from .models import Project


def home(request):
    """Display the home page."""

    display_columns = [
        field.name
        for field in Project._meta.get_fields()  # pylint: disable=no-member, protected-access
        if field.name != "id"
    ]
    print(f"Projects: {Project.objects.all()}")  # pylint: disable=no-member
    context = {
        "columns": display_columns,
        "projects": {
            proj.id: [getattr(proj, col) for col in display_columns]
            for proj in Project.objects.all()  # pylint: disable=no-member
        },
    }
    print(f"Context: {context!r}")  # Debugging output
    request.session['project_home'] = 'test'
    return render(request, "home/index.html", context)


def select_project(request, project_id):
    """Redirect to the processing progress page for the selected project."""

    request.session.flush()
    project = Project.objects.get(id=project_id)  # pylint: disable=no-member
    request.session["project_db_path"] = os.path.join(
        project.path, "autowisp.db"
    )

    return redirect("processing:progress")
