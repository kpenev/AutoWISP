"""Define the vies available on the home page."""

import os.path
import json
from io import StringIO

from django.shortcuts import render, redirect, HttpResponse

from autowisp.database.defaults import master_info

from .create_project_view import (  # pylint: disable=unused-import
    CreateProjectView,
    MasterConfigView,
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
    request.session["project_home"] = "test"
    return render(request, "home/index.html", context)


def select_project(request, project_id):
    """Redirect to the processing progress page for the selected project."""

    request.session.flush()
    project = Project.objects.get(id=project_id)  # pylint: disable=no-member
    request.session["project_db_path"] = os.path.join(
        project.path, "autowisp.db"
    )

    return redirect("processing:progress")


def reset_project_config(request):
    """Reset the configuration of project being created to defaults."""

    request.session.flush()
    return redirect("home:new_project")


def export_master_config(request):
    """Generate a JSON file with the current master config for new project."""

    master_config = {}
    for master_type in master_info:
        if master_type in ["highflat", "lowflat"]:
            master_type = "flat"
        master_config[master_type] = {
            param: (
                request.session[f"master-{master_type}-{param}"]
                if param == "enabled"
                else list(
                    filter(
                        None,
                        request.session[f"master-{master_type}-{param}"],
                    )
                )
            )
            for param in ["enabled", "split", "match"]
        }
    with StringIO() as export_stream:
        json.dump(master_config, export_stream, indent=4)
        return HttpResponse(
            export_stream.getvalue().encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Content-Disposition": (
                    'attachment; filename="master_config.json"'
                ),
            },
        )
