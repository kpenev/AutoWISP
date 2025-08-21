"""Define the vies available on the home page."""

from django.shortcuts import render

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
    print(f'Context: {context!r}')  # Debugging output
    return render(request, "home/index.html", context)
