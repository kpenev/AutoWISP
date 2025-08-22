"""Define the URL paths used by the processing BUI app."""

from django.urls import path

from . import views

app_name = "home"

urlpatterns = [
    path("", views.home, name="home"),
    path(
        "new_project",
        views.CreateProjectView.as_view(),
        name="new_project",
    ),
    path(
        "new_project/<path:dirname>/",
        views.CreateProjectView.as_view(),
        name="new_project",
    ),
    path(
        "create_directory/<path:dirname>/",
        views.CreateProjectView.as_view(mode='create_dir'),
        name="create_directory",
    ),
    path(
        "select_project/<int:project_id>/",
        views.select_project,
        name="select_project",
    )
]
