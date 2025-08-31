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
        "reset_project_config",
        views.reset_project_config,
        name="reset_project_config",
    ),
    path(
        "select_project_home/",
        views.CreateProjectView.as_view(
            mode="select_home",
            url_name="home:select_project_home",
            template="home/select_project_home.html",
        ),
        name="select_project_home",
    ),
    path(
        "select_project_home/<path:dirname>/",
        views.CreateProjectView.as_view(
            mode="select_home",
            url_name="home:select_project_home",
            template="home/select_project_home.html",
        ),
        name="select_project_home",
    ),
    path(
        "create_directory/<path:dirname>/",
        views.CreateProjectView.as_view(
            mode="create_dir",
            url_name="home:select_project_home",
            template="home/create_directory.html",
        ),
        name="create_directory",
    ),
    path(
        "select_project/<int:project_id>/",
        views.select_project,
        name="select_project",
    ),
    path(
        "change_master_config/<slug:master_type>",
        views.MasterConfigView.as_view(),
        name="change_master_config",
    ),
    path(
        "export_master_config",
        views.export_master_config,
        name="export_master_config",
    ),
]
