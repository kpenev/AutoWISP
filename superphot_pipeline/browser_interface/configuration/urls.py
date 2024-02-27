from django.urls import path

from . import views

app_name = 'configuration'

urlpatterns = [
    path("save_config/<int:version>", views.save_config, name="save_config"),
    path("", views.config_tree, name="config_tree"),
    path("<str:step>/<int:version>", views.config_tree, name="config_tree"),
]
