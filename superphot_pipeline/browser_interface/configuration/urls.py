from django.urls import path

from . import views

app_name = 'configuration'

urlpatterns = [
    path("", views.config_tree, name="config_tree"),
    path("<str:step>/<int:version>", views.config_tree, name="config_tree"),
]
