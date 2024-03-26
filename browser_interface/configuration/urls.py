"""Configure the URLs for the interface for editing the configuration."""

from django.urls import path

from . import views

app_name = 'configuration'

urlpatterns = [
    path("save_config/<int:version>",
         views.save_config,
         name="save_config"),
    path("",
         views.config_tree,
         name="config_tree"),
    path("<str:step>/<int:version>",
         views.config_tree,
         name="config_tree"),
    path("<str:step>/<int:version>/<int:force_unlock>",
         views.config_tree,
         name="config_tree"),
    path("survey",
         views.edit_survey,
         name='survey'),
    path("survey/<slug:selected_component_type>/<int:selected_id>",
         views.edit_survey,
         name='survey'),
    path("survey/<slug:selected_component_type>/<int:selected_id>/"
         "<create_new_types>",
         views.edit_survey,
         name='survey')

]
