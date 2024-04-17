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
    path("survey/<slug:selected_component>/<slug:selected_id>",
         views.edit_survey,
         name='survey'),
    path("survey/<slug:selected_component>//<slug:selected_type_id>",
         views.edit_survey,
         name='survey'),
    path("survey/<slug:selected_component>/<slug:selected_id>/"
         "<create_new_types>",
         views.edit_survey,
         name='survey'),
    path("delete_from_survey/<slug:component_type>/<int:component_id>",
         views.delete_from_survey,
         name='delete_from_survey'),
    path("delete_from_survey/<slug:component_type>/T<int:component_type_id>",
         views.delete_from_survey,
         name='delete_from_survey'),
    path("update_survey_component/<slug:component_type>/<slug:component_id>",
         views.update_survey_component,
         name='update_survey_component'),
    path("update_survey_component_type/<slug:component_type>/<slug:type_id>",
         views.update_survey_component_type,
         name='update_survey_component_type'),
    path('change_access/<int:new_access>/<slug:selected_component>'
         '/<int:selected_id>/<slug:target_component>/<int:target_id>',
         views.change_access,
         name='change_access'),
]
