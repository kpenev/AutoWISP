"""The views to display and edit pipeline configuration."""
from django.shortcuts import render

from superphot_pipeline.database.user_interface import\
    get_json_config,\
    list_steps

def config_tree(request, version=0, step='All'):
    """Landing page for the configuration interface."""

    return render(
        request,
        'configuration/config_tree.html',
        {
            'selected_step': step,
            'selected_version': version,
            'config_json': get_json_config(version, step=step, indent=4),
            'pipeline_steps': ['All'] + list_steps(),
            'config_versions': [0],
        }
    )
