"""The views to display and edit pipeline configuration."""

from sqlalchemy import select, func

from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from superphot_pipeline.database.user_interface import\
    get_json_config,\
    save_json_config,\
    list_steps
from superphot_pipeline.database.interface import Session
#False positive
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import \
    Configuration,\
    ImageProcessingProgress
#pylint: enable=no-name-in-module


def config_tree(request, version=0, step='All', force_unlock=False):
    """Landing page for the configuration interface."""

    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        defined_versions = sorted(
            db_session.scalars(
                select(
                    func.distinct(Configuration.version)
                )
            ).all()
        )
        max_used_version = db_session.scalar(
            select(
                func.max(ImageProcessingProgress.configuration_version)
            )
        )

    return render(
        request,
        'configuration/config_tree.html',
        {
            'selected_step': step,
            'selected_version': version,
            'config_json': get_json_config(version, step=step, indent=4),
            'pipeline_steps': ['All'] + list_steps(),
            'config_versions': defined_versions,
            'max_locked_version': max_used_version,
            'locked': (not force_unlock) and version <= max_used_version
        }
    )


def save_config(request, version):
    """Save a user-defined configuration to the database."""

    print('Saving version: ' + repr(version))
    save_json_config(request.body, version)
    return HttpResponseRedirect(reverse("configuration:config_tree"))
