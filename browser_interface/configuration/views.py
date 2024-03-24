"""The views to display and edit pipeline configuration."""

from collections import namedtuple

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
from superphot_pipeline.database.data_model import\
    Configuration,\
    ImageProcessingProgress
from superphot_pipeline.database.data_model import provenance
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
        if max_used_version is None:
            max_used_version = -1

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

    save_json_config(request.body, version)
    return HttpResponseRedirect(reverse("configuration:config_tree"))


def edit_survey(request, selected_type=None, selected_id=None):
    """
    Add/delete instruments/observers to the currently configured survey.

    Args:
        request:    See django.

        selected_type(str):    What type of survey component is currently
            selected. One of ``'observer'``, ``'observatory'``, ``'camera'``,
            ``'mount'``, ``'telescope'``

        selected_id(int):    The ID of the selected component within the
            corresponding database table.
    """


    context = {}
    #False positive:
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        if selected_type is not None:
            assert selected_id is not None
            selected_type = getattr(provenance, selected_type.title())
            selected = db_session.scalar(
                select(
                    selected_type
                ).where(
                    selected_type.id == selected_id
                )
            )
        for component_type in ['camera', 'mount', 'telescope']:
            context[component_type + 's'] = [
                namedtuple(
                    component_type,
                    ['id', 'serial', 'make', 'model']
                )(
                    equipment.id,
                    equipment.serial_number,
                    getattr(equipment, component_type + '_type').make,
                    getattr(equipment, component_type + '_type').model
                )
                for equipment in db_session.scalars(
                    select(
                        getattr(provenance, component_type.title())
                    )
                ).all()
            ]

    return render(
        request,
        'configuration/edit_survey.html',
        context
    )

